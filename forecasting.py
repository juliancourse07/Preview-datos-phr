"""Motor de pronóstico de facturación diaria (firma de servicios legales).

Enfoque: pronóstico mensual (nivel por día hábil x estacionalidad mensual x
días hábiles) distribuido al día con la curva acumulada intra-mes.
Los modelos candidatos se comparan con backtesting de origen móvil.
"""
from __future__ import annotations

import warnings

import holidays
import numpy as np
import pandas as pd

# Estacionalidad mensual de referencia (firmas legales en Colombia): enero bajo,
# cierres de trimestre/año altos. Se usa cuando la historia no permite estimarla.
PRIOR_SEASONALITY = {
    1: 0.82, 2: 0.92, 3: 1.00, 4: 0.96, 5: 1.00, 6: 1.02,
    7: 0.96, 8: 1.00, 9: 1.02, 10: 1.05, 11: 1.08, 12: 1.17,
}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie"]
GRID = np.linspace(0, 1, 101)


# ------------------------------------------------------------------ datos
def load_data(source) -> pd.DataFrame:
    raw = pd.read_excel(source)
    cols = {c: str(c).strip().lower() for c in raw.columns}
    raw = raw.rename(columns=cols)
    fecha = next((c for c in raw.columns if c.startswith("fecha")), None)
    valor = next((c for c in raw.columns if "total" in c or "fact" in c and c != fecha), None)
    area = "area" if "area" in raw.columns else None
    if fecha is None or valor is None or area is None:
        raise ValueError("El archivo debe tener las columnas Area, FechaFctd y TotalFactPesos.")
    df = pd.DataFrame({
        "Area": raw[area].astype(str).str.strip(),
        "Fecha": pd.to_datetime(raw[fecha], dayfirst=True, errors="coerce"),
        "Valor": pd.to_numeric(raw[valor], errors="coerce"),
    }).dropna()
    df["Fecha"] = df["Fecha"].dt.normalize()
    return df


def co_holidays(years) -> set:
    return set(pd.to_datetime(list(holidays.CO(years=sorted(set(years))).keys())))


def business_days(start, end) -> pd.DatetimeIndex:
    rng = pd.bdate_range(start, end)
    hol = co_holidays(range(rng[0].year, rng[-1].year + 1))
    return pd.DatetimeIndex([d for d in rng if d not in hol])


def daily_series(df: pd.DataFrame) -> pd.Series:
    """Serie diaria sobre todos los días hábiles del rango (0 si no hubo facturación)."""
    s = df.groupby("Fecha")["Valor"].sum()
    idx = business_days(s.index.min(), s.index.max()).union(s.index)
    return s.reindex(idx, fill_value=0.0).sort_index()


def winsorize(s: pd.Series, q: float | None) -> pd.Series:
    if not q:
        return s
    return s.clip(upper=s.quantile(q))


def monthly_table(daily: pd.Series) -> pd.DataFrame:
    g = daily.groupby(daily.index.to_period("M"))
    out = pd.DataFrame({"total": g.sum(), "dias_habiles": g.size()})
    out["por_dia"] = out["total"] / out["dias_habiles"]
    return out


def month_bdays(period: pd.Period) -> int:
    return len(business_days(period.start_time, period.end_time.normalize()))


# --------------------------------------------------------- estacionalidad
def monthly_seasonality(monthly: pd.DataFrame, shrink: float = 1.0) -> pd.Series:
    """Índice mensual: datos (si hay años con >=9 meses) encogidos hacia el prior."""
    per = monthly["por_dia"]
    ratios: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for _, grp in per.groupby(per.index.year):
        if len(grp) >= 9:
            rel = grp / grp.mean()
            for p, v in rel.items():
                ratios[p.month].append(v)
    prior = pd.Series(PRIOR_SEASONALITY)
    idx = {}
    for m in range(1, 13):
        n = len(ratios[m])
        idx[m] = (np.sum(ratios[m]) + shrink * prior[m]) / (n + shrink)
    s = pd.Series(idx)
    return s / s.mean()


def weekday_index(daily: pd.Series) -> pd.Series:
    d = daily[daily.index.dayofweek < 5]
    g = d.groupby(d.index.dayofweek).mean()
    return (g / g.mean()).reindex(range(5))


def intramonth_curve(daily: pd.Series, smooth: float = 0.4) -> np.ndarray:
    """Curva acumulada promedio (0-1) del avance de facturación dentro del mes."""
    curves = []
    for _, m in daily.groupby(daily.index.to_period("M")):
        tot = m.sum()
        if len(m) < 10 or tot <= 0:
            continue
        cum = np.concatenate([[0], m.cumsum().values / tot])
        x = np.linspace(0, 1, len(cum))
        curves.append(np.interp(GRID, x, cum))
    base = np.mean(curves, axis=0) if curves else GRID.copy()
    return (1 - smooth) * base + smooth * GRID


def day_profile(curve: np.ndarray, n: int) -> np.ndarray:
    cum = np.interp(np.arange(n + 1) / n, GRID, curve)
    return np.diff(cum)


# ----------------------------------------------------- modelos mensuales
def _deseason(monthly: pd.DataFrame, idx: pd.Series) -> pd.Series:
    return pd.Series(
        [v / idx[p.month] for p, v in monthly["por_dia"].items()], index=monthly.index
    )


def _level(monthly, idx, window):
    return _deseason(monthly, idx).tail(window).mean()


def _model_runrate(window):
    def f(monthly, idx, target):
        return _level(monthly, idx, window) * idx[target.month]
    return f


def _model_ets(monthly, idx, target):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    ds = _deseason(monthly, idx)
    if len(ds) < 6:
        raise ValueError("historia insuficiente")
    steps = (target.year - ds.index[-1].year) * 12 + target.month - ds.index[-1].month
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            ds.values, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit()
    return float(fit.forecast(steps)[-1]) * idx[target.month]


def _model_snaive(monthly, idx, target):
    prev = pd.Period(year=target.year - 1, month=target.month, freq="M")
    if prev not in monthly.index:
        raise ValueError("sin mismo mes del año anterior")
    ds = _deseason(monthly, idx)
    growth = ds.tail(3).mean() / ds.loc[prev - 2 : prev].mean() if (prev - 2) in ds.index else 1.0
    return monthly.loc[prev, "por_dia"] * float(np.clip(growth, 0.5, 2.0))


BASE_MODELS = {
    "Run-rate 3 meses (desestacionalizado)": _model_runrate(3),
    "Run-rate 6 meses (desestacionalizado)": _model_runrate(6),
    "Holt-Winters amortiguado (mensual)": _model_ets,
    "Estacional ingenuo + tendencia": _model_snaive,
}
ENSEMBLE = "Ensamble (mediana de modelos)"
DEFAULT_MODEL = "Run-rate 3 meses (desestacionalizado)"


def _predict(name, monthly, idx, target) -> float | None:
    try:
        if name == ENSEMBLE:
            vals = []
            for f in BASE_MODELS.values():
                try:
                    vals.append(f(monthly, idx, target))
                except Exception:
                    pass
            return float(np.median(vals)) if vals else None
        return float(BASE_MODELS[name](monthly, idx, target))
    except Exception:
        return None


def backtest(monthly: pd.DataFrame, max_origins: int = 6):
    """Origen móvil a 1 mes. Devuelve (tabla de métricas, detalle)."""
    n = len(monthly)
    if n < 4:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for t in range(max(3, n - max_origins), n):
        train, target = monthly.iloc[:t], monthly.index[t]
        idx = monthly_seasonality(train)
        for name in [*BASE_MODELS, ENSEMBLE]:
            p = _predict(name, train, idx, target)
            if p is not None:
                rows.append({
                    "Modelo": name, "Mes": str(target),
                    "Real": monthly.loc[target, "total"],
                    "Pronóstico": p * monthly.loc[target, "dias_habiles"],
                })
    det = pd.DataFrame(rows)
    if det.empty:
        return pd.DataFrame(), det
    det["Error"] = det["Pronóstico"] - det["Real"]
    g = det.groupby("Modelo")
    met = pd.DataFrame({
        "WAPE %": g.apply(lambda x: x["Error"].abs().sum() / x["Real"].abs().sum() * 100),
        "Sesgo %": g.apply(lambda x: x["Error"].sum() / x["Real"].sum() * 100),
        "Meses evaluados": g.size(),
    }).sort_values("WAPE %")
    return met.round(2), det


# ------------------------------------------------------------ incertidumbre
def bootstrap_ratio(daily: pd.Series, n_days: int, sims: int = 3000, seed: int = 7):
    """Cuantiles (p10,p90) de suma de n días / (n*media), remuestreando días históricos."""
    rng = np.random.default_rng(seed)
    d = daily.tail(130).values
    mean = d.mean()
    if mean <= 0 or len(d) < 5:
        return 0.8, 1.2
    tot = rng.choice(d, size=(sims, n_days)).sum(axis=1) / (n_days * mean)
    return float(np.quantile(tot, 0.10)), float(np.quantile(tot, 0.90))


# ------------------------------------------------------------- orquestador
def run_forecast(
    df: pd.DataFrame,
    model: str | None,
    end_month: pd.Period,
    annual_adj: float = 0.0,
    winsor_q: float | None = 0.98,
    smooth: float = 0.4,
    stretch: float = 0.05,
    budget_flat: float = 0.5,
    seasonality_override: dict | None = None,
):
    daily_raw = daily_series(df)
    daily = winsorize(daily_raw, winsor_q)
    monthly = monthly_table(daily)
    monthly_raw = monthly_table(daily_raw)

    # último mes con datos incompletos se descarta de entrenamiento
    last = monthly.index[-1]
    if daily.index[-1] < business_days(last.start_time, last.end_time.normalize())[-1] - pd.Timedelta(days=3) \
            and len(monthly) > 1:
        monthly = monthly.iloc[:-1]

    idx = monthly_seasonality(monthly)
    if seasonality_override:
        idx = pd.Series({m: seasonality_override.get(m, idx[m]) for m in range(1, 13)})
        idx = idx / idx.mean()

    met, det = backtest(monthly)
    if model is None:
        model = met.index[0] if not met.empty else DEFAULT_MODEL
    validated = not met.empty

    curve = intramonth_curve(daily, smooth)
    first = monthly.index[-1] + 1
    months = pd.period_range(first, end_month, freq="M")
    drows, mrows = [], []
    for p in months:
        bd = business_days(p.start_time, p.end_time.normalize())
        per_day = _predict(model, monthly, idx, p)
        if per_day is None:
            per_day = _predict(DEFAULT_MODEL, monthly, idx, p)
        steps = (p.year - monthly.index[-1].year) * 12 + p.month - monthly.index[-1].month
        per_day *= (1 + annual_adj) ** (steps / 12)
        total = per_day * len(bd)
        lo_r, hi_r = bootstrap_ratio(daily, len(bd))
        prof = day_profile(curve, len(bd))
        flat = np.full(len(bd), 1 / len(bd))
        bprof = (1 - budget_flat) * prof + budget_flat * flat
        budget_total = total * (1 + stretch)
        for d, w, bw in zip(bd, prof, bprof):
            drows.append({
                "Fecha": d, "Mes": str(p), "Pronóstico": total * w,
                "Presupuesto ideal": budget_total * bw,
            })
        mrows.append({
            "Mes": str(p), "Días hábiles": len(bd), "Pronóstico": total,
            "P10": total * lo_r, "P90": total * hi_r,
            "Presupuesto ideal": budget_total,
            "Índice estacional": idx[p.month],
        })
    fc_daily = pd.DataFrame(drows)
    fc_daily["Pronóstico acum. mes"] = fc_daily.groupby("Mes")["Pronóstico"].cumsum()
    fc_daily["Presupuesto acum. mes"] = fc_daily.groupby("Mes")["Presupuesto ideal"].cumsum()
    return {
        "model": model, "validated": validated, "metrics": met, "backtest": det,
        "daily_hist": daily_raw, "monthly_hist": monthly_raw,
        "monthly_fc": pd.DataFrame(mrows), "daily_fc": fc_daily,
        "seasonality": idx, "weekday": weekday_index(daily), "curve": curve,
        "uses_prior": len(monthly) < 12,
    }
