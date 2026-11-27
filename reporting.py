"""Reportería: cruce Facturación + Horas -> Facturación, Proformas, Horas pendientes y Análisis.

Reglas de negocio:
  * Facturación y Horas se cruzan por «Numero Serial» (se trae el comentario de la línea).
  * Facturación  : Estado Proforma = F  y hora_facturable en S o vacío  (fecha: Fecha de Factura).
  * Proformas    : Estado Proforma <> F y hora_facturable en S o vacío  (fecha: Fec. Doc.).
  * Horas pendientes: archivo de Horas con Estado Proforma vacío        (fecha: Fecha Movimiento TR).
"""
from __future__ import annotations

import io
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VERDE, VERDE_CL, LIMA, GRIS, CREMA = "#007A3D", "#8DC63F", "#C4D82E", "#6D6E71", "#F2F8EC"
LOGO = Path(__file__).with_name("logo-phr-horizontal-1024x93.png")
LOGO_BLANCO = Path(__file__).with_name("logo-phr-blanco.png")
MESES_L = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
           "noviembre", "diciembre"]


def fecha_larga(d, corta=False) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} de {MESES_L[d.month - 1]}" + ("" if corta else f" de {d.year}")

# Significado resumido de los estados de proforma (se completa con «Nombre Estado Prof.» de Horas).
ESTADOS_BASE = {"F": "Facturado", "O": "Abierta / en elaboración"}

# (nombre, columnas candidatas (normalizadas), tipo)  tipo: t=texto d=fecha h=horas c=COP
FACT_SPEC = [
    ("Fecha de Factura", ["fecha de factura"], "d"),
    ("Fec. Doc.", ["fec doc"], "d"),
    ("N° Factura", ["numero fact prefijo", "numero de factura"], "t"),
    ("Nro. Doc.", ["nro doc"], "t"),
    ("Cliente", ["cliente"], "t"),
    ("Cód. Cliente", ["cod cliente"], "t"),
    ("Holding", ["nombre holding"], "t"),
    ("Cód. Caso", ["cod caso"], "t"),
    ("Caso", ["nombre de caso"], "t"),
    ("Área", ["area empleado", "area"], "t"),
    ("Socio", ["nombre socio resp", "socio"], "t"),
    ("Empleado", ["nombre del empleado"], "t"),
    ("Categoría", ["nombre de la categoria"], "t"),
    ("Fecha Trabajo", ["fecha hora"], "d"),
    ("Horas", ["horas facturadas"], "h"),
    ("Total Moneda Original", ["total facturado moneda original"], "c"),
    ("Total COP", ["total real facturado en cop"], "c"),
    ("Total Tarifa Estándar COP", ["total facturado a tarifa estandar cop"], "c"),
    ("Facturable", ["hora facturable"], "t"),
    ("Estado", ["estado proforma"], "t"),
    ("N° Serial", ["numero serial"], "t"),
]
HORAS_SPEC = [
    ("Fecha Movimiento TR", ["fecha movimiento tr"], "d"),
    ("Área", ["area", "nom depto empl"], "t"),
    ("Empleado", ["nombre del empleado"], "t"),
    ("Categoría", ["nombre de la categoria"], "t"),
    ("Cód. Caso", ["codigo de caso"], "t"),
    ("Caso", ["nombre de caso"], "t"),
    ("Cliente", ["cliente"], "t"),
    ("Socio", ["socio"], "t"),
    ("Tarea", ["nombre de tarea"], "t"),
    ("Comentario", ["comentario"], "t"),
    ("Horas", ["tiempo reportado"], "h"),
    ("Tiempo Facturable", ["tiempo facturable"], "h"),
    ("Facturable", ["facturable"], "t"),
    ("Tarifa Estándar", ["tarifa std"], "c"),
    ("Tarifa Deal", ["tarifa deal"], "c"),
    ("Estado", ["estado proforma"], "t"),
    ("Nombre Estado", ["nombre estado prof"], "t"),
    ("N° Serial", ["numero serial"], "t"),
]


# ------------------------------------------------------------------ lectura
def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def find_col(df: pd.DataFrame, cands) -> list:
    names = {norm(c): c for c in reversed(df.columns)}
    out = []
    for cand in cands:
        if cand in names:
            out.append(names[cand])
        else:
            out += [c for n, c in names.items() if n.startswith(cand + " ") and not n[len(cand) + 1:].isdigit()][:1]
    return out


def to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        t = str(v).strip().replace("$", "").replace("%", "").replace(" ", "")
        if not t:
            return np.nan
        if "," in t:
            t = t.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
        try:
            return float(t)
        except ValueError:
            return np.nan

    return s.map(conv)


def to_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    return pd.to_datetime(s, dayfirst=True, errors="coerce").dt.normalize()


def to_text(s: pd.Series) -> pd.Series:
    t = s.astype(object).where(s.notna(), "").astype(str).str.strip()
    return t.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True).replace({"nan": "", "NaT": "", "None": ""})


def standardize(raw: pd.DataFrame, spec) -> pd.DataFrame:
    out = {}
    for name, cands, kind in spec:
        cols = find_col(raw, cands)
        if kind == "t":
            ser = pd.Series("", index=raw.index)
            for c in cols:
                t = to_text(raw[c])
                ser = ser.where(ser != "", t)
        elif kind == "d":
            ser = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
            for c in cols:
                ser = ser.fillna(to_date(raw[c]))
        else:
            ser = pd.Series(np.nan, index=raw.index)
            for c in cols:
                ser = ser.fillna(to_num(raw[c]))
        out[name] = ser
    return pd.DataFrame(out)


def kind_of(raw: pd.DataFrame) -> str | None:
    if find_col(raw, ["total real facturado en cop"]):
        return "facturacion"
    if find_col(raw, ["fecha movimiento tr"]):
        return "horas"
    return None


def read_sheets(src) -> dict:
    return pd.read_excel(src, sheet_name=None)


def remap(raw: pd.DataFrame, spec, mapping: dict) -> pd.DataFrame:
    """Aplica una asignación manual {campo estándar: columna del archivo}; vacío = no existe."""
    out = {}
    for name, cands, kind in spec:
        col = mapping.get(name)
        out[cands[0]] = raw[col] if col else pd.Series(np.nan, index=raw.index)
    return pd.DataFrame(out)


def read_excel(src) -> pd.DataFrame:
    """Lee la primera hoja que se reconozca como Facturación u Horas."""
    sheets = pd.read_excel(src, sheet_name=None)
    for df in sheets.values():
        if kind_of(df):
            return df
    return next(iter(sheets.values()))


# --------------------------------------------------------------- transformación
def build_report(fact_raw: pd.DataFrame, horas_raw: pd.DataFrame) -> dict:
    """Devuelve {'facturacion','proformas','pendientes'} ya cruzados y filtrados por reglas de negocio."""
    if kind_of(fact_raw) == "horas" and kind_of(horas_raw) == "facturacion":
        fact_raw, horas_raw = horas_raw, fact_raw
    if kind_of(fact_raw) != "facturacion":
        raise ValueError("No se reconoce el archivo de Facturación (falta «Total Real Facturado en COP$»).")
    if kind_of(horas_raw) != "horas":
        raise ValueError("No se reconoce el archivo de Horas (falta «Fecha Movimiento TR»).")

    return build_from_std(standardize(fact_raw, FACT_SPEC), standardize(horas_raw, HORAS_SPEC))


def build_from_std(f: pd.DataFrame, h: pd.DataFrame, estados_fact=("F",), solo_facturable=True,
                   estados_pend=("",)) -> dict:
    """Construye el informe desde tablas ya estandarizadas; los criterios de los indicadores son configurables."""
    f, h = f.copy(), h.copy()

    # Cruce por serial: comentario (y otros datos de la hora) sin duplicar filas de facturación
    hs = h[h["N° Serial"] != ""].drop_duplicates("N° Serial")
    f = f.merge(hs[["N° Serial", "Comentario"]], on="N° Serial", how="left")
    f["Comentario"] = f["Comentario"].fillna("")

    f["Estado"] = f["Estado"].str.upper()
    f["Facturable"] = f["Facturable"].str.upper()
    h["Estado"] = h["Estado"].str.upper()
    facturable_ok = f["Facturable"].isin(["S", ""]) if solo_facturable else pd.Series(True, index=f.index)
    es_fact = f["Estado"].isin([e.upper() for e in estados_fact])

    names = dict(ESTADOS_BASE)
    names.update({e: n.title() for e, n in zip(h["Estado"], h["Nombre Estado"]) if e and n})

    def etiqueta(code):
        return f"{code} – {names.get(code, 'Otro estado')}" if code else "Sin estado"

    f["Realización %"] = np.where(f["Total Tarifa Estándar COP"] > 0, f["Total COP"] / f["Total Tarifa Estándar COP"], np.nan)

    fac = f[es_fact & facturable_ok].copy()
    fac["Mes"] = fac["Fecha de Factura"].dt.to_period("M").astype(str).replace("NaT", "")

    pro = f[~es_fact & facturable_ok].copy()
    pro["Proforma"] = pro["Estado"].map(etiqueta)
    pro["Mes"] = pro["Fec. Doc."].dt.to_period("M").astype(str).replace("NaT", "")
    hoy = pd.Timestamp(datetime.now().date())
    pro["Antigüedad (días)"] = (hoy - pro["Fec. Doc."]).dt.days

    pen = h[h["Estado"].isin([e.upper() for e in estados_pend])].copy()
    pen["Valor Estimado COP"] = pen["Horas"].fillna(0) * pen["Tarifa Estándar"].fillna(0)
    pen["Antigüedad (días)"] = (hoy - pen["Fecha Movimiento TR"]).dt.days
    pen["Rango Antigüedad"] = pd.cut(
        pen["Antigüedad (días)"], [-np.inf, 30, 60, 90, 180, np.inf],
        labels=["0-30 días", "31-60 días", "61-90 días", "91-180 días", "> 180 días"],
    ).astype(str).replace("nan", "")
    pen["Mes"] = pen["Fecha Movimiento TR"].dt.to_period("M").astype(str).replace("NaT", "")

    fac = fac.sort_values("Fecha de Factura", ascending=False, ignore_index=True)
    pro = pro.sort_values("Fec. Doc.", ascending=False, ignore_index=True)
    pen = pen.sort_values("Fecha Movimiento TR", ascending=True, ignore_index=True)

    # Columnas de presentación (orden ejecutivo)
    return {
        "facturacion": fac[[c for c in [
            "Fecha de Factura", "N° Factura", "Cliente", "Cód. Cliente", "Holding", "Cód. Caso", "Caso", "Área", "Socio",
            "Empleado", "Categoría", "Fecha Trabajo", "Horas", "Total Moneda Original", "Total COP",
            "Total Tarifa Estándar COP", "Realización %", "Facturable", "N° Serial", "Comentario", "Mes"] if c in fac]],
        "proformas": pro[[
            "Proforma", "Fec. Doc.", "Nro. Doc.", "Cliente", "Cód. Cliente", "Holding", "Cód. Caso", "Caso", "Área",
            "Socio", "Empleado", "Categoría", "Fecha Trabajo", "Horas", "Total COP", "Total Tarifa Estándar COP",
            "Facturable", "N° Serial", "Comentario", "Antigüedad (días)", "Mes"]],
        "pendientes": pen[[
            "Fecha Movimiento TR", "Área", "Empleado", "Categoría", "Cód. Caso", "Caso", "Cliente", "Socio", "Tarea",
            "Comentario", "Horas", "Facturable", "Tarifa Estándar", "Valor Estimado COP", "Antigüedad (días)",
            "Rango Antigüedad", "N° Serial", "Mes"]],
    }


DATE_COL = {"facturacion": "Fecha de Factura", "proformas": "Fec. Doc.", "pendientes": "Fecha Movimiento TR"}
FMT = {
    "Horas": "hrs", "Tiempo Facturable": "hrs", "Total Moneda Original": "cop", "Total COP": "cop",
    "Total Tarifa Estándar COP": "cop", "Tarifa Estándar": "cop", "Tarifa Deal": "cop", "Valor Estimado COP": "cop",
    "Realización %": "pct", "Antigüedad (días)": "int",
    "Fecha de Factura": "date", "Fec. Doc.": "date", "Fecha Trabajo": "date", "Fecha Movimiento TR": "date",
}
WIDTH = {"Comentario": 60, "Caso": 38, "Cliente": 34, "Holding": 26, "Tarea": 28, "Socio": 26, "Empleado": 26,
         "Proforma": 26, "Categoría": 14, "Área": 16}


def apply_filters(data: dict, desde=None, hasta=None, columnas: dict | None = None) -> dict:
    """Filtra por rango de fechas (columna de fecha de cada hoja) y por {columna: [valores]} donde exista."""
    out = {}
    for k, df in data.items():
        m = pd.Series(True, index=df.index)
        dc = DATE_COL[k]
        if desde is not None:
            m &= df[dc] >= pd.Timestamp(desde)
        if hasta is not None:
            m &= df[dc] <= pd.Timestamp(hasta)
        for col, vals in (columnas or {}).items():
            if col in df and vals:
                m &= df[col].isin(vals)
        out[k] = df[m].reset_index(drop=True)
    return out


# ------------------------------------------------------------------ análisis
def _share(s):
    t = s.sum()
    return s / t if t else s * 0


def analyze(d: dict) -> dict:
    fac, pro, pen = d["facturacion"], d["proformas"], d["pendientes"]
    tot, std, hrs = fac["Total COP"].sum(), fac["Total Tarifa Estándar COP"].sum(), fac["Horas"].sum()
    pen_val, pen_h = pen["Valor Estimado COP"].sum(), pen["Horas"].sum()
    viejas = pen[pen["Antigüedad (días)"] > 90]
    kpis = [
        ("Total facturado COP", tot, "cop"), ("Horas facturadas", hrs, "hrs"),
        ("Tarifa efectiva COP/hora", tot / hrs if hrs else 0, "cop"),
        ("Realización vs tarifa estándar", tot / std if std else 0, "pct"),
        ("Proformas pendientes COP", pro["Total COP"].sum(), "cop"),
        ("Horas pendientes", pen_h, "hrs"), ("Valor estimado horas pendientes", pen_val, "cop"),
        ("Horas pendientes > 90 días", viejas["Horas"].sum(), "hrs"),
    ]

    def grp(df, key, agg, top=None, sort=None):
        g = df.groupby(key, dropna=False).agg(**agg).reset_index()
        g = g[g[key] != ""]
        g = g.sort_values(sort or list(agg)[0], ascending=False)
        return g.head(top) if top else g

    def fact_group(key, top=None):
        g = grp(fac, key, {"Total COP": ("Total COP", "sum"), "Horas": ("Horas", "sum"),
                           "Estándar": ("Total Tarifa Estándar COP", "sum")}, top)
        g["Tarifa efectiva"] = np.where(g["Horas"] > 0, g["Total COP"] / g["Horas"], 0)
        g["Realización %"] = np.where(g["Estándar"] > 0, g["Total COP"] / g["Estándar"], 0)
        g["% del total"] = g["Total COP"] / tot if tot else 0
        return g.drop(columns="Estándar")

    mes = grp(fac, "Mes", {"Total COP": ("Total COP", "sum"), "Horas": ("Horas", "sum")}).sort_values("Mes")
    mes["Variación vs mes anterior"] = mes["Total COP"].pct_change().replace([np.inf, -np.inf], np.nan)

    bucket_order = ["0-30 días", "31-60 días", "61-90 días", "91-180 días", "> 180 días"]
    ant = grp(pen, "Rango Antigüedad", {"Horas": ("Horas", "sum"), "Valor estimado": ("Valor Estimado COP", "sum"),
                                        "Registros": ("Horas", "size")})
    ant = ant.set_index("Rango Antigüedad").reindex(bucket_order).fillna(0).reset_index()
    ant["% de horas"] = _share(ant["Horas"])

    pro_est = grp(pro, "Proforma", {"Registros": ("Horas", "size"), "Total COP": ("Total COP", "sum"),
                                    "Horas": ("Horas", "sum"), "Antigüedad prom. (días)": ("Antigüedad (días)", "mean")})
    pro_cli = grp(pro, "Cliente", {"Total COP": ("Total COP", "sum"), "Horas": ("Horas", "sum"),
                                   "Antigüedad prom. (días)": ("Antigüedad (días)", "mean")}, 10)
    pen_soc = grp(pen, "Socio", {"Horas": ("Horas", "sum"), "Valor estimado": ("Valor Estimado COP", "sum"),
                                 "Antigüedad prom. (días)": ("Antigüedad (días)", "mean")}, 10)
    pen_area = grp(pen, "Área", {"Horas": ("Horas", "sum"), "Valor estimado": ("Valor Estimado COP", "sum"),
                                 "Antigüedad prom. (días)": ("Antigüedad (días)", "mean")})
    pen_caso = grp(pen, "Caso", {"Horas": ("Horas", "sum"), "Valor estimado": ("Valor Estimado COP", "sum"),
                                 "Antigüedad máx. (días)": ("Antigüedad (días)", "max")}, 10)

    emp = pd.concat([
        pen.assign(Origen="Pendiente")[["Empleado", "Caso", "Horas", "Valor Estimado COP"]],
        fac[["Empleado", "Caso", "Horas", "Total COP"]].rename(columns={"Total COP": "Valor Estimado COP"}).assign(Origen="Facturado"),
    ])
    emp = emp[emp["Empleado"] != ""]
    emp_caso = (emp.pivot_table(index=["Caso", "Empleado"], columns="Origen", values=["Horas", "Valor Estimado COP"],
                                aggfunc="sum", fill_value=0))
    emp_caso.columns = [{"Horas": "Horas ", "Valor Estimado COP": "Valor COP "}[a] + {"Pendiente": "pendientes", "Facturado": "facturadas"}[b]
                        for a, b in emp_caso.columns]
    emp_caso = emp_caso.reset_index().sort_values(["Caso", "Horas pendientes" if "Horas pendientes" in emp_caso else "Empleado"],
                                                  ascending=[True, False])

    tablas = [
        ("Facturación por mes", mes, "col", "Mes", "Total COP"),
        ("Facturación por área", fact_group("Área"), "col", "Área", "Total COP"),
        ("Top 10 clientes facturados", fact_group("Cliente", 10), "bar", "Cliente", "Total COP"),
        ("Top 10 socios por facturación", fact_group("Socio", 10), "bar", "Socio", "Total COP"),
        ("Proformas por estado (sin facturar)", pro_est, None, None, None),
        ("Top 10 clientes con proformas pendientes de facturar", pro_cli, "bar", "Cliente", "Total COP"),
        ("Horas pendientes por antigüedad", ant, "col", "Rango Antigüedad", "Horas"),
        ("Horas pendientes por área", pen_area, "col", "Área", "Horas"),
        ("Top 10 socios con horas pendientes (para gestión de las asistentes)", pen_soc, "bar", "Socio", "Horas"),
        ("Top 10 casos con más horas pendientes", pen_caso, "bar", "Caso", "Horas"),
        ("Empleados asociados por caso (horas y valor)", emp_caso, None, None, None),
    ]

    ins = []
    if tot:
        cli = fact_group("Cliente", 1)
        if len(cli):
            ins.append(f"El cliente {cli['Cliente'].iloc[0]} concentra el {cli['% del total'].iloc[0]:.0%} de lo facturado.")
        top3 = fact_group("Cliente", 3)["% del total"].sum()
        ins.append(f"Los 3 principales clientes representan el {top3:.0%} de la facturación (riesgo de concentración).")
        ins.append(f"Se factura a {tot / hrs:,.0f} COP por hora, con una realización del {tot / std:.0%} frente a la tarifa estándar."
                   if hrs and std else "Sin horas/tarifa estándar para calcular realización.")
        if len(mes) >= 2 and pd.notna(mes["Variación vs mes anterior"].iloc[-1]):
            ins.append(f"El último mes ({mes['Mes'].iloc[-1]}) varió {mes['Variación vs mes anterior'].iloc[-1]:+.0%} frente al anterior.")
    if len(pro):
        ins.append(f"Hay {len(pro):,} líneas de proforma sin facturar por {pro['Total COP'].sum():,.0f} COP; "
                   f"antigüedad promedio {pro['Antigüedad (días)'].mean():.0f} días. Priorizar su facturación.")
    if len(pen):
        ins.append(f"Hay {pen_h:,.1f} horas pendientes (≈ {pen_val:,.0f} COP a tarifa estándar); "
                   f"{viejas['Horas'].sum():,.1f} h llevan más de 90 días ({_pct(viejas['Horas'].sum(), pen_h)} del total).")
        if len(pen_soc):
            ins.append(f"El socio con más horas pendientes es {pen_soc['Socio'].iloc[0]} ({pen_soc['Horas'].iloc[0]:,.1f} h): "
                       "las asistentes deben gestionar su proforma.")
        sin_com = (pen["Comentario"] == "").mean()
        if sin_com > 0.05:
            ins.append(f"El {sin_com:.0%} de las horas pendientes no tiene comentario: riesgo de glosa o descuento al cobrar.")
    return {"kpis": kpis, "tablas": tablas, "insights": ins}


def _pct(a, b):
    return f"{a / b:.0%}" if b else "0%"


# --------------------------------------------------------------------- Excel
def to_excel(data: dict, filtros_txt: str = "", desde=None, hasta=None) -> bytes:
    a = analyze(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}) as xw:
        wb = xw.book
        base = {"font_name": "Calibri", "font_size": 10, "valign": "vcenter"}
        F = {
            "title": wb.add_format({"font_name": "Calibri", "bold": True, "font_size": 18, "font_color": VERDE}),
            "sub": wb.add_format({"font_name": "Calibri", "italic": True, "font_color": GRIS}),
            "hdr": wb.add_format({**base, "bold": True, "font_color": "#FFFFFF", "bg_color": VERDE, "text_wrap": True,
                                  "align": "center", "border": 1, "border_color": VERDE_CL}),
            "band": wb.add_format({**base, "bold": True, "font_color": "#FFFFFF", "bg_color": GRIS, "font_size": 12}),
            "kl": wb.add_format({**base, "font_color": GRIS, "bg_color": CREMA, "align": "center", "text_wrap": True,
                                 "top": 2, "top_color": VERDE_CL, "left": 1, "right": 1, "border_color": "#DDDDDD"}),
            "txt": wb.add_format({**base, "text_wrap": True, "valign": "top"}),
            "wrap": wb.add_format({"font_name": "Calibri", "font_size": 11, "text_wrap": True, "valign": "top"}),
        }
        numfmt = {"cop": "$ #,##0", "hrs": "#,##0.00", "pct": "0.0%", "int": "#,##0", "date": "dd/mm/yyyy"}
        cell = {k: wb.add_format({**base, "num_format": v, "valign": "top",
                                  **({"align": "center"} if k in ("date", "int") else {})}) for k, v in numfmt.items()}
        cell["txt"] = F["txt"]
        kpi = {k: wb.add_format({**base, "bold": True, "font_size": 15, "font_color": VERDE, "bg_color": CREMA,
                                 "align": "center", "num_format": v, "bottom": 2, "bottom_color": VERDE_CL,
                                 "left": 1, "right": 1, "border_color": "#DDDDDD"}) for k, v in numfmt.items()}

        bf = {"font_name": "Calibri", "bg_color": GRIS, "font_color": "#FFFFFF"}
        B = {"t": wb.add_format({**bf, "bold": True, "font_size": 16}),
             "b": wb.add_format({**bf, "bold": True, "font_size": 11}),
             "n": wb.add_format({**bf, "font_size": 11}),
             "bg": wb.add_format(bf), "line": wb.add_format({"bg_color": VERDE_CL})}

        def header(ws, title, sub, ncols, caso=None):
            ws.hide_gridlines(2)
            for r, h in enumerate([50, 26, 20, 20, 20, 10]):
                ws.set_row(r, h, B["line"] if r == 5 else B["bg"])
            ws.insert_image(0, 0, str(LOGO_BLANCO), {"x_scale": 0.3, "y_scale": 0.3, "x_offset": 12, "y_offset": 10,
                                                     "object_position": 3})
            ws.write(1, 0, "  " + title, B["t"])
            if caso:
                ws.write_rich_string(2, 0, B["b"], "  CASO: ", B["n"], caso, B["n"])
                ws.write(3, 0, "  " + periodo, B["n"])
            else:
                ws.write(2, 0, "  " + periodo, B["n"])
            ws.write_rich_string(4, 0, B["b"], "  ACTUALIZACIÓN: ", B["n"], fecha_larga(corte), B["n"])
            ws.set_landscape()
            ws.fit_to_pages(1, 0)
            ws.set_paper(9)

        corte = pd.Timestamp(hasta) if hasta is not None else pd.Timestamp(datetime.now().date())
        pend = data["pendientes"]
        fmin = pd.Timestamp(desde) if desde is not None else pend["Fecha Movimiento TR"].min()
        periodo = (f"Horas del {fecha_larga(fmin, True)} al {fecha_larga(corte)}" if pd.notna(fmin)
                   else f"Horas hasta el {fecha_larga(corte)}")
        casos = sorted({c for df in data.values() for c in df["Caso"].dropna().astype(str) if c.strip()})
        caso_txt = " | ".join(casos) if casos else "—"
        gen = filtros_txt
        sheets = [("facturacion", "Facturación", "DETALLE DE FACTURACIÓN", "Total COP"),
                  ("proformas", "Proformas", "DETALLE DE PROFORMAS POR FACTURAR", "Total COP"),
                  ("pendientes", "Horas Pendientes", "DETALLE DE TIEMPOS PENDIENTES POR FACTURAR", "Valor Estimado COP")]
        for key, sname, title, money in sheets:
            df = data[key].drop(columns=["Mes"])
            ws = wb.add_worksheet(sname)
            xw.sheets[sname] = ws
            ws.set_tab_color(VERDE)
            header(ws, title, gen, len(df.columns), caso_txt)
            # tarjetas resumen
            hrs = df["Horas"].sum()
            cards = [("Registros", len(df), "int"), ("Horas", hrs, "hrs"), ("Valor COP", df[money].sum(), "cop")]
            for i, (lab, val, f) in enumerate(cards):
                ws.write(7, i * 2, lab, F["kl"])
                ws.write(8, i * 2, val, kpi[f])
            ws.set_row(8, 26)
            top = 10
            _write_grouped(ws, df, top, DATE_COL[key], money, F, cell, WB_FMT(wb, base))
            ws.set_row(top, 32)
            ws.set_default_row(18)
            ws.repeat_rows(top)

        _write_analysis(wb, xw, a, F, cell, kpi, header, gen, caso_txt)
    return buf.getvalue()


def WB_FMT(wb, base):
    def mk(bg, fc="#000000", fmt=None, bold=True):
        d = {**base, "bold": bold, "bg_color": bg, "font_color": fc, "top": 1, "bottom": 1, "border_color": "#BBBBBB"}
        if fmt:
            d["num_format"] = fmt
        return wb.add_format(d)
    return {
        "mes": {"l": mk("#E3F1D3", fmt=None), "hrs": mk("#E3F1D3", fmt="#,##0.00"), "cop": mk("#E3F1D3", fmt="$ #,##0")},
        "anio": {"l": mk("#C4D82E"), "hrs": mk("#C4D82E", fmt="#,##0.00"), "cop": mk("#C4D82E", fmt="$ #,##0")},
        "tot": {"l": mk(GRIS, "#FFFFFF"), "hrs": mk(GRIS, "#FFFFFF", "#,##0.00"), "cop": mk(GRIS, "#FFFFFF", "$ #,##0")},
    }


def _write_grouped(ws, df, top, date_col, money, F, cell, S):
    """Tabla cronológica con filas de subtotal por mes, por año (si hay más de uno) y total general."""
    from xlsxwriter.utility import xl_col_to_name, xl_rowcol_to_cell

    d = df.sort_values(date_col, kind="stable", na_position="last").reset_index(drop=True)
    dt = d[date_col]
    d.insert(0, "Mes Reportado", dt.dt.month.map(lambda m: MESES_L[int(m) - 1].capitalize() if pd.notna(m) else ""))
    d.insert(0, "Año Reportado", dt.dt.year.astype("Int64"))
    cols = list(d.columns)
    sums = {c: cols.index(c) for c in ("Horas", money) if c in cols}
    for j, c in enumerate(cols):
        ws.write(top, j, c, F["hdr"])
        ws.set_column(j, j, WIDTH.get(c) or (11 if j < 2 else max(12, min(26, len(c) + 4))))
    cellfmt = {c: cell[FMT.get(c, "txt")] for c in cols}
    cellfmt["Año Reportado"] = cell["int"]
    cellfmt["Mes Reportado"] = cell["date"]
    multi_year = d["Año Reportado"].dropna().nunique() > 1
    r = top + 1

    def subtotal(kind, label, first, last, frame):
        ws.write(r, 2 if len(cols) > 2 else 0, label, S[kind]["l"])
        for j in range(len(cols)):
            if j in sums.values():
                c = [k for k, v in sums.items() if v == j][0]
                ref = f"{xl_rowcol_to_cell(first, j)}:{xl_rowcol_to_cell(last, j)}"
                ws.write_formula(r, j, f"=SUBTOTAL(109,{ref})", S[kind]["hrs" if c == "Horas" else "cop"],
                                 float(frame[c].sum()))
            elif j != 2:
                ws.write_blank(r, j, None, S[kind]["l"])

    first_row = r
    for y, gy in d.groupby("Año Reportado", dropna=False, sort=True):
        y_first = r
        for m, gm in gy.groupby(gy[date_col].dt.month, dropna=False, sort=True):
            m_first = r
            for row in gm.itertuples(index=False):
                for j, v in enumerate(row):
                    if v is None or (not isinstance(v, str) and pd.isna(v)):
                        ws.write_blank(r, j, None, cellfmt[cols[j]])
                    elif isinstance(v, pd.Timestamp):
                        ws.write_datetime(r, j, v.to_pydatetime(), cellfmt[cols[j]])
                    else:
                        ws.write(r, j, v, cellfmt[cols[j]])
                r += 1
            lab = f"TOTAL {MESES_L[int(m) - 1].upper()}" if pd.notna(m) else "TOTAL SIN FECHA"
            subtotal("mes", lab, m_first, r - 1, gm)
            r += 1
        if multi_year:
            subtotal("anio", f"TOTAL {int(y)}" if pd.notna(y) else "TOTAL SIN FECHA", y_first, r - 1, gy)
            r += 1
    subtotal("tot", "TOTAL", first_row, r - 1, d)
    ws.autofilter(top, 0, max(r - 1, top), len(cols) - 1)
    ws.freeze_panes(top + 1, 2)


def _write_analysis(wb, xw, a, F, cell, kpi, header, gen, caso_txt):
    ws = wb.add_worksheet("Análisis")
    xw.sheets["Análisis"] = ws
    ws.set_tab_color(LIMA)
    header(ws, "ANÁLISIS EJECUTIVO", gen, 8, caso_txt)
    ws.set_column(0, 0, 3)
    ws.set_column(1, 1, 42)
    ws.set_column(2, 7, 17)
    ws.set_column(8, 8, 3)
    ws.set_column(9, 18, 10)

    r = 7
    # tarjetas KPI: 4 por fila
    layout = [(1, 1), (2, 3), (4, 5), (6, 7)]
    for i, (lab, val, f) in enumerate(a["kpis"]):
        row = r + (i // 4) * 3
        c0, c1 = layout[i % 4]
        if c0 == c1:
            ws.write(row, c0, lab, F["kl"])
            ws.write(row + 1, c0, val, kpi[f])
        else:
            ws.merge_range(row, c0, row, c1, lab, F["kl"])
            ws.merge_range(row + 1, c0, row + 1, c1, val, kpi[f])
        ws.set_row(row, 28)
        ws.set_row(row + 1, 28)
    r += 7

    ws.merge_range(r, 1, r, 7, "Hallazgos clave", F["band"])
    r += 1
    for t in a["insights"]:
        ws.merge_range(r, 1, r, 7, "•  " + t, F["wrap"])
        ws.set_row(r, 32)
        r += 1
    r += 1

    for title, df, ctype, cat, val in a["tablas"]:
        ws.merge_range(r, 1, r, 7, title, F["band"])
        r += 1
        if df.empty:
            ws.write(r, 1, "Sin datos", F["txt"])
            r += 2
            continue
        for j, c in enumerate(df.columns):
            ws.write(r, 1 + j, c, F["hdr"])
        for i, row in enumerate(df.itertuples(index=False)):
            for j, (c, v) in enumerate(zip(df.columns, row)):
                f = _afmt(c)
                if v is None or (not isinstance(v, str) and pd.isna(v)):
                    ws.write_blank(r + 1 + i, 1 + j, None, cell[f])
                else:
                    ws.write(r + 1 + i, 1 + j, v, cell[f])
        first, last = r + 1, r + len(df)
        if ctype:
            ch = wb.add_chart({"type": "bar" if ctype == "bar" else "column"})
            ci, vi = list(df.columns).index(cat), list(df.columns).index(val)
            ch.add_series({
                "name": val, "categories": ["Análisis", first, 1 + ci, last, 1 + ci],
                "values": ["Análisis", first, 1 + vi, last, 1 + vi],
                "fill": {"color": VERDE}, "gap": 60,
                "data_labels": {"value": False},
            })
            ch.set_legend({"none": True})
            ch.set_title({"name": title, "name_font": {"size": 10, "color": GRIS}})
            ch.set_chartarea({"border": {"color": "#DDDDDD"}})
            if ctype == "bar":
                ch.set_y_axis({"reverse": True})
            ch.set_size({"width": 520, "height": max(240, 24 * (len(df) + 4))})
            ws.insert_chart(r, 9, ch)
        r = max(last + 2, r + (13 if ctype else 0) + (max(len(df) - 10, 0) if ctype else 0)) + 1


def _afmt(c: str) -> str:
    if c.startswith("%") or "Realización" in c or "Variación" in c:
        return "pct"
    if c in ("Horas", "Antigüedad prom. (días)", "Antigüedad máx. (días)") or c.startswith("Horas"):
        return "hrs" if c.startswith("Horas") else "int"
    if c in ("Registros",):
        return "int"
    if any(k in c for k in ("COP", "Valor", "Tarifa")):
        return "cop"
    return "txt"
