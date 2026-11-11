import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecasting as fc

VERDE, VERDE_CL, LIMA, GRIS = "#007A3D", "#8DC63F", "#C4D82E", "#6D6E71"

st.set_page_config(page_title="Forecast Facturación | PHR", page_icon="🟢", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{background: linear-gradient(160deg,#ffffff 0%,#f2f8ec 55%,#e3f1d3 100%);}}
    [data-testid="stSidebar"] {{background:#f4f6f2;border-right:3px solid {VERDE_CL};}}
    h1,h2,h3 {{color:{GRIS};}}
    [data-testid="stMetric"] {{background:#fff;border-left:6px solid {VERDE};border-radius:8px;
        padding:12px 16px;box-shadow:0 1px 6px rgba(0,0,0,.08);}}
    [data-testid="stMetricValue"] {{color:{VERDE};}}
    .stTabs [aria-selected="true"] {{color:{VERDE};border-bottom-color:{VERDE};}}
    </style>""",
    unsafe_allow_html=True,
)
st.image("logo-phr-horizontal-1024x93.png", width=520)
st.title("Pronóstico de facturación y presupuesto diario")


def cop(v):
    return f"${v / 1e6:,.0f} M"


def style(fig, h=420):
    fig.update_layout(
        height=h, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.7)", legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


@st.cache_data
def load(src):
    return fc.load_data(src)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Datos")
up = st.sidebar.file_uploader("Excel (Area, FechaFctd, TotalFactPesos)", type="xlsx")
try:
    df = load(up if up else "datapreview.xlsx")
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

st.sidebar.header("Filtros")
areas = sorted(df["Area"].unique())
sel = st.sidebar.multiselect("Áreas", areas, default=areas)
fmin, fmax = df["Fecha"].min().date(), df["Fecha"].max().date()
rango = st.sidebar.date_input("Rango de fechas históricas", (fmin, fmax), fmin, fmax)
if len(rango) != 2 or not sel:
    st.warning("Selecciona al menos un área y un rango de fechas válido.")
    st.stop()
dff = df[df["Area"].isin(sel) & df["Fecha"].between(pd.Timestamp(rango[0]), pd.Timestamp(rango[1]))]
if dff.empty:
    st.warning("Sin datos para los filtros elegidos.")
    st.stop()

st.sidebar.header("Parámetros")
last_year = dff["Fecha"].max().year
end_year = st.sidebar.selectbox("Pronosticar hasta", [last_year + 1, last_year + 2], 0,
                                format_func=lambda y: f"Dic {y}")
modelo = st.sidebar.selectbox("Modelo", ["Automático (mejor en backtest)", fc.ENSEMBLE, *fc.BASE_MODELS])
adj = st.sidebar.slider("Ajuste anual (inflación/tarifas) %", 0.0, 15.0, 5.0, 0.5) / 100
winsor = st.sidebar.checkbox("Suavizar facturas atípicas (p98) — resta fuerza al cierre de diciembre", False)
smooth = st.sidebar.slider("Suavizado curva intra-mes", 0.0, 1.0, 0.4, 0.1)
stretch = st.sidebar.slider("Meta sobre pronóstico (presupuesto) %", 0.0, 20.0, 5.0, 1.0) / 100
flat = st.sidebar.slider("Uniformidad del presupuesto diario", 0.0, 1.0, 0.5, 0.1)

with st.sidebar.expander("Estacionalidad mensual (editable)"):
    base_idx = fc.monthly_seasonality(fc.monthly_table(fc.daily_series(dff)))
    over = {m: st.number_input(fc.MESES[m - 1], 0.1, 3.0, round(float(base_idx[m]), 2), 0.01)
            for m in range(1, 13)}

res = fc.run_forecast(
    dff, None if modelo.startswith("Auto") else modelo, pd.Period(f"{end_year}-12"),
    adj, 0.98 if winsor else None, smooth, stretch, flat, over,
)
mf, dfc = res["monthly_fc"], res["daily_fc"]
mf["Mes_dt"] = pd.PeriodIndex(mf["Mes"], freq="M").to_timestamp()

if not res["validated"]:
    st.info("Historia < 4 meses: el modelo no se puede validar con backtest. Se usa run-rate "
            "desestacionalizado con estacionalidad de referencia. Carga más historia para "
            "que el sistema elija el mejor modelo.")
elif res["uses_prior"]:
    st.caption("Estacionalidad mensual basada parcialmente en valores de referencia del sector.")

nov = mf[mf["Mes"] == f"{last_year}-11"]
next_year = mf[mf["Mes"].str.startswith(str(last_year + 1))]
c = st.columns(4)
c[0].metric("Modelo", res["model"].split(" (")[0])
if not nov.empty:
    c[1].metric("Cierre Nov (facturado + proyectado)", cop(nov["Pronóstico"].iloc[0]),
                f"P10–P90: {cop(nov['P10'].iloc[0])} – {cop(nov['P90'].iloc[0])}", delta_color="off")
    c[2].metric("Presupuesto Nov", cop(nov["Presupuesto ideal"].iloc[0]),
                f"Facturado a la fecha: {cop(nov['Facturado a la fecha'].iloc[0])}", delta_color="off")
c[3].metric(f"Total {last_year + 1}", cop(next_year["Pronóstico"].sum()) if not next_year.empty else "–")

t1, t2, t3, t4, t5 = st.tabs(["Mensual", "Diario", "Estacionalidad", "Presupuesto diario", "Validación"])

with t1:
    h = res["monthly_hist"].reset_index()
    h["Mes_dt"] = h["Mes"].dt.to_timestamp() if "Mes" in h else h["index"].dt.to_timestamp()
    fig = go.Figure()
    fig.add_bar(x=h["Mes_dt"], y=h["total"], name="Histórico", marker_color=GRIS)
    fig.add_bar(x=mf["Mes_dt"], y=mf["Pronóstico"], name="Pronóstico", marker_color=VERDE_CL,
                error_y=dict(type="data", symmetric=False, array=mf["P90"] - mf["Pronóstico"],
                             arrayminus=mf["Pronóstico"] - mf["P10"], color=VERDE))
    fig.add_scatter(x=mf["Mes_dt"], y=mf["Presupuesto ideal"], name="Presupuesto", mode="lines+markers",
                    line=dict(color=LIMA, width=3))
    st.plotly_chart(style(fig), width="stretch")
    st.dataframe(mf.drop(columns="Mes_dt").style.format(
        {c: "{:,.0f}" for c in ["Facturado a la fecha", "Pronóstico", "P10", "P90", "Presupuesto ideal"]} | {"Índice estacional": "{:.2f}"}),
        width="stretch", hide_index=True)

with t2:
    mes = st.selectbox("Mes", mf["Mes"], key="mes_d")
    d = dfc[dfc["Mes"] == mes]
    hist = res["daily_hist"]
    fig = go.Figure()
    fig.add_bar(x=hist.index, y=hist.values, name="Histórico", marker_color=GRIS)
    fig.add_bar(x=d["Fecha"], y=d["Pronóstico"], name="Pronóstico", marker_color=VERDE_CL)
    fig.add_scatter(x=d["Fecha"], y=d["Pronóstico acum. mes"], name="Acumulado", yaxis="y2",
                    line=dict(color=VERDE, width=3))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(style(fig), width="stretch")
    st.dataframe(d.style.format({c: "{:,.0f}" for c in d.columns if c not in ("Fecha", "Mes")}),
                 width="stretch", hide_index=True)

with t3:
    a, b = st.columns(2)
    s = res["seasonality"]
    fig = go.Figure(go.Bar(x=fc.MESES, y=s.values, marker_color=[VERDE if v >= 1 else VERDE_CL for v in s.values]))
    fig.add_hline(y=1, line_dash="dash", line_color=GRIS)
    a.subheader("Índice mensual")
    a.plotly_chart(style(fig, 320), width="stretch")
    w = res["weekday"]
    fig = go.Figure(go.Bar(x=fc.DIAS, y=w.values, marker_color=VERDE))
    fig.add_hline(y=1, line_dash="dash", line_color=GRIS)
    b.subheader("Índice por día de la semana")
    b.plotly_chart(style(fig, 320), width="stretch")
    fig = go.Figure(go.Scatter(x=fc.GRID * 100, y=res["curve"] * 100, line=dict(color=VERDE, width=3), name="Curva"))
    fig.add_scatter(x=fc.GRID * 100, y=fc.GRID * 100, line=dict(color=GRIS, dash="dash"), name="Uniforme")
    fig.update_layout(xaxis_title="% del mes transcurrido", yaxis_title="% facturado acumulado")
    st.subheader("Avance de facturación dentro del mes")
    st.plotly_chart(style(fig, 340), width="stretch")
    ar = dff.groupby("Area")["Valor"].sum().sort_values()
    st.subheader("Mix por área (histórico filtrado)")
    st.plotly_chart(style(go.Figure(go.Bar(x=ar.values, y=ar.index, orientation="h", marker_color=VERDE_CL)), 420),
                    width="stretch")

with t4:
    mes = st.selectbox("Mes", mf["Mes"], key="mes_b")
    d = dfc[dfc["Mes"] == mes]
    fig = go.Figure()
    fig.add_bar(x=d["Fecha"], y=d["Presupuesto ideal"], name="Presupuesto diario", marker_color=VERDE)
    fig.add_scatter(x=d["Fecha"], y=d["Pronóstico"], name="Pronóstico", line=dict(color=LIMA, width=3))
    fig.add_scatter(x=d["Fecha"], y=d["Presupuesto acum. mes"], name="Presupuesto acum.", yaxis="y2",
                    line=dict(color=GRIS, dash="dot"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(style(fig), width="stretch")
    st.dataframe(d[["Fecha", "Presupuesto ideal", "Presupuesto acum. mes"]].style.format(
        {"Presupuesto ideal": "{:,.0f}", "Presupuesto acum. mes": "{:,.0f}"}), width="stretch", hide_index=True)

with t5:
    if res["metrics"].empty:
        st.write("Se requieren al menos 4 meses de historia para el backtest (origen móvil a 1 mes).")
    else:
        st.write("Backtest de origen móvil (WAPE: menor es mejor). El modo automático elige el de menor WAPE.")
        st.dataframe(res["metrics"], width="stretch")
        st.dataframe(res["backtest"], width="stretch", hide_index=True)

# --------------------------------------------------------------- descarga
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
    mf.drop(columns="Mes_dt").to_excel(xw, sheet_name="Mensual", index=False)
    dfc.to_excel(xw, sheet_name="Diario", index=False)
    dfc[["Fecha", "Mes", "Presupuesto ideal", "Presupuesto acum. mes"]].to_excel(xw, sheet_name="Presupuesto", index=False)
    pd.DataFrame({"Mes": fc.MESES, "Índice": res["seasonality"].values}).to_excel(xw, sheet_name="Estacionalidad", index=False)
    pd.DataFrame({"Día": fc.DIAS, "Índice": res["weekday"].values}).to_excel(xw, sheet_name="Dia semana", index=False)
    if not res["metrics"].empty:
        res["metrics"].to_excel(xw, sheet_name="Backtest")
st.sidebar.download_button("⬇️ Descargar proyectado (Excel)", buf.getvalue(), "proyeccion_facturacion.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.sidebar.download_button("⬇️ Descargar diario (CSV)", dfc.to_csv(index=False).encode(), "proyeccion_diaria.csv")
