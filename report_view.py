import pandas as pd
import plotly.express as px
import streamlit as st

import reporting as rp


def style_page():
    st.markdown(
        f"""<style>
    .stApp {{background: linear-gradient(160deg,#ffffff 0%,#f2f8ec 55%,#e3f1d3 100%);}}
    [data-testid="stSidebar"] {{background:#f4f6f2;border-right:3px solid {rp.VERDE_CL};}}
    h1,h2,h3 {{color:{rp.GRIS};}}
    [data-testid="stMetric"] {{background:#fff;border-left:6px solid {rp.VERDE};border-radius:8px;
        padding:12px 16px;box-shadow:0 1px 6px rgba(0,0,0,.08);}}
    [data-testid="stMetricValue"] {{color:{rp.VERDE};}}
    .stTabs [aria-selected="true"] {{color:{rp.VERDE};border-bottom-color:{rp.VERDE};}}
    </style>""",
        unsafe_allow_html=True,
    )


def render(full: dict, file_name: str = "reporte_facturacion_horas.xlsx"):
    """Filtros, KPIs, hallazgos, tablas y descarga para un informe {'facturacion','proformas','pendientes'}."""
    # ---------------------------------------------------------------- filtros
    st.sidebar.header("Filtros del informe")
    dates = pd.concat([full[k][c] for k, c in rp.DATE_COL.items()]).dropna()
    desde = hasta = None
    if not dates.empty:
        dmin, dmax = dates.min().date(), dates.max().date()
        rango = st.sidebar.date_input("Rango de fechas (factura / documento / movimiento TR)", (dmin, dmax), dmin, dmax)
        if len(rango) == 2:
            desde, hasta = rango


    def opciones(col):
        return sorted({v for df in full.values() if col in df for v in df[col].dropna().unique() if v != ""})


    cols_sel = {}
    for col in ["Área", "Socio", "Cliente"]:
        v = st.sidebar.multiselect(col, opciones(col))
        if v:
            cols_sel[col] = v
    with st.sidebar.expander("Filtro avanzado (cualquier columna)"):
        extra = sorted({c for df in full.values() for c in df.columns
                        if df[c].dtype == object and c not in ("Comentario", "Mes")} - {"Área", "Socio", "Cliente"})
        for col in st.multiselect("Columnas a filtrar", extra):
            v = st.multiselect(col, opciones(col), key=f"adv_{col}")
            if v:
                cols_sel[col] = v

    data = rp.apply_filters(full, desde, hasta, cols_sel)
    a = rp.analyze(data)

    filtros = "; ".join(f"{k}: {', '.join(map(str, v[:3]))}{'…' if len(v) > 3 else ''}" for k, v in cols_sel.items())
    if desde:
        filtros = f"{desde:%d/%m/%Y} a {hasta:%d/%m/%Y}" + (f" | {filtros}" if filtros else "")
    st.sidebar.download_button(
        "⬇️ Descargar informe (Excel)", rp.to_excel(data, filtros, desde, hasta), file_name,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
    )

    # ------------------------------------------------------------------ vista
    fmt = {"cop": "${:,.0f}", "hrs": "{:,.1f}", "pct": "{:.1%}", "int": "{:,.0f}"}
    cols = st.columns(4)
    for i, (lab, val, f) in enumerate(a["kpis"]):
        cols[i % 4].metric(lab, fmt[f].format(val))

    st.subheader("Hallazgos clave")
    for t in a["insights"]:
        st.markdown(f"- {t}")

    tabs = st.tabs(["Facturación", "Proformas", "Horas pendientes", "Análisis"])
    for tab, key in zip(tabs[:3], ["facturacion", "proformas", "pendientes"]):
        with tab:
            df = data[key].drop(columns="Mes")
            st.caption(f"{len(df):,} registros")
            st.dataframe(
                df.style.format({c: fmt[rp.FMT[c]] for c in df.columns if rp.FMT.get(c) in fmt}
                                | {c: "{:%d/%m/%Y}" for c in df.columns if rp.FMT.get(c) == "date"}, na_rep=""),
                width="stretch", hide_index=True,
            )
    with tabs[3]:
        for title, df, ctype, cat, val in a["tablas"]:
            st.markdown(f"**{title}**")
            if df.empty:
                st.write("Sin datos")
                continue
            if ctype:
                fig = px.bar(df, x=val if ctype == "bar" else cat, y=cat if ctype == "bar" else val,
                             orientation="h" if ctype == "bar" else "v", color_discrete_sequence=[rp.VERDE])
                fig.update_layout(height=300, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
                if ctype == "bar":
                    fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, width="stretch")
            st.dataframe(df.style.format({c: fmt[rp._afmt(c)] for c in df.columns if rp._afmt(c) in fmt}, na_rep=""),
                         width="stretch", hide_index=True)
