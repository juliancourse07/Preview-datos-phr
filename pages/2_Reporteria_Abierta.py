import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reporting as rp  # noqa: E402
from report_view import render, style_page  # noqa: E402

st.set_page_config(page_title="Reportería abierta | PHR", page_icon="🟢", layout="wide")
style_page()
st.image(str(rp.LOGO), width=520)
st.title("Reportería abierta")
st.caption("Sube una o varias bases (cada archivo puede tener varias hojas), indica qué es cada hoja, "
           "ajusta las columnas y define los indicadores. La salida es la misma del informe estándar.")

TIPOS = ["Facturación", "Horas", "Ignorar"]
SPECS = {"Facturación": rp.FACT_SPEC, "Horas": rp.HORAS_SPEC}
OBLIG = {"Facturación": ["Total COP", "Estado"], "Horas": ["Horas", "Estado"]}


@st.cache_data(show_spinner="Leyendo archivos…")
def leer(nombre, data):
    return rp.read_sheets(io.BytesIO(data))


files = st.file_uploader("Bases en Excel (una o varias)", type=["xlsx"], accept_multiple_files=True)
if not files:
    st.info("Sube al menos una base. Puede ser una sola (p. ej. solo Facturación) o varias.")
    st.stop()

hojas = {}
for f in files:
    try:
        for sh, df in leer(f.name, f.getvalue()).items():
            hojas[(f.name, sh)] = df
    except Exception as e:
        st.error(f"No se pudo leer {f.name}: {e}")
if not hojas:
    st.stop()

# ----------------------------------------------------------- 1. tipo de hoja
st.subheader("1. ¿Qué contiene cada hoja?")
asign = {}
for (fn, sh), df in hojas.items():
    k = rp.kind_of(df)
    default = {"facturacion": 0, "horas": 1}.get(k, 2)
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**{fn}** · hoja «{sh}» · {len(df):,} filas")
    asign[(fn, sh)] = c2.selectbox("Tipo", TIPOS, default, key=f"t_{fn}_{sh}", label_visibility="collapsed")

# ------------------------------------------------- 2. mapeo de columnas
st.subheader("2. Columnas")
std = {"Facturación": [], "Horas": []}
for (fn, sh), tipo in asign.items():
    if tipo == "Ignorar":
        continue
    raw, spec = hojas[(fn, sh)], SPECS[tipo]
    cols = [""] + list(raw.columns)
    mapping = {}
    with st.expander(f"{tipo}: {fn} · {sh} (detección automática, editable)"):
        g = st.columns(3)
        for i, (name, cands, _) in enumerate(spec):
            auto = rp.find_col(raw, cands)
            mapping[name] = g[i % 3].selectbox(
                name, cols, cols.index(auto[0]) if auto else 0, key=f"m_{fn}_{sh}_{name}")
    faltan = [n for n in OBLIG[tipo] if not mapping.get(n)]
    if faltan:
        st.warning(f"{fn} · {sh}: falta asignar {', '.join(faltan)}.")
    std[tipo].append(rp.standardize(rp.remap(raw, spec, mapping), spec))

if not std["Facturación"]:
    st.warning("Marca al menos una hoja como Facturación.")
    st.stop()


def unir(lst, spec):
    if lst:
        return pd.concat(lst, ignore_index=True)
    return rp.standardize(pd.DataFrame(), spec)


f_all, h_all = unir(std["Facturación"], rp.FACT_SPEC), unir(std["Horas"], rp.HORAS_SPEC)
f_all["Estado"], h_all["Estado"] = f_all["Estado"].str.upper(), h_all["Estado"].str.upper()
if len(std["Facturación"]) > 1 or len(std["Horas"]) > 1:
    st.caption("Varias hojas del mismo tipo se apilan y se eliminan filas totalmente duplicadas.")
    f_all, h_all = f_all.drop_duplicates(ignore_index=True), h_all.drop_duplicates(ignore_index=True)
if not len(h_all):
    st.caption("Sin hoja de Horas: no habrá horas pendientes ni comentarios cruzados.")

# --------------------------------------------------- 3. indicadores clave
st.subheader("3. Indicadores clave")
VACIO = "(vacío)"
est_f = sorted(set(f_all["Estado"]) - {""}) + ([VACIO] if (f_all["Estado"] == "").any() else [])
est_h = sorted(set(h_all["Estado"]) - {""}) + ([VACIO] if (h_all["Estado"] == "").any() else [])
c1, c2, c3 = st.columns(3)
sel_fact = c1.multiselect("Estados que cuentan como FACTURADO", est_f, [e for e in est_f if e == "F"],
                          help="Todo lo demás (facturable) se considera proforma pendiente.")
solo_fact = c1.checkbox("Solo horas facturables (S o vacío)", True)
sel_pen = c2.multiselect("Estados de HORAS PENDIENTES", est_h, [VACIO] if VACIO in est_h else [],
                         help="Por defecto, horas sin proforma asociada.")
c3.markdown("**Proformas pendientes** = registros de Facturación que no están en los estados facturados.")
if not sel_fact:
    st.warning("Elige al menos un estado de facturado.")
    st.stop()

full = rp.build_from_std(
    f_all, h_all, [("" if e == VACIO else e) for e in sel_fact], solo_fact,
    [("" if e == VACIO else e) for e in sel_pen],
)
render(full, "reporte_abierto.xlsx")
