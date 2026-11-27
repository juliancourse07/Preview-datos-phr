import io
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reporting as rp  # noqa: E402
from report_view import render, style_page  # noqa: E402

st.set_page_config(page_title="Reportería | PHR", page_icon="🟢", layout="wide")
style_page()
st.image(str(rp.LOGO), width=520)
st.title("Reportería: Facturación, Proformas y Horas pendientes")
st.caption("Carga el Excel de Facturación y el de Horas; se cruzan por «Numero Serial» y se genera el informe ejecutivo.")


@st.cache_data(show_spinner="Procesando archivos…")
def process(fb, hb):
    return rp.build_report(rp.read_excel(io.BytesIO(fb)), rp.read_excel(io.BytesIO(hb)))


c1, c2 = st.columns(2)
up_f = c1.file_uploader("Excel de Facturación", type=["xlsx"])
up_h = c2.file_uploader("Excel de Horas", type=["xlsx"])
if not (up_f and up_h):
    st.info("Sube ambos archivos para generar el informe.")
    st.stop()
try:
    full = process(up_f.getvalue(), up_h.getvalue())
except Exception as e:
    st.error(f"No se pudo procesar: {e}")
    st.stop()

render(full)
