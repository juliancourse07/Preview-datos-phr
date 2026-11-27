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
st.caption("1) Sube tus bases · 2) Filtra lo que te interesa · 3) Define qué es facturado y pendiente · 4) Mira el informe.")

TIPOS = ["Facturación", "Horas", "Ignorar"]
SPECS = {"Facturación": rp.FACT_SPEC, "Horas": rp.HORAS_SPEC}
OBLIG = {"Facturación": ["Total COP", "Estado"], "Horas": ["Horas", "Estado"]}
VACIO = "(vacío)"
FILTRABLES = ["Área", "Socio", "Cliente", "Holding", "Caso", "Empleado", "Categoría", "Facturable"]


# Resultados cacheados como recursos (sin copias): nunca se modifican en sitio.
@st.cache_resource(show_spinner="Leyendo archivo…")
def leer(nombre, tam, data):
    return rp.read_sheets(io.BytesIO(data))


@st.cache_resource(show_spinner=False)
def estandarizar(_raw, firma, tipo, mapping):
    return rp.standardize(rp.remap(_raw, SPECS[tipo], dict(mapping)), SPECS[tipo])


@st.cache_resource(show_spinner="Calculando informe…")
def informe(_f, _h, firma, est_f, solo_fact, est_p):
    return rp.build_from_std(_f, _h, list(est_f), solo_fact, list(est_p))


def err(msg):
    st.error(msg)
    st.stop()


# ------------------------------------------------------------ 1. archivos
st.subheader("1. Sube tus bases")
files = st.file_uploader("Excel (uno o varios; pueden traer varias hojas)", type=["xlsx"], accept_multiple_files=True)
if not files:
    st.info("Sube al menos un Excel. Basta con uno (por ejemplo solo Facturación).")
    st.stop()

hojas = {}
for f in files:
    try:
        for sh, df in leer(f.name, f.size, f.getvalue()).items():
            if len(df.columns):
                hojas[(f.name, f.size, sh)] = df
    except Exception as e:
        st.warning(f"No se pudo leer «{f.name}»: {e}")
if not hojas:
    err("Ningún archivo se pudo leer. Verifica que sean .xlsx válidos.")

std = {"Facturación": [], "Horas": []}
firmas = []
for (fn, tam, sh), raw in hojas.items():
    auto = {"facturacion": 0, "horas": 1}.get(rp.kind_of(raw), 2)
    c1, c2 = st.columns([4, 1])
    c1.markdown(f"**{fn}** · hoja «{sh}» · {len(raw):,} filas")
    tipo = c2.selectbox("Tipo de hoja", TIPOS, auto, key=f"t_{fn}_{tam}_{sh}", label_visibility="collapsed")
    if tipo == "Ignorar":
        continue
    spec = SPECS[tipo]
    mapping = {n: (rp.find_col(raw, c) or [""])[0] for n, c, _ in spec}
    faltan = [n for n in OBLIG[tipo] if not mapping[n]]
    if faltan:
        st.caption(f"⚠️ No encontré {', '.join(faltan)}; indícalo:")
        g = st.columns(len(faltan))
        for col, n in zip(g, faltan):
            mapping[n] = col.selectbox(n, [""] + list(raw.columns), key=f"m_{fn}_{tam}_{sh}_{n}")
        if any(not mapping[n] for n in faltan):
            err(f"Falta indicar {', '.join(n for n in faltan if not mapping[n])} en «{sh}» (o márcala como Ignorar).")
    firma = (fn, tam, sh, tipo, tuple(sorted(mapping.items())))
    firmas.append(firma)
    std[tipo].append(estandarizar(raw, firma[:3], tipo, firma[4]))

if not std["Facturación"]:
    err("Marca al menos una hoja como «Facturación».")


def unir(lst, spec):
    df = pd.concat(lst, ignore_index=True).drop_duplicates(ignore_index=True) if lst else rp.standardize(pd.DataFrame(), spec)
    df["Estado"] = df["Estado"].str.upper()
    return df


f_all, h_all = unir(std["Facturación"], rp.FACT_SPEC), unir(std["Horas"], rp.HORAS_SPEC)
if h_all.empty:
    st.caption("Sin hoja de Horas: no habrá horas pendientes.")

# ------------------------------------------------------------ 2. filtros previos
st.subheader("2. Filtra antes de calcular (opcional)")
st.caption("Los filtros se aplican a todas las bases ANTES de definir facturado/pendiente.")
disp = [c for c in FILTRABLES if c in f_all or c in h_all]
elegidas = st.multiselect("Columnas por las que quieres filtrar", disp, placeholder="Ej.: Área, Socio, Cliente")
filtros = {}
if elegidas:
    cols = st.columns(min(3, len(elegidas)))
    for i, c in enumerate(elegidas):
        vals = sorted({v for d in (f_all, h_all) if c in d for v in d[c].unique() if v != ""})
        sel = cols[i % len(cols)].multiselect(c, vals, key=f"flt_{c}", placeholder="Todos")
        if sel:
            filtros[c] = sel


def aplicar(df):
    m = pd.Series(True, index=df.index)
    for c, v in filtros.items():
        if c in df:
            m &= df[c].isin(v)
    return df[m].reset_index(drop=True)


f_use, h_use = aplicar(f_all), aplicar(h_all)
st.caption(f"Facturación: {len(f_use):,} de {len(f_all):,} filas · Horas: {len(h_use):,} de {len(h_all):,} filas")
if f_use.empty:
    err("Con esos filtros no queda ninguna fila de Facturación. Quita algún filtro.")

# ------------------------------------------------------------ 3. reglas
st.subheader("3. Define los indicadores")


def opciones(df):
    cuenta = df["Estado"].replace("", VACIO).value_counts()
    return list(cuenta.index), {k: f"{k}  ({v:,} filas)" for k, v in cuenta.items()}


op_f, lab_f = opciones(f_use)
op_h, lab_h = opciones(h_use) if len(h_use) else ([], {})
c1, c2 = st.columns(2)
sel_fact = c1.multiselect("**Facturación** = filas con estado…", op_f, [e for e in op_f if e == "F"] or op_f[:1],
                          format_func=lab_f.get, help="El resto de lo facturable cuenta como proforma pendiente.")
solo_fact = c1.checkbox("Contar solo horas facturables", True)
sel_pen = c2.multiselect("**Horas pendientes** = horas con estado…", op_h, [VACIO] if VACIO in op_h else [],
                         format_func=lab_h.get, help="Normalmente las que aún no tienen proforma (vacío).")
c2.caption("**Proformas pendientes** = facturación facturable que NO está en los estados de arriba.")
if not sel_fact:
    err("Elige al menos un estado para «Facturación».")


def quitar(xs):
    return tuple("" if e == VACIO else e for e in xs)


clave = (tuple(firmas), tuple(sorted((k, tuple(v)) for k, v in filtros.items())))
try:
    full = informe(f_use, h_use, clave, quitar(sel_fact), solo_fact, quitar(sel_pen))
except Exception as e:
    err(f"No pude calcular el informe: {e}")

# ------------------------------------------------------------ 4. informe
st.subheader("4. Informe")
try:
    render(full, "reporte_abierto.xlsx")
except Exception as e:
    st.error(f"No pude mostrar el informe con estos datos: {e}")
