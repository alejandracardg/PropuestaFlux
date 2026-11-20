"""Freixenet FLUX · simulador educativo.

Vista del proyector:  https://<tu-app>.streamlit.app/
Mando externo (QR):   https://<tu-app>.streamlit.app/?vista=mando
"""
from __future__ import annotations

import io
import threading
import time
import uuid

import altair as alt
import pandas as pd
import qrcode
import streamlit as st

from engine import (AREAS, INCERTIDUMBRE, INCIDENCIAS, Escenario, Supuestos, comparar, configurar,
                    entrenar_modelo, organigrama)

st.set_page_config(page_title="Freixenet FLUX", page_icon="🍾", layout="wide")

URL_POR_DEFECTO = "https://freixenet-flux.streamlit.app"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@400;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, span, div { font-family: 'Lato', sans-serif; }
h1, h2, h3, h1 span, h2 span, h3 span, .flux-title { font-family: 'Playfair Display', serif !important; letter-spacing: 0; font-weight:700; }
.flux-dec { border-left:3px solid #BC9A4C; background:#121212; padding:8px 12px; margin-bottom:8px; }
.flux-dec b { color:#BC9A4C; }
.stApp { background-color:#070707; background-image:
  linear-gradient(#141414 1px, transparent 1px), linear-gradient(90deg, #141414 1px, transparent 1px);
  background-size: 28px 28px; }
.flux-kicker { color:#BC9A4C; font-weight:700; letter-spacing:.18em; text-transform:uppercase; font-size:.78rem; }
.flux-card { border:1px solid #2E2E2E; background:#121212; padding:16px 18px; border-radius:4px; }
.flux-card.gold { border-color:#BC9A4C; box-shadow:0 0 18px rgba(188,154,76,.35); }
.flux-big { font-family:'Playfair Display', serif; font-size:3rem; color:#BC9A4C; line-height:1.05; font-weight:700; }
.flux-mode { font-family:'Playfair Display', serif; font-size:1.6rem; color:#F4F1EA; }
.flux-muted { color:#9C9C9C; font-size:.9rem; }
.flux-band { background:#6E5A2C; color:#F4F1EA; padding:10px 16px; border-radius:3px; }
div[data-testid="stMetricValue"] { font-family:'Playfair Display', serif; color:#BC9A4C; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- recursos compartidos
@st.cache_resource
def modelo():
    return entrenar_modelo()


@st.cache_resource
def almacen():
    """Escenario compartido entre el mando externo (móvil) y el proyector (mismo proceso del servidor)."""
    return {"escenario": Escenario(), "t": None, "lock": threading.Lock()}


def guardar_escenario(e: Escenario) -> None:
    a = almacen()
    with a["lock"]:
        a["escenario"], a["t"] = e, time.time()


def leer_escenario() -> tuple[Escenario, float | None]:
    a = almacen()
    with a["lock"]:
        return a["escenario"], a["t"]


def leer_controles(prefijo: str) -> Escenario:
    ss = st.session_state
    return Escenario(ss[prefijo + "dem"], ss[prefijo + "disp"], INCIDENCIAS.index(ss[prefijo + "inc"]),
                     ss[prefijo + "mer"], INCERTIDUMBRE.index(ss[prefijo + "inc2"]))


def controles(prefijo: str, por_defecto: Escenario = Escenario(), al_cambiar=None) -> Escenario:
    kw = {"on_change": al_cambiar} if al_cambiar else {}
    st.slider("Variación de la demanda (%)", -30, 80, int(por_defecto.demanda), 5, key=prefijo + "dem",
              help="Cuánto se aleja la demanda de lo previsto para la campaña", **kw)
    st.slider("Producto disponible (% de lo previsto)", 40, 120, int(por_defecto.disponibilidad), 5, key=prefijo + "disp", **kw)
    st.select_slider("Incidencias logísticas", options=INCIDENCIAS, value=INCIDENCIAS[por_defecto.incidencia], key=prefijo + "inc", **kw)
    st.slider("Mercados afectados", 1, 6, por_defecto.mercados, key=prefijo + "mer", **kw)
    st.select_slider("Incertidumbre de las previsiones", options=INCERTIDUMBRE, value=INCERTIDUMBRE[por_defecto.incertidumbre], key=prefijo + "inc2", **kw)
    return leer_controles(prefijo)


def qr_png(url: str) -> bytes:
    q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=2)
    q.add_data(url)
    q.make(fit=True)
    img = q.make_image(fill_color=(5, 5, 5), back_color=(244, 241, 234))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def url_por_defecto() -> str:
    try:
        return st.secrets.get("APP_URL", URL_POR_DEFECTO)
    except Exception:
        return URL_POR_DEFECTO


# ---------------------------------------------------------------- vista mando (móvil)
def vista_mando() -> None:
    st.markdown('<div class="flux-kicker">Freixenet FLUX · mando externo</div>', unsafe_allow_html=True)
    st.markdown("<h2>Tú controlas el escenario</h2>", unsafe_allow_html=True)
    st.markdown('<p class="flux-muted">Cada cambio se aplica al instante en la pantalla de clase.</p>', unsafe_allow_html=True)

    def aplicar():
        guardar_escenario(leer_controles("m_"))

    actual, _ = leer_escenario()
    e = controles("m_", actual, al_cambiar=aplicar)
    if st.button("Volver a la situación estable", width="stretch"):
        for k in ["m_dem", "m_disp", "m_inc", "m_mer", "m_inc2"]:
            st.session_state.pop(k, None)
        guardar_escenario(Escenario())
        st.rerun()
    cfg = configurar(e, modelo().riesgo(e))
    st.markdown(f'<div class="flux-card gold"><div class="flux-kicker">En pantalla</div>'
                f'<div class="flux-mode">{cfg.nombre}</div>'
                f'<div class="flux-muted">Riesgo de desajuste estimado: {cfg.riesgo:.0%}</div></div>', unsafe_allow_html=True)
    st.caption("Simulación educativa con datos sintéticos. No son datos de Freixenet.")


# ---------------------------------------------------------------- vista proyector
def panel(e: Escenario, fuente: str, supuestos: Supuestos) -> None:
    m = modelo()
    riesgo = m.riesgo(e)
    cfg = configurar(e, riesgo)
    comp = comparar(e, riesgo, supuestos, cfg.modo)

    c1, c2, c3, c4 = st.columns([1.1, 1.6, 1, 1])
    with c1:
        st.markdown(f'<div class="flux-card gold"><div class="flux-kicker">Riesgo de desajuste</div>'
                    f'<div class="flux-big">{riesgo:.0%}</div><div class="flux-muted">Estimado por el modelo</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="flux-card gold"><div class="flux-kicker">Configuración activa</div>'
                    f'<div class="flux-mode">{cfg.nombre}</div>'
                    f'<div class="flux-muted">La IA recomienda · las reglas asignan la autoridad · las personas deciden</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="flux-card"><div class="flux-kicker">Escenario</div>'
                    f'<div class="flux-muted">Demanda {e.demanda:+.0f} % · Producto {e.disponibilidad:.0f} %<br>'
                    f'Incidencia {INCIDENCIAS[e.incidencia].lower()} · {e.mercados} mercado(s)<br>'
                    f'Incertidumbre {INCERTIDUMBRE[e.incertidumbre].lower()}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="flux-card"><div class="flux-kicker">Quién controla</div>'
                    f'<div class="flux-mode">{fuente}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="flux-kicker">Organigrama en vivo</div>', unsafe_allow_html=True)
    st.graphviz_chart(organigrama(cfg), width="stretch")
    c_por, c_quien, c_dec = st.columns([1.1, 1, 1.1])
    with c_por:
        st.markdown('<div class="flux-kicker">Por qué esta configuración</div>', unsafe_allow_html=True)
        for motivo in cfg.motivos:
            st.markdown(f"- {motivo}")
    with c_quien:
        st.markdown('<div class="flux-kicker">Quién entra y por qué</div>', unsafe_allow_html=True)
        filas = [f"<b>{AREAS[k]}</b>: {porque}" for k, porque in cfg.miembros]
        if cfg.analista:
            filas.append("<b>Análisis de datos</b>: la previsión es muy incierta y hay que revisarla")
        if cfg.enlace:
            filas.append("<b>Enlace con el grupo</b>: informa a Henkell sin autoridad formal (posición de enlace)")
        if cfg.lateral:
            filas.append("<b>Comercial, Operaciones y Logística</b>: se coordinan de forma lateral, sin crear una célula")
        if not filas:
            filas.append("Nadie: cada área trabaja en su ámbito con la jerarquía habitual")
        st.markdown("<br>".join(f"· {f}" for f in filas), unsafe_allow_html=True)
    with c_dec:
        st.markdown('<div class="flux-kicker">Quién puede decidir qué</div>', unsafe_allow_html=True)
        for decision, quien in cfg.decisiones:
            st.markdown(f'<div class="flux-dec">{decision}<br><b>→ {quien}</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="flux-kicker" style="margin-top:8px">Mismo escenario, dos formas de organizarse</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    t, f = comp["tradicional"], comp["flux"]
    igual = comp["sin_celula"]
    k1.metric("Horas medias hasta decidir · FLUX", f"{f['horas_media']:.0f} h",
              None if igual else f"{f['horas_media'] - t['horas_media']:+.0f} h frente al tradicional ({t['horas_media']:.0f} h)", delta_color="inverse")
    k2.metric("Niveles de autorización por decisión · FLUX", f"{f['niveles_media']:.1f}".replace(".", ","),
              None if igual else f"{f['niveles_media'] - t['niveles_media']:+.1f} frente al tradicional".replace(".", ","), delta_color="inverse")
    k3.metric("Incidencias resueltas en 24 h · FLUX", f"{f['en_24h']:.0f} %",
              None if igual else f"{f['en_24h'] - t['en_24h']:+.0f} puntos frente al tradicional")

    df = pd.DataFrame([
        {"Modelo": "Tradicional", "Indicador": "Horas hasta decidir", "Valor": t["horas_media"]},
        {"Modelo": "FLUX", "Indicador": "Horas hasta decidir", "Valor": f["horas_media"]},
        {"Modelo": "Tradicional", "Indicador": "Niveles de autorización", "Valor": t["niveles_media"]},
        {"Modelo": "FLUX", "Indicador": "Niveles de autorización", "Valor": f["niveles_media"]},
    ])
    graf = alt.Chart(df).mark_bar(cornerRadiusEnd=2).encode(
        x=alt.X("Valor:Q", title=None, axis=alt.Axis(labelColor="#9C9C9C", gridColor="#222")),
        y=alt.Y("Modelo:N", title=None, sort=["Tradicional", "FLUX"], axis=alt.Axis(labelColor="#F4F1EA", labelFontSize=13)),
        color=alt.Color("Modelo:N", scale=alt.Scale(domain=["Tradicional", "FLUX"], range=["#6E6E6E", "#BC9A4C"]), legend=None),
        column=alt.Column("Indicador:N", title=None, header=alt.Header(labelColor="#CAB796", labelFontSize=13)),
    ).properties(height=110, width=380).resolve_scale(x="independent").configure_view(stroke=None).configure(background="transparent")
    st.altair_chart(graf)
    if comp["sin_celula"]:
        st.caption("En situación estable FLUX no activa ninguna célula: funciona igual que la estructura funcional.")
    st.markdown(f'<div class="flux-band">Simulación con supuestos explícitos y datos sintéticos: '
                f'{comp["incidencias"]} incidencias estimadas en la campaña ({comp["locales"]} locales, '
                f'{comp["excepciones"]} excepciones, {comp["internacionales"]} internacionales). No son datos de Freixenet.</div>',
                unsafe_allow_html=True)


def vista_proyector() -> None:
    m = modelo()
    with st.sidebar:
        st.markdown('<div class="flux-kicker">Control del presentador</div>', unsafe_allow_html=True)
        control = st.radio("Quién controla el escenario", ["Manual", "Mando externo (QR)"], horizontal=True)
        manual = None
        if control == "Manual":
            manual = controles("p_")
        else:
            url = st.text_input("URL de la app", url_por_defecto())
            st.image(qr_png(url.rstrip("/") + "/?vista=mando"), caption="Escanea para tomar el mando", width="stretch")
            if st.button("Volver a la situación estable", width="stretch"):
                guardar_escenario(Escenario())
        with st.expander("Supuestos de la comparación"):
            h = st.slider("Horas de espera por nivel jerárquico", 2, 24, 8)
            c = st.slider("Horas de una reunión de la célula", 1, 8, 3)
            pen = st.slider("Retraso extra por nivel de incertidumbre", 0.0, 1.0, 0.5, 0.1)
        with st.expander("Sobre el modelo de IA"):
            st.write(f"Gradient boosting entrenado con {m.n_entrenamiento:,} campañas **sintéticas**. "
                     f"AUC en validación: {m.auc:.2f}. Estima la probabilidad de que la demanda supere al producto disponible. "
                     "Recomienda; no decide: la autoridad la asignan las reglas.".replace(",", "."))

    st.markdown('<div class="flux-kicker">Freixenet FLUX · simulación educativa</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="margin-top:0">La estructura cambia según la situación</h1>', unsafe_allow_html=True)
    supuestos = Supuestos(horas_por_nivel=h, horas_celula=c, penalizacion_incertidumbre=pen)

    if control == "Manual":
        panel(manual, "Presentador", supuestos)
    else:
        @st.fragment(run_every="2s")
        def en_vivo():
            e, t = leer_escenario()
            if t is None:
                st.info("Esperando al mando externo… Que alguien escanee el QR del panel lateral. Mientras tanto, situación estable.")
            panel(e, "Mando externo", supuestos)
        en_vivo()


if st.query_params.get("vista") in ("mando", "movil"):
    vista_mando()
else:
    vista_proyector()
