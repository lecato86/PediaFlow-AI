"""
PediaFlow-AI - Predicción de riesgo de fracaso de CNAF al ingreso.

Carga el modelo entrenado (modelo_canula_ingreso.pkl) y calcula la probabilidad
de fracaso a partir de cinco variables clínicas.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(__file__).parent / "modelo_canula_ingreso.pkl"

# Nombres de columnas EXACTOS con los que fue entrenado el modelo, en orden.
FEATURES = ["tal posta", "fc", "fr", "sat", "l/kg"]

# Punto de corte óptimo (índice de Youden) y su sensibilidad asociada.
YOUDEN_PCT = 45.9
YOUDEN_SENS = 71

st.set_page_config(
    page_title="PediaFlow-AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 860px;}

      .pf-hero {
        background: linear-gradient(135deg, #0f4c81 0%, #1d7fb8 60%, #38b6c9 100%);
        color: #fff; border-radius: 18px; padding: 28px 32px; margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(15, 76, 129, .25);
      }
      .pf-hero h1 {margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: .5px;}
      .pf-hero p  {margin: 6px 0 0; opacity: .92; font-size: 1.02rem;}

      .pf-section {font-weight: 700; color: #0f4c81; font-size: 1.05rem; margin-bottom: 6px;}
      .pf-hint {color: #6b7a90; font-size: .85rem; margin-top: -4px; margin-bottom: 12px;}

      .pf-result {
        border-radius: 18px; padding: 26px 30px; margin-top: 8px; color: #fff;
        box-shadow: 0 10px 30px rgba(0,0,0,.12);
      }
      .pf-result .label {font-size: .95rem; opacity: .9; text-transform: uppercase; letter-spacing: 1px;}
      .pf-result .value {font-size: 3.4rem; font-weight: 800; line-height: 1.05; margin: 4px 0 2px;}
      .pf-result .level {font-size: 1.15rem; font-weight: 700; opacity: .95;}

      .pf-bar-wrap {position: relative; margin: 18px 0 58px;}
      .pf-bar-bg {
        width: 100%; height: 22px; background: rgba(255,255,255,.28);
        border-radius: 999px; overflow: hidden;
      }
      .pf-bar-fill {height: 100%; border-radius: 999px; background: #fff; transition: width .6s ease;}

      /* Marca fija del índice de Youden: línea que atraviesa la barra + leyenda debajo */
      .pf-youden-line {
        position: absolute; top: -7px; bottom: -7px; width: 3px;
        background: #fff; border-radius: 2px; transform: translateX(-50%);
        box-shadow: 0 0 0 1.5px rgba(0,0,0,.35);
      }
      .pf-youden-tag {
        position: absolute; top: calc(100% + 10px); transform: translateX(-50%);
        white-space: nowrap; text-align: center; line-height: 1.25;
        font-size: .74rem; font-weight: 800; letter-spacing: .4px;
        background: rgba(0,0,0,.30); padding: 5px 10px; border-radius: 8px;
      }
      .pf-youden-tag::before {
        content: ""; position: absolute; left: 50%; top: -6px; transform: translateX(-50%);
        border: 6px solid transparent; border-top: 0; border-bottom-color: rgba(0,0,0,.30);
      }
      .pf-youden-tag span {font-weight: 600; letter-spacing: .2px; opacity: .95;}

      .pf-scale {display: flex; justify-content: space-between; font-size: .8rem; opacity: .9;}

      .pf-alert {
        border-radius: 14px; padding: 16px 20px; margin-top: 16px; font-size: 1rem;
        border-left: 6px solid; background: #fff; color: #1f2937;
        box-shadow: 0 4px 14px rgba(20, 40, 80, .06);
      }
      .pf-alert b {display: block; font-size: 1.05rem; margin-bottom: 4px;}

      div.stButton > button {
        width: 100%; border-radius: 12px; font-weight: 700; font-size: 1.05rem;
        padding: .7rem 1rem; border: none; color: #fff;
        background: linear-gradient(135deg, #0f4c81, #1d7fb8);
        box-shadow: 0 6px 18px rgba(15, 76, 129, .3);
      }
      div.stButton > button:hover {background: linear-gradient(135deg, #0c3d68, #17679a); color: #fff;}

      .pf-foot {color: #8a97a8; font-size: .8rem; text-align: center; margin-top: 30px;}

      .pf-legend {
        background: #f6f9fc; border: 1px solid #e6ebf1; border-radius: 14px;
        padding: 16px 20px; margin-top: 26px; color: #1f2937;
      }
      .pf-legend-title {font-weight: 700; color: #0f4c81; margin-bottom: 10px; font-size: .98rem;}
      .pf-legend-row {display: flex; gap: 12px; align-items: flex-start; margin: 8px 0; font-size: .92rem; line-height: 1.4;}
      .pf-legend-row b {display: block; margin-bottom: 2px;}
      .pf-dot {flex: 0 0 14px; width: 14px; height: 14px; border-radius: 50%; margin-top: 4px;}

      /* Controles táctiles más cómodos (todas las pantallas) */
      div.stButton > button {min-height: 52px;}
      [data-baseweb="slider"] [role="slider"] {width: 22px !important; height: 22px !important;}

      /* ------------------------------------------------------------------ */
      /* Celular: ancho <= 640px                                             */
      /* ------------------------------------------------------------------ */
      @media (max-width: 640px) {
        .block-container {padding: 1rem .9rem 2.5rem !important;}

        .pf-hero {padding: 18px 18px; border-radius: 14px; margin-bottom: 16px;}
        .pf-hero h1 {font-size: 1.45rem;}
        .pf-hero p  {font-size: .9rem; line-height: 1.35;}

        .pf-section {font-size: .98rem;}
        .pf-hint {font-size: .8rem;}

        /* Slider y campo numérico en la MISMA fila (Streamlit los apila por defecto) */
        [data-testid="stHorizontalBlock"] {
          flex-direction: row !important; flex-wrap: nowrap !important;
          gap: .6rem !important; align-items: flex-end;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {
          flex: 1 1 0 !important; min-width: 0 !important; width: auto !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {
          flex: 0 0 88px !important; min-width: 88px !important; width: 88px !important;
        }
        /* Ocultar botones +/- del campo numérico para ganar ancho */
        .stNumberInput button {display: none !important;}
        .stNumberInput input {text-align: center; font-weight: 700; padding: .45rem .3rem;}

        .pf-result {padding: 20px 18px; border-radius: 14px;}
        .pf-result .label {font-size: .78rem;}
        .pf-result .value {font-size: 2.6rem;}
        .pf-result .level {font-size: 1rem;}
        .pf-bar-wrap {margin: 14px 0 54px;}
        .pf-bar-bg {height: 18px;}
        .pf-youden-tag {font-size: .66rem; padding: 4px 8px;}
        .pf-scale {font-size: .72rem;}

        .pf-alert {padding: 14px 14px; font-size: .93rem; border-radius: 12px;}
        .pf-alert b {font-size: .98rem;}

        div.stButton > button {font-size: 1rem;}
        .pf-foot {font-size: .72rem; margin-top: 22px;}
        .pf-legend {padding: 14px 14px; margin-top: 20px;}
        .pf-legend-row {font-size: .85rem; gap: 10px;}
      }

      /* Pantallas muy chicas (<= 380px) */
      @media (max-width: 380px) {
        .pf-hero h1 {font-size: 1.3rem;}
        .pf-result .value {font-size: 2.2rem;}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {
          flex: 0 0 76px !important; min-width: 76px !important; width: 76px !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@st.cache_resource
def cargar_modelo(path: Path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        f"No se encontró `{MODEL_PATH.name}`. Debe estar en la raíz del proyecto, "
        "junto a `app.py`."
    )
    st.stop()

modelo = cargar_modelo(MODEL_PATH)

# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="pf-hero">
      <h1>🫁 PediaFlow-AI</h1>
      <p>Predicción de riesgo de fracaso de cánula nasal de alto flujo (CNAF) al ingreso</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Formulario
# --------------------------------------------------------------------------- #
def entrada(label, key, vmin, vmax, vdef, step, fmt, ayuda):
    """Slider + campo numérico sincronizados para una misma variable."""
    if key not in st.session_state:
        st.session_state[key] = vdef

    def _desde_slider():
        st.session_state[key] = st.session_state[f"{key}_sl"]

    def _desde_num():
        st.session_state[key] = st.session_state[f"{key}_num"]

    c1, c2 = st.columns([3, 1.1])
    with c1:
        st.slider(
            label, vmin, vmax, st.session_state[key], step,
            key=f"{key}_sl", on_change=_desde_slider, help=ayuda, format=fmt,
        )
    with c2:
        st.number_input(
            label, vmin, vmax, st.session_state[key], step,
            key=f"{key}_num", on_change=_desde_num, format=fmt,
            label_visibility="hidden",
        )
    return st.session_state[key]


st.markdown('<div class="pf-section">📋 Variables clínicas al ingreso</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="pf-hint">Usá los deslizadores o escribí el valor exacto en el campo numérico.</div>',
    unsafe_allow_html=True,
)

tal = entrada("Score de Tal", "tal", 0, 12, 6, 1, "%d",
              "Puntaje clínico de Tal (0 a 12).")
fc = entrada("Frecuencia Cardíaca (FC) · lpm", "fc", 60, 230, 140, 1, "%d",
             "Latidos por minuto.")
fr = entrada("Frecuencia Respiratoria (FR) · rpm", "fr", 15, 110, 50, 1, "%d",
             "Respiraciones por minuto.")
sat = entrada("Saturación de Oxígeno (Sat) · %", "sat", 70, 100, 93, 1, "%d",
              "Saturación periférica de oxígeno.")
lkg = entrada("Flujo (L/kg)", "lkg", 0.5, 3.0, 1.5, 0.1, "%.1f",
              "Litros por minuto por kilogramo de peso.")

st.write("")
calcular = st.button("🔎 Calcular Riesgo")

# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #
if calcular:
    X = pd.DataFrame([[tal, fc, fr, sat, lkg]], columns=FEATURES)

    try:
        proba = modelo.predict_proba(X)[0]
        clases = list(getattr(modelo, "classes_", [0, 1]))
        idx = clases.index(1) if 1 in clases else len(proba) - 1
        p = float(proba[idx])
    except Exception as e:  # noqa: BLE001
        st.error(f"Error al ejecutar la predicción: {e}")
        st.stop()

    pct = p * 100
    pct_txt = f"{pct:.1f}".replace(".", ",")
    youden_txt = f"{YOUDEN_PCT:.1f}".replace(".", ",")

    if pct < 30:
        color, color_dark, icono = "#22c55e", "#15803d", "🟢"
        nivel, subtitulo = "RIESGO BAJO", "Paciente Seguro"
        mensaje = (
            "Probabilidad de fracaso menor al 30 %. El paciente satura bien y presenta "
            "un Score de Tal bajo. Continuar con el soporte actual y el monitoreo habitual."
        )
    elif pct <= 70:
        color, color_dark, icono = "#f59e0b", "#b45309", "🟡"
        nivel, subtitulo = "RIESGO MODERADO", "Monitoreo Estricto"
        mensaje = (
            "Probabilidad de fracaso entre 30 % y 70 %. Zona gris donde el paciente "
            "empieza a descompensarse. Se recomienda vigilancia estrecha, reevaluación "
            "clínica frecuente y optimización de parámetros."
        )
    else:
        color, color_dark, icono = "#ef4444", "#b91c1c", "🚨"
        nivel, subtitulo = "RIESGO ALTO", "Alerta de Fallo"
        mensaje = (
            "Probabilidad de fracaso mayor al 70 %. El modelo confirma de forma "
            "multivariada que el paciente comparte el perfil del Cluster 1 de fracaso "
            "histórico. Considerar escalar el soporte respiratorio y evaluar ingreso o "
            "traslado a cuidados intensivos."
        )

    st.markdown(
        f"""
        <div class="pf-result" style="background: linear-gradient(135deg, {color}, {color_dark});">
          <div class="label">Probabilidad de fracaso de CNAF</div>
          <div class="value">{pct_txt}%</div>
          <div class="level">{icono} {nivel} · {subtitulo}</div>
          <div class="pf-bar-wrap">
            <div class="pf-bar-bg"><div class="pf-bar-fill" style="width:{pct:.1f}%"></div></div>
            <div class="pf-youden-line" style="left:{YOUDEN_PCT}%"></div>
            <div class="pf-youden-tag" style="left:{YOUDEN_PCT}%">
              ÍNDICE DE YOUDEN {youden_txt}%<br>
              <span>Sensibilidad {YOUDEN_SENS}%</span>
            </div>
          </div>
          <div class="pf-scale"><span>0%</span><span>30%</span><span>70%</span><span>100%</span></div>
        </div>
        <div class="pf-alert" style="border-color:{color};">
          <b>{icono} Alerta clínica · {nivel.title()} ({subtitulo})</b>
          {mensaje}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Ver valores utilizados"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Variable": ["Score de Tal", "FC (lpm)", "FR (rpm)", "Sat (%)", "L/kg"],
                    "Valor": [tal, fc, fr, sat, f"{lkg:.1f}"],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

st.markdown(
    """
    <div class="pf-legend">
      <div class="pf-legend-title">Escala de riesgo</div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#22c55e"></span>
        <div><b>🟢 Riesgo Bajo · Paciente Seguro</b>
        Probabilidad de fracaso menor al 30 % (saturando bien, TAL bajo).</div>
      </div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#f59e0b"></span>
        <div><b>🟡 Riesgo Moderado · Monitoreo Estricto</b>
        Entre 30 % y 70 %: zona gris donde el paciente empieza a descompensarse.</div>
      </div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#ef4444"></span>
        <div><b>🚨 Riesgo Alto · Alerta de Fallo</b>
        Mayor al 70 %: el modelo confirma de forma multivariada que el paciente comparte
        el perfil del Cluster 1 de fracaso histórico.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="pf-foot">Herramienta de apoyo a la decisión clínica. '
    "No reemplaza el criterio médico.</div>",
    unsafe_allow_html=True,
)
