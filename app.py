"""
PediaFlow-AI - Predicción de riesgo de fracaso de CNAF al ingreso.

Carga el modelo entrenado (modelo_canula_ingreso.pkl) y calcula la probabilidad
de fracaso a partir de cinco variables clínicas.
"""

import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "modelo_canula_ingreso.pkl"
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

# Nombres de columnas EXACTOS con los que fue entrenado el modelo, en orden.
FEATURES = ["tal posta", "fc", "fr", "sat", "l/kg"]

# Punto de corte óptimo (índice de Youden) y su sensibilidad asociada.
YOUDEN_PCT = 45.9
YOUDEN_SENS = 71

st.set_page_config(
    page_title="PediaFlow-AI",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

      html, body, [class*="css"], .stApp {font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;}
      #MainMenu, footer, header {visibility: hidden;}

      /* Fondo general: azul profundo con destellos */
      .stApp {
        background:
          radial-gradient(1100px 600px at 10% -10%, rgba(56,182,201,.28), transparent 60%),
          radial-gradient(900px 500px at 100% 0%, rgba(99,102,241,.22), transparent 60%),
          radial-gradient(700px 500px at 50% 110%, rgba(16,185,129,.14), transparent 60%),
          linear-gradient(180deg, #0b1220 0%, #0d1730 55%, #0b1220 100%);
        background-attachment: fixed;
        color: #e5ecf6;
      }
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 880px;}

      /* Tarjetas translúcidas */
      .pf-glass {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 20px;
        box-shadow: 0 18px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
      }

      /* Encabezado */
      .pf-hero {
        display: flex; align-items: center; gap: 22px;
        padding: 26px 30px; margin-bottom: 22px;
      }
      .pf-hero img, .pf-hero .pf-logo-fallback {
        width: 88px; height: 88px; flex: 0 0 88px; border-radius: 22px; object-fit: contain;
      }
      .pf-hero .pf-logo-fallback {
        display: flex; align-items: center; justify-content: center; font-size: 2.4rem;
        background: linear-gradient(135deg, #1d7fb8, #38b6c9);
        box-shadow: 0 10px 25px rgba(56,182,201,.35);
      }
      .pf-hero h1 {
        margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px; line-height: 1.1;
        background: linear-gradient(90deg, #ffffff 0%, #9be7f2 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .pf-hero p {margin: 8px 0 0; color: #a9b8cf; font-size: 1rem; line-height: 1.4;}
      .pf-chip {
        display: inline-block; margin-top: 10px; padding: 4px 10px; border-radius: 999px;
        font-size: .72rem; font-weight: 700; letter-spacing: .6px; text-transform: uppercase;
        color: #9be7f2; background: rgba(56,182,201,.14); border: 1px solid rgba(56,182,201,.35);
      }

      /* Formulario */
      .pf-section {font-weight: 800; color: #e5ecf6; font-size: 1.1rem; margin: 6px 0 2px;}
      .pf-hint {color: #8fa1bb; font-size: .86rem; margin-bottom: 10px;}
      .stSlider label, .stNumberInput label {color: #cdd8e8 !important; font-weight: 600 !important;}
      .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {color: #7f91ab !important;}
      .stNumberInput input {
        background: rgba(255,255,255,.06) !important; color: #fff !important;
        border-radius: 10px !important; font-weight: 700 !important; text-align: center;
      }
      .stNumberInput > div > div {background: transparent !important; border-color: rgba(255,255,255,.14) !important; border-radius: 10px !important;}
      .stNumberInput button {background: rgba(255,255,255,.06) !important; color: #cdd8e8 !important; border-color: rgba(255,255,255,.14) !important;}

      div.stButton > button {
        width: 100%; min-height: 54px; border-radius: 14px; font-weight: 800; font-size: 1.05rem;
        letter-spacing: .3px; border: none; color: #06121f;
        background: linear-gradient(135deg, #38b6c9 0%, #7ee0ee 100%);
        box-shadow: 0 12px 30px rgba(56,182,201,.35);
        transition: transform .12s ease, box-shadow .12s ease;
      }
      div.stButton > button:hover {transform: translateY(-1px); box-shadow: 0 16px 36px rgba(56,182,201,.45); color: #06121f;}
      div.stButton > button:active {transform: translateY(0);}

      /* Resultado */
      .pf-result {
        border-radius: 20px; padding: 28px 30px; margin-top: 10px; color: #fff;
        box-shadow: 0 20px 50px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.18);
        border: 1px solid rgba(255,255,255,.18);
      }
      .pf-result .label {font-size: .82rem; opacity: .9; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700;}
      .pf-result .value {font-size: 3.6rem; font-weight: 800; line-height: 1.05; margin: 6px 0 2px; letter-spacing: -1px;}
      .pf-result .level {font-size: 1.15rem; font-weight: 800;}

      .pf-bar-wrap {position: relative; margin: 20px 0 60px;}
      .pf-bar-bg {
        width: 100%; height: 22px; background: rgba(0,0,0,.28);
        border-radius: 999px; overflow: hidden; box-shadow: inset 0 2px 6px rgba(0,0,0,.35);
      }
      .pf-bar-fill {
        height: 100%; border-radius: 999px; transition: width .6s ease;
        background: linear-gradient(90deg, rgba(255,255,255,.75), #ffffff);
        box-shadow: 0 0 14px rgba(255,255,255,.55);
      }

      /* Marca fija del índice de Youden */
      .pf-youden-line {
        position: absolute; top: -8px; bottom: -8px; width: 3px;
        background: #fff; border-radius: 2px; transform: translateX(-50%);
        box-shadow: 0 0 0 1.5px rgba(0,0,0,.45), 0 0 12px rgba(255,255,255,.6);
      }
      .pf-youden-tag {
        position: absolute; top: calc(100% + 12px); transform: translateX(-50%);
        white-space: nowrap; text-align: center; line-height: 1.25;
        font-size: .74rem; font-weight: 800; letter-spacing: .4px;
        background: rgba(0,0,0,.42); padding: 5px 10px; border-radius: 8px;
        border: 1px solid rgba(255,255,255,.18);
      }
      .pf-youden-tag::before {
        content: ""; position: absolute; left: 50%; top: -6px; transform: translateX(-50%);
        border: 6px solid transparent; border-top: 0; border-bottom-color: rgba(0,0,0,.42);
      }
      .pf-youden-tag span {font-weight: 600; letter-spacing: .2px; opacity: .95;}
      .pf-scale {display: flex; justify-content: space-between; font-size: .8rem; opacity: .9; font-weight: 600;}

      /* Alerta clínica */
      .pf-alert {
        border-radius: 16px; padding: 18px 22px; margin-top: 16px; font-size: .98rem;
        border-left: 6px solid; color: #e5ecf6; line-height: 1.5;
      }
      .pf-alert b {display: block; font-size: 1.05rem; margin-bottom: 4px; color: #fff;}

      /* Leyenda */
      .pf-legend {padding: 18px 22px; margin-top: 26px; color: #d5deeb;}
      .pf-legend-title {font-weight: 800; color: #fff; margin-bottom: 12px; font-size: 1rem; letter-spacing: .2px;}
      .pf-legend-row {display: flex; gap: 12px; align-items: flex-start; margin: 10px 0; font-size: .92rem; line-height: 1.45;}
      .pf-legend-row b {display: block; margin-bottom: 2px; color: #fff;}
      .pf-dot {flex: 0 0 14px; width: 14px; height: 14px; border-radius: 50%; margin-top: 5px; box-shadow: 0 0 10px currentColor;}

      .pf-foot {color: #7f91ab; font-size: .8rem; text-align: center; margin-top: 30px;}

      /* Expander y tabla en oscuro */
      [data-testid="stExpander"] {border: 1px solid rgba(255,255,255,.10) !important; border-radius: 14px !important; background: rgba(255,255,255,.04);}

      /* Controles táctiles */
      [data-baseweb="slider"] [role="slider"] {width: 22px !important; height: 22px !important; box-shadow: 0 0 0 4px rgba(56,182,201,.25) !important;}

      /* ------------------------------------------------------------------ */
      /* Celular: ancho <= 640px                                             */
      /* ------------------------------------------------------------------ */
      @media (max-width: 640px) {
        .block-container {padding: 1rem .9rem 2.5rem !important;}

        .pf-hero {padding: 18px 18px; gap: 14px; border-radius: 16px; margin-bottom: 16px;}
        .pf-hero img, .pf-hero .pf-logo-fallback {width: 60px; height: 60px; flex-basis: 60px; border-radius: 16px;}
        .pf-hero .pf-logo-fallback {font-size: 1.7rem;}
        .pf-hero h1 {font-size: 1.45rem;}
        .pf-hero p  {font-size: .88rem; line-height: 1.35;}
        .pf-chip {font-size: .66rem;}

        .pf-section {font-size: .98rem;}
        .pf-hint {font-size: .8rem;}

        /* Slider y campo numérico en la MISMA fila */
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
        .stNumberInput button {display: none !important;}
        .stNumberInput input {padding: .45rem .3rem;}

        .pf-result {padding: 20px 18px; border-radius: 16px;}
        .pf-result .label {font-size: .74rem;}
        .pf-result .value {font-size: 2.7rem;}
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
# Modelo y recursos
# --------------------------------------------------------------------------- #
@st.cache_resource
def cargar_modelo(path: Path):
    return joblib.load(path)


@st.cache_data
def logo_base64(path: Path):
    """Devuelve el logo como data-URI para incrustarlo en el encabezado."""
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


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
_logo = logo_base64(LOGO_PATH)
logo_html = (
    f'<img src="data:image/png;base64,{_logo}" alt="PediaFlow-AI">'
    if _logo
    else '<div class="pf-logo-fallback">🫁</div>'
)

st.markdown(
    f"""
    <div class="pf-glass pf-hero">
      {logo_html}
      <div>
        <h1>PediaFlow-AI</h1>
        <p>Predicción de riesgo de fracaso de cánula nasal de alto flujo (CNAF) al ingreso</p>
        <span class="pf-chip">Modelo predictivo · Pediatría</span>
      </div>
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
        color, color_dark, icono = "#22c55e", "#0f6b3a", "🟢"
        nivel, subtitulo = "RIESGO BAJO", "Paciente Seguro"
        mensaje = (
            "Probabilidad de fracaso menor al 30 %. El paciente satura bien y presenta "
            "un Score de Tal bajo. Continuar con el soporte actual y el monitoreo habitual."
        )
    elif pct <= 70:
        color, color_dark, icono = "#f59e0b", "#9a4a06", "🟡"
        nivel, subtitulo = "RIESGO MODERADO", "Monitoreo Estricto"
        mensaje = (
            "Probabilidad de fracaso entre 30 % y 70 %. Zona gris donde el paciente "
            "empieza a descompensarse. Se recomienda vigilancia estrecha, reevaluación "
            "clínica frecuente y optimización de parámetros."
        )
    else:
        color, color_dark, icono = "#ef4444", "#8f1d1d", "🚨"
        nivel, subtitulo = "RIESGO ALTO", "Alerta de Fallo"
        mensaje = (
            "Probabilidad de fracaso mayor al 70 %. El modelo confirma de forma "
            "multivariada que el paciente comparte el perfil del Cluster 1 de fracaso "
            "histórico. Considerar escalar el soporte respiratorio y evaluar ingreso o "
            "traslado a cuidados intensivos."
        )

    st.markdown(
        f"""
        <div class="pf-result" style="background: linear-gradient(135deg, {color} 0%, {color_dark} 100%);">
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
        <div class="pf-glass pf-alert" style="border-left-color:{color};">
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
    <div class="pf-glass pf-legend">
      <div class="pf-legend-title">Escala de riesgo</div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#22c55e; color:#22c55e"></span>
        <div><b>🟢 Riesgo Bajo · Paciente Seguro</b>
        Probabilidad de fracaso menor al 30 % (saturando bien, TAL bajo).</div>
      </div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#f59e0b; color:#f59e0b"></span>
        <div><b>🟡 Riesgo Moderado · Monitoreo Estricto</b>
        Entre 30 % y 70 %: zona gris donde el paciente empieza a descompensarse.</div>
      </div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#ef4444; color:#ef4444"></span>
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
