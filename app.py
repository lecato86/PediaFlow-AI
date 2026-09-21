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
ASSETS_DIR = BASE_DIR / "assets"


def buscar_logo():
    """Devuelve la ruta del logo (assets/logo.*, sin importar mayúsculas) o None."""
    if not ASSETS_DIR.exists():
        return None
    for f in sorted(ASSETS_DIR.iterdir()):
        if f.is_file() and f.stem.lower() == "logo" and f.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"
        }:
            return f
    return None


def mime_imagen(data: bytes, sufijo: str) -> str:
    """Detecta el tipo real de imagen por sus primeros bytes (el nombre puede engañar)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if sufijo.lower() == ".svg":
        return "image/svg+xml"
    return "image/png"


LOGO_PATH = buscar_logo()

# Nombres de columnas EXACTOS con los que fue entrenado el modelo, en orden.
FEATURES = ["tal posta", "fc", "fr", "sat", "l/kg"]

# Punto de corte óptimo (índice de Youden) y su sensibilidad asociada.
YOUDEN_PCT = 45.9
YOUDEN_SENS = 71

def icono_pagina():
    """Ícono de pestaña: el logo como imagen PIL (evita problemas de extensión) o un emoji."""
    if LOGO_PATH is None or LOGO_PATH.suffix.lower() == ".svg":
        return "🫁"
    try:
        from PIL import Image

        return Image.open(LOGO_PATH)
    except Exception:  # noqa: BLE001
        return "🫁"


st.set_page_config(
    page_title="PediaFlow-AI",
    page_icon=icono_pagina(),
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

      /* Fondo general: mismo degradado radial que el logo
         (muestreado del archivo: #09405F arriba-centro, #0A2945 medio, #02081E abajo) */
      .stApp {
        background:
          radial-gradient(ellipse 130% 75% at 50% 0%, #0a4262 0%, #0a2f4d 35%, #0a2945 55%, #04142a 80%, #02081e 100%);
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
        color: #7fe6e0; border: 1px solid rgba(95,227,217,.35);
        background: linear-gradient(90deg, rgba(47,134,214,.22), rgba(95,227,217,.18));
      }

      /* Encabezado con logo-banner (el logo trae nombre y fondo propios) */
      .pf-hero-banner {text-align: center; margin: -10px auto 14px; padding: 0;}
      .pf-hero-banner img {
        display: block; margin: 0 auto; width: 100%; max-width: 460px; height: auto;
        border: 0; border-radius: 0; box-shadow: none; background: transparent;
        /* Los bordes se funden con el fondo de la página: sin marco ni rectángulo */
        -webkit-mask-image:
          linear-gradient(to right,  transparent 0%, #000 9%, #000 91%, transparent 100%),
          linear-gradient(to bottom, transparent 0%, #000 9%, #000 91%, transparent 100%);
        -webkit-mask-composite: source-in;
                mask-image:
          linear-gradient(to right,  transparent 0%, #000 9%, #000 91%, transparent 100%),
          linear-gradient(to bottom, transparent 0%, #000 9%, #000 91%, transparent 100%);
                mask-composite: intersect;
      }
      .pf-hero-banner p {margin: 4px auto 0; max-width: 560px; color: #a9b8cf; font-size: 1rem; line-height: 1.45;}
      .pf-hero-banner .pf-chip {margin-top: 12px;}

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
        letter-spacing: .3px; border: none; color: #04101f;
        /* Degradé azul → turquesa, como los pulmones y el "AI" del logo */
        background: linear-gradient(135deg, #2f86d6 0%, #3fb8d8 55%, #5fe3d9 100%);
        box-shadow: 0 12px 30px rgba(63,184,216,.35);
        transition: transform .12s ease, box-shadow .12s ease;
      }
      div.stButton > button:hover {transform: translateY(-1px); box-shadow: 0 16px 36px rgba(63,184,216,.5); color: #04101f;}
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
        .pf-hero-banner {margin: -6px auto 10px;}
        .pf-hero-banner img {max-width: 320px;}
        .pf-hero-banner p {font-size: .88rem; margin-top: 2px;}

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
def _logo_data_uri(path_str: str, mtime: float, size: int):
    """Data-URI del logo. La caché se invalida si el archivo cambia (mtime/tamaño)."""
    path = Path(path_str)
    data = path.read_bytes()
    mime = mime_imagen(data, path.suffix)
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def logo_data_uri():
    """Devuelve el logo como data-URI, o None si no hay archivo (sin cachear el None)."""
    if LOGO_PATH is None or not LOGO_PATH.exists():
        return None
    st_ = LOGO_PATH.stat()
    return _logo_data_uri(str(LOGO_PATH), st_.st_mtime, st_.st_size)


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
_logo = logo_data_uri()

if _logo:
    # El logo ya incluye el nombre y el fondo oscuro: se muestra como banner.
    st.markdown(
        f"""
        <div class="pf-hero-banner">
          <img src="{_logo}" alt="PediaFlow-AI">
          <p>Predicción de riesgo de fracaso de cánula nasal de alto flujo (CNAF) al ingreso</p>
          <span class="pf-chip">Modelo predictivo · Pediatría</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="pf-glass pf-hero">
          <div class="pf-logo-fallback">🫁</div>
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
