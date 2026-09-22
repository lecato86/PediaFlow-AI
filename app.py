"""
PediaFlow-AI - Predicción de riesgo de fracaso de CNAF al ingreso.

Carga el modelo entrenado (modelo_canula_ingreso.pkl) y calcula la probabilidad
de fracaso a partir de tres variables: Score de Tal, flujo inicial (L/min) y pROX.
El pROX se calcula automáticamente a partir de la saturación, la FiO2, la
frecuencia respiratoria y la edad en meses.
"""

import base64
from pathlib import Path

import joblib
import numpy as np
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
FEATURES = ["TAL", "FLUJO", "pROX"]

# Frecuencia respiratoria normal para el pROX según la edad.
PROX_EDAD_CORTE_MESES = 12   # hasta 12 meses inclusive → FR normal 40; más de 12 → 30
PROX_FR_NORMAL_LACTANTE = 40
PROX_FR_NORMAL_MAYOR = 30

# Signo aplicado al z del pROX antes de la predicción (ver `normalizar`).
PROX_SIGNO_Z = -1.0

# Punto de corte óptimo (índice de Youden) y su sensibilidad asociada.
YOUDEN_PCT = 44.4
YOUDEN_SENS = 100.0
YOUDEN_ESPEC = 66.7

# Umbral inferior de la zona de riesgo moderado.
RIESGO_BAJO_PCT = 30.0


def fr_normal(edad_meses: float) -> int:
    """FR normal por edad: 40 rpm hasta los 12 meses inclusive, 30 rpm a partir de ahí."""
    return PROX_FR_NORMAL_LACTANTE if edad_meses <= PROX_EDAD_CORTE_MESES else PROX_FR_NORMAL_MAYOR


def calcular_rrsd(fr: float, edad_meses: float) -> float:
    """RRSD = FR observada / FR normal por edad."""
    return fr / fr_normal(edad_meses)


def calcular_prox(sat: float, fio2: float, fr: float, edad_meses: float) -> float:
    """pROX crudo = (SpO2 / FiO2) / RRSD.

    La FiO2 llega como proporción (0,21 a 1,00), que equivale a FiO2 % / 100.
    """
    return (sat / fio2) / calcular_rrsd(fr, edad_meses)


def normalizar(X: pd.DataFrame, modelo) -> pd.DataFrame:
    """Estandariza las variables con los metadatos guardados en el .pkl.

    El modelo se entrenó sobre valores normalizados (z = (x - media) / desvío).
    Las medias y desvíos vienen en `modelo.means_` y `modelo.stds_`, en el mismo
    orden que FEATURES. Si el archivo no los trae, se devuelven los valores crudos.

    El z del pROX se multiplica por PROX_SIGNO_Z (-1): el coeficiente del pROX en
    el modelo es levemente positivo por el ajuste conjunto con el TAL, y la
    inversión garantiza que un pROX alto (mejor oxigenación) reduzca el riesgo.
    """
    means = getattr(modelo, "means_", None)
    stds = getattr(modelo, "stds_", None)
    if means is None or stds is None:
        return X
    means = np.asarray(means, dtype=float).reshape(-1)
    stds = np.asarray(stds, dtype=float).reshape(-1)
    if means.shape[0] != len(FEATURES) or stds.shape[0] != len(FEATURES):
        raise ValueError(
            f"Los metadatos de escala del modelo tienen {means.shape[0]} valores, "
            f"pero se esperaban {len(FEATURES)} ({', '.join(FEATURES)})."
        )
    stds = np.where(stds == 0, 1.0, stds)  # evita división por cero
    Z = pd.DataFrame((X.to_numpy(dtype=float) - means) / stds, columns=FEATURES)
    Z["pROX"] = Z["pROX"] * PROX_SIGNO_Z
    return Z


def predecir_riesgo(modelo, tal, flujo, fr, sat, fio2, edad_meses):
    """Pasos que se ejecutan al pulsar «Calcular Riesgo».

    1. FR normal por edad (40 si edad <= 12 meses, 30 si es mayor).
    2. RRSD = FR / FR normal.
    3. pROX crudo = (SpO2 / FiO2) / RRSD.
    4. Vector [TAL, FLUJO, pROX] en el orden que espera el modelo.
    5. Normalización con modelo.means_ y modelo.stds_, con el z del pROX invertido.
    6. predict_proba → probabilidad de fracaso acotada entre 0 y 100 %.
    Devuelve (porcentaje, pROX crudo, DataFrame normalizado).
    """
    prox_crudo = calcular_prox(sat, fio2, fr, edad_meses)
    X = pd.DataFrame([[tal, flujo, prox_crudo]], columns=FEATURES, dtype=float)
    Z = normalizar(X, modelo)
    proba = modelo.predict_proba(Z)[0]
    clases = list(getattr(modelo, "classes_", [0, 1]))
    idx = clases.index(1) if 1 in clases else len(proba) - 1
    pct = float(np.clip(proba[idx] * 100.0, 0.0, 100.0))
    return pct, prox_crudo, Z

# Crédito de autoría mostrado bajo el logo y en el pie.
AUTOR = "Catriel Rossi"

# Desempeño del modelo (área bajo la curva ROC, en %).
AUC_PCT = 85

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
        display: block; margin: 0 auto; width: 100%; max-width: 440px; height: auto;
        border: 0; border-radius: 0; box-shadow: none; background: transparent;
        /* PNG con transparencia real: solo un halo suave detrás */
        filter: drop-shadow(0 18px 40px rgba(0,0,0,.45)) drop-shadow(0 0 28px rgba(63,184,216,.18));
      }
      .pf-hero-banner p {margin: 18px auto 0; max-width: 560px; color: #a9b8cf; font-size: 1rem; line-height: 1.45;}
      .pf-hero-banner .pf-chip {margin-top: 12px;}

      /* Crédito de autoría */
      .pf-credit {
        margin-top: 6px; color: #8fa1bb; font-size: .8rem; letter-spacing: 1.4px;
        text-transform: uppercase; font-weight: 600;
      }
      .pf-credit b {
        font-weight: 800; letter-spacing: .6px; text-transform: none; font-size: .92rem;
        background: linear-gradient(90deg, #6fb6ff 0%, #5fe3d9 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .pf-hero .pf-credit {margin-top: 4px;}

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

      /* Sobre el modelo */
      .pf-method {padding: 18px 22px; margin-top: 16px; color: #d5deeb;}
      .pf-method-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 4px 0 14px;}
      .pf-stat {
        background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.09);
        border-radius: 12px; padding: 12px 12px; text-align: center;
      }
      .pf-stat-label {font-size: .68rem; letter-spacing: 1.2px; text-transform: uppercase; color: #8fa1bb; font-weight: 700;}
      .pf-stat-value {font-size: .95rem; font-weight: 800; color: #fff; margin-top: 4px; line-height: 1.2;}
      .pf-stat-hl {
        background: linear-gradient(135deg, rgba(47,134,214,.28), rgba(95,227,217,.22));
        border-color: rgba(95,227,217,.45);
      }
      .pf-stat-hl .pf-stat-value {
        font-size: 1.6rem; letter-spacing: -.5px;
        background: linear-gradient(90deg, #ffffff, #7fe6e0);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .pf-method-text {margin: 0; font-size: .9rem; line-height: 1.55; color: #b9c6d9;}
      .pf-method-text b {color: #fff; font-weight: 700;}

      /* pROX calculado automáticamente */
      .pf-subsection {
        font-weight: 700; color: #cdd8e8; font-size: .95rem; margin: 18px 0 2px;
        display: flex; align-items: center; gap: 8px;
      }
      .pf-prox {
        display: flex; align-items: center; justify-content: space-between; gap: 16px;
        padding: 16px 22px; margin: 14px 0 6px;
        background: linear-gradient(135deg, rgba(47,134,214,.22), rgba(95,227,217,.16));
        border-color: rgba(95,227,217,.40);
      }
      .pf-prox-value {
        font-size: 2.2rem; font-weight: 800; letter-spacing: -.5px; line-height: 1;
        background: linear-gradient(90deg, #ffffff, #7fe6e0);
        -webkit-background-clip: text; background-clip: text; color: transparent;
      }
      .pf-prox-formula {color: #a9b8cf; font-size: .82rem; line-height: 1.45; text-align: right;}
      .pf-prox-formula b {color: #fff; font-weight: 700;}

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
        .pf-hero-banner {margin: 0 auto 12px;}
        .pf-hero-banner img {max-width: 300px;}
        .pf-hero-banner p {font-size: .88rem; margin-top: 12px;}
        .pf-credit {font-size: .72rem; letter-spacing: 1.1px;}
        .pf-credit b {font-size: .84rem;}

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
        .pf-method {padding: 14px 14px; margin-top: 14px;}
        .pf-method-grid {grid-template-columns: repeat(2, 1fr); gap: 8px;}
        .pf-stat {padding: 10px 8px;}
        .pf-stat-value {font-size: .88rem;}
        .pf-stat-hl .pf-stat-value {font-size: 1.4rem;}
        .pf-method-text {font-size: .85rem;}
        .pf-prox {padding: 14px 16px; flex-direction: column; align-items: flex-start; gap: 8px;}
        .pf-prox-value {font-size: 1.9rem;}
        .pf-prox-formula {text-align: left; font-size: .78rem;}
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
          <div class="pf-credit">Powered by <b>{AUTOR}</b></div>
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
            <div class="pf-credit">Powered by <b>{AUTOR}</b></div>
            <p>Predicción de riesgo de fracaso de cánula nasal de alto flujo (CNAF) al ingreso</p>
            <span class="pf-chip">Modelo predictivo · Pediatría</span>
          </div>
        </div>
        """.replace("{AUTOR}", AUTOR),
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
flujo = entrada("Flujo inicial colocado · L/min", "flujo", 1, 60, 10, 1, "%d",
                "Flujo total inicial de la cánula, en litros por minuto.")

st.markdown(
    '<div class="pf-subsection">🧮 Datos para el cálculo automático del pROX</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="pf-hint">pROX = (Sat / FiO2) / RRSD, donde RRSD = FR / FR normal por edad. '
    'La FR normal es 40 rpm hasta los 12 meses inclusive y 30 rpm a partir de ahí. '
    'El modelo está desarrollado para pacientes de hasta 24 meses.</div>',
    unsafe_allow_html=True,
)

fr = entrada("Frecuencia Respiratoria (FR) · rpm", "fr", 15, 110, 50, 1, "%d",
             "Respiraciones por minuto.")
sat = entrada("Saturación de Oxígeno (Sat) · %", "sat", 70, 100, 93, 1, "%d",
              "Saturación periférica de oxígeno.")
fio2 = entrada("FiO2 · proporción (0,21 a 1,00)", "fio2", 0.21, 1.00, 0.40, 0.01, "%.2f",
               "Fracción inspirada de oxígeno como proporción: 0,21 = aire ambiente, 1,00 = 100 %.")
edad = entrada("Edad · meses (0 a 24)", "edad", 0, 24, 6, 1, "%d",
               "Edad del paciente en meses. El modelo aplica hasta los 24 meses; "
               "la edad define la FR normal del pROX (40 rpm hasta los 12 meses inclusive, 30 rpm después).")

prox = calcular_prox(sat, fio2, fr, edad)
fr_ref = fr_normal(edad)
edad_txt = "hasta 12 meses" if edad <= PROX_EDAD_CORTE_MESES else "mayor de 12 meses"
prox_txt = f"{prox:.1f}".replace(".", ",")
fio2_txt = f"{fio2:.2f}".replace(".", ",")

st.markdown(
    f"""
    <div class="pf-glass pf-prox">
      <div>
        <div class="pf-stat-label">pROX calculado</div>
        <div class="pf-prox-value">{prox_txt}</div>
      </div>
      <div class="pf-prox-formula">
        ({sat} / {fio2_txt}) / ({fr} / <b>{fr_ref}</b>)<br>
        Paciente {edad_txt} · FR normal <b>{fr_ref} rpm</b>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
calcular = st.button("🔎 Calcular Riesgo")

# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #
if calcular:
    try:
        pct, prox, Z = predecir_riesgo(modelo, tal, flujo, fr, sat, fio2, edad)
    except Exception as e:  # noqa: BLE001
        st.error(f"Error al ejecutar la predicción: {e}")
        st.stop()

    prox_txt = f"{prox:.1f}".replace(".", ",")
    pct_txt = f"{pct:.1f}".replace(".", ",")
    youden_txt = f"{YOUDEN_PCT:.1f}".replace(".", ",")
    sens_txt = f"{YOUDEN_SENS:.1f}".replace(".", ",")
    espec_txt = f"{YOUDEN_ESPEC:.1f}".replace(".", ",")

    if pct < RIESGO_BAJO_PCT:
        color, color_dark, icono = "#22c55e", "#0f6b3a", "🟢"
        nivel, subtitulo = "RIESGO BAJO", "Estabilidad Clínica"
        mensaje = "Riesgo Bajo. Perfil compatible con estabilidad clínica."
    elif pct < YOUDEN_PCT:
        color, color_dark, icono = "#f59e0b", "#9a4a06", "🟡"
        nivel, subtitulo = "RIESGO MODERADO", "Monitoreo Estricto"
        mensaje = "Riesgo Moderado. Se sugiere monitoreo estricto."
    else:
        color, color_dark, icono = "#ef4444", "#8f1d1d", "🚨"
        nivel, subtitulo = "ALERTA CRÍTICA DE FRACASO", f"Sensibilidad del modelo: {sens_txt} %"
        mensaje = (
            f"🚨 ALERTA CRÍTICA DE FRACASO (Sensibilidad del modelo: {sens_txt} %). "
            "El perfil comparte criterios con el grupo de fallo histórico. "
            "Evaluar de inmediato estrategias alternativas."
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
              <span>Sensibilidad {sens_txt}% · Especificidad {espec_txt}%</span>
            </div>
          </div>
          <div class="pf-scale"><span>0%</span><span>30%</span><span>{youden_txt}%</span><span>100%</span></div>
        </div>
        <div class="pf-glass pf-alert" style="border-left-color:{color};">
          <b>{icono} Alerta clínica · {nivel.title()}</b>
          {mensaje}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Ver valores utilizados"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Variable": [
                        "Score de Tal (TAL)", "Flujo inicial (FLUJO, L/min)",
                        "pROX calculado", "FR (rpm)", "Sat (%)", "FiO2 (proporción)",
                        "Edad (meses)", "FR normal por edad (rpm)", "RRSD (FR / FR normal)",
                        "z TAL", "z FLUJO", "z pROX (invertido)",
                    ],
                    "Valor": [
                        str(tal), str(flujo), prox_txt, str(fr), str(sat), fio2_txt,
                        str(edad), str(fr_ref),
                        f"{calcular_rrsd(fr, edad):.2f}".replace(".", ","),
                        f"{Z['TAL'].iloc[0]:+.2f}".replace(".", ","),
                        f"{Z['FLUJO'].iloc[0]:+.2f}".replace(".", ","),
                        f"{Z['pROX'].iloc[0]:+.2f}".replace(".", ","),
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

_youden_txt = f"{YOUDEN_PCT:.1f}".replace(".", ",")
_sens_txt = f"{YOUDEN_SENS:.1f}".replace(".", ",")
_espec_txt = f"{YOUDEN_ESPEC:.1f}".replace(".", ",")
st.markdown(
    f"""
    <div class="pf-glass pf-legend">
      <div class="pf-legend-title">Escala de riesgo · índice de Youden {_youden_txt} %</div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#22c55e; color:#22c55e"></span>
        <div><b>🟢 Riesgo Bajo · Estabilidad clínica</b>
        Probabilidad de fracaso menor al 30 %. Perfil compatible con estabilidad clínica.</div>
      </div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#f59e0b; color:#f59e0b"></span>
        <div><b>🟡 Riesgo Moderado · Monitoreo estricto</b>
        Entre 30 % y {_youden_txt} %. Se sugiere monitoreo estricto.</div>
      </div>
      <div class="pf-legend-row">
        <span class="pf-dot" style="background:#ef4444; color:#ef4444"></span>
        <div><b>🚨 Alerta crítica de fracaso</b>
        Igual o mayor al {_youden_txt} % (índice de Youden, sensibilidad {_sens_txt} %,
        especificidad {_espec_txt} %):
        el perfil comparte criterios con el grupo de fallo histórico.
        Evaluar de inmediato estrategias alternativas.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="pf-glass pf-method">
      <div class="pf-legend-title">Sobre el modelo</div>
      <div class="pf-method-grid">
        <div class="pf-stat">
          <div class="pf-stat-label">Enfoque</div>
          <div class="pf-stat-value">Análisis multivariado</div>
        </div>
        <div class="pf-stat">
          <div class="pf-stat-label">Segmentación</div>
          <div class="pf-stat-value">K-Means sobre PCA</div>
        </div>
        <div class="pf-stat">
          <div class="pf-stat-label">Algoritmo</div>
          <div class="pf-stat-value">Regresión logística</div>
        </div>
        <div class="pf-stat pf-stat-hl">
          <div class="pf-stat-label">AUC</div>
          <div class="pf-stat-value">{AUC_PCT}%</div>
        </div>
      </div>
      <p class="pf-method-text">
        PediaFlow-AI se construyó a partir de un <b>análisis multivariado</b> de pacientes
        pediátricos con CNAF. Primero se redujo la dimensionalidad con <b>PCA</b> (análisis de
        componentes principales) y se identificaron perfiles de pacientes mediante
        <b>K-Means</b>, distinguiendo el clúster de fracaso histórico. Luego se entrenó un
        modelo de <b>regresión logística</b> (logistic regression) con tres variables:
        <b>Score de Tal</b>, <b>flujo inicial</b> (L/min) y <b>pROX</b>, un índice de
        oxigenación ajustado por edad que la app calcula automáticamente a partir de la
        saturación, la FiO2, la frecuencia respiratoria y la edad. El modelo alcanzó un
        <b>AUC de {AUC_PCT}%</b>; el punto de corte óptimo se fijó con el índice de Youden
        ({_youden_txt}%, sensibilidad {_sens_txt}%, especificidad {_espec_txt}%). Las
        variables se estandarizan con las
        medias y desvíos guardados en el modelo antes de cada predicción.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="pf-foot">PediaFlow-AI · Powered by {AUTOR}<br>'
    "Herramienta de apoyo a la decisión clínica. No reemplaza el criterio médico.</div>",
    unsafe_allow_html=True,
)
