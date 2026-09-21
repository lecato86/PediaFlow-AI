"""
PediaFlow-AI - Predicción de riesgo de requerimiento de CNAF al ingreso.

Aplicación Streamlit que carga el modelo entrenado (modelo_canula_ingreso.pkl)
y permite ingresar las variables clínicas para obtener una predicción.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "modelo_canula_ingreso.pkl"

st.set_page_config(page_title="PediaFlow-AI", page_icon="🩺", layout="centered")
st.title("🩺 PediaFlow-AI")
st.caption("Riesgo de requerimiento de cánula nasal de alto flujo (CNAF) al ingreso")


@st.cache_resource
def cargar_modelo(path: Path):
    """Carga el modelo serializado con joblib (se cachea entre ejecuciones)."""
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        f"No se encontró el archivo del modelo: `{MODEL_PATH.name}`.\n\n"
        "Colocá `modelo_canula_ingreso.pkl` en la raíz del repositorio, "
        "al lado de `app.py`."
    )
    st.stop()

modelo = cargar_modelo(MODEL_PATH)

# Nombres de variables: se toman del modelo si fue entrenado con un DataFrame.
features = list(getattr(modelo, "feature_names_in_", []))

if not features:
    n = getattr(modelo, "n_features_in_", None)
    if n is None:
        st.error("No se pudo determinar la cantidad de variables de entrada del modelo.")
        st.stop()
    features = [f"variable_{i + 1}" for i in range(n)]
    st.info(
        "El modelo no guardó los nombres de las variables; "
        "se usan nombres genéricos en el mismo orden del entrenamiento."
    )

st.subheader("Datos del paciente")
with st.form("formulario_prediccion"):
    valores = {}
    columnas = st.columns(2)
    for i, feat in enumerate(features):
        with columnas[i % 2]:
            valores[feat] = st.number_input(feat, value=0.0, format="%.3f")
    enviar = st.form_submit_button("Calcular riesgo", type="primary")

if enviar:
    X = pd.DataFrame([valores], columns=features)
    try:
        pred = modelo.predict(X)[0]
        st.divider()
        st.subheader("Resultado")

        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X)[0]
            clases = list(getattr(modelo, "classes_", range(len(proba))))
            idx_pos = clases.index(1) if 1 in clases else int(np.argmax(proba))
            riesgo = float(proba[idx_pos])
            st.metric("Probabilidad de requerir CNAF", f"{riesgo:.1%}")
            st.progress(min(max(riesgo, 0.0), 1.0))
            if riesgo >= 0.5:
                st.warning("Riesgo elevado de requerimiento de CNAF.")
            else:
                st.success("Riesgo bajo de requerimiento de CNAF.")
        else:
            st.metric("Predicción", str(pred))

        with st.expander("Ver datos ingresados"):
            st.dataframe(X, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Error al ejecutar la predicción: {e}")

st.divider()
st.caption(
    "Herramienta de apoyo a la decisión clínica. No reemplaza el criterio médico."
)
