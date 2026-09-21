# PediaFlow-AI

Aplicación Streamlit para estimar el riesgo de requerimiento de cánula nasal de
alto flujo (CNAF) al ingreso, a partir de un modelo de scikit-learn.

## Estructura

```
PediaFlow-AI/
├── app.py                      # aplicación Streamlit
├── modelo_canula_ingreso.pkl   # modelo entrenado (joblib)
├── requirements.txt
└── README.md
```

## Uso local

```bash
pip install -r requirements.txt
streamlit run app.py
```

El archivo `modelo_canula_ingreso.pkl` debe estar en la raíz del repositorio,
en la misma carpeta que `app.py`.
