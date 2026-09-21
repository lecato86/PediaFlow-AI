# PediaFlow-AI

Aplicación Streamlit para estimar el riesgo de fracaso de cánula nasal de
alto flujo (CNAF) al ingreso, a partir de un modelo de scikit-learn
(regresión logística).

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

## Variables de entrada

| Campo en la app               | Columna del modelo | Rango     |
|-------------------------------|--------------------|-----------|
| Score de Tal                  | `tal posta`        | 0 – 12    |
| Frecuencia Cardíaca (lpm)     | `fc`               | 60 – 230  |
| Frecuencia Respiratoria (rpm) | `fr`               | 15 – 110  |
| Saturación de Oxígeno (%)     | `sat`              | 70 – 100  |
| Flujo (L/kg)                  | `l/kg`             | 0.5 – 3.0 |

La salida es la probabilidad de fracaso de CNAF con semáforo:
verde < 30 %, amarillo 30–70 %, rojo > 70 %.
