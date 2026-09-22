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

El modelo usa tres columnas: `TAL`, `FLUJO` y `pROX`. Las dos primeras se
cargan directamente; el `pROX` se calcula de forma automática a partir de
cuatro datos adicionales.

| Campo en la app                 | Columna del modelo | Rango       |
|---------------------------------|--------------------|-------------|
| Score de Tal                    | `TAL`              | 0 – 12      |
| Flujo inicial colocado (L/min)  | `FLUJO`            | 1 – 60      |
| Frecuencia Respiratoria (rpm)   | → `pROX`           | 15 – 110    |
| Saturación de Oxígeno (%)       | → `pROX`           | 70 – 100    |
| FiO2 (proporción)               | → `pROX`           | 0.21 – 1.00 |
| Edad (meses)                    | → `pROX`           | 0 – 24      |

El modelo está desarrollado para pacientes de hasta 24 meses.

### Pasos al pulsar «Calcular Riesgo»

1. **FR normal por edad**: 40 rpm si la edad es ≤ 12 meses, 30 rpm si es mayor.
2. **RRSD** = FR / FR normal.
3. **pROX crudo** = (SpO2 / FiO2) / RRSD. La FiO2 se ingresa como proporción
   (0.21 = aire ambiente, 1.00 = 100 %), que equivale a FiO2 % / 100.
4. Vector `[TAL, FLUJO, pROX]` en el orden que espera el modelo.
5. **Normalización** con los metadatos guardados en el `.pkl`
   (`modelo.means_` y `modelo.stds_`): `z = (x - media) / desvío`.
   El z del pROX se multiplica por −1 antes de la predicción: el coeficiente del
   pROX es levemente positivo (+0.05) por el ajuste conjunto con el TAL, y la
   inversión garantiza que un pROX alto (mejor oxigenación) reduzca el riesgo.
6. `predict_proba` → probabilidad de fracaso, acotada entre 0 % y 100 %.

## Escala de riesgo (índice de Youden 44,4 %)
- 🟢 **Riesgo Bajo**: menor al 30 %. Perfil compatible con estabilidad clínica.
- 🟡 **Riesgo Moderado**: entre 30 % y 44,3 %. Se sugiere monitoreo estricto.
- 🚨 **Alerta crítica de fracaso**: igual o mayor al 44,4 % (Youden, sensibilidad
  100 %, especificidad 66,7 %). El perfil comparte criterios con el grupo de fallo
  histórico. Evaluar de inmediato estrategias alternativas.

## Sobre el modelo

- **Enfoque**: análisis multivariado de pacientes pediátricos con CNAF.
- **Segmentación**: reducción de dimensionalidad con PCA y clustering con K-Means,
  que identificó el clúster de fracaso histórico (Cluster 1).
- **Algoritmo**: regresión logística (logistic regression) con tres variables
  (`TAL`, `FLUJO`, `pROX`), entrenada con scikit-learn 1.6.1.
- **Desempeño**: AUC de 85 %.
- **Punto de corte**: índice de Youden en 44,4 % de probabilidad, con
  sensibilidad del 100 % y especificidad del 66,7 %.
- **Autor**: Catriel Rossi.
