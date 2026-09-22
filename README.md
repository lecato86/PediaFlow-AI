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

El modelo declara sus variables en `modelo.variables_` (o `feature_names_in_`)
y la app las construye a partir del formulario. La app admite dos modelos:

| Modelo                          | Variables del `.pkl`                              |
|---------------------------------|---------------------------------------------------|
| 3 variables                     | `TAL`, `FLUJO`, `pROX`                            |
| Flujo estratificado por edad    | `TAL`, `pROX`, `FLUJO_menores`, `FLUJO_mayores`   |

`FLUJO_menores` vale el flujo si la edad es ≤ 12 meses y 0 en caso contrario;
`FLUJO_mayores` es el complemento. El `pROX` se calcula de forma automática a
partir de cuatro datos adicionales.

| Campo en la app                 | Variable                          | Rango       |
|---------------------------------|-----------------------------------|-------------|
| Score de Tal                    | `TAL`                             | 0 – 12      |
| Flujo inicial colocado (L/min)  | `FLUJO` o `FLUJO_menores/mayores` | 1 – 60      |
| Frecuencia Respiratoria (rpm)   | → `pROX`                          | 15 – 110    |
| Saturación de Oxígeno (%)       | → `pROX`                          | 70 – 100    |
| FiO2 (proporción)               | → `pROX`                          | 0.21 – 1.00 |
| Edad (meses)                    | → `pROX` y estrato de flujo       | 0 – 24      |

El modelo está desarrollado para pacientes de hasta 24 meses. Si el `.pkl`
pide una variable que la app no conoce, se muestra un error explicativo.

### Pasos al pulsar «Calcular Riesgo»

1. **FR normal por edad**: 40 rpm si la edad es ≤ 12 meses, 30 rpm si es mayor.
2. **RRSD** = FR / FR normal.
3. **pROX crudo** = (SpO2 / FiO2) / RRSD. La FiO2 se ingresa como proporción
   (0.21 = aire ambiente, 1.00 = 100 %), que equivale a FiO2 % / 100.
4. **Flujo estratificado** (si el modelo lo pide): `FLUJO_menores` /
   `FLUJO_mayores` según la edad, y vector con las variables en el orden que
   declara el modelo.
5. **Normalización** con los metadatos guardados en el `.pkl`
   (`modelo.means_` y `modelo.stds_`): `z = (x - media) / desvío`.
   El z del pROX se multiplica por −1 antes de la predicción: el coeficiente del
   pROX es levemente positivo (+0.05) por el ajuste conjunto con el TAL, y la
   inversión garantiza que un pROX alto (mejor oxigenación) reduzca el riesgo.
6. `predict_proba` → probabilidad de fracaso, acotada entre 0 % y 100 %.

## Escala de riesgo

El corte de alerta es el índice de Youden del modelo cargado (ver tabla abajo).

- 🟢 **Riesgo Bajo**: menor al 30 %. Perfil compatible con estabilidad clínica.
- 🟡 **Riesgo Moderado**: entre 30 % y el corte de Youden. Se sugiere monitoreo
  estricto.
- 🚨 **Alerta crítica de fracaso**: igual o mayor al corte de Youden. El perfil
  comparte criterios con el grupo de fallo histórico. Evaluar de inmediato
  estrategias alternativas.

## Sobre el modelo

- **Enfoque**: análisis multivariado de pacientes pediátricos con CNAF.
- **Segmentación**: reducción de dimensionalidad con PCA y clustering con K-Means,
  que identificó el clúster de fracaso histórico (Cluster 1).
- **Algoritmo**: regresión logística (logistic regression), entrenada con
  scikit-learn 1.6.1 sobre el 80 % de los pacientes y validada en el 20 % restante.
- **Métricas por modelo** (conjunto de validación), elegidas automáticamente
  según las variables del `.pkl`:

| Modelo                       | AUC    | Corte Youden | Sensibilidad | Especificidad |
|------------------------------|--------|--------------|--------------|---------------|
| 3 variables                  | 85,6 % | 44,4 %       | 100 %        | 66,7 %        |
| Flujo estratificado por edad | 80,6 % | 50,2 %       | 75 %         | 80 %          |

- **Autor**: Catriel Rossi.
