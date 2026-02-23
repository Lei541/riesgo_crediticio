📊 Proyecto Integrador – Riesgo Crediticio
📌 Objetivo del Proyecto

Desarrollar, evaluar y monitorear un modelo de Machine Learning capaz de predecir el pago a tiempo de créditos, utilizando información histórica de clientes, e incorporando buenas prácticas de MLOps para asegurar reproducibilidad, trazabilidad y escalabilidad.

🧠 Caso de Negocio

En el sector financiero, una correcta predicción del comportamiento de pago de los clientes permite:

Reducir el riesgo crediticio.

Optimizar la aprobación de créditos.

Mejorar la rentabilidad del portafolio.

Detectar cambios en el perfil de los solicitantes a lo largo del tiempo.

El modelo desarrollado busca anticipar el riesgo de incumplimiento y habilitar decisiones más informadas en la originación de créditos.

🗂️ Estructura del Proyecto

El proyecto sigue una estructura de carpetas estricta, compatible con procesos de despliegue automatizados mediante pipelines CI/CD (Jenkins):

riesgo_crediticio/
│
├── src/
│   ├── Cargar_datos.py
│   ├── ft_engineering.py
│   ├── modeling.py
│   └── model_monitoring.py
│   └── app.py
│   └── comprension_EDA.ipynb
│
└── README.md

⚙️ Pipeline del Proyecto

El flujo completo del proyecto se organiza en las siguientes etapas:

Carga de datos

Análisis exploratorio de datos (EDA)

Ingeniería de características

Entrenamiento de modelos supervisados

Evaluación y selección del mejor modelo

Monitoreo de data drift en producción

🔍 Análisis Exploratorio (EDA)

Durante el EDA se realizó:

Análisis univariable, bivariable y multivariable.

Corrección de tipos de datos.

Identificación y tratamiento de valores erróneos y nulos.

Evaluación de relaciones entre variables y la variable objetivo (Pago_atiempo).

Identificación de variables con alta correlación y posible data leakage.

Este análisis permitió definir reglas de validación y transformaciones aplicadas en etapas posteriores.

🧪 Feature Engineering

En esta etapa se implementaron:

Imputación de valores faltantes.

Codificación de variables categóricas.

Escalado de variables numéricas.

Eliminación de variables no informativas o con riesgo de fuga de información.

Separación de conjuntos de entrenamiento y evaluación.

El proceso se encapsuló en pipelines reutilizables, facilitando la reproducibilidad del modelo.

🤖 Modelamiento y Evaluación

Se entrenaron y evaluaron múltiples modelos supervisados:

Regresión Logística

Random Forest

Gradient Boosting

Las métricas utilizadas para la evaluación fueron:

Accuracy

Precision

Recall

F1-score

ROC-AUC

📊 Resultados

Si bien algunos modelos alcanzaron métricas cercanas al desempeño perfecto, se seleccionó la Regresión Logística como modelo principal por:

Alta performance.

Simplicidad.

Interpretabilidad.

Facilidad de monitoreo y mantenimiento en producción.

📈 Monitoreo de Data Drift

Se implementó un módulo de monitoreo que permite comparar datos históricos con datos actuales para detectar cambios en la población.

Métricas utilizadas:

Population Stability Index (PSI)

PSI < 0.1 → Sin drift

0.1 ≤ PSI < 0.25 → Drift moderado

PSI ≥ 0.25 → Drift severo

Kolmogorov-Smirnov Test (KS)

p-value < 0.05 → Cambio significativo

Jensen-Shannon Divergence

Valores altos indican diferencias relevantes entre distribuciones

Chi-cuadrado (variables categóricas)

p-value < 0.05 → Cambio significativo

Estas métricas permiten anticipar posibles degradaciones del modelo en producción.

🕒 Análisis Temporal y Tendencias

El sistema incorpora un análisis de la evolución de las métricas a lo largo del tiempo. Esto permite:
- Identificar si el drift es un cambio abrupto o una degradación gradual.
- Detectar estacionalidad en el perfil de los solicitantes.
- Visualizar la tendencia del PSI para anticipar necesidades de re-entrenamiento.

🖥️ Aplicación Streamlit

Se desarrolló una aplicación interactiva en Streamlit que permite:

Comparar distribuciones históricas vs actuales.

Visualizar métricas de data drift por variable.

Mostrar indicadores visuales de alerta (semáforo).

Generar recomendaciones automáticas ante drift significativo.

Facilitar el monitoreo continuo del modelo.

Visualizar la evolución histórica del drift mediante gráficos de tendencia.

Analizar la distribución de los pronósticos entregados para asegurar la estabilidad del modelo.


🔁 Próximos Pasos

Automatizar el retraining del modelo ante drift severo.

Integrar alertas automáticas en el pipeline CI/CD.

Incorporar monitoreo del desempeño del modelo (model drift).

Versionado de modelos y datos.

🚀 **Instrucciones de Ejecución**

Para levantar el entorno de monitoreo localmente, siga estos pasos:

1. **Clonar el repositorio:**
   `git clone https://github.com/Lei941/riesgo_crediticio.git`

2. **Instalar las dependencias:**
   `pip install -r requirements.txt`

3. **Lanzar la aplicación:**
   `streamlit run src/app.py`

🛠️ Tecnologías Utilizadas

Python

pandas, numpy

scikit-learn

seaborn, matplotlib

Streamlit

Git / GitHub