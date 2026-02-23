import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Cargar_datos import cargar_datos
from model_monitoring import monitor_data_drift

import sys
from pathlib import Path

root_path = Path.cwd().parent 
sys.path.append(str(root_path / "src"))

st.set_page_config(page_title="Data Drift Monitoring", layout="wide")

st.title("📊 Dashboard de Detección de Data Drift")

# =====================================================
# 1. Carga y simulación de datos
# =====================================================
df = cargar_datos()

df = df.dropna().reset_index(drop=True)

mitad = len(df)//2
reference_df = df.iloc[:mitad]
current_df = df.iloc[mitad:]

st.success(f"Datos cargados correctamente | Referencia: {len(reference_df)} | Producción: {len(current_df)}")

# =====================================================
# 2. Selección de variables
# =====================================================
numeric_features = reference_df.select_dtypes(include="number").columns.tolist()
categorical_features = reference_df.select_dtypes(exclude="number").columns.tolist()

with st.sidebar:
    st.header("⚙️ Configuración de Monitoreo")

    p_value_threshold = st.slider(
        "Umbral p-value (Drift)",
        min_value=0.01,
        max_value=0.10,
        value=0.05
    )

    psi_threshold = st.slider(
        "Umbral PSI",
        min_value=0.05,
        max_value=0.50,
        value=0.20
    )

# =====================================================
# 3. Cálculo de Data Drift
# =====================================================
drift_results = monitor_data_drift(
    reference_df,
    current_df,
    numeric_features,
    categorical_features
)

st.subheader("📋 Resumen de métricas de drift")
st.dataframe(drift_results)

# =====================================================
# 4. Indicadores de alerta
# =====================================================
st.subheader("🚦 Indicadores de alerta")
st.write("Columnas disponibles en drift_results:", drift_results.columns.tolist())

drift_detected = drift_results[
    ((drift_results["tipo"] == "numeric") & (drift_results["ks_p_value"] < p_value_threshold)) |
    ((drift_results["tipo"] == "numeric") & (drift_results["psi"] > psi_threshold))
]

if len(drift_detected) > 0:
    st.error(f"⚠️ Se detectó Data Drift en {len(drift_detected)} variables.")
else:
    st.success("✅ No se detectó Data Drift significativo.")

# =====================================================
# 5. Visualización de distribuciones
# =====================================================
st.subheader("📈 Comparación de distribuciones")

numeric_drift_vars = drift_results[
    drift_results["tipo"] == "numeric"
]["variable"].tolist()

selected_var = st.selectbox(
    "Seleccioná una variable numérica",
    numeric_drift_vars
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(reference_df[selected_var], bins=30, alpha=0.6, density=True, label="Referencia")
ax.hist(current_df[selected_var], bins=30, alpha=0.6, density=True, label="Producción")

ax.set_title(f"Distribución de {selected_var}")
ax.legend()

st.pyplot(fig)

# =====================================================
# 6. Recomendaciones automáticas
# =====================================================
st.subheader("💡 Recomendaciones")

if len(drift_detected) > 0:
    st.warning("Se recomienda evaluar re-entrenamiento del modelo o revisión de variables con drift.")
else:
    st.info("El modelo puede seguir operando sin ajustes inmediatos.")