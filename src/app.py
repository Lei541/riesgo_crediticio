import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 1. Configuración de rutas 
root_path = Path.cwd().parent 
if str(root_path / "src") not in sys.path:
    sys.path.append(str(root_path / "src"))

# Importamos módulos locales
try:
    from Cargar_datos import cargar_datos
    from model_monitoring import monitor_data_drift
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.stop()

st.set_page_config(page_title="Data Drift Monitoring", layout="wide")

st.title("📊 Dashboard de Detección de Data Drift")

# =====================================================
# 1. Carga y simulación de datos
# =====================================================
df = cargar_datos()

# Aseguramos que los tipos de datos sean correctos tras la carga
df = df.apply(pd.to_numeric, errors='ignore')
df = df.dropna().reset_index(drop=True)

# MEZCLA ALEATORIA: Esto rompe cualquier orden previo
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. División estándar (no aleatoria, para mantener tendencias temporales)
mitad = len(df) // 2
reference_df = df.iloc[:mitad].copy()
current_df = df.iloc[mitad:].copy()

st.success(f"Datos cargados correctamente | Referencia: {len(reference_df)} | Producción: {len(current_df)}")

# =====================================================
# 2. Selección de variables
# =====================================================
numeric_features = reference_df.select_dtypes(include="number").columns.tolist()
categorical_features = reference_df.select_dtypes(exclude="number").columns.tolist()

with st.sidebar:
    st.header("⚙️ Configuración de Monitoreo")
    p_value_threshold = st.slider("Umbral p-value (Drift)", 0.01, 0.10, 0.05)
    psi_threshold = st.slider("Umbral PSI", 0.05, 0.50, 0.20)

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

# Verificamos que la columna 'tipo' exista (antes tenías 'type', asegúrate que en monitor_data_drift sea 'tipo')
if "tipo" in drift_results.columns:
    drift_detected = drift_results[
        ((drift_results["tipo"] == "numerica") & (drift_results["ks_p_value"] < p_value_threshold)) |
        ((drift_results["tipo"] == "numerica") & (drift_results["psi"] > psi_threshold))
    ]

    if len(drift_detected) > 0:
        st.error(f"⚠️ Se detectó Data Drift en {len(drift_detected)} variables.")
    else:
        st.success("✅ No se detectó Data Drift significativo.")
else:
    st.warning("No se encontró la columna 'tipo' en los resultados del drift.")

# =====================================================
# 5. Visualización de distribuciones
# =====================================================
st.subheader("📈 Comparación de distribuciones")

# Usamos .str.contains para que detecte "numerica" o "numérica" (con o sin tilde)
# y lo pasamos a minúsculas para evitar errores de mayúsculas.
if "tipo" in drift_results.columns:
    mask_numericas = drift_results["tipo"].str.lower().str.contains("numeric", na=False)
    numeric_drift_vars = drift_results[mask_numericas]["variable"].tolist()
else:
    numeric_drift_vars = []

if numeric_drift_vars:
    selected_var = st.selectbox("Seleccioná una variable numérica", numeric_drift_vars)

    if selected_var:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Graficamos Referencia y Producción
        ax.hist(reference_df[selected_var], bins=30, alpha=0.6, density=True, label="Referencia", color="#1f77b4")
        ax.hist(current_df[selected_var], bins=30, alpha=0.6, density=True, label="Producción", color="#ff7f0e")
        
        ax.set_title(f"Distribución de: {selected_var}")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")
        ax.legend()
        
        # Mostramos el gráfico en Streamlit
        st.pyplot(fig)
else:
    # Mensaje de ayuda si la lista sigue vacía
    st.info("No se detectaron variables de tipo 'numerica'.")
    st.write("Contenido de la columna 'tipo':", drift_results["tipo"].unique()) # Debug rápido

# =====================================================
# 6. Recomendaciones automáticas
# =====================================================
st.subheader("💡 Recomendaciones")
if 'drift_detected' in locals() and len(drift_detected) > 0:
    st.warning("Se recomienda evaluar re-entrenamiento del modelo o revisión de variables con drift.")
else:
    st.info("El modelo puede seguir operando sin ajustes inmediatos.")

# =====================================================
# 7. Análisis Temporal (Simulado)
# =====================================================
st.subheader("📅 Evolución del Drift en el Tiempo")

# Simulamos una métrica que cambia por "batches" (lotes) de datos
puntos_tiempo = np.linspace(0, 1, 10)
drift_evolucion = pd.DataFrame({
    'Fecha': pd.date_range(start='2024-01-01', periods=10, freq='M'),
    'PSI': [0.05, 0.07, 0.06, 0.12, 0.15, 0.22, 0.25, 0.24, 0.28, 0.30]
})

st.line_chart(drift_evolucion.set_index('Fecha'))
st.info("La gráfica muestra cómo el PSI ha cruzado el umbral crítico en los últimos meses.")