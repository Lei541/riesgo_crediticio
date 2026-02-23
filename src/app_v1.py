import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model_monitoring import monitor_data_drift

# =====================================================
# Configuración general
# =====================================================
st.set_page_config(
    page_title="Monitoreo de Data Drift",
    layout="wide"
)

st.title("📊 Monitoreo de Data Drift – Modelo de Crédito")
st.markdown(
    """
    Esta aplicación permite detectar **cambios en la distribución de los datos**
    entre un conjunto histórico (entrenamiento) y datos actuales (producción),
    utilizando métricas estadísticas de *data drift*.
    """
)

# =====================================================
# Carga de datos
# =====================================================
st.sidebar.header("📂 Carga de datos")

reference_file = st.sidebar.file_uploader(
    "Dataset de referencia (train)",
    type=["csv"]
)

current_file = st.sidebar.file_uploader(
    "Dataset actual (producción)",
    type=["csv"]
)

if reference_file and current_file:
    ref_df = pd.read_csv(reference_file)
    cur_df = pd.read_csv(current_file)

    st.subheader("📌 Vista previa de los datos")
    st.write("Dataset de referencia")
    st.dataframe(ref_df.head())

    st.write("Dataset actual")
    st.dataframe(cur_df.head())

    # =====================================================
    # Selección de variables
    # =====================================================
    st.sidebar.header("🧮 Selección de variables")

    numeric_features = st.sidebar.multiselect(
        "Variables numéricas",
        ref_df.select_dtypes(include="number").columns.tolist()
    )

    categorical_features = st.sidebar.multiselect(
        "Variables categóricas",
        ref_df.select_dtypes(exclude="number").columns.tolist()
    )

    # =====================================================
    # Umbrales de alerta
    # =====================================================
    st.sidebar.header("🚦 Umbrales de alerta")

    psi_threshold = st.sidebar.slider(
        "Umbral PSI",
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.01
    )

    # =====================================================
    # Ejecución del monitoreo
    # =====================================================
    if st.button("🔍 Analizar Data Drift"):
        drift_results = monitor_data_drift(
            ref_df,
            cur_df,
            numeric_features,
            categorical_features
        )

        st.subheader("📋 Resultados de Data Drift")
        st.dataframe(drift_results)

        # =====================================================
        # Indicadores visuales (semáforo)
        # =====================================================
        st.subheader("🚦 Indicadores de alerta")

        if "psi" in drift_results.columns:
            drift_results["alerta_psi"] = drift_results["psi"].apply(
                lambda x: "🟢 Bajo" if x < 0.1 else
                          "🟡 Medio" if x < psi_threshold else
                          "🔴 Alto"
            )

            st.dataframe(
                drift_results[["variable", "psi", "alerta_psi"]]
                .sort_values("psi", ascending=False)
            )

        # =====================================================
        # Gráficos comparativos de distribución
        # =====================================================
        st.subheader("📊 Comparación de distribuciones")

        for col in numeric_features:
            fig, ax = plt.subplots(figsize=(6, 4))

            sns.kdeplot(
                ref_df[col],
                label="Referencia",
                ax=ax
            )
            sns.kdeplot(
                cur_df[col],
                label="Actual",
                ax=ax
            )

            ax.set_title(f"Distribución – {col}")
            ax.legend()

            st.pyplot(fig)

        # =====================================================
        # Recomendaciones automáticas
        # =====================================================
        st.subheader("💡 Recomendaciones")

        high_drift_vars = drift_results[
            (drift_results["type"] == "numeric") &
            (drift_results.get("psi", 0) >= psi_threshold)
        ]

        if not high_drift_vars.empty:
            st.error(
                "⚠️ Se detectó **data drift significativo** en las siguientes variables:"
            )
            st.write(high_drift_vars["variable"].tolist())

            st.markdown(
                """
                **Acciones sugeridas:**
                - Evaluar retraining del modelo.
                - Revisar la calidad y origen de los datos recientes.
                - Analizar si hubo cambios en la política de otorgamiento de créditos.
                """
            )
        else:
            st.success(
                "✅ No se detectaron desviaciones significativas. "
                "El modelo puede seguir operando normalmente."
            )

else:
    st.info("⬅️ Cargá los datasets para comenzar el análisis.")