import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chisquare
from scipy.spatial.distance import jensenshannon

#Este módulo implementa el monitoreo de data drift comparando una población histórica contra una actual, utilizando métricas estadísticas para variables numéricas y categóricas.

# =====================================================
# PSI – Population Stability Index (variables numéricas)
# =====================================================
def calculate_psi(expected, actual, bins=10):
    """
    Calcula el Population Stability Index (PSI) entre
    una distribución de referencia y una actual.
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Definición de bins a partir de percentiles del histórico
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum(
        (actual_counts - expected_counts)
        * np.log((actual_counts + 1e-6) / (expected_counts + 1e-6))
    )

    return psi


# =====================================================
# KS Test – Kolmogorov-Smirnov (numéricas)
# =====================================================
def calculate_ks(reference, current):
    """
    Test KS para comparar dos distribuciones continuas.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    statistic, p_value = ks_2samp(reference, current)
    return statistic, p_value


# =====================================================
# Jensen-Shannon Divergence (numéricas)
# =====================================================
def calculate_js(reference, current, bins=10):
    """
    Distancia de Jensen-Shannon entre dos distribuciones.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)

    return jensenshannon(ref_hist, cur_hist)


# =====================================================
# Chi-cuadrado (variables categóricas)
# =====================================================


def calculate_chi_square(reference_series, current_series):
    """
    Chi-square robusto para variables categóricas
    - Alinea categorías
    - Normaliza distribuciones
    - Evita errores por tamaños distintos
    """

    ref_counts = reference_series.value_counts(dropna=False)
    cur_counts = current_series.value_counts(dropna=False)

    # Alinear categorías
    all_categories = ref_counts.index.union(cur_counts.index)

    ref_counts = ref_counts.reindex(all_categories, fill_value=0)
    cur_counts = cur_counts.reindex(all_categories, fill_value=0)

    # Normalizar
    ref_dist = ref_counts / ref_counts.sum()
    cur_dist = cur_counts / cur_counts.sum()

    # Escalar para que tengan la misma suma
    cur_dist_scaled = cur_dist * ref_dist.sum()

    # Evitar ceros absolutos
    epsilon = 1e-6
    ref_dist += epsilon
    cur_dist_scaled += epsilon

    chi_stat, p_value = chisquare(
        f_obs=cur_dist_scaled,
        f_exp=ref_dist
    )

    return chi_stat, p_value

# =====================================================
# Función principal de monitoreo de Data Drift
# =====================================================
def monitor_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list
) -> pd.DataFrame:
    """
    Ejecuta métricas de data drift entre un dataset histórico
    y uno actual, para variables numéricas y categóricas.
    """
    results = []

    # ---------------------
    # Variables numéricas
    # ---------------------
    for col in numeric_features:
        psi = calculate_psi(reference_df[col], current_df[col])
        ks_stat, ks_p = calculate_ks(reference_df[col], current_df[col])
        js = calculate_js(reference_df[col], current_df[col])

        results.append({
            "variable": col,
            "tipo": "numerica",
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_p_value": ks_p,
            "js_divergence": js
        })

    # ---------------------
    # Variables categóricas
    # ---------------------
    for col in categorical_features:
        chi_stat, chi_p = calculate_chi_square(
            reference_df[col],
            current_df[col]
        )

        results.append({
            "variable": col,
            "tipo": "categorica",
            "chi_square": chi_stat,
            "chi_p_value": chi_p
        })

    return pd.DataFrame(results)