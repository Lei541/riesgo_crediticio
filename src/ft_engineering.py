import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def feature_engineering(df: pd.DataFrame, target: str = "Pago_atiempo"):
    """
    Realiza el proceso completo de feature engineering:
    - Limpieza básica
    - Definición de tipos de variables
    - Pipelines de imputación y encoding
    - Split de datos en train y test

    Retorna:
    X_train, X_test, y_train, y_test, preprocessor
    """

    # ============================
    # 1. Limpieza básica
    # ============================

    df = df.copy()

    # Unificar nulos
    df.replace(
        to_replace=["", " ", "NA", "N/A", "na", "null"],
        value=np.nan,
        inplace=True
    )

    
    # Corrección de valores erróneos en "tendencia_ingresos"
    # Definir categorías válidas
    categorias_validas = ["Creciente", "Estable", "Decreciente"]
    
    # Reemplazar valores erróneos por NaN
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(categorias_validas),
        np.nan
    )
    
    
    # Definir rango válido de edad
    edad_min = 18
    edad_max = 100
    
    # Reemplazar valores fuera de rango por NaN
    df["edad_cliente"] = df["edad_cliente"].where(
        df["edad_cliente"].between(edad_min, edad_max),
        np.nan
    )

    # NO usar fecha de crédito (evitar leakage temporal)
    df.drop(columns=["fecha_prestamo"], errors="ignore", inplace=True)
    

    # ============================
    # 2. Definición de variables
    # ============================

    numeric_features = [
        "capital_prestado",
        "plazo_meses",
        "edad_cliente",
        "salario_cliente",
        "total_otros_prestamos",
        "cuota_pactada",
        "puntaje",
        "puntaje_datacredito",
        "cant_creditosvigentes",
        "huella_consulta",
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "saldo_mora_codeudor",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
        "promedio_ingresos_datacredito"
    ]

    categorical_nominal = [
        "tipo_credito",
        "tipo_laboral"
    ]

    categorical_ordinal = [
        "tendencia_ingresos"
    ]

    ordinal_categories = [
        ["Decreciente", "Estable", "Creciente"]
    ]

    # ============================
    # 3. Pipelines por tipo
    # ============================

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    categorical_ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=ordinal_categories))
    ])

    # ============================
    # 4. ColumnTransformer
    # ============================

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat_nom", categorical_nominal_pipeline, categorical_nominal),
            ("cat_ord", categorical_ordinal_pipeline, categorical_ordinal)
        ],
        remainder="drop"
    )

    # ============================
    # 5. Split de datos
    # ============================

    # Aseguramos que no haya nulos en el target antes de hacer el split
    df = df[df[target].notna()]
    
    X = df.drop(columns=[target])
    y = df[target]

    

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor