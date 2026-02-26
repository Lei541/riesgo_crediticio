import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)


# =========================================================
# 1. Función resumen de clasificación
# =========================================================
def summarize_classification(y_true, y_pred, y_proba=None):
    """
    Retorna un diccionario con métricas de evaluación
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


# =========================================================
# 2. Construcción de modelos
# =========================================================
def build_model(model_name: str, preprocessor):
    """
    Construye un pipeline completo según el modelo solicitado
    """

    if model_name == "logistic":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        )

    elif model_name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            random_state=42
        )

    else:
        raise ValueError("Modelo no soportado")

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    return model


# =========================================================
# 3. Entrenamiento y evaluación de múltiples modelos
# =========================================================
def train_and_evaluate_models(
    X_train, X_test, y_train, y_test, preprocessor
):
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)
    return results_df, trained_models


# =========================================================
# Guardar modelo entrenado
# =========================================================

import os
import joblib
from pathlib import Path

def save_best_model(model, preprocessor, model_name="logistic_model"):
    """
    Guarda el mejor modelo y el preprocessor por separado
    """
    artifacts_path = Path("artifacts")
    artifacts_path.mkdir(exist_ok=True)

    joblib.dump(model, artifacts_path / f"{model_name}.pkl")
    joblib.dump(preprocessor, artifacts_path / "preprocessor.pkl")