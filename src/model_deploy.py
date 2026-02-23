from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# =====================================================
# Inicialización de la API
# =====================================================
app = FastAPI(
    title="API Modelo Riesgo Crediticio",
    description="Servicio de predicción de pago a tiempo (batch)",
    version="1.0.0"
)

# =====================================================
# Carga del modelo y preprocesador
# =====================================================
model = joblib.load("src/model.joblib")
preprocessor = joblib.load("src/preprocessor.joblib")

# =====================================================
# Esquema de entrada (batch)
# =====================================================
class CreditRequest(BaseModel):
    records: list[dict]

# =====================================================
# Endpoint de salud
# =====================================================
@app.get("/health")
def health_check():
    return {"status": "ok"}

# =====================================================
# Endpoint de predicción batch
# =====================================================
@app.post("/predict")
def predict(data: CreditRequest):
    """
    Recibe múltiples registros y retorna predicciones
    """
    df = pd.DataFrame(data.records)

    X = preprocessor.transform(df)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }