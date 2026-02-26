from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib



# =====================================================
# Inicialización de la API
# =====================================================
EXPECTED_FEATURES = [
    "capital_prestado",
    "tipo_credito",
    "salario_cliente",
    "plazo_meses",
    "edad_cliente",
    "puntaje_datacredito",
    "cuota_pactada",
    "cant_creditosvigentes",
    "puntaje",
    "total_otros_prestamos",
    "saldo_total",
    "saldo_principal",
    "saldo_mora",
    "saldo_mora_codeudor",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
    "huella_consulta",
    "tipo_laboral",
    "creditos_sectorFinanciero",
    "creditos_sectorReal",
    "creditos_sectorCooperativo"
]

app = FastAPI(
    title="API Modelo Riesgo Crediticio",
    description="Servicio de predicción de pago a tiempo (batch)",
    version="1.0.0"
)

# =====================================================
# Carga del modelo y preprocesador
# =====================================================

model = joblib.load("artifacts/logistic_model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

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
    df = pd.DataFrame(data.records)

    # Agregar columnas faltantes como NaN
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = None

    # Reordenar columnas como espera el modelo
    df = df[EXPECTED_FEATURES]

    X = preprocessor.transform(df)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }