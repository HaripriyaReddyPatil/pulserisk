from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from backend.schemas import PatientFeatures, PredictionResponse
from backend.services import (
    load_data,
    load_model,
    load_metrics,
    global_feature_importance,
    predict,
    log_action,
)

app = FastAPI(
    title="Healthcare Patient Risk & Insights API",
    version="1.0.0",
    description="Synthetic portfolio demonstration API for patient risk scoring.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def startup():
    global model
    try:
        model = load_model()
    except Exception:
        model = None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "disclaimer": "Synthetic portfolio demo only. Not for clinical use.",
    }

@app.get("/metrics")
def metrics():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "evaluation": load_metrics(),
        "feature_importance": global_feature_importance(model),
    }

@app.get("/patients")
def patients(limit: int = 100):
    df = load_data().head(min(limit, 1000)).copy()
    return df.fillna("").to_dict(orient="records")

@app.get("/patients/{patient_id}")
def patient(patient_id: str):
    df = load_data()
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Patient not found.")
    log_action("PATIENT_VIEWED", patient_id)
    return row.iloc[0].fillna("").to_dict()

@app.post("/predict", response_model=PredictionResponse)
def score_patient(payload: PatientFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run ml/generate_data.py and ml/train_model.py first.",
        )
    result = predict(model, payload)
    log_action("RISK_SCORED", f"risk={result['risk_percent']}%")
    return result
