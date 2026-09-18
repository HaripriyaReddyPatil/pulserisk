from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from backend.schemas import (
    PatientFeatures,
    PredictionResponse,
)

from backend.services import (
    load_data,
    load_model,
    load_metrics,
    load_metadata,
    load_model_comparison,
    load_evaluation_summary,
    global_feature_importance,
    predict,
    log_action,
)


# =========================================================
# Model lifecycle
# =========================================================

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained ML model when the API starts
    and release the reference when the API stops.
    """

    global model

    try:
        model = load_model()

    except Exception as exc:
        print(
            f"Model loading failed: {exc}"
        )

        model = None

    yield

    model = None


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="PulseRisk API",
    version="1.1.0",
    description=(
        "Machine-learning API for synthetic "
        "30-day patient readmission risk analysis. "
        "Portfolio demonstration only."
    ),
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",

        "model_loaded": (
            model is not None
        ),

        "api_version": "1.1.0",

        "disclaimer": (
            "Synthetic portfolio "
            "demonstration only. "
            "Not for clinical use."
        ),
    }


# =========================================================
# Model information
# =========================================================

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    return {
        "metadata": (
            load_metadata()
        ),

        "comparison": (
            load_model_comparison()
        ),

        "evaluation": (
            load_evaluation_summary()
        ),

        "feature_importance": (
            global_feature_importance(
                model
            )
        ),
    }


# =========================================================
# Metrics
# =========================================================

@app.get("/metrics")
def metrics():
    return load_metrics()


# =========================================================
# Patients
# =========================================================

@app.get("/patients")
def patients(
    limit: int = 100,
):
    """
    Return a limited collection of
    synthetic patient records.
    """

    data = load_data()

    limit = max(
        1,
        min(
            limit,
            500,
        ),
    )

    records = (
        data
        .head(limit)
        .fillna("")
        .to_dict(
            orient="records"
        )
    )

    return {
        "count": len(records),
        "patients": records,
    }


# =========================================================
# Individual patient
# =========================================================

@app.get(
    "/patients/{patient_id}"
)
def patient_by_id(
    patient_id: str,
):
    data = load_data()

    result = data[
        data[
            "patient_id"
        ].astype(str)
        == str(patient_id)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Patient not found."
            ),
        )

    patient = (
        result.iloc[0]
        .fillna("")
        .to_dict()
    )

    log_action(
        "PATIENT_API_VIEWED",
        patient_id,
    )

    return patient


# =========================================================
# Prediction
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_risk(
    patient: PatientFeatures,
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded."
            ),
        )

    try:
        patient_data = (
            patient.model_dump()
            if hasattr(
                patient,
                "model_dump",
            )
            else patient.dict()
        )

        result = predict(
            patient_data,
            model,
        )

        log_action(
            "RISK_API_SCORED",
            (
                f"risk="
                f"{result['risk_probability']:.4f}; "
                f"level="
                f"{result['risk_level']}; "
                f"model="
                f"{result['model_type']}"
            ),
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: "
                f"{str(exc)}"
            ),
        )


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():
    return {
        "application":
            "PulseRisk API",

        "version":
            "1.1.0",

        "status":
            "running",

        "documentation":
            "/docs",

        "health":
            "/health",

        "model_information":
            "/model-info",

        "disclaimer": (
            "Synthetic-data portfolio "
            "demonstration only. "
            "Not for clinical use."
        ),
    }