from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: int = Field(ge=18, le=100)
    bmi: float = Field(ge=10, le=70)
    systolic_bp: float = Field(ge=70, le=260)
    hba1c: float = Field(ge=3, le=20)
    ldl: float = Field(ge=20, le=350)
    length_of_stay: float = Field(ge=0, le=90)
    prior_admissions: int = Field(ge=0, le=30)
    comorbidity_count: int = Field(ge=0, le=20)


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_percent: float
    risk_level: str
    model_type: str
    top_risk_drivers: list[dict]