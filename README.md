# Healthcare Patient Risk & Insights System

A portfolio-ready end-to-end healthcare analytics application built with **Streamlit + FastAPI + scikit-learn**.

> **Important:** This project uses synthetic patient data and is intended for software/ML demonstration only. It is not a medical device and must not be used for clinical decision-making.

## Why this project stands out

This is more than a model-in-a-notebook. It demonstrates:

- End-to-end ML product design
- REST API development with FastAPI
- Interactive Streamlit UI
- Patient-level risk scoring
- Explainable ML
- Cohort analytics
- Patient timeline and trend analysis
- Data-quality monitoring
- Audit logging
- Production-style project structure
- Unit tests
- Docker support

## Key Features

### Executive Dashboard
- Total patient count
- High-risk patient count
- Average risk score
- Readmission rate
- Risk distribution
- Age/risk trends
- Top clinical risk drivers

### Patient Explorer
Search or select a patient and view:
- Demographics
- Clinical profile
- Vitals and labs
- Utilization history
- Risk score
- Risk category
- Personalized risk drivers
- Patient timeline
- Automated care-team summary

### Risk Prediction
Predict risk for a new patient using:
- Age
- BMI
- Systolic blood pressure
- HbA1c
- LDL
- Length of stay
- Prior admissions
- Comorbidity count

### Explainability
Global and patient-level feature importance using a model-agnostic contribution view.

### Cohort Insights
Filter by:
- Age
- Gender
- Risk level
- Diabetes
- Hypertension
- Readmission status

### Data Quality
Detect:
- Missing values
- Duplicate records
- Out-of-range values
- Invalid categories

### Audit Log
Tracks application actions such as:
- Patient viewed
- Risk scored
- Filter applied
- Export requested

---

## Architecture

```text
Synthetic Patient Data
        |
        v
ML Training Pipeline
        |
        v
Saved Random Forest Model
        |
        v
FastAPI Backend
  - /health
  - /patients
  - /patients/{id}
  - /predict
  - /metrics
        |
        v
Streamlit Frontend
  - Dashboard
  - Patient Explorer
  - Risk Predictor
  - Cohort Analytics
  - Data Quality
  - Audit Log
```

---

## Tech Stack

**Frontend**
- Streamlit
- Plotly
- Pandas

**Backend**
- FastAPI
- Pydantic
- Uvicorn

**Machine Learning**
- scikit-learn
- RandomForestClassifier
- Standard preprocessing pipeline
- ROC-AUC evaluation

**Data**
- Synthetic healthcare dataset
- CSV
- SQLite audit log

---

## Run locally

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate data and train model

```bash
python ml/generate_data.py
python ml/train_model.py
```

### 4. Start FastAPI

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Start Streamlit in a second terminal

```bash
streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

FastAPI docs:

```text
http://localhost:8000/docs
```

---

## Resume bullet examples

- Built an end-to-end healthcare risk analytics platform using Streamlit, FastAPI, scikit-learn, and Plotly to score patient risk and surface cohort-level insights.
- Developed a Random Forest risk model with reusable preprocessing pipelines, REST-based inference, evaluation metrics, and feature-level explainability.
- Designed an interactive clinical dashboard with patient timelines, cohort filters, data-quality monitoring, audit logging, and automated patient summaries.
- Implemented production-style architecture with modular frontend/backend services, testing, Docker support, and synthetic healthcare data for privacy-safe deployment.

---

## Future Enhancements

- PostgreSQL
- JWT authentication
- Role-based access control
- MLflow experiment tracking
- SHAP explainability
- FHIR ingestion
- LLM-generated care summaries
- CI/CD with GitHub Actions
- AWS/Azure deployment
- Drift monitoring
- Model versioning
