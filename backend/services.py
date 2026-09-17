from pathlib import Path
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "patients.csv"
MODEL_PATH = BASE / "ml" / "risk_model.joblib"
METRICS_PATH = BASE / "ml" / "metrics.json"
AUDIT_DB = BASE / "data" / "audit.db"

FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "hba1c",
    "ldl",
    "length_of_stay",
    "prior_admissions",
    "comorbidity_count",
]

FEATURE_LABELS = {
    "age": "Age",
    "bmi": "BMI",
    "systolic_bp": "Systolic BP",
    "hba1c": "HbA1c",
    "ldl": "LDL",
    "length_of_stay": "Length of stay",
    "prior_admissions": "Prior admissions",
    "comorbidity_count": "Comorbidity count",
}

def load_data():
    return pd.read_csv(DATA_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}

def risk_level(prob):
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"

def global_feature_importance(model):
    rf = model.named_steps["model"]
    pairs = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    return [
        {"feature": FEATURE_LABELS.get(name, name), "importance": float(value)}
        for name, value in pairs
    ]

def patient_drivers(model, row):
    # Portable approximation of local contributions without requiring SHAP.
    rf = model.named_steps["model"]
    medians = load_data()[FEATURES].median(numeric_only=True)

    base_df = pd.DataFrame([medians])
    row_df = pd.DataFrame([row], columns=FEATURES)

    base_prob = model.predict_proba(base_df)[:, 1][0]
    contributions = []

    for feature in FEATURES:
        changed = base_df.copy()
        changed.loc[0, feature] = row_df.loc[0, feature]
        prob = model.predict_proba(changed)[:, 1][0]
        contributions.append({
            "feature": FEATURE_LABELS[feature],
            "impact": float(prob - base_prob),
            "direction": "increases risk" if prob - base_prob >= 0 else "reduces risk",
        })

    return sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)[:5]

def predict(model, payload):
    row = {k: getattr(payload, k) for k in FEATURES}
    df = pd.DataFrame([row])
    prob = float(model.predict_proba(df)[:, 1][0])
    return {
        "risk_probability": round(prob, 4),
        "risk_percent": round(prob * 100, 1),
        "risk_level": risk_level(prob),
        "top_risk_drivers": patient_drivers(model, row),
    }

def init_audit_db():
    con = sqlite3.connect(AUDIT_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TEXT DEFAULT CURRENT_TIMESTAMP,
            action TEXT,
            details TEXT
        )
    """)
    con.commit()
    con.close()

def log_action(action, details=""):
    init_audit_db()
    con = sqlite3.connect(AUDIT_DB)
    con.execute(
        "INSERT INTO audit_log(action, details) VALUES (?, ?)",
        (action, details),
    )
    con.commit()
    con.close()

def get_audit_log(limit=100):
    init_audit_db()
    con = sqlite3.connect(AUDIT_DB)
    df = pd.read_sql_query(
        "SELECT event_time, action, details FROM audit_log ORDER BY id DESC LIMIT ?",
        con,
        params=(limit,),
    )
    con.close()
    return df
