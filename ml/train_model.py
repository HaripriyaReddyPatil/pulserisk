from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "patients.csv"
MODEL_PATH = BASE / "ml" / "risk_model.joblib"
METRICS_PATH = BASE / "ml" / "metrics.json"

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
TARGET = "readmitted_30d"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, FEATURES)]
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("Saved model:", MODEL_PATH)
    print(metrics)

if __name__ == "__main__":
    main()
