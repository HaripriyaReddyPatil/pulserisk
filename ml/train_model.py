from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data" / "patients.csv"
MODEL_PATH = BASE / "ml" / "risk_model.joblib"
METRICS_PATH = BASE / "ml" / "metrics.json"
COMPARISON_PATH = BASE / "ml" / "model_comparison.json"
METADATA_PATH = BASE / "ml" / "model_metadata.json"


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

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

RANDOM_STATE = 42
TEST_SIZE = 0.22
THRESHOLD = 0.50


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def make_tree_preprocessor():
    """
    Tree-based models do not require feature scaling.
    Missing numeric values are replaced with the median.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURES),
        ]
    )


def make_scaled_preprocessor():
    """
    Logistic Regression benefits from standardized features.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURES),
        ]
    )


# ---------------------------------------------------------
# Candidate models
# ---------------------------------------------------------

def build_models():
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", make_scaled_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                ("preprocessor", make_tree_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=8,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", make_tree_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "roc_auc": round(
            float(roc_auc_score(y_test, probabilities)),
            4,
        ),
        "accuracy": round(
            float(accuracy_score(y_test, predictions)),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
    }


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def main():

    print("\nPulseRisk Model Training")
    print("=" * 55)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {len(df)} rows")

    X = df[FEATURES]
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    models = build_models()

    comparison_results = {}
    trained_models = {}

    # -----------------------------------------------------
    # Train every candidate model
    # -----------------------------------------------------

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        comparison_results[model_name] = metrics
        trained_models[model_name] = model

        print(
            f"ROC-AUC={metrics['roc_auc']} | "
            f"Accuracy={metrics['accuracy']} | "
            f"Precision={metrics['precision']} | "
            f"Recall={metrics['recall']} | "
            f"F1={metrics['f1']}"
        )

    # -----------------------------------------------------
    # Select best model based on ROC-AUC
    # -----------------------------------------------------

    best_model_name = max(
        comparison_results,
        key=lambda name: comparison_results[name]["roc_auc"],
    )

    best_model = trained_models[best_model_name]

    best_metrics = comparison_results[best_model_name].copy()

    best_metrics.update(
        {
            "model": best_model_name,
            "decision_threshold": THRESHOLD,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
    )

    # -----------------------------------------------------
    # Save winning pipeline
    # -----------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    # Existing API can continue reading this file
    METRICS_PATH.write_text(
        json.dumps(
            best_metrics,
            indent=2,
        )
    )

    # Save comparison of all models
    comparison_output = {
        "selection_metric": "roc_auc",
        "selected_model": best_model_name,
        "models": comparison_results,
    }

    COMPARISON_PATH.write_text(
        json.dumps(
            comparison_output,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Save model metadata
    # -----------------------------------------------------

    metadata = {
        "project": "PulseRisk",
        "model_version": "1.1.0",
        "selected_model": best_model_name,
        "selection_metric": "roc_auc",
        "decision_threshold": THRESHOLD,
        "features": FEATURES,
        "target": TARGET,
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "total_records": int(len(df)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("MODEL SELECTION COMPLETE")
    print("=" * 55)

    print(f"\nBest model: {best_model_name}")

    print("\nBest model metrics:")

    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

    print("\nSaved files:")
    print(f"Model:       {MODEL_PATH}")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Comparison:  {COMPARISON_PATH}")
    print(f"Metadata:    {METADATA_PATH}")


if __name__ == "__main__":
    main()