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
METADATA_PATH = BASE / "ml" / "model_metadata.json"
COMPARISON_PATH = BASE / "ml" / "model_comparison.json"

EVALUATION_PATH = (
    BASE
    / "ml"
    / "evaluation"
    / "evaluation_summary.json"
)

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


# ---------------------------------------------------------
# Data / model loaders
# ---------------------------------------------------------

def load_data():
    return pd.read_csv(DATA_PATH)


def load_model():
    return joblib.load(MODEL_PATH)


def load_json(path):
    if path.exists():
        return json.loads(
            path.read_text()
        )

    return {}


def load_metrics():
    return load_json(
        METRICS_PATH
    )


def load_metadata():
    return load_json(
        METADATA_PATH
    )


def load_model_comparison():
    return load_json(
        COMPARISON_PATH
    )


def load_evaluation_summary():
    return load_json(
        EVALUATION_PATH
    )


# ---------------------------------------------------------
# Risk classification
# ---------------------------------------------------------

def risk_level(probability):
    """
    Display-oriented risk bands.

    These bands are used for the UI and are separate
    from the binary decision threshold used during
    formal model evaluation.
    """

    if probability >= 0.70:
        return "High"

    if probability >= 0.40:
        return "Medium"

    return "Low"


def model_type(model):
    estimator = model.named_steps["model"]

    return type(
        estimator
    ).__name__


# ---------------------------------------------------------
# Global model explainability
# ---------------------------------------------------------

def global_feature_importance(model):

    estimator = model.named_steps[
        "model"
    ]

    results = []

    # Logistic Regression / linear models
    if hasattr(
        estimator,
        "coef_",
    ):

        coefficients = (
            estimator.coef_[0]
        )

        for feature, coefficient in zip(
            FEATURES,
            coefficients,
        ):

            results.append(
                {
                    "feature": FEATURE_LABELS[
                        feature
                    ],
                    "importance": float(
                        abs(coefficient)
                    ),
                    "coefficient": float(
                        coefficient
                    ),
                    "direction": (
                        "increases risk"
                        if coefficient >= 0
                        else "reduces risk"
                    ),
                }
            )

    # Tree-based models
    elif hasattr(
        estimator,
        "feature_importances_",
    ):

        importances = (
            estimator.feature_importances_
        )

        for feature, importance in zip(
            FEATURES,
            importances,
        ):

            results.append(
                {
                    "feature": FEATURE_LABELS[
                        feature
                    ],
                    "importance": float(
                        importance
                    ),
                    "coefficient": None,
                    "direction": (
                        "importance only"
                    ),
                }
            )

    return sorted(
        results,
        key=lambda item: item[
            "importance"
        ],
        reverse=True,
    )


# ---------------------------------------------------------
# Patient-level model explanation
# ---------------------------------------------------------

def patient_drivers(
    model,
    row,
):

    estimator = model.named_steps[
        "model"
    ]

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    row_df = pd.DataFrame(
        [row],
        columns=FEATURES,
    )

    # Logistic Regression contributions
    if hasattr(
        estimator,
        "coef_",
    ):

        transformed = (
            preprocessor.transform(
                row_df
            )
        )

        transformed_values = (
            np.asarray(
                transformed
            ).reshape(-1)
        )

        coefficients = (
            estimator.coef_[0]
        )

        contributions = (
            transformed_values
            * coefficients
        )

        results = []

        for feature, contribution in zip(
            FEATURES,
            contributions,
        ):

            results.append(
                {
                    "feature": FEATURE_LABELS[
                        feature
                    ],
                    "impact": round(
                        float(
                            contribution
                        ),
                        4,
                    ),
                    "direction": (
                        "increases risk"
                        if contribution >= 0
                        else "reduces risk"
                    ),
                }
            )

        return sorted(
            results,
            key=lambda item: abs(
                item["impact"]
            ),
            reverse=True,
        )[:5]

    # -----------------------------------------------------
    # Fallback explanation for non-linear models
    # -----------------------------------------------------

    data = load_data()

    medians = (
        data[
            FEATURES
        ]
        .median(
            numeric_only=True
        )
    )

    base_df = pd.DataFrame(
        [medians]
    )

    base_probability = float(
        model.predict_proba(
            base_df
        )[:, 1][0]
    )

    results = []

    for feature in FEATURES:

        changed = (
            base_df.copy()
        )

        changed.loc[
            0,
            feature,
        ] = row_df.loc[
            0,
            feature,
        ]

        probability = float(
            model.predict_proba(
                changed
            )[:, 1][0]
        )

        impact = (
            probability
            - base_probability
        )

        results.append(
            {
                "feature": FEATURE_LABELS[
                    feature
                ],
                "impact": round(
                    float(impact),
                    4,
                ),
                "direction": (
                    "increases risk"
                    if impact >= 0
                    else "reduces risk"
                ),
            }
        )

    return sorted(
        results,
        key=lambda item: abs(
            item["impact"]
        ),
        reverse=True,
    )[:5]


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(
    model,
    payload,
):

    row = {
        feature: getattr(
            payload,
            feature,
        )
        for feature in FEATURES
    }

    df = pd.DataFrame(
        [row]
    )

    probability = float(
        model.predict_proba(
            df
        )[:, 1][0]
    )

    return {
        "risk_probability": round(
            probability,
            4,
        ),
        "risk_percent": round(
            probability * 100,
            1,
        ),
        "risk_level": risk_level(
            probability
        ),
        "model_type": model_type(
            model
        ),
        "top_risk_drivers": (
            patient_drivers(
                model,
                row,
            )
        ),
    }


# ---------------------------------------------------------
# Audit logging
# ---------------------------------------------------------

def init_audit_db():

    con = sqlite3.connect(
        AUDIT_DB
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TEXT DEFAULT CURRENT_TIMESTAMP,
            action TEXT,
            details TEXT
        )
        """
    )

    con.commit()
    con.close()


def log_action(
    action,
    details="",
):

    init_audit_db()

    con = sqlite3.connect(
        AUDIT_DB
    )

    con.execute(
        """
        INSERT INTO audit_log(
            action,
            details
        )
        VALUES (?, ?)
        """,
        (
            action,
            details,
        ),
    )

    con.commit()
    con.close()


def get_audit_log(
    limit=100,
):

    init_audit_db()

    con = sqlite3.connect(
        AUDIT_DB
    )

    df = pd.read_sql_query(
        """
        SELECT
            event_time,
            action,
            details
        FROM audit_log
        ORDER BY id DESC
        LIMIT ?
        """,
        con,
        params=(limit,),
    )

    con.close()

    return df