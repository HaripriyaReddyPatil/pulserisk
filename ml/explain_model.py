from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data" / "patients.csv"
MODEL_PATH = BASE / "ml" / "risk_model.joblib"

OUTPUT_DIR = BASE / "ml" / "explanations"
OUTPUT_DIR.mkdir(exist_ok=True)

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


def main():
    print("\nPulseRisk Model Explainability")
    print("=" * 55)

    # Load trained pipeline and dataset
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # This script is designed for the selected Logistic Regression model
    if not hasattr(model, "coef_"):
        raise ValueError(
            "The selected production model does not expose coefficients. "
            "Run ml/train_model.py and confirm Logistic Regression "
            "is the selected model."
        )

    coefficients = model.coef_[0]

    # ---------------------------------------------------------
    # Global model explanation
    # ---------------------------------------------------------

    global_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "coefficient": coefficients,
            "absolute_importance": np.abs(coefficients),
        }
    )

    global_df["direction"] = np.where(
        global_df["coefficient"] > 0,
        "increases risk",
        "decreases risk",
    )

    global_df = global_df.sort_values(
        "absolute_importance",
        ascending=False,
    )

    global_df.to_csv(
        OUTPUT_DIR / "global_feature_importance.csv",
        index=False,
    )

    print("\nGlobal feature effects:\n")

    print(
        global_df[
            ["feature", "coefficient", "direction"]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Global importance chart
    # ---------------------------------------------------------

    importance_plot = global_df.sort_values(
        "absolute_importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        importance_plot["feature"],
        importance_plot["absolute_importance"],
    )

    ax.set_xlabel("Absolute standardized coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("PulseRisk Global Feature Importance")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "global_feature_importance.png",
        dpi=200,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Feature direction chart
    # ---------------------------------------------------------

    direction_plot = global_df.sort_values(
        "coefficient",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        direction_plot["feature"],
        direction_plot["coefficient"],
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel("Logistic regression coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("PulseRisk Feature Direction")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "feature_direction.png",
        dpi=200,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Patient-level explanations
    # ---------------------------------------------------------

    X = df[FEATURES]

    probabilities = pipeline.predict_proba(X)[:, 1]

    df_with_risk = df.copy()
    df_with_risk["predicted_risk"] = probabilities

    low_index = df_with_risk["predicted_risk"].idxmin()

    high_index = df_with_risk["predicted_risk"].idxmax()

    medium_index = (
        df_with_risk["predicted_risk"] - 0.50
    ).abs().idxmin()

    example_indices = {
        "low_risk_example": low_index,
        "medium_risk_example": medium_index,
        "high_risk_example": high_index,
    }

    patient_explanations = {}

    for label, row_index in example_indices.items():

        patient_df = X.loc[[row_index]]

        transformed = preprocessor.transform(
            patient_df
        )

        transformed_values = np.asarray(
            transformed
        ).reshape(-1)

        contributions = (
            transformed_values * coefficients
        )

        contribution_df = pd.DataFrame(
            {
                "feature": FEATURES,
                "raw_value": [
                    patient_df.iloc[0][feature]
                    for feature in FEATURES
                ],
                "standardized_value": transformed_values,
                "coefficient": coefficients,
                "contribution": contributions,
            }
        )

        contribution_df["absolute_contribution"] = (
            contribution_df["contribution"].abs()
        )

        contribution_df["effect"] = np.where(
            contribution_df["contribution"] > 0,
            "pushes risk higher",
            "pushes risk lower",
        )

        contribution_df = contribution_df.sort_values(
            "absolute_contribution",
            ascending=False,
        )

        predicted_risk = float(
            probabilities[row_index]
        )

        patient_explanations[label] = {
            "row_index": int(row_index),
            "predicted_risk": round(
                predicted_risk,
                4,
            ),
            "top_contributors": [
                {
                    "feature": row["feature"],
                    "value": (
                        None
                        if pd.isna(row["raw_value"])
                        else float(row["raw_value"])
                    ),
                    "contribution": round(
                        float(row["contribution"]),
                        4,
                    ),
                    "effect": row["effect"],
                }
                for _, row in contribution_df.head(5).iterrows()
            ],
        }

        contribution_df.to_csv(
            OUTPUT_DIR
            / f"{label}_contributions.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # Save explanations as JSON
    # ---------------------------------------------------------

    explanation_json = {
        "model_type": type(model).__name__,
        "interpretation": (
            "Positive coefficients increase predicted "
            "readmission risk while negative coefficients "
            "decrease predicted risk. Patient-level "
            "contributions are calculated after preprocessing."
        ),
        "global_feature_effects": [
            {
                "feature": row["feature"],
                "coefficient": round(
                    float(row["coefficient"]),
                    4,
                ),
                "direction": row["direction"],
            }
            for _, row in global_df.iterrows()
        ],
        "patient_examples": patient_explanations,
    }

    json_path = (
        OUTPUT_DIR / "model_explanations.json"
    )

    json_path.write_text(
        json.dumps(
            explanation_json,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # Terminal output
    # ---------------------------------------------------------

    print("\nPatient examples:")

    for label, info in patient_explanations.items():

        print(
            f"\n{label}: "
            f"predicted risk = "
            f"{info['predicted_risk']:.2%}"
        )

        for contributor in info["top_contributors"]:

            print(
                f"  {contributor['feature']}: "
                f"{contributor['effect']} "
                f"({contributor['contribution']:+.4f})"
            )

    print("\nSaved explanation artifacts:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path)


if __name__ == "__main__":
    main()