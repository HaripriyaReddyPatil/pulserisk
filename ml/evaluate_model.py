from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data" / "patients.csv"
MODEL_PATH = BASE / "ml" / "risk_model.joblib"

OUTPUT_DIR = BASE / "ml" / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD_CSV = OUTPUT_DIR / "threshold_analysis.csv"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.json"

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


def main():
    print("\nPulseRisk Model Evaluation")
    print("=" * 55)

    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Load trained production model
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"ROC-AUC: {roc_auc:.4f}")

    # -----------------------------------------------------
    # Threshold analysis
    # -----------------------------------------------------

    threshold_rows = []

    for threshold in [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)

    threshold_df.to_csv(
        THRESHOLD_CSV,
        index=False,
    )

    best_f1_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    best_threshold = float(best_f1_row["threshold"])

    print("\nBest threshold by F1:")
    print(best_f1_row.to_dict())

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    final_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        final_predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Readmitted",
            "Readmitted",
        ],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        f"PulseRisk Confusion Matrix\nThreshold = {best_threshold:.2f}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # ROC curve
    # -----------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        label=f"PulseRisk (AUC = {roc_auc:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("PulseRisk ROC Curve")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Precision-recall curve
    # -----------------------------------------------------

    precision_values, recall_values, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        recall_values,
        precision_values,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "PulseRisk Precision-Recall Curve"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "precision_recall_curve.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Threshold trade-off plot
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="o",
        label="Precision",
    )

    ax.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        marker="o",
        label="Recall",
    )

    ax.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        marker="o",
        label="F1",
    )

    ax.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best F1 threshold = {best_threshold:.2f}",
    )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        "Precision, Recall and F1 by Decision Threshold"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "threshold_tradeoff.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    final_precision = precision_score(
        y_test,
        final_predictions,
        zero_division=0,
    )

    final_recall = recall_score(
        y_test,
        final_predictions,
        zero_division=0,
    )

    final_f1 = f1_score(
        y_test,
        final_predictions,
        zero_division=0,
    )

    summary = {
        "roc_auc": round(float(roc_auc), 4),
        "recommended_threshold": round(
            best_threshold,
            2,
        ),
        "precision_at_recommended_threshold": round(
            float(final_precision),
            4,
        ),
        "recall_at_recommended_threshold": round(
            float(final_recall),
            4,
        ),
        "f1_at_recommended_threshold": round(
            float(final_f1),
            4,
        ),
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        },
        "test_records": int(len(X_test)),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("\nEvaluation complete.")
    print(json.dumps(summary, indent=2))

    print("\nSaved evaluation artifacts:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path)


if __name__ == "__main__":
    main()