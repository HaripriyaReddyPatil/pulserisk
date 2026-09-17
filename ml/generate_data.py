from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1400

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    out = Path(__file__).resolve().parents[1] / "data" / "patients.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    age = RNG.integers(18, 91, N)
    gender = RNG.choice(["Female", "Male"], N, p=[0.52, 0.48])
    bmi = np.clip(RNG.normal(28, 6, N), 16, 52)
    systolic_bp = np.clip(RNG.normal(128 + 0.18*(age-50), 18, N), 85, 210)
    hba1c = np.clip(RNG.normal(5.9 + 0.02*np.maximum(age-45, 0), 1.15, N), 4.0, 13.5)
    ldl = np.clip(RNG.normal(116 + 0.15*(age-50), 30, N), 35, 250)
    length_of_stay = np.clip(RNG.gamma(2.2, 2.0, N), 1, 25).round(1)
    prior_admissions = np.clip(RNG.poisson(0.7 + 0.012*np.maximum(age-50, 0), N), 0, 8)
    comorbidity_count = np.clip(
        RNG.poisson(0.8 + 0.02*np.maximum(age-45, 0), N), 0, 8
    )

    diabetes = (hba1c >= 6.5) | (RNG.random(N) < sigmoid((bmi-31)/5)*0.15)
    hypertension = (systolic_bp >= 140) | (RNG.random(N) < 0.08)
    smoker = RNG.random(N) < np.clip(0.22 - 0.001*(age-40), 0.08, 0.25)

    logit = (
        -5.2
        + 0.035*(age-50)
        + 0.055*(bmi-27)
        + 0.020*(systolic_bp-125)
        + 0.34*(hba1c-5.5)
        + 0.010*(ldl-110)
        + 0.13*(length_of_stay-3)
        + 0.58*prior_admissions
        + 0.46*comorbidity_count
        + 0.55*diabetes.astype(int)
        + 0.35*hypertension.astype(int)
        + 0.32*smoker.astype(int)
    )
    risk_probability = sigmoid(logit)
    readmitted_30d = RNG.binomial(1, risk_probability)

    patient_ids = [f"PT-{10000+i}" for i in range(N)]
    admit_dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        RNG.integers(0, 250, N), unit="D"
    )

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "gender": gender,
        "bmi": bmi.round(1),
        "systolic_bp": systolic_bp.round(0).astype(int),
        "hba1c": hba1c.round(1),
        "ldl": ldl.round(0).astype(int),
        "length_of_stay": length_of_stay,
        "prior_admissions": prior_admissions,
        "comorbidity_count": comorbidity_count,
        "diabetes": diabetes.astype(int),
        "hypertension": hypertension.astype(int),
        "smoker": smoker.astype(int),
        "readmitted_30d": readmitted_30d,
        "last_admission_date": admit_dates.strftime("%Y-%m-%d"),
    })

    # Add a tiny amount of missingness for data-quality demos
    for col in ["bmi", "ldl", "hba1c"]:
        idx = RNG.choice(N, size=max(1, N//100), replace=False)
        df.loc[idx, col] = np.nan

    df.to_csv(out, index=False)
    print(f"Saved {len(df)} synthetic patients to {out}")

if __name__ == "__main__":
    main()
