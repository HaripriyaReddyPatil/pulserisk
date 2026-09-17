import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import joblib

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "patients.csv"
MODEL_PATH = BASE / "ml" / "risk_model.joblib"
METRICS_PATH = BASE / "ml" / "metrics.json"
AUDIT_DB = BASE / "data" / "audit.db"
API_URL = os.getenv("API_URL", "http://localhost:8000")

FEATURES = [
    "age", "bmi", "systolic_bp", "hba1c", "ldl",
    "length_of_stay", "prior_admissions", "comorbidity_count",
]

FEATURE_LABELS = {
    "age": "Age",
    "bmi": "BMI",
    "systolic_bp": "Systolic BP",
    "hba1c": "HbA1c",
    "ldl": "LDL",
    "length_of_stay": "Length of Stay",
    "prior_admissions": "Prior Admissions",
    "comorbidity_count": "Comorbidity Count",
}

st.set_page_config(
    page_title="PulseRisk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(66, 153, 225, .12), transparent 30%),
            radial-gradient(circle at 90% 5%, rgba(56, 178, 172, .10), transparent 25%),
            #07111f;
        color: #eef4ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081423 0%, #0d1b2a 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
    }

    .hero {
        padding: 1.5rem 1.6rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(22, 101, 170, .28), rgba(15, 118, 110, .20));
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 18px 50px rgba(0,0,0,.18);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: .25rem;
    }

    .hero-sub {
        color: #aac0d8;
        font-size: 1rem;
    }

    .metric-card {
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.085);
        padding: 1.05rem 1.1rem;
        border-radius: 18px;
        min-height: 118px;
    }

    .metric-label {
        color: #8fa7bf;
        font-size: .82rem;
        margin-bottom: .35rem;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
    }

    .metric-help {
        color: #6f89a3;
        font-size: .76rem;
        margin-top: .35rem;
    }

    .risk-high {
        color: #ff7b7b;
        font-weight: 800;
    }

    .risk-medium {
        color: #ffd166;
        font-weight: 800;
    }

    .risk-low {
        color: #72e1b3;
        font-weight: 800;
    }

    .patient-header {
        padding: 1rem 1.15rem;
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        margin: .4rem 0 1rem;
    }

    .insight-box {
        background: rgba(65,105,225,.08);
        border-left: 4px solid #5aa9ff;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        margin-top: .6rem;
    }

    .warning-box {
        background: rgba(255,193,7,.08);
        border-left: 4px solid #ffd166;
        padding: .9rem 1rem;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.07);
        padding: 14px;
        border-radius: 16px;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 42px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def risk_level(prob):
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"

def score_rows(df, model):
    scored = df.copy()
    probs = model.predict_proba(scored[FEATURES])[:, 1]
    scored["risk_score"] = probs
    scored["risk_level"] = pd.cut(
        probs,
        bins=[-0.01, 0.40, 0.70, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return scored

def log_action(action, details=""):
    con = sqlite3.connect(AUDIT_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TEXT DEFAULT CURRENT_TIMESTAMP,
            action TEXT,
            details TEXT
        )
    """)
    con.execute("INSERT INTO audit_log(action, details) VALUES (?,?)", (action, details))
    con.commit()
    con.close()

def read_audit():
    con = sqlite3.connect(AUDIT_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TEXT DEFAULT CURRENT_TIMESTAMP,
            action TEXT,
            details TEXT
        )
    """)
    out = pd.read_sql_query(
        "SELECT event_time, action, details FROM audit_log ORDER BY id DESC LIMIT 200", con
    )
    con.close()
    return out

def probability_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Predicted 30-Day Readmission Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.24},
            "steps": [
                {"range": [0, 40]},
                {"range": [40, 70]},
                {"range": [70, 100]},
            ],
            "threshold": {
                "line": {"width": 3},
                "thickness": 0.75,
                "value": prob*100,
            },
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaf2fb"),
    )
    return fig

def local_driver_table(row, model):
    med = df[FEATURES].median(numeric_only=True)
    base = pd.DataFrame([med])
    base_prob = model.predict_proba(base)[:,1][0]
    values = []

    for f in FEATURES:
        x = base.copy()
        x.loc[0, f] = row[f]
        p = model.predict_proba(x)[:,1][0]
        values.append({
            "Feature": FEATURE_LABELS[f],
            "Impact": p - base_prob,
            "Patient value": row[f],
        })
    return pd.DataFrame(values).sort_values("Impact", key=lambda s: s.abs(), ascending=False)

def summary_text(row, prob):
    risk = risk_level(prob)
    drivers = []
    if row["prior_admissions"] >= 2:
        drivers.append("multiple prior admissions")
    if row["comorbidity_count"] >= 3:
        drivers.append("higher comorbidity burden")
    if row["hba1c"] >= 6.5:
        drivers.append("elevated HbA1c")
    if row["systolic_bp"] >= 140:
        drivers.append("elevated systolic blood pressure")
    if row["bmi"] >= 30:
        drivers.append("elevated BMI")
    if row["length_of_stay"] >= 6:
        drivers.append("longer recent length of stay")

    if not drivers:
        driver_text = "no single dominant risk factor in the available variables"
    else:
        driver_text = ", ".join(drivers[:4])

    return (
        f"This synthetic patient is currently categorized as **{risk} risk** "
        f"with an estimated **{prob*100:.1f}%** 30-day readmission probability. "
        f"Notable factors include {driver_text}. "
        f"This output is a portfolio demonstration and is not clinical guidance."
    )

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Project data/model not found. Run `python ml/generate_data.py` "
        "and `python ml/train_model.py` first."
    )
    st.stop()

df = load_data()
model = load_model()
scored_df = score_rows(df, model)

with st.sidebar:
    st.markdown("## 🩺 PulseRisk AI")
    st.caption("Patient Risk & Insights Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Patient Explorer",
            "Risk Predictor",
            "Cohort Analytics",
            "Data Quality",
            "Audit Log",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("SYSTEM STATUS")
    try:
        api_status = requests.get(f"{API_URL}/health", timeout=1.0).json()
        st.success("FastAPI connected")
    except Exception:
        st.warning("Frontend running locally")
        st.caption("Start FastAPI on port 8000 for API mode.")

    st.caption("Demo data • Synthetic only")
    st.caption("Not for clinical use")

if page == "Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Healthcare Patient Risk & Insights System</div>
        <div class="hero-sub">
            Turn patient-level data into risk signals, cohort insights, and explainable ML outputs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    high_risk = int((scored_df["risk_level"] == "High").sum())
    avg_risk = float(scored_df["risk_score"].mean())
    readmission_rate = float(scored_df["readmitted_30d"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(scored_df):,}", "Synthetic cohort")
    c2.metric("High-risk patients", f"{high_risk:,}", f"{high_risk/len(scored_df):.1%} of cohort")
    c3.metric("Average risk", f"{avg_risk:.1%}", "Predicted 30-day risk")
    c4.metric("Observed readmission", f"{readmission_rate:.1%}", "Synthetic label rate")

    st.write("")
    left, right = st.columns([1.2, 1])

    with left:
        risk_counts = (
            scored_df["risk_level"]
            .astype(str)
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .reset_index()
        )
        risk_counts.columns = ["Risk level", "Patients"]
        fig = px.bar(
            risk_counts,
            x="Risk level",
            y="Patients",
            title="Risk distribution",
            text="Patients",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        age_bins = pd.cut(scored_df["age"], [17, 30, 45, 60, 75, 100])
        age_risk = scored_df.assign(age_group=age_bins).groupby(
            "age_group", observed=False
        )["risk_score"].mean().reset_index()
        age_risk["age_group"] = age_risk["age_group"].astype(str)
        fig = px.line(
            age_risk,
            x="age_group",
            y="risk_score",
            markers=True,
            title="Average predicted risk by age group",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top risk drivers")
    importances = model.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS[x] for x in FEATURES],
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Global model feature importance",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Portfolio value:</b> this landing page shows product thinking, ML inference,
    analytics, data visualization, and application design — not just a notebook model.
    </div>
    """, unsafe_allow_html=True)

elif page == "Patient Explorer":
    st.title("Patient Explorer")
    st.caption("Search, review, and explain a synthetic patient risk profile.")

    search = st.text_input("Search patient ID", placeholder="Example: PT-10042")
    options = scored_df["patient_id"].tolist()
    default_idx = 0
    if search and search in options:
        default_idx = options.index(search)

    selected = st.selectbox("Select patient", options, index=default_idx)
    row = scored_df[scored_df["patient_id"] == selected].iloc[0]
    log_action("PATIENT_VIEWED", selected)

    st.markdown(
        f"""
        <div class="patient-header">
        <b>{selected}</b> &nbsp; • &nbsp; {int(row['age'])} years &nbsp; • &nbsp;
        {row['gender']} &nbsp; • &nbsp;
        Last admission: {row['last_admission_date']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    prob = float(row["risk_score"])
    c1, c2 = st.columns([1, 1.35])

    with c1:
        st.plotly_chart(probability_gauge(prob), use_container_width=True)

    with c2:
        st.subheader("Patient summary")
        st.markdown(summary_text(row, prob))

        level = risk_level(prob)
        css = {
            "High": "risk-high",
            "Medium": "risk-medium",
            "Low": "risk-low",
        }[level]
        st.markdown(
            f'Risk category: <span class="{css}">{level}</span>',
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Prior admissions", int(row["prior_admissions"]))
        m2.metric("Comorbidities", int(row["comorbidity_count"]))
        m3.metric("LOS", f"{row['length_of_stay']:.1f} days")

    st.subheader("Clinical snapshot")
    a,b,c,d = st.columns(4)
    a.metric("BMI", f"{row['bmi']:.1f}")
    b.metric("Systolic BP", f"{int(row['systolic_bp'])} mmHg")
    c.metric("HbA1c", f"{row['hba1c']:.1f}%")
    d.metric("LDL", f"{int(row['ldl'])} mg/dL")

    st.subheader("Why the model assigned this risk")
    drivers = local_driver_table(row, model).head(6)
    fig = px.bar(
        drivers.sort_values("Impact"),
        x="Impact",
        y="Feature",
        orientation="h",
        hover_data=["Patient value"],
        title="Approximate patient-level contribution vs cohort median",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=390,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Patient timeline")
    admit_date = pd.to_datetime(row["last_admission_date"])
    timeline = pd.DataFrame({
        "Date": [
            admit_date - timedelta(days=120),
            admit_date - timedelta(days=65),
            admit_date - timedelta(days=10),
            admit_date,
        ],
        "Event": [
            "Primary care visit",
            "Lab panel",
            "Emergency visit",
            "Most recent admission",
        ],
        "Details": [
            "Routine follow-up",
            "Metabolic and lipid labs",
            "Acute symptom evaluation",
            f"Length of stay: {row['length_of_stay']:.1f} days",
        ],
    })
    st.dataframe(timeline, use_container_width=True, hide_index=True)

elif page == "Risk Predictor":
    st.title("New Patient Risk Predictor")
    st.caption("Enter patient features and score risk through the trained ML pipeline.")

    with st.form("predict_form"):
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 18, 100, 58)
        bmi = c2.number_input("BMI", 10.0, 70.0, 29.0, 0.1)
        systolic_bp = c3.number_input("Systolic BP", 70, 260, 138)
        hba1c = c4.number_input("HbA1c", 3.0, 20.0, 6.4, 0.1)

        c5, c6, c7, c8 = st.columns(4)
        ldl = c5.number_input("LDL", 20, 350, 130)
        los = c6.number_input("Length of stay", 0.0, 90.0, 4.0, 0.5)
        prior = c7.number_input("Prior admissions", 0, 30, 1)
        comorb = c8.number_input("Comorbidity count", 0, 20, 2)

        submitted = st.form_submit_button("Calculate Risk", use_container_width=True)

    if submitted:
        payload = {
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "hba1c": hba1c,
            "ldl": ldl,
            "length_of_stay": los,
            "prior_admissions": prior,
            "comorbidity_count": comorb,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
            response.raise_for_status()
            result = response.json()
            prob = result["risk_probability"]
            drivers = pd.DataFrame(result["top_risk_drivers"])
            api_used = True
        except Exception:
            x = pd.DataFrame([payload])
            prob = float(model.predict_proba(x)[:,1][0])
            dummy = scored_df.iloc[0].copy()
            for k, v in payload.items():
                dummy[k] = v
            drivers = local_driver_table(dummy, model).head(5).rename(
                columns={"Feature":"feature","Impact":"impact"}
            )
            api_used = False

        log_action("RISK_SCORED", f"{prob*100:.1f}%")

        left, right = st.columns([1, 1.2])
        with left:
            st.plotly_chart(probability_gauge(prob), use_container_width=True)
        with right:
            st.subheader("Prediction result")
            st.metric("Risk probability", f"{prob:.1%}")
            st.metric("Risk level", risk_level(prob))
            st.caption("Scored through FastAPI" if api_used else "Scored locally")

            if "feature" in drivers.columns:
                st.write("Top model drivers")
                show = drivers[["feature","impact"]].copy()
                show["impact"] = show["impact"].map(lambda x: f"{x:+.3f}")
                st.dataframe(show, use_container_width=True, hide_index=True)

elif page == "Cohort Analytics":
    st.title("Cohort Analytics")
    st.caption("Slice the population and identify high-risk patient segments.")

    with st.expander("Cohort filters", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        age_range = c1.slider("Age", 18, 90, (18, 90))
        genders = c2.multiselect("Gender", sorted(scored_df["gender"].unique()), default=sorted(scored_df["gender"].unique()))
        risks = c3.multiselect("Risk level", ["Low","Medium","High"], default=["Low","Medium","High"])
        readmitted = c4.selectbox("Readmitted", ["All","Yes","No"])

    filt = scored_df[
        scored_df["age"].between(*age_range)
        & scored_df["gender"].isin(genders)
        & scored_df["risk_level"].astype(str).isin(risks)
    ].copy()

    if readmitted == "Yes":
        filt = filt[filt["readmitted_30d"] == 1]
    elif readmitted == "No":
        filt = filt[filt["readmitted_30d"] == 0]

    log_action("COHORT_FILTER_APPLIED", f"n={len(filt)}")

    c1,c2,c3 = st.columns(3)
    c1.metric("Patients in cohort", f"{len(filt):,}")
    c2.metric("Average predicted risk", f"{filt['risk_score'].mean():.1%}" if len(filt) else "—")
    c3.metric("Observed readmission", f"{filt['readmitted_30d'].mean():.1%}" if len(filt) else "—")

    left,right = st.columns(2)
    with left:
        fig = px.scatter(
            filt,
            x="age",
            y="risk_score",
            size="comorbidity_count",
            hover_name="patient_id",
            title="Age vs predicted risk",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        tmp = filt.copy()
        tmp["Comorbidity group"] = pd.cut(
            tmp["comorbidity_count"],
            [-1,0,2,4,20],
            labels=["0","1–2","3–4","5+"],
        )
        grp = tmp.groupby("Comorbidity group", observed=False)["risk_score"].mean().reset_index()
        fig = px.bar(grp, x="Comorbidity group", y="risk_score", title="Risk by comorbidity burden")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-risk patients in filtered cohort")
    show_cols = [
        "patient_id", "age", "gender", "risk_score", "prior_admissions",
        "comorbidity_count", "hba1c", "systolic_bp"
    ]
    table = filt.sort_values("risk_score", ascending=False)[show_cols].head(25).copy()
    table["risk_score"] = table["risk_score"].map(lambda x: f"{x:.1%}")
    st.dataframe(table, use_container_width=True, hide_index=True)

elif page == "Data Quality":
    st.title("Data Quality Center")
    st.caption("A production-minded project should show what happens before model inference.")

    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated(subset=["patient_id"]).sum())

    out_of_range = {
        "BMI outside 10–70": int(((df["bmi"] < 10) | (df["bmi"] > 70)).sum()),
        "Systolic BP outside 70–260": int(((df["systolic_bp"] < 70) | (df["systolic_bp"] > 260)).sum()),
        "HbA1c outside 3–20": int(((df["hba1c"] < 3) | (df["hba1c"] > 20)).sum()),
    }

    c1,c2,c3 = st.columns(3)
    c1.metric("Missing cells", total_missing)
    c2.metric("Duplicate patient IDs", duplicate_rows)
    c3.metric("Rows", f"{len(df):,}")

    missing = df.isna().sum().reset_index()
    missing.columns = ["Column","Missing values"]
    missing = missing[missing["Missing values"] > 0]

    left,right = st.columns(2)
    with left:
        st.subheader("Missingness")
        if len(missing):
            st.dataframe(missing, use_container_width=True, hide_index=True)
        else:
            st.success("No missing values detected.")

    with right:
        st.subheader("Range checks")
        range_df = pd.DataFrame(
            [{"Check":k, "Violations":v} for k,v in out_of_range.items()]
        )
        st.dataframe(range_df, use_container_width=True, hide_index=True)

    completeness = 1 - total_missing/(df.shape[0]*df.shape[1])
    st.progress(completeness)
    st.caption(f"Dataset completeness: {completeness:.2%}")

elif page == "Audit Log":
    st.title("Audit Log")
    st.caption("Tracks major user actions for traceability.")
    audit = read_audit()
    st.dataframe(audit, use_container_width=True, hide_index=True)

    if st.button("Clear audit log"):
        con = sqlite3.connect(AUDIT_DB)
        con.execute("DELETE FROM audit_log")
        con.commit()
        con.close()
        st.rerun()

elif page == "About":
    st.title("About the Project")
    st.markdown("""
### What problem does it solve?

Healthcare teams often have patient data spread across tables, reports, and isolated systems.
This application demonstrates how a unified analytics layer can:

- identify higher-risk patients,
- explain model predictions,
- analyze population-level patterns,
- inspect patient history,
- monitor data quality,
- and expose ML predictions through an API.

### Why it is portfolio-worthy

The strongest part of this project is **system integration**. It combines:

1. data generation and preprocessing,
2. machine-learning training,
3. backend APIs,
4. frontend product design,
5. explainability,
6. monitoring,
7. testing,
8. containerization.

### Clinical safety note

This application is intentionally built with synthetic data and is **not intended for diagnosis,
treatment, triage, or any real clinical decision-making**.
""")
