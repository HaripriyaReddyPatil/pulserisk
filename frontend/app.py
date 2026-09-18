import json
import os
import sqlite3

from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# Paths / configuration
# =========================================================

BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data" / "patients.csv"
MODEL_PATH = BASE / "ml" / "risk_model.joblib"

METRICS_PATH = BASE / "ml" / "metrics.json"
METADATA_PATH = BASE / "ml" / "model_metadata.json"
COMPARISON_PATH = BASE / "ml" / "model_comparison.json"

EVALUATION_DIR = BASE / "ml" / "evaluation"
EVALUATION_PATH = EVALUATION_DIR / "evaluation_summary.json"

AUDIT_DB = BASE / "data" / "audit.db"

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
)


# =========================================================
# Features
# =========================================================

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
    "length_of_stay": "Length of Stay",
    "prior_admissions": "Prior Admissions",
    "comorbidity_count": "Comorbidity Count",
}


# =========================================================
# Streamlit setup
# =========================================================

st.set_page_config(
    page_title="PulseRisk",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(66, 153, 225, .12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 5%,
            rgba(56, 178, 172, .10),
            transparent 25%
        ),
        #07111f;

    color: #eef4ff;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #081423 0%,
            #0d1b2a 100%
        );

    border-right:
        1px solid
        rgba(255,255,255,.08);
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

    background:
        linear-gradient(
            135deg,
            rgba(22,101,170,.28),
            rgba(15,118,110,.20)
        );

    border:
        1px solid
        rgba(255,255,255,.10);

    box-shadow:
        0 18px 50px
        rgba(0,0,0,.18);

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

    background:
        rgba(255,255,255,.045);

    border:
        1px solid
        rgba(255,255,255,.08);

    border-radius: 18px;

    margin:
        .4rem 0 1rem;
}

div[data-testid="stMetric"] {
    background:
        rgba(255,255,255,.045);

    border:
        1px solid
        rgba(255,255,255,.07);

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

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# =========================================================
# Loaders
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_PATH
    )


@st.cache_resource
def load_model():
    return joblib.load(
        MODEL_PATH
    )


@st.cache_data
def load_json_file(path_string):
    path = Path(path_string)

    if path.exists():
        return json.loads(
            path.read_text()
        )

    return {}


# =========================================================
# Helpers
# =========================================================

def risk_level(probability):
    if probability >= 0.70:
        return "High"

    if probability >= 0.40:
        return "Medium"

    return "Low"


def score_rows(
    dataframe,
    model,
):
    scored = dataframe.copy()

    probabilities = model.predict_proba(
        scored[FEATURES]
    )[:, 1]

    scored["risk_score"] = probabilities

    scored["risk_level"] = pd.cut(
        probabilities,
        bins=[
            -0.01,
            0.40,
            0.70,
            1.01,
        ],
        labels=[
            "Low",
            "Medium",
            "High",
        ],
    )

    return scored


def model_display_name(
    model_name,
):
    mapping = {
        "logistic_regression":
            "Logistic Regression",

        "LogisticRegression":
            "Logistic Regression",

        "random_forest":
            "Random Forest",

        "RandomForestClassifier":
            "Random Forest",

        "gradient_boosting":
            "Gradient Boosting",

        "GradientBoostingClassifier":
            "Gradient Boosting",
    }

    return mapping.get(
        model_name,
        model_name,
    )


def safe_float(
    value,
    digits=1,
):
    if pd.isna(value):
        return "—"

    return f"{float(value):.{digits}f}"


def safe_int(value):
    if pd.isna(value):
        return "—"

    return str(
        int(value)
    )


# =========================================================
# Audit logging
# =========================================================

def log_action(
    action,
    details="",
):
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


def read_audit():
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

    result = pd.read_sql_query(
        """
        SELECT
            event_time,
            action,
            details
        FROM audit_log
        ORDER BY id DESC
        LIMIT 200
        """,
        con,
    )

    con.close()

    return result


# =========================================================
# Explainability
# =========================================================

def get_global_importance(
    model,
):
    estimator = model.named_steps[
        "model"
    ]

    if hasattr(
        estimator,
        "coef_",
    ):
        coefficients = (
            estimator.coef_[0]
        )

        result = pd.DataFrame(
            {
                "Feature": [
                    FEATURE_LABELS[x]
                    for x in FEATURES
                ],
                "Coefficient":
                    coefficients,
                "Importance":
                    np.abs(
                        coefficients
                    ),
            }
        )

        result[
            "Direction"
        ] = np.where(
            result[
                "Coefficient"
            ] >= 0,
            "Increases risk",
            "Reduces risk",
        )

        return result.sort_values(
            "Importance",
            ascending=False,
        )

    if hasattr(
        estimator,
        "feature_importances_",
    ):
        result = pd.DataFrame(
            {
                "Feature": [
                    FEATURE_LABELS[x]
                    for x in FEATURES
                ],
                "Importance":
                    estimator.feature_importances_,
                "Coefficient":
                    np.nan,
                "Direction":
                    "Model importance",
            }
        )

        return result.sort_values(
            "Importance",
            ascending=False,
        )

    return pd.DataFrame()


def local_driver_table(
    row,
    model,
):
    estimator = model.named_steps[
        "model"
    ]

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    row_df = pd.DataFrame(
        [
            {
                feature:
                    row[feature]
                for feature in FEATURES
            }
        ]
    )

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

        records = []

        for (
            feature,
            contribution,
        ) in zip(
            FEATURES,
            contributions,
        ):
            records.append(
                {
                    "Feature":
                        FEATURE_LABELS[
                            feature
                        ],

                    "Impact":
                        float(
                            contribution
                        ),

                    "Patient value":
                        row[
                            feature
                        ],

                    "Direction": (
                        "Increases risk"
                        if contribution >= 0
                        else "Reduces risk"
                    ),
                }
            )

        return (
            pd.DataFrame(
                records
            )
            .sort_values(
                "Impact",
                key=lambda s:
                    s.abs(),
                ascending=False,
            )
        )

    medians = (
        df[
            FEATURES
        ]
        .median(
            numeric_only=True
        )
    )

    base = pd.DataFrame(
        [medians]
    )

    base_probability = (
        model.predict_proba(
            base
        )[:, 1][0]
    )

    records = []

    for feature in FEATURES:
        modified = base.copy()

        modified.loc[
            0,
            feature,
        ] = row[
            feature
        ]

        probability = (
            model.predict_proba(
                modified
            )[:, 1][0]
        )

        impact = (
            probability
            - base_probability
        )

        records.append(
            {
                "Feature":
                    FEATURE_LABELS[
                        feature
                    ],

                "Impact":
                    float(
                        impact
                    ),

                "Patient value":
                    row[
                        feature
                    ],

                "Direction": (
                    "Increases risk"
                    if impact >= 0
                    else "Reduces risk"
                ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "Impact",
            key=lambda s:
                s.abs(),
            ascending=False,
        )
    )


# =========================================================
# Visualizations
# =========================================================

def probability_gauge(
    probability,
):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=(
                probability
                * 100
            ),

            number={
                "suffix": "%",
                "font": {
                    "size": 40
                },
            },

            title={
                "text":
                    "Predicted 30-Day "
                    "Readmission Risk"
            },

            gauge={
                "axis": {
                    "range": [
                        0,
                        100,
                    ]
                },

                "bar": {
                    "thickness":
                        0.24
                },

                "steps": [
                    {
                        "range":
                            [0, 40]
                    },

                    {
                        "range":
                            [40, 70]
                    },

                    {
                        "range":
                            [70, 100]
                    },
                ],

                "threshold": {
                    "line": {
                        "width": 3
                    },

                    "thickness":
                        0.75,

                    "value":
                        probability
                        * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=320,

        margin=dict(
            l=35,
            r=35,
            t=70,
            b=25,
        ),

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        font=dict(
            color="#eaf2fb"
        ),
    )

    return fig


def summary_text(
    row,
    probability,
):
    risk = risk_level(
        probability
    )

    drivers = []

    if (
        row[
            "prior_admissions"
        ]
        >= 2
    ):
        drivers.append(
            "multiple prior admissions"
        )

    if (
        row[
            "comorbidity_count"
        ]
        >= 3
    ):
        drivers.append(
            "higher comorbidity burden"
        )

    if (
        not pd.isna(
            row["hba1c"]
        )
        and
        row[
            "hba1c"
        ] >= 6.5
    ):
        drivers.append(
            "elevated HbA1c"
        )

    if (
        row[
            "systolic_bp"
        ]
        >= 140
    ):
        drivers.append(
            "elevated systolic "
            "blood pressure"
        )

    if (
        not pd.isna(
            row["bmi"]
        )
        and
        row[
            "bmi"
        ] >= 30
    ):
        drivers.append(
            "elevated BMI"
        )

    if (
        row[
            "length_of_stay"
        ]
        >= 6
    ):
        drivers.append(
            "longer recent "
            "length of stay"
        )

    if not drivers:
        driver_text = (
            "no single dominant "
            "risk factor in the "
            "available variables"
        )

    else:
        driver_text = ", ".join(
            drivers[:4]
        )

    return (
        f"This synthetic patient is "
        f"currently categorized as "
        f"**{risk} risk** with an "
        f"estimated "
        f"**{probability * 100:.1f}%** "
        f"30-day readmission probability. "
        f"Notable factors include "
        f"{driver_text}. "
        f"This estimate is intended "
        f"for demonstration only."
    )


# =========================================================
# Startup checks
# =========================================================

if (
    not DATA_PATH.exists()
    or
    not MODEL_PATH.exists()
):
    st.error(
        "Project data/model not found. "
        "Run the ML training pipeline first."
    )

    st.stop()


df = load_data()

model = load_model()

scored_df = score_rows(
    df,
    model,
)

metadata = load_json_file(
    str(
        METADATA_PATH
    )
)

comparison = load_json_file(
    str(
        COMPARISON_PATH
    )
)

evaluation_summary = (
    load_json_file(
        str(
            EVALUATION_PATH
        )
    )
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown(
        "## 🩺 PulseRisk"
    )

    st.caption(
        "Patient Risk & "
        "Insights Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",

        [
            "Overview",
            "Patient Explorer",
            "Risk Predictor",
            "Model Performance",
            "Cohort Analytics",
            "Data Quality",
            "Audit Log",
            "About",
        ],

        label_visibility=(
            "collapsed"
        ),
    )

    st.divider()

    st.caption(
        "SYSTEM STATUS"
    )

    try:
        api_status = (
            requests.get(
                f"{API_URL}/health",
                timeout=1.0,
            )
            .json()
        )

        if api_status.get(
            "model_loaded"
        ):
            st.success(
                "FastAPI connected"
            )

        else:
            st.warning(
                "API connected — "
                "model unavailable"
            )

    except Exception:
        st.warning(
            "Frontend running locally"
        )

        st.caption(
            "Start FastAPI on "
            "port 8000 for API mode."
        )

    selected_model = (
        metadata.get(
            "selected_model",
            "Unknown",
        )
    )

    st.caption(
        "MODEL"
    )

    st.write(
        model_display_name(
            selected_model
        )
    )

    st.caption(
        "Synthetic data only"
    )

    st.caption(
        "Not for clinical use"
    )


# =========================================================
# Overview
# =========================================================

if page == "Overview":

    st.markdown(
        """
<div class="hero">
    <div class="hero-title">PulseRisk</div>
    <div class="hero-sub">
        Explainable machine-learning system for synthetic 30-day patient readmission risk analysis.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    high_risk = int(
        (
            scored_df[
                "risk_level"
            ]
            == "High"
        ).sum()
    )

    average_risk = float(
        scored_df[
            "risk_score"
        ].mean()
    )

    readmission_rate = float(
        scored_df[
            "readmitted_30d"
        ].mean()
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Patients",
        f"{len(scored_df):,}",
        "Synthetic cohort",
    )

    c2.metric(
        "High-risk patients",
        f"{high_risk:,}",
        (
            f"{high_risk / len(scored_df):.1%} "
            f"of cohort"
        ),
    )

    c3.metric(
        "Average predicted risk",
        f"{average_risk:.1%}",
    )

    c4.metric(
        "Observed readmission",
        f"{readmission_rate:.1%}",
        "Synthetic outcome rate",
    )

    st.write("")

    left, right = st.columns(
        [
            1.2,
            1,
        ]
    )

    with left:
        risk_counts = (
            scored_df[
                "risk_level"
            ]
            .astype(str)
            .value_counts()
            .reindex(
                [
                    "Low",
                    "Medium",
                    "High",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        risk_counts.columns = [
            "Risk level",
            "Patients",
        ]

        fig = px.bar(
            risk_counts,
            x="Risk level",
            y="Patients",
            text="Patients",
            title=(
                "Predicted risk "
                "distribution"
            ),
        )

        fig.update_layout(
            template="plotly_dark",

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            height=360,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        age_bins = pd.cut(
            scored_df["age"],
            [
                17,
                30,
                45,
                60,
                75,
                100,
            ],
        )

        age_risk = (
            scored_df
            .assign(
                age_group=age_bins
            )
            .groupby(
                "age_group",
                observed=False,
            )[
                "risk_score"
            ]
            .mean()
            .reset_index()
        )

        age_risk[
            "age_group"
        ] = (
            age_risk[
                "age_group"
            ].astype(str)
        )

        fig = px.line(
            age_risk,
            x="age_group",
            y="risk_score",
            markers=True,
            title=(
                "Average predicted "
                "risk by age group"
            ),
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        fig.update_layout(
            template="plotly_dark",

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            height=360,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        "Global model drivers"
    )

    importance_df = (
        get_global_importance(
            model
        )
    )

    if not importance_df.empty:
        plot_df = (
            importance_df
            .sort_values(
                "Importance"
            )
        )

        fig = px.bar(
            plot_df,
            x="Importance",
            y="Feature",
            orientation="h",

            hover_data=[
                "Direction"
            ],

            title=(
                "Standardized "
                "feature influence"
            ),
        )

        fig.update_layout(
            template="plotly_dark",

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            height=420,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with st.expander(
        "How to interpret this chart"
    ):
        st.write(
            "For the current Logistic "
            "Regression model, feature "
            "importance is represented "
            "by the absolute value of "
            "the standardized coefficient. "
            "Direction indicates whether "
            "a feature pushes the model "
            "toward higher or lower "
            "predicted risk."
        )


# =========================================================
# Patient Explorer
# =========================================================

elif page == "Patient Explorer":

    st.title(
        "Patient Explorer"
    )

    st.caption(
        "Review a synthetic patient "
        "profile and inspect the model's "
        "risk explanation."
    )

    search = st.text_input(
        "Search patient ID",
        placeholder=(
            "Example: PT-10042"
        ),
    )

    options = (
        scored_df[
            "patient_id"
        ].tolist()
    )

    default_index = 0

    if (
        search
        and search in options
    ):
        default_index = (
            options.index(
                search
            )
        )

    selected = st.selectbox(
        "Select patient",
        options,
        index=default_index,
    )

    row = (
        scored_df[
            scored_df[
                "patient_id"
            ]
            == selected
        ]
        .iloc[0]
    )

    log_action(
        "PATIENT_VIEWED",
        selected,
    )

    st.markdown(
        f"""
<div class="patient-header">
    <b>{selected}</b>
    &nbsp; • &nbsp;
    {int(row['age'])} years
    &nbsp; • &nbsp;
    {row['gender']}
    &nbsp; • &nbsp;
    Last admission: {row['last_admission_date']}
</div>
""",
        unsafe_allow_html=True,
    )

    probability = float(
        row[
            "risk_score"
        ]
    )

    c1, c2 = (
        st.columns(
            [
                1,
                1.35,
            ]
        )
    )

    with c1:
        st.plotly_chart(
            probability_gauge(
                probability
            ),
            use_container_width=True,
        )

    with c2:
        st.subheader(
            "Patient summary"
        )

        st.markdown(
            summary_text(
                row,
                probability,
            )
        )

        level = risk_level(
            probability
        )

        css = {
            "High":
                "risk-high",

            "Medium":
                "risk-medium",

            "Low":
                "risk-low",
        }[
            level
        ]

        st.markdown(
            (
                f'Risk category: '
                f'<span class="{css}">'
                f'{level}'
                f'</span>'
            ),
            unsafe_allow_html=True,
        )

        m1, m2, m3 = (
            st.columns(3)
        )

        m1.metric(
            "Prior admissions",
            int(
                row[
                    "prior_admissions"
                ]
            ),
        )

        m2.metric(
            "Comorbidities",
            int(
                row[
                    "comorbidity_count"
                ]
            ),
        )

        m3.metric(
            "Length of stay",
            (
                f"{row['length_of_stay']:.1f} "
                f"days"
            ),
        )

    st.subheader(
        "Clinical snapshot"
    )

    a, b, c, d = (
        st.columns(4)
    )

    a.metric(
        "BMI",
        safe_float(
            row["bmi"]
        ),
    )

    b.metric(
        "Systolic BP",
        (
            f"{safe_int(row['systolic_bp'])} "
            f"mmHg"
        ),
    )

    c.metric(
        "HbA1c",
        (
            f"{safe_float(row['hba1c'])}%"
        ),
    )

    d.metric(
        "LDL",
        (
            f"{safe_int(row['ldl'])} "
            f"mg/dL"
        ),
    )

    st.subheader(
        "Why the model assigned this risk"
    )

    drivers = (
        local_driver_table(
            row,
            model,
        )
        .head(6)
    )

    chart_df = (
        drivers
        .sort_values(
            "Impact"
        )
    )

    fig = px.bar(
        chart_df,
        x="Impact",
        y="Feature",
        orientation="h",
        color="Direction",

        hover_data=[
            "Patient value",
            "Direction",
        ],

        title=(
            "Patient-level "
            "model contributions"
        ),
    )

    fig.update_layout(
        template="plotly_dark",

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        height=390,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Positive contributions push "
        "the prediction toward higher "
        "risk; negative contributions "
        "push it toward lower risk."
    )

    st.subheader(
        "Patient timeline"
    )

    admission_date = (
        pd.to_datetime(
            row[
                "last_admission_date"
            ]
        )
    )

    timeline = pd.DataFrame(
        {
            "Date": [
                admission_date
                - timedelta(
                    days=120
                ),

                admission_date
                - timedelta(
                    days=65
                ),

                admission_date
                - timedelta(
                    days=10
                ),

                admission_date,
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

                (
                    f"Length of stay: "
                    f"{row['length_of_stay']:.1f} "
                    f"days"
                ),
            ],
        }
    )

    timeline[
        "Date"
    ] = pd.to_datetime(
        timeline[
            "Date"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    st.dataframe(
        timeline,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# Risk Predictor
# =========================================================

elif page == "Risk Predictor":

    st.title(
        "New Patient Risk Predictor"
    )

    st.caption(
        "Enter patient features and "
        "score 30-day readmission risk "
        "using the selected production "
        "model."
    )

    with st.form(
        "predict_form"
    ):
        c1, c2, c3, c4 = (
            st.columns(4)
        )

        age = c1.number_input(
            "Age",
            18,
            100,
            58,
        )

        bmi = c2.number_input(
            "BMI",
            10.0,
            70.0,
            29.0,
            0.1,
        )

        systolic_bp = (
            c3.number_input(
                "Systolic BP",
                70,
                260,
                138,
            )
        )

        hba1c = c4.number_input(
            "HbA1c",
            3.0,
            20.0,
            6.4,
            0.1,
        )

        c5, c6, c7, c8 = (
            st.columns(4)
        )

        ldl = c5.number_input(
            "LDL",
            20,
            350,
            130,
        )

        length_of_stay = (
            c6.number_input(
                "Length of stay",
                0.0,
                90.0,
                4.0,
                0.5,
            )
        )

        prior_admissions = (
            c7.number_input(
                "Prior admissions",
                0,
                30,
                1,
            )
        )

        comorbidity_count = (
            c8.number_input(
                "Comorbidity count",
                0,
                20,
                2,
            )
        )

        submitted = (
            st.form_submit_button(
                "Calculate Risk",
                use_container_width=True,
            )
        )

    if submitted:
        payload = {
            "age":
                age,

            "bmi":
                bmi,

            "systolic_bp":
                systolic_bp,

            "hba1c":
                hba1c,

            "ldl":
                ldl,

            "length_of_stay":
                length_of_stay,

            "prior_admissions":
                prior_admissions,

            "comorbidity_count":
                comorbidity_count,
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=3,
            )

            response.raise_for_status()

            result = (
                response.json()
            )

            probability = (
                result[
                    "risk_probability"
                ]
            )

            model_used = (
                result.get(
                    "model_type",
                    "Unknown",
                )
            )

            drivers = pd.DataFrame(
                result[
                    "top_risk_drivers"
                ]
            )

            api_used = True

        except Exception:
            patient_df = (
                pd.DataFrame(
                    [payload]
                )
            )

            probability = float(
                model.predict_proba(
                    patient_df
                )[:, 1][0]
            )

            dummy = (
                scored_df.iloc[
                    0
                ].copy()
            )

            for (
                key,
                value,
            ) in payload.items():

                dummy[
                    key
                ] = value

            local = (
                local_driver_table(
                    dummy,
                    model,
                )
                .head(5)
            )

            drivers = (
                local.rename(
                    columns={
                        "Feature":
                            "feature",

                        "Impact":
                            "impact",

                        "Direction":
                            "direction",
                    }
                )
            )

            model_used = (
                type(
                    model.named_steps[
                        "model"
                    ]
                ).__name__
            )

            api_used = False

        log_action(
            "RISK_SCORED",
            (
                f"{probability * 100:.1f}%"
            ),
        )

        left, right = (
            st.columns(
                [
                    1,
                    1.2,
                ]
            )
        )

        with left:
            st.plotly_chart(
                probability_gauge(
                    probability
                ),
                use_container_width=True,
            )

        with right:
            st.subheader(
                "Prediction result"
            )

            st.metric(
                "Risk probability",
                f"{probability:.1%}",
            )

            st.metric(
                "Risk level",
                risk_level(
                    probability
                ),
            )

            st.metric(
                "Model",
                model_display_name(
                    model_used
                ),
            )

            st.caption(
                (
                    "Scored through FastAPI"
                    if api_used
                    else "Scored locally"
                )
            )

        if (
            not drivers.empty
            and
            "feature"
            in drivers.columns
        ):
            st.subheader(
                "Top model drivers"
            )

            show = (
                drivers[
                    [
                        "feature",
                        "impact",
                    ]
                ]
                .copy()
            )

            show[
                "impact"
            ] = (
                show[
                    "impact"
                ]
                .map(
                    lambda x:
                        f"{x:+.3f}"
                )
            )

            st.dataframe(
                show,
                use_container_width=True,
                hide_index=True,
            )


# =========================================================
# Model Performance
# =========================================================

elif page == "Model Performance":

    st.title(
        "Model Performance"
    )

    st.caption(
        "Model selection, validation "
        "metrics, decision-threshold "
        "analysis, and evaluation "
        "diagnostics."
    )

    selected_model = (
        metadata.get(
            "selected_model",
            "Unknown",
        )
    )

    selected_label = (
        model_display_name(
            selected_model
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Production model",
        selected_label,
    )

    c2.metric(
        "ROC-AUC",
        (
            f"{evaluation_summary.get('roc_auc', 0):.3f}"
        ),
    )

    c3.metric(
        "Recall",
        (
            f"{evaluation_summary.get('recall_at_recommended_threshold', 0):.3f}"
        ),
    )

    c4.metric(
        "F1",
        (
            f"{evaluation_summary.get('f1_at_recommended_threshold', 0):.3f}"
        ),
    )

    c5, c6, c7 = (
        st.columns(3)
    )

    c5.metric(
        "Precision",
        (
            f"{evaluation_summary.get('precision_at_recommended_threshold', 0):.3f}"
        ),
    )

    c6.metric(
        "Recommended threshold",
        (
            f"{evaluation_summary.get('recommended_threshold', 0.5):.2f}"
        ),
    )

    c7.metric(
        "Test records",
        evaluation_summary.get(
            "test_records",
            metadata.get(
                "testing_records",
                "—",
            ),
        ),
    )

    st.divider()

    st.subheader(
        "Candidate model comparison"
    )

    model_results = (
        comparison.get(
            "models",
            {},
        )
    )

    comparison_rows = []

    for (
        model_name,
        metrics,
    ) in model_results.items():

        comparison_rows.append(
            {
                "Model":
                    model_display_name(
                        model_name
                    ),

                "ROC-AUC":
                    metrics.get(
                        "roc_auc"
                    ),

                "Accuracy":
                    metrics.get(
                        "accuracy"
                    ),

                "Precision":
                    metrics.get(
                        "precision"
                    ),

                "Recall":
                    metrics.get(
                        "recall"
                    ),

                "F1":
                    metrics.get(
                        "f1"
                    ),
            }
        )

    if comparison_rows:
        comparison_df = (
            pd.DataFrame(
                comparison_rows
            )
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            comparison_df,
            x="Model",
            y="ROC-AUC",
            text="ROC-AUC",
            title=(
                "ROC-AUC across "
                "candidate models"
            ),
        )

        fig.update_layout(
            template="plotly_dark",

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            height=380,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.subheader(
        "Evaluation diagnostics"
    )

    left, right = (
        st.columns(2)
    )

    roc_path = (
        EVALUATION_DIR
        / "roc_curve.png"
    )

    confusion_path = (
        EVALUATION_DIR
        / "confusion_matrix.png"
    )

    precision_recall_path = (
        EVALUATION_DIR
        / "precision_recall_curve.png"
    )

    threshold_path = (
        EVALUATION_DIR
        / "threshold_tradeoff.png"
    )

    with left:
        if roc_path.exists():
            st.image(
                str(
                    roc_path
                ),
                caption=(
                    "ROC Curve"
                ),
                use_container_width=True,
            )

        if (
            precision_recall_path.exists()
        ):
            st.image(
                str(
                    precision_recall_path
                ),
                caption=(
                    "Precision–Recall Curve"
                ),
                use_container_width=True,
            )

    with right:
        if confusion_path.exists():
            st.image(
                str(
                    confusion_path
                ),
                caption=(
                    "Confusion Matrix"
                ),
                use_container_width=True,
            )

        if threshold_path.exists():
            st.image(
                str(
                    threshold_path
                ),
                caption=(
                    "Decision Threshold "
                    "Trade-off"
                ),
                use_container_width=True,
            )

    st.subheader(
        "Model-selection methodology"
    )

    st.markdown(
        """
PulseRisk trains several candidate classifiers
using the same stratified train/test split.

**Candidate models**

- Logistic Regression
- Random Forest
- Gradient Boosting

The production model is selected using ROC-AUC
while accuracy, precision, recall, and F1 are
also retained for evaluation.

Decision-threshold analysis is performed
separately because different thresholds create
different trade-offs between false positives
and false negatives.
        """
    )

    st.info(
        "All performance measurements are "
        "derived from synthetic demonstration "
        "data and should not be interpreted "
        "as clinical validation."
    )


# =========================================================
# Cohort Analytics
# =========================================================

elif page == "Cohort Analytics":

    st.title(
        "Cohort Analytics"
    )

    st.caption(
        "Filter the synthetic population "
        "and investigate patient risk "
        "patterns."
    )

    with st.expander(
        "Cohort filters",
        expanded=True,
    ):
        c1, c2, c3, c4 = (
            st.columns(4)
        )

        age_range = c1.slider(
            "Age",
            18,
            90,
            (
                18,
                90,
            ),
        )

        gender_values = sorted(
            scored_df[
                "gender"
            ]
            .dropna()
            .unique()
        )

        genders = c2.multiselect(
            "Gender",
            gender_values,
            default=gender_values,
        )

        risks = c3.multiselect(
            "Risk level",
            [
                "Low",
                "Medium",
                "High",
            ],
            default=[
                "Low",
                "Medium",
                "High",
            ],
        )

        readmitted = c4.selectbox(
            "Readmitted",
            [
                "All",
                "Yes",
                "No",
            ],
        )

    filt = (
        scored_df[
            scored_df[
                "age"
            ].between(
                *age_range
            )

            &

            scored_df[
                "gender"
            ].isin(
                genders
            )

            &

            scored_df[
                "risk_level"
            ]
            .astype(str)
            .isin(
                risks
            )
        ]
        .copy()
    )

    if (
        readmitted
        == "Yes"
    ):
        filt = filt[
            filt[
                "readmitted_30d"
            ]
            == 1
        ]

    elif (
        readmitted
        == "No"
    ):
        filt = filt[
            filt[
                "readmitted_30d"
            ]
            == 0
        ]

    log_action(
        "COHORT_FILTER_APPLIED",
        f"n={len(filt)}",
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Patients in cohort",
        f"{len(filt):,}",
    )

    c2.metric(
        "Average predicted risk",
        (
            f"{filt['risk_score'].mean():.1%}"
            if len(filt)
            else "—"
        ),
    )

    c3.metric(
        "Observed readmission",
        (
            f"{filt['readmitted_30d'].mean():.1%}"
            if len(filt)
            else "—"
        ),
    )

    left, right = (
        st.columns(2)
    )

    with left:
        fig = px.scatter(
            filt,
            x="age",
            y="risk_score",
            size="comorbidity_count",
            hover_name="patient_id",
            title=(
                "Age vs predicted risk"
            ),
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        fig.update_layout(
            template="plotly_dark",

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            height=400,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        tmp = filt.copy()

        tmp[
            "Comorbidity group"
        ] = pd.cut(
            tmp[
                "comorbidity_count"
            ],
            [
                -1,
                0,
                2,
                4,
                20,
            ],
            labels=[
                "0",
                "1–2",
                "3–4",
                "5+",
            ],
        )

        grouped = (
            tmp.groupby(
                "Comorbidity group",
                observed=False,
            )[
                "risk_score"
            ]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            grouped,
            x="Comorbidity group",
            y="risk_score",
            title=(
                "Risk by comorbidity "
                "burden"
            ),
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        fig.update_layout(
            template="plotly_dark",

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            height=400,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        "Highest-risk patients "
        "in filtered cohort"
    )

    show_columns = [
        "patient_id",
        "age",
        "gender",
        "risk_score",
        "prior_admissions",
        "comorbidity_count",
        "hba1c",
        "systolic_bp",
    ]

    table = (
        filt
        .sort_values(
            "risk_score",
            ascending=False,
        )[
            show_columns
        ]
        .head(25)
        .copy()
    )

    table[
        "risk_score"
    ] = (
        table[
            "risk_score"
        ]
        .map(
            lambda value:
                f"{value:.1%}"
        )
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# Data Quality
# =========================================================

elif page == "Data Quality":

    st.title(
        "Data Quality Center"
    )

    st.caption(
        "Review missing values, "
        "duplicates, and feature "
        "range checks."
    )

    total_missing = int(
        df.isna()
        .sum()
        .sum()
    )

    duplicate_rows = int(
        df.duplicated(
            subset=[
                "patient_id"
            ]
        ).sum()
    )

    out_of_range = {
        "BMI outside 10–70":
            int(
                (
                    (
                        df[
                            "bmi"
                        ]
                        < 10
                    )
                    |
                    (
                        df[
                            "bmi"
                        ]
                        > 70
                    )
                ).sum()
            ),

        "Systolic BP outside 70–260":
            int(
                (
                    (
                        df[
                            "systolic_bp"
                        ]
                        < 70
                    )
                    |
                    (
                        df[
                            "systolic_bp"
                        ]
                        > 260
                    )
                ).sum()
            ),

        "HbA1c outside 3–20":
            int(
                (
                    (
                        df[
                            "hba1c"
                        ]
                        < 3
                    )
                    |
                    (
                        df[
                            "hba1c"
                        ]
                        > 20
                    )
                ).sum()
            ),
    }

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Missing cells",
        total_missing,
    )

    c2.metric(
        "Duplicate patient IDs",
        duplicate_rows,
    )

    c3.metric(
        "Rows",
        f"{len(df):,}",
    )

    missing = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing values",
    ]

    missing = (
        missing[
            missing[
                "Missing values"
            ]
            > 0
        ]
    )

    left, right = (
        st.columns(2)
    )

    with left:
        st.subheader(
            "Missingness"
        )

        if len(
            missing
        ):
            st.dataframe(
                missing,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.success(
                "No missing values detected."
            )

    with right:
        st.subheader(
            "Range checks"
        )

        range_df = pd.DataFrame(
            [
                {
                    "Check": key,
                    "Violations": value,
                }
                for key, value
                in out_of_range.items()
            ]
        )

        st.dataframe(
            range_df,
            use_container_width=True,
            hide_index=True,
        )

    completeness = (
        1
        -
        total_missing
        /
        (
            df.shape[0]
            *
            df.shape[1]
        )
    )

    st.progress(
        completeness
    )

    st.caption(
        (
            f"Dataset completeness: "
            f"{completeness:.2%}"
        )
    )


# =========================================================
# Audit Log
# =========================================================

elif page == "Audit Log":

    st.title(
        "Audit Log"
    )

    st.caption(
        "Trace major interactions "
        "with the demonstration system."
    )

    audit = read_audit()

    st.dataframe(
        audit,
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Clear audit log"
    ):
        con = sqlite3.connect(
            AUDIT_DB
        )

        con.execute(
            """
            DELETE FROM audit_log
            """
        )

        con.commit()
        con.close()

        st.rerun()


# =========================================================
# About
# =========================================================

elif page == "About":

    st.title(
        "About PulseRisk"
    )

    st.markdown(
        """
### Overview

PulseRisk is an end-to-end machine-learning
risk analytics platform for synthetic
30-day hospital readmission prediction.

The project demonstrates:

- model training and comparison,
- model selection,
- patient-level risk scoring,
- explainable machine learning,
- decision-threshold analysis,
- cohort analytics,
- data-quality monitoring,
- REST API inference,
- audit logging,
- and containerized deployment.

### Machine Learning

PulseRisk compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

Candidate models are evaluated using:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1 score

The production model is selected using
ROC-AUC and additional threshold analysis
is performed to evaluate precision-recall
trade-offs.

### Architecture

**Frontend**

Streamlit + Plotly

**Backend**

FastAPI

**Machine Learning**

scikit-learn

**Data**

Pandas + synthetic patient dataset

**Audit storage**

SQLite

**Deployment**

Docker-compatible services

### Explainability

PulseRisk provides:

- global feature influence
- patient-level feature contributions

For the current Logistic Regression model,
patient-level contributions are calculated
using standardized feature values and model
coefficients.

### Clinical safety

PulseRisk uses synthetic data exclusively.

The application is not intended for
diagnosis, treatment, triage, medical
decision-making, or real-world clinical use.
        """
    )