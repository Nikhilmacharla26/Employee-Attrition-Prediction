"""

------------------------------------------------------------
Streamlit entrypoint for the Employee Turnover Risk Prediction
and Early Intervention System.

Run from the project root:
    streamlit run app/main.py

Requires models/model_bundle.pkl, produced by:
    python -m src.train --data data/HR-Employee-Attrition-New.csv
------------------------------------------------------------
"""

import sys
from pathlib import Path

# Make the project root importable (so `import config`, `from src...` work
# regardless of the working directory Streamlit is launched from).
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

import config
from app.components import (
    attrition_distribution_figure,
    confusion_matrix_figure,
    feature_importance_figure,
    model_comparison_figure,
    render_metric_strip,
    risk_gauge_figure,
)
from app.theme import apply_theme, render_hero
from src.predict import assess_employee, load_bundle

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Employee Turnover Risk Predictor", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
apply_theme(st)
render_hero(st)


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle(config.MODEL_BUNDLE_PATH)


bundle = get_bundle()

if bundle is None:
    st.error(
        "No trained model found yet. From the project root, run:\n\n"
        "```\npython -m src.train --data data/HR-Employee-Attrition-New.csv\n```\n\n"
        "This creates `models/model_bundle.pkl`, which this app loads."
    )
    st.stop()

metrics = bundle["metrics"]
per_model_f1 = bundle["per_model_f1"]
numeric_columns = bundle["numeric_columns"]
categorical_columns = bundle["categorical_columns"]
categorical_options = bundle["categorical_options"]
numeric_ranges = bundle["numeric_ranges"]

# ============================================================
# TOP METRIC STRIP
# ============================================================
render_metric_strip(st, metrics)
st.write("")

tab_predict, tab_dashboard, tab_about = st.tabs(["🔮 Predict Risk", "📊 Model Insights", "ℹ️ About"])

# ============================================================
# TAB 1 — PREDICTION FORM
# ============================================================
with tab_predict:
    left, right = st.columns([1.05, 1.4], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Employee Profile</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Fill in the key factors below. Fields not shown are '
            "auto-filled with typical (median/most common) values from the training data.</div>",
            unsafe_allow_html=True,
        )

        with st.form("employee_form"):
            c1, c2 = st.columns(2)

            def num_input(container, col, label):
                rng = numeric_ranges.get(col)
                if rng is None:
                    return container.number_input(label, value=0)
                return container.number_input(
                    label, min_value=float(rng["min"]), max_value=float(max(rng["max"], rng["min"] + 1)),
                    value=float(rng["median"]),
                )

            def scale_select(container, label, mapping):
                return container.select_slider(label, options=list(mapping.keys()), value=3, format_func=lambda x: mapping[x])

            user_data = {}
            if "Age" in numeric_columns:
                user_data["Age"] = num_input(c1, "Age", "Age")
            if "MonthlyIncome" in numeric_columns:
                user_data["MonthlyIncome"] = num_input(c2, "MonthlyIncome", "Monthly Income")
            if "OverTime" in categorical_columns:
                user_data["OverTime"] = c1.selectbox("OverTime?", ["No", "Yes"])
            if "JobRole" in categorical_columns:
                user_data["JobRole"] = c2.selectbox("Job Role", categorical_options["JobRole"])
            if "JobSatisfaction" in numeric_columns:
                user_data["JobSatisfaction"] = scale_select(c1, "Job Satisfaction", config.SATISFACTION_LABELS)
            if "EnvironmentSatisfaction" in numeric_columns:
                user_data["EnvironmentSatisfaction"] = scale_select(c2, "Environment Satisfaction", config.SATISFACTION_LABELS)
            if "JobInvolvement" in numeric_columns:
                user_data["JobInvolvement"] = scale_select(c1, "Job Involvement", config.INVOLVEMENT_LABELS)
            if "WorkLifeBalance" in numeric_columns:
                user_data["WorkLifeBalance"] = scale_select(c2, "Work-Life Balance", config.BALANCE_LABELS)
            if "JobLevel" in numeric_columns:
                user_data["JobLevel"] = c1.slider("Job Level", 1, 5, 2)
            if "YearsAtCompany" in numeric_columns:
                user_data["YearsAtCompany"] = c2.number_input("Years At Company", min_value=0, value=3, step=1)
            if "YearsInCurrentRole" in numeric_columns:
                user_data["YearsInCurrentRole"] = c1.number_input("Years In Current Role", min_value=0, value=2, step=1)
            if "YearsSinceLastPromotion" in numeric_columns:
                user_data["YearsSinceLastPromotion"] = c2.number_input("Years Since Last Promotion", min_value=0, value=1, step=1)

            submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if not submitted:
            st.markdown(
                """<div class="glass-card" style="text-align:center; padding:60px 20px;">
                        <div style="font-size:2.2rem;">🧭</div>
                        <div class="section-title" style="margin-top:10px;">Awaiting employee details</div>
                        <div class="section-sub">Fill the form and click <b>Assess Risk</b> to see the prediction,
                        risk gauge and recommended action.</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            result = assess_employee(user_data, bundle)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            g1, g2 = st.columns([1, 1.3])
            with g1:
                st.plotly_chart(risk_gauge_figure(result["probability"], result["risk_level"]), use_container_width=True)
            with g2:
                st.markdown(f"### {result['result_text']}")
                st.markdown(f'<span class="risk-pill risk-{result["risk_level"]}">RISK: {result["risk_level"]}</span>', unsafe_allow_html=True)
                st.write("")
                st.metric("Attrition probability", f"{result['probability']*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Key Risk Factors</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<span class="factor-chip">⚠️ {f}</span>' for f in result["risk_factors"]), unsafe_allow_html=True)
            st.write("")
            st.markdown('<div class="section-title">Recommended HR Action</div>', unsafe_allow_html=True)
            st.info(result["recommendation"])
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — MODEL INSIGHTS DASHBOARD
# ============================================================
with tab_dashboard:
    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Comparison (F1 Score)</div>', unsafe_allow_html=True)
        st.plotly_chart(model_comparison_figure(per_model_f1), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confusion Matrix — Stacking Classifier</div>', unsafe_allow_html=True)
        st.plotly_chart(confusion_matrix_figure(metrics["confusion_matrix"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    d3, d4 = st.columns(2, gap="large")
    with d3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 15 Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
        st.plotly_chart(feature_importance_figure(bundle["feature_importance"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Attrition Distribution (Training Data)</div>', unsafe_allow_html=True)
        st.plotly_chart(attrition_distribution_figure(bundle["attrition_counts"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classification Report — Stacking Classifier</div>', unsafe_allow_html=True)
    st.code(metrics["classification_report"], language=None)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3 — ABOUT
# ============================================================
with tab_about:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About this system</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
- **Goal:** identify employees with elevated turnover risk so HR can intervene early.
- **Pipeline:** IQR outlier capping → Ordinal encoding of categoricals → base learners
  (Random Forest, Decision Tree, KNN, Logistic Regression) → Logistic Regression stacking
  meta-model, all tuned with `GridSearchCV` (5-fold CV, F1-optimized).
- **Data split:** {bundle['n_train']} training rows / {bundle['n_test']} held-out test rows.
- **Risk bands:** LOW < 35%, MEDIUM 35–65%, HIGH ≥ 66% predicted attrition probability.
- **Note:** predictions are decision-support only, not a substitute for HR judgment.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center; color:#9aa3b8; font-size:0.8rem; margin-top:20px;">'
    "Employee Turnover Risk Prediction and Early Intervention System</div>",
    unsafe_allow_html=True,
)
