"""
src/predict.py
------------------------------------------------------------
Inference-time helpers used by the Streamlit app: loading the
trained bundle, building a model-ready input row from partial
user input, predicting, and deriving risk level / risk factors /
recommended action.
------------------------------------------------------------
"""

from pathlib import Path

import joblib
import pandas as pd

import config
from src.preprocessing import IQRCapper  # noqa: F401  (needed to unpickle the bundle)


def load_bundle(path: Path = config.MODEL_BUNDLE_PATH):
    """Loads the trained model bundle, or returns None if it doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)


def build_employee_row(user_data: dict, bundle: dict) -> pd.DataFrame:
    """Fills in a full model input row: user-provided fields override the
    bundle's stored defaults (median/mode from training data)."""
    row = dict(bundle["default_row"])
    row.update(user_data)
    return pd.DataFrame([row])[bundle["columns"]]


def predict_employee(user_data: dict, bundle: dict):
    """Returns (prediction, probability) for a single employee."""
    employee_row = build_employee_row(user_data, bundle)
    model = bundle["model"]
    prediction = model.predict(employee_row)[0]
    probability = float(model.predict_proba(employee_row)[0][1])
    return prediction, probability


def risk_level_from_probability(probability: float) -> str:
    if probability < config.RISK_LOW_MAX:
        return "LOW"
    if probability < config.RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def recommended_action(risk_level: str) -> str:
    return config.RISK_RECOMMENDATIONS[risk_level]


def identify_risk_factors(user_data: dict) -> list:
    """Human-readable list of the key risk factors present in the given
    (partial) employee data, mirroring the rules from the original notebook."""
    factors = []

    if user_data.get("OverTime") == "Yes":
        factors.append("Overtime = Yes")
    if user_data.get("JobSatisfaction", 3) <= 2:
        factors.append(f"Job Satisfaction = {user_data.get('JobSatisfaction')}")
    if user_data.get("EnvironmentSatisfaction", 3) <= 2:
        factors.append(f"Environment Satisfaction = {user_data.get('EnvironmentSatisfaction')}")
    if user_data.get("JobInvolvement", 3) <= 2:
        factors.append(f"Job Involvement = {user_data.get('JobInvolvement')}")
    if user_data.get("WorkLifeBalance", 3) <= 2:
        factors.append(f"Work-Life Balance = {user_data.get('WorkLifeBalance')}")
    if user_data.get("YearsAtCompany", 99) <= 2:
        factors.append(f"Short tenure = {user_data.get('YearsAtCompany')} yrs")
    if user_data.get("YearsInCurrentRole", 99) <= 1:
        factors.append(f"Short time in role = {user_data.get('YearsInCurrentRole')} yrs")
    if user_data.get("YearsSinceLastPromotion", 0) >= 4:
        factors.append(f"No promotion in {user_data.get('YearsSinceLastPromotion')} yrs")
    if user_data.get("MonthlyIncome", float("inf")) < 30000:
        factors.append(f"Lower income = \u20b9{user_data.get('MonthlyIncome'):,.0f}")

    if not factors:
        factors = ["No major risk factors identified"]

    return factors


def assess_employee(user_data: dict, bundle: dict) -> dict:
    """One-call convenience wrapper the app uses: prediction, probability,
    risk level, risk factors and recommendation for a given employee."""
    prediction, probability = predict_employee(user_data, bundle)
    risk_level = risk_level_from_probability(probability)
    return {
        "prediction": prediction,
        "result_text": "May Leave" if prediction == 1 else "Likely to Stay",
        "probability": probability,
        "risk_level": risk_level,
        "risk_factors": identify_risk_factors(user_data),
        "recommendation": recommended_action(risk_level),
    }
