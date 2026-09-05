"""
tests/test_pipeline.py
------------------------------------------------------------
Smoke test: trains on a small synthetic dataset (fast mode) and
verifies the bundle can be saved, reloaded, and used to predict.

Run:
    pytest tests/test_pipeline.py -q
------------------------------------------------------------
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import assess_employee, load_bundle  # noqa: E402
from src.train import run_training  # noqa: E402


def make_synthetic_df(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Age": rng.integers(18, 60, n),
        "Attrition": rng.choice(["Yes", "No"], n, p=[0.2, 0.8]),
        "BusinessTravel": rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n),
        "DailyRate": rng.integers(100, 1500, n),
        "Department": rng.choice(["Sales", "R&D", "HR"], n),
        "DistanceFromHome": rng.integers(1, 30, n),
        "Education": rng.integers(1, 5, n),
        "EducationField": rng.choice(["Life Sciences", "Medical", "Marketing"], n),
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(n),
        "EnvironmentSatisfaction": rng.integers(1, 5, n),
        "Gender": rng.choice(["Male", "Female"], n),
        "HourlyRate": rng.integers(30, 100, n),
        "JobInvolvement": rng.integers(1, 5, n),
        "JobLevel": rng.integers(1, 6, n),
        "JobRole": rng.choice(["Sales Executive", "Research Scientist", "Manager"], n),
        "JobSatisfaction": rng.integers(1, 5, n),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
        "MonthlyIncome": rng.integers(1000, 20000, n),
        "MonthlyRate": rng.integers(2000, 27000, n),
        "NumCompaniesWorked": rng.integers(0, 10, n),
        "Over18": "Y",
        "OverTime": rng.choice(["Yes", "No"], n),
        "PercentSalaryHike": rng.integers(11, 25, n),
        "PerformanceRating": rng.integers(3, 5, n),
        "RelationshipSatisfaction": rng.integers(1, 5, n),
        "StandardHours": 80,
        "StockOptionLevel": rng.integers(0, 4, n),
        "TotalWorkingYears": rng.integers(0, 40, n),
        "TrainingTimesLastYear": rng.integers(0, 6, n),
        "WorkLifeBalance": rng.integers(1, 5, n),
        "YearsAtCompany": rng.integers(0, 20, n),
        "YearsInCurrentRole": rng.integers(0, 15, n),
        "YearsSinceLastPromotion": rng.integers(0, 15, n),
        "YearsWithCurrManager": rng.integers(0, 15, n),
    })


def test_train_save_load_predict(tmp_path):
    df = make_synthetic_df()

    bundle = run_training(df, fast=True, verbose=False)
    assert "model" in bundle
    assert 0.0 <= bundle["metrics"]["accuracy"] <= 1.0

    bundle_path = tmp_path / "model_bundle.pkl"
    joblib.dump(bundle, bundle_path)

    loaded = load_bundle(bundle_path)
    assert loaded is not None

    result = assess_employee(
        {"Age": 30, "OverTime": "Yes", "JobSatisfaction": 1, "MonthlyIncome": 20000},
        loaded,
    )
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["risk_factors"], list) and len(result["risk_factors"]) > 0


def test_load_bundle_missing_file_returns_none(tmp_path):
    assert load_bundle(tmp_path / "does_not_exist.pkl") is None
