"""
config.py
------------------------------------------------------------
Central configuration: file paths and constants shared by
src/ (training/inference logic) and app/ (Streamlit UI).
------------------------------------------------------------
"""

from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_DATA_PATH = DATA_DIR / "HR-Employee-Attrition-New.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "model_bundle.pkl"

# ------------------------------------------------------------
# Columns dropped before modeling (IDs / constant columns)
# ------------------------------------------------------------
DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
TARGET_COLUMN = "Attrition"
TARGET_MAP = {"No": 0, "Yes": 1}

# ------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# ------------------------------------------------------------
# Risk bands (predicted attrition probability)
# ------------------------------------------------------------
RISK_LOW_MAX = 0.35
RISK_MEDIUM_MAX = 0.66

RISK_RECOMMENDATIONS = {
    "LOW": "Continue regular engagement and retention activities.",
    "MEDIUM": "Monitor engagement, workload, satisfaction and career development.",
    "HIGH": "Review workload and career progression. Consider immediate retention action.",
}

# ------------------------------------------------------------
# Web form field options (used by app/main.py)
# ------------------------------------------------------------
SATISFACTION_LABELS = {1: "1 - Low", 2: "2 - Medium", 3: "3 - Good", 4: "4 - Very Good"}
INVOLVEMENT_LABELS = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
BALANCE_LABELS = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
