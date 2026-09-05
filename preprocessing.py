"""
src/preprocessing.py
------------------------------------------------------------
Custom transformer + pipeline/param-grid builders shared by
training and inference.

IMPORTANT: IQRCapper must live in an importable module (not in a
__main__ script) so joblib can unpickle the trained pipeline
correctly wherever it's loaded (e.g. from app/main.py).
------------------------------------------------------------
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class IQRCapper(BaseEstimator, TransformerMixin):
    """Caps numeric columns to the 1.5*IQR range. Fitted only on
    training folds when used inside a Pipeline/GridSearchCV."""

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_columns_ = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.lower_bounds_, self.upper_bounds_ = {}, {}

        for col in self.numeric_columns_:
            Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - 1.5 * IQR
            self.upper_bounds_[col] = Q3 + 1.5 * IQR

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.numeric_columns_:
            X[col] = X[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
        return X


def build_preprocessor(numeric_columns, categorical_columns):
    """ColumnTransformer: ordinal-encode categoricals, passthrough numerics."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )


def build_base_models(preprocessor):
    """The four base-learner pipelines used inside the stacking ensemble."""
    return {
        "RF": Pipeline([
            ("iqr", IQRCapper()),
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42)),
        ]),
        "DT": Pipeline([
            ("iqr", IQRCapper()),
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]),
        "KNN": Pipeline([
            ("iqr", IQRCapper()),
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        "LR": Pipeline([
            ("iqr", IQRCapper()),
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000)),
        ]),
    }


def build_param_grids(fast: bool = False):
    """Hyperparameter grids for each base model. `fast=True` gives a small
    grid for quick smoke tests; `fast=False` is the full original grid."""
    if fast:
        return {
            "RF": {"model__n_estimators": [150], "model__max_depth": [12], "model__class_weight": ["balanced"]},
            "DT": {"model__criterion": ["gini"], "model__max_depth": [10], "model__class_weight": ["balanced"]},
            "KNN": {"model__n_neighbors": [7], "model__weights": ["distance"]},
            "LR": {
                "model__C": [1],
                "model__penalty": ["l2"],
                "model__solver": ["liblinear"],
                "model__class_weight": ["balanced"],
            },
        }

    return {
        "RF": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 15],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__class_weight": ["balanced"],
        },
        "DT": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [5, 10, 15],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__class_weight": ["balanced"],
        },
        "KNN": {
            "model__n_neighbors": [5, 7, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
        "LR": {
            "model__C": [0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
            "model__class_weight": ["balanced"],
        },
    }


def build_stack_param_grid(fast: bool = False):
    return {
        "passthrough": [False],
        "final_estimator__C": [1] if fast else [0.1, 1, 10],
        "final_estimator__penalty": ["l2"],
        "final_estimator__solver": ["liblinear"],
        "final_estimator__class_weight": ["balanced"],
    }
