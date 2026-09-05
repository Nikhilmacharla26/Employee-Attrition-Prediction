"""
src/train.py
------------------------------------------------------------
Trains the IQR-capping + OrdinalEncoder + Stacking pipeline
(RF + DT + KNN + LR -> Logistic-Regression meta-model) and saves
a deployable artifact bundle to models/model_bundle.pkl.

CLI usage:
    python -m src.train --data data/HR-Employee-Attrition-New.csv
    python -m src.train --data data/HR-Employee-Attrition-New.csv --fast

Library usage (e.g. from a notebook or the app):
    from src.train import run_training
    bundle = run_training(df, fast=True)
------------------------------------------------------------
"""

import argparse
import sys

import joblib
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

import config
from src.preprocessing import (
    build_base_models,
    build_param_grids,
    build_preprocessor,
    build_stack_param_grid,
)


def run_training(df: pd.DataFrame, fast: bool = False, verbose: bool = True) -> dict:
    """Runs the full training pipeline on an in-memory DataFrame and
    returns the artifact bundle (does not write to disk)."""

    drop_cols = [c for c in config.DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.drop(config.TARGET_COLUMN, axis=1)
    y = df[config.TARGET_COLUMN].map(config.TARGET_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    models = build_base_models(preprocessor)
    params = build_param_grids(fast=fast)

    best_models = {}
    for name, pipeline in models.items():
        if verbose:
            print(f"=== Training: {name} ===")
        grid = GridSearchCV(pipeline, params[name], cv=config.CV_FOLDS, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        if verbose:
            print("Best Params:", grid.best_params_)
            print("Best CV F1 :", round(grid.best_score_, 4))

    estimators = [(k.lower(), v) for k, v in best_models.items()]
    meta = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, class_weight="balanced")
    stack = StackingClassifier(estimators=estimators, final_estimator=meta, cv=config.CV_FOLDS, passthrough=False, n_jobs=-1)

    if verbose:
        print("\n=== Training Stacking Model ===")
    stack_grid = GridSearchCV(stack, build_stack_param_grid(fast=fast), cv=config.CV_FOLDS, scoring="f1", n_jobs=1)
    stack_grid.fit(X_train, y_train)
    best_stack = stack_grid.best_estimator_

    if verbose:
        print("Best Stack Params:", stack_grid.best_params_)
        print("Best Stack CV F1 :", round(stack_grid.best_score_, 4))

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    y_pred = best_stack.predict(X_test)
    y_proba = best_stack.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, digits=2),
    }

    per_model_f1 = {name: f1_score(y_test, m.predict(X_test), zero_division=0) for name, m in best_models.items()}
    per_model_f1["Stacking"] = metrics["f1"]

    rf_preprocessor = best_models["RF"].named_steps["preprocessor"]
    rf_model = best_models["RF"].named_steps["model"]
    feature_importance = (
        pd.DataFrame({"Feature": rf_preprocessor.get_feature_names_out(), "Importance": rf_model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    attrition_counts = df[config.TARGET_COLUMN].value_counts().reindex(["No", "Yes"], fill_value=0).to_dict()

    default_row = {
        col: (X_train[col].median() if col in numeric_columns else X_train[col].mode().iloc[0])
        for col in X_train.columns
    }
    categorical_options = {col: sorted(X_train[col].dropna().unique().tolist()) for col in categorical_columns}
    numeric_ranges = {
        col: {"min": float(X_train[col].min()), "max": float(X_train[col].max()), "median": float(X_train[col].median())}
        for col in numeric_columns
    }

    return {
        "model": best_stack,
        "best_models": best_models,
        "columns": X_train.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_options": categorical_options,
        "numeric_ranges": numeric_ranges,
        "default_row": default_row,
        "metrics": metrics,
        "per_model_f1": per_model_f1,
        "feature_importance": feature_importance,
        "attrition_counts": attrition_counts,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the employee turnover stacking model.")
    parser.add_argument("--data", type=str, default=str(config.DEFAULT_DATA_PATH), help="Path to the HR attrition CSV.")
    parser.add_argument("--out", type=str, default=str(config.MODEL_BUNDLE_PATH), help="Output bundle path.")
    parser.add_argument("--fast", action="store_true", help="Use a smaller grid for a quick smoke-test run.")
    args = parser.parse_args()

    if not config.DATA_DIR.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not config.MODELS_DIR.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    from pathlib import Path

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(
            f"ERROR: dataset not found at {data_path}\n"
            "Place HR-Employee-Attrition-New.csv in data/, or pass --data <path>."
        )

    print("Loading dataset from:", data_path)
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)

    bundle = run_training(df, fast=args.fast, verbose=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"\nSaved model bundle -> {out_path}")


if __name__ == "__main__":
    main()
