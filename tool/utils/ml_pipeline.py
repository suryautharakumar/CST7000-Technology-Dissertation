from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def detect_problem_type(target: pd.Series) -> str:
    clean_target = target.dropna()
    if clean_target.empty:
        return "classification"
    if pd.api.types.is_numeric_dtype(clean_target) and clean_target.nunique() > 10:
        return "regression"
    return "classification"


def build_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    missing_by_column = df.isna().sum().to_dict()
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": len(df.select_dtypes(include=["number", "bool"]).columns),
        "categorical_columns": len(df.select_dtypes(exclude=["number", "bool"]).columns),
        "missing_by_column": missing_by_column,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _models(problem_type: str) -> dict[str, Any]:
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=120, random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=120, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    }


def _classification_metrics(y_test, y_pred, pipeline, X_test) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }

    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)
            if probs.shape[1] == 2:
                metrics["ROC-AUC"] = round(roc_auc_score(y_test, probs[:, 1]), 4)
            else:
                metrics["ROC-AUC"] = round(roc_auc_score(y_test, probs, multi_class="ovr"), 4)
        else:
            metrics["ROC-AUC"] = "N/A"
    except Exception:
        metrics["ROC-AUC"] = "N/A"

    return metrics


def _regression_metrics(y_test, y_pred) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(float(rmse), 4),
        "R² Score": round(r2_score(y_test, y_pred), 4),
    }


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def _extract_feature_importance(pipeline: Pipeline) -> list[dict[str, Any]]:
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    feature_names = _get_feature_names(preprocessor)

    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
    return [{"feature": name, "importance": round(float(score), 5)} for name, score in pairs]


def train_models(df: pd.DataFrame, target_col: str, problem_type: str) -> dict[str, Any]:
    df = df.copy()
    df = df.dropna(axis=0, subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Remove columns that are completely empty.
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No usable feature columns found after removing empty columns.")

    stratify = y if problem_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    results = []
    best_pipeline = None
    best_model_name = None
    best_score = -np.inf

    for model_name, model in _models(problem_type).items():
        start_time = time.time()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(X_train)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        training_time = round(time.time() - start_time, 4)

        if problem_type == "classification":
            metric_values = _classification_metrics(y_test, y_pred, pipeline, X_test)
            score = float(metric_values["F1-Score"])
        else:
            metric_values = _regression_metrics(y_test, y_pred)
            score = float(metric_values["R² Score"])

        metric_values["Training Time (s)"] = training_time
        metric_values["Model"] = model_name
        results.append(metric_values)

        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_model_name = model_name

    assert best_pipeline is not None

    if problem_type == "classification":
        chart_metric = "F1-Score"
    else:
        chart_metric = "R² Score"

    chart_labels = [row["Model"] for row in results]
    chart_values = [float(row[chart_metric]) if isinstance(row[chart_metric], (int, float)) else 0 for row in results]

    return {
        "metrics": results,
        "best_model_name": best_model_name,
        "chart_labels": chart_labels,
        "chart_values": chart_values,
        "chart_metric": chart_metric,
        "model_bundle": {
            "pipeline": best_pipeline,
            "problem_type": problem_type,
            "target_column": target_col,
            "feature_columns": list(X.columns),
            "feature_importance": _extract_feature_importance(best_pipeline),
        },
    }


def prepare_prediction_input(form_data, features: list[str], original_df: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for feature in features:
        value = form_data.get(feature, "")
        if feature in original_df.columns and pd.api.types.is_numeric_dtype(original_df[feature]):
            try:
                row[feature] = float(value)
            except ValueError:
                row[feature] = np.nan
        else:
            row[feature] = value
    return pd.DataFrame([row], columns=features)
