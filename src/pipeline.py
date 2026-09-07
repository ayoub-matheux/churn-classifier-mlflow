"""Pipeline construction module for tabular churn classification."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_type: str = "logreg",
    model_params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a scikit-learn Pipeline with ColumnTransformer and classification model.

    Args:
        numeric: List of numerical column names.
        categorical: List of categorical column names.
        model_type: Model identifier, 'logreg' or 'random_forest'.
        model_params: Optional dictionary of keyword arguments passed to the model.
        random_state: Random seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline containing 'pre' and 'model' steps.
    """
    model_params = model_params or {}

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric),
            ("cat", cat_transformer, categorical),
        ]
    )

    if model_type == "logreg":
        base_params: dict[str, Any] = {
            "max_iter": 1000,
            "random_state": random_state,
        }
        base_params.update(model_params)
        model = LogisticRegression(**base_params)
    elif model_type == "random_forest":
        base_params = {
            "random_state": random_state,
        }
        base_params.update(model_params)
        model = RandomForestClassifier(**base_params)
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Expected 'logreg' or 'random_forest'."
        )

    return Pipeline(steps=[("pre", preprocessor), ("model", model)])

