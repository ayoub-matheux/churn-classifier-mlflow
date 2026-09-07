"""Unit tests for pipeline construction, data cleaning, and prediction sanity."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.pipeline import build_pipeline
from src.utils import clean_churn_dataframe, load_config


def test_build_pipeline_structure():
    """Verify pipeline has expected 'pre' and 'model' steps for both models."""
    logreg_pipe = build_pipeline(["num1"], ["cat1"], "logreg")
    assert isinstance(logreg_pipe, Pipeline)
    assert "pre" in dict(logreg_pipe.named_steps)
    assert "model" in dict(logreg_pipe.named_steps)
    assert logreg_pipe.named_steps["model"].__class__.__name__ == "LogisticRegression"

    rf_pipe = build_pipeline(["num1"], ["cat1"], "random_forest")
    assert isinstance(rf_pipe, Pipeline)
    assert "pre" in dict(rf_pipe.named_steps)
    assert "model" in dict(rf_pipe.named_steps)
    assert rf_pipe.named_steps["model"].__class__.__name__ == "RandomForestClassifier"


def test_build_pipeline_unsupported_model():
    """Verify that an unsupported model type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_pipeline(["num1"], ["cat1"], "unsupported_model")


def test_clean_churn_dataframe():
    """Verify data cleaning properly formats target, drops ID, and parses numeric strings."""
    sample_data = pd.DataFrame({
        "customerID": ["001-A", "002-B", "003-C"],
        "tenure": [1, 24, 60],
        "MonthlyCharges": [29.85, 56.95, 105.65],
        "TotalCharges": ["29.85", " ", "6339.00"],  # Notice whitespace here
        "gender": ["Female", "Male", "Female"],
        "Churn": ["No", "Yes", "No"],
    })

    X, y = clean_churn_dataframe(sample_data, target_col="Churn")

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert list(y) == [0, 1, 0]
    assert np.isnan(X.loc[1, "TotalCharges"])  # The whitespace should become NaN
    assert X.loc[0, "TotalCharges"] == 29.85


def test_pipeline_fit_and_predict():
    """Verify end-to-end fitting and probability predictions on sample tabular data."""
    train_df = pd.DataFrame({
        "num_a": [10.0, np.nan, 30.0, 40.0, 50.0],
        "cat_b": ["A", "B", "A", np.nan, "B"],
    })
    y = np.array([0, 1, 0, 1, 0])

    pipe = build_pipeline(numeric=["num_a"], categorical=["cat_b"], model_type="logreg")
    pipe.fit(train_df, y)

    test_df = pd.DataFrame({
        "num_a": [20.0, 35.0],
        "cat_b": ["A", "UnknownCat"],
    })
    preds = pipe.predict(test_df)
    probs = pipe.predict_proba(test_df)

    assert len(preds) == 2
    assert preds.shape == (2,)
    assert probs.shape == (2, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), rtol=1e-5)


def test_config_loading():
    """Verify configuration file exists and contains essential sections."""
    config = load_config("configs/config.yaml")
    assert "data" in config
    assert "features" in config
    assert "model" in config
    assert "cv" in config
    assert "mlflow" in config
    assert "csv_path" in config["data"]
    assert "numeric" in config["features"]
    assert "categorical" in config["features"]

