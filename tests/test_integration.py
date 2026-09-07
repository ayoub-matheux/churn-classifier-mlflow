"""Small end-to-end tests for the train, evaluate and predict workflows."""

import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.evaluate import evaluate
from src.pipeline import build_pipeline
from src.predict import predict
from src.train import make_cv_splitter, make_search, train


def _write_test_project(tmp_path: Path) -> Path:
    """Create a balanced, local dataset and deliberately tiny tuning grid."""
    rows = []
    for index in range(40):
        churn = index % 2
        rows.append(
            {
                "customerID": f"customer-{index}",
                "num": float(index),
                "category": "high" if churn else "low",
                "Churn": "Yes" if churn else "No",
            }
        )
    csv_path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config = {
        "data": {
            "csv_path": str(csv_path),
            "target": "Churn",
            "test_size": 0.25,
            "random_state": 7,
        },
        "features": {"numeric": ["num"], "categorical": ["category"]},
        "model": {"type": "logreg", "logreg": {"param_grid": {"model__C": [1.0]}}},
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits": 2,
            "shuffle": True,
            "n_jobs": 1,
            "scoring": "roc_auc",
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "experiment_name": "integration-test",
            "registered_model_name": "IntegrationChurnClassifier",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_train_evaluate_and_predict_end_to_end(tmp_path, monkeypatch):
    """A tuned run, its model, evaluation artifacts and prediction CSV are created."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    config_path = _write_test_project(tmp_path)

    run_id = train(str(config_path))
    metadata = json.loads((tmp_path / "data/processed/latest_run.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == run_id
    assert "baseline_run_id" in metadata

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    client = mlflow.tracking.MlflowClient()
    assert client.get_run(run_id).data.metrics["tuned_auc_minus_baseline_auc"] == pytest.approx(
        metadata["tuned_auc_minus_baseline_auc"]
    )
    assert client.list_artifacts(run_id, "model")

    metrics = evaluate(str(config_path), run_id=run_id)
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0
    assert client.list_artifacts(run_id, "evaluation")

    output_path = tmp_path / "predictions.csv"
    predict(str(tmp_path / "raw.csv"), str(output_path), config_path=str(config_path))
    prediction_df = pd.read_csv(output_path)
    assert len(prediction_df) == 40
    assert {"churn_prediction", "churn_probability"}.issubset(prediction_df.columns)


def test_cv_splitter_factory_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown cv.strategy 'not-a-splitter'"):
        make_cv_splitter({"strategy": "not-a-splitter"}, random_state=42)


def test_search_factory_supports_grid_and_random_search():
    pipeline = build_pipeline(numeric=["num"], categorical=["category"])
    cv = make_cv_splitter({"n_splits": 2}, random_state=42)

    grid = make_search(pipeline, {"model__C": [1.0]}, cv, {}, "roc_auc", 42)
    random = make_search(
        pipeline,
        {"model__C": [0.1, 1.0]},
        cv,
        {"search_type": "random", "n_iter": 2},
        "roc_auc",
        42,
    )

    assert isinstance(grid, GridSearchCV)
    assert isinstance(random, RandomizedSearchCV)
    assert random.n_iter == 2
