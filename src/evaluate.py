"""Evaluation module for computing test metrics and logging diagnostic artifacts to MLflow."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import (
    load_and_clean_data,
    load_config,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained churn model on test set.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional MLflow run ID to evaluate and log artifacts to.",
    )
    return parser.parse_args()


def evaluate(config_path: str = "configs/config.yaml", run_id: str | None = None) -> dict[str, float]:
    """Evaluate trained model on test data, generate visual plots, and log to MLflow.

    Args:
        config_path: Path to YAML config.
        run_id: Optional MLflow run ID.

    Returns:
        Dictionary of computed test metrics.
    """
    config = load_config(config_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.get("mlflow", {}).get("tracking_uri", "mlruns"))
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", config.get("mlflow", {}).get("experiment_name", "churn-classifier")
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Determine run_id and model_uri
    metadata_path = "data/processed/latest_run.json"
    if run_id is None:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                run_id = meta.get("run_id")
        else:
            raise FileNotFoundError(
                "No run_id specified and 'data/processed/latest_run.json' was not found. "
                "Please run train.py first or specify --run-id."
            )

    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from MLflow run: {run_id} ({model_uri})")
    model = mlflow.sklearn.load_model(model_uri)

    # Load test split
    x_test_path = "data/processed/X_test.csv"
    y_test_path = "data/processed/y_test.csv"

    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")
    else:
        print("Processed test set not found. Re-splitting from raw data...")
        from sklearn.model_selection import train_test_split

        X, y = load_and_clean_data(
            config["data"]["csv_path"],
            target_col=config["data"]["target"],
            download_url=config["data"].get("download_url"),
        )
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=config["data"].get("test_size", 0.2),
            random_state=config["data"].get("random_state", 42),
            stratify=y,
        )

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_log_loss": float(log_loss(y_test, y_prob)),
    }

    print("\n--- Test Set Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")

    # Generate and save diagnostic plots
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    roc_path = plot_roc_curve(y_test.to_numpy(), y_prob, os.path.join(reports_dir, "roc_curve.png"))
    pr_path = plot_pr_curve(y_test.to_numpy(), y_prob, os.path.join(reports_dir, "pr_curve.png"))
    cm_path = plot_confusion_matrix(
        y_test.to_numpy(), y_pred, os.path.join(reports_dir, "confusion_matrix.png")
    )

    numeric_cols = config["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]
    fi_path = plot_feature_importance(
        model,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        output_path=os.path.join(reports_dir, "feature_importance.png"),
    )

    # Save detailed predictions for error analysis
    pred_df = X_test.copy()
    pred_df["actual"] = y_test
    pred_df["predicted"] = y_pred
    pred_df["churn_probability"] = np.round(y_prob, 4)
    pred_path = os.path.join(reports_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # Save classification report as JSON
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_json_path = os.path.join(reports_dir, "classification_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(clf_report, f, indent=2)

    # Log metrics and artifacts back to the parent MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(roc_path, artifact_path="evaluation")
        mlflow.log_artifact(pr_path, artifact_path="evaluation")
        mlflow.log_artifact(cm_path, artifact_path="evaluation")
        if fi_path and os.path.exists(fi_path):
            mlflow.log_artifact(fi_path, artifact_path="evaluation")
        mlflow.log_artifact(pred_path, artifact_path="evaluation")
        mlflow.log_artifact(report_json_path, artifact_path="evaluation")

    print(f"\nArtifacts successfully logged to MLflow run {run_id} under 'evaluation/'.")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    evaluate(config_path=args.config, run_id=args.run_id)

