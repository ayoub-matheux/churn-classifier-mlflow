"""Batch inference script for churn prediction using a saved MLflow model."""

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

import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils import load_config

load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for batch prediction."""
    parser = argparse.ArgumentParser(description="Run batch inference with trained churn model.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file containing features.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/predictions.csv",
        help="Path where output predictions CSV should be saved.",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="MLflow model URI (e.g., 'runs:/<run_id>/model' or 'models:/ChurnClassifier/staging').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def predict(
    input_path: str,
    output_path: str,
    model_uri: str | None = None,
    config_path: str = "configs/config.yaml",
) -> str:
    """Load model and perform batch prediction on input data.

    Args:
        input_path: Path to input CSV.
        output_path: Path to output predictions CSV.
        model_uri: MLflow model URI. If None, reads from latest_run.json.
        config_path: Path to YAML config.

    Returns:
        The output CSV path.
    """
    config = load_config(config_path)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.get("mlflow", {}).get("tracking_uri", "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)

    if model_uri is None:
        metadata_path = "data/processed/latest_run.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_uri = meta.get("model_uri")
        else:
            model_name = config.get("mlflow", {}).get("registered_model_name", "ChurnClassifier")
            model_uri = f"models:/{model_name}/staging"

    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    print(f"Reading input data from: {input_path}")
    df = pd.read_csv(input_path)

    # Pre-clean known tabular issues like TotalCharges whitespace
    features = df.copy()
    if "TotalCharges" in features.columns:
        features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")

    # Drop target or ID column if present in input
    for col in ["Churn", "churn", "customerID", "CustomerID", "id", "ID"]:
        if col in features.columns:
            features = features.drop(columns=[col])

    # Run predictions
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    output_df = df.copy()
    output_df["churn_prediction"] = predictions
    output_df["churn_probability"] = np.round(probabilities, 4)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions successfully saved to {output_path} ({len(output_df)} rows).")
    return output_path


if __name__ == "__main__":
    args = parse_args()
    predict(
        input_path=args.input,
        output_path=args.output,
        model_uri=args.model_uri,
        config_path=args.config,
    )
