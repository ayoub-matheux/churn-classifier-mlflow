"""Training and hyperparameter tuning script with MLflow tracking and model registry."""

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
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.pipeline import build_pipeline
from src.utils import load_and_clean_data, load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train churn classification pipeline with MLflow.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Override model type: 'logreg' or 'random_forest'.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register best model in MLflow Model Registry.",
    )
    return parser.parse_args()


def train(
    config_path: str,
    model_type_override: str | None = None,
    register_model: bool = False,
) -> str:
    """Run training, tuning with GridSearchCV, and log to MLflow.

    Args:
        config_path: Path to configuration file.
        model_type_override: Optional model type to override config.
        register_model: Whether to register the model in the MLflow Model Registry.

    Returns:
        The MLflow run_id of the parent training run.
    """
    config = load_config(config_path)

    model_type = model_type_override or config.get("model", {}).get("type", "logreg")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.get("mlflow", {}).get("tracking_uri", "mlruns"))
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", config.get("mlflow", {}).get("experiment_name", "churn-classifier")
    )
    registered_model_name = config.get("mlflow", {}).get("registered_model_name", "ChurnClassifier")
    target_stage = config.get("mlflow", {}).get("stage", "Staging")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment: {experiment_name}")
    print(f"Selected Model Type: {model_type}")

    # Load and clean dataset
    csv_path = config["data"]["csv_path"]
    target_col = config["data"]["target"]
    download_url = config["data"].get("download_url")

    X, y = load_and_clean_data(csv_path, target_col=target_col, download_url=download_url)
    print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # Features
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    # Stratified Train/Test Split
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Persist split for evaluation and reproducibility
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # Build Pipeline
    pipeline = build_pipeline(
        numeric=numeric_features,
        categorical=categorical_features,
        model_type=model_type,
        random_state=random_state,
    )

    # Cross-validation and parameter grid
    cv_config = config.get("cv", {})
    n_splits = cv_config.get("n_splits", 5)
    scoring = cv_config.get("scoring", "roc_auc")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    param_grid = {}
    if model_type in config.get("model", {}):
        param_grid = config["model"][model_type].get("param_grid", {})
    elif "params" in config.get("model", {}):
        # Allow prefix-less or prefixed params
        for k, v in config["model"]["params"].items():
            param_key = k if k.startswith("model__") else f"model__{k}"
            param_grid[param_key] = v

    print(f"Grid search param grid: {param_grid}")

    # Autologging
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
    )

    run_name = f"train_{model_type}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run: {run_id}")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_train)

        best_score = float(grid_search.best_score_)
        best_params = grid_search.best_params_
        print(f"Best CV Score ({scoring}): {best_score:.4f}")
        print(f"Best Parameters: {best_params}")

        # Explicitly log best model metrics and params to parent run
        mlflow.log_metric(f"best_cv_{scoring}", best_score)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("cv_strategy", cv_config.get("strategy", "StratifiedKFold"))

        # Save metadata for evaluate.py
        run_metadata = {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "model_type": model_type,
            "best_score": best_score,
            "scoring": scoring,
            "model_uri": f"runs:/{run_id}/model",
        }
        with open("data/processed/latest_run.json", "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2)

        # Optional Model Registry
        if register_model:
            print(f"Registering model under '{registered_model_name}'...")
            model_uri = f"runs:/{run_id}/model"
            reg_model = mlflow.register_model(model_uri, registered_model_name)
            client = MlflowClient()
            version = reg_model.version
            print(f"Model registered as version {version}.")

            # Transition stage / set alias
            try:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=version,
                    stage=target_stage,
                    archive_existing_versions=False,
                )
                print(f"Transitioned version {version} to stage '{target_stage}'.")
            except Exception as e:  # noqa: BLE001
                print(f"Stage transition notice: {e}")

            try:
                client.set_registered_model_alias(
                    name=registered_model_name,
                    alias="staging",
                    version=version,
                )
                print(f"Set alias 'staging' for version {version}.")
            except Exception as e:  # noqa: BLE001
                print(f"Alias notice: {e}")

    print("Training and tuning completed successfully.")
    return run_id


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        model_type_override=args.model_type,
        register_model=args.register,
    )

