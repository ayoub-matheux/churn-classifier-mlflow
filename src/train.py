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
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from src.pipeline import build_pipeline
from src.utils import load_and_clean_data, load_config

load_dotenv(PROJECT_ROOT / ".env")


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


def make_cv_splitter(cv_config: dict, random_state: int):
    """Build the configured cross-validation splitter.

    The names deliberately match the names exposed in ``config.yaml`` so a
    misspelling fails before an expensive training job starts.
    """
    strategy = cv_config.get("strategy", "StratifiedKFold")
    n_splits = cv_config.get("n_splits", 5)
    shuffle = cv_config.get("shuffle", True)
    random_state_arg = random_state if shuffle else None

    if strategy == "StratifiedKFold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state_arg)
    if strategy == "KFold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state_arg)
    if strategy == "RepeatedStratifiedKFold":
        return RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=cv_config.get("n_repeats", 2),
            random_state=random_state,
        )

    supported = "StratifiedKFold, KFold, RepeatedStratifiedKFold"
    raise ValueError(f"Unknown cv.strategy '{strategy}'. Supported strategies: {supported}.")


def make_search(estimator, param_space: dict, cv, cv_config: dict, scoring: str, random_state: int):
    """Create the configured hyperparameter-search strategy.

    Random search is preferable for the wider production search spaces: it
    samples a fixed budget rather than exhaustively evaluating an expensive
    Cartesian product.  Grid search remains available for small, deliberate
    experiments and backwards compatibility.
    """
    search_type = cv_config.get("search_type", "grid").lower()
    common_args = {
        "estimator": estimator,
        "cv": cv,
        "scoring": scoring,
        "n_jobs": cv_config.get("n_jobs", -1),
        "refit": True,
        "return_train_score": False,
    }
    if search_type == "grid":
        return GridSearchCV(param_grid=param_space, **common_args)
    if search_type == "random":
        return RandomizedSearchCV(
            param_distributions=param_space,
            n_iter=cv_config.get("n_iter", 30),
            random_state=random_state,
            **common_args,
        )

    raise ValueError("Unknown cv.search_type " f"'{search_type}'. Supported values: grid, random.")


def train(
    config_path: str,
    model_type_override: str | None = None,
    register_model: bool = False,
) -> str:
    """Train a baseline, tune it, and log both runs to MLflow.

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
    scoring = cv_config.get("scoring", "roc_auc")
    cv = make_cv_splitter(cv_config, random_state)

    param_grid = {}
    if model_type in config.get("model", {}):
        param_grid = config["model"][model_type].get("param_grid", {})
    elif "params" in config.get("model", {}):
        # Allow prefix-less or prefixed params
        for k, v in config["model"]["params"].items():
            param_key = k if k.startswith("model__") else f"model__{k}"
            param_grid[param_key] = v

    search_type = cv_config.get("search_type", "grid")
    print(f"{search_type.title()} search parameter space: {param_grid}")

    # Log the single selected estimator ourselves. This avoids a large number
    # of implicit child runs from GridSearchCV and keeps ``runs:/.../model``
    # stable for evaluate.py and predict.py.
    mlflow.sklearn.autolog(
        log_models=False,
        log_input_examples=True,
        log_model_signatures=True,
    )

    # A genuine baseline: default pipeline, fitted once on exactly the same
    # train split as the tuned candidate and scored on the held-out split.
    baseline_pipeline = build_pipeline(
        numeric=numeric_features,
        categorical=categorical_features,
        model_type=model_type,
        random_state=random_state,
    )
    with mlflow.start_run(run_name=f"baseline_{model_type}") as baseline_run:
        baseline_pipeline.fit(X_train, y_train)
        baseline_auc = float(roc_auc_score(y_test, baseline_pipeline.predict_proba(X_test)[:, 1]))
        mlflow.log_metric("validation_roc_auc", baseline_auc)
        mlflow.set_tag("run_type", "baseline")
        mlflow.set_tag("model_type", model_type)
        # MLflow 3 defaults to the safer ``skops`` format. The current
        # sklearn pipeline includes numpy.dtype, so record that precise,
        # known-safe type as trusted for serialization and later loading.
        mlflow.sklearn.log_model(
            baseline_pipeline,
            artifact_path="model",
            skops_trusted_types=["numpy.dtype"],
        )
        baseline_run_id = baseline_run.info.run_id
        print(f"Baseline validation ROC-AUC: {baseline_auc:.4f}")

    run_name = f"tuned_{model_type}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run: {run_id}")

        grid_search = make_search(pipeline, param_grid, cv, cv_config, scoring, random_state)

        grid_search.fit(X_train, y_train)

        best_score = float(grid_search.best_score_)
        best_score_std = float(grid_search.cv_results_["std_test_score"][grid_search.best_index_])
        best_params = grid_search.best_params_
        print(f"Best CV Score ({scoring}): {best_score:.4f} (+/- {best_score_std:.4f})")
        print(f"Best Parameters: {best_params}")

        tuned_auc = float(roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1]))
        auc_improvement = tuned_auc - baseline_auc

        # Explicitly log best model metrics and params to parent run
        mlflow.log_metric(f"best_cv_{scoring}", best_score)
        mlflow.log_metric(f"best_cv_{scoring}_std", best_score_std)
        mlflow.log_metric("search_candidates", len(grid_search.cv_results_["params"]))
        mlflow.log_metric("validation_roc_auc", tuned_auc)
        mlflow.log_metric("baseline_validation_roc_auc", baseline_auc)
        mlflow.log_metric("tuned_auc_minus_baseline_auc", auc_improvement)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("cv_strategy", cv_config.get("strategy", "StratifiedKFold"))
        mlflow.set_tag("search_type", search_type)
        mlflow.set_tag("run_type", "tuned")
        mlflow.set_tag("baseline_run_id", baseline_run_id)
        mlflow.sklearn.log_model(
            grid_search.best_estimator_,
            artifact_path="model",
            skops_trusted_types=["numpy.dtype"],
        )
        print(f"Tuned validation ROC-AUC: {tuned_auc:.4f} (delta: {auc_improvement:+.4f})")

        # Save metadata for evaluate.py
        run_metadata = {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "model_type": model_type,
            "best_score": best_score,
            "best_score_std": best_score_std,
            "baseline_run_id": baseline_run_id,
            "baseline_auc": baseline_auc,
            "tuned_auc": tuned_auc,
            "tuned_auc_minus_baseline_auc": auc_improvement,
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
