# Customer Churn Classification with MLflow

An end-to-end MLOps project for binary customer-churn prediction using the IBM Telco Customer Churn dataset. It combines a scikit-learn preprocessing and modelling pipeline with experiment tracking, model registration, evaluation artifacts, batch inference, automated tests, and a Docker runtime.

## What the project does

- Downloads the configured dataset automatically when `data/raw.csv` is missing.
- Cleans the Telco-specific `TotalCharges` column and removes identifier columns.
- Splits data into reproducible, stratified training and test sets.
- Builds a `ColumnTransformer` pipeline:
  - numeric features: median imputation and standard scaling;
  - categorical features: most-frequent imputation and one-hot encoding with unknown-category handling.
- Trains a baseline and tunes either Logistic Regression or Random Forest.
- Tracks parameters, metrics, models, and evaluation artifacts in MLflow.
- Optionally registers the tuned model as `ChurnClassifier` and assigns the `staging` alias.
- Produces batch predictions in CSV format.

## Project layout

```text
.
├── configs/config.yaml       # Data, features, tuning, and MLflow settings
├── src/
│   ├── pipeline.py           # Preprocessing and estimator pipeline factory
│   ├── train.py              # Baseline, tuning, MLflow logging, optional registry
│   ├── evaluate.py           # Test metrics and diagnostic artifacts
│   ├── predict.py            # Batch-inference CLI
│   └── utils.py              # Data, configuration, and plotting helpers
├── tests/                    # Unit and integration tests
├── Dockerfile
├── docker-entrypoint.sh
├── Makefile
├── requirements.txt
└── .env.example
```

Runtime data, reports, local MLflow storage, and virtual environments are intentionally ignored by Git.

## Requirements

- Python 3.11 is used by the Docker image; Python 3.10+ is recommended locally.
- `make` is optional.
- Docker is optional for containerised execution.

## Installation

Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional MLflow environment overrides can be created from the supplied template:

```powershell
Copy-Item .env.example .env
```

The scripts load `.env` from the repository root. `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME`, when set, override the values in `configs/config.yaml`.

## Configuration

All workflow settings are in [configs/config.yaml](configs/config.yaml):

- `data`: source path, download URL, target, split size, and random seed;
- `features`: numeric and categorical input columns;
- `model`: default model and each model's search space;
- `cv`: cross-validation strategy, scoring, parallelism, and search budget;
- `mlflow`: tracking URI, experiment name, registered model name, and target stage.

The default configuration uses `StratifiedKFold`, five folds, ROC-AUC scoring, and a 30-candidate random search. Supported model types are `logreg` and `random_forest`. Supported CV strategies are `StratifiedKFold`, `KFold`, and `RepeatedStratifiedKFold`.

## Run the workflow

### Train and tune

This downloads the dataset if necessary, writes the stratified split to `data/processed/`, logs a baseline and a tuned run, and records the most recent tuned run in `data/processed/latest_run.json`.

```powershell
python src/train.py --config configs/config.yaml
```

Register the tuned model in MLflow:

```powershell
python src/train.py --config configs/config.yaml --register
```

Train the Random Forest instead of the configured default:

```powershell
python src/train.py --config configs/config.yaml --model-type random_forest --register
```

### Evaluate

Evaluation loads the model from the latest run (or from `--run-id`), scores it on the held-out test set, saves artifacts under `reports/`, and logs them to that MLflow run.

```powershell
python src/evaluate.py --config configs/config.yaml
python src/evaluate.py --config configs/config.yaml --run-id <MLFLOW_RUN_ID>
```

Metrics are ROC-AUC, accuracy, precision, recall, F1, and log loss. Artifacts include ROC and precision-recall curves, a confusion matrix, feature importance or coefficient plot, test predictions, and a classification report.

### Predict

By default, prediction uses the model URI stored in `data/processed/latest_run.json`.

```powershell
python src/predict.py --input data/raw.csv --output data/predictions.csv --config configs/config.yaml
```

To load a registered model explicitly:

```powershell
python src/predict.py --input data/raw.csv --output data/predictions.csv --model-uri "models:/ChurnClassifier@staging"
```

The input CSV must include the feature columns defined in the configuration. It may also include `Churn` and an ID column; these are removed before inference. The output preserves the input columns and adds `churn_prediction` and `churn_probability`.

### View experiments

With the default local SQLite backend:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1
```

Open `http://localhost:5000` in a browser.

## Make targets

```text
make init       # Create .venv and install dependencies
make train      # Tune Logistic Regression and register the model
make train-rf   # Tune Random Forest and register the model
make evaluate   # Evaluate the latest trained model
make predict    # Run batch prediction on data/raw.csv
make test       # Run the test suite
make lint       # Run Ruff
make ui         # Start the MLflow UI
```

## Tests and linting

Run only the repository test suite (this avoids collecting unrelated local directories):

```powershell
python -m pytest -v tests
python -m ruff check src tests
```

The integration test creates its own small dataset and isolated MLflow SQLite store, then exercises training, evaluation, and prediction end to end.

## Docker

Docker lets you run the project with its Python version and dependencies already installed; a local virtual environment is not needed. Install and start Docker Desktop first, then run the following commands **from the repository root**.

### 1. Build the image

This creates a local image named `churn-classifier:latest`. Rebuild it after changing the source code, dependencies, or Dockerfile.

```bash
docker build -t churn-classifier:latest .
```

### 2. Choose a workflow

The container entry point accepts `train` (the default), `evaluate`, or `predict`. Additional options are passed directly to the corresponding Python script. Running without a mounted volume is useful for a quick test, but generated data, MLflow runs, reports, and predictions are removed with the container.

```bash
docker run --rm churn-classifier:latest train
docker run --rm churn-classifier:latest evaluate
docker run --rm churn-classifier:latest predict --input data/raw.csv --output data/predictions.csv
```

`train` downloads the dataset if it is not already present, trains and logs a model. `evaluate` needs a previous training run, and `predict` needs either the latest training-run metadata or an explicit `--model-uri`. Therefore, use the persistent setup below when chaining commands.

### 3. Persist data and MLflow runs

Bind mounts connect folders on your machine to folders inside the container:

- `data` keeps the downloaded dataset, processed files, and prediction CSVs;
- `mlflow` keeps the local MLflow SQLite database and artifacts;
- `reports` keeps evaluation plots and reports.

The `MLFLOW_TRACKING_URI` must use the **container path** (`/app/mlflow/mlflow.db`), not a path from the host machine. On Linux or macOS, train with:

```bash
docker run --rm \
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/mlflow:/app/mlflow" \
  -v "$(pwd)/reports:/app/reports" \
  churn-classifier:latest train
```

On Windows PowerShell, use `${PWD}` and backticks for line continuation:

```powershell
docker run --rm `
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/mlflow:/app/mlflow" `
  -v "${PWD}/reports:/app/reports" `
  churn-classifier:latest train
```

Run evaluation against that same persistent store (use the same mounts and tracking URI):

```powershell
docker run --rm `
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/mlflow:/app/mlflow" `
  -v "${PWD}/reports:/app/reports" `
  churn-classifier:latest evaluate
```

For a batch prediction, add the workflow and its arguments after the image name. The result will be written to `data/predictions.csv` on the host:

```powershell
docker run --rm `
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/mlflow:/app/mlflow" `
  churn-classifier:latest predict --input data/raw.csv --output data/predictions.csv
```

To inspect the locally persisted experiments, start the UI on the host after installing the Python dependencies, using the command in [View experiments](#view-experiments). Alternatively, run an MLflow tracking server that is reachable from the container.

For shared or production tracking, set `MLFLOW_TRACKING_URI` to a reachable MLflow tracking server rather than using SQLite in the container.
