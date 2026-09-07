# 📊 Churn Classifier with Pipelines + MLflow

An end-to-end, production-ready MLOps project demonstrating reproducible tabular classification using **scikit-learn** pipelines (`Pipeline` + `ColumnTransformer`) and comprehensive experiment tracking, model tuning, artifact logging, and model registry with **MLflow**.

---

## 📁 Repository Structure

```text
Project/
├─ configs/
│  └─ config.yaml             # Central configuration (data, features, model grids, CV, MLflow)
├─ data/                      # Raw and processed datasets (gitignored)
│  ├─ raw.csv                 # Telco Customer Churn dataset
│  └─ processed/              # Stratified train/test splits & latest run metadata
├─ src/
│  ├─ __init__.py
│  ├─ pipeline.py             # Modular ColumnTransformer + classifier pipeline constructor
│  ├─ train.py                # GridSearchCV hyperparameter tuning + MLflow autolog + Model Registry
│  ├─ evaluate.py             # Test split evaluation + diagnostic artifact generation & logging
│  ├─ predict.py              # CLI for batch inference on new/unseen CSV data
│  └─ utils.py                # Data loading, cleaning, plotting, and config utilities
├─ tests/
│  ├─ __init__.py
│  └─ test_pipeline.py        # Sanity tests for pipeline architecture, fitting, and data cleaning
├─ reports/                   # Saved evaluation plots and predictions
├─ Dockerfile                 # Containerized runtime for training and inference
├─ Makefile                   # Automation shortcuts for workflow tasks
├─ requirements.txt           # Python dependencies
├─ .env.example               # Template for MLflow environment variables
├─ .gitignore                 # Standard ML / Python ignore patterns
└─ README.md                  # Project documentation
```

---

## 🚀 Key Features

1. **Robust Feature Preprocessing**:
   - **Numerical Pipeline**: Median imputation (`SimpleImputer`) + feature standardization (`StandardScaler`).
   - **Categorical Pipeline**: Most frequent imputation (`SimpleImputer`) + one-hot encoding (`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`).
   - **Data Cleaning**: Automatically handles the Telco churn dataset quirk where empty strings in `TotalCharges` for new customers (`tenure == 0`) are converted to `NaN` and imputed gracefully.

2. **MLflow Tracking & Autologging**:
   - Automatically logs estimator parameters, model schemas, signatures, and input examples with `mlflow.sklearn.autolog`.
   - Organizes runs under hierarchical experiments with full parent/child relationship during cross-validation.

3. **Hyperparameter Tuning & Cross-Validation**:
   - Stratified $K$-fold cross-validation ($k=5$) scored on `roc_auc`.
   - Grid search across regularizations (`C`), penalties, solvers for Logistic Regression, and depth / estimators for Random Forest.

4. **Rich Evaluation & Artifact Tracking**:
   - Held-out test set evaluation with ROC-AUC, Accuracy, Precision, Recall, F1-Score, and Log Loss.
   - Generates and logs diagnostic figures directly to MLflow:
     - **ROC Curve** (`roc_curve.png`)
     - **Precision-Recall Curve** (`pr_curve.png`)
     - **Confusion Matrix Heatmap** (`confusion_matrix.png`)
     - **Top Feature Importances / Coefficients** (`feature_importance.png`)
     - **Test Predictions CSV** (`test_predictions.csv`) for residual and error analysis
     - **Classification Report JSON** (`classification_report.json`)

5. **Model Registry & Staging**:
   - Automatically registers best models under `ChurnClassifier`.
   - Supports stage transitions (e.g. `Staging`) and modern MLflow model aliases (`@staging`).

---

## 📈 Benchmark & Experimental Results

Evaluated on the 20% held-out test split (1,409 customers):

| Model | Best CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test F1-Score | Best Parameters |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **`0.8463`** | **`0.8411`** | **`80.48%`** | **`60.32%`** | `C=10.0, solver=liblinear, penalty=l2` |
| **Random Forest** | **`0.8458`** | **`0.8399`** | **`79.35%`** | **`51.74%`** | `n_estimators=200, max_depth=5, min_samples_split=2` |

All diagnostic curves (`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `feature_importance.png`) are saved in `reports/` and logged directly to MLflow.

---

## 🛠️ Setup Instructions (Virtual Environment)

### 1. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Alternatively, run `make init` if `make` is installed).*

---

## 🏃 Quickstart: Training, Evaluation & Tracking

### 1. Train and Tune Models with MLflow

Run hyperparameter tuning for Logistic Regression and register the best model to the Model Registry:

```powershell
python src/train.py --config configs/config.yaml --register
```

Or train a Random Forest classifier:

```powershell
python src/train.py --config configs/config.yaml --model-type random_forest --register
```

### 2. Evaluate Model & Log Diagnostic Artifacts

Evaluate the trained model on the held-out test set and log ROC/PR plots and confusion matrices:

```powershell
python src/evaluate.py --config configs/config.yaml
```

### 3. Launch MLflow UI

View all experiment runs, metrics, parameters, and logged artifacts:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1
```
Open your browser at [http://localhost:5000](http://localhost:5000).

### 4. Run Batch Predictions

Make churn predictions on raw or new customer data using the latest run:

```powershell
python src/predict.py --input data/raw.csv --output data/predictions.csv
```

Or make predictions directly using the **registered staging model**:

```powershell
python src/predict.py --input data/raw.csv --output data/predictions.csv --model-uri "models:/ChurnClassifier@staging"
```

---

## 🧪 Testing and Linting

Run automated unit tests:

```powershell
pytest -v tests/
```

Check code quality with `ruff`:

```powershell
ruff check src tests
```

---

## 🐳 Docker Usage

Build the Docker image:

```bash
docker build -t churn-classifier:latest .
```

Run training inside container:

```bash
docker run --rm -v $(pwd)/mlruns:/app/mlruns churn-classifier:latest
```

---

## ⚙️ Configuration (`configs/config.yaml`)

All parameters are centrally managed in `configs/config.yaml`:
- Dataset path and automatic download URL.
- Numeric and categorical feature definitions.
- Model hyperparameters search space for `logreg` and `random_forest`.
- Cross-validation split count and evaluation metrics.
- MLflow experiment name and registered model naming.

