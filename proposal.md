
#  “Churn Classifier with Pipelines + MLflow”


Build a robust tabular classification pipeline (scikit-learn `Pipeline` + `ColumnTransformer`) and **track everything with MLflow** (params, metrics, artifacts, and model registry).
Choose one of these small, public, easy-to-use datasets (CSV):

* **Telco Customer Churn** (binary classification, mixed categorical/numeric)
* **Bank Marketing** (deposit yes/no)
* **Adult Income** (>=50K vs <50K)

> Target skills: reproducible training with Pipelines, feature preprocessing, cross-validation & tuning, MLflow tracking/registry, CLI + config, Makefile, and (optional) Docker.

---

## What you’ll deliver

**Repo layout**

```
mlops-project/
├─ data/                         # raw & processed (gitignored)
├─ configs/
│  └─ config.yaml                # paths, target, split, model/type/params
├─ src/
│  ├─ pipeline.py                # build ColumnTransformer + model
│  ├─ train.py                   # train + tune + MLflow autolog + registry
│  ├─ evaluate.py                # final eval + plot & log artifacts
│  └─ utils.py                   # small helpers (split, plotting, etc.)
├─ tests/
│  └─ test_pipeline.py           # sanity checks for pipeline
├─ Makefile
├─ requirements.txt (or pyproject.toml)
├─ .env.example                  # MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
└─ README.md
```

**Makefile (example)**

```make
PY=python
ENV?=.venv
EXP?=churn-exp

init:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt


train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/train.py --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/evaluate.py --config configs/config.yaml


test:
	pytest -q
```

**configs/config.yaml (example)**

```yaml
data:
  csv_path: data/raw.csv
  target: churn
  test_size: 0.2
  random_state: 42

features:
  numeric: ["tenure", "MonthlyCharges", "TotalCharges"]
  categorical: ["gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "InternetService", "Contract",
                "PaperlessBilling", "PaymentMethod"]

model:
  type: "logreg"  # or "random_forest"
  params:
    C: [0.1, 1.0, 10.0]
    penalty: ["l2"]
    solver: ["lbfgs", "liblinear"]

cv:
  strategy: "StratifiedKFold"
  n_splits: 5
  scoring: "roc_auc"
```

---

## Core code snippets

**src/pipeline.py**

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(numeric, categorical, model_type="logreg"):
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[("num", num, numeric), ("cat", cat, categorical)]
    )
    if model_type == "logreg":
        model = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    else:
        raise ValueError("Unsupported model_type")
    return Pipeline(steps=[("pre", pre), ("model", model)])
```



**tests/test\_pipeline.py**

```python
from src.pipeline import build_pipeline

def test_build_pipeline():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    assert "pre" in dict(pipe.named_steps)
    assert "model" in dict(pipe.named_steps)
```

---

## Tasks

1. **Setup & Data**

   * Create repo, venv, install deps (`scikit-learn`, `pandas`, `mlflow`, `matplotlib`, `pyyaml`, `ruff`, `pytest`).
   * Place CSV in `data/raw.csv`.

2. **Baseline Pipeline**

   * Implement `pipeline.py`.
   * Quick train/test split; single model; baseline ROC-AUC; verify pipeline works.

3. **MLflow Integration + Tuning**

   * Enable `mlflow.autolog`.
   * Add `GridSearchCV` with proper param grid; log best metrics; log best model; check MLflow UI.

4. **Artifacts & Evaluation**

   * Compute & log ROC/PR plots, confusion matrix as artifacts.
   * Save predictions CSV for error analysis (optional).


6. **(Optional) Registry & Docker**

   * Register model to `ChurnClassifier` and set stage “Staging”.
   * Dockerfile to run training or a tiny `predict.py` script for batch inference.
