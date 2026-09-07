# Makefile for Churn Classifier with Pipelines and MLflow

PY ?= python
ENV ?= .venv
EXP ?= churn-classifier
CONFIG ?= configs/config.yaml

# On Windows virtualenv python is in .venv/Scripts/python, on Linux/macOS in .venv/bin/python
VENV_BIN = $(wildcard $(ENV)/Scripts/python.exe)
ifeq ($(VENV_BIN),)
    VENV_PY = $(wildcard $(ENV)/bin/python)
    ifeq ($(VENV_PY),)
        PYTHON = $(PY)
        PYTEST = pytest
        RUFF = ruff
        MLFLOW = mlflow
    else
        PYTHON = $(ENV)/bin/python
        PYTEST = $(ENV)/bin/pytest
        RUFF = $(ENV)/bin/ruff
        MLFLOW = $(ENV)/bin/mlflow
    endif
else
    PYTHON = $(ENV)/Scripts/python.exe
    PYTEST = $(ENV)/Scripts/pytest.exe
    RUFF = $(ENV)/Scripts/ruff.exe
    MLFLOW = $(ENV)/Scripts/mlflow.exe
endif

.PHONY: help init install train train-rf evaluate test lint ui predict clean

help:
	@echo "Available targets:"
	@echo "  init       : Create virtual environment and install requirements"
	@echo "  install    : Install dependencies in active environment"
	@echo "  train      : Run training and hyperparameter tuning for Logistic Regression"
	@echo "  train-rf   : Run training and hyperparameter tuning for Random Forest"
	@echo "  evaluate   : Evaluate the latest model and log artifacts to MLflow"
	@echo "  predict    : Run batch inference on raw dataset"
	@echo "  test       : Run pytest suite"
	@echo "  lint       : Check code style with ruff"
	@echo "  ui         : Launch MLflow UI tracking dashboard (http://localhost:5000)"
	@echo "  clean      : Remove temporary caches and outputs"

init:
	$(PY) -m venv $(ENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) src/train.py --config $(CONFIG) --register

train-rf:
	$(PYTHON) src/train.py --config $(CONFIG) --model-type random_forest --register

evaluate:
	$(PYTHON) src/evaluate.py --config $(CONFIG)

predict:
	$(PYTHON) src/predict.py --input data/raw.csv --output data/predictions.csv --config $(CONFIG)

test:
	$(PYTEST) -v tests/

lint:
	$(RUFF) check src tests

ui:
	$(MLFLOW) ui --port 5000

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache .ruff_cache

