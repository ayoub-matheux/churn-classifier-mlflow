"""Utility functions for data loading, preprocessing, plotting, and configuration."""

import os
import ssl
import urllib.request
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing parsed configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_data_if_missing(csv_path: str, url: str) -> None:
    """Download dataset if the file does not already exist locally.

    Args:
        csv_path: Destination local path for the CSV file.
        url: URL to download the raw dataset from.
    """
    if os.path.exists(csv_path):
        return

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    print(f"Downloading dataset from {url} to {csv_path}...")
    try:
        urllib.request.urlretrieve(url, csv_path)
    except Exception:  # noqa: BLE001
        # Fallback with unverified context in case of corporate SSL interception
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, open(
            csv_path, "wb"
        ) as out_file:
            out_file.write(response.read())
    print("Dataset downloaded successfully.")


def clean_churn_dataframe(
    df: pd.DataFrame, target_col: str = "Churn"
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the raw Telco customer churn dataset and separate features and target.

    Args:
        df: Raw input DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    data = df.copy()

    # Case-insensitive search for target column if needed
    matched_target = None
    for col in data.columns:
        if col.strip().lower() == target_col.strip().lower():
            matched_target = col
            break

    if not matched_target:
        raise KeyError(
            f"Target column '{target_col}' not found in DataFrame columns: {list(data.columns)}"
        )

    # Encode target column to binary 0/1
    target_series = data[matched_target]
    if pd.api.types.is_numeric_dtype(target_series):
        cleaned_target = target_series.astype(int)
    else:
        mapping = {
            "Yes": 1, "yes": 1, "1": 1, "True": 1, "true": 1,
            "No": 0, "no": 0, "0": 0, "False": 0, "false": 0,
        }
        cleaned_target = (
            target_series.astype(str)
            .str.strip()
            .map(mapping)
        )
        if cleaned_target.isna().any():
            cleaned_target = cleaned_target.fillna(0)
        cleaned_target = cleaned_target.astype(int)

    # Clean numeric features with empty spaces (classic Telco churn dataset issue)
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Drop ID and target columns from feature set
    cols_to_drop = [matched_target]
    for id_col in ["customerID", "CustomerID", "id", "ID"]:
        if id_col in data.columns:
            cols_to_drop.append(id_col)

    features_df = data.drop(columns=cols_to_drop)

    return features_df, cleaned_target


def load_and_clean_data(
    csv_path: str, target_col: str = "Churn", download_url: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset from path (downloading if absent) and clean it.

    Args:
        csv_path: Local path to the CSV file.
        target_col: Name of the target column.
        download_url: Optional URL to download the dataset if missing.

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    if not os.path.exists(csv_path):
        if download_url:
            download_data_if_missing(csv_path, download_url)
        else:
            raise FileNotFoundError(f"Data file not found at: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    return clean_churn_dataframe(raw_df, target_col=target_col)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str) -> str:
    """Plot and save ROC curve.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probability for the positive class.
        output_path: Filepath where plot should be saved.

    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot(ax=plt.gca(), name="Classifier")
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str) -> str:
    """Plot and save Precision-Recall curve.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probability for the positive class.
        output_path: Filepath where plot should be saved.

    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot(ax=plt.gca(), name="Classifier")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> str:
    """Plot and save confusion matrix heatmap.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        output_path: Filepath where plot should be saved.

    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn (0)", "Churn (1)"],
        yticklabels=["No Churn (0)", "Churn (1)"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_feature_importance(
    pipeline: Any,
    numeric_cols: list[str],
    categorical_cols: list[str],
    output_path: str,
    top_n: int = 15,
) -> str | None:
    """Plot and save top feature importances or model coefficients.

    Args:
        pipeline: Fitted sklearn Pipeline.
        numeric_cols: List of numeric feature names.
        categorical_cols: List of categorical feature names.
        output_path: Filepath where plot should be saved.
        top_n: Number of top features to display.

    Returns:
        output_path if successful, else None.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        estimator = getattr(pipeline, "best_estimator_", pipeline)
        preprocessor = estimator.named_steps["pre"]
        cat_ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        ohe_cols = list(cat_ohe.get_feature_names_out(categorical_cols))
        feature_names = list(numeric_cols) + ohe_cols

        model = estimator.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
            metric_label = "Feature Importance"
        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_[0])
            metric_label = "Absolute Coefficient Magnitude"
        else:
            return None

        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": values})
            .sort_values(by="importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=fi_df,
            x="importance",
            y="feature",
            palette="viridis",
            hue="feature",
            legend=False,
        )
        plt.title(f"Top {top_n} Features ({metric_label})")
        plt.xlabel(metric_label)
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Could not plot feature importance: {exc}")
        return None

