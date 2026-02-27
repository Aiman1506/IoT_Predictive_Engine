"""
FactoryGuard AI - Model Evaluation
====================================
STEP 6: Comprehensive evaluation with focus on Precision-Recall
metrics (NOT Accuracy) due to extreme class imbalance.

Outputs:
  - Classification report (Precision, Recall, F1)
  - Confusion matrix
  - Precision-Recall curve plot
  - False positive / false negative analysis
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    feature_names: list | None = None,
    save_plots: bool = True,
) -> Dict[str, float]:
    """
    Run full evaluation suite on a trained model.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    X_test, y_test : array-like
    model_name : str
        Label for plots and logs.
    feature_names : list[str], optional
    save_plots : bool
        Whether to save plots to disk.

    Returns
    -------
    dict
        Evaluation metrics: precision, recall, f1, pr_auc.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # --- Metrics -------------------------------------------------------------
    pr_auc = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
    }

    logger.info("=" * 60)
    logger.info("EVALUATION: %s", model_name)
    logger.info("=" * 60)
    logger.info("  PR-AUC     : %.4f", pr_auc)
    logger.info("  Precision  : %.4f", precision)
    logger.info("  Recall     : %.4f", recall)
    logger.info("  F1-Score   : %.4f", f1)

    # --- Classification report -----------------------------------------------
    report = classification_report(
        y_test, y_pred, target_names=["Normal", "Failure"], zero_division=0
    )
    logger.info("\n%s", report)

    # --- Confusion matrix ----------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    logger.info("Confusion Matrix:")
    logger.info("  TN=%d  FP=%d", tn, fp)
    logger.info("  FN=%d  TP=%d", fn, tp)
    logger.info(
        "  False alarm rate (FPR): %.4f",
        fp / max(fp + tn, 1)
    )
    logger.info(
        "  Missed failure rate (FNR): %.4f",
        fn / max(fn + tp, 1)
    )

    if save_plots:
        _plot_confusion_matrix(cm, model_name)
        _plot_precision_recall_curve(y_test, y_proba, pr_auc, model_name)

    return metrics


def _plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = OUTPUT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", path)


def _plot_precision_recall_curve(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    pr_auc: float,
    model_name: str,
) -> None:
    """Save Precision-Recall curve."""
    precision_vals, recall_vals, thresholds = precision_recall_curve(
        y_test, y_proba
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, linewidth=2,
            label=f"PR-AUC = {pr_auc:.4f}")
    ax.fill_between(recall_vals, precision_vals, alpha=0.2)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    path = OUTPUT_DIR / f"pr_curve_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved: %s", path)


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table of all model metrics.

    Parameters
    ----------
    results : dict
        {model_name: {metric_name: value, ...}, ...}

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by PR-AUC descending.
    """
    df = pd.DataFrame(results).T
    df = df.sort_values("pr_auc", ascending=False)
    logger.info("\nModel Comparison:\n%s", df.to_string())
    return df
