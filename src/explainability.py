"""
FactoryGuard AI - Model Explainability (SHAP)
===============================================
STEP 7: Generate global and local SHAP explanations
so maintenance engineers understand *why* the model predicts failure.

Uses TreeExplainer for LightGBM/XGBoost models.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def explain_model(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    feature_names: List[str],
    n_samples: int = 500,
    save_plots: bool = True,
) -> dict:
    """
    Generate SHAP explanations for the production model.

    Parameters
    ----------
    model : fitted tree-based model (LightGBM / XGBoost)
    X_test : array-like
        Test features.
    feature_names : list[str]
        Feature column names.
    n_samples : int
        Number of samples to explain (for performance).
    save_plots : bool
        Whether to save SHAP plots.

    Returns
    -------
    dict with keys:
        - shap_values: SHAP values array
        - feature_importance: sorted feature importance DataFrame
    """
    import shap

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame for interpretability
    if isinstance(X_test, np.ndarray):
        X_explain = pd.DataFrame(X_test[:n_samples], columns=feature_names)
    else:
        X_explain = X_test.iloc[:n_samples].copy()
        X_explain.columns = feature_names

    logger.info("Computing SHAP values for %d samples...", len(X_explain))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # For binary classification, some explainers return a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # --- Global Feature Importance -------------------------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info("\nTop 10 Features by SHAP Importance:")
    for _, row in importance_df.head(10).iterrows():
        logger.info("  %-40s  %.4f", row["feature"], row["mean_abs_shap"])

    if save_plots:
        _plot_global_importance(shap_values, X_explain, feature_names)
        _plot_local_explanation(shap_values, X_explain, feature_names, explainer)

    return {
        "shap_values": shap_values,
        "feature_importance": importance_df,
    }


def _plot_global_importance(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    feature_names: List[str],
) -> None:
    """Save SHAP summary (beeswarm) plot."""
    import shap

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title("Global Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    path = OUTPUT_DIR / "shap_global_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP global importance plot saved: %s", path)


def _plot_local_explanation(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    feature_names: List[str],
    explainer: Any,
) -> None:
    """
    Generate a local SHAP explanation for the highest-risk prediction.
    This is the kind of explanation a maintenance engineer needs:
    "Why did the model predict failure for THIS reading?"
    """
    import shap

    # Find the sample with the highest predicted failure probability
    # (highest sum of SHAP values = highest risk)
    risk_scores = shap_values.sum(axis=1)
    high_risk_idx = int(np.argmax(risk_scores))

    logger.info(
        "Generating local explanation for sample %d (highest risk)",
        high_risk_idx
    )

    # Generate human-readable explanation
    sample_shap = shap_values[high_risk_idx]
    sample_features = X_explain.iloc[high_risk_idx]

    top_contributing = np.argsort(np.abs(sample_shap))[::-1][:5]

    explanation_parts = []
    for idx in top_contributing:
        feat_name = feature_names[idx]
        feat_val = sample_features.iloc[idx]
        shap_val = sample_shap[idx]
        direction = "increased" if shap_val > 0 else "decreased"
        explanation_parts.append(
            f"  • {feat_name} = {feat_val:.2f} ({direction} failure risk "
            f"by {abs(shap_val):.4f})"
        )

    explanation = (
        "FAILURE PREDICTION EXPLANATION:\n"
        "The model predicted high failure risk because:\n"
        + "\n".join(explanation_parts)
    )
    logger.info("\n%s", explanation)

    # Save waterfall plot
    try:
        fig = plt.figure(figsize=(12, 6))
        explanation_obj = shap.Explanation(
            values=shap_values[high_risk_idx],
            base_values=explainer.expected_value if not isinstance(
                explainer.expected_value, list
            ) else explainer.expected_value[1],
            data=X_explain.iloc[high_risk_idx].values,
            feature_names=feature_names,
        )
        shap.waterfall_plot(explanation_obj, show=False, max_display=12)
        path = OUTPUT_DIR / "shap_local_explanation.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP local explanation plot saved: %s", path)
    except Exception as e:
        logger.warning("Could not save waterfall plot: %s", e)

    # Save explanation text
    text_path = OUTPUT_DIR / "failure_explanation.txt"
    text_path.write_text(explanation, encoding="utf-8")
    logger.info("Explanation text saved: %s", text_path)
