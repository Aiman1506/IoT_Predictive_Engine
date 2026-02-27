"""
FactoryGuard AI - Model Training Module
========================================
STEP 4: Baseline models  (Logistic Regression, Random Forest)
STEP 5: Production model (LightGBM with hyperparameter tuning)

Key design decisions:
  - All models are evaluated on PR-AUC (NOT accuracy) due to extreme
    class imbalance.
  - Class weights / scale_pos_weight are used to handle imbalance.
  - GridSearchCV uses PR-AUC as the scoring metric.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scorer: Precision-Recall AUC (Average Precision)
# ---------------------------------------------------------------------------
pr_auc_scorer = make_scorer(average_precision_score, response_method="predict_proba")


# ===== STEP 4: Baseline Models ==============================================

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str = "balanced",
) -> LogisticRegression:
    """Train a Logistic Regression baseline with balanced class weights."""
    logger.info("Training Logistic Regression baseline...")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    train_pr_auc = average_precision_score(
        y_train, model.predict_proba(X_train)[:, 1]
    )
    logger.info("Logistic Regression train PR-AUC: %.4f", train_pr_auc)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str = "balanced",
    n_estimators: int = 200,
) -> RandomForestClassifier:
    """Train a Random Forest baseline with balanced class weights."""
    logger.info("Training Random Forest baseline...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_pr_auc = average_precision_score(
        y_train, model.predict_proba(X_train)[:, 1]
    )
    logger.info("Random Forest train PR-AUC: %.4f", train_pr_auc)
    return model


# ===== STEP 5: Production Model =============================================

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str] | None = None,
    tune: bool = True,
) -> Any:
    """
    Train a LightGBM classifier optimized for PR-AUC.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and labels.
    feature_names : list[str], optional
        Column names for interpretability.
    tune : bool
        Whether to perform GridSearchCV hyperparameter tuning.

    Returns
    -------
    LGBMClassifier (best estimator if tuned)
    """
    import lightgbm as lgb

    # Compute scale_pos_weight for imbalanced classes
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    spw = n_neg / n_pos
    logger.info("Class ratio: neg=%d, pos=%d, scale_pos_weight=%.1f",
                n_neg, n_pos, spw)

    base_params = {
        "objective": "binary",
        "scale_pos_weight": spw,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    if not tune:
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            **base_params,
        )
        model.fit(X_train, y_train)
        return model

    # --- Hyperparameter grid -------------------------------------------------
    logger.info("Starting GridSearchCV hyperparameter tuning (PR-AUC)...")
    param_grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.01, 0.05],
        "max_depth": [4, 6, 8],
        "num_leaves": [15, 31],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    estimator = lgb.LGBMClassifier(**base_params)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=pr_auc_scorer,
        cv=3,                    # Time-series aware CV could be used
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best params: %s", grid_search.best_params_)
    logger.info("Best CV PR-AUC: %.4f", grid_search.best_score_)
    return best_model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
) -> Any:
    """
    Train an XGBoost classifier (alternative production model).
    """
    import xgboost as xgb

    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    spw = n_neg / n_pos

    base_params = {
        "scale_pos_weight": spw,
        "use_label_encoder": False,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    if not tune:
        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            **base_params,
        )
        model.fit(X_train, y_train)
        return model

    logger.info("Starting XGBoost GridSearchCV tuning (PR-AUC)...")
    param_grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.01, 0.05],
        "max_depth": [4, 6],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    estimator = xgb.XGBClassifier(**base_params)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=pr_auc_scorer,
        cv=3,
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best XGB params: %s", grid_search.best_params_)
    logger.info("Best XGB CV PR-AUC: %.4f", grid_search.best_score_)
    return best_model
