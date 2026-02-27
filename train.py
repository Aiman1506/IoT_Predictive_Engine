"""
FactoryGuard AI - Training Pipeline Orchestrator
==================================================
STEPS 1-8: End-to-end pipeline that:
  1. Loads and validates sensor data
  2. Engineers rolling-window features
  3. Splits temporally (no leakage)
  4. Trains baseline models (LR, RF)
  5. Trains production model (LightGBM)
  6. Evaluates all models with PR-AUC
  7. Generates SHAP explanations
  8. Saves model + preprocessor via joblib

Usage:
    python train.py
    python train.py --data data/sensor_data.csv --model lgbm
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- Local modules -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import load_sensor_data, get_data_summary
from src.feature_engineering import build_features, get_feature_columns
from src.splitting import time_based_split
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_lightgbm,
)
from src.evaluation import evaluate_model, compare_models
from src.explainability import explain_model

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
        ),
        logging.FileHandler("outputs/training.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("FactoryGuard.Train")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "sensor_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactoryGuard AI - Training Pipeline"
    )
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                        help="Path to sensor CSV data")
    parser.add_argument("--model", type=str, default="lgbm",
                        choices=["lgbm", "xgb"],
                        help="Production model type (lgbm or xgb)")
    parser.add_argument("--tune", action="store_true", default=False,
                        help="Enable GridSearchCV hyperparameter tuning")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP explanation generation")
    return parser.parse_args()


def save_model_artifacts(
    model: object,
    scaler: StandardScaler,
    feature_names: list,
    model_name: str,
) -> Path:
    """
    Save model, scaler, and feature schema together via joblib.
    Versioned by datetime for reproducibility.

    Returns the path to the saved artifact.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = MODEL_DIR / f"factoryguard_{model_name}_v{timestamp}.pkl"

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": model_name,
        "version": timestamp,
        "description": "FactoryGuard AI - IoT Predictive Maintenance",
    }
    joblib.dump(artifact, artifact_path, compress=3)
    logger.info("Model artifact saved: %s", artifact_path)

    # Also save as 'latest' symlink-equivalent for the API to load
    latest_path = MODEL_DIR / "factoryguard_latest.pkl"
    joblib.dump(artifact, latest_path, compress=3)
    logger.info("Latest model updated: %s", latest_path)

    return artifact_path


def run_pipeline(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Load & Validate Data
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: Loading and Validating Sensor Data")
    logger.info("=" * 70)
    df = load_sensor_data(args.data)
    summary = get_data_summary(df)
    logger.info("Data summary: %s", {
        k: v for k, v in summary.items() if k != "sensor_stats"
    })

    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 70)
    df_features = build_features(df)
    feature_cols = get_feature_columns(df_features)
    logger.info("Features created: %d", len(feature_cols))

    # =========================================================================
    # STEP 3: Time-Aware Train/Test Split
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 3: Temporal Train/Test Split (80/20)")
    logger.info("=" * 70)
    train_df, test_df = time_based_split(df_features, train_fraction=0.80)

    X_train_raw = train_df[feature_cols].values
    y_train = train_df["failure_label"].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df["failure_label"].values

    # =========================================================================
    # Preprocessing: StandardScaler (fit only on train to prevent leakage)
    # =========================================================================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # =========================================================================
    # STEP 4: Baseline Models
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 4: Training Baseline Models")
    logger.info("=" * 70)
    results = {}

    lr_model = train_logistic_regression(X_train, y_train)
    results["Logistic Regression"] = evaluate_model(
        lr_model, X_test, y_test, model_name="Logistic Regression"
    )

    rf_model = train_random_forest(X_train, y_train)
    results["Random Forest"] = evaluate_model(
        rf_model, X_test, y_test, model_name="Random Forest"
    )

    # =========================================================================
    # STEP 5: Production Model (LightGBM / XGBoost)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 5: Training Production Model (%s)", args.model.upper())
    logger.info("=" * 70)

    if args.model == "lgbm":
        prod_model = train_lightgbm(
            X_train, y_train,
            feature_names=feature_cols,
            tune=args.tune,
        )
        model_label = "LightGBM"
    else:
        from src.models import train_xgboost
        prod_model = train_xgboost(X_train, y_train, tune=args.tune)
        model_label = "XGBoost"

    # =========================================================================
    # STEP 6: Evaluate Production Model
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 6: Evaluating Production Model")
    logger.info("=" * 70)
    results[model_label] = evaluate_model(
        prod_model, X_test, y_test,
        model_name=model_label,
        feature_names=feature_cols,
    )

    # =========================================================================
    # Model Comparison Table
    # =========================================================================
    comparison = compare_models(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv")

    # =========================================================================
    # STEP 7: SHAP Explainability
    # =========================================================================
    if not args.no_shap:
        logger.info("=" * 70)
        logger.info("STEP 7: Generating SHAP Explanations")
        logger.info("=" * 70)
        try:
            explain_model(
                model=prod_model,
                X_test=X_test,
                feature_names=feature_cols,
                n_samples=min(500, len(X_test)),
                save_plots=True,
            )
        except Exception as exc:
            logger.warning("SHAP explanation failed (non-critical): %s", exc)

    # =========================================================================
    # STEP 8: Save Model Artifacts
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 8: Saving Model Artifacts")
    logger.info("=" * 70)
    artifact_path = save_model_artifacts(
        model=prod_model,
        scaler=scaler,
        feature_names=feature_cols,
        model_name=args.model.lower(),
    )

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Best model   : %s", model_label)
    logger.info("  PR-AUC       : %.4f", results[model_label]["pr_auc"])
    logger.info("  Artifact     : %s", artifact_path)
    logger.info("=" * 70)


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
