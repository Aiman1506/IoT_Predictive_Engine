"""
FactoryGuard AI - Flask REST API
==================================
STEP 9: Production deployment endpoint.

Endpoint: POST /predict
Input:  JSON with current sensor readings
Output: Failure probability + risk level

Design targets:
  - Response time < 50 ms
  - Model loaded once at startup (not per request)
  - Thread-safe inference
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("FactoryGuard.API")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "factoryguard_latest.pkl"
_artifact: Dict[str, Any] = {}  # Loaded once at startup


def load_model_artifact() -> None:
    """Load model and scaler into memory at application startup."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python train.py` first to train and save the model."
        )
    global _artifact
    _artifact = joblib.load(MODEL_PATH)
    logger.info(
        "Model loaded: %s (version %s)",
        _artifact.get("model_name", "unknown"),
        _artifact.get("version", "unknown"),
    )


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def _build_feature_vector(
    readings: Dict[str, float],
    feature_names: list,
) -> np.ndarray:
    """
    Build a feature vector from raw sensor readings.

    For a real-time API, the rolling/lag features require a history buffer.
    Here we implement a stateless fallback: the single reading is broadcast
    into all derived features to enable zero-latency inference on the
    current snapshot.

    In production, integrate with a time-series store (e.g., Redis, InfluxDB)
    to compute accurate rolling features.
    """
    temp = float(readings["temperature"])
    vibr = float(readings["vibration"])
    pres = float(readings["pressure"])

    # Map raw readings to feature vector matching training schema
    feature_map: Dict[str, float] = {}

    for col, val in [("temperature", temp), ("vibration", vibr),
                     ("pressure", pres)]:
        # Rolling means — approximate with current value as fallback
        for w in [1, 6, 12]:
            feature_map[f"{col}_rolling_mean_{w}h"] = val
            feature_map[f"{col}_rolling_std_{w}h"] = 0.0  # no history
        # EMA
        feature_map[f"{col}_ema_12h"] = val
        # Lags — approximate as current value
        feature_map[f"{col}_lag_1"] = val
        feature_map[f"{col}_lag_2"] = val
        # Diff
        feature_map[f"{col}_diff_1"] = 0.0  # no prior reading

    # Assemble in the correct order
    vector = np.array([feature_map.get(f, 0.0) for f in feature_names],
                      dtype=np.float32)
    return vector.reshape(1, -1)


def _classify_risk(probability: float) -> str:
    """Convert probability to a human-readable risk level."""
    if probability >= 0.80:
        return "CRITICAL"
    elif probability >= 0.50:
        return "HIGH"
    elif probability >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index() -> str:
    """Serve the frontend UI."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health() -> tuple:
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_version": _artifact.get("version", "not_loaded"),
        "model_name": _artifact.get("model_name", "unknown"),
    }), 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    """
    Predict failure probability from current sensor readings.

    Request body (JSON):
    {
        "temperature": 72.5,   // °C
        "vibration": 3.8,      // mm/s
        "pressure": 5.2        // bar
    }

    Response (JSON):
    {
        "failure_probability": 0.234,
        "risk_level": "MEDIUM",
        "prediction": 0,
        "response_time_ms": 12.4
    }
    """
    t_start = time.perf_counter()

    # --- Input validation ---------------------------------------------------
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    required_fields = ["temperature", "vibration", "pressure"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # --- Feature engineering -------------------------------------------
        model = _artifact["model"]
        scaler = _artifact["scaler"]
        feature_names = _artifact["feature_names"]

        X_raw = _build_feature_vector(data, feature_names)
        X_scaled = scaler.transform(X_raw)

        # --- Inference -------------------------------------------------------
        prob = float(model.predict_proba(X_scaled)[0, 1])
        pred = int(prob >= 0.5)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000

        response = {
            "failure_probability": round(prob, 4),
            "risk_level": _classify_risk(prob),
            "prediction": pred,
            "response_time_ms": round(t_elapsed_ms, 2),
        }

        logger.info(
            "Prediction: prob=%.4f risk=%s latency=%.1f ms",
            prob, response["risk_level"], t_elapsed_ms,
        )
        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return jsonify({"error": "Internal inference error"}), 500


@app.route("/model-info", methods=["GET"])
def model_info() -> tuple:
    """Return metadata about the currently loaded model."""
    return jsonify({
        "model_name": _artifact.get("model_name"),
        "version": _artifact.get("version"),
        "description": _artifact.get("description"),
        "n_features": len(_artifact.get("feature_names", [])),
        "feature_names": _artifact.get("feature_names", [])[:10],  # preview
    }), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Load model when module is imported (works with gunicorn)
load_model_artifact()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)