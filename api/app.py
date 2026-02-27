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
"""
FactoryGuard AI - Flask REST API
==================================
Production Deployment Version
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
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "factoryguard_latest.pkl"
_artifact: Any = None  # Loaded once at startup


# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------
def load_model_artifact() -> None:
    """Load model at startup (Gunicorn compatible)."""
    global _artifact

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Make sure factoryguard_latest.pkl exists inside models/ folder."
        )

    _artifact = joblib.load(MODEL_PATH)

    if isinstance(_artifact, dict):
        logger.info(
            "Model loaded successfully | Name: %s | Version: %s",
            _artifact.get("model_name", "unknown"),
            _artifact.get("version", "unknown"),
        )
    else:
        logger.info("Model loaded successfully (raw model object).")


# ---------------------------------------------------------------------------
# Feature Builder
# ---------------------------------------------------------------------------
def _build_feature_vector(readings: Dict[str, float],
                          feature_names: list) -> np.ndarray:
    """
    Build feature vector matching training schema.
    Stateless fallback for production.
    """
    temp = float(readings["temperature"])
    vibr = float(readings["vibration"])
    pres = float(readings["pressure"])

    feature_map: Dict[str, float] = {}

    for col, val in [
        ("temperature", temp),
        ("vibration", vibr),
        ("pressure", pres),
    ]:
        for w in [1, 6, 12]:
            feature_map[f"{col}_rolling_mean_{w}h"] = val
            feature_map[f"{col}_rolling_std_{w}h"] = 0.0

        feature_map[f"{col}_ema_12h"] = val
        feature_map[f"{col}_lag_1"] = val
        feature_map[f"{col}_lag_2"] = val
        feature_map[f"{col}_diff_1"] = 0.0

    vector = np.array(
        [feature_map.get(f, 0.0) for f in feature_names],
        dtype=np.float32,
    )

    return vector.reshape(1, -1)


# ---------------------------------------------------------------------------
# Risk Classification
# ---------------------------------------------------------------------------
def _classify_risk(probability: float) -> str:
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
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": _artifact is not None,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    t_start = time.perf_counter()

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    required_fields = ["temperature", "vibration", "pressure"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # ---------------------------------------------------------
        # Case 1: Model saved as dictionary (recommended)
        # ---------------------------------------------------------
        if isinstance(_artifact, dict) and "model" in _artifact:
            model = _artifact["model"]
            scaler = _artifact.get("scaler")
            feature_names = _artifact.get("feature_names")

            if scaler and feature_names:
                X_raw = _build_feature_vector(data, feature_names)
                X = scaler.transform(X_raw)
            else:
                # fallback to raw features
                X = np.array([[
                    float(data["temperature"]),
                    float(data["vibration"]),
                    float(data["pressure"])
                ]])

        # ---------------------------------------------------------
        # Case 2: Only model saved
        # ---------------------------------------------------------
        else:
            model = _artifact
            X = np.array([[
                float(data["temperature"]),
                float(data["vibration"]),
                float(data["pressure"])
            ]])

        # Inference
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= 0.5)

        response_time = (time.perf_counter() - t_start) * 1000

        response = {
            "failure_probability": round(prob, 4),
            "risk_level": _classify_risk(prob),
            "prediction": pred,
            "response_time_ms": round(response_time, 2),
        }

        logger.info(
            "Prediction successful | Prob: %.4f | Risk: %s | Time: %.2f ms",
            prob, response["risk_level"], response_time
        )

        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    if isinstance(_artifact, dict):
        return jsonify({
            "model_name": _artifact.get("model_name"),
            "version": _artifact.get("version"),
            "n_features": len(_artifact.get("feature_names", [])),
        }), 200
    else:
        return jsonify({
            "model_type": str(type(_artifact)),
        }), 200


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
load_model_artifact()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
    """
FactoryGuard AI - Flask REST API
==================================
Production Deployment Version
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
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "factoryguard_latest.pkl"
_artifact: Any = None  # Loaded once at startup


# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------
def load_model_artifact() -> None:
    """Load model at startup (Gunicorn compatible)."""
    global _artifact

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Make sure factoryguard_latest.pkl exists inside models/ folder."
        )

    _artifact = joblib.load(MODEL_PATH)

    if isinstance(_artifact, dict):
        logger.info(
            "Model loaded successfully | Name: %s | Version: %s",
            _artifact.get("model_name", "unknown"),
            _artifact.get("version", "unknown"),
        )
    else:
        logger.info("Model loaded successfully (raw model object).")


# ---------------------------------------------------------------------------
# Feature Builder
# ---------------------------------------------------------------------------
def _build_feature_vector(readings: Dict[str, float],
                          feature_names: list) -> np.ndarray:
    """
    Build feature vector matching training schema.
    Stateless fallback for production.
    """
    temp = float(readings["temperature"])
    vibr = float(readings["vibration"])
    pres = float(readings["pressure"])

    feature_map: Dict[str, float] = {}

    for col, val in [
        ("temperature", temp),
        ("vibration", vibr),
        ("pressure", pres),
    ]:
        for w in [1, 6, 12]:
            feature_map[f"{col}_rolling_mean_{w}h"] = val
            feature_map[f"{col}_rolling_std_{w}h"] = 0.0

        feature_map[f"{col}_ema_12h"] = val
        feature_map[f"{col}_lag_1"] = val
        feature_map[f"{col}_lag_2"] = val
        feature_map[f"{col}_diff_1"] = 0.0

    vector = np.array(
        [feature_map.get(f, 0.0) for f in feature_names],
        dtype=np.float32,
    )

    return vector.reshape(1, -1)


# ---------------------------------------------------------------------------
# Risk Classification
# ---------------------------------------------------------------------------
def _classify_risk(probability: float) -> str:
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
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": _artifact is not None,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    t_start = time.perf_counter()

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    required_fields = ["temperature", "vibration", "pressure"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # ---------------------------------------------------------
        # Case 1: Model saved as dictionary (recommended)
        # ---------------------------------------------------------
        if isinstance(_artifact, dict) and "model" in _artifact:
            model = _artifact["model"]
            scaler = _artifact.get("scaler")
            feature_names = _artifact.get("feature_names")

            if scaler and feature_names:
                X_raw = _build_feature_vector(data, feature_names)
                X = scaler.transform(X_raw)
            else:
                # fallback to raw features
                X = np.array([[
                    float(data["temperature"]),
                    float(data["vibration"]),
                    float(data["pressure"])
                ]])

        # ---------------------------------------------------------
        # Case 2: Only model saved
        # ---------------------------------------------------------
        else:
            model = _artifact
            X = np.array([[
                float(data["temperature"]),
                float(data["vibration"]),
                float(data["pressure"])
            ]])

        # Inference
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= 0.5)

        response_time = (time.perf_counter() - t_start) * 1000

        response = {
            "failure_probability": round(prob, 4),
            "risk_level": _classify_risk(prob),
            "prediction": pred,
            "response_time_ms": round(response_time, 2),
        }

        logger.info(
            "Prediction successful | Prob: %.4f | Risk: %s | Time: %.2f ms",
            prob, response["risk_level"], response_time
        )

        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    if isinstance(_artifact, dict):
        return jsonify({
            "model_name": _artifact.get("model_name"),
            "version": _artifact.get("version"),
            "n_features": len(_artifact.get("feature_names", [])),
        }), 200
    else:
        return jsonify({
            "model_type": str(type(_artifact)),
        }), 200


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
load_model_artifact()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)