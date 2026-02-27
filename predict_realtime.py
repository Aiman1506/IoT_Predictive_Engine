"""
FactoryGuard AI - Stateful Prediction Client
=============================================
Demonstrates real-time streaming inference using a rolling history
buffer for accurate feature computation.

In production, replace the simulated readings with actual sensor feeds
(OPC-UA, MQTT, Kafka, etc.)
"""

import sys
import time
import logging
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.feature_engineering import (
    ROLLING_WINDOWS, EMA_SPAN, LAG_STEPS, SENSOR_COLUMNS
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = Path(__file__).parent / "models" / "factoryguard_latest.pkl"
ALERT_THRESHOLD = 0.50
CRITICAL_THRESHOLD = 0.80


class RealtimePredictor:
    """
    Maintains a rolling buffer of sensor readings to compute
    accurate time-series features for real-time inference.
    """

    def __init__(self, buffer_size: int = 24):
        artifact = joblib.load(MODEL_PATH)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_names = artifact["feature_names"]
        self.buffer_size = max(buffer_size, 14)
        self.history: deque = deque(maxlen=self.buffer_size)
        logger.info("RealtimePredictor initialized (buffer=%d)",
                    self.buffer_size)

    def update(self, temperature: float, vibration: float,
               pressure: float) -> dict:
        """
        Accept a new sensor reading and return failure prediction.

        Parameters
        ----------
        temperature, vibration, pressure : float

        Returns
        -------
        dict with failure_probability, risk_level, prediction
        """
        self.history.append({
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
        })

        if len(self.history) < 3:
            return {"failure_probability": 0.0, "risk_level": "INSUFFICIENT_HISTORY",
                    "prediction": 0}

        # Build mini-DataFrame from history
        hist_df = pd.DataFrame(list(self.history))

        features = {}
        for col in SENSOR_COLUMNS:
            s = hist_df[col]
            for w in ROLLING_WINDOWS:
                effective_window = min(w, len(s))
                features[f"{col}_rolling_mean_{w}h"] = s.rolling(
                    effective_window, min_periods=1).mean().iloc[-1]
                features[f"{col}_rolling_std_{w}h"] = s.rolling(
                    effective_window, min_periods=1).std().fillna(0).iloc[-1]
            features[f"{col}_ema_{EMA_SPAN}h"] = s.ewm(
                span=min(EMA_SPAN, len(s)), adjust=False).mean().iloc[-1]
            for lag in LAG_STEPS:
                features[f"{col}_lag_{lag}"] = (
                    s.iloc[-1 - lag] if len(s) > lag else s.iloc[0]
                )
            features[f"{col}_diff_1"] = s.diff(1).fillna(0).iloc[-1]

        X_raw = np.array(
            [features.get(f, 0.0) for f in self.feature_names],
            dtype=np.float32
        ).reshape(1, -1)

        X_scaled = self.scaler.transform(X_raw)
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        pred = int(prob >= ALERT_THRESHOLD)

        risk = "CRITICAL" if prob >= CRITICAL_THRESHOLD else (
            "HIGH" if prob >= ALERT_THRESHOLD else
            "MEDIUM" if prob >= 0.25 else "LOW"
        )

        return {
            "failure_probability": round(prob, 4),
            "risk_level": risk,
            "prediction": pred,
            "buffer_size": len(self.history),
        }


def simulate_streaming(n_readings: int = 50) -> None:
    """Run a simulated streaming inference session."""
    if not MODEL_PATH.exists():
        logger.error("No trained model found. Run `python train.py` first.")
        return

    rng = np.random.default_rng(99)
    predictor = RealtimePredictor(buffer_size=24)

    logger.info("=" * 60)
    logger.info("Starting simulated real-time inference (%d readings)", n_readings)
    logger.info("=" * 60)

    for i in range(n_readings):
        # Simulate readings — inject degradation towards the end
        degradation = min(1.0, i / 35)
        temp = rng.normal(55 + 30 * degradation, 2.0)
        vibr = abs(rng.normal(2.0 + 5 * degradation, 0.3))
        pres = rng.normal(6.0 - 2 * degradation, 0.4)

        t0 = time.perf_counter()
        result = predictor.update(temp, vibr, pres)
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "t=%03d | T=%.1f°C V=%.2fmm/s P=%.1fbar | "
            "p_fail=%.3f | risk=%-10s | %.1fms",
            i, temp, vibr, pres,
            result["failure_probability"],
            result["risk_level"],
            latency_ms,
        )

        if result["risk_level"] in ("HIGH", "CRITICAL"):
            logger.warning(">>> MAINTENANCE ALERT: Failure risk detected!")

        time.sleep(0.01)  # Simulate 10ms sensor polling


if __name__ == "__main__":
    simulate_streaming(n_readings=50)
