"""
FactoryGuard AI - Feature Engineering
======================================
STEP 2: Create rolling-window time-series features for each sensor.

All windows are **backward-looking only** to prevent data leakage.
Vectorized Pandas operations are used for performance.

Features generated per sensor (temperature, vibration, pressure):
  - Rolling mean       (1 h, 6 h, 12 h)
  - Rolling std        (1 h, 6 h, 12 h)
  - Exponential MA     (span = 12 h)
  - Lag features       (t-1, t-2)
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENSOR_COLUMNS = ["temperature", "vibration", "pressure"]
ROLLING_WINDOWS = [1, 6, 12]        # hours
EMA_SPAN = 12                        # hours
LAG_STEPS = [1, 2]                   # t-1, t-2
MIN_HISTORY_HOURS = 12               # rows to drop at start (max window)


def build_features(df: pd.DataFrame,
                   sensor_cols: List[str] | None = None) -> pd.DataFrame:
    """
    Generate rolling-window features from raw sensor data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: machine_id, timestamp, and sensor columns.
    sensor_cols : list[str], optional
        Sensor columns to featurize. Defaults to SENSOR_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Original data augmented with engineered features.
        Rows with insufficient history are dropped.
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_COLUMNS

    df = df.copy()
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    feature_frames = []

    for machine_id, group in df.groupby("machine_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group = _compute_features_for_machine(group, sensor_cols)
        feature_frames.append(group)

    result = pd.concat(feature_frames, ignore_index=True)

    # Drop rows that lack sufficient history for the largest window
    n_before = len(result)
    result = result.dropna(subset=_get_feature_column_names(sensor_cols))
    result = result.reset_index(drop=True)
    n_dropped = n_before - len(result)

    logger.info(
        "Feature engineering complete: %d features created, "
        "%d rows dropped (insufficient history), %d rows remaining",
        len(_get_feature_column_names(sensor_cols)), n_dropped, len(result),
    )
    return result


def _compute_features_for_machine(group: pd.DataFrame,
                                  sensor_cols: List[str]) -> pd.DataFrame:
    """Compute all features for a single machine (vectorized)."""
    for col in sensor_cols:
        series = group[col]

        # --- Rolling mean & std (backward-looking) -----------------------
        for window in ROLLING_WINDOWS:
            group[f"{col}_rolling_mean_{window}h"] = (
                series.rolling(window=window, min_periods=window).mean()
            )
            # ddof=0 (population std) avoids NaN when window=1
            group[f"{col}_rolling_std_{window}h"] = (
                series.rolling(window=window, min_periods=window).std(ddof=0)
            )

        # --- Exponential Moving Average ----------------------------------
        group[f"{col}_ema_{EMA_SPAN}h"] = (
            series.ewm(span=EMA_SPAN, adjust=False).mean()
        )

        # --- Lag features ------------------------------------------------
        for lag in LAG_STEPS:
            group[f"{col}_lag_{lag}"] = series.shift(lag)

        # --- Rate of change (first derivative approximation) -------------
        group[f"{col}_diff_1"] = series.diff(1)

    return group


def _get_feature_column_names(sensor_cols: List[str]) -> List[str]:
    """Return the list of all generated feature column names."""
    names = []
    for col in sensor_cols:
        for w in ROLLING_WINDOWS:
            names.append(f"{col}_rolling_mean_{w}h")
            names.append(f"{col}_rolling_std_{w}h")
        names.append(f"{col}_ema_{EMA_SPAN}h")
        for lag in LAG_STEPS:
            names.append(f"{col}_lag_{lag}")
        names.append(f"{col}_diff_1")
    return names


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return all feature columns present in a DataFrame
    (excludes raw sensor, timestamp, machine_id, and label columns).
    """
    exclude = {"timestamp", "machine_id", "failure_label",
               "temperature", "vibration", "pressure"}
    return [c for c in df.columns if c not in exclude]
