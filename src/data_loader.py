"""
FactoryGuard AI - Data Loader & Validator
==========================================
STEP 1: Load raw sensor CSV, validate schema, handle missing values,
and enforce temporal ordering to prevent data leakage.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "timestamp", "machine_id", "temperature", "vibration",
    "pressure", "failure_label",
]
SENSOR_COLUMNS = ["temperature", "vibration", "pressure"]

# Physical sanity bounds (anything outside is treated as corrupt)
SENSOR_BOUNDS = {
    "temperature": (-40.0, 200.0),   # °C
    "vibration":   (0.0, 50.0),      # mm/s
    "pressure":    (0.0, 30.0),      # bar
}


def load_sensor_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load and validate sensor data from CSV.

    Parameters
    ----------
    filepath : str | Path
        Path to the raw sensor CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned, temporally-sorted sensor data.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing or data is corrupt.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Sensor data not found: {filepath}")

    logger.info("Loading sensor data from %s", filepath)
    df = pd.read_csv(filepath, parse_dates=["timestamp"])

    # --- Column validation ---------------------------------------------------
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # --- Type enforcement ----------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df["failure_label"] = df["failure_label"].astype(int)

    # --- Sort by machine + time (prevents future leakage) --------------------
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    # --- Sanity-bound clipping (flag & replace out-of-range as NaN) ----------
    for col, (lo, hi) in SENSOR_BOUNDS.items():
        oob_mask = ~df[col].between(lo, hi) & df[col].notna()
        n_oob = oob_mask.sum()
        if n_oob > 0:
            logger.warning(
                "Column '%s': %d out-of-bound values clipped to NaN", col, n_oob
            )
            df.loc[oob_mask, col] = np.nan

    # --- Handle missing values (forward-fill within each machine) ------------
    n_missing_before = df[SENSOR_COLUMNS].isnull().sum().sum()
    df[SENSOR_COLUMNS] = (
        df.groupby("machine_id")[SENSOR_COLUMNS]
        .transform(lambda s: s.ffill().bfill())
    )
    n_missing_after = df[SENSOR_COLUMNS].isnull().sum().sum()
    logger.info(
        "Missing values: %d -> %d (after forward/back-fill)",
        n_missing_before, n_missing_after,
    )

    # Drop any residual NaN rows (e.g., entire machine series missing)
    if n_missing_after > 0:
        df = df.dropna(subset=SENSOR_COLUMNS).reset_index(drop=True)
        logger.warning("Dropped %d rows with irrecoverable NaN values",
                       n_missing_after)

    # --- Final leakage guard: assert monotonic time per machine --------------
    _verify_temporal_monotonicity(df)

    logger.info(
        "Data loaded: %d rows, %d machines, date range %s to %s",
        len(df), df["machine_id"].nunique(),
        df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


def _verify_temporal_monotonicity(df: pd.DataFrame) -> None:
    """Ensure timestamps are non-decreasing within each machine."""
    violations = (
        df.groupby("machine_id")["timestamp"]
        .apply(lambda s: (s.diff().dt.total_seconds().dropna() < 0).any())
    )
    bad_machines = violations[violations].index.tolist()
    if bad_machines:
        raise ValueError(
            f"Temporal monotonicity violated for machines: {bad_machines}. "
            "This indicates potential data leakage."
        )


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a compact summary dict for logging / display."""
    return {
        "n_rows": len(df),
        "n_machines": df["machine_id"].nunique(),
        "date_range": (df["timestamp"].min().isoformat(), df["timestamp"].max().isoformat()),
        "failure_rate_pct": round(df["failure_label"].mean() * 100, 3),
        "missing_values": int(df[SENSOR_COLUMNS].isnull().sum().sum()),
        "sensor_stats": df[SENSOR_COLUMNS].describe().to_dict(),
    }
