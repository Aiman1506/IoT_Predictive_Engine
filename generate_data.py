"""
FactoryGuard AI - Synthetic Sensor Data Generator
==================================================
Generates realistic time-series sensor data for 500 robotic arms
with rare catastrophic failures (<1% failure rate).

Sensors: temperature (°C), vibration (mm/s), pressure (bar)
Failure signature: gradual degradation over 24-48 hours before failure.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NUM_MACHINES = 500
DAYS = 90  # 3 months of data
READINGS_PER_HOUR = 1  # hourly readings
FAILURE_RATE = 0.007  # ~0.7% of total readings are failure events
OUTPUT_DIR = Path(__file__).parent / "data"

# Normal operating ranges
TEMP_NORMAL = {"mean": 55.0, "std": 5.0}       # °C
VIBRATION_NORMAL = {"mean": 2.0, "std": 0.5}   # mm/s
PRESSURE_NORMAL = {"mean": 6.0, "std": 0.8}    # bar


def _generate_base_signal(n_points: int, rng: np.random.Generator,
                          mean: float, std: float) -> np.ndarray:
    """Generate a base sensor signal with slight autocorrelation."""
    noise = rng.normal(0, std, n_points)
    signal = np.full(n_points, mean)
    # Add autocorrelated noise (AR(1) process)
    for i in range(1, n_points):
        noise[i] = 0.7 * noise[i - 1] + 0.3 * noise[i]
    return signal + noise


def _inject_failure_pattern(signal: np.ndarray, start_idx: int,
                            ramp_hours: int, peak_multiplier: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Inject a gradual degradation ramp before a failure event."""
    end_idx = min(start_idx + ramp_hours, len(signal))
    ramp = np.linspace(0, 1, end_idx - start_idx)
    baseline = signal[start_idx]
    signal[start_idx:end_idx] += baseline * (peak_multiplier - 1) * ramp
    # Add extra noise near failure
    signal[start_idx:end_idx] += rng.normal(0, baseline * 0.05,
                                            end_idx - start_idx)
    return signal


def generate_sensor_data() -> pd.DataFrame:
    """
    Generate synthetic sensor data with realistic failure signatures.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, machine_id, temperature, vibration,
                 pressure, failure_label
    """
    rng = np.random.default_rng(RANDOM_SEED)
    total_hours = DAYS * 24 * READINGS_PER_HOUR
    records = []

    for machine_id in range(1, NUM_MACHINES + 1):
        # Base signals
        temp = _generate_base_signal(total_hours, rng,
                                     TEMP_NORMAL["mean"], TEMP_NORMAL["std"])
        vibration = _generate_base_signal(total_hours, rng,
                                          VIBRATION_NORMAL["mean"],
                                          VIBRATION_NORMAL["std"])
        pressure = _generate_base_signal(total_hours, rng,
                                         PRESSURE_NORMAL["mean"],
                                         PRESSURE_NORMAL["std"])

        labels = np.zeros(total_hours, dtype=int)

        # Decide failure events for this machine (0-3 failures in 90 days)
        n_failures = rng.poisson(lam=0.8)
        n_failures = min(n_failures, 3)

        failure_times = sorted(
            rng.choice(range(48, total_hours - 24), size=n_failures,
                       replace=False)
        ) if n_failures > 0 else []

        for ft in failure_times:
            ramp = rng.integers(18, 36)  # 18-36 hour degradation ramp
            temp = _inject_failure_pattern(temp, ft - ramp, ramp, 1.6, rng)
            vibration = _inject_failure_pattern(vibration, ft - ramp, ramp,
                                                2.5, rng)
            pressure = _inject_failure_pattern(pressure, ft - ramp, ramp,
                                               0.6, rng)
            # Label the 24-hour window BEFORE failure as positive
            label_start = max(ft - 24, 0)
            labels[label_start:ft + 1] = 1

        # Create timestamps (starting Jan 1, 2026)
        timestamps = pd.date_range(
            start="2026-01-01", periods=total_hours, freq="h"
        )

        machine_df = pd.DataFrame({
            "timestamp": timestamps,
            "machine_id": machine_id,
            "temperature": np.round(temp, 2),
            "vibration": np.round(np.abs(vibration), 4),
            "pressure": np.round(pressure, 2),
            "failure_label": labels,
        })
        records.append(machine_df)

    df = pd.concat(records, ignore_index=True)

    # Inject ~2% missing values randomly
    mask_size = int(0.02 * len(df))
    for col in ["temperature", "vibration", "pressure"]:
        missing_idx = rng.choice(df.index, size=mask_size, replace=False)
        df.loc[missing_idx, col] = np.nan

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sensor_data.csv"

    print("Generating synthetic sensor data for FactoryGuard AI...")
    df = generate_sensor_data()

    failure_pct = df["failure_label"].mean() * 100
    print(f"  Total records   : {len(df):,}")
    print(f"  Failure rate    : {failure_pct:.2f}%")
    print(f"  Machines        : {df['machine_id'].nunique()}")
    print(f"  Date range      : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Missing values  : {df.isnull().sum().sum():,}")

    df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
