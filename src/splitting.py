"""
FactoryGuard AI - Time-Aware Train/Test Splitting
===================================================
STEP 3: Perform a strict temporal split to prevent data leakage.

Rules:
  - Train set: all data BEFORE the cutoff date
  - Test set:  all data ON or AFTER the cutoff date
  - No random shuffling
  - Class distribution is preserved (reported, not forced)
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_FRACTION = 0.80  # 80 % train, 20 % test


def time_based_split(
    df: pd.DataFrame,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    cutoff_date: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally: train on the past, test on the future.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset with 'timestamp' and 'failure_label'.
    train_fraction : float
        Approximate fraction of data for training (used if cutoff_date
        is not provided).
    cutoff_date : str, optional
        Explicit cutoff date string (ISO 8601). Data before this date
        goes to train, data on or after goes to test.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    if cutoff_date is None:
        # Derive cutoff from fraction
        cutoff_idx = int(len(df) * train_fraction)
        cutoff_date = df.iloc[cutoff_idx]["timestamp"]
        logger.info("Auto-computed cutoff date: %s (%.0f%% split)",
                    cutoff_date, train_fraction * 100)
    else:
        cutoff_date = pd.Timestamp(cutoff_date)
        logger.info("Using explicit cutoff date: %s", cutoff_date)

    train_df = df[df["timestamp"] < cutoff_date].copy()
    test_df = df[df["timestamp"] >= cutoff_date].copy()

    _log_split_stats(train_df, test_df)
    return train_df, test_df


def _log_split_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log class distribution for both splits."""
    train_fail = train_df["failure_label"].mean() * 100
    test_fail = test_df["failure_label"].mean() * 100

    logger.info(
        "Train: %d rows (%.2f%% failures) | "
        "Test: %d rows (%.2f%% failures)",
        len(train_df), train_fail, len(test_df), test_fail,
    )

    if len(test_df) == 0:
        logger.warning("Test set is empty! Adjust cutoff date.")
    if train_df["failure_label"].sum() == 0:
        logger.warning("No failures in training set!")
