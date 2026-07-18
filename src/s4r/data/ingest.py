"""Strict loaders for the feature table and Kaggle sample submission.

Fail loudly on any schema drift — a silently malformed feature table is how a
1000-point MSE mistake ships. See data/README.md for the porting contract.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from s4r import config


class DataValidationError(ValueError):
    pass


def load_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"feature table not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")
    if len(df) != config.N_VILLAGES:
        raise DataValidationError(f"expected {config.N_VILLAGES} rows (one per village), got {len(df)}")
    if df["village_id"].duplicated().any():
        dupes = df.loc[df["village_id"].duplicated(), "village_id"].tolist()
        raise DataValidationError(f"duplicate village_id values: {dupes}")
    area = df["area_ha"]
    if area.isna().any() or (area <= 0).any():
        bad = df.loc[area.isna() | (area <= 0), "village_id"].tolist()
        raise DataValidationError(f"area_ha must be positive and non-null; bad villages: {bad}")

    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    df["is_synthetic"] = df["is_synthetic"].astype(bool)
    return df.reset_index(drop=True)


def load_sample_submission(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"sample submission not found: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != config.SUBMISSION_COLUMNS:
        raise DataValidationError(
            f"sample submission columns {list(df.columns)} != required {config.SUBMISSION_COLUMNS}"
        )
    if len(df) != config.N_VILLAGES:
        raise DataValidationError(f"sample submission must have {config.N_VILLAGES} rows, got {len(df)}")
    return df


def standardize_features(df: pd.DataFrame) -> np.ndarray:
    """Z-score MODEL_FEATURES (NaN-aware); NaNs become 0 (== column mean).

    Zero-coverage villages therefore sit at the feature mean, and their
    predictions are governed entirely by the confidence blend downstream.
    """
    X = df[config.MODEL_FEATURES].to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma
    return np.nan_to_num(X, nan=0.0)
