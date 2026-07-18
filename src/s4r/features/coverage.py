"""Per-village SAR coverage confidence in [0, 1].

Confidence weights the blend between the feature-driven model output and the
regional-mean baseline. Zero-coverage villages get confidence 0 — their
predictions become exactly the regional mean (the V4 lesson: shrink, never
zero out, never extrapolate).
"""

import numpy as np
import pandas as pd

from s4r import config


def coverage_confidence(df: pd.DataFrame, saturation: float = 0.5) -> np.ndarray:
    cov = df[[f"coverage_{d}" for d in config.DATES]].to_numpy(dtype=float)
    mean_cov = np.nan_to_num(cov, nan=0.0).mean(axis=1)
    return np.clip(mean_cov / saturation, 0.0, 1.0)
