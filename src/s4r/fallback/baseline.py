"""Hedged regional-mean allocation — the V5-style safety-net anchor (MSE 1662).

Every village gets the same cultivated fraction (regional point estimate) and
the regional crop mix. This is both the fallback submission and the shrinkage
target for low-confidence villages.
"""

import numpy as np

from s4r import config


def baseline_allocation(area_ha: np.ndarray) -> np.ndarray:
    totals = config.BASELINE_FRAC * area_ha  # (29,)
    return totals[:, None] * config.MIX_VECTOR[None, :]  # (29, 5)
