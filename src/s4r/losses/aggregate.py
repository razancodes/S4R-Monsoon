"""Aggregate-only LLP loss components (no per-village labels exist).

Scaling conventions:
- loss_total: 1.0 == the whole region overshooting the band by 100 ha.
- loss_mix: 1.0 == one crop share off its tolerance band edge by 1 percentage point.
- loss_shrink / loss_anchor: fraction errors are relative to BASELINE_FRAC.

The per-village cap and non-negativity are enforced STRUCTURALLY in the head
(sigmoid/softmax); cap_violations() exists as a post-hoc audit that must always
return all-False before any submission.
"""

import numpy as np
import pandas as pd

from s4r import config


def band_penalty(x: float, lo: float, hi: float) -> float:
    return max(0.0, lo - x) ** 2 + max(0.0, x - hi) ** 2


def loss_total(totals: np.ndarray) -> float:
    return band_penalty(float(totals.sum()), *config.TOTAL_AREA_BAND) / 100.0**2


def loss_mix(pred: np.ndarray) -> float:
    shares = pred.sum(axis=0) / pred.sum()
    out = 0.0
    for c, target in zip(range(len(config.CROPS)), config.MIX_VECTOR):
        out += band_penalty(float(shares[c]), target - config.MIX_TOL, target + config.MIX_TOL)
    return out / 0.01**2


def loss_shrink(frac_model: np.ndarray, shares_model: np.ndarray, conf: np.ndarray) -> float:
    frac_dev = ((frac_model - config.BASELINE_FRAC) / config.BASELINE_FRAC) ** 2
    share_dev = ((shares_model - config.MIX_VECTOR[None, :]) ** 2).sum(axis=1)
    return float(np.mean((1.0 - conf) * (frac_dev + share_dev)))


def loss_anchor(frac: np.ndarray, anchors: pd.DataFrame | None) -> float:
    if anchors is None or len(anchors) == 0:
        return 0.0
    idx = anchors["village_index"].to_numpy(dtype=int)
    est = anchors["cultivated_fraction_est"].to_numpy(dtype=float)
    w = anchors["weight"].to_numpy(dtype=float)
    dev = (frac[idx] - est) ** 2 / config.BASELINE_FRAC**2
    return float(np.mean(w * dev))


def l2_penalty(theta: np.ndarray, lam: float) -> float:
    return float(lam * np.sum(theta**2))


def l2_penalty_with_prior(theta: np.ndarray, prior_W_s: np.ndarray, lam: float) -> float:
    # W_s is at offset N_FEATURES + 1 and has size N_CROPS * N_FEATURES
    off = len(config.MODEL_FEATURES) + 1
    w_size = len(config.CROPS) * len(config.MODEL_FEATURES)
    
    # Non-W_s parameters get standard L2
    penalty_other = np.sum(theta[:off]**2) + np.sum(theta[off + w_size:]**2)
    
    # W_s parameters get squared deviation from prior
    W_s_flat = theta[off : off + w_size]
    prior_flat = prior_W_s.flatten()
    penalty_W_s = np.sum((W_s_flat - prior_flat)**2)
    
    return float(lam * (penalty_other + penalty_W_s))


def cap_violations(pred: np.ndarray, area_ha: np.ndarray, alpha: float) -> np.ndarray:
    return pred.sum(axis=1) > alpha * area_ha + 1e-9
