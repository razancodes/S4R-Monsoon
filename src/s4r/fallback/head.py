"""Route C prediction head: 66-parameter linear head with structural constraints.

frac_model  = alpha * sigmoid(X @ w_t + b_t)        -> cultivated fraction, cap-safe
shares_model = softmax(X @ W_s.T + b_s)             -> simplex, non-negative
frac  = conf * frac_model  + (1-conf) * BASELINE_FRAC     (structural shrinkage)
shares = conf * shares_model + (1-conf) * MIX_VECTOR
pred  = (frac * area_ha)[:, None] * shares

The cap holds by construction: frac_model <= alpha, BASELINE_FRAC < alpha, and
a convex blend of the two stays <= alpha. Zero-confidence villages reproduce
baseline_allocation exactly.
"""

import numpy as np

from s4r import config

N_FEATURES = len(config.MODEL_FEATURES)
N_CROPS = len(config.CROPS)


def n_params() -> int:
    return (N_FEATURES + 1) + (N_CROPS * N_FEATURES + N_CROPS)


def unflatten(theta: np.ndarray):
    w_t = theta[:N_FEATURES]
    b_t = theta[N_FEATURES]
    off = N_FEATURES + 1
    W_s = theta[off : off + N_CROPS * N_FEATURES].reshape(N_CROPS, N_FEATURES)
    b_s = theta[off + N_CROPS * N_FEATURES :]
    return w_t, b_t, W_s, b_s


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(
    theta: np.ndarray,
    X: np.ndarray,
    area_ha: np.ndarray,
    conf: np.ndarray,
    alpha: float = config.ALPHA_CAP,
) -> dict:
    w_t, b_t, W_s, b_s = unflatten(theta)

    frac_model = alpha * _sigmoid(X @ w_t + b_t)
    shares_model = _softmax(X @ W_s.T + b_s[None, :])

    frac = conf * frac_model + (1.0 - conf) * config.BASELINE_FRAC
    shares = conf[:, None] * shares_model + (1.0 - conf)[:, None] * config.MIX_VECTOR[None, :]

    totals = frac * area_ha
    pred = totals[:, None] * shares
    return {
        "frac_model": frac_model,
        "shares_model": shares_model,
        "frac": frac,
        "shares": shares,
        "totals": totals,
        "pred": pred,
    }
