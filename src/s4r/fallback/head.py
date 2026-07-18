"""Route C prediction head: 66-parameter linear head with structural constraints.

frac_model  = alpha * sigmoid(X @ w_t + b_t)        -> cultivated fraction, cap-safe
shares_model = softmax(X @ W_s.T + b_s)             -> simplex, non-negative
frac  = conf * frac_model  + (1-conf) * base_frac          (structural shrinkage)
shares = conf * shares_model + (1-conf) * MIX_VECTOR
pred  = (frac * area_ha)[:, None] * shares

base_frac defaults to BASELINE_FRAC everywhere; weak-label anchors may replace
it per village (clipped to alpha) so that manual estimates can substitute for
missing SAR signal on zero-coverage villages (spec section 4.3).

The cap holds by construction: frac_model <= alpha, base_frac <= alpha, and a
convex blend of the two stays <= alpha. Zero-confidence villages without an
anchor reproduce baseline_allocation exactly.
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
    base_frac: np.ndarray | None = None,
) -> dict:
    w_t, b_t, W_s, b_s = unflatten(theta)

    if base_frac is None:
        base_frac = np.full(X.shape[0], config.BASELINE_FRAC)
    base_frac = np.clip(base_frac, 0.0, alpha)

    frac_model = alpha * _sigmoid(X @ w_t + b_t)
    shares_model = _softmax(X @ W_s.T + b_s[None, :])

    frac = conf * frac_model + (1.0 - conf) * base_frac
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
