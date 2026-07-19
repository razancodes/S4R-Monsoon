"""Torch mirror of the Route C head (src/s4r/fallback/head.py).

Identical math, identical 66-parameter layout, identical structural
constraints: alpha*sigmoid caps the fraction, softmax keeps shares on the
simplex, and the confidence blend shrinks toward base_frac/MIX_VECTOR.
Numerical equivalence with the numpy head is asserted in tests/test_route_a.py
— any change here must keep that test passing.
"""

import torch

from s4r import config
from s4r.fallback.head import N_CROPS, N_FEATURES


def unflatten(theta: torch.Tensor):
    w_t = theta[:N_FEATURES]
    b_t = theta[N_FEATURES]
    off = N_FEATURES + 1
    W_s = theta[off : off + N_CROPS * N_FEATURES].reshape(N_CROPS, N_FEATURES)
    b_s = theta[off + N_CROPS * N_FEATURES :]
    return w_t, b_t, W_s, b_s


def head_forward(
    theta: torch.Tensor,
    X: torch.Tensor,
    area_ha: torch.Tensor,
    conf: torch.Tensor,
    alpha: float = config.ALPHA_CAP,
    base_frac: torch.Tensor | None = None,
) -> dict:
    w_t, b_t, W_s, b_s = unflatten(theta)
    mix = torch.as_tensor(config.MIX_VECTOR, dtype=X.dtype, device=X.device)

    if base_frac is None:
        base_frac = torch.full((X.shape[0],), config.BASELINE_FRAC, dtype=X.dtype, device=X.device)
    base_frac = torch.clamp(base_frac, 0.0, alpha)

    frac_model = alpha * torch.sigmoid(torch.clamp(X @ w_t + b_t, -60, 60))
    shares_model = torch.softmax(X @ W_s.T + b_s[None, :], dim=1)

    frac = conf * frac_model + (1.0 - conf) * base_frac
    shares = conf[:, None] * shares_model + (1.0 - conf)[:, None] * mix[None, :]

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
