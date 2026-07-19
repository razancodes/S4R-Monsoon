"""Torch mirrors of the aggregate LLP losses (src/s4r/losses/aggregate.py).

Same scaling conventions as the numpy versions; equivalence is asserted in
tests/test_route_a.py.
"""

import pandas as pd
import torch

from s4r import config


def band_penalty(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.relu(lo - x) ** 2 + torch.relu(x - hi) ** 2


def loss_total(totals: torch.Tensor) -> torch.Tensor:
    return band_penalty(totals.sum(), *config.TOTAL_AREA_BAND) / 100.0**2


def loss_mix(pred: torch.Tensor) -> torch.Tensor:
    shares = pred.sum(dim=0) / pred.sum()
    out = pred.new_zeros(())
    for c, target in enumerate(config.MIX_VECTOR):
        out = out + band_penalty(shares[c], target - config.MIX_TOL, target + config.MIX_TOL)
    return out / 0.01**2


def loss_shrink(
    frac_model: torch.Tensor, shares_model: torch.Tensor, conf: torch.Tensor
) -> torch.Tensor:
    mix = torch.as_tensor(config.MIX_VECTOR, dtype=frac_model.dtype, device=frac_model.device)
    frac_dev = ((frac_model - config.BASELINE_FRAC) / config.BASELINE_FRAC) ** 2
    share_dev = ((shares_model - mix[None, :]) ** 2).sum(dim=1)
    return ((1.0 - conf) * (frac_dev + share_dev)).mean()


def loss_anchor(frac: torch.Tensor, anchors: pd.DataFrame | None) -> torch.Tensor:
    if anchors is None or len(anchors) == 0:
        return frac.new_zeros(())
    idx = torch.as_tensor(anchors["village_index"].to_numpy(dtype=int).copy(), device=frac.device)
    est = torch.as_tensor(
        anchors["cultivated_fraction_est"].to_numpy(dtype=float).copy(),
        dtype=frac.dtype,
        device=frac.device,
    )
    w = torch.as_tensor(
        anchors["weight"].to_numpy(dtype=float).copy(), dtype=frac.dtype, device=frac.device
    )
    dev = (frac[idx] - est) ** 2 / config.BASELINE_FRAC**2
    return (w * dev).mean()


def l2_penalty(params: list[torch.Tensor], lam: float) -> torch.Tensor:
    return lam * sum((p**2).sum() for p in params)
