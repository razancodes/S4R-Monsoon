"""Multi-restart L-BFGS trainer for the Route C head on aggregate-only losses.

Every run serializes its full hyperparameter configuration and loss breakdown
to a JSON run log — Kaggle submissions are an extremely scarce validation
signal, so every candidate must be traceable to its exact configuration.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from s4r import config
from s4r.fallback.head import forward, n_params
from s4r.losses.aggregate import l2_penalty, loss_anchor, loss_mix, loss_shrink, loss_total


@dataclass
class TrainConfig:
    alpha: float = config.ALPHA_CAP
    w_total: float = 1.0
    w_mix: float = 1.0
    w_shrink: float = 0.1
    w_anchor: float = 1.0
    lam: float = 1e-3
    n_restarts: int = 8
    seed: int = 0
    maxiter: int = 300


def anchor_base_frac(anchors: pd.DataFrame | None, n: int, alpha: float) -> np.ndarray:
    """Blend target per village: weak-label estimate (weighted by annotation
    confidence, clipped to the cap) where available, else the regional mean.

    This lets manual inspection substitute for missing SAR signal on
    zero-coverage villages, where the model path is fully shrunk out.
    """
    base = np.full(n, config.BASELINE_FRAC)
    if anchors is not None and len(anchors):
        idx = anchors["village_index"].to_numpy(dtype=int)
        est = anchors["cultivated_fraction_est"].to_numpy(dtype=float)
        w = anchors["weight"].to_numpy(dtype=float)
        base[idx] = np.clip(w * est + (1.0 - w) * config.BASELINE_FRAC, 0.0, alpha)
    return base


def _components(theta, X, area_ha, conf, cfg: TrainConfig, anchors: pd.DataFrame | None):
    base_frac = anchor_base_frac(anchors, X.shape[0], cfg.alpha)
    out = forward(theta, X, area_ha, conf, alpha=cfg.alpha, base_frac=base_frac)
    return {
        "total": loss_total(out["totals"]),
        "mix": loss_mix(out["pred"]),
        "shrink": loss_shrink(out["frac_model"], out["shares_model"], conf),
        "anchor": loss_anchor(out["frac"], anchors),
        "l2": l2_penalty(theta, cfg.lam),
    }


def objective(theta, X, area_ha, conf, cfg: TrainConfig, anchors: pd.DataFrame | None = None) -> float:
    c = _components(theta, X, area_ha, conf, cfg, anchors)
    return (
        cfg.w_total * c["total"]
        + cfg.w_mix * c["mix"]
        + cfg.w_shrink * c["shrink"]
        + cfg.w_anchor * c["anchor"]
        + c["l2"]
    )


def train(
    X: np.ndarray,
    area_ha: np.ndarray,
    conf: np.ndarray,
    cfg: TrainConfig,
    anchors: pd.DataFrame | None = None,
    run_dir: str | None = "experiments/runs",
) -> dict:
    rng = np.random.default_rng(cfg.seed)
    best_theta, best_loss = None, np.inf
    restart_losses = []
    for i in range(cfg.n_restarts):
        theta0 = np.zeros(n_params()) if i == 0 else rng.normal(0, 0.5, size=n_params())
        res = minimize(
            objective,
            theta0,
            args=(X, area_ha, conf, cfg, anchors),
            method="L-BFGS-B",
            options={"maxiter": cfg.maxiter},
        )
        restart_losses.append(float(res.fun))
        if res.fun < best_loss:
            best_loss, best_theta = float(res.fun), res.x

    base_frac = anchor_base_frac(anchors, X.shape[0], cfg.alpha)
    out = forward(best_theta, X, area_ha, conf, alpha=cfg.alpha, base_frac=base_frac)
    comps = _components(best_theta, X, area_ha, conf, cfg, anchors)

    run_log_path = None
    if run_dir is not None:
        run_dir_p = Path(run_dir)
        run_dir_p.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_log_path = run_dir_p / f"route_c_{stamp}.json"
        log = {
            "route": "C",
            "timestamp": stamp,
            "config": asdict(cfg),
            "n_anchors": 0 if anchors is None else int(len(anchors)),
            "loss": best_loss,
            "loss_components": comps,
            "restart_losses": restart_losses,
            "aggregate_total": float(out["pred"].sum()),
            "aggregate_mix": {
                crop: float(s)
                for crop, s in zip(config.CROPS, out["pred"].sum(axis=0) / out["pred"].sum())
            },
            "per_village_totals": [float(t) for t in out["totals"]],
        }
        run_log_path.write_text(json.dumps(log, indent=2))

    return {
        "theta": best_theta,
        "pred": out["pred"],
        "frac": out["frac"],
        "totals": out["totals"],
        "loss": best_loss,
        "loss_components": comps,
        "run_log_path": str(run_log_path) if run_log_path else None,
    }
