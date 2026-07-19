"""Adam trainer for Route A on the same aggregate-only LLP losses as Route C.

Only the Capella adapter (patch embed + projection) and the 66-parameter head
train; the OlmoEarth backbone stays frozen. The L2 penalty applies to the head
theta exactly as in Route C (the adapter is regularized by the frozen
backbone and the tiny head bottleneck). Runs are seeded and serialized to
JSON run logs, mirroring fallback/train.py.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from s4r import config
from s4r.fallback.train import anchor_base_frac
from s4r.route_a.adapter import RouteAModel, load_frozen_backbone
from s4r.route_a.losses_torch import l2_penalty, loss_anchor, loss_mix, loss_shrink, loss_total


@dataclass
class TrainAConfig:
    alpha: float = config.ALPHA_CAP
    w_total: float = 1.0
    w_mix: float = 1.0
    w_shrink: float = 0.1
    w_anchor: float = 1.0
    lam: float = 1e-3
    lr: float = 0.01
    epochs: int = 300
    seed: int = 0


def _loss_components(out: dict, conf: torch.Tensor, cfg: TrainAConfig, anchors, theta):
    return {
        "total": loss_total(out["totals"]),
        "mix": loss_mix(out["pred"]),
        "shrink": loss_shrink(out["frac_model"], out["shares_model"], conf),
        "anchor": loss_anchor(out["frac"], anchors),
        "l2": l2_penalty([theta], cfg.lam),
    }


def train_route_a(
    chips: torch.Tensor,
    area_ha: np.ndarray,
    conf: np.ndarray,
    cfg: TrainAConfig,
    anchors: pd.DataFrame | None = None,
    run_dir: str | None = "experiments/runs",
    backbone=None,
) -> dict:
    torch.manual_seed(cfg.seed)
    if backbone is None:
        backbone = load_frozen_backbone(load_weights=True)
    model = RouteAModel(backbone, alpha=cfg.alpha)
    model.train()

    area_t = torch.as_tensor(np.array(area_ha, dtype=np.float32))
    conf_t = torch.as_tensor(np.array(conf, dtype=np.float32))
    base = anchor_base_frac(anchors, len(area_ha), cfg.alpha)
    base_t = torch.as_tensor(base, dtype=torch.float32)
    chips = chips.float()

    opt = torch.optim.Adam(model.trainable_parameters(), lr=cfg.lr)
    loss_curve = []
    for _ in range(cfg.epochs):
        opt.zero_grad()
        out = model(chips, area_t, conf_t, base_frac=base_t)
        comps = _loss_components(out, conf_t, cfg, anchors, model.theta)
        loss = (
            cfg.w_total * comps["total"]
            + cfg.w_mix * comps["mix"]
            + cfg.w_shrink * comps["shrink"]
            + cfg.w_anchor * comps["anchor"]
            + comps["l2"]
        )
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        out = model(chips, area_t, conf_t, base_frac=base_t)
        comps = _loss_components(out, conf_t, cfg, anchors, model.theta)
    pred = out["pred"].numpy().astype(float)
    comps_f = {k: float(v) for k, v in comps.items()}

    run_log_path = None
    if run_dir is not None:
        run_dir_p = Path(run_dir)
        run_dir_p.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_log_path = run_dir_p / f"route_a_{stamp}.json"
        log = {
            "route": "A",
            "timestamp": stamp,
            "config": asdict(cfg),
            "n_anchors": 0 if anchors is None else int(len(anchors)),
            "loss": loss_curve[-1],
            "loss_components": comps_f,
            "aggregate_total": float(pred.sum()),
            "aggregate_mix": {
                crop: float(s)
                for crop, s in zip(config.CROPS, pred.sum(axis=0) / pred.sum())
            },
            "per_village_totals": [float(t) for t in pred.sum(axis=1)],
            "trainable_params": int(sum(p.numel() for p in model.trainable_parameters())),
            "frozen_backbone_params": int(sum(p.numel() for p in model.backbone_parameters())),
        }
        run_log_path.write_text(json.dumps(log, indent=2))

    return {
        "model": model,
        "pred": pred,
        "frac": out["frac"].numpy().astype(float),
        "totals": out["totals"].numpy().astype(float),
        "loss": loss_curve[-1],
        "loss_curve": loss_curve,
        "loss_components": comps_f,
        "run_log_path": str(run_log_path) if run_log_path else None,
    }
