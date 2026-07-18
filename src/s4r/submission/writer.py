"""Submission CSV writer with hard-constraint gate and comparison reporting.

Nothing reaches disk unless every hard constraint holds: non-negativity, the
per-village alpha cap, the regional total band, and exact ID/order agreement
with the Kaggle sample file. Synthetic-sourced predictions are refused unless
explicitly allowed (dry-runs only).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from s4r import config


class SubmissionError(ValueError):
    pass


def validate_predictions(pred: np.ndarray, area_ha: np.ndarray, alpha: float) -> list[str]:
    msgs = []
    if (pred < 0).any():
        rows = np.unique(np.where(pred < 0)[0]).tolist()
        msgs.append(f"negative predictions in village rows {rows}")
    totals = pred.sum(axis=1)
    over = totals > alpha * area_ha + 1e-9
    if over.any():
        msgs.append(f"cap exceeded (alpha={alpha}) in village rows {np.where(over)[0].tolist()}")
    lo, hi = config.TOTAL_AREA_BAND
    grand = float(pred.sum())
    if not (lo - 1.0 <= grand <= hi + 1.0):
        msgs.append(f"grand total {grand:.1f} ha outside band [{lo}, {hi}]")
    return msgs


def write_submission(
    pred: np.ndarray,
    features_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    out_path: str | Path,
    allow_synthetic: bool = False,
) -> Path:
    if features_df["is_synthetic"].any() and not allow_synthetic:
        raise SubmissionError(
            "refusing to write a submission from synthetic data (pass allow_synthetic=True for dry-runs)"
        )

    area = features_df["area_ha"].to_numpy()
    msgs = validate_predictions(pred, area, config.ALPHA_CAP)
    if msgs:
        raise SubmissionError("; ".join(msgs))

    feature_ids = set(int(v) for v in features_df["village_id"])
    sample_ids = [int(v) for v in sample_df["ID"]]
    if feature_ids != set(sample_ids):
        raise SubmissionError(
            f"ID mismatch: features vs sample differ by {feature_ids ^ set(sample_ids)}"
        )

    by_id = {
        int(vid): pred[i] for i, vid in enumerate(features_df["village_id"])
    }
    rows = []
    for sid in sample_ids:
        p = by_id[sid]
        row = {"ID": sid}
        for crop in config.CROPS:
            row[f"{crop}_ha"] = p[config.CROPS.index(crop)]
        rows.append(row)
    out = pd.DataFrame(rows)[config.SUBMISSION_COLUMNS]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def comparison_report(
    preds: dict[str, np.ndarray], features_df: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    area = features_df["area_ha"].to_numpy()
    report = pd.DataFrame(
        {
            "village_id": features_df["village_id"],
            "village_name": features_df["village_name"],
            "area_ha": area,
            "cap_ha": alpha * area,
        }
    )
    for name, pred in preds.items():
        totals = pred.sum(axis=1)
        report[f"{name}_total_ha"] = totals
        report[f"{name}_frac"] = totals / area
        report[f"{name}_over_cap"] = totals > alpha * area + 1e-9
    return report


def report_summary(preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, pred in preds.items():
        row = {"route": name, "grand_total_ha": float(pred.sum())}
        shares = pred.sum(axis=0) / pred.sum()
        for crop, s, target in zip(config.CROPS, shares, config.MIX_VECTOR):
            row[f"{crop}_share"] = float(s)
            row[f"{crop}_target"] = float(target)
        rows.append(row)
    return pd.DataFrame(rows)
