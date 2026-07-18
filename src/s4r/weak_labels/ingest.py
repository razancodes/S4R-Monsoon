"""Manual weak-label annotation ingestion (spec section 4.3).

Annotations come from visual inspection of free public imagery (Google Earth
Pro historical, Bing Maps, Sentinel-2 true-color). They are weak, noisy, but
genuinely independent signal — used as anchor terms and blend targets, and
documented for compliance ("validate rule-based approaches" per host ruling).
"""

from pathlib import Path

import pandas as pd

from s4r import config


class WeakLabelError(ValueError):
    pass


REQUIRED = ["village_id", "cultivated_fraction_est", "dominant_crop", "confidence", "source", "notes"]


def load_weak_labels(path: str | Path, features_df: pd.DataFrame) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise WeakLabelError(f"weak-label file not found: {path}")
    df = pd.read_csv(path, keep_default_na=False)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise WeakLabelError(f"missing columns: {missing}")

    id_to_index = {int(v): i for i, v in enumerate(features_df["village_id"])}
    unknown = [v for v in df["village_id"] if int(v) not in id_to_index]
    if unknown:
        raise WeakLabelError(f"unknown village_id values: {unknown}")

    frac = pd.to_numeric(df["cultivated_fraction_est"], errors="raise")
    if ((frac < 0) | (frac > 1)).any():
        raise WeakLabelError("cultivated fraction estimates must be in [0, 1]")

    bad_crops = [c for c in df["dominant_crop"] if c != "" and c not in config.CROPS]
    if bad_crops:
        raise WeakLabelError(f"dominant_crop must be one of {config.CROPS} or empty; got {bad_crops}")

    conf = pd.to_numeric(df["confidence"], errors="raise")
    if ((conf <= 0) | (conf > 1)).any():
        raise WeakLabelError("confidence must be in (0, 1]")

    out = df.copy()
    out["cultivated_fraction_est"] = frac
    out["weight"] = conf
    out["village_index"] = [id_to_index[int(v)] for v in df["village_id"]]
    return out
