"""OlmoEarth-based pseudo-labels: per-village cultivated-fraction estimates.

Pipeline: S1 dB stacks -> frozen OlmoEarth encoder -> per-token (80 m) pooled
embeddings -> k-means over all villages' tokens -> clusters flagged
"agricultural" by their seasonal VH dynamics -> per-village fraction of
in-village tokens in agricultural clusters.

The output is deliberately weak signal: `dominant_crop` stays empty (an
unsupervised pipeline cannot credibly name crops) and confidence is modest,
so anchors nudge rather than dominate (anchor blend in fallback/train.py).
Training-time only per AGENTS.md invariant 7.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from s4r import config

PSEUDO_CONFIDENCE = 0.3
SOURCE_TAG = "olmoearth_s1_v1"


def fraction_from_clusters(
    cluster_ids: np.ndarray, ag_clusters: set[int], valid: np.ndarray
) -> float | None:
    """Fraction of valid tokens assigned to agricultural clusters; None when a
    village has no valid tokens (caller must then omit the village — never
    fabricate an anchor)."""
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None
    ag = np.isin(cluster_ids, list(ag_clusters)) & valid
    return float(ag.sum() / n_valid)


def identify_ag_clusters(
    cluster_ids: np.ndarray, vh_range: np.ndarray, min_range_db: float = 3.0
) -> set[int]:
    """Clusters whose mean seasonal VH range (max-min dB across dates) exceeds
    the threshold. Monsoon cropland shows strong temporal dynamics; water,
    urban and bare surfaces stay comparatively flat."""
    out = set()
    for cid in np.unique(cluster_ids):
        if float(vh_range[cluster_ids == cid].mean()) >= min_range_db:
            out.add(int(cid))
    return out


def build_annotations(
    fractions: dict[int, float],
    confidence: float,
    source: str,
    notes: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Annotation table in the exact s4r.weak_labels.ingest schema."""
    if not 0.0 < confidence <= 1.0:
        raise ValueError(f"confidence must be in (0, 1]; got {confidence}")
    bad = {v: f for v, f in fractions.items() if not 0.0 <= f <= 1.0}
    if bad:
        raise ValueError(f"cultivated fraction out of [0, 1] for villages {bad}")
    rows = []
    for vid in sorted(fractions):
        rows.append(
            {
                "village_id": int(vid),
                "cultivated_fraction_est": float(fractions[vid]),
                "dominant_crop": "",
                "confidence": float(confidence),
                "source": source,
                "notes": (notes or {}).get(vid, "auto pseudo-label"),
            }
        )
    return pd.DataFrame(rows, columns=[
        "village_id", "cultivated_fraction_est", "dominant_crop",
        "confidence", "source", "notes",
    ])


# --- model-driven pipeline (needs the route-a dependency group) --------------


def _load_encoder():
    from olmoearth_pretrain_minimal import ModelID, load_model_from_id

    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=True)
    model.eval()
    return model


def embed_village_tokens(
    stack_db: np.ndarray,
    in_village: np.ndarray,
    timestamps: np.ndarray,
    model,
    patch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one village stack through the frozen encoder.

    Returns (tokens (P_H, P_W, D) pooled over time/band-sets,
             token_valid (P_H, P_W) — token cell majority-inside the village
             with finite data, and vh_range (P_H, P_W) seasonal VH dB range).
    """
    import torch
    from olmoearth_pretrain_minimal import Normalizer
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
        MaskedOlmoEarthSample,
    )

    h, w, t, c = stack_db.shape
    # encoder's spatial position encoding assumes a square token grid
    side = min(h, w) // patch_size
    ph = pw = side
    if ph == 0 or pw == 0:
        return (
            np.zeros((0, 0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
            np.zeros((0, 0), dtype=np.float32),
        )
    hc, wc = ph * patch_size, pw * patch_size
    stack_db = stack_db[:hc, :wc]
    in_village = in_village[:hc, :wc]

    finite = np.isfinite(stack_db).all(axis=(2, 3))
    filled = np.where(np.isfinite(stack_db), stack_db, -25.0).astype(np.float32)

    normalizer = Normalizer(std_multiplier=2.0)
    normed = normalizer.normalize(Modality.SENTINEL1, filled).astype(np.float32)

    sample = MaskedOlmoEarthSample(
        timestamps=torch.from_numpy(np.asarray(timestamps, dtype=np.int64))[None],
        sentinel1=torch.from_numpy(normed)[None],
        sentinel1_mask=torch.zeros(1, hc, wc, t, dtype=torch.long),
    )
    with torch.no_grad():
        out = model.encoder(sample, patch_size=patch_size, input_res=10, fast_pass=True)
    toks = out["tokens_and_masks"].sentinel1[0]  # (P_H, P_W, T, band_sets, D)
    tokens = toks.mean(dim=(2, 3)).numpy()  # pool time + band sets -> (P_H, P_W, D)

    cell = (in_village & finite)[: ph * patch_size, : pw * patch_size]
    cell = cell.reshape(ph, patch_size, pw, patch_size).mean(axis=(1, 3))
    token_valid = cell >= 0.5

    vh = filled[..., 1].reshape(ph, patch_size, pw, patch_size, t).mean(axis=(1, 3))
    vh_range = (vh.max(axis=-1) - vh.min(axis=-1)).astype(np.float32)
    return tokens.astype(np.float32), token_valid, vh_range


def generate_pseudo_labels(
    stacks: dict[int, dict],
    n_clusters: int = 6,
    min_range_db: float = 3.0,
    seed: int = 0,
    model=None,
) -> tuple[dict[int, float], dict]:
    """Full unsupervised pipeline over all villages.

    Returns (fractions per village_id, diagnostics dict).
    """
    from scipy.cluster.vq import kmeans2

    if model is None:
        model = _load_encoder()

    per_village = {}
    all_tokens, all_vh = [], []
    for vid, d in sorted(stacks.items()):
        tokens, valid, vh_range = embed_village_tokens(
            d["stack"], d["in_village"], d["timestamps"], model
        )
        per_village[vid] = (tokens, valid, vh_range)
        if valid.any():
            all_tokens.append(tokens[valid])
            all_vh.append(vh_range[valid])

    if not all_tokens:
        raise RuntimeError("no valid tokens in any village")
    X = np.concatenate(all_tokens)
    vh_flat = np.concatenate(all_vh)
    # standardize embedding dims so no dimension dominates the distance metric
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    _, labels = kmeans2(X, n_clusters, seed=seed, minit="++")
    ag = identify_ag_clusters(labels, vh_flat, min_range_db=min_range_db)

    fractions: dict[int, float] = {}
    offset = 0
    for vid, (tokens, valid, _) in sorted(per_village.items()):
        n = int(valid.sum())
        if n == 0:
            continue
        ids = labels[offset : offset + n]
        offset += n
        frac = fraction_from_clusters(ids, ag, np.ones(n, dtype=bool))
        if frac is not None:
            fractions[vid] = frac

    diagnostics = {
        "n_clusters": n_clusters,
        "ag_clusters": sorted(ag),
        "min_range_db": min_range_db,
        "n_tokens_total": int(X.shape[0]),
        "cluster_mean_vh_range": {
            int(c): float(vh_flat[labels == c].mean()) for c in np.unique(labels)
        },
    }
    return fractions, diagnostics


def write_pseudo_annotations(
    fractions: dict[int, float], out_path: str | Path
) -> Path:
    df = build_annotations(fractions, confidence=PSEUDO_CONFIDENCE, source=SOURCE_TAG)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
