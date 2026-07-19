"""Capella chip dataset for Route A.

One 1-channel chip per village: boundless windowed reads of the four GEO
preview rasters over the village polygon, downsampled to chip_px², DN -> dB,
averaged across the dates where each pixel is valid, then z-scored over valid
pixels (invalid and out-of-polygon pixels are 0 — which is the valid-pixel
mean post-z-score, same convention as standardize_features).

Zero-coverage villages produce all-zero chips; the confidence blend already
pins them to the regional mean, so the adapter never learns from them.
Capella-only signal (AGENTS.md invariant 7); rasters are read locally and
never transmitted.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

from s4r import config
from s4r.features.extract import find_preview_tifs, load_villages


def _read_chip(ds: rasterio.DatasetReader, geom, chip_px: int) -> np.ndarray:
    win = from_bounds(*geom.bounds, transform=ds.transform)
    win = Window(
        int(np.floor(win.col_off)),
        int(np.floor(win.row_off)),
        int(np.ceil(win.width)) + 1,
        int(np.ceil(win.height)) + 1,
    )
    dn = ds.read(
        1,
        window=win,
        out_shape=(chip_px, chip_px),
        boundless=True,
        fill_value=0,
        resampling=Resampling.nearest,  # preserves DN=0 nodata semantics
    )
    tfm = ds.window_transform(win)
    tfm = tfm * tfm.scale(win.width / chip_px, win.height / chip_px)
    inside = geometry_mask([geom], out_shape=(chip_px, chip_px), transform=tfm, invert=True)
    # cast before log10: on uint8 input numpy returns float16, whose sums
    # overflow to inf for well-covered villages
    dn_f = np.maximum(dn, 1).astype(np.float32)
    db = np.where(dn > 0, 20.0 * np.log10(dn_f), np.nan)
    db[~inside] = np.nan
    return db


def village_chips(data_dir: str | Path, chip_px: int = 64):
    """Returns (chips tensor (N, 1, chip_px, chip_px), village_ids list)."""
    import torch

    villages = load_villages(data_dir)
    tifs = find_preview_tifs(data_dir)

    per_date = []
    for key in config.DATES:
        with rasterio.open(tifs[key]) as ds:
            per_date.append(
                np.stack([_read_chip(ds, geom, chip_px) for geom in villages.geometry])
            )
    stack = np.stack(per_date, axis=1)  # (N, T, H, W)

    import warnings

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # all-NaN slices are the documented zero-coverage case, not an error
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        composite = np.nanmean(stack, axis=1)  # (N, H, W)

    chips = np.zeros_like(composite, dtype=np.float32)
    for i in range(composite.shape[0]):
        valid = np.isfinite(composite[i])
        if valid.sum() < 2:
            continue
        mu = composite[i][valid].mean()
        sd = composite[i][valid].std()
        if sd > 0:
            chips[i][valid] = (composite[i][valid] - mu) / sd

    ids = [int(v) for v in villages["village_id"]]
    return torch.from_numpy(chips[:, None]), ids
