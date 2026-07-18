"""Regenerate the village feature table from raw Capella GEO preview rasters.

All per-village/date statistics derive from exact DN histograms (256 bins,
uint8 previews, nodata=0) accumulated over the village polygon:

- coverage  = valid pixels / polygon pixels
- mean/std  = of dB = 20*log10(DN) over valid pixels (exact, from histogram)
- flood     = fraction of valid pixels with DN <= the date's global 15th
              percentile DN (all-village histogram) — June dates averaged
- trajectory features = fits over the 4-date mean-dB series (day offsets)

Reads are windowed and boundless (polygon-sized), so memory stays flat
regardless of raster size (~27k x 27k). Every signal here is Capella-only
(see docs/compliance/provenance.md).
"""

import glob
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

from s4r import config

UTM_CRS = "EPSG:32643"
DATE_STAMPS = {"jun06": "20250606", "jun19": "20250619", "aug14": "20250814", "oct13": "20251013"}
# Days since the first acquisition (2025-06-06, 06-19, 08-14, 10-13).
DAY_OFFSETS = [0, 13, 69, 129]
FLOOD_PERCENTILE = 0.15
_DB_LUT = np.concatenate([[np.nan], 20.0 * np.log10(np.arange(1, 256))])


def dn_to_db(dn: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(dn)


def stats_from_hist(hist: np.ndarray) -> tuple[float, float]:
    """Exact mean/std of dB over valid pixels (DN >= 1) from a 256-bin histogram."""
    w = hist[1:].astype(float)
    n = w.sum()
    if n == 0:
        return float("nan"), float("nan")
    db = _DB_LUT[1:]
    mean = float((w * db).sum() / n)
    var = float((w * (db - mean) ** 2).sum() / n)
    return mean, float(np.sqrt(var))


def percentile_dn_from_hist(hist: np.ndarray, q: float) -> int:
    """Smallest DN >= 1 at which the cumulative valid-pixel fraction reaches q."""
    w = hist[1:].astype(float)
    n = w.sum()
    if n == 0:
        return 0
    cum = np.cumsum(w) / n
    return int(np.searchsorted(cum, q) + 1)


def hist_for_geometry(ds: rasterio.DatasetReader, geom) -> tuple[np.ndarray, int]:
    """DN histogram over the polygon and the polygon's total pixel count.

    The window is polygon-sized and boundless: pixels outside the raster
    footprint read as nodata (0), so the denominator always reflects the full
    polygon regardless of how much of it the acquisition covers.
    """
    win = from_bounds(*geom.bounds, transform=ds.transform)
    win = Window(
        int(np.floor(win.col_off)),
        int(np.floor(win.row_off)),
        int(np.ceil(win.width)) + 1,
        int(np.ceil(win.height)) + 1,
    )
    data = ds.read(1, window=win, boundless=True, fill_value=0)
    transform = ds.window_transform(win)
    mask = geometry_mask([geom], out_shape=data.shape, transform=transform, invert=True)
    n_px = int(mask.sum())
    hist = np.bincount(data[mask], minlength=256).astype(np.int64)
    hist[0] = 0  # contract: histogram holds valid pixels only (DN 0 = nodata)
    return hist, n_px


def trajectory_features(means: dict[str, float]) -> dict[str, float]:
    """Temporal-fit features over the 4-date mean-dB series (NaN-tolerant)."""
    keys = list(DATE_STAMPS)
    y = np.array([means[k] for k in keys], dtype=float)
    t = np.array(DAY_OFFSETS, dtype=float)
    ok = np.isfinite(y)

    out = {
        "traj_slope": float("nan"),
        "traj_range": float("nan"),
        "traj_curvature": float("nan"),
        "delta_aug14_jun19": float(means["aug14"] - means["jun19"]),
        "delta_oct13_aug14": float(means["oct13"] - means["aug14"]),
    }
    if ok.sum() >= 2:
        out["traj_slope"] = float(np.polyfit(t[ok], y[ok], 1)[0])
        out["traj_range"] = float(y[ok].max() - y[ok].min())
    if ok.sum() >= 3:
        out["traj_curvature"] = float(2.0 * np.polyfit(t[ok], y[ok], 2)[0])
    return out


def find_preview_tifs(data_dir: str | Path) -> dict[str, Path]:
    data_dir = Path(data_dir)
    tifs: dict[str, Path] = {}
    for key, stamp in DATE_STAMPS.items():
        matches = glob.glob(str(data_dir / f"CAPELLA_*_{stamp}*" / "*GEO*preview.tif"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one GEO preview for {stamp} under {data_dir}, found {matches}"
            )
        tifs[key] = Path(matches[0])
    return tifs


def load_villages(data_dir: str | Path) -> gpd.GeoDataFrame:
    shp = Path(data_dir) / "villages_clean" / "villages_clean.shp"
    gdf = gpd.read_file(shp).to_crs(UTM_CRS)
    if len(gdf) != config.N_VILLAGES:
        raise ValueError(f"expected {config.N_VILLAGES} villages, got {len(gdf)}")
    gdf = gdf.rename(columns={"ID": "village_id", "VILLAGE": "village_name"})
    gdf["area_ha"] = gdf.geometry.area / 10_000.0
    return gdf.sort_values("village_id").reset_index(drop=True)


def extract_features(data_dir: str | Path, out_csv: str | Path | None = None) -> pd.DataFrame:
    villages = load_villages(data_dir)
    tifs = find_preview_tifs(data_dir)

    hists: dict[str, list[np.ndarray]] = {}
    n_pixels: dict[str, list[int]] = {}
    for key, tif in tifs.items():
        with rasterio.open(tif) as ds:
            if str(ds.crs) != UTM_CRS:
                raise ValueError(f"{tif} CRS {ds.crs} != {UTM_CRS}")
            per_village = [hist_for_geometry(ds, geom) for geom in villages.geometry]
        hists[key] = [h for h, _ in per_village]
        n_pixels[key] = [n for _, n in per_village]

    flood_dn = {
        key: percentile_dn_from_hist(np.sum(hists[key], axis=0), FLOOD_PERCENTILE)
        for key in tifs
    }

    rows = []
    for i in range(len(villages)):
        row = {
            "village_id": int(villages["village_id"].iloc[i]),
            "village_name": villages["village_name"].iloc[i],
            "area_ha": float(villages["area_ha"].iloc[i]),
        }
        means = {}
        flood_fracs = {}
        for key in DATE_STAMPS:
            hist, n_px = hists[key][i], n_pixels[key][i]
            valid = int(hist[1:].sum())
            mean_db, std_db = stats_from_hist(hist)
            row[f"coverage_{key}"] = valid / n_px if n_px else 0.0
            row[f"mean_{key}"] = mean_db
            row[f"std_{key}"] = std_db
            means[key] = mean_db
            thr = flood_dn[key]
            flood_fracs[key] = float(hist[1 : thr + 1].sum() / valid) if valid else float("nan")
        # Flood signal is the June (early monsoon) low-backscatter fraction.
        row["flood_frac_avg"] = float(np.mean([flood_fracs["jun06"], flood_fracs["jun19"]]))
        row.update(trajectory_features(means))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["is_synthetic"] = False
    df = df[["village_id", "village_name", "area_ha"] + config.FEATURE_COLUMNS + ["is_synthetic"]]
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
