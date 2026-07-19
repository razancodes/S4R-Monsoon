"""Sentinel-1 RTC fetch for the 29-village AOI (Microsoft Planetary Computer).

Training-time signal ONLY (AGENTS.md invariant 7): Sentinel-1 never becomes an
inference input. Data flows outward only as STAC queries over public
catalogs — no competition data leaves the machine.

Per-village output: a (H, W, T, 2) float32 stack in dB, band order
['vv', 'vh'] (OlmoEarth Modality.SENTINEL1 order), one timestep per Capella
acquisition date, plus an in-village validity mask. Cached to an .npz so the
model step and tests never re-hit the network.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from s4r import config

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"

# Capella acquisition dates — Route B mirrors them so pseudo-labels describe
# the same phenological moments the Capella features see.
TARGET_DATES = {
    "jun06": datetime(2025, 6, 6),
    "jun19": datetime(2025, 6, 19),
    "aug14": datetime(2025, 8, 14),
    "oct13": datetime(2025, 10, 13),
}


def nearest_scenes(
    scene_dts: list[datetime], targets: list[datetime], max_days: int = 30
) -> list[int | None]:
    """Index of the temporally nearest scene per target date, or None if the
    nearest is more than max_days away."""
    out: list[int | None] = []
    for t in targets:
        if not scene_dts:
            out.append(None)
            continue
        deltas = [abs((dt - t).days) for dt in scene_dts]
        best = int(np.argmin(deltas))
        out.append(best if deltas[best] <= max_days else None)
    return out


def village_geometries(data_dir: str | Path):
    """Village polygons from the competition shapefile, in EPSG:4326."""
    import geopandas as gpd

    shp = next(Path(data_dir).glob("villages_clean/*.shp"))
    gdf = gpd.read_file(shp)
    gdf["village_id"] = gdf["ID"].astype(int)
    return gdf.to_crs("EPSG:4326")


def fetch_village_stacks(
    data_dir: str | Path,
    out_npz: str | Path,
    res_m: float = 10.0,
    max_days: int = 30,
) -> dict:
    """Fetch S1 RTC clips for every village at each target date; cache to npz.

    Returns {village_id: {"stack": (H,W,T,2) dB, "in_village": (H,W) bool,
    "timestamps": (T,3) day/month0/year}}.
    """
    import planetary_computer
    import pystac_client
    import rasterio
    import rasterio.features
    import rasterio.warp
    from rasterio.windows import from_bounds

    gdf = village_geometries(data_dir)
    minx, miny, maxx, maxy = gdf.total_bounds
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=[minx, miny, maxx, maxy],
        datetime="2025-05-01/2025-11-15",
        query={"sar:polarizations": {"eq": ["VV", "VH"]}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("no Sentinel-1 RTC items found for AOI")
    scene_dts = [i.datetime.replace(tzinfo=None) for i in items]
    targets = list(TARGET_DATES.values())
    picks = nearest_scenes(scene_dts, targets, max_days=max_days)
    if any(p is None for p in picks):
        missing = [k for k, p in zip(TARGET_DATES, picks) if p is None]
        raise RuntimeError(f"no S1 scene within {max_days} days of {missing}")

    chosen = [items[p] for p in picks]
    timestamps = np.array(
        [[t.day, t.month - 1, t.year] for t in targets], dtype=np.int64
    )

    result: dict[int, dict] = {}
    for _, row in gdf.iterrows():
        vid = int(row["village_id"])
        geom = row.geometry
        stacks, masks = [], []
        for item in chosen:
            bands = []
            for asset_key in ("vv", "vh"):
                href = item.assets[asset_key].href
                with rasterio.open(href) as src:
                    g = rasterio.warp.transform_geom("EPSG:4326", src.crs, geom.__geo_interface__)
                    from shapely.geometry import shape

                    b = shape(g).bounds
                    win = from_bounds(*b, transform=src.transform)
                    scale = res_m / abs(src.transform.a)
                    out_h = max(1, int(round(win.height / scale)))
                    out_w = max(1, int(round(win.width / scale)))
                    arr = src.read(1, window=win, out_shape=(out_h, out_w), boundless=True, fill_value=0)
                    win_transform = src.window_transform(win)
                    win_transform = win_transform * win_transform.scale(
                        win.width / out_w, win.height / out_h
                    )
                    bands.append((arr, win_transform, src.crs, g))
            arrs = [a for a, *_ in bands]
            h = min(a.shape[0] for a in arrs)
            w = min(a.shape[1] for a in arrs)
            date_stack = np.stack([a[:h, :w] for a in arrs], axis=-1).astype(np.float32)
            with np.errstate(divide="ignore"):
                date_db = 10.0 * np.log10(np.where(date_stack > 0, date_stack, np.nan))
            stacks.append(date_db)
            _, tfm, crs, g = bands[0]
            inside = rasterio.features.geometry_mask(
                [g], out_shape=(h, w), transform=tfm, invert=True
            )
            masks.append(inside)

        h = min(s.shape[0] for s in stacks)
        w = min(s.shape[1] for s in stacks)
        stack = np.stack([s[:h, :w] for s in stacks], axis=2)  # (H, W, T, 2)
        in_village = np.logical_and.reduce([m[:h, :w] for m in masks])
        result[vid] = {"stack": stack, "in_village": in_village, "timestamps": timestamps}

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for vid, d in result.items():
        payload[f"stack_{vid}"] = d["stack"]
        payload[f"mask_{vid}"] = d["in_village"]
    payload["timestamps"] = timestamps
    payload["scene_ids"] = np.array([i.id for i in chosen])
    np.savez_compressed(out_npz, **payload)
    return result


def load_village_stacks(npz_path: str | Path) -> dict:
    """Load the cached per-village stacks written by fetch_village_stacks."""
    z = np.load(npz_path, allow_pickle=False)
    timestamps = z["timestamps"]
    out: dict[int, dict] = {}
    for key in z.files:
        if key.startswith("stack_"):
            vid = int(key.removeprefix("stack_"))
            out[vid] = {
                "stack": z[key],
                "in_village": z[f"mask_{vid}"],
                "timestamps": timestamps,
            }
    return out
