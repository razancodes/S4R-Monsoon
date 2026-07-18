import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from s4r.features.extract import (
    DAY_OFFSETS,
    dn_to_db,
    hist_for_geometry,
    percentile_dn_from_hist,
    stats_from_hist,
    trajectory_features,
)


def _write_raster(path, data, origin_x=0.0, origin_y=40.0, res=1.0):
    transform = from_origin(origin_x, origin_y, res, res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:32643",
        transform=transform,
        nodata=0,
    ) as ds:
        ds.write(data, 1)
    return path


# ---------- dn_to_db / stats_from_hist ----------


def test_dn_to_db_monotone():
    db = dn_to_db(np.arange(1, 256))
    assert (np.diff(db) > 0).all()
    assert db[0] == 0.0  # 20*log10(1)


def test_stats_from_hist_single_value():
    hist = np.zeros(256, dtype=np.int64)
    hist[100] = 50
    mean_db, std_db = stats_from_hist(hist)
    assert abs(mean_db - 20 * np.log10(100)) < 1e-9
    assert abs(std_db) < 1e-9


def test_stats_from_hist_two_values():
    hist = np.zeros(256, dtype=np.int64)
    hist[10] = 1
    hist[100] = 1
    mean_db, std_db = stats_from_hist(hist)
    lo, hi = 20 * np.log10(10), 20 * np.log10(100)
    assert abs(mean_db - (lo + hi) / 2) < 1e-9
    assert abs(std_db - (hi - lo) / 2) < 1e-9


def test_stats_from_hist_empty_is_nan():
    mean_db, std_db = stats_from_hist(np.zeros(256, dtype=np.int64))
    assert np.isnan(mean_db) and np.isnan(std_db)


# ---------- percentile threshold ----------


def test_percentile_dn_from_hist():
    hist = np.zeros(256, dtype=np.int64)
    hist[10] = 15  # lowest 15%
    hist[200] = 85
    assert percentile_dn_from_hist(hist, 0.15) == 10
    assert percentile_dn_from_hist(hist, 0.10) == 10
    assert percentile_dn_from_hist(hist, 0.50) == 200


def test_percentile_empty_hist():
    assert percentile_dn_from_hist(np.zeros(256, dtype=np.int64), 0.15) == 0


# ---------- hist_for_geometry ----------


def test_hist_full_inside_polygon(tmp_path):
    data = np.full((40, 60), 100, dtype=np.uint8)
    data[:, ::2] = 200  # checkerboard columns: half 100, half 200
    path = _write_raster(tmp_path / "r.tif", data)
    geom = box(10, 10, 30, 30)  # 20x20 = 400 px fully inside
    with rasterio.open(path) as ds:
        hist, n_px = hist_for_geometry(ds, geom)
    assert n_px == 400
    assert hist.sum() == 400
    assert hist[100] == 200 and hist[200] == 200


def test_hist_polygon_half_outside_footprint(tmp_path):
    data = np.full((40, 60), 50, dtype=np.uint8)
    path = _write_raster(tmp_path / "r.tif", data)
    # raster x in [0,60]; polygon x in [50,70] -> half the pixels have data
    geom = box(50, 10, 70, 30)
    with rasterio.open(path) as ds:
        hist, n_px = hist_for_geometry(ds, geom)
    assert n_px == 400
    assert hist.sum() == 200  # only covered half has valid DN
    assert hist[50] == 200


def test_hist_polygon_fully_outside(tmp_path):
    data = np.full((40, 60), 50, dtype=np.uint8)
    path = _write_raster(tmp_path / "r.tif", data)
    geom = box(1000, 1000, 1020, 1020)
    with rasterio.open(path) as ds:
        hist, n_px = hist_for_geometry(ds, geom)
    assert n_px == 400
    assert hist.sum() == 0


def test_hist_respects_nodata_inside_footprint(tmp_path):
    data = np.full((40, 60), 80, dtype=np.uint8)
    data[10:20, 10:20] = 0  # nodata hole
    path = _write_raster(tmp_path / "r.tif", data)
    geom = box(10, 20, 20, 30)  # exactly the hole (rows 10..20)
    with rasterio.open(path) as ds:
        hist, n_px = hist_for_geometry(ds, geom)
    assert n_px == 100
    assert hist.sum() == 0  # all nodata


# ---------- trajectory features ----------


def test_trajectory_linear():
    days = DAY_OFFSETS  # [0, 13, 69, 129]
    means = {k: 2.0 + 0.1 * d for k, d in zip(("jun06", "jun19", "aug14", "oct13"), days)}
    out = trajectory_features(means)
    assert abs(out["traj_slope"] - 0.1) < 1e-9
    assert abs(out["traj_range"] - 0.1 * 129) < 1e-9
    assert abs(out["traj_curvature"]) < 1e-9
    assert abs(out["delta_aug14_jun19"] - 0.1 * (69 - 13)) < 1e-9
    assert abs(out["delta_oct13_aug14"] - 0.1 * (129 - 69)) < 1e-9


def test_trajectory_with_missing_dates():
    means = {"jun06": np.nan, "jun19": 1.0, "aug14": 3.0, "oct13": np.nan}
    out = trajectory_features(means)
    assert abs(out["traj_slope"] - (3.0 - 1.0) / (69 - 13)) < 1e-9
    assert abs(out["traj_range"] - 2.0) < 1e-9
    assert np.isnan(out["traj_curvature"])  # needs >= 3 points
    assert np.isnan(out["delta_oct13_aug14"])
    assert abs(out["delta_aug14_jun19"] - 2.0) < 1e-9


def test_trajectory_all_missing():
    means = {k: np.nan for k in ("jun06", "jun19", "aug14", "oct13")}
    out = trajectory_features(means)
    for v in out.values():
        assert np.isnan(v)
