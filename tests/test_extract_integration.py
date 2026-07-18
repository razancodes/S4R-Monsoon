"""Integration test against the real competition data (skipped if absent).

These assertions encode externally known facts from the legacy analysis — if
extraction reproduces them from raw rasters, the pipeline is reading the data
correctly:
- 29 villages, area sum 21,006.71 ha
- exactly-zero SAR coverage for IDs 1 (Manpura), 12 (Kotna), 25 (Pilol), 27 (Alindra)
- very low coverage (< 5%) for IDs 3 (Sankhyad), 5 (Khanpur), 11 (Chhani)
"""

from pathlib import Path

import numpy as np
import pytest

from s4r import config

DATA_DIR = Path(__file__).resolve().parents[1] / (
    "anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge_copy"
)

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="real competition data not present"
)


@pytest.fixture(scope="module")
def features():
    from s4r.features.extract import extract_features

    return extract_features(DATA_DIR)


def test_shape_and_schema(features):
    assert len(features) == config.N_VILLAGES
    for col in config.REQUIRED_COLUMNS:
        assert col in features.columns
    assert not features["is_synthetic"].any()


def test_area_sum(features):
    assert abs(features["area_ha"].sum() - config.TOTAL_LANDMASS_HA) < 0.1


def test_known_zero_coverage_villages(features):
    df = features.set_index("village_id")
    for vid in config.ZERO_COVERAGE_IDS:
        for d in config.DATES:
            assert df.loc[vid, f"coverage_{d}"] == 0.0, (vid, d)
        assert np.isnan(df.loc[vid, f"mean_{d}"])


def test_known_low_coverage_villages(features):
    df = features.set_index("village_id")
    for vid in config.LOW_COVERAGE_IDS:
        mean_cov = np.mean([df.loc[vid, f"coverage_{d}"] for d in config.DATES])
        assert 0.0 < mean_cov < 0.05, (vid, mean_cov)


def test_stats_finite_exactly_where_covered(features):
    # Real coverage is a continuum (many villages are partially covered);
    # the invariant is: finite stats iff the date has any valid pixels.
    for d in config.DATES:
        covered = features[f"coverage_{d}"] > 0
        assert np.isfinite(features.loc[covered, f"mean_{d}"]).all()
        assert np.isfinite(features.loc[covered, f"std_{d}"]).all()
        assert features.loc[~covered, f"mean_{d}"].isna().all()


def test_well_covered_villages_have_all_derived_features(features):
    df = features[~features["village_id"].isin(config.ZERO_COVERAGE_IDS + config.LOW_COVERAGE_IDS)]
    mean_cov = df[[f"coverage_{d}" for d in config.DATES]].mean(axis=1)
    assert (mean_cov > 0.05).all()
    fully = df[(df[[f"coverage_{d}" for d in config.DATES]] > 0).all(axis=1)]
    for col in ("flood_frac_avg", "traj_slope", "traj_range", "traj_curvature",
                "delta_aug14_jun19", "delta_oct13_aug14"):
        assert np.isfinite(fully[col]).all(), col


def test_backscatter_in_plausible_db_range(features):
    for d in config.DATES:
        vals = features[f"mean_{d}"].dropna()
        assert (vals > 0).all() and (vals < 48.2).all()  # uint8 dB range (0, 20*log10(255))


def test_cli_extract_roundtrips_through_ingestion(tmp_path):
    import subprocess
    import sys

    from s4r.data.ingest import load_features

    out = tmp_path / "village_features.csv"
    res = subprocess.run(
        [sys.executable, "-m", "s4r.cli", "extract", "--data-dir", str(DATA_DIR), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    df = load_features(out)  # must satisfy the full ingestion contract
    assert len(df) == config.N_VILLAGES
    assert not df["is_synthetic"].any()


def test_village_names_match_shapefile(features):
    df = features.set_index("village_id")
    assert df.loc[1, "village_name"] == "Manpura"
    assert df.loc[13, "village_name"] == "Koyali"
    assert df.loc[17, "village_name"] == "Angadh"
    assert df.loc[24, "village_name"] == "Asoj"
