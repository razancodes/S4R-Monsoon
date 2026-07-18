import numpy as np

from s4r import config
from s4r.data.synthetic import make_synthetic_features


def test_shape_and_columns():
    df = make_synthetic_features()
    assert len(df) == config.N_VILLAGES
    for col in config.REQUIRED_COLUMNS:
        assert col in df.columns
    assert df["is_synthetic"].all()


def test_area_sum_matches_landmass():
    df = make_synthetic_features()
    assert abs(df["area_ha"].sum() - config.TOTAL_LANDMASS_HA) < 0.01


def test_zero_coverage_villages():
    df = make_synthetic_features().set_index("village_id")
    for vid in config.ZERO_COVERAGE_IDS:
        row = df.loc[vid]
        for d in config.DATES:
            assert row[f"coverage_{d}"] == 0.0
            assert np.isnan(row[f"mean_{d}"])


def test_low_coverage_villages():
    df = make_synthetic_features().set_index("village_id")
    expected = {3: 0.004, 5: 0.012, 11: 0.006}
    for vid, cov in expected.items():
        covs = [df.loc[vid][f"coverage_{d}"] for d in config.DATES]
        assert abs(np.mean(covs) - cov) < 0.01


def test_known_village_names():
    df = make_synthetic_features().set_index("village_id")
    assert df.loc[1, "village_name"] == "Manpura"
    assert df.loc[12, "village_name"] == "Kotna"
    assert df.loc[25, "village_name"] == "Pilol"
    assert df.loc[27, "village_name"] == "Alindra"
    assert "Koyali" in set(df["village_name"])
    assert "Angadh" in set(df["village_name"])


def test_deterministic():
    a = make_synthetic_features(seed=7)
    b = make_synthetic_features(seed=7)
    assert a.equals(b)
