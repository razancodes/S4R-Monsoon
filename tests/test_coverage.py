import numpy as np

from s4r import config
from s4r.data.synthetic import make_synthetic_features
from s4r.features.coverage import coverage_confidence


def test_zero_coverage_gives_zero_confidence():
    df = make_synthetic_features()
    c = coverage_confidence(df)
    zero_mask = df["village_id"].isin(config.ZERO_COVERAGE_IDS).to_numpy()
    assert np.allclose(c[zero_mask], 0.0)


def test_full_coverage_saturates_to_one():
    df = make_synthetic_features()
    for d in config.DATES:
        df[f"coverage_{d}"] = 1.0
    c = coverage_confidence(df)
    assert np.allclose(c, 1.0)


def test_monotone_in_coverage():
    df = make_synthetic_features()
    for d in config.DATES:
        df[f"coverage_{d}"] = 0.1
    c_low = coverage_confidence(df)[0]
    for d in config.DATES:
        df[f"coverage_{d}"] = 0.3
    c_high = coverage_confidence(df)[0]
    assert c_high > c_low


def test_range_and_shape():
    df = make_synthetic_features()
    c = coverage_confidence(df)
    assert c.shape == (config.N_VILLAGES,)
    assert (c >= 0).all() and (c <= 1).all()
