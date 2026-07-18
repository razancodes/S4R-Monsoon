import numpy as np

from s4r import config
from s4r.data.synthetic import make_synthetic_features
from s4r.fallback.baseline import baseline_allocation


def test_grand_total_hits_point_estimate():
    df = make_synthetic_features()
    pred = baseline_allocation(df["area_ha"].to_numpy())
    assert abs(pred.sum() - config.TOTAL_AREA_POINT) < 0.5


def test_aggregate_mix_matches_priors():
    df = make_synthetic_features()
    pred = baseline_allocation(df["area_ha"].to_numpy())
    shares = pred.sum(axis=0) / pred.sum()
    assert np.allclose(shares, config.MIX_VECTOR, atol=1e-9)


def test_cap_and_nonnegativity():
    df = make_synthetic_features()
    area = df["area_ha"].to_numpy()
    pred = baseline_allocation(area)
    assert (pred >= 0).all()
    assert (pred.sum(axis=1) <= config.ALPHA_CAP * area + 1e-9).all()


def test_shape():
    area = make_synthetic_features()["area_ha"].to_numpy()
    assert baseline_allocation(area).shape == (config.N_VILLAGES, len(config.CROPS))
