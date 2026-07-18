import numpy as np

from s4r import config
from s4r.data.ingest import standardize_features
from s4r.data.synthetic import make_synthetic_features
from s4r.fallback.baseline import baseline_allocation
from s4r.fallback.head import forward, n_params, unflatten
from s4r.features.coverage import coverage_confidence


def _inputs():
    df = make_synthetic_features()
    X = standardize_features(df)
    area = df["area_ha"].to_numpy()
    conf = coverage_confidence(df)
    return df, X, area, conf


def test_n_params():
    assert n_params() == (10 + 1) + (5 * 10 + 5)  # 66


def test_unflatten_shapes():
    w_t, b_t, W_s, b_s = unflatten(np.zeros(n_params()))
    assert w_t.shape == (10,) and np.isscalar(b_t) or b_t.shape == ()
    assert W_s.shape == (5, 10) and b_s.shape == (5,)


def test_forward_shapes():
    _, X, area, conf = _inputs()
    out = forward(np.zeros(n_params()), X, area, conf)
    assert out["pred"].shape == (config.N_VILLAGES, 5)
    assert out["totals"].shape == (config.N_VILLAGES,)
    assert out["frac"].shape == (config.N_VILLAGES,)
    assert out["shares"].shape == (config.N_VILLAGES, 5)


def test_structural_constraints_random_thetas():
    _, X, area, conf = _inputs()
    rng = np.random.default_rng(0)
    for _ in range(200):
        theta = rng.normal(0, 3, size=n_params())
        out = forward(theta, X, area, conf)
        assert (out["pred"] >= 0).all()
        assert (out["totals"] <= config.ALPHA_CAP * area + 1e-9).all()
        assert np.allclose(out["shares"].sum(axis=1), 1.0)


def test_zero_confidence_rows_equal_baseline():
    df, X, area, conf = _inputs()
    theta = np.random.default_rng(1).normal(0, 3, size=n_params())
    out = forward(theta, X, area, conf)
    base = baseline_allocation(area)
    zero_mask = df["village_id"].isin(config.ZERO_COVERAGE_IDS).to_numpy()
    assert np.allclose(out["pred"][zero_mask], base[zero_mask])
