import json

import numpy as np
import pandas as pd
import pytest

from s4r import config
from s4r.data.ingest import standardize_features
from s4r.data.synthetic import make_synthetic_features
from s4r.fallback.head import forward, n_params
from s4r.fallback.train import TrainConfig, objective, train
from s4r.features.coverage import coverage_confidence
from s4r.losses.aggregate import cap_violations


@pytest.fixture(scope="module")
def inputs():
    df = make_synthetic_features()
    X = standardize_features(df)
    area = df["area_ha"].to_numpy()
    conf = coverage_confidence(df)
    return df, X, area, conf


@pytest.fixture(scope="module")
def trained(inputs, tmp_path_factory):
    _, X, area, conf = inputs
    cfg = TrainConfig(n_restarts=4, maxiter=200, seed=0)
    run_dir = tmp_path_factory.mktemp("runs")
    return train(X, area, conf, cfg, run_dir=str(run_dir)), cfg


def test_objective_finite(inputs):
    _, X, area, conf = inputs
    theta = np.zeros(n_params())
    val = objective(theta, X, area, conf, TrainConfig())
    assert np.isfinite(val)


def test_trained_total_in_band(trained, inputs):
    result, _ = trained
    total = result["pred"].sum()
    lo, hi = config.TOTAL_AREA_BAND
    assert lo - 1.0 <= total <= hi + 1.0


def test_trained_mix_near_priors(trained):
    result, _ = trained
    shares = result["pred"].sum(axis=0) / result["pred"].sum()
    assert np.abs(shares - config.MIX_VECTOR).max() <= config.MIX_TOL + 0.005


def test_no_cap_violations(trained, inputs):
    result, cfg = trained
    _, _, area, _ = inputs
    assert not cap_violations(result["pred"], area, cfg.alpha).any()


def test_loss_improves_over_init(trained, inputs):
    result, cfg = trained
    _, X, area, conf = inputs
    init = objective(np.zeros(n_params()), X, area, conf, cfg)
    assert result["loss"] <= init + 1e-9


def test_run_log_written(trained):
    result, _ = trained
    with open(result["run_log_path"]) as f:
        log = json.load(f)
    for key in ("config", "loss_components", "restart_losses", "aggregate_total", "aggregate_mix"):
        assert key in log
    assert log["config"]["alpha"] == config.ALPHA_CAP


def test_anchor_pulls_prediction(inputs):
    _, X, area, conf = inputs
    cfg = TrainConfig(n_restarts=2, maxiter=200, w_anchor=50.0, seed=1)
    anchors = pd.DataFrame(
        {"village_index": [0], "cultivated_fraction_est": [0.05], "weight": [1.0]}
    )
    res_anchored = train(X, area, conf, cfg, anchors=anchors, run_dir=None)
    frac0 = res_anchored["frac"][0]
    assert frac0 < config.BASELINE_FRAC  # pulled toward 0.05
