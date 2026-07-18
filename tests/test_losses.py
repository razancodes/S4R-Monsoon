import numpy as np
import pandas as pd

from s4r import config
from s4r.losses.aggregate import (
    band_penalty,
    cap_violations,
    l2_penalty,
    loss_anchor,
    loss_mix,
    loss_shrink,
    loss_total,
)


def test_band_penalty_inside_zero():
    assert band_penalty(5300.0, 5200.0, 5500.0) == 0.0
    assert band_penalty(5200.0, 5200.0, 5500.0) == 0.0


def test_band_penalty_outside_quadratic():
    assert band_penalty(5600.0, 5200.0, 5500.0) == 100.0**2
    assert band_penalty(5100.0, 5200.0, 5500.0) == 100.0**2


def test_loss_total_scaling():
    totals = np.full(29, 5600.0 / 29)  # sums to 5600 -> 100 over
    assert abs(loss_total(totals) - 1.0) < 1e-9
    assert loss_total(np.full(29, 5350.0 / 29)) == 0.0


def test_loss_mix_zero_at_prior():
    pred = np.outer(np.full(29, 100.0), config.MIX_VECTOR)
    assert loss_mix(pred) == 0.0


def test_loss_mix_penalizes_deviation():
    uniform = np.full((29, 5), 100.0)
    assert loss_mix(uniform) > 0.0


def test_loss_shrink_zero_at_full_confidence():
    frac = np.full(29, 0.1)
    shares = np.tile(config.MIX_VECTOR, (29, 1))
    assert loss_shrink(frac, shares, np.ones(29)) == 0.0


def test_loss_shrink_zero_at_baseline():
    frac = np.full(29, config.BASELINE_FRAC)
    shares = np.tile(config.MIX_VECTOR, (29, 1))
    assert loss_shrink(frac, shares, np.zeros(29)) < 1e-12


def test_loss_shrink_positive_when_deviating_with_low_conf():
    frac = np.full(29, config.BASELINE_FRAC * 1.5)
    shares = np.tile(config.MIX_VECTOR, (29, 1))
    assert loss_shrink(frac, shares, np.zeros(29)) > 0.0


def test_loss_anchor_none_is_zero():
    assert loss_anchor(np.full(29, 0.25), None) == 0.0
    empty = pd.DataFrame(columns=["village_index", "cultivated_fraction_est", "weight"])
    assert loss_anchor(np.full(29, 0.25), empty) == 0.0


def test_loss_anchor_arithmetic():
    frac = np.full(29, 0.25)
    anchors = pd.DataFrame(
        {"village_index": [0], "cultivated_fraction_est": [0.05], "weight": [1.0]}
    )
    expected = (0.25 - 0.05) ** 2 / config.BASELINE_FRAC**2
    assert abs(loss_anchor(frac, anchors) - expected) < 1e-9


def test_l2_penalty():
    theta = np.array([3.0, 4.0])
    assert abs(l2_penalty(theta, 0.1) - 0.1 * 25.0) < 1e-12


def test_cap_violations_mask():
    area = np.full(29, 100.0)
    pred = np.zeros((29, 5))
    pred[3, :] = 10.0  # total 50 > 38 = 0.38*100
    mask = cap_violations(pred, area, alpha=0.38)
    assert mask[3] and mask.sum() == 1
