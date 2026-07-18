import numpy as np

from s4r import config


def test_mix_sums_to_one():
    assert abs(sum(config.REGIONAL_MIX.values()) - 1.0) < 1e-9


def test_mix_vector_follows_crop_order():
    for i, crop in enumerate(config.CROPS):
        assert config.MIX_VECTOR[i] == config.REGIONAL_MIX[crop]
    assert isinstance(config.MIX_VECTOR, np.ndarray)


def test_submission_columns_exact():
    assert config.SUBMISSION_COLUMNS == [
        "ID", "Rice_ha", "Cotton_ha", "Maize_ha", "Bajra_ha", "Groundnut_ha",
    ]


def test_baseline_frac_below_cap():
    assert abs(config.BASELINE_FRAC - 5269.0 / 21006.71) < 1e-12
    assert 0.24 < config.BASELINE_FRAC < 0.26
    assert config.BASELINE_FRAC < config.ALPHA_CAP


def test_alpha_in_spec_range():
    assert 0.35 <= config.ALPHA_CAP <= 0.40


def test_model_features_subset_of_schema():
    assert set(config.MODEL_FEATURES) <= set(config.FEATURE_COLUMNS)
    assert len(config.MODEL_FEATURES) == 10


def test_required_columns_include_identity():
    for col in ("village_id", "village_name", "area_ha"):
        assert col in config.REQUIRED_COLUMNS


def test_total_area_band():
    lo, hi = config.TOTAL_AREA_BAND
    assert lo == 5200.0 and hi == 5500.0
    assert lo <= config.TOTAL_AREA_POINT <= hi
