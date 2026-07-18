import numpy as np
import pandas as pd
import pytest

from s4r import config
from s4r.data.ingest import DataValidationError, load_features, load_sample_submission, standardize_features
from s4r.data.synthetic import make_synthetic_features


@pytest.fixture
def features_csv(tmp_path):
    path = tmp_path / "village_features.csv"
    make_synthetic_features().to_csv(path, index=False)
    return path


def test_load_features_ok(features_csv):
    df = load_features(features_csv)
    assert len(df) == config.N_VILLAGES
    assert df["is_synthetic"].all()


def test_load_features_marks_real_when_flag_absent(tmp_path):
    df = make_synthetic_features().drop(columns=["is_synthetic"])
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    out = load_features(path)
    assert not out["is_synthetic"].any()


def test_missing_column_raises(tmp_path):
    df = make_synthetic_features().drop(columns=["flood_frac_avg"])
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="flood_frac_avg"):
        load_features(path)


def test_wrong_row_count_raises(tmp_path):
    df = make_synthetic_features().iloc[:28]
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="29"):
        load_features(path)


def test_bad_area_raises(tmp_path):
    df = make_synthetic_features()
    df.loc[0, "area_ha"] = -5.0
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="area_ha"):
        load_features(path)


def test_duplicate_id_raises(tmp_path):
    df = make_synthetic_features()
    df.loc[1, "village_id"] = df.loc[0, "village_id"]
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="duplicate"):
        load_features(path)


def test_load_sample_submission(tmp_path):
    sample = pd.DataFrame({c: (range(29) if c == "ID" else np.zeros(29)) for c in config.SUBMISSION_COLUMNS})
    path = tmp_path / "sample.csv"
    sample.to_csv(path, index=False)
    df = load_sample_submission(path)
    assert list(df.columns) == config.SUBMISSION_COLUMNS


def test_load_sample_submission_bad_columns(tmp_path):
    sample = pd.DataFrame({"ID": range(29), "Rice_ha": np.zeros(29)})
    path = tmp_path / "sample.csv"
    sample.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_sample_submission(path)


def test_standardize_features(features_csv):
    df = load_features(features_csv)
    X = standardize_features(df)
    assert X.shape == (config.N_VILLAGES, len(config.MODEL_FEATURES))
    assert np.isfinite(X).all()
    # zero-coverage rows had NaN features -> exactly 0 after fill
    zero_rows = df["village_id"].isin(config.ZERO_COVERAGE_IDS).to_numpy()
    assert np.allclose(X[zero_rows], 0.0)
    # covered columns should be ~zero-mean
    assert abs(np.nanmean(X[~zero_rows, 0])) < 0.5
