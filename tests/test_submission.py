import numpy as np
import pandas as pd
import pytest

from s4r import config
from s4r.data.synthetic import make_synthetic_features
from s4r.fallback.baseline import baseline_allocation
from s4r.submission.writer import (
    SubmissionError,
    comparison_report,
    validate_predictions,
    write_submission,
)


@pytest.fixture
def features_df():
    return make_synthetic_features()


@pytest.fixture
def sample_df(features_df):
    # sample IDs are village_ids in reversed order to prove order mapping works
    ids = features_df["village_id"].tolist()[::-1]
    data = {"ID": ids}
    for c in config.SUBMISSION_COLUMNS[1:]:
        data[c] = np.zeros(len(ids))
    return pd.DataFrame(data)


@pytest.fixture
def pred(features_df):
    return baseline_allocation(features_df["area_ha"].to_numpy())


def test_validate_ok(pred, features_df):
    area = features_df["area_ha"].to_numpy()
    assert validate_predictions(pred, area, config.ALPHA_CAP) == []


def test_validate_flags_negative(pred, features_df):
    bad = pred.copy()
    bad[0, 0] = -1.0
    msgs = validate_predictions(bad, features_df["area_ha"].to_numpy(), config.ALPHA_CAP)
    assert any("negative" in m for m in msgs)


def test_validate_flags_cap(pred, features_df):
    area = features_df["area_ha"].to_numpy()
    bad = pred.copy()
    bad[2, :] = area[2]  # 5x area total, way over cap
    msgs = validate_predictions(bad, area, config.ALPHA_CAP)
    assert any("cap" in m for m in msgs)


def test_validate_flags_total_band(features_df):
    area = features_df["area_ha"].to_numpy()
    tiny = baseline_allocation(area) * 0.1
    msgs = validate_predictions(tiny, area, config.ALPHA_CAP)
    assert any("band" in m for m in msgs)


def test_write_refuses_synthetic(pred, features_df, sample_df, tmp_path):
    with pytest.raises(SubmissionError, match="synthetic"):
        write_submission(pred, features_df, sample_df, tmp_path / "sub.csv")


def test_write_roundtrip_in_sample_order(pred, features_df, sample_df, tmp_path):
    out = write_submission(
        pred, features_df, sample_df, tmp_path / "sub.csv", allow_synthetic=True
    )
    written = pd.read_csv(out)
    assert list(written.columns) == config.SUBMISSION_COLUMNS
    assert len(written) == config.N_VILLAGES
    assert written["ID"].tolist() == sample_df["ID"].tolist()
    # row for last sample ID (first village in features order) matches pred row 0
    row = written.iloc[-1]
    assert row["ID"] == features_df["village_id"].iloc[0]
    assert abs(row["Rice_ha"] - pred[0, config.CROPS.index("Rice")]) < 1e-9


def test_write_refuses_violations(features_df, sample_df, tmp_path):
    area = features_df["area_ha"].to_numpy()
    bad = baseline_allocation(area)
    bad[0, 0] = -5.0
    with pytest.raises(SubmissionError, match="negative"):
        write_submission(bad, features_df, sample_df, tmp_path / "s.csv", allow_synthetic=True)


def test_write_refuses_id_mismatch(pred, features_df, sample_df, tmp_path):
    sample_df.loc[0, "ID"] = 999
    with pytest.raises(SubmissionError, match="ID"):
        write_submission(pred, features_df, sample_df, tmp_path / "s.csv", allow_synthetic=True)


def test_comparison_report(pred, features_df):
    report = comparison_report({"baseline": pred, "route_c": pred * 1.01}, features_df, config.ALPHA_CAP)
    assert len(report) == config.N_VILLAGES
    assert "baseline_total_ha" in report.columns and "route_c_total_ha" in report.columns
    assert "cap_ha" in report.columns
    assert not report["route_c_over_cap"].any()
