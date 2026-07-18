import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from s4r import config
from s4r.data.synthetic import make_synthetic_features


@pytest.fixture
def data_dir(tmp_path):
    df = make_synthetic_features()
    features = tmp_path / "village_features.csv"
    df.to_csv(features, index=False)
    sample = tmp_path / "sample.csv"
    data = {"ID": df["village_id"].tolist()}
    for c in config.SUBMISSION_COLUMNS[1:]:
        data[c] = np.zeros(len(df))
    pd.DataFrame(data).to_csv(sample, index=False)
    return tmp_path, features, sample


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "s4r.cli", *args],
        capture_output=True,
        text=True,
    )


def test_baseline_only_writes_submission(data_dir):
    tmp, features, sample = data_dir
    out = tmp / "sub.csv"
    res = _run(
        ["route-c", "--features", str(features), "--sample", str(sample),
         "--out", str(out), "--baseline-only", "--allow-synthetic",
         "--run-dir", str(tmp / "runs")]
    )
    assert res.returncode == 0, res.stderr
    written = pd.read_csv(out)
    assert list(written.columns) == config.SUBMISSION_COLUMNS
    assert abs(written.drop(columns="ID").to_numpy().sum() - config.TOTAL_AREA_POINT) < 1.0


def test_refuses_synthetic_without_flag(data_dir):
    tmp, features, sample = data_dir
    res = _run(
        ["route-c", "--features", str(features), "--sample", str(sample),
         "--out", str(tmp / "sub.csv"), "--baseline-only",
         "--run-dir", str(tmp / "runs")]
    )
    assert res.returncode == 1
    assert "synthetic" in res.stderr.lower()


def test_trained_route_c(data_dir):
    tmp, features, sample = data_dir
    out = tmp / "sub_trained.csv"
    res = _run(
        ["route-c", "--features", str(features), "--sample", str(sample),
         "--out", str(out), "--allow-synthetic", "--restarts", "2",
         "--run-dir", str(tmp / "runs")]
    )
    assert res.returncode == 0, res.stderr
    written = pd.read_csv(out)
    total = written.drop(columns="ID").to_numpy().sum()
    lo, hi = config.TOTAL_AREA_BAND
    assert lo - 1.0 <= total <= hi + 1.0
    assert "grand_total" in res.stdout or "route" in res.stdout
