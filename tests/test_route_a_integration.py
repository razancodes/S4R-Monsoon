"""Route A Capella chip extraction against the real competition rasters.

Auto-skips when the raw data directory (or torch) is absent, mirroring
tests/test_extract_integration.py.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from s4r import config

DATA_DIR = Path("anrf-aise-hack-2026-round-1-sar-crop-mapping-challenge_copy")

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="raw competition data not present"
)


@pytest.fixture(scope="module")
def chips_and_ids():
    from s4r.route_a.data import village_chips

    return village_chips(DATA_DIR, chip_px=64)


def test_chip_shapes_and_order(chips_and_ids):
    chips, ids = chips_and_ids
    assert chips.shape == (config.N_VILLAGES, 1, 64, 64)
    assert list(ids) == list(range(1, config.N_VILLAGES + 1))
    assert torch.isfinite(chips).all()


def test_zero_coverage_villages_have_empty_chips(chips_and_ids):
    chips, ids = chips_and_ids
    for vid in config.ZERO_COVERAGE_IDS:
        i = list(ids).index(vid)
        assert float(chips[i].abs().sum()) == 0.0


def test_covered_villages_have_signal(chips_and_ids):
    chips, ids = chips_and_ids
    covered = [
        i
        for i, vid in enumerate(ids)
        if vid not in config.ZERO_COVERAGE_IDS + config.LOW_COVERAGE_IDS
    ]
    frac_nonzero = np.mean([float((chips[i] != 0).float().mean()) for i in covered])
    assert frac_nonzero > 0.3
