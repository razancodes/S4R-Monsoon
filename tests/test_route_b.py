"""Route B: Sentinel-1 + OlmoEarth pseudo-label generation (pure logic).

Network fetch and model inference are exercised by the integration path only;
these tests cover scene selection, token aggregation, and the annotation
schema contract against s4r.weak_labels.ingest.
"""

from datetime import datetime

import numpy as np
import pytest

from s4r.data.synthetic import make_synthetic_features
from s4r.route_b.pseudo_labels import (
    build_annotations,
    fraction_from_clusters,
    identify_ag_clusters,
)
from s4r.route_b.s1_fetch import nearest_scenes
from s4r.weak_labels.ingest import load_weak_labels


@pytest.fixture
def features_df():
    return make_synthetic_features()


# --- scene selection ---------------------------------------------------------


def test_nearest_scene_per_target_date():
    scenes = [datetime(2025, 6, 4), datetime(2025, 6, 21), datetime(2025, 8, 10)]
    targets = [datetime(2025, 6, 6), datetime(2025, 8, 14)]
    assert nearest_scenes(scenes, targets, max_days=30) == [0, 2]


def test_nearest_scene_none_beyond_max_days():
    scenes = [datetime(2025, 6, 4)]
    targets = [datetime(2025, 10, 13)]
    assert nearest_scenes(scenes, targets, max_days=30) == [None]


# --- token aggregation -------------------------------------------------------


def test_fraction_from_clusters_counts_valid_ag_tokens():
    cluster_ids = np.array([[0, 1], [2, 1]])
    valid = np.ones((2, 2), dtype=bool)
    assert fraction_from_clusters(cluster_ids, {1}, valid) == pytest.approx(0.5)


def test_fraction_from_clusters_ignores_invalid_tokens():
    cluster_ids = np.array([[0, 1], [1, 1]])
    valid = np.array([[True, True], [False, False]])
    assert fraction_from_clusters(cluster_ids, {1}, valid) == pytest.approx(0.5)


def test_fraction_from_clusters_no_valid_tokens_is_none():
    cluster_ids = np.zeros((2, 2), dtype=int)
    valid = np.zeros((2, 2), dtype=bool)
    assert fraction_from_clusters(cluster_ids, {0}, valid) is None


def test_identify_ag_clusters_picks_high_seasonal_dynamics():
    # cluster 0: flat backscatter trajectory (urban/water); cluster 1: strong
    # monsoon dynamics (cropland); cluster 2: moderate.
    cluster_ids = np.array([0] * 10 + [1] * 10 + [2] * 10)
    vh_range = np.array([0.5] * 10 + [8.0] * 10 + [4.0] * 10)
    ag = identify_ag_clusters(cluster_ids, vh_range, min_range_db=3.0)
    assert ag == {1, 2}


# --- annotation schema contract ---------------------------------------------


def test_build_annotations_roundtrip_through_ingest(tmp_path, features_df):
    fractions = {8: 0.05, 14: 0.31, 1: 0.20}
    df = build_annotations(fractions, confidence=0.3, source="olmoearth_s1_v1")
    path = tmp_path / "annotations.csv"
    df.to_csv(path, index=False)
    anchors = load_weak_labels(path, features_df)
    assert len(anchors) == 3
    assert (anchors["weight"] == 0.3).all()
    # unsupervised pseudo-labels must not claim a crop type
    assert (anchors["dominant_crop"] == "").all()


def test_build_annotations_rejects_out_of_range_fraction():
    with pytest.raises(ValueError, match="fraction"):
        build_annotations({8: 1.3}, confidence=0.3, source="olmoearth_s1_v1")


def test_build_annotations_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        build_annotations({8: 0.5}, confidence=0.0, source="olmoearth_s1_v1")
