import pytest

from s4r.data.synthetic import make_synthetic_features
from s4r.weak_labels.ingest import WeakLabelError, load_weak_labels


@pytest.fixture
def features_df():
    return make_synthetic_features()


def _write(tmp_path, rows):
    path = tmp_path / "annotations.csv"
    header = "village_id,cultivated_fraction_est,dominant_crop,confidence,source,notes\n"
    path.write_text(header + "\n".join(rows) + "\n")
    return path


def test_valid_file_loads(tmp_path, features_df):
    path = _write(
        tmp_path,
        [
            '8,0.05,,0.9,"Google Earth Pro 2025-08","Koyali refinery - near-zero cropland"',
            '14,0.30,Cotton,0.6,"Bing Maps","Angadh capacity check"',
        ],
    )
    anchors = load_weak_labels(path, features_df)
    assert len(anchors) == 2
    assert set(anchors.columns) >= {"village_index", "cultivated_fraction_est", "weight"}
    # village_index maps to row position of village_id in features_df
    assert features_df.iloc[anchors["village_index"].iloc[0]]["village_id"] == 8
    assert anchors["weight"].iloc[1] == 0.6


def test_unknown_village_raises(tmp_path, features_df):
    path = _write(tmp_path, ['999,0.3,Rice,0.5,"src","note"'])
    with pytest.raises(WeakLabelError, match="999"):
        load_weak_labels(path, features_df)


def test_fraction_out_of_range_raises(tmp_path, features_df):
    path = _write(tmp_path, ['8,1.2,Rice,0.5,"src","note"'])
    with pytest.raises(WeakLabelError, match="fraction"):
        load_weak_labels(path, features_df)


def test_bad_crop_raises(tmp_path, features_df):
    path = _write(tmp_path, ['8,0.3,Wheat,0.5,"src","note"'])
    with pytest.raises(WeakLabelError, match="Wheat"):
        load_weak_labels(path, features_df)


def test_bad_confidence_raises(tmp_path, features_df):
    path = _write(tmp_path, ['8,0.3,Rice,0.0,"src","note"'])
    with pytest.raises(WeakLabelError, match="confidence"):
        load_weak_labels(path, features_df)
