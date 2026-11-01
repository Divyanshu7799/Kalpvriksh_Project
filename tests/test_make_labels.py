import csv

from machineguard.data.make_labels import build_labels, lookup_condition


def test_build_labels_writes_expected_metadata(tmp_path):
    raw_data_dir = tmp_path / "raw" / "cwru"
    normal_dir = raw_data_dir / "norml"
    normal_dir.mkdir(parents=True)

    # Dummy filenames are sufficient; label generation does not load MAT data.
    (normal_dir / "97.mat").touch()
    (normal_dir / "98.mat").touch()

    output_path = tmp_path / "interim" / "cwru" / "labels.csv"

    rows = build_labels(data_root=raw_data_dir, output_path=output_path)

    assert len(rows) == 2
    assert output_path.is_file()

    with output_path.open(newline="", encoding="utf-8") as file:
        saved_rows = list(csv.DictReader(file))

    assert len(saved_rows) == 2
    assert saved_rows[0]["fault_type"] == "Normal"
    assert saved_rows[0]["rpm"] == "1797"
    assert saved_rows[1]["fault_type"] == "Normal"
    assert saved_rows[1]["rpm"] == "1772"


def test_file_id_maps_to_size_position_and_load():
    assert lookup_condition("IR", 106) == (0.007, "-", 1)
    assert lookup_condition("OR", 237) == (0.021, "@6", 3)


def test_unknown_file_id_is_not_guessed():
    assert lookup_condition("IR", 9999) is None


def test_file_with_unknown_id_gets_question_marks(tmp_path):
    folder = tmp_path / "raw" / "IR"
    folder.mkdir(parents=True)
    (folder / "9999.mat").touch()

    rows = build_labels(data_root=tmp_path / "raw", output_path=tmp_path / "labels.csv")

    assert rows == [["9999.mat", "IR", "?", "?", "?", "?"]]
