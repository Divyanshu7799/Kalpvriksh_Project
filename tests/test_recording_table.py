import numpy as np
import pandas as pd
import pytest

from machineguard.data.recording_table import build_window_table, file_name_of, load_recording_table


def write_tables(tmp_path, labels_rows):
    pd.DataFrame(
        {
            "source_file": ["data\\raw\\cwru\\BF\\118.mat", "data/raw/cwru/norml/97.mat"],
            "output_file": ["BF_118_DE.npy", "norml_97_DE.npy"],
        }
    ).to_csv(tmp_path / "manifest.csv", index=False)
    pd.DataFrame(labels_rows).to_csv(tmp_path / "labels.csv", index=False)
    return tmp_path / "manifest.csv", tmp_path / "labels.csv"


def good_labels():
    return {
        "file": ["118.mat", "97.mat"],
        "fault_type": ["B", "Normal"],
        "size_in": [0.007, 0],
        "load_hp": [0, 0],
    }


def test_file_name_of_accepts_windows_and_posix_paths():
    assert file_name_of("data\\raw\\cwru\\BF\\118.mat") == "118.mat"
    assert file_name_of("data/raw/cwru/BF/118.mat") == "118.mat"


def test_recordings_are_joined_with_their_labels(tmp_path):
    manifest_file, labels_file = write_tables(tmp_path, good_labels())

    table = load_recording_table(manifest_file, labels_file)

    assert table.set_index("file").loc["118.mat", "fault_type"] == "B"
    assert table.set_index("file").loc["97.mat", "fault_type"] == "Normal"


def test_duplicate_label_file_names_are_rejected(tmp_path):
    labels = good_labels()
    labels = {key: values + [values[0]] for key, values in labels.items()}
    manifest_file, labels_file = write_tables(tmp_path, labels)

    with pytest.raises(ValueError, match="duplicate file names"):
        load_recording_table(manifest_file, labels_file)


def test_recording_without_a_label_is_rejected(tmp_path):
    labels = {key: values[:1] for key, values in good_labels().items()}
    manifest_file, labels_file = write_tables(tmp_path, labels)

    with pytest.raises(ValueError, match="without a known load or fault type"):
        load_recording_table(manifest_file, labels_file)


def test_window_table_has_one_row_per_window_with_labels(tmp_path):
    np.save(tmp_path / "BF_118_DE.npy", np.arange(4096, dtype=float))
    recordings = pd.DataFrame(
        {"file": ["118.mat"], "output_file": ["BF_118_DE.npy"], "fault_type": ["B"], "load_hp": [2.0]}
    )

    table = build_window_table(recordings, tmp_path, ["rms"], window_size=2048, overlap=0.5)

    assert len(table) == 3
    assert set(table["fault_type"]) == {"B"}
    assert set(table["load_hp"]) == {2}
    assert list(table.columns) == ["rms", "file", "fault_type", "load_hp"]
