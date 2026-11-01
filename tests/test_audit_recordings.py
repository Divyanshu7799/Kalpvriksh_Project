import numpy as np
from experiments.cwru import audit_recordings
from scipy.io import savemat


def write_recording(folder, name, de_signal):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    savemat(path, {"X097_DE_time": de_signal, "X097RPM": np.array([[1797]])})
    return path


def test_second_copy_of_a_signal_is_marked_as_duplicate(tmp_path):
    signal = np.arange(10.0).reshape(-1, 1)
    first = write_recording(tmp_path / "norml", "97.mat", signal)
    second = write_recording(tmp_path / "norml", "98.mat", signal)
    first_seen = {}

    first_rows = audit_recordings.rows_for_recording(first, {}, first_seen)
    second_rows = audit_recordings.rows_for_recording(second, {}, first_seen)

    assert first_rows[0]["duplicate_of"] == ""
    assert second_rows[0]["duplicate_of"].endswith("97.mat:X097_DE_time")


def test_unlabelled_recording_keeps_empty_label_columns(tmp_path):
    path = write_recording(tmp_path / "BF", "118.mat", np.arange(10.0).reshape(-1, 1))

    row = audit_recordings.rows_for_recording(path, {}, {})[0]

    assert row["fault_type"] == ""
    assert row["rpm_mat"] == "1797"
    assert row["sampling_rate_hz"] == 12000
