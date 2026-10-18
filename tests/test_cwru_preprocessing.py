import numpy as np
import pytest
from experiments.cwru import preprocess_recordings


def make_row(**overrides):
    """Create a minimal manifest row for testing."""
    row = {
        "file": "97.mat",
        "folder": "norml",
        "fault_type": "Normal",
        "channel": "DE",
        "internal_id": "097",
        "sampling_rate_hz": "48000",
        "duplicate_of": "",
    }
    row.update(overrides)
    return row


def test_duplicate_recording_is_skipped(tmp_path, monkeypatch):
    """Duplicate manifest entries should not load or process signals."""

    def unexpected_load(*args, **kwargs):
        pytest.fail("A duplicate recording should not be loaded.")

    monkeypatch.setattr(preprocess_recordings, "loadmat", unexpected_load)

    row = make_row(duplicate_of="data/raw/cwru/norml/98.mat:X098_DE_time")

    result = preprocess_recordings.process_recording(row, tmp_path)

    assert result is None


def test_missing_signal_variable_raises_error(tmp_path, monkeypatch):
    """A missing MATLAB signal should produce a clear error."""
    monkeypatch.setattr(preprocess_recordings, "loadmat", lambda path: {})

    with pytest.raises(KeyError, match="Missing signal"):
        preprocess_recordings.process_recording(make_row(), tmp_path)


def test_invalid_sampling_rate_is_rejected(tmp_path):
    """Missing or invalid sampling rates must not be guessed."""
    row = make_row(sampling_rate_hz="")

    with pytest.raises(ValueError, match="valid sampling rate"):
        preprocess_recordings.process_recording(row, tmp_path)


def test_valid_recording_is_resampled(tmp_path, monkeypatch):
    """A valid signal should be resampled and have correct metadata."""
    original_signal = np.arange(16, dtype=float).reshape(-1, 1)

    monkeypatch.setattr(
        preprocess_recordings,
        "loadmat",
        lambda path: {"X097_DE_time": original_signal},
    )

    result = preprocess_recordings.process_recording(make_row(), tmp_path)

    assert result is not None
    processed_signal, metadata = result

    assert processed_signal.size == 4
    assert np.isfinite(processed_signal).all()
    assert metadata["original_rate_hz"] == 48000
    assert metadata["target_rate_hz"] == 12000
    assert metadata["original_samples"] == 16
    assert metadata["processed_samples"] == 4
    assert metadata["fault_type"] == "Normal"
