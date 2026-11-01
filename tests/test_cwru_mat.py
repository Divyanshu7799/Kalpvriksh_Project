from pathlib import Path

import numpy as np

from machineguard.data.cwru_mat import (
    calculate_signal_hash,
    discover_signal_variables,
    extract_recorded_rpm,
    get_sampling_rate_hz,
)


def test_normal_baseline_files_are_48_khz():
    assert get_sampling_rate_hz(Path("raw/norml/97.mat")) == 48000


def test_fault_folders_are_12_khz():
    assert get_sampling_rate_hz(Path("raw/BF/118.mat")) == 12000


def test_unknown_recording_has_no_sampling_rate_instead_of_a_guess():
    assert get_sampling_rate_hz(Path("raw/other/1.mat")) is None
    assert get_sampling_rate_hz(Path("raw/norml/12345.mat")) is None


def test_only_vibration_signals_are_discovered():
    mat_data = {"__header__": b"x", "X097_DE_time": np.zeros((4, 1)), "X097RPM": np.array([[1797]])}

    signals = discover_signal_variables(mat_data)

    assert [(s["variable"], s["internal_id"], s["channel"]) for s in signals] == [("X097_DE_time", "097", "DE")]


def test_rpm_values_are_collected_without_duplicates():
    mat_data = {"X097RPM": np.array([[1797]]), "X098RPM": np.array([[1797]]), "X099RPM": np.array([[1772]])}

    assert extract_recorded_rpm(mat_data) == [1772, 1797]


def test_equal_signals_have_equal_hashes_and_different_signals_do_not():
    signal = np.arange(10.0)

    assert calculate_signal_hash(signal) == calculate_signal_hash(signal.copy())
    assert calculate_signal_hash(signal) != calculate_signal_hash(signal + 1)
