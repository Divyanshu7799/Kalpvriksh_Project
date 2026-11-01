import numpy as np
import pytest

from machineguard.features.bearing_frequencies import defect_frequencies_hz
from machineguard.features.frequency_domain import band_energy_ratio, envelope_peak_ratio
from machineguard.features.registry import compute_features

SAMPLE_RATE_HZ = 12000
WINDOW_SIZE = 2048
RPM = 1797


def noise_windows(count=4, seed=0):
    return np.random.default_rng(seed).normal(size=(count, WINDOW_SIZE))


def bursts_at(defect_hz, count=4, seed=0):
    """Noise plus a 4 kHz carrier whose amplitude is switched on and off defect_hz times per second."""
    time_s = np.arange(WINDOW_SIZE) / SAMPLE_RATE_HZ
    carrier = np.sin(2 * np.pi * 4000 * time_s)
    modulation = 1 + np.cos(2 * np.pi * defect_hz * time_s)
    return noise_windows(count, seed) * 0.2 + 3 * modulation * carrier


def test_band_energy_ratio_is_one_for_a_tone_inside_the_band():
    time_s = np.arange(WINDOW_SIZE) / SAMPLE_RATE_HZ
    windows = np.tile(np.sin(2 * np.pi * 4000 * time_s), (2, 1))

    assert band_energy_ratio(windows, SAMPLE_RATE_HZ) == pytest.approx([1.0, 1.0], abs=0.01)


def test_band_energy_ratio_is_zero_for_a_tone_outside_the_band():
    time_s = np.arange(WINDOW_SIZE) / SAMPLE_RATE_HZ
    windows = np.tile(np.sin(2 * np.pi * 500 * time_s), (2, 1))

    assert band_energy_ratio(windows, SAMPLE_RATE_HZ) == pytest.approx([0.0, 0.0], abs=0.01)


def test_band_energy_ratio_of_an_all_zero_window_is_zero_not_nan():
    assert band_energy_ratio(np.zeros((1, WINDOW_SIZE)), SAMPLE_RATE_HZ)[0] == 0.0


def test_envelope_peak_ratio_is_large_at_the_true_defect_frequency():
    bpfo_hz = defect_frequencies_hz(RPM)["BPFO"]

    ratio = envelope_peak_ratio(bursts_at(bpfo_hz), SAMPLE_RATE_HZ, bpfo_hz)

    assert np.all(ratio > 10)


def test_envelope_peak_ratio_is_small_at_a_different_defect_frequency():
    frequencies = defect_frequencies_hz(RPM)

    ratio = envelope_peak_ratio(bursts_at(frequencies["BPFO"]), SAMPLE_RATE_HZ, frequencies["BPFI"])

    assert np.all(ratio < 5)


def test_envelope_peak_ratio_for_plain_noise_is_small():
    ratio = envelope_peak_ratio(noise_windows(), SAMPLE_RATE_HZ, defect_frequencies_hz(RPM)["BPFO"])

    assert np.all(ratio < 5)


def test_compute_features_returns_one_column_per_name_and_one_row_per_window():
    table = compute_features(["rms", "band_energy_ratio", "env_bpfo"], noise_windows(3), SAMPLE_RATE_HZ, RPM)

    assert list(table.columns) == ["rms", "band_energy_ratio", "env_bpfo"]
    assert len(table) == 3


def test_compute_features_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown features"):
        compute_features(["not_a_feature"], noise_windows(), SAMPLE_RATE_HZ, RPM)
