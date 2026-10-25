import numpy as np

from machineguard.features.time_domain import crest_factor_per_window, kurtosis_per_window, rms_per_window


def test_rms_of_each_window():
    windows = np.array([[3.0, 4.0], [0.0, 0.0]])

    np.testing.assert_allclose(rms_per_window(windows), [np.sqrt(12.5), 0.0])


def test_kurtosis_of_alternating_signal_is_minus_two():
    windows = np.array([[-1.0, 1.0, -1.0, 1.0]])

    np.testing.assert_allclose(kurtosis_per_window(windows), [-2.0])


def test_kurtosis_is_larger_for_a_spiky_window_than_for_a_flat_one():
    flat = np.tile([-1.0, 1.0], 50)
    spiky = np.zeros(100)
    spiky[10] = 10.0
    spiky[60] = -10.0

    values = kurtosis_per_window(np.array([flat, spiky]))

    assert values[1] > values[0]


def test_crest_factor_is_peak_over_rms():
    windows = np.array([[3.0, 4.0]])

    np.testing.assert_allclose(crest_factor_per_window(windows), [4.0 / np.sqrt(12.5)])
