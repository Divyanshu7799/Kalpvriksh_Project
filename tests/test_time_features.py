import numpy as np

from machineguard.features.time_domain import (
    clearance_factor_per_window,
    crest_factor_per_window,
    impulse_factor_per_window,
    kurtosis_per_window,
    mean_abs_per_window,
    peak_per_window,
    peak_to_peak_per_window,
    rms_per_window,
    shape_factor_per_window,
    skewness_per_window,
    std_per_window,
)


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


def test_std_ignores_a_constant_offset():
    windows = np.array([[1.0, 3.0], [11.0, 13.0]])

    np.testing.assert_allclose(std_per_window(windows), [1.0, 1.0])


def test_crest_factor_of_an_all_zero_window_is_nan_not_an_error():
    windows = np.zeros((1, 8))

    assert np.isnan(crest_factor_per_window(windows)[0])


# --- Candidate pool: the remaining standard time-domain statistics ------------------------------
def sine_window(samples=4000, cycles=10):
    return np.sin(2 * np.pi * cycles * np.arange(samples) / samples)[np.newaxis, :]


def test_peak_and_peak_to_peak():
    windows = np.array([[-3.0, 1.0, 2.0]])

    np.testing.assert_allclose(peak_per_window(windows), [3.0])
    np.testing.assert_allclose(peak_to_peak_per_window(windows), [5.0])


def test_mean_abs_ignores_the_sign():
    np.testing.assert_allclose(mean_abs_per_window(np.array([[-2.0, 2.0, -4.0, 4.0]])), [3.0])


def test_skewness_is_zero_for_a_symmetric_signal_and_positive_for_a_tail_to_the_right():
    symmetric = sine_window()
    right_tail = np.array([[0.0] * 99 + [10.0]])

    assert abs(skewness_per_window(symmetric)[0]) < 1e-9
    assert skewness_per_window(right_tail)[0] > 0


def test_factors_of_a_sine_wave_match_the_textbook_values():
    window = sine_window()

    np.testing.assert_allclose(crest_factor_per_window(window), [np.sqrt(2)], rtol=1e-3)
    np.testing.assert_allclose(shape_factor_per_window(window), [np.pi / (2 * np.sqrt(2))], rtol=1e-3)
    np.testing.assert_allclose(impulse_factor_per_window(window), [np.pi / 2], rtol=1e-3)


def test_factors_of_a_square_wave_are_all_one():
    window = np.tile([-1.0, 1.0], 50)[np.newaxis, :]

    for function in (
        crest_factor_per_window,
        shape_factor_per_window,
        impulse_factor_per_window,
        clearance_factor_per_window,
    ):
        np.testing.assert_allclose(function(window), [1.0])


def test_ratio_features_are_nan_for_an_all_zero_window():
    zeros = np.zeros((1, 100))

    for function in (shape_factor_per_window, impulse_factor_per_window, clearance_factor_per_window):
        assert np.isnan(function(zeros)[0])
