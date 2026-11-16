import numpy as np
import pytest

from machineguard.data.preprocessing import split_into_windows


def test_split_signal_into_overlapping_windows():
    signal = np.arange(5000)

    windows = split_into_windows(signal)

    assert windows.shape == (3, 2048)
    np.testing.assert_array_equal(windows[0], signal[:2048])
    np.testing.assert_array_equal(windows[1], signal[1024:3072])
    np.testing.assert_array_equal(windows[2], signal[2048:4096])


def test_signal_shorter_than_window_returns_empty_array():
    windows = split_into_windows(np.arange(100))

    assert windows.shape == (0, 2048)


def test_signal_equal_to_window_produces_one_window():
    windows = split_into_windows(np.arange(2048))

    assert windows.shape == (1, 2048)


def test_empty_signal_is_rejected():
    with pytest.raises(ValueError, match="Signal must not be empty"):
        split_into_windows([])


@pytest.mark.parametrize("window_size", [0, -1])
def test_non_positive_window_size_is_rejected(window_size):
    with pytest.raises(ValueError, match="Window size must be positive"):
        split_into_windows(np.arange(10), window_size=window_size)


@pytest.mark.parametrize("overlap", [-0.1, 1, 1.1])
def test_invalid_overlap_is_rejected(overlap):
    with pytest.raises(ValueError, match="Overlap must be"):
        split_into_windows(np.arange(10), overlap=overlap)


def test_a_quarter_overlap_moves_each_window_by_three_quarters_of_its_length():
    signal = np.arange(1000)

    windows = split_into_windows(signal, window_size=100, overlap=0.25)

    assert windows.shape == (13, 100)  # starts 0, 75, ... 900; start 975 would run past the end
    np.testing.assert_array_equal(windows[1], signal[75:175])
