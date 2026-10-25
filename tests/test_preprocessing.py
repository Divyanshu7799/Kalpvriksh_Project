import numpy as np
import pytest

from machineguard.data.preprocessing import resample_signal


def test_resample_48khz_to_12khz_keeps_expected_length():
    original_signal = np.random.default_rng(0).normal(size=4801)
    original_copy = original_signal.copy()

    resampled_signal = resample_signal(
        original_signal,
        original_rate_hz=48000,
        target_rate_hz=12000,
    )

    assert resampled_signal.size == (original_signal.size + 3) // 4
    assert np.isfinite(resampled_signal).all()
    assert np.array_equal(original_signal, original_copy)


def test_resample_rejects_empty_signal():
    with pytest.raises(ValueError, match="Signal must not be empty"):
        resample_signal(
            np.array([]),
            original_rate_hz=12000,
            target_rate_hz=12000,
        )


def test_resample_same_rate_returns_copy():
    original_signal = np.array([1.0, 2.0, 3.0])

    result = resample_signal(
        original_signal,
        original_rate_hz=12000,
        target_rate_hz=12000,
    )

    assert np.array_equal(result, original_signal)
    assert result is not original_signal
