import numpy as np
from scipy.io import loadmat

from machineguard.data.preprocessing import resample_signal


def test_resample_normal_de_signal():
    mat_data = loadmat("data/raw/cwru/norml/97.mat")
    original_signal = mat_data["X097_DE_time"].reshape(-1)
    original_copy = original_signal.copy()

    resampled_signal = resample_signal(
        original_signal,
        original_rate_hz=48000,
        target_rate_hz=12000,
    )

    assert resampled_signal.size == (original_signal.size + 3) // 4
    assert np.isfinite(resampled_signal).all()
    assert np.array_equal(original_signal, original_copy)
