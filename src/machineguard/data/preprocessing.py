from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_signal(signal, original_rate_hz, target_rate_hz):
    """Resample a vibration signal to a target sampling rate."""
    if original_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("Sampling rates must be positive.")

    signal = np.asarray(signal, dtype=float).reshape(-1)

    if signal.size == 0:
        raise ValueError("Signal must not be empty.")

    if original_rate_hz == target_rate_hz:
        return signal.copy()

    common_divisor = gcd(original_rate_hz, target_rate_hz)
    up = target_rate_hz // common_divisor
    down = original_rate_hz // common_divisor

    return resample_poly(signal, up, down)


def split_into_windows(signal, window_size=2048, overlap=0.5):
    """Split a vibration signal into fixed-size, overlapping windows."""
    signal = np.asarray(signal, dtype=float).reshape(-1)

    if signal.size == 0:
        raise ValueError("Signal must not be empty.")

    if window_size <= 0:
        raise ValueError("Window size must be positive.")

    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in the range [0, 1).")

    step_size = int(window_size * (1 - overlap))

    if step_size <= 0:
        raise ValueError("Overlap produces an invalid step size.")

    if signal.size < window_size:
        return np.empty((0, window_size), dtype=float)

    windows = [signal[start : start + window_size] for start in range(0, signal.size - window_size + 1, step_size)]

    return np.asarray(windows)
