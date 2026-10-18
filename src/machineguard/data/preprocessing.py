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
