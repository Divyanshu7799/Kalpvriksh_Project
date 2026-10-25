"""Time-domain features computed for every window of a vibration signal.

Each feature function takes a 2D array (one window per row) and returns one value per window.
To add a feature: write a function like the ones below and register it in FEATURE_FUNCTIONS.
"""

import numpy as np
from scipy.stats import kurtosis


def rms_per_window(windows):
    """Root mean square of each window."""
    return np.sqrt(np.mean(windows**2, axis=1))


def kurtosis_per_window(windows):
    """Excess kurtosis of each window: about 0 for Gaussian noise, large for impulsive signals."""
    return kurtosis(windows, axis=1)


def crest_factor_per_window(windows):
    """Peak absolute amplitude divided by RMS, for each window."""
    return np.max(np.abs(windows), axis=1) / rms_per_window(windows)


FEATURE_FUNCTIONS = {
    "rms": rms_per_window,
    "kurtosis": kurtosis_per_window,
    "crest_factor": crest_factor_per_window,
}
