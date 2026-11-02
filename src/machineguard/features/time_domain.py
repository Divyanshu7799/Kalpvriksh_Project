"""Time-domain features computed for every window of a vibration signal.

Each feature function takes a 2D array (one window per row) and returns one value per window.
To add a feature: write a function like the ones below and register it in FEATURE_FUNCTIONS.
"""

import numpy as np
from scipy.stats import kurtosis, skew


def rms_per_window(windows):
    """Root mean square of each window."""
    return np.sqrt(np.mean(windows**2, axis=1))


def kurtosis_per_window(windows):
    """Excess kurtosis of each window: about 0 for Gaussian noise, large for impulsive signals."""
    return kurtosis(windows, axis=1)


def std_per_window(windows):
    """Standard deviation of each window."""
    return np.std(windows, axis=1)


def crest_factor_per_window(windows):
    """Peak absolute amplitude divided by RMS, for each window. NaN for a window of all zeros."""
    return _safe_divide(peak_per_window(windows), rms_per_window(windows))


def peak_per_window(windows):
    """Largest absolute amplitude in each window."""
    return np.max(np.abs(windows), axis=1)


def peak_to_peak_per_window(windows):
    """Largest minus smallest amplitude in each window."""
    return np.max(windows, axis=1) - np.min(windows, axis=1)


def mean_abs_per_window(windows):
    """Mean of the absolute amplitude of each window."""
    return np.mean(np.abs(windows), axis=1)


def skewness_per_window(windows):
    """Asymmetry of the amplitude distribution of each window (0 for a symmetric signal)."""
    return skew(windows, axis=1)


def shape_factor_per_window(windows):
    """RMS divided by mean absolute amplitude. NaN for a window of all zeros."""
    return _safe_divide(rms_per_window(windows), mean_abs_per_window(windows))


def impulse_factor_per_window(windows):
    """Peak divided by mean absolute amplitude. NaN for a window of all zeros."""
    return _safe_divide(peak_per_window(windows), mean_abs_per_window(windows))


def clearance_factor_per_window(windows):
    """Peak divided by the squared mean of the square-rooted absolute amplitude. NaN for all zeros."""
    return _safe_divide(peak_per_window(windows), np.mean(np.sqrt(np.abs(windows)), axis=1) ** 2)


def _safe_divide(numerator, denominator):
    """Element-wise division that gives NaN, not a warning or infinity, where the denominator is zero."""
    return np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


FEATURE_FUNCTIONS = {
    "rms": rms_per_window,
    "std": std_per_window,
    "kurtosis": kurtosis_per_window,
    "crest_factor": crest_factor_per_window,
    "peak": peak_per_window,
    "peak_to_peak": peak_to_peak_per_window,
    "mean_abs": mean_abs_per_window,
    "skewness": skewness_per_window,
    "shape_factor": shape_factor_per_window,
    "impulse_factor": impulse_factor_per_window,
    "clearance_factor": clearance_factor_per_window,
}
