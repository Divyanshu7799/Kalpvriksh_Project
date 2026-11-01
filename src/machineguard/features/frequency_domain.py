"""Frequency-domain features computed for every window of a vibration signal.

Two families, each tied to a physical hypothesis (see docs/feature_engineering):
- band energy ratio: a defect excites structural resonances, so more energy sits in the resonance band;
- envelope peak ratio: a defect makes repeated bursts at its defect frequency, which shows up as a peak
  in the envelope spectrum of the resonance band, compared with the background around it.
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from machineguard.features.bearing_frequencies import defect_frequencies_hz

# Provisional: read from the amplitude spectra, then justify in the decision register.
RESONANCE_BAND_HZ = (2500, 5500)
BACKGROUND_BAND_HZ = (20, 400)  # envelope-spectrum range used to measure the "normal" background level
PEAK_TOLERANCE_BINS = 1  # a peak may sit this many frequency bins away from the calculated frequency
BANDPASS_ORDER = 4


def band_energy_ratio(windows, sample_rate_hz, band_hz=RESONANCE_BAND_HZ):
    """Share of each window's spectral energy that lies inside band_hz (between 0 and 1)."""
    power = np.abs(np.fft.rfft(windows * np.hanning(windows.shape[1]), axis=1)) ** 2
    frequencies_hz = np.fft.rfftfreq(windows.shape[1], d=1.0 / sample_rate_hz)
    in_band = (frequencies_hz >= band_hz[0]) & (frequencies_hz <= band_hz[1])
    total = power.sum(axis=1)
    return np.divide(power[:, in_band].sum(axis=1), total, out=np.zeros_like(total), where=total > 0)


def envelope_peak_ratio(windows, sample_rate_hz, defect_hz, band_hz=RESONANCE_BAND_HZ):
    """Envelope-spectrum amplitude near defect_hz divided by the median amplitude of the background."""
    frequencies_hz, spectra = envelope_spectra(windows, sample_rate_hz, band_hz)
    bin_width_hz = frequencies_hz[1]
    near_defect = np.abs(frequencies_hz - defect_hz) <= PEAK_TOLERANCE_BINS * bin_width_hz
    background = (frequencies_hz >= BACKGROUND_BAND_HZ[0]) & (frequencies_hz <= BACKGROUND_BAND_HZ[1])

    peak = spectra[:, near_defect].max(axis=1)
    background_level = np.median(spectra[:, background], axis=1)
    return np.divide(peak, background_level, out=np.full(peak.shape, np.nan), where=background_level > 0)


def envelope_spectra(windows, sample_rate_hz, band_hz):
    """Envelope spectrum of every window. Returns (frequencies_hz, one spectrum per row)."""
    sos = butter(BANDPASS_ORDER, band_hz, btype="bandpass", fs=sample_rate_hz, output="sos")
    envelopes = np.abs(hilbert(sosfiltfilt(sos, windows, axis=1), axis=1))
    centred = envelopes - envelopes.mean(axis=1, keepdims=True)
    spectra = np.abs(np.fft.rfft(centred * np.hanning(windows.shape[1]), axis=1))
    return np.fft.rfftfreq(windows.shape[1], d=1.0 / sample_rate_hz), spectra


def defect_ratio_feature(defect_name):
    """Feature function for the envelope peak ratio at one named defect frequency (BPFO, BPFI, ...)."""

    def feature(windows, sample_rate_hz, rpm):
        return envelope_peak_ratio(windows, sample_rate_hz, defect_frequencies_hz(rpm)[defect_name])

    return feature
