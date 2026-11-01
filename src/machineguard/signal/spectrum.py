"""Frequency-domain tools for vibration segments. Pure maths: no files, no plotting."""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

BANDPASS_ORDER = 4


def amplitude_spectrum(segment, sample_rate_hz):
    """Hann-windowed amplitude spectrum. Returns (frequencies_hz, amplitudes)."""
    window = np.hanning(segment.size)
    amplitudes = np.abs(np.fft.rfft(segment * window)) * 2.0 / window.sum()
    frequencies_hz = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate_hz)
    return frequencies_hz, amplitudes


def envelope_spectrum(signal, sample_rate_hz, band_hz):
    """Spectrum of the signal's envelope inside a frequency band: it shows how often bursts repeat."""
    envelope = envelope_in_band(signal, sample_rate_hz, band_hz)
    return amplitude_spectrum(envelope - envelope.mean(), sample_rate_hz)


def envelope_in_band(signal, sample_rate_hz, band_hz):
    """Band-pass the signal, then return the outline (envelope) of what remains."""
    sos = butter(BANDPASS_ORDER, band_hz, btype="bandpass", fs=sample_rate_hz, output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, signal)))
