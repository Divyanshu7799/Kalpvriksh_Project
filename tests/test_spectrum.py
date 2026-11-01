import numpy as np
import pytest

from machineguard.signal.spectrum import amplitude_spectrum, envelope_spectrum

SAMPLE_RATE_HZ = 12000


def test_sine_wave_peaks_at_its_own_frequency_and_amplitude():
    t = np.arange(SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
    signal = 2.0 * np.sin(2 * np.pi * 100 * t)

    frequencies_hz, amplitudes = amplitude_spectrum(signal, SAMPLE_RATE_HZ)

    peak = amplitudes.argmax()
    assert frequencies_hz[peak] == 100.0
    assert amplitudes[peak] == pytest.approx(2.0, rel=1e-2)


def test_envelope_spectrum_finds_the_rate_at_which_a_high_frequency_signal_pulses():
    t = np.arange(SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
    carrier = np.cos(2 * np.pi * 3500 * t)
    pulsing_signal = (1 + 0.8 * np.cos(2 * np.pi * 107 * t)) * carrier

    frequencies_hz, amplitudes = envelope_spectrum(pulsing_signal, SAMPLE_RATE_HZ, band_hz=(2500, 5500))

    low_rates = (frequencies_hz > 20) & (frequencies_hz < 400)
    assert frequencies_hz[low_rates][amplitudes[low_rates].argmax()] == pytest.approx(107, abs=1)
