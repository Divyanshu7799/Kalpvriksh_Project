"""Look before you compute: envelope spectra of one recording per class, with the defect frequencies marked.

Question answered: "Does the repetition rate of the bursts match BPFO / BPFI / 2xBSF / FTF?"
If a peak appears at the marked line for the matching class, envelope features are worth building.

Run from the project root:
    python experiments/cwru/plot_envelope_spectra.py
"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from machineguard.config import CWRU_LABELS_FILE, CWRU_PROCESSED_DIR, FIGURES_DIR
from scipy.signal import butter, hilbert, sosfiltfilt

logger = logging.getLogger(__name__)

# --- Configuration -------------------------------------------------------------------------------
FIGURE_FILE = FIGURES_DIR / "envelope_spectra.png"

SAMPLE_RATE_HZ = 12000
SEGMENT_SAMPLES = 16384  # about 1.4 s: long enough to resolve peaks about 1 Hz apart
RESONANCE_BAND_HZ = (2500, 5500)  # provisional: read from the amplitude spectra, then justify in the log
MAX_PLOTTED_HZ = 400

# One example per class: (processed file, label file). Choose by looking at the data, not by habit.
EXAMPLES = {
    "Normal": ("norml_100_DE.npy", "100.mat"),
    "B": ("BF_118_DE.npy", "118.mat"),
    "IR": ("IR_105_DE.npy", "105.mat"),
    "OR": ("OR_130_DE.npy", "130.mat"),
}

# Defect frequency = multiplier x shaft speed. Computed from the bearing geometry
# (9 balls, ball 0.3126 in, pitch 1.537 in, contact angle 0). VERIFY the geometry on the CWRU page.
DEFECT_MULTIPLIERS = {"BPFO": 3.5848, "BPFI": 5.4152, "2xBSF": 4.7134, "FTF": 0.39831}


# --- Entry point ---------------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        plot_envelope_spectra()
    except (FileNotFoundError, ValueError, KeyError) as error:
        logger.error("Plot stopped: %s", error)
        sys.exit(1)


def plot_envelope_spectra():
    labels = pd.read_csv(CWRU_LABELS_FILE).set_index("file")
    figure, axes = plt.subplots(len(EXAMPLES), 1, figsize=(10, 9), sharex=True)

    for axis, (class_name, (npy_name, label_name)) in zip(axes, EXAMPLES.items()):
        signal = load_segment(CWRU_PROCESSED_DIR / npy_name)
        rpm = labels.loc[label_name, "rpm"]
        frequencies_hz, amplitudes = envelope_spectrum(signal, SAMPLE_RATE_HZ, RESONANCE_BAND_HZ)
        draw_panel(axis, class_name, npy_name, frequencies_hz, amplitudes, defect_frequencies_hz(rpm))

    axes[-1].set_xlabel("Envelope frequency (Hz)")
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=150)
    logger.info("Saved %s", FIGURE_FILE)


# --- Signal processing (no plotting, no files: easy to test) -------------------------------------
def load_segment(npy_path):
    signal = np.load(npy_path)
    if signal.size < SEGMENT_SAMPLES:
        raise ValueError(f"{npy_path.name} has only {signal.size} samples, need {SEGMENT_SAMPLES}.")
    return signal[:SEGMENT_SAMPLES]


def defect_frequencies_hz(rpm):
    shaft_hz = rpm / 60
    return {name: multiplier * shaft_hz for name, multiplier in DEFECT_MULTIPLIERS.items()}


def envelope_spectrum(signal, sample_rate_hz, band_hz):
    """Band-pass around the resonance, take the envelope, and return its amplitude spectrum."""
    sos = butter(4, band_hz, btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(filtered))
    envelope = envelope - envelope.mean()
    amplitudes = np.abs(np.fft.rfft(envelope * np.hanning(envelope.size))) / envelope.size
    frequencies_hz = np.fft.rfftfreq(envelope.size, d=1 / sample_rate_hz)
    return frequencies_hz, amplitudes


# --- Drawing -------------------------------------------------------------------------------------
def draw_panel(axis, class_name, npy_name, frequencies_hz, amplitudes, defect_frequencies):
    visible = frequencies_hz <= MAX_PLOTTED_HZ
    axis.plot(frequencies_hz[visible], amplitudes[visible], linewidth=0.8)
    for label, frequency in defect_frequencies.items():
        axis.axvline(frequency, linestyle="--", linewidth=0.8, color="tab:red", alpha=0.7)
        axis.text(frequency, axis.get_ylim()[1], label, fontsize=7, rotation=90, va="top", ha="right")
    axis.set_title(f"{class_name} - {npy_name}", fontsize=9)
    axis.set_ylabel("Amplitude")


if __name__ == "__main__":
    main()
