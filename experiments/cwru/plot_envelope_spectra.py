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

from machineguard.config import CWRU_LABELS_FILE, CWRU_PROCESSED_DIR, FIGURES_DIR, PROCESSED_SAMPLE_RATE_HZ
from machineguard.features.bearing_frequencies import defect_frequencies_hz
from machineguard.signal.spectrum import envelope_spectrum

logger = logging.getLogger(__name__)

FIGURE_FILE = FIGURES_DIR / "envelope_spectra.png"

SAMPLE_RATE_HZ = PROCESSED_SAMPLE_RATE_HZ
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


# --- Level 1: the whole job ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        make_envelope_figure()
    except (OSError, ValueError, KeyError) as error:
        logger.error("Plot stopped: %s", error)
        sys.exit(1)


def make_envelope_figure():
    rpm_by_file = pd.read_csv(CWRU_LABELS_FILE).set_index("file")["rpm"]
    figure = draw_envelope_spectra(rpm_by_file)
    save_figure(figure, FIGURE_FILE)


# --- Level 2: the steps --------------------------------------------------------------------------
def draw_envelope_spectra(rpm_by_file):
    """One panel per example: its envelope spectrum with the defect frequencies marked."""
    figure, axes = plt.subplots(len(EXAMPLES), 1, figsize=(10, 9), sharex=True)

    for axis, (class_name, (npy_name, label_name)) in zip(axes, EXAMPLES.items()):
        signal = load_segment(CWRU_PROCESSED_DIR / npy_name)
        frequencies_hz, amplitudes = envelope_spectrum(signal, SAMPLE_RATE_HZ, RESONANCE_BAND_HZ)
        draw_panel(axis, f"{class_name} - {npy_name}", frequencies_hz, amplitudes)
        mark_defect_frequencies(axis, defect_frequencies_hz(rpm_by_file[label_name]))

    axes[-1].set_xlabel("Envelope frequency (Hz)")
    figure.tight_layout()
    return figure


def save_figure(figure, figure_file):
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    logger.info("Saved %s", figure_file)


# --- Level 3: details ----------------------------------------------------------------------------
def load_segment(npy_path):
    signal = np.load(npy_path)
    if signal.size < SEGMENT_SAMPLES:
        raise ValueError(f"{npy_path.name} has only {signal.size} samples, need {SEGMENT_SAMPLES}.")
    return signal[:SEGMENT_SAMPLES]


def draw_panel(axis, title, frequencies_hz, amplitudes):
    visible = frequencies_hz <= MAX_PLOTTED_HZ
    axis.plot(frequencies_hz[visible], amplitudes[visible], linewidth=0.8)
    axis.set_title(title, fontsize=9)
    axis.set_ylabel("Amplitude")


def mark_defect_frequencies(axis, defect_frequencies):
    for label, frequency in defect_frequencies.items():
        axis.axvline(frequency, linestyle="--", linewidth=0.8, color="tab:red", alpha=0.7)
        axis.text(frequency, axis.get_ylim()[1], label, fontsize=7, rotation=90, va="top", ha="right")


if __name__ == "__main__":
    main()
