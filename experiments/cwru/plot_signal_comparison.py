"""Look at the data first: a waveform and an amplitude spectrum for one recording of each class.

Run from the project root (after preprocess_recordings.py):
    python experiments/cwru/plot_signal_comparison.py
"""

import logging
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from machineguard.config import CLASS_ORDER, DOCS_DIR
from machineguard.data.recording_table import load_processed_signal, read_processed_manifest
from machineguard.signal.spectrum import amplitude_spectrum

logger = logging.getLogger(__name__)

OUTPUT_PATH = DOCS_DIR / "cwru_signal_comparison.png"

SAMPLES_FOR_SPECTRUM = 6000  # 0.5 seconds at 12 kHz
SAMPLES_FOR_WAVEFORM = 1200  # 0.1 seconds at 12 kHz
MAX_FREQUENCY_HZ = 3000

Example = namedtuple("Example", ["filename", "signal", "sample_rate_hz"])


# --- Level 1: the whole job ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        make_comparison_figure()
    except (OSError, ValueError, KeyError) as error:
        logger.error("Plot stopped: %s", error)
        sys.exit(1)


def make_comparison_figure():
    examples = choose_examples(read_processed_manifest())
    figure = draw_comparison(examples)
    save_figure(figure, OUTPUT_PATH)


# --- Level 2: the steps --------------------------------------------------------------------------
def choose_examples(manifest_rows):
    """The first usable processed recording of each class, as {class: Example}."""
    examples = {}

    for row in sorted(manifest_rows, key=lambda item: item["source_file"]):
        fault_type = row["fault_type"]
        if fault_type not in CLASS_ORDER or fault_type in examples:
            continue

        signal = load_processed_signal(row["output_file"])
        if signal.size < SAMPLES_FOR_SPECTRUM:
            logger.info("Skipping short recording: %s", row["output_file"])
            continue

        examples[fault_type] = Example(row["output_file"], signal, int(row["target_rate_hz"]))

    missing = set(CLASS_ORDER) - set(examples)
    if missing:
        raise ValueError(f"No usable recordings found for: {sorted(missing)}")
    return examples


def draw_comparison(examples):
    """One row per class: waveform on the left, amplitude spectrum on the right."""
    figure, axes = plt.subplots(nrows=len(CLASS_ORDER), ncols=2, figsize=(14, 14), constrained_layout=True)

    for (waveform_axis, spectrum_axis), fault_type in zip(axes, CLASS_ORDER):
        example = examples[fault_type]
        segment = example.signal[:SAMPLES_FOR_SPECTRUM]  # the same duration for every class

        draw_waveform(waveform_axis, fault_type, example, segment)
        draw_spectrum(spectrum_axis, fault_type, example, segment)
    return figure


def save_figure(figure, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    logger.info("Plot saved to: %s", output_path)


# --- Level 3: one panel --------------------------------------------------------------------------
def draw_waveform(axis, fault_type, example, segment):
    """The first 0.1 seconds of the signal."""
    time = np.arange(SAMPLES_FOR_WAVEFORM) / example.sample_rate_hz
    axis.plot(time, segment[:SAMPLES_FOR_WAVEFORM], linewidth=0.7)
    axis.set_title(f"{fault_type} — {example.filename}")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, alpha=0.3)


def draw_spectrum(axis, fault_type, example, segment):
    frequencies_hz, amplitudes = amplitude_spectrum(segment, example.sample_rate_hz)
    axis.plot(frequencies_hz, amplitudes, linewidth=0.7)
    axis.set_xlim(0, MAX_FREQUENCY_HZ)
    axis.set_title(f"{fault_type} — amplitude spectrum")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, alpha=0.3)


if __name__ == "__main__":
    main()
