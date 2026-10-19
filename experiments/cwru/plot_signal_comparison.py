
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cwru"
MANIFEST_PATH = PROCESSED_DIR / "processed_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "cwru_signal_comparison.png"

SAMPLES_FOR_SPECTRUM = 6000  # 0.5 seconds at 12 kHz
SAMPLES_FOR_WAVEFORM = 1200  # 0.1 seconds at 12 kHz
MAX_FREQUENCY_HZ = 3000

CLASS_ORDER = ["Normal", "B", "IR", "OR"]


def load_recordings():
    """Select one processed recording for each available fault class."""
    with MANIFEST_PATH.open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    recordings = {}

    for row in sorted(rows, key=lambda item: item["source_file"]):
        fault_type = row["fault_type"]

        if row["status"] != "processed":
            continue

        if fault_type not in CLASS_ORDER:
            continue

        if fault_type in recordings:
            continue

        signal_path = PROCESSED_DIR / row["output_file"]
        signal = np.asarray(np.load(signal_path), dtype=float).ravel()
        sampling_rate = int(row["target_rate_hz"])

        if signal.size < SAMPLES_FOR_SPECTRUM:
            print(f"Skipping short recording: {signal_path.name}")
            continue

        if not np.isfinite(signal).all():
            raise ValueError(f"Non-finite values in {signal_path.name}")

        recordings[fault_type] = {
            "signal": signal,
            "sampling_rate": sampling_rate,
            "filename": signal_path.name,
        }

    missing = set(CLASS_ORDER) - set(recordings)
    if missing:
        raise ValueError(f"No usable recordings found for: {sorted(missing)}")

    return recordings


def plot_comparison(recordings):
    """Plot time waveforms and amplitude spectra for each class."""
    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(14, 14),
        constrained_layout=True,
    )

    for row_index, fault_type in enumerate(CLASS_ORDER):
        recording = recordings[fault_type]
        signal = recording["signal"]
        sampling_rate = recording["sampling_rate"]
        filename = recording["filename"]

        # Use the same duration for each recording.
        segment = signal[:SAMPLES_FOR_SPECTRUM]

        # Time-domain waveform: first 0.1 seconds.
        time = (
            np.arange(SAMPLES_FOR_WAVEFORM) / sampling_rate
        )
        axes[row_index, 0].plot(
            time,
            segment[:SAMPLES_FOR_WAVEFORM],
            linewidth=0.7,
        )
        axes[row_index, 0].set_title(
            f"{fault_type} — {filename}"
        )
        axes[row_index, 0].set_xlabel("Time (seconds)")
        axes[row_index, 0].set_ylabel("Amplitude")
        axes[row_index, 0].grid(True, alpha=0.3)

        # Frequency-domain amplitude spectrum.
        # A Hann window reduces spectral leakage.
        window = np.hanning(segment.size)
        spectrum = np.abs(
            np.fft.rfft(segment * window)
        )
        spectrum *= 2.0 / window.sum()

        frequencies = np.fft.rfftfreq(
            segment.size,
            d=1.0 / sampling_rate,
        )

        axes[row_index, 1].plot(
            frequencies,
            spectrum,
            linewidth=0.7,
        )
        axes[row_index, 1].set_xlim(0, MAX_FREQUENCY_HZ)
        axes[row_index, 1].set_title(
            f"{fault_type} — amplitude spectrum"
        )
        axes[row_index, 1].set_xlabel("Frequency (Hz)")
        axes[row_index, 1].set_ylabel("Amplitude")
        axes[row_index, 1].grid(True, alpha=0.3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.show()

    print(f"Plot saved to: {OUTPUT_PATH}")


def main():
    recordings = load_recordings()
    plot_comparison(recordings)


if __name__ == "__main__":
    main()
