"""Resample every unique CWRU drive-end recording to one common rate and save it as a .npy file.

Run from the project root (after audit_recordings.py):
    python experiments/cwru/preprocess_recordings.py
"""

import csv
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from machineguard.config import (
    CWRU_MANIFEST_FILE,
    CWRU_PROCESSED_DIR,
    CWRU_PROCESSED_MANIFEST_FILE,
    CWRU_RAW_DIR,
    project_relative,
)
from machineguard.data.preprocessing import resample_signal

logger = logging.getLogger(__name__)

TARGET_RATE_HZ = 12000
CHANNEL = "DE"


# --- Entry point: the only place that catches errors --------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        process_recordings()
    except (OSError, ValueError, KeyError) as error:
        logger.error("Preprocessing stopped: %s", error)
        sys.exit(1)


# --- All recordings ------------------------------------------------------------------------------
def process_recordings():
    """Process unique Drive End recordings and save metadata."""
    processed_rows = []

    for row in load_manifest():
        if row["channel"] != CHANNEL:
            continue

        result = process_recording(row)

        if result is None:
            logger.info("Skipping duplicate: %s / %s", row["file"], row["internal_id"])
            continue

        signal, metadata = result
        processed_rows.append(save_processed_recording(signal, metadata))
        logger.info(
            "Processed: %s (%d -> %d samples)", row["file"], metadata["original_samples"], metadata["processed_samples"]
        )

    if not processed_rows:
        raise ValueError("No recordings were processed.")

    save_processed_manifest(processed_rows)
    logger.info("Processed recordings: %d", len(processed_rows))
    logger.info("Metadata file: %s", CWRU_PROCESSED_MANIFEST_FILE)


def load_manifest(manifest_file=CWRU_MANIFEST_FILE):
    """Load recording metadata from the audit manifest."""
    with manifest_file.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save_processed_manifest(processed_rows):
    CWRU_PROCESSED_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CWRU_PROCESSED_MANIFEST_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(processed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(processed_rows)


# --- One recording -------------------------------------------------------------------------------
def process_recording(row, raw_data_dir=CWRU_RAW_DIR):
    """Load, validate and resample one unique recording. Returns None for a duplicate."""
    if row["channel"] != CHANNEL:
        raise ValueError(f"Expected {CHANNEL} channel, got {row['channel']}.")

    if row["duplicate_of"].strip():
        return None

    original_rate_hz = parse_sampling_rate(row["sampling_rate_hz"])
    source_path = raw_data_dir / row["folder"] / row["file"]
    variable_name = f"X{row['internal_id']}_{CHANNEL}_time"

    signal = load_signal(source_path, variable_name)
    processed = resample_signal(signal, original_rate_hz, TARGET_RATE_HZ)

    return processed, build_metadata(row, source_path, variable_name, original_rate_hz, signal, processed)


def parse_sampling_rate(text):
    """Turn the manifest's sampling-rate text into a positive integer, never guessing a missing value."""
    try:
        rate_hz = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError("A valid sampling rate is required.") from error

    if rate_hz <= 0:
        raise ValueError("Sampling rate must be positive.")
    return rate_hz


def load_signal(source_path, variable_name):
    """Read one vibration signal from a MAT file as a finite, non-empty 1D array."""
    mat_data = loadmat(source_path)

    if variable_name not in mat_data:
        raise KeyError(f"Missing signal {variable_name} in {source_path}")

    signal = np.asarray(mat_data[variable_name], dtype=float).reshape(-1)

    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Empty or non-finite signal: {source_path}:{variable_name}")
    return signal


def build_metadata(row, source_path, variable_name, original_rate_hz, signal, processed):
    """Describe what was done to one recording."""
    return {
        "source_file": str(project_relative(source_path)),
        "source_variable": variable_name,
        "folder": row["folder"],
        "fault_type": row["fault_type"],
        "original_rate_hz": original_rate_hz,
        "target_rate_hz": TARGET_RATE_HZ,
        "original_samples": signal.size,
        "processed_samples": processed.size,
        "status": "processed",
    }


def save_processed_recording(signal, metadata, output_dir=CWRU_PROCESSED_DIR):
    """Save a processed signal and return metadata with its filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stem = Path(metadata["source_file"]).stem
    output_name = f"{metadata['folder']}_{source_stem}_{CHANNEL}.npy"

    np.save(output_dir / output_name, signal, allow_pickle=False)

    return {**metadata, "output_file": output_name}


if __name__ == "__main__":
    main()
