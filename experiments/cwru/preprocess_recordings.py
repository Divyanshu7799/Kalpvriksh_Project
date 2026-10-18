import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from machineguard.data.preprocessing import resample_signal

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
MANIFEST_FILE = PROJECT_ROOT / "data" / "interim" / "cwru" / "recording_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "cwru"
METADATA_FILE = OUTPUT_DIR / "processed_manifest.csv"

TARGET_RATE_HZ = 12000
CHANNEL = "DE"


def process_recordings():
    """Process unique Drive End recordings and save metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_rows = []

    for row in load_manifest():
        if row["channel"] != CHANNEL:
            continue

        result = process_recording(row)

        if result is None:
            print(f"Skipping duplicate: {row['file']} / {row['internal_id']}")
            continue

        signal, metadata = result
        processed_row = save_processed_recording(signal, metadata)
        processed_rows.append(processed_row)

        print(
            f"Processed: {row['file']} "
            f"({metadata['original_samples']} -> "
            f"{metadata['processed_samples']} samples)"
        )

    if not processed_rows:
        raise ValueError("No recordings were processed.")

    fieldnames = list(processed_rows[0].keys())

    with METADATA_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"\nProcessed recordings: {len(processed_rows)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata file: {METADATA_FILE}")


def load_manifest(manifest_file=MANIFEST_FILE):
    """Load recording metadata from the audit manifest."""
    with manifest_file.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def process_recording(row, raw_data_dir=RAW_DATA_DIR):
    """Load, validate and resample one unique recording."""
    if row["channel"] != CHANNEL:
        raise ValueError(f"Expected {CHANNEL} channel, got {row['channel']}.")

    if row["duplicate_of"].strip():
        return None

    try:
        original_rate = int(row["sampling_rate_hz"])
    except (TypeError, ValueError) as error:
        raise ValueError("A valid sampling rate is required.") from error

    if original_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    source_path = raw_data_dir / row["folder"] / row["file"]
    variable_name = f"X{row['internal_id']}_{CHANNEL}_time"
    mat_data = loadmat(source_path)

    if variable_name not in mat_data:
        raise KeyError(f"Missing signal {variable_name} in {source_path}")

    signal = np.asarray(mat_data[variable_name], dtype=float).reshape(-1)

    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Empty or non-finite signal: {source_path}:{variable_name}")

    processed = resample_signal(signal, original_rate, TARGET_RATE_HZ)

    source_description = (
        str(source_path.relative_to(PROJECT_ROOT)) if source_path.is_relative_to(PROJECT_ROOT) else str(source_path)
    )

    metadata = {
        "source_file": source_description,
        "source_variable": variable_name,
        "folder": row["folder"],
        "fault_type": row["fault_type"],
        "original_rate_hz": original_rate,
        "target_rate_hz": TARGET_RATE_HZ,
        "original_samples": signal.size,
        "processed_samples": processed.size,
        "status": "processed",
    }

    return processed, metadata


def save_processed_recording(signal, metadata, output_dir=OUTPUT_DIR):
    """Save a processed signal and return metadata with its filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stem = Path(metadata["source_file"]).stem
    output_name = f"{metadata['folder']}_{source_stem}_{CHANNEL}.npy"
    output_path = output_dir / output_name

    np.save(output_path, signal, allow_pickle=False)

    return {**metadata, "output_file": output_name}


if __name__ == "__main__":
    process_recordings()
