import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
LABELS_FILE = PROJECT_ROOT / "data" / "interim" / "cwru" / "labels.csv"
MANIFEST_FILE = PROJECT_ROOT / "data" / "interim" / "cwru" / "recording_manifest.csv"

SIGNAL_PATTERN = re.compile(r"^X(\d+)_(DE|FE|BA)_time$")
RPM_PATTERN = re.compile(r"^X(\d+)RPM$")

CHANNELS = ("DE", "FE", "BA")
NORMAL_BASELINE_FILES = {"97.mat", "98.mat", "99.mat", "100.mat"}
FAULT_FOLDERS = {"BF", "IR", "OR"}


def get_sampling_rate_hz(recording_path):
    """Return the sampling rate verified for the selected CWRU collection."""
    folder = recording_path.parent.name
    filename = recording_path.name

    if folder == "norml" and filename in NORMAL_BASELINE_FILES:
        return 48000

    if folder in FAULT_FOLDERS:
        return 12000

    return ""


def load_recording_labels():
    """Load CSV metadata indexed by the physical MAT filename."""
    with LABELS_FILE.open(newline="", encoding="utf-8-sig") as file:
        return {row["file"].strip(): row for row in csv.DictReader(file)}


def discover_recording_files():
    """Find all MAT files beneath the CWRU raw-data directory."""
    return sorted(RAW_DATA_DIR.rglob("*.mat"))


def discover_signal_variables(mat_data):
    """Return recognized vibration signals and their internal IDs."""
    signals = []

    for variable_name, values in mat_data.items():
        match = SIGNAL_PATTERN.fullmatch(variable_name)

        if match:
            internal_id, channel = match.groups()
            signals.append(
                {
                    "variable": variable_name,
                    "internal_id": internal_id,
                    "channel": channel,
                    "values": values,
                }
            )

    return signals


def extract_recorded_rpm(mat_data):
    """Extract available RPM values without inventing missing values."""
    rpm_values = []

    for variable_name, values in mat_data.items():
        if RPM_PATTERN.fullmatch(variable_name):
            rpm_values.append(int(values.item()))

    return sorted(set(rpm_values))


def calculate_signal_hash(signal):
    """Calculate a stable hash of a signal's array contents."""
    values = np.ascontiguousarray(signal)
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_recording_manifest(recording_files, labels):
    """Build one manifest row per discovered vibration signal."""
    manifest_rows = []
    previously_seen_signals = {}

    for recording_path in recording_files:
        mat_data = loadmat(recording_path)
        recording_labels = labels.get(recording_path.name, {})

        signals = discover_signal_variables(mat_data)
        recorded_rpm = extract_recorded_rpm(mat_data)

        sampling_rate_hz = get_sampling_rate_hz(recording_path)

        for signal in signals:
            signal_hash = calculate_signal_hash(signal["values"])
            duplicate_key = (signal["channel"], signal_hash)

            # Preserve the first occurrence as the reference.
            duplicate_of = previously_seen_signals.get(duplicate_key)

            if duplicate_of is None:
                previously_seen_signals[duplicate_key] = (
                    f"{recording_path.relative_to(PROJECT_ROOT)}:" f"{signal['variable']}"
                )

            manifest_rows.append(
                {
                    "file": recording_path.name,
                    "folder": recording_path.parent.name,
                    "fault_type": recording_labels.get("fault_type", ""),
                    "size_in": recording_labels.get("size_in", ""),
                    "or_position": recording_labels.get("or_position", ""),
                    "load_hp": recording_labels.get("load_hp", ""),
                    "rpm_csv": recording_labels.get("rpm", ""),
                    "rpm_mat": ";".join(map(str, recorded_rpm)),
                    "internal_id": signal["internal_id"],
                    "channel": signal["channel"],
                    "samples": signal["values"].shape[0],
                    "sampling_rate_hz": sampling_rate_hz,
                    "sha256": signal_hash,
                    "duplicate_of": duplicate_of or "",
                }
            )

    return manifest_rows


def save_recording_manifest(manifest_rows):
    """Write the recording manifest to the interim-data directory."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "folder",
        "fault_type",
        "size_in",
        "or_position",
        "load_hp",
        "rpm_csv",
        "rpm_mat",
        "internal_id",
        "channel",
        "samples",
        "sampling_rate_hz",
        "sha256",
        "duplicate_of",
    ]

    with MANIFEST_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def print_audit_summary(recording_files, manifest_rows, labels):
    """Print counts and identify files absent from the label CSV."""
    duplicate_count = sum(bool(row["duplicate_of"]) for row in manifest_rows)

    unlabeled_files = sorted({path.name for path in recording_files if path.name not in labels})

    print(f"MAT files found: {len(recording_files)}")
    print(f"Signal variables found: {len(manifest_rows)}")
    print(f"Duplicate signal entries: {duplicate_count}")
    print(f"Files missing from labels.csv: {len(unlabeled_files)}")

    if unlabeled_files:
        print("Unlabeled files:", ", ".join(unlabeled_files))

    print(f"Manifest saved to: {MANIFEST_FILE}")


def main():
    """Run the complete CWRU recording audit."""
    labels = load_recording_labels()
    recording_files = discover_recording_files()
    manifest_rows = build_recording_manifest(recording_files, labels)

    save_recording_manifest(manifest_rows)
    print_audit_summary(recording_files, manifest_rows, labels)


if __name__ == "__main__":
    main()
