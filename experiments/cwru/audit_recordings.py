"""Audit the raw CWRU recordings and write a manifest with one row per vibration signal.

For every signal the manifest records its label, rpm, sampling rate, length and a SHA-256 hash,
so that duplicates can be found and later steps never have to guess.

Run from the project root (after make_labels.py):
    python experiments/cwru/audit_recordings.py
"""

import csv
import hashlib
import logging
import re
import sys

import numpy as np
from machineguard.config import CWRU_LABELS_FILE, CWRU_MANIFEST_FILE, CWRU_RAW_DIR, PROJECT_ROOT
from scipy.io import loadmat

logger = logging.getLogger(__name__)

SIGNAL_PATTERN = re.compile(r"^X(\d+)_(DE|FE|BA)_time$")
RPM_PATTERN = re.compile(r"^X(\d+)RPM$")

# Sampling rates verified for the selected CWRU collection.
NORMAL_BASELINE_FOLDER = "norml"
NORMAL_BASELINE_FILES = {"97.mat", "98.mat", "99.mat", "100.mat"}
NORMAL_BASELINE_RATE_HZ = 48000
FAULT_FOLDERS = {"BF", "IR", "OR"}
FAULT_RATE_HZ = 12000

MANIFEST_COLUMNS = [
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


# --- Entry point: the only place that catches errors --------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_audit()
    except (OSError, ValueError) as error:
        logger.error("Audit stopped: %s", error)
        sys.exit(1)


# --- The audit, step by step ---------------------------------------------------------------------
def run_audit():
    labels = load_recording_labels()
    recording_files = discover_recording_files()
    manifest_rows = build_recording_manifest(recording_files, labels)

    save_recording_manifest(manifest_rows)
    print_audit_summary(recording_files, manifest_rows, labels)


def load_recording_labels():
    """Load CSV metadata indexed by the physical MAT filename."""
    if not CWRU_LABELS_FILE.exists():
        raise FileNotFoundError(f"{CWRU_LABELS_FILE} not found. Run make_labels.py first.")
    with CWRU_LABELS_FILE.open(newline="", encoding="utf-8-sig") as file:
        return {row["file"].strip(): row for row in csv.DictReader(file)}


def discover_recording_files():
    """Find all MAT files beneath the CWRU raw-data directory."""
    return sorted(CWRU_RAW_DIR.rglob("*.mat"))


def build_recording_manifest(recording_files, labels):
    """Build one manifest row per discovered vibration signal."""
    manifest_rows = []
    first_seen = {}  # (channel, hash) -> where this signal was first found

    for recording_path in recording_files:
        recording_labels = labels.get(recording_path.name, {})
        manifest_rows.extend(rows_for_recording(recording_path, recording_labels, first_seen))

    return manifest_rows


def save_recording_manifest(manifest_rows):
    """Write the recording manifest to the interim-data directory."""
    CWRU_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CWRU_MANIFEST_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
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

    print(f"Manifest saved to: {CWRU_MANIFEST_FILE}")


# --- One recording -------------------------------------------------------------------------------
def rows_for_recording(recording_path, recording_labels, first_seen):
    """Manifest rows for every signal in one MAT file. Updates first_seen to detect duplicates."""
    mat_data = loadmat(recording_path)
    rpm_in_file = ";".join(map(str, extract_recorded_rpm(mat_data)))
    sampling_rate_hz = get_sampling_rate_hz(recording_path)

    rows = []
    for signal in discover_signal_variables(mat_data):
        signal_hash = calculate_signal_hash(signal["values"])
        duplicate_of = first_seen.get((signal["channel"], signal_hash))

        if duplicate_of is None:  # the first occurrence is the reference
            first_seen[(signal["channel"], signal_hash)] = (
                f"{recording_path.relative_to(PROJECT_ROOT)}:{signal['variable']}"
            )

        rows.append(
            {
                "file": recording_path.name,
                "folder": recording_path.parent.name,
                **label_fields(recording_labels),
                "rpm_mat": rpm_in_file,
                "internal_id": signal["internal_id"],
                "channel": signal["channel"],
                "samples": signal["values"].shape[0],
                "sampling_rate_hz": sampling_rate_hz,
                "sha256": signal_hash,
                "duplicate_of": duplicate_of or "",
            }
        )
    return rows


def label_fields(recording_labels):
    """The label columns of a manifest row; unknown values stay empty, never invented."""
    return {
        "fault_type": recording_labels.get("fault_type", ""),
        "size_in": recording_labels.get("size_in", ""),
        "or_position": recording_labels.get("or_position", ""),
        "load_hp": recording_labels.get("load_hp", ""),
        "rpm_csv": recording_labels.get("rpm", ""),
    }


# --- Small helpers -------------------------------------------------------------------------------
def get_sampling_rate_hz(recording_path):
    """Return the verified sampling rate of a recording, or None when it is not known."""
    folder = recording_path.parent.name

    if folder == NORMAL_BASELINE_FOLDER and recording_path.name in NORMAL_BASELINE_FILES:
        return NORMAL_BASELINE_RATE_HZ

    if folder in FAULT_FOLDERS:
        return FAULT_RATE_HZ

    return None


def discover_signal_variables(mat_data):
    """Return recognized vibration signals and their internal IDs."""
    signals = []

    for variable_name, values in mat_data.items():
        match = SIGNAL_PATTERN.fullmatch(variable_name)

        if match:
            internal_id, channel = match.groups()
            signals.append(
                {"variable": variable_name, "internal_id": internal_id, "channel": channel, "values": values}
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


if __name__ == "__main__":
    main()
