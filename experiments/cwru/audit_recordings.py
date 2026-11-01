"""Audit the raw CWRU recordings and write a manifest with one row per vibration signal.

For every signal the manifest records its label, rpm, sampling rate, length and a SHA-256 hash,
so that duplicates can be found and later steps never have to guess.

Run from the project root (after make_labels.py):
    python experiments/cwru/audit_recordings.py
"""

import csv
import logging
import sys

from scipy.io import loadmat

from machineguard.config import CWRU_LABELS_FILE, CWRU_MANIFEST_FILE, CWRU_RAW_DIR, project_relative
from machineguard.data.cwru_mat import (
    calculate_signal_hash,
    discover_signal_variables,
    extract_recorded_rpm,
    get_sampling_rate_hz,
)

logger = logging.getLogger(__name__)

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


# --- Level 1: the whole audit --------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_audit()
    except (OSError, ValueError) as error:
        logger.error("Audit stopped: %s", error)
        sys.exit(1)


def run_audit():
    labels = load_recording_labels()
    recording_files = discover_recording_files()
    manifest_rows = build_recording_manifest(recording_files, labels)

    save_recording_manifest(manifest_rows)
    print_audit_summary(recording_files, manifest_rows, labels)


# --- Level 2: the steps of the audit -------------------------------------------------------------
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


# --- Level 3: one recording, one signal ----------------------------------------------------------
def rows_for_recording(recording_path, recording_labels, first_seen):
    """Manifest rows for every signal in one MAT file."""
    mat_data = loadmat(recording_path)
    facts = facts_about_recording(recording_path, recording_labels, mat_data)

    return [manifest_row(facts, recording_path, signal, first_seen) for signal in discover_signal_variables(mat_data)]


def facts_about_recording(recording_path, recording_labels, mat_data):
    """The manifest columns that are the same for every signal of one recording."""
    return {
        "file": recording_path.name,
        "folder": recording_path.parent.name,
        **label_fields(recording_labels),
        "rpm_mat": ";".join(map(str, extract_recorded_rpm(mat_data))),
        "sampling_rate_hz": get_sampling_rate_hz(recording_path),
    }


def manifest_row(facts, recording_path, signal, first_seen):
    """One manifest row: the recording's facts plus what is specific to this signal."""
    signal_hash = calculate_signal_hash(signal["values"])

    return {
        **facts,
        "internal_id": signal["internal_id"],
        "channel": signal["channel"],
        "samples": signal["values"].shape[0],
        "sha256": signal_hash,
        "duplicate_of": note_first_occurrence(first_seen, recording_path, signal, signal_hash),
    }


# --- Level 4: details ----------------------------------------------------------------------------
def label_fields(recording_labels):
    """The label columns of a manifest row; unknown values stay empty, never invented."""
    return {
        "fault_type": recording_labels.get("fault_type", ""),
        "size_in": recording_labels.get("size_in", ""),
        "or_position": recording_labels.get("or_position", ""),
        "load_hp": recording_labels.get("load_hp", ""),
        "rpm_csv": recording_labels.get("rpm", ""),
    }


def note_first_occurrence(first_seen, recording_path, signal, signal_hash):
    """Return where this signal was first seen ('' if this is its first appearance); remember new ones."""
    key = (signal["channel"], signal_hash)

    if key in first_seen:
        return first_seen[key]

    first_seen[key] = f"{project_relative(recording_path)}:{signal['variable']}"
    return ""


if __name__ == "__main__":
    main()
