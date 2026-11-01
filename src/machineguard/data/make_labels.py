"""Build labels.csv (fault type, size, position, load, rpm) from the CWRU file names."""

import csv
import logging
import re
import sys
from pathlib import Path

from machineguard.config import CWRU_LABELS_FILE, CWRU_RAW_DIR

logger = logging.getLogger(__name__)

CLASS_OF = {
    "norml": "Normal",
    "normal": "Normal",
    "ir": "IR",
    "bf": "B",
    "or": "OR",
}

# Each starting ID represents four consecutive load conditions.
GROUPS = {
    "Normal": [(97, 0, "-")],
    "IR": [
        (105, 0.007, "-"),
        (169, 0.014, "-"),
        (209, 0.021, "-"),
        (3001, 0.028, "-"),
    ],
    "B": [
        (118, 0.007, "-"),
        (185, 0.014, "-"),
        (222, 0.021, "-"),
        (3005, 0.028, "-"),
    ],
    "OR": [
        (130, 0.007, "@6"),
        (144, 0.007, "@3"),
        (156, 0.007, "@12"),
        (197, 0.014, "@6"),
        (234, 0.021, "@6"),
        (246, 0.021, "@3"),
        (258, 0.021, "@12"),
    ],
}

FILES_PER_CONDITION = 4  # one recording for each load, 0 to 3 HP
RPM_BY_LOAD = [1797, 1772, 1750, 1730]
FIELDNAMES = ["file", "fault_type", "size_in", "or_position", "load_hp", "rpm"]
UNKNOWN = "?"


# --- Level 1: the whole job ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        rows = build_labels()
    except FileNotFoundError as error:
        logger.error("Label generation stopped: %s", error)
        sys.exit(1)
    print(f"{len(rows)} files written to {CWRU_LABELS_FILE}")


def build_labels(data_root=CWRU_RAW_DIR, output_path=CWRU_LABELS_FILE):
    """Build a CWRU labels CSV from the raw recording directory."""
    rows = [label_row(fault_type, path) for fault_type, path in find_recordings(Path(data_root))]
    rows.sort(key=lambda row: (row[1], row[0]))

    write_labels(rows, Path(output_path))
    return rows


# --- Level 2: the steps --------------------------------------------------------------------------
def find_recordings(data_root):
    """List (fault type, path) for every recording file inside a known class folder."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {data_root}")

    recordings = []
    for folder in sorted(data_root.iterdir()):
        fault_type = CLASS_OF.get(folder.name.lower()) if folder.is_dir() else None
        if fault_type is None:
            continue

        recordings.extend((fault_type, path) for path in sorted(folder.rglob("*")) if is_recording(path))
    return recordings


def label_row(fault_type, path):
    """One labels row for a recording; unknown IDs get '?' and a warning, never a guess."""
    condition = lookup_condition(fault_type, file_id_of(path))

    if condition is None:
        logger.warning("Metadata mapping not found for %s", path.name)
        return [path.name, fault_type, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN]

    size, position, load = condition
    return [path.name, fault_type, size, position, load, RPM_BY_LOAD[load]]


def write_labels(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)


# --- Level 3: details ----------------------------------------------------------------------------
def lookup_condition(fault_type, file_id):
    """Return (size in inches, position, load index) for a file ID, or None if the ID is unknown."""
    for first_id, size, position in GROUPS[fault_type]:
        if first_id <= file_id < first_id + FILES_PER_CONDITION:
            return size, position, file_id - first_id
    return None


def is_recording(path):
    return path.is_file() and re.match(r"\d+", path.stem) is not None


def file_id_of(path):
    return int(re.match(r"\d+", path.stem).group())


if __name__ == "__main__":
    main()
