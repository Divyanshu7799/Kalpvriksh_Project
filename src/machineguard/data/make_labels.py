import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
LABELS_FILE = PROJECT_ROOT / "data" / "interim" / "cwru" / "labels.csv"

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

RPM_BY_LOAD = [1797, 1772, 1750, 1730]
FIELDNAMES = ["file", "fault_type", "size_in", "or_position", "load_hp", "rpm"]


def build_labels(data_root=RAW_DATA_DIR, output_path=LABELS_FILE):
    """Build a CWRU labels CSV from the raw recording directory."""
    data_root = Path(data_root)
    output_path = Path(output_path)

    if not data_root.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {data_root}")

    rows = []

    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir():
            continue

        fault_type = CLASS_OF.get(folder.name.lower())
        if fault_type is None:
            continue

        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue

            match = re.match(r"(\d+)", file_path.stem)
            if match is None:
                continue

            file_id = int(match.group(1))
            size, position, load = "?", "?", "?"

            for first_id, fault_size, fault_position in GROUPS[fault_type]:
                if first_id <= file_id < first_id + 4:
                    size = fault_size
                    position = fault_position
                    load = file_id - first_id
                    break

            if load == "?":
                print(f"Warning: metadata mapping not found for {file_path.name}")

            rpm = RPM_BY_LOAD[load] if load != "?" else "?"

            rows.append([file_path.name, fault_type, size, position, load, rpm])

    rows.sort(key=lambda row: (row[1], row[0]))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)

    return rows


def main():
    """Generate labels and report the output location."""
    rows = build_labels()
    print(f"{len(rows)} files written to {LABELS_FILE}")


if __name__ == "__main__":
    main()
