import csv
import re
from pathlib import Path

ROOT = Path(r"D:\CWRU")

# folder name -> class label
class_of = {"norml": "Normal", "normal": "Normal", "ir": "IR", "bf": "B", "or": "OR"}

# (first_id, size_inch, or_position): 4 consecutive IDs = load 0,1,2,3 HP
groups = {
    "Normal": [(97, 0, "-")],
    "IR": [(105, 0.007, "-"), (169, 0.014, "-"), (209, 0.021, "-"), (3001, 0.028, "-")],
    "B": [(118, 0.007, "-"), (185, 0.014, "-"), (222, 0.021, "-"), (3005, 0.028, "-")],
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
rpm = [1797, 1772, 1750, 1730]

rows = []
for folder in ROOT.iterdir():
    if not folder.is_dir():
        continue
    cls = class_of.get(folder.name.lower())
    if cls is None:
        continue
    for f in folder.rglob("*"):
        m = re.match(r"(\d+)", f.stem)
        if not (f.is_file() and m):
            continue
        fid = int(m.group(1))
        size, pos, load = "?", "?", "?"
        for start, s, p in groups[cls]:
            if start <= fid < start + 4:
                size, pos, load = s, p, fid - start
                break
        else:
            print("Check on website:", cls, f.name)
        r = rpm[load] if load != "?" else "?"
        rows.append([f.name, cls, size, pos, load, r])

with open(ROOT / "labels.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "fault_type", "size_in", "or_position", "load_hp", "rpm"])
    w.writerows(sorted(rows))

print(len(rows), "files written to labels.csv")
