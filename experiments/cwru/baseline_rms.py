"""Rungs 1-2 of the feature ladder: a dummy baseline and an RMS-only classifier.

Question answered: "How much does RMS alone tell us about the 4 CWRU classes
(Normal, B, IR, OR) on a load the model has never seen?"

Run from the project root:
    python experiments/cwru/baseline_rms.py
"""

from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from machineguard.data.preprocessing import split_into_windows

# --- Configuration -------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cwru"
PROCESSED_MANIFEST = PROCESSED_DIR / "processed_manifest.csv"
LABELS_FILE = PROJECT_ROOT / "data" / "interim" / "cwru" / "labels.csv"

# Provisional: to be chosen by the window-size experiment, not by habit.
WINDOW_SIZE = 2048
OVERLAP = 0.5

CLASS_ORDER = ["Normal", "B", "IR", "OR"]
LOADS_HP = [0, 1, 2, 3]


def rms_per_window(windows):
    """Root mean square of each window (one value per row)."""
    return np.sqrt(np.mean(windows**2, axis=1))


# To add a feature (rung 3 and beyond): write a function like rms_per_window,
# register it here, and add its name to FEATURE_NAMES. Nothing else changes.
FEATURE_FUNCTIONS = {"rms": rms_per_window}
FEATURE_NAMES = ["rms"]


# --- Step 1: know which recording is what --------------------------------------------------------
def file_name_of(source_file):
    """'data\\raw\\cwru\\BF\\118.mat' -> '118.mat' (accepts Windows and POSIX separators)."""
    return PureWindowsPath(source_file).name


def load_recording_table():
    """Join the processed manifest with the labels (fault type, size, load) by file name."""
    manifest = pd.read_csv(PROCESSED_MANIFEST)
    labels = pd.read_csv(LABELS_FILE)

    if not labels["file"].is_unique:
        raise ValueError("labels.csv contains duplicate file names.")

    manifest["file"] = manifest["source_file"].map(file_name_of)
    table = manifest[["file", "output_file"]].merge(
        labels[["file", "fault_type", "size_in", "load_hp"]], on="file", how="left"
    )
    table["load_hp"] = pd.to_numeric(table["load_hp"], errors="coerce")

    unlabelled = table[table["load_hp"].isna() | table["fault_type"].isna()]
    if not unlabelled.empty:
        print("WARNING: dropping recordings without a known load or fault type:")
        print(unlabelled["file"].tolist())
    return table.drop(unlabelled.index).reset_index(drop=True)


# --- Step 2: cut each recording into windows and compute features --------------------------------
def build_window_table(recordings):
    """One row per window: its features plus which recording, class and load it came from."""
    parts = []
    for recording in recordings.itertuples():
        signal = np.load(PROCESSED_DIR / recording.output_file)
        windows = split_into_windows(signal, WINDOW_SIZE, OVERLAP)
        part = pd.DataFrame({name: FEATURE_FUNCTIONS[name](windows) for name in FEATURE_NAMES})
        part["file"] = recording.file
        part["fault_type"] = recording.fault_type
        part["load_hp"] = int(recording.load_hp)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# --- Step 3: train on three loads, test on the fourth --------------------------------------------
def make_models():
    return {
        "dummy (most frequent class)": DummyClassifier(strategy="most_frequent"),
        "logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
    }


def evaluate_held_out_load(window_table, held_out_load):
    """Return {model name: (macro-F1, confusion matrix)} for one held-out load."""
    train = window_table[window_table["load_hp"] != held_out_load]
    test = window_table[window_table["load_hp"] == held_out_load]

    results = {}
    for name, model in make_models().items():
        model.fit(train[FEATURE_NAMES], train["fault_type"])
        predicted = model.predict(test[FEATURE_NAMES])
        macro_f1 = f1_score(test["fault_type"], predicted, labels=CLASS_ORDER, average="macro", zero_division=0)
        matrix = confusion_matrix(test["fault_type"], predicted, labels=CLASS_ORDER)
        results[name] = (macro_f1, matrix)
    return results


# --- Step 4: report ------------------------------------------------------------------------------
def report(all_results):
    model_names = list(next(iter(all_results.values())).keys())

    print(f"\nFeatures: {FEATURE_NAMES} | window={WINDOW_SIZE} overlap={OVERLAP}")
    print("\nMacro-F1 when each load is held out:")
    for name in model_names:
        scores = [all_results[load][name][0] for load in LOADS_HP]
        per_load = "  ".join(f"load {load}: {score:.2f}" for load, score in zip(LOADS_HP, scores))
        print(f"  {name:28s} {per_load}   mean: {np.mean(scores):.2f}")

    print(f"\nConfusion matrix (rows = true, columns = predicted), summed over folds. Order: {CLASS_ORDER}")
    summed = sum(all_results[load]["logistic regression"][1] for load in LOADS_HP)
    print(summed)


def main():
    recordings = load_recording_table()
    print(f"{len(recordings)} labelled recordings")
    window_table = build_window_table(recordings)
    print(f"{len(window_table)} windows")

    all_results = {load: evaluate_held_out_load(window_table, load) for load in LOADS_HP}
    report(all_results)


if __name__ == "__main__":
    main()