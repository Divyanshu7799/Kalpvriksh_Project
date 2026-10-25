"""Feature ladder experiment: a dummy baseline versus a classifier on chosen features.

Question answered: "How much do these features tell us about the 4 CWRU classes
(Normal, B, IR, OR) on a load the model has never seen?"

Run from the project root, naming the features to use:
    python experiments/cwru/baseline_rms.py --features rms                 # rung 1
    python experiments/cwru/baseline_rms.py --features rms kurtosis        # rung 2a
    python experiments/cwru/baseline_rms.py --features rms crest_factor    # rung 2b

Add --final only once, when the feature set is frozen: it trains on loads 0-2 and tests on load 3.
"""

import argparse
import logging
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from machineguard.data.preprocessing import split_into_windows

logger = logging.getLogger(__name__)

# --- Configuration -------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cwru"
PROCESSED_MANIFEST = PROCESSED_DIR / "processed_manifest.csv"
LABELS_FILE = PROJECT_ROOT / "data" / "interim" / "cwru" / "labels.csv"

# Provisional: to be chosen by the window-size experiment, not by habit.
WINDOW_SIZE = 2048
OVERLAP = 0.5

CLASS_ORDER = ["Normal", "B", "IR", "OR"]
# Loads 0-2 are used to develop and compare feature sets. Load 3 is a lock-box: it is
# touched once, at the very end, with --final.
DEVELOPMENT_LOADS_HP = [0, 1, 2]
LOCKBOX_LOAD_HP = 3


def rms_per_window(windows):
    """Root mean square of each window (one value per row)."""
    return np.sqrt(np.mean(windows**2, axis=1))


def kurtosis_per_window(windows):
    """Excess kurtosis of each window: about 0 for Gaussian noise, large for impulsive signals."""
    return kurtosis(windows, axis=1)


def crest_factor_per_window(windows):
    """Peak absolute amplitude divided by RMS, for each window."""
    return np.max(np.abs(windows), axis=1) / rms_per_window(windows)


# To add a feature: write a function like the ones above and register it here.
FEATURE_FUNCTIONS = {
    "rms": rms_per_window,
    "kurtosis": kurtosis_per_window,
    "crest_factor": crest_factor_per_window,
}


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
        raise ValueError(f"Recordings without a known load or fault type: {unlabelled['file'].tolist()}")
    return table


# --- Step 2: cut each recording into windows and compute features --------------------------------
def build_window_table(recordings, feature_names):
    """One row per window: its features plus which recording, class and load it came from."""
    parts = []
    for recording in recordings.itertuples():
        signal = np.load(PROCESSED_DIR / recording.output_file)
        windows = split_into_windows(signal, WINDOW_SIZE, OVERLAP)
        part = pd.DataFrame({name: FEATURE_FUNCTIONS[name](windows) for name in feature_names})
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


def evaluate_held_out_load(window_table, feature_names, held_out_load):
    """Return {model name: (macro-F1, confusion matrix)} for one held-out load."""
    train = window_table[window_table["load_hp"] != held_out_load]
    test = window_table[window_table["load_hp"] == held_out_load]

    results = {}
    for name, model in make_models().items():
        model.fit(train[feature_names], train["fault_type"])
        predicted = model.predict(test[feature_names])
        macro_f1 = f1_score(test["fault_type"], predicted, labels=CLASS_ORDER, average="macro", zero_division=0)
        matrix = confusion_matrix(test["fault_type"], predicted, labels=CLASS_ORDER)
        results[name] = (macro_f1, matrix)
    return results


# --- Step 4: report ------------------------------------------------------------------------------
def report(all_results, feature_names):
    loads = list(all_results)
    model_names = list(next(iter(all_results.values())).keys())

    print(f"\nFeatures: {feature_names} | window={WINDOW_SIZE} overlap={OVERLAP}")
    print("\nMacro-F1 when each load is held out:")
    for name in model_names:
        scores = [all_results[load][name][0] for load in loads]
        per_load = "  ".join(f"load {load}: {score:.2f}" for load, score in zip(loads, scores))
        print(f"  {name:28s} {per_load}   mean: {np.mean(scores):.2f}")

    print(f"\nConfusion matrix (rows = true, columns = predicted), summed over folds. Order: {CLASS_ORDER}")
    summed = sum(all_results[load]["logistic regression"][1] for load in loads)
    print(summed)


# --- Entry point: the only place that catches errors --------------------------------------------
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", nargs="+", default=["rms"], choices=sorted(FEATURE_FUNCTIONS))
    parser.add_argument("--final", action="store_true", help="evaluate once on the lock-box load")
    return parser.parse_args(argv)


def run_baseline(feature_names, final):
    recordings = load_recording_table()
    logger.info("%d labelled recordings", len(recordings))
    window_table = build_window_table(recordings, feature_names)
    logger.info("%d windows", len(window_table))

    if final:
        # Train on the development loads, test once on the untouched load.
        loads_to_test = [LOCKBOX_LOAD_HP]
        usable_windows = window_table
    else:
        # The lock-box load is used neither for training nor for testing.
        loads_to_test = DEVELOPMENT_LOADS_HP
        usable_windows = window_table[window_table["load_hp"].isin(DEVELOPMENT_LOADS_HP)]

    all_results = {load: evaluate_held_out_load(usable_windows, feature_names, load) for load in loads_to_test}
    report(all_results, feature_names)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parse_arguments(argv)
    try:
        run_baseline(arguments.features, arguments.final)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Baseline stopped: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
