"""Compare time-domain features across the four classes, one value per recording.

Run from the project root (after preprocess_recordings.py):
    python experiments/cwru/compare_time_features.py
"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from machineguard.config import CLASS_ORDER, DOCS_DIR
from machineguard.data.recording_table import load_processed_signal, read_processed_manifest
from machineguard.features.time_domain import FEATURE_FUNCTIONS

logger = logging.getLogger(__name__)

FEATURES = ["rms", "std", "kurtosis", "crest_factor"]
TABLE_PATH = DOCS_DIR / "cwru_recording_time_features.csv"
FIGURE_PATH = DOCS_DIR / "cwru_time_feature_distributions.png"


# --- Level 1: the whole job ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_comparison()
    except (OSError, ValueError, KeyError) as error:
        logger.error("Comparison stopped: %s", error)
        sys.exit(1)


def run_comparison():
    table = build_feature_table(read_processed_manifest())

    save_table(table, TABLE_PATH)
    print_summary(table)
    save_figure(draw_distributions(table), FIGURE_PATH)


# --- Level 2: the steps --------------------------------------------------------------------------
def build_feature_table(manifest_rows):
    """One row per processed recording, with the features computed over the whole recording."""
    return pd.DataFrame([feature_record(row) for row in manifest_rows])


def save_table(table, table_path):
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    logger.info("Feature table saved to: %s", table_path)


def print_summary(table):
    print(f"Recordings analyzed: {len(table)}")
    print("\nRecordings per class:")
    print(table["fault_type"].value_counts().reindex(CLASS_ORDER))
    print("\nFeature medians by class:")
    print(table.groupby("fault_type")[FEATURES].median().reindex(CLASS_ORDER).round(5))


def draw_distributions(table):
    """One box plot per feature, comparing the four classes."""
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), constrained_layout=True)

    for axis, feature in zip(axes.flat, FEATURES):
        values_by_class = [
            table.loc[table["fault_type"] == fault_type, feature].dropna().to_numpy() for fault_type in CLASS_ORDER
        ]
        axis.boxplot(values_by_class, tick_labels=CLASS_ORDER, showfliers=True)
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel("Fault class")
        axis.set_ylabel("Feature value")
        axis.grid(axis="y", alpha=0.3)
    return figure


def save_figure(figure, figure_path):
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    logger.info("Plot saved to: %s", figure_path)


# --- Level 3: one recording ----------------------------------------------------------------------
def feature_record(row):
    """The manifest columns that identify a recording, plus its features."""
    return {
        "source_file": row["source_file"],
        "output_file": row["output_file"],
        "fault_type": row["fault_type"],
        "sampling_rate_hz": int(row["target_rate_hz"]),
        **recording_features(load_processed_signal(row["output_file"])),
    }


def recording_features(signal):
    """Features of a whole recording, using the same definitions as the windowed experiments."""
    as_one_window = signal[np.newaxis, :]
    return {name: float(FEATURE_FUNCTIONS[name](as_one_window)[0]) for name in FEATURES}


if __name__ == "__main__":
    main()
