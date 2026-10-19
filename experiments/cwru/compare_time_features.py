
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cwru"
MANIFEST_PATH = PROCESSED_DIR / "processed_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs"

FEATURES = ["rms", "std", "kurtosis", "crest_factor"]
CLASS_ORDER = ["Normal", "B", "IR", "OR"]


def calculate_features(signal):
    """Calculate exploratory time-domain features for one recording."""
    signal = np.asarray(signal, dtype=float).ravel()

    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Signal must be non-empty and finite.")

    rms = np.sqrt(np.mean(signal**2))
    standard_deviation = np.std(signal)
    peak = np.max(np.abs(signal))

    return {
        "rms": rms,
        "std": standard_deviation,
        "kurtosis": kurtosis(signal, fisher=True, bias=False),
        "crest_factor": peak / rms if rms > 0 else np.nan,
    }


def build_feature_table():
    """Calculate features for every successfully processed recording."""
    with MANIFEST_PATH.open(
        newline="", encoding="utf-8"
    ) as file:
        manifest = list(csv.DictReader(file))

    records = []

    for row in manifest:
        if row["status"] != "processed":
            continue

        signal_path = PROCESSED_DIR / row["output_file"]
        signal = np.load(signal_path)

        features = calculate_features(signal)

        records.append(
            {
                "source_file": row["source_file"],
                "output_file": row["output_file"],
                "fault_type": row["fault_type"],
                "sampling_rate_hz": int(row["target_rate_hz"]),
                **features,
            }
        )

    return pd.DataFrame(records)


def plot_feature_distributions(features):
    """Compare feature distributions across fault classes."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    for axis, feature in zip(axes.flat, FEATURES):
        grouped_values = [
            features.loc[
                features["fault_type"] == fault_type, feature
            ].dropna().to_numpy()
            for fault_type in CLASS_ORDER
        ]

        axis.boxplot(
            grouped_values,
            tick_labels=CLASS_ORDER,
            showfliers=True,
        )
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel("Fault class")
        axis.set_ylabel("Feature value")
        axis.grid(axis="y", alpha=0.3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cwru_time_feature_distributions.png"
    fig.savefig(output_path, dpi=160)
    plt.show()

    print(f"Plot saved to: {output_path}")


def main():
    features = build_feature_table()

    csv_path = OUTPUT_DIR / "cwru_recording_time_features.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(csv_path, index=False)

    print(f"Recordings analyzed: {len(features)}")
    print("\nRecordings per class:")
    print(features["fault_type"].value_counts().reindex(CLASS_ORDER))

    print("\nFeature medians by class:")
    print(
        features.groupby("fault_type")[FEATURES]
        .median()
        .reindex(CLASS_ORDER)
        .round(5)
    )

    print(f"\nFeature table saved to: {csv_path}")
    plot_feature_distributions(features)


if __name__ == "__main__":
    main()
