"""Screen the candidate feature pool with evidence, before any selection: redundancy and relevance.

Question answered: "Which candidates duplicate each other, which carry class information on their own,
and which small set is left after removing duplicates?"

Run from the project root (uses development loads only; the lock-box load is never touched):
    python experiments/cwru/screen_feature_pool.py
    python experiments/cwru/screen_feature_pool.py --window-size 4096
Output: a printed report and docs/feature_engineering/feature_screening.csv (for the decision register).
"""

import argparse
import logging
import sys

import pandas as pd

from machineguard.config import DEFAULT_WINDOW_SIZE, DEVELOPMENT_LOADS_HP, FEATURE_ENGINEERING_DIR
from machineguard.data.recording_table import load_window_table
from machineguard.features.registry import FEATURES
from machineguard.features.screening import (
    REDUNDANCY_THRESHOLD,
    correlation_matrix,
    mutual_information,
    prune_redundant,
    redundant_pairs,
    require_complete,
    single_feature_scores,
)

logger = logging.getLogger(__name__)

OUTPUT_FILE = FEATURE_ENGINEERING_DIR / "feature_screening.csv"


# --- Entry point: the only place that catches errors --------------------------------------------
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parse_arguments(argv)
    try:
        run_screening(sorted(FEATURES), arguments.window_size)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Screening stopped: %s", error)
        sys.exit(1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="samples per window")
    return parser.parse_args(argv)


# --- The screening, step by step -----------------------------------------------------------------
def run_screening(feature_names, window_size):
    window_table = load_development_windows(feature_names, window_size)
    correlations = correlation_matrix(window_table, feature_names)
    relevance = single_feature_scores(window_table, feature_names, DEVELOPMENT_LOADS_HP)
    information = mutual_information(window_table, feature_names)
    kept, dropped = prune_redundant(relevance, correlations)

    print_report(correlations, relevance, information, kept, dropped)
    save_screening(feature_names, relevance, information, kept, dropped)


def load_development_windows(feature_names, window_size):
    """Windows of the development loads only, with every candidate feature computed and checked."""
    window_table = load_window_table(feature_names, window_size)
    window_table = window_table[window_table["load_hp"].isin(DEVELOPMENT_LOADS_HP)]
    require_complete(window_table, feature_names)
    logger.info("%d development windows, %d candidate features", len(window_table), len(feature_names))
    return window_table


def print_report(correlations, relevance, information, kept, dropped):
    print("\nRelevance of each feature alone (macro-F1 on held-out development loads; chance is about 0.25):")
    for name, score in relevance.items():
        print(f"  {name:18s} F1 {score:.2f}   mutual information {information[name]:.2f}")

    print(f"\nRedundant pairs (|Spearman correlation| > {REDUNDANCY_THRESHOLD}):")
    for first, second, value in redundant_pairs(correlations):
        print(f"  {first} ~ {second}: {value:+.2f}")

    print("\nDropped as duplicates (the more relevant of the pair is kept):")
    for name, (duplicate_of, value) in dropped.items():
        print(f"  {name:18s} duplicates {duplicate_of} ({value:+.2f})")

    print(f"\nSurvivors for forward selection ({len(kept)}): {kept}")


def save_screening(feature_names, relevance, information, kept, dropped):
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "single_feature_f1": [relevance[name] for name in feature_names],
            "mutual_information": [information[name] for name in feature_names],
            "decision": ["keep" if name in kept else "drop: duplicate" for name in feature_names],
            "duplicate_of": [dropped.get(name, ("", ""))[0] for name in feature_names],
            "correlation": [dropped.get(name, ("", ""))[1] for name in feature_names],
        }
    ).sort_values("single_feature_f1", ascending=False)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
