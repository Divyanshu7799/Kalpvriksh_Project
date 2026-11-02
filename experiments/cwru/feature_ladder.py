"""Feature ladder experiment: a dummy baseline versus a classifier on chosen features.

Question answered: "How much do these features tell us about the 4 CWRU classes
(Normal, B, IR, OR) on a load the model has never seen?"

Run from the project root, naming the features to use:
    python experiments/cwru/feature_ladder.py --features rms
    python experiments/cwru/feature_ladder.py --features rms kurtosis
    python experiments/cwru/feature_ladder.py --features rms kurtosis crest_factor
    python experiments/cwru/feature_ladder.py --features rms kurtosis --window-size 4096   # window-size study
    python experiments/cwru/feature_ladder.py --features rms band_energy_ratio env_bpfo env_bpfi env_2xbsf

Add --final only once, when the feature set is frozen: it trains on loads 0-2 and tests on load 3.
"""

import argparse
import logging
import sys

import numpy as np

from machineguard.config import CLASS_ORDER, DEFAULT_WINDOW_SIZE, DEVELOPMENT_LOADS_HP, LOCKBOX_LOAD_HP, WINDOW_OVERLAP
from machineguard.data.recording_table import load_window_table
from machineguard.experiment_log import count_lockbox_runs, record_experiment
from machineguard.features.registry import FEATURES
from machineguard.models.evaluation import LOGISTIC_REGRESSION, evaluate_held_out_load, summarise

logger = logging.getLogger(__name__)


# --- Entry point: the only place that catches errors --------------------------------------------
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parse_arguments(argv)
    try:
        run_experiment(arguments.features, arguments.window_size, arguments.final)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Experiment stopped: %s", error)
        sys.exit(1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", nargs="+", default=["rms"], choices=sorted(FEATURES))
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="samples per window")
    parser.add_argument("--final", action="store_true", help="evaluate once on the lock-box load")
    return parser.parse_args(argv)


# --- The experiment, step by step ----------------------------------------------------------------
def run_experiment(feature_names, window_size, final):
    if final:
        warn_if_lockbox_already_used()
    window_table = load_window_table(feature_names, window_size)
    logger.info("%d windows", len(window_table))

    loads_to_test, usable_windows = choose_loads(window_table, final)
    all_results = {load: evaluate_held_out_load(usable_windows, feature_names, load) for load in loads_to_test}
    report(all_results, feature_names, window_size)
    arguments = {"features": feature_names, "window_size": window_size, "final": final}
    record_experiment("feature_ladder", arguments, summarise(all_results))


def warn_if_lockbox_already_used():
    previous_looks = count_lockbox_runs()
    if previous_looks:
        logger.warning(
            "The lock-box load was already evaluated %d time(s). Every extra look weakens the final test.",
            previous_looks,
        )


def choose_loads(window_table, final):
    """Return (loads to hold out one by one, the windows that may be used)."""
    if final:
        # Train on the development loads, test once on the untouched load.
        return [LOCKBOX_LOAD_HP], window_table
    # The lock-box load is used neither for training nor for testing.
    development_windows = window_table[window_table["load_hp"].isin(DEVELOPMENT_LOADS_HP)]
    return DEVELOPMENT_LOADS_HP, development_windows


def report(all_results, feature_names, window_size):
    loads = list(all_results)
    model_names = list(next(iter(all_results.values())).keys())

    print(f"\nFeatures: {feature_names} | window={window_size} overlap={WINDOW_OVERLAP}")
    print("\nMacro-F1 when each load is held out:")
    for name in model_names:
        scores = [all_results[load][name].macro_f1 for load in loads]
        per_load = "  ".join(f"load {load}: {score:.2f}" for load, score in zip(loads, scores))
        print(f"  {name:28s} {per_load}   mean: {np.mean(scores):.2f}")

    print(f"\nConfusion matrix (rows = true, columns = predicted), summed over folds. Order: {CLASS_ORDER}")
    print(sum(all_results[load][LOGISTIC_REGRESSION].confusion for load in loads))


if __name__ == "__main__":
    main()
