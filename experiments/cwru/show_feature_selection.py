"""Show, in one terminal run, how the feature pool shrinks to the final set, and why.

Question answered: "Which features were removed, which were kept, and what is the evidence?"

Run from the project root (development loads only; the lock-box load is never touched):
    python experiments/cwru/show_feature_selection.py
    python experiments/cwru/show_feature_selection.py --model "logistic regression" --window-size 4096
"""

import argparse
import logging
import sys

from machineguard.config import DEFAULT_WINDOW_SIZE, DEVELOPMENT_LOADS_HP
from machineguard.data.recording_table import load_window_table
from machineguard.experiment_log import record_experiment
from machineguard.features.registry import FEATURES
from machineguard.features.screening import (
    REDUNDANCY_THRESHOLD,
    correlation_matrix,
    prune_redundant,
    require_complete,
    single_feature_scores,
)
from machineguard.models.evaluation import DECISION_TREE
from machineguard.models.selection import (
    CHANCE_MACRO_F1,
    SELECTION_MODELS,
    ablation,
    forward_select,
    selected_features,
)
from machineguard.reporting import describe_final_set, describe_removed_duplicate, describe_selection_round

logger = logging.getLogger(__name__)


# --- Entry point: the only place that catches errors --------------------------------------------
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parse_arguments(argv)
    try:
        show_feature_selection(arguments.window_size, arguments.model)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Report stopped: %s", error)
        sys.exit(1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=SELECTION_MODELS, default=DECISION_TREE, help="model that judges features")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="samples per window")
    return parser.parse_args(argv)


# --- The report, one section per question ---------------------------------------------------------
def show_feature_selection(window_size, model_name):
    candidates = sorted(FEATURES)
    window_table = load_development_windows(candidates, window_size)

    survivors = show_duplicate_removal(window_table, candidates)
    steps = show_forward_selection(window_table, survivors, model_name)
    chosen = selected_features(steps)
    final_score = show_ablation(window_table, chosen, model_name)

    print_heading("5. Result")
    print(describe_final_set(chosen, final_score))
    record_experiment(
        "show_feature_selection",
        {"window_size": window_size, "model": model_name},
        {"survivors": survivors, "selected": chosen, "macro_f1": round(final_score, 4)},
    )


def load_development_windows(candidates, window_size):
    window_table = load_window_table(candidates, window_size)
    window_table = window_table[window_table["load_hp"].isin(DEVELOPMENT_LOADS_HP)]
    require_complete(window_table, candidates)
    return window_table


def show_duplicate_removal(window_table, candidates):
    print_heading(f"1. Candidate pool ({len(candidates)} features)")
    print(", ".join(candidates))

    print_heading(f"2. Duplicates removed (|Spearman correlation| > {REDUNDANCY_THRESHOLD})")
    relevance = single_feature_scores(window_table, candidates, DEVELOPMENT_LOADS_HP)
    survivors, dropped = prune_redundant(relevance, correlation_matrix(window_table, candidates))
    for name, (kept_name, correlation) in dropped.items():
        print("  " + describe_removed_duplicate(name, kept_name, correlation))

    print_heading(f"3. Survivors, each scored alone (chance is {CHANCE_MACRO_F1:.2f})")
    for name in survivors:
        print(f"  {name:18s} macro-F1 {relevance[name]:.2f}")
    return survivors


def show_forward_selection(window_table, survivors, model_name):
    print_heading(f"4. Forward selection judged by {model_name}")
    steps = forward_select(window_table, survivors, DEVELOPMENT_LOADS_HP, model_name)
    for number, step in enumerate(steps, start=1):
        print(f"  round {number}: " + describe_selection_round(*step[:5]))
    return steps


def show_ablation(window_table, chosen, model_name):
    if not chosen:
        return CHANCE_MACRO_F1
    full_score, losses = ablation(window_table, chosen, DEVELOPMENT_LOADS_HP, model_name)
    print("\n  Ablation: score lost when one selected feature is removed")
    for name, loss in sorted(losses.items(), key=lambda item: -item[1]):
        print(f"    without {name:18s} loses {loss:+.2f}")
    return full_score


def print_heading(text):
    print(f"\n=== {text} ===")


if __name__ == "__main__":
    main()