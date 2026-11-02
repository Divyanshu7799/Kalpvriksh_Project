"""Choose the feature set by forward selection with a stopping rule, then check it with an ablation.

Question answered: "Which features does the classifier actually need, judged on loads it has never seen?"

Run from the project root, after screen_feature_pool.py (pass the survivors it printed):
    python experiments/cwru/select_features.py --candidates rms kurtosis env_bpfo env_bpfi band_energy_ratio
Uses development loads only. Output: printed trace and docs/feature_engineering/feature_selection.csv.
"""

import argparse
import logging
import sys

import pandas as pd

from machineguard.config import DEFAULT_WINDOW_SIZE, DEVELOPMENT_LOADS_HP, FEATURE_ENGINEERING_DIR
from machineguard.data.recording_table import load_window_table
from machineguard.experiment_log import record_experiment
from machineguard.features.registry import FEATURES
from machineguard.features.screening import require_complete
from machineguard.models.evaluation import LOGISTIC_REGRESSION
from machineguard.models.selection import (
    CHANCE_MACRO_F1,
    SELECTION_MODELS,
    ablation,
    forward_select,
    selected_features,
)

logger = logging.getLogger(__name__)

OUTPUT_FILE = FEATURE_ENGINEERING_DIR / "feature_selection.csv"  # the latest run; the log keeps all runs


# --- Entry point: the only place that catches errors --------------------------------------------
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parse_arguments(argv)
    try:
        run_selection(arguments.candidates, arguments.window_size, arguments.model)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Selection stopped: %s", error)
        sys.exit(1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", required=True, choices=sorted(FEATURES))
    parser.add_argument(
        "--model", choices=SELECTION_MODELS, default=LOGISTIC_REGRESSION, help="model that judges features"
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="samples per window")
    return parser.parse_args(argv)


# --- The selection, step by step -----------------------------------------------------------------
def run_selection(candidates, window_size, model_name):
    window_table = load_window_table(candidates, window_size)
    window_table = window_table[window_table["load_hp"].isin(DEVELOPMENT_LOADS_HP)]
    require_complete(window_table, candidates)

    steps = forward_select(window_table, candidates, DEVELOPMENT_LOADS_HP, model_name)
    chosen = selected_features(steps)
    full_score, losses = (
        ablation(window_table, chosen, DEVELOPMENT_LOADS_HP, model_name) if chosen else (CHANCE_MACRO_F1, {})
    )

    print_report(steps, chosen, full_score, losses, model_name)
    save_selection(steps, model_name)
    results = {
        "selected": chosen,
        "steps": [
            {"feature": step.feature, "mean_f1": round(step.mean_f1, 4), "accepted": step.accepted} for step in steps
        ],
        "ablation_loss": {name: round(loss, 4) for name, loss in losses.items()},
    }
    record_experiment(
        "select_features", {"candidates": candidates, "window_size": window_size, "model": model_name}, results
    )


def print_report(steps, chosen, full_score, losses, model_name):
    print(f"\nForward selection judged by: {model_name}")
    print(f"Start: chance level {CHANCE_MACRO_F1:.2f}; a feature is kept if gain > fold spread")
    for number, step in enumerate(steps, start=1):
        verdict = "KEEP" if step.accepted else "STOP: not worth adding"
        print(
            f"  {number}. + {step.feature:18s} F1 {step.mean_f1:.2f}  gain {step.gain:+.3f}  "
            f"fold spread {step.fold_std:.3f}  {verdict}"
        )

    print(f"\nSelected set ({len(chosen)}): {chosen}")
    if chosen:
        print(f"\nAblation (macro-F1 {full_score:.2f} with all selected; loss when one is removed):")
        for name, loss in sorted(losses.items(), key=lambda item: -item[1]):
            print(f"  without {name:18s} loses {loss:+.3f}")


def save_selection(steps, model_name):
    table = pd.DataFrame(steps)
    table["model"] = model_name
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
