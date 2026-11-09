"""Choose features by evidence: forward selection with a written stopping rule, then an ablation.

Both use held-out-load macro-F1 of a chosen model (logistic regression by default), so a feature is judged on whether it
helps on a load the model has never seen. Only development loads are ever passed in here.
"""

from collections import namedtuple

import numpy as np

from machineguard.config import CLASS_ORDER
from machineguard.models.evaluation import (
    DECISION_TREE,
    LOGISTIC_REGRESSION,
    evaluate_held_out_load,
    make_models,
    shallow_decision_tree,
)

# Macro-F1 of a classifier that knows nothing, with four balanced classes: the starting point of the first round.
CHANCE_MACRO_F1 = 1 / len(CLASS_ORDER)

# One round of forward selection: the best feature of the round, the set's mean score, its spread across folds,
# the gain over the previous set, whether the gain was large enough to keep the feature, and the scoreboard:
# {every candidate tried this round: mean macro-F1 of the set with that candidate added}.
SelectionStep = namedtuple("SelectionStep", ["feature", "mean_f1", "fold_std", "gain", "accepted", "scoreboard"])


SELECTION_MODELS = (LOGISTIC_REGRESSION, DECISION_TREE)


def selection_model(model_name):
    """An untrained model by name. The logistic regression is linear; the tree can split a feature into ranges."""
    if model_name == LOGISTIC_REGRESSION:
        return make_models()[LOGISTIC_REGRESSION]
    if model_name == DECISION_TREE:
        return shallow_decision_tree()
    raise ValueError(f"Unknown selection model {model_name!r}. Choose from {SELECTION_MODELS}")


def fold_scores(window_table, feature_names, held_out_loads, model_name=LOGISTIC_REGRESSION):
    """Macro-F1 of the named model for each held-out load."""
    return [
        evaluate_held_out_load(window_table, feature_names, load, {model_name: selection_model(model_name)})[
            model_name
        ].macro_f1
        for load in held_out_loads
    ]


def forward_select(window_table, candidates, held_out_loads, model_name=LOGISTIC_REGRESSION):
    """Add one feature at a time (the one that helps most) until the gain is no larger than the fold spread.

    Stopping rule, fixed before looking at results: a feature is kept only if the mean macro-F1 rises by more
    than the standard deviation of the new set's fold scores. Returns every round as a SelectionStep.
    """
    selected, remaining, steps = [], list(candidates), []
    previous_mean = CHANCE_MACRO_F1

    while remaining:
        scored = {name: fold_scores(window_table, selected + [name], held_out_loads, model_name) for name in remaining}
        best = max(scored, key=lambda name: np.mean(scored[name]))
        mean_f1, fold_std = float(np.mean(scored[best])), float(np.std(scored[best], ddof=1))
        gain = mean_f1 - previous_mean
        accepted = gain > fold_std
        scoreboard = {name: float(np.mean(scores)) for name, scores in scored.items()}
        steps.append(SelectionStep(best, mean_f1, fold_std, gain, accepted, scoreboard))
        if not accepted:
            break
        selected.append(best)
        remaining.remove(best)
        previous_mean = mean_f1
    return steps


def selected_features(steps):
    """Names of the features that forward selection kept."""
    return [step.feature for step in steps if step.accepted]


def ablation(window_table, feature_names, held_out_loads, model_name=LOGISTIC_REGRESSION):
    """Mean macro-F1 lost when each feature is removed in turn (a large loss means the feature matters).

    Removing the only feature leaves nothing to classify with, so it is compared with chance level.
    """
    full = float(np.mean(fold_scores(window_table, feature_names, held_out_loads, model_name)))
    losses = {}
    for name in feature_names:
        without = [other for other in feature_names if other != name]
        score_without = (
            np.mean(fold_scores(window_table, without, held_out_loads, model_name)) if without else CHANCE_MACRO_F1
        )
        losses[name] = full - float(score_without)
    return full, losses
