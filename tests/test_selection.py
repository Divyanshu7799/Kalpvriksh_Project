import numpy as np
import pandas as pd

from machineguard.models.evaluation import DECISION_TREE, LOGISTIC_REGRESSION
from machineguard.models.selection import ablation, forward_select, selected_features, selection_model

CLASSES = ["Normal", "B", "IR", "OR"]


def make_table(per_class_per_load=40, seed=0):
    """'useful_a' and 'useful_b' each separate two of the four classes; 'noise' separates none."""
    rng = np.random.default_rng(seed)
    parts = []
    for load in (0, 1, 2):
        for class_index, fault_type in enumerate(CLASSES):
            size = per_class_per_load
            parts.append(
                pd.DataFrame(
                    {
                        "useful_a": (class_index // 2) * 4 + rng.normal(scale=0.5, size=size),
                        "useful_b": (class_index % 2) * 4 + rng.normal(scale=0.5, size=size),
                        "noise": rng.normal(size=size),
                        "fault_type": fault_type,
                        "load_hp": load,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def test_forward_selection_keeps_both_useful_features_and_rejects_noise():
    steps = forward_select(make_table(), ["noise", "useful_a", "useful_b"], held_out_loads=[0, 1, 2])

    assert set(selected_features(steps)) == {"useful_a", "useful_b"}
    assert steps[-1].feature == "noise" and not steps[-1].accepted


def test_forward_selection_picks_a_helpful_feature_first():
    steps = forward_select(make_table(), ["noise", "useful_a", "useful_b"], held_out_loads=[0, 1, 2])

    assert steps[0].feature in {"useful_a", "useful_b"}
    assert steps[0].gain > 0.15


def test_forward_selection_with_only_noise_keeps_nothing():
    steps = forward_select(make_table(), ["noise"], held_out_loads=[0, 1, 2])

    assert selected_features(steps) == []


def test_ablation_shows_a_large_loss_for_useful_features_and_none_for_noise():
    full, losses = ablation(make_table(), ["useful_a", "useful_b", "noise"], held_out_loads=[0, 1, 2])

    assert full > 0.8
    assert losses["useful_a"] > 0.2 and losses["useful_b"] > 0.2
    assert abs(losses["noise"]) < 0.05


def test_ablation_of_a_single_feature_is_compared_with_chance_level():
    full, losses = ablation(make_table(), ["useful_a"], held_out_loads=[0, 1, 2])

    assert losses["useful_a"] > 0.1


def make_two_range_table(per_class_per_load=40, seed=1):
    """One feature where every fault class sits on both sides of Normal, so no single linear cut isolates a class."""
    rng = np.random.default_rng(seed)
    centres = {"Normal": [0], "B": [-4, 4], "IR": [-8, 8], "OR": [-12, 12]}
    parts = []
    for load in (0, 1, 2):
        for fault_type, centre_list in centres.items():
            centre = rng.choice(centre_list, size=per_class_per_load)
            parts.append(
                pd.DataFrame(
                    {
                        "x": centre + rng.normal(scale=0.4, size=per_class_per_load),
                        "fault_type": fault_type,
                        "load_hp": load,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def test_a_tree_judges_a_feature_with_separate_ranges_better_than_logistic_regression():
    table = make_two_range_table()

    linear = forward_select(table, ["x"], [0, 1, 2], LOGISTIC_REGRESSION)[0].mean_f1
    tree = forward_select(table, ["x"], [0, 1, 2], DECISION_TREE)[0].mean_f1

    assert tree > linear + 0.1


def test_selection_model_rejects_an_unknown_name():
    import pytest

    with pytest.raises(ValueError, match="Unknown selection model"):
        selection_model("magic")


def test_recall_per_class_and_missed_faults():
    from machineguard.models.evaluation import faults_missed_as_normal, recall_per_class

    confusion = np.array([[10, 0, 0, 0], [3, 7, 0, 0], [1, 0, 9, 0], [0, 2, 0, 8]])

    recalls = recall_per_class(confusion)

    assert recalls == {"Normal": 1.0, "B": 0.7, "IR": 0.9, "OR": 0.8}
    assert faults_missed_as_normal(confusion) == 4


def test_recall_of_a_class_with_no_windows_is_nan():
    from machineguard.models.evaluation import recall_per_class

    confusion = np.array([[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]])

    assert np.isnan(recall_per_class(confusion)["B"])


def test_each_round_records_the_score_of_every_candidate_tried():
    steps = forward_select(make_table(), ["noise", "useful_a", "useful_b"], held_out_loads=[0, 1, 2])

    assert set(steps[0].scoreboard) == {"noise", "useful_a", "useful_b"}
    assert max(steps[0].scoreboard, key=steps[0].scoreboard.get) == steps[0].feature
    assert set(steps[1].scoreboard) == {"noise", "useful_a", "useful_b"} - {steps[0].feature}
