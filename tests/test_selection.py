import numpy as np
import pandas as pd

from machineguard.models.selection import ablation, forward_select, selected_features

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
