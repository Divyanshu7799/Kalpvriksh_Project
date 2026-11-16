import numpy as np
import pandas as pd
import pytest

from machineguard.features.screening import (
    correlation_matrix,
    mutual_information,
    prune_redundant,
    redundant_pairs,
    require_complete,
    single_feature_scores,
)

CLASSES = ["Normal", "B", "IR", "OR"]


def make_table(per_class_per_load=30, seed=0):
    """Windows over three loads. 'informative' separates the classes, 'copy' duplicates it, 'noise' is random."""
    rng = np.random.default_rng(seed)
    rows = []
    for load in (0, 1, 2):
        for class_index, fault_type in enumerate(CLASSES):
            informative = class_index + rng.normal(scale=0.1, size=per_class_per_load)
            rows.append(
                pd.DataFrame(
                    {
                        "informative": informative,
                        "copy": informative * 3 + 1,
                        "noise": rng.normal(size=per_class_per_load),
                        "fault_type": fault_type,
                        "load_hp": load,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_correlation_of_a_rescaled_copy_is_one():
    correlations = correlation_matrix(make_table(), ["informative", "copy", "noise"])

    assert correlations.loc["informative", "copy"] == pytest.approx(1.0)
    assert abs(correlations.loc["informative", "noise"]) < 0.3


def test_redundant_pairs_finds_only_the_copy():
    correlations = correlation_matrix(make_table(), ["informative", "copy", "noise"])

    pairs = redundant_pairs(correlations)

    assert [(first, second) for first, second, _ in pairs] == [("informative", "copy")]


def test_mutual_information_ranks_informative_above_noise():
    scores = mutual_information(make_table(), ["informative", "noise"])

    assert scores.index[0] == "informative"
    assert scores["informative"] > scores["noise"] + 0.5


def test_single_feature_scores_are_high_for_informative_and_low_for_noise():
    scores = single_feature_scores(make_table(), ["informative", "noise"], held_out_loads=[0, 1, 2])

    assert scores["informative"] > 0.9
    assert scores["noise"] < 0.5


def test_prune_keeps_the_more_relevant_of_two_duplicates_and_says_why():
    table = make_table()
    correlations = correlation_matrix(table, ["informative", "copy", "noise"])
    ranking = pd.Series({"copy": 0.9, "informative": 0.8, "noise": 0.1})

    kept, dropped = prune_redundant(ranking, correlations)

    assert kept == ["copy", "noise"]
    assert dropped["informative"][0] == "copy"
    assert dropped["informative"][1] == pytest.approx(1.0)


def test_require_complete_rejects_missing_values():
    table = make_table()
    table.loc[0, "noise"] = np.nan

    with pytest.raises(ValueError, match="noise"):
        require_complete(table, ["informative", "noise"])


def test_redundant_pairs_lists_the_strongest_pair_first():
    names = ["a", "b", "c"]
    correlations = pd.DataFrame(
        [[1.0, 0.96, 0.99], [0.96, 1.0, 0.0], [0.99, 0.0, 1.0]], index=names, columns=names
    )

    pairs = redundant_pairs(correlations)

    assert [(first, second) for first, second, _ in pairs] == [("a", "c"), ("a", "b")]
