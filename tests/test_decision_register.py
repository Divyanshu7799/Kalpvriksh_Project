import pandas as pd

from machineguard.features.decision_register import build_register
from machineguard.features.provenance import PROVENANCE


def screening_table():
    return pd.DataFrame(
        {
            "feature": ["rms", "std", "kurtosis", "skewness"],
            "single_feature_f1": [0.7, 0.69, 0.5, 0.2],
            "mutual_information": [0.5, 0.5, 0.3, 0.0],
            "decision": ["keep", "drop: duplicate", "keep", "keep"],
            "duplicate_of": ["", "rms", "", ""],
            "correlation": [None, 1.0, None, None],
        }
    )


def selection_table():
    return pd.DataFrame(
        {
            "feature": ["rms", "kurtosis"],
            "mean_f1": [0.7, 0.71],
            "fold_std": [0.05, 0.04],
            "gain": [0.45, 0.01],
            "accepted": [True, False],
        }
    )


def decisions():
    register = build_register(screening_table(), selection_table()).set_index("feature")
    return register


def test_a_kept_feature_with_a_large_gain_is_selected_with_the_numbers_as_reason():
    row = decisions().loc["rms"]

    assert row["final_decision"] == "selected"
    assert "+0.450" in row["reason"]


def test_a_duplicate_is_rejected_and_names_the_feature_it_duplicates():
    row = decisions().loc["std"]

    assert row["final_decision"] == "rejected"
    assert "duplicate of rms" in row["reason"]


def test_a_feature_whose_gain_was_within_the_fold_spread_is_rejected():
    row = decisions().loc["kurtosis"]

    assert row["final_decision"] == "rejected"
    assert "not larger than the fold spread" in row["reason"]


def test_a_feature_never_tried_in_selection_is_rejected_with_an_explanation():
    row = decisions().loc["skewness"]

    assert row["final_decision"] == "rejected"
    assert pd.isna(row["selection_gain"])


def test_every_registered_feature_has_a_source_and_an_honest_status():
    for name, provenance in PROVENANCE.items():
        assert provenance.source and provenance.hypothesis, name
        assert provenance.source_status in {"read", "exists", "none yet"}, name
