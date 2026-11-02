"""Evidence for choosing features without trial and error: redundancy, relevance, and a pruning rule.

Every function here is pure (tables in, tables out). The experiment script decides what to do with the output.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

from machineguard.models.evaluation import evaluate_held_out_load

REDUNDANCY_THRESHOLD = 0.95  # |Spearman correlation| above this means "carries the same information"
SINGLE_FEATURE_TREE_DEPTH = 3  # shallow, so one feature cannot be memorised; allows non-linear class ranges


def require_complete(window_table, feature_names):
    """Fail loudly if any feature has a missing value, instead of silently dropping or filling it."""
    missing = [name for name in feature_names if window_table[name].isna().any()]
    if missing:
        raise ValueError(f"Features with missing values (check all-zero windows): {missing}")


def correlation_matrix(window_table, feature_names):
    """Spearman correlation between every pair of features (rank-based, so robust to skewed features)."""
    return window_table[feature_names].corr(method="spearman")


def mutual_information(window_table, feature_names):
    """How much knowing the feature reduces uncertainty about the class (0 = nothing). Sorted, highest first."""
    scores = mutual_info_classif(window_table[feature_names], window_table["fault_type"], random_state=0)
    return pd.Series(scores, index=feature_names).sort_values(ascending=False)


def single_feature_scores(window_table, feature_names, held_out_loads):
    """Mean macro-F1 of each feature used alone, over folds that each hold one load out."""
    models = {
        "tree": DecisionTreeClassifier(max_depth=SINGLE_FEATURE_TREE_DEPTH, class_weight="balanced", random_state=0)
    }
    scores = {}
    for name in feature_names:
        fold_scores = [
            evaluate_held_out_load(window_table, [name], load, models)["tree"].macro_f1 for load in held_out_loads
        ]
        scores[name] = float(np.mean(fold_scores))
    return pd.Series(scores).sort_values(ascending=False)


def redundant_pairs(correlations, threshold=REDUNDANCY_THRESHOLD):
    """Feature pairs whose absolute correlation exceeds the threshold, strongest first."""
    names = list(correlations.columns)
    pairs = [
        (first, second, float(correlations.loc[first, second]))
        for position, first in enumerate(names)
        for second in names[position + 1 :]
        if abs(correlations.loc[first, second]) > threshold
    ]
    return sorted(pairs, key=lambda pair: -abs(pair[2]))


def prune_redundant(ranking, correlations, threshold=REDUNDANCY_THRESHOLD):
    """Walk features from most to least relevant; drop one that duplicates a feature already kept.

    Returns (kept feature names, {dropped name: (kept name it duplicates, correlation)}).
    """
    kept, dropped = [], {}
    for name in ranking.index:
        duplicate_of = next((other for other in kept if abs(correlations.loc[name, other]) > threshold), None)
        if duplicate_of is None:
            kept.append(name)
        else:
            dropped[name] = (duplicate_of, float(correlations.loc[name, duplicate_of]))
    return kept, dropped
