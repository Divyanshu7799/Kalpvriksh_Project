"""Join provenance, screening results and selection results into one table: one row per feature, with a reason."""

import pandas as pd

from machineguard.features.provenance import PROVENANCE

COLUMNS = [
    "feature",
    "family",
    "definition",
    "hypothesis",
    "source",
    "source_status",
    "single_feature_f1",
    "mutual_information",
    "screening",
    "selection_gain",
    "selection_fold_spread",
    "final_decision",
    "reason",
]


def build_register(screening, selection):
    """screening: table from screen_feature_pool.py; selection: table from select_features.py."""
    screening = screening.set_index("feature")
    selection = selection.set_index("feature")
    rows = [register_row(name, PROVENANCE[name], screening, selection) for name in screening.index]
    return pd.DataFrame(rows, columns=COLUMNS)


def register_row(name, provenance, screening, selection):
    screened = screening.loc[name]
    tried = selection.loc[name] if name in selection.index else None
    decision, reason = decide(screened, tried)
    return {
        "feature": name,
        "family": provenance.family,
        "definition": provenance.definition,
        "hypothesis": provenance.hypothesis,
        "source": provenance.source,
        "source_status": provenance.source_status,
        "single_feature_f1": round(float(screened["single_feature_f1"]), 3),
        "mutual_information": round(float(screened["mutual_information"]), 3),
        "screening": screened["decision"],
        "selection_gain": None if tried is None else round(float(tried["gain"]), 3),
        "selection_fold_spread": None if tried is None else round(float(tried["fold_std"]), 3),
        "final_decision": decision,
        "reason": reason,
    }


def decide(screened, tried):
    """(final decision, reason) from what screening and selection said about one feature."""
    if screened["decision"] != "keep":
        return (
            "rejected",
            f"duplicate of {screened['duplicate_of']} (correlation {float(screened['correlation']):+.2f})",
        )
    if tried is None:
        return "rejected", "not selected: no remaining feature raised the score by more than the fold spread"
    if bool(tried["accepted"]):
        return (
            "selected",
            f"gain {float(tried['gain']):+.3f} is larger than the fold spread {float(tried['fold_std']):.3f}",
        )
    return (
        "rejected",
        f"gain {float(tried['gain']):+.3f} is not larger than the fold spread {float(tried['fold_std']):.3f}",
    )
