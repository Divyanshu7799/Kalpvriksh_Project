
"""Turn plain values into printable lines. No data loading, no models: easy to test."""
 
 
def describe_removed_duplicate(name, kept_name, correlation):
     return f"{name} removed: duplicate of {kept_name} (correlation {correlation:.2f})"
    # return "rms removed: duplicate of std (correlation 0.98)"
 
 
def describe_selection_round(feature, mean_f1, fold_std, gain, accepted):
    """One printable line for a forward-selection round."""
    verdict = "kept" if accepted else "rejected (gain not above fold spread)"
    return f"{feature}: F1 {mean_f1:.2f}, gain {gain:+.2f}, spread {fold_std:.2f} -> {verdict}"
 
 
def describe_final_set(features, macro_f1):
    """The closing line: which features were chosen and how well they score."""
    return f"final set ({len(features)}): {', '.join(features)} -> macro-F1 {macro_f1:.2f}"
 