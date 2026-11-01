"""The list of every feature the experiments can ask for, and one function to compute them.

Every registered feature takes (windows, sample_rate_hz, rpm) and returns one value per window.
To add a feature: write the function in time_domain.py or frequency_domain.py, then register it here.
"""

import pandas as pd

from machineguard.features import frequency_domain, time_domain


def _ignores_context(function):
    """Adapt a feature that needs only the windows to the common (windows, sample_rate_hz, rpm) form."""
    return lambda windows, sample_rate_hz, rpm: function(windows)


FEATURES = {
    **{name: _ignores_context(function) for name, function in time_domain.FEATURE_FUNCTIONS.items()},
    "band_energy_ratio": lambda windows, sample_rate_hz, rpm: frequency_domain.band_energy_ratio(
        windows, sample_rate_hz
    ),
    "env_bpfo": frequency_domain.defect_ratio_feature("BPFO"),
    "env_bpfi": frequency_domain.defect_ratio_feature("BPFI"),
    "env_2xbsf": frequency_domain.defect_ratio_feature("2xBSF"),
}


def compute_features(feature_names, windows, sample_rate_hz, rpm):
    """Table with one column per requested feature and one row per window."""
    unknown = [name for name in feature_names if name not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown features {unknown}. Available: {sorted(FEATURES)}")
    return pd.DataFrame({name: FEATURES[name](windows, sample_rate_hz, rpm) for name in feature_names})
