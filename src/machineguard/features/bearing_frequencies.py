"""Bearing defect frequencies: how often a fault produces a burst, from bearing geometry and shaft speed."""

# Multipliers of the shaft rotation frequency, computed from the CWRU bearing geometry
# (9 balls, ball 0.3126 in, pitch 1.537 in, contact angle 0). VERIFY the geometry on the CWRU page.
DEFECT_MULTIPLIERS = {"BPFO": 3.5848, "BPFI": 5.4152, "2xBSF": 4.7134, "FTF": 0.39831}


def defect_frequencies_hz(rpm):
    """Defect frequencies in Hz for a given shaft speed in revolutions per minute."""
    shaft_hz = rpm / 60
    return {name: multiplier * shaft_hz for name, multiplier in DEFECT_MULTIPLIERS.items()}
