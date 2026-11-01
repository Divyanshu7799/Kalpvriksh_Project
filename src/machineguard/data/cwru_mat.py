"""Reading CWRU MAT files: finding signals, rpm, duplicates, and the verified sampling rate."""

import hashlib
import re

import numpy as np

SIGNAL_PATTERN = re.compile(r"^X(\d+)_(DE|FE|BA)_time$")
RPM_PATTERN = re.compile(r"^X(\d+)RPM$")

# Sampling rates verified for the selected CWRU collection.
NORMAL_BASELINE_FOLDER = "norml"
NORMAL_BASELINE_FILES = {"97.mat", "98.mat", "99.mat", "100.mat"}
NORMAL_BASELINE_RATE_HZ = 48000
FAULT_FOLDERS = {"BF", "IR", "OR"}
FAULT_RATE_HZ = 12000


def get_sampling_rate_hz(recording_path):
    """Return the verified sampling rate of a recording, or None when it is not known."""
    folder = recording_path.parent.name

    if folder == NORMAL_BASELINE_FOLDER and recording_path.name in NORMAL_BASELINE_FILES:
        return NORMAL_BASELINE_RATE_HZ

    if folder in FAULT_FOLDERS:
        return FAULT_RATE_HZ

    return None


def discover_signal_variables(mat_data):
    """Return recognized vibration signals and their internal IDs."""
    signals = []

    for variable_name, values in mat_data.items():
        match = SIGNAL_PATTERN.fullmatch(variable_name)

        if match:
            internal_id, channel = match.groups()
            signals.append(
                {"variable": variable_name, "internal_id": internal_id, "channel": channel, "values": values}
            )

    return signals


def extract_recorded_rpm(mat_data):
    """Extract available RPM values without inventing missing values."""
    rpm_values = []

    for variable_name, values in mat_data.items():
        if RPM_PATTERN.fullmatch(variable_name):
            rpm_values.append(int(values.item()))

    return sorted(set(rpm_values))


def calculate_signal_hash(signal):
    """Calculate a stable hash of a signal's array contents."""
    values = np.ascontiguousarray(signal)
    return hashlib.sha256(values.tobytes()).hexdigest()
