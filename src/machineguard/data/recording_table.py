"""Tables that tie each processed recording to its label, and each window to its features."""

from pathlib import PureWindowsPath

import numpy as np
import pandas as pd

from machineguard.data.preprocessing import split_into_windows
from machineguard.features.time_domain import FEATURE_FUNCTIONS


def file_name_of(source_file):
    """'data\\raw\\cwru\\BF\\118.mat' -> '118.mat' (accepts Windows and POSIX separators)."""
    return PureWindowsPath(source_file).name


def load_recording_table(manifest_file, labels_file):
    """Join the processed manifest with the labels (fault type, size, load) by file name."""
    manifest = pd.read_csv(manifest_file)
    labels = pd.read_csv(labels_file)

    if not labels["file"].is_unique:
        raise ValueError("labels.csv contains duplicate file names.")

    manifest["file"] = manifest["source_file"].map(file_name_of)
    table = manifest[["file", "output_file"]].merge(
        labels[["file", "fault_type", "size_in", "load_hp"]], on="file", how="left"
    )
    table["load_hp"] = pd.to_numeric(table["load_hp"], errors="coerce")

    unlabelled = table[table["load_hp"].isna() | table["fault_type"].isna()]
    if not unlabelled.empty:
        raise ValueError(f"Recordings without a known load or fault type: {unlabelled['file'].tolist()}")
    return table


def build_window_table(recordings, processed_dir, feature_names, window_size, overlap):
    """One row per window: its features plus which recording, class and load it came from."""
    parts = []
    for recording in recordings.itertuples():
        signal = np.load(processed_dir / recording.output_file)
        windows = split_into_windows(signal, window_size, overlap)
        part = pd.DataFrame({name: FEATURE_FUNCTIONS[name](windows) for name in feature_names})
        part["file"] = recording.file
        part["fault_type"] = recording.fault_type
        part["load_hp"] = int(recording.load_hp)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
