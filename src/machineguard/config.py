"""Project-wide file locations. Every script imports its paths from here, so each is defined once.

Assumes the package is installed in editable mode (`python -m pip install -e .`), so that
this file stays inside the project's src/ folder.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CWRU_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
CWRU_INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "cwru"
CWRU_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cwru"

CWRU_LABELS_FILE = CWRU_INTERIM_DIR / "labels.csv"
CWRU_MANIFEST_FILE = CWRU_INTERIM_DIR / "recording_manifest.csv"
CWRU_PROCESSED_MANIFEST_FILE = CWRU_PROCESSED_DIR / "processed_manifest.csv"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
