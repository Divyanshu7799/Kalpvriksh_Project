"""Project-wide file locations and constants. Every script imports them from here, so each is defined once."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CWRU_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
CWRU_INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "cwru"
CWRU_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cwru"

CWRU_LABELS_FILE = CWRU_INTERIM_DIR / "labels.csv"
CWRU_MANIFEST_FILE = CWRU_INTERIM_DIR / "recording_manifest.csv"
CWRU_PROCESSED_MANIFEST_FILE = CWRU_PROCESSED_DIR / "processed_manifest.csv"

DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Every processed CWRU recording is resampled to this rate (see preprocess_recordings.py).
PROCESSED_SAMPLE_RATE_HZ = 12000

FEATURE_ENGINEERING_DIR = DOCS_DIR / "feature_engineering"

# Windowing defaults (provisional until the window-size experiment) and the lock-box protocol:
# loads 0-2 are used to develop and compare feature sets; load 3 is touched once, with --final.
DEFAULT_WINDOW_SIZE = 2048
WINDOW_OVERLAP = 0.5
DEVELOPMENT_LOADS_HP = [0, 1, 2]
LOCKBOX_LOAD_HP = 3

# Display and evaluation order of the four bearing conditions.
CLASS_ORDER = ["Normal", "B", "IR", "OR"]


def project_relative(path):
    """Path relative to the project when possible (for readable logs and manifests), else the full path."""
    path = Path(path)
    return path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
