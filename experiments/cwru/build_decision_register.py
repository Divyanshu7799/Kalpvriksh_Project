"""Build the decision register: for every feature, its source, hypothesis, test results and final decision.

Run from the project root, after screen_feature_pool.py and select_features.py:
    python experiments/cwru/build_decision_register.py
Reads docs/feature_engineering/feature_screening.csv and feature_selection.csv.
Writes docs/feature_engineering/decision_register.csv. The reasons come from the saved results, not from memory.
"""

import logging
import sys

import pandas as pd

from machineguard.config import FEATURE_ENGINEERING_DIR
from machineguard.features.decision_register import build_register
from machineguard.features.provenance import PROVENANCE

logger = logging.getLogger(__name__)

SCREENING_FILE = FEATURE_ENGINEERING_DIR / "feature_screening.csv"
SELECTION_FILE = FEATURE_ENGINEERING_DIR / "feature_selection.csv"
REGISTER_FILE = FEATURE_ENGINEERING_DIR / "decision_register.csv"


# --- Entry point: the only place that catches errors --------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        write_register()
    except (FileNotFoundError, ValueError, KeyError) as error:
        logger.error("Register not built: %s", error)
        sys.exit(1)


def write_register():
    screening = pd.read_csv(SCREENING_FILE, keep_default_na=False, na_values=[""])
    selection = pd.read_csv(SELECTION_FILE)
    unknown = [name for name in screening["feature"] if name not in PROVENANCE]
    if unknown:
        raise ValueError(f"No provenance written for: {unknown}. Add them to features/provenance.py")

    register = build_register(screening, selection)
    register.to_csv(REGISTER_FILE, index=False)
    print_summary(register)
    logger.info("Saved %s", REGISTER_FILE)


def print_summary(register):
    for decision in ("selected", "rejected"):
        print(f"\n{decision.upper()}")
        for row in register[register["final_decision"] == decision].itertuples():
            print(f"  {row.feature:18s} {row.reason}")
    unverified = register[register["source_status"] != "read"]["feature"].tolist()
    print(f"\nSources not yet read or missing for: {unverified}")


if __name__ == "__main__":
    main()
