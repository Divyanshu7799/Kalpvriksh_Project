"""A permanent record of every experiment run: when, with which code, which settings, which results.

Each run appends one row to docs/feature_engineering/experiment_log.csv, so every number in the decision
register can be traced to a run, and the number of looks at the lock-box load can be counted.
"""

import csv
import json
import subprocess
from datetime import datetime

from machineguard.config import FEATURE_ENGINEERING_DIR, PROJECT_ROOT

LOG_FILE = FEATURE_ENGINEERING_DIR / "experiment_log.csv"
COLUMNS = ["timestamp", "git_commit", "uncommitted_changes", "script", "arguments", "results"]


def record_experiment(script, arguments, results, log_file=LOG_FILE):
    """Append one row describing a finished run. arguments and results must be JSON-serialisable."""
    commit, uncommitted = current_git_state()
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "uncommitted_changes": uncommitted,
        "script": script,
        "arguments": json.dumps(arguments, sort_keys=True),
        "results": json.dumps(results, sort_keys=True),
    }
    log_file.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def count_lockbox_runs(log_file=LOG_FILE):
    """How many logged runs looked at the lock-box load (arguments contain "final": true)."""
    if not log_file.exists():
        return 0
    with log_file.open(newline="", encoding="utf-8") as file:
        return sum(json.loads(row["arguments"]).get("final") is True for row in csv.DictReader(file))


def current_git_state():
    """(short commit hash, whether there are uncommitted changes), or ("unknown", "unknown") outside git."""
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        uncommitted = bool(_git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"
    return commit, uncommitted


def _git(*arguments):
    result = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=10
    )
    return result.stdout.strip()
