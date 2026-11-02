import csv
import json

from machineguard.experiment_log import COLUMNS, count_lockbox_runs, current_git_state, record_experiment


def read_rows(log_file):
    with log_file.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_first_record_creates_the_file_with_a_header(tmp_path):
    log_file = tmp_path / "log.csv"

    record_experiment("ladder", {"features": ["rms"], "final": False}, {"mean_f1": 0.5}, log_file)

    rows = read_rows(log_file)
    assert list(rows[0].keys()) == COLUMNS
    assert rows[0]["script"] == "ladder"
    assert json.loads(rows[0]["results"]) == {"mean_f1": 0.5}


def test_later_records_are_appended_not_overwritten(tmp_path):
    log_file = tmp_path / "log.csv"

    record_experiment("ladder", {"final": False}, {}, log_file)
    record_experiment("screen", {"final": False}, {}, log_file)

    assert [row["script"] for row in read_rows(log_file)] == ["ladder", "screen"]


def test_lockbox_runs_are_counted_only_when_final_is_true(tmp_path):
    log_file = tmp_path / "log.csv"
    record_experiment("ladder", {"final": False}, {}, log_file)
    record_experiment("ladder", {"final": True}, {}, log_file)
    record_experiment("screen", {}, {}, log_file)

    assert count_lockbox_runs(log_file) == 1


def test_lockbox_count_is_zero_when_there_is_no_log_yet(tmp_path):
    assert count_lockbox_runs(tmp_path / "missing.csv") == 0


def test_git_state_never_raises():
    commit, uncommitted = current_git_state()

    assert isinstance(commit, str)
    assert uncommitted in (True, False, "unknown")


def test_summarise_gives_per_load_and_mean_scores():
    from collections import namedtuple

    from machineguard.models.evaluation import summarise

    result = namedtuple("Result", ["macro_f1", "confusion"])
    all_results = {0: {"model": result(0.5, None)}, 1: {"model": result(0.7, None)}}

    assert summarise(all_results) == {"model": {"per_load": {0: 0.5, 1: 0.7}, "mean": 0.6}}
