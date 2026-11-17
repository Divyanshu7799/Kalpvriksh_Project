"""Plant one deliberate bug at a time and see whether the test suite notices."""
import subprocess, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()  # repo root; default: current folder
MUTANTS = [
    ("split: step uses overlap instead of 1-overlap", "src/machineguard/data/preprocessing.py", "step_size = int(window_size * (1 - overlap))", "step_size = int(window_size * overlap)"),
    ("split: last full window dropped (off by one)", "src/machineguard/data/preprocessing.py", "signal.size - window_size + 1, step_size", "signal.size - window_size, step_size"),
    ("resample: up and down swapped", "src/machineguard/data/preprocessing.py", "up = target_rate_hz // common_divisor\n    down = original_rate_hz // common_divisor", "up = original_rate_hz // common_divisor\n    down = target_rate_hz // common_divisor"),
    ("rms: sum instead of mean", "src/machineguard/features/time_domain.py", "np.sqrt(np.mean(windows**2, axis=1))", "np.sqrt(np.sum(windows**2, axis=1))"),
    ("mean_abs: median instead of mean", "src/machineguard/features/time_domain.py", "return np.mean(np.abs(windows), axis=1)", "return np.median(np.abs(windows), axis=1)"),
    ("clearance factor: wrong power", "src/machineguard/features/time_domain.py", "axis=1) ** 2)", "axis=1) ** 1)"),
    ("defect frequency: 2xBSF multiplier wrong", "src/machineguard/features/bearing_frequencies.py", '"2xBSF": 4.7134', '"2xBSF": 4.0'),
    ("spectrum: amplitude scale lost", "src/machineguard/signal/spectrum.py", "* 2.0 / window.sum()", "/ window.sum()"),
    ("envelope: DC not removed", "src/machineguard/signal/spectrum.py", "envelope - envelope.mean()", "envelope"),
    ("envelope peak: tolerance 5 bins instead of 1", "src/machineguard/features/frequency_domain.py", "PEAK_TOLERANCE_BINS = 1", "PEAK_TOLERANCE_BINS = 5"),
    ("screening: pairs sorted weakest first", "src/machineguard/features/screening.py", "key=lambda pair: -abs(pair[2])", "key=lambda pair: abs(pair[2])"),
    ("selection: fold spread uses ddof=0", "src/machineguard/models/selection.py", "np.std(scored[best], ddof=1)", "np.std(scored[best], ddof=0)"),
    ("selection: first round starts from 0, not chance", "src/machineguard/models/selection.py", "previous_mean = CHANCE_MACRO_F1\n", "previous_mean = 0.0\n"),
    ("selection: stopping rule ignores the spread", "src/machineguard/models/selection.py", "accepted = gain > fold_std", "accepted = gain > 0"),
    ("LEAKAGE: training data includes the held-out load", "src/machineguard/models/evaluation.py", 'train = window_table[window_table["load_hp"] != held_out_load]', "train = window_table"),
    ("labels: wrong rpm for load 2", "src/machineguard/data/make_labels.py", "RPM_BY_LOAD = [1797, 1772, 1750, 1730]", "RPM_BY_LOAD = [1797, 1772, 1751, 1730]"),
    ("cwru: normal files treated as 12 kHz", "src/machineguard/data/cwru_mat.py", "NORMAL_BASELINE_RATE_HZ = 48000", "NORMAL_BASELINE_RATE_HZ = 12000"),
    ("missed faults: also counts correct Normal", "src/machineguard/models/evaluation.py", "for row in range(len(CLASS_ORDER)) if row != normal", "for row in range(len(CLASS_ORDER))"),
    ("register: duplicates treated as kept", "src/machineguard/features/decision_register.py", 'if screened["decision"] != "keep"', 'if screened["decision"] == "keep"'),
    ("log: lock-box count ignores 'final' flag", "src/machineguard/experiment_log.py", '.get("final") is True', '.get("final") is not False'),
]
survived, killed = [], []
for name, rel, old, new in MUTANTS:
    path = ROOT / rel
    with path.open(newline="", encoding="utf-8") as handle:  # newline="" keeps Windows line endings as they are
        original = handle.read()
    eol = "\r\n" if "\r\n" in original else "\n"
    old, new = old.replace("\n", eol), new.replace("\n", eol)
    assert original.count(old) == 1, (name, original.count(old))
    path.write_text(original.replace(old, new), newline="", encoding="utf-8")
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    finally:
        path.write_text(original, newline="", encoding="utf-8")
    last = [l for l in result.stdout.splitlines() if l.startswith("FAILED") or " failed" in l or "error" in l.lower()]
    first_failure = next((l for l in result.stdout.splitlines() if l.startswith("FAILED")), "")
    status = "caught" if result.returncode != 0 else "SURVIVED"
    (killed if status == "caught" else survived).append(name)
    print(f"{status:9s} {name}   {first_failure[7:90]}")
print(f"\ncaught {len(killed)} of {len(MUTANTS)}; survived {len(survived)}")