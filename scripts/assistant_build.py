"""Phase 2b build: both case sets, their historians (readings + alarm log) and answer keys, and the PCA normal reference.

Steps: (1) add the normal reference to results/phase-2/pca.npz if it has none (mean contribution per reading over the
`fit` normal runs); (2) per set, pick the cases (src/cases.py), load and crop their runs, write
data/processed/historian_{set}.sqlite and results/phase-2b/answer_key_{set}.csv; (3) print and save the sanity checks.
The test set is FROZEN: if its key or historian exists the script stops before doing anything. --dev-only rebuilds dev.
--rebuild-unused-test is a one-off: it deletes and rebuilds the test set, but refuses once answers_test.jsonl exists.
Usage: .venv/Scripts/python scripts/assistant_build.py [--dev-only | --rebuild-unused-test]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import cases, contributions, data, historian, mspc, prepare, split  # noqa: E402

MODEL = ROOT / "results" / "phase-2" / "pca.npz"
OUT = ROOT / "results" / "phase-2b"
NORMAL_TRAINING = "TEP_FaultFree_Training"


def paths(set_name):
    return OUT / f"answer_key_{set_name}.csv", data.PROCESSED / f"historian_{set_name}.sqlite"


def ensure_reference(model_path):
    """Add the normal reference to the model file if it has none. Returns True if it was computed now."""
    if contributions.load_reference(model_path) is not None:
        return False
    monitor = mspc.PCAMonitor.load(model_path)
    fit = data.load_runs(NORMAL_TRAINING, split.ids("fit", split.load_split()))
    assert (fit["faultNumber"] == 0).all() and fit["simulationRun"].nunique() == 330
    contributions.add_reference(model_path, contributions.normal_reference(monitor, fit))
    return True


def build_set(set_name, monitor, machine):
    key_path, db_path = paths(set_name)
    if set_name == "test":
        key = cases.build_test_key()
    else:
        key = cases.build_dev_key(split.ids("dev", split.load_split()))
    frames = cases.load_case_frames(key)
    tmp = db_path.with_name(db_path.name + ".tmp")
    historian.write_historian(tmp, frames, monitor)
    os.replace(tmp, db_path)
    cases.write_answer_key(key, key_path, machine)
    checks = cases.sanity_checks(key, db_path, MODEL)
    (OUT / f"build_checks_{set_name}.json").write_text(json.dumps({"machine": machine, "set": set_name, **checks},
                                                                   indent=1))
    return key, checks, db_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-only", action="store_true", help="rebuild the dev set only (the test set is frozen)")
    parser.add_argument("--rebuild-unused-test", action="store_true",
                        help="one-off: rebuild the test set too; refused once any test session is recorded")
    args = parser.parse_args()
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    sets = ["dev"] if args.dev_only else ["dev", "test"]
    if args.rebuild_unused_test:
        key_path, db_path = paths("test")
        cases.unfreeze_unused_test(key_path, db_path, OUT / "answers_test.jsonl")
    if "test" in sets:
        cases.guard_frozen("test", paths("test"))  # stop before touching anything
    OUT.mkdir(parents=True, exist_ok=True)
    print("normal reference computed now" if ensure_reference(MODEL) else "normal reference already in pca.npz")
    monitor = mspc.PCAMonitor.load(MODEL)
    assert monitor.limits is not None, "no limits in pca.npz; run scripts/mspc_dev.py"
    for set_name in sets:
        t_set = time.perf_counter()
        key, checks, db_path = build_set(set_name, monitor, machine)
        share = checks["normal_share_above"]
        print(f"\n{set_name}: {len(key)} cases, {db_path.stat().st_size / 1e6:.0f} MB, "
              f"{time.perf_counter() - t_set:.0f} s")
        print(f"  normal cases ({share['n_cases']}): share of samples above the 1% limits: "
              f"T2 {share['t2']:.4f}, SPE {share['spe']:.4f}  (expect 0.003-0.03)")
        print(f"  fault 6 cases: " + (", ".join(f"{c}: {'A feed in top 5' if v['a_feed_in_top5'] else 'NO A feed'} "
                                               f"{v['top5']}" for c, v in checks["fault6"].items()) or "none"))
        print(f"  onset inside the set's range: {checks['onset_in_range']}")
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
