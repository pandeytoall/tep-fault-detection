"""Score the assistant's answers: writes results/phase-2b/diagnosis_{set}.csv and prints the summary.

Usage: .venv/Scripts/python scripts/assistant_score.py --set dev|test
"""
import argparse
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, prepare  # noqa: E402
from src.assistant import score  # noqa: E402

MODEL = ROOT / "results" / "phase-2" / "pca.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="case_set", required=True, choices=["dev", "test"])
    args = ap.parse_args()
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    key = score.load_key(args.case_set)
    answers = score.load_answers(score.RESULTS / f"answers_{args.case_set}.jsonl")
    baseline = score.compute_baseline(key, data.PROCESSED / f"historian_{args.case_set}.sqlite", MODEL)
    rows, summary = score.score(answers, key, baseline)
    rows.assign(machine=machine).to_csv(score.RESULTS / f"diagnosis_{args.case_set}.csv", index=False)
    print(score.format_summary(summary))
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
