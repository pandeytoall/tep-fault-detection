"""Score the assistant's answers: writes results/phase-2b/diagnosis_{set}[_v1.1].csv and prints the summary.

With --version v1.1 --set test it also scores v1 (answers_test.jsonl), prints the v1 vs v1.1 comparison and the split by
use of classify_fault, and writes results/phase-3/assistant_v1_vs_v1.1.csv.
--classify-check --set dev prints the hgb-l2 class accuracy over the cases (classify_fault, default window); it needs no answers.
Usage: .venv/Scripts/python scripts/assistant_score.py --set dev|test [--version v1|v1.1]
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

MODEL = data.PCA_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="case_set", required=True, choices=["dev", "test"])
    ap.add_argument("--version", default="v1", choices=list(score.VERSIONS))
    ap.add_argument("--classify-check", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    key = score.load_key(args.case_set)
    if args.classify_check:
        tops = score.compute_classify_tops(key, data.PROCESSED / f"historian_{args.case_set}.sqlite")
        acc = score.classify_accuracy(key, tops)
        for name in ("top1", "top3", "normal"):
            print(f"hgb-l2 {name} on {args.case_set} cases: {acc[name]['k']} / {acc[name]['n']}")
        truth = key.set_index("case_id")["fault"]
        print("per case (true fault: top-3): " + "; ".join(f"{c} ({truth[c]}): {', '.join(t)}" for c, t in tops.items()))
        print("faults 3/9/15 as labelled: " + (str(acc["hard"]) or "none in this set"))
        print(prepare.runtime_line(t0))
        return
    baseline = score.compute_baseline(key, data.PROCESSED / f"historian_{args.case_set}.sqlite", MODEL)
    rows, summary = score.score(score.load_answers(score.answers_path(args.case_set, args.version)), key, baseline)
    rows.assign(machine=machine).to_csv(score.diagnosis_path(args.case_set, args.version), index=False)
    print(score.format_summary(summary))
    if args.version == "v1.1":
        split = score.classify_split(rows)
        for name, s in split.items():
            print(f"classify_fault {name.replace('_', ' ')}: {s['sessions']} sessions; headline {s['headline']['k']} / "
                  f"{s['headline']['n']}, IDV top-1 {s['idv_top1_all']['k']} / {s['idv_top1_all']['n']}, "
                  f"top-3 {s['idv_top3_all']['k']} / {s['idv_top3_all']['n']}")
        if args.case_set == "test":
            v1 = score.score(score.load_answers(score.answers_path("test", "v1")), key, baseline)
            table = score.compare(v1, (rows, summary))
            data.RESULTS_3.mkdir(parents=True, exist_ok=True)
            table.assign(machine=machine).to_csv(data.RESULTS_3 / "assistant_v1_vs_v1.1.csv", index=False)
            print(score.format_comparison(table))
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
