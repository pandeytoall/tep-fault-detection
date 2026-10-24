"""Check that faultNumber 4-7 in Faulty_Training behave like IDV(4)-(7).

Paired first-hour method on the 20 `dev` runs: faulty run k is compared with FaultFree_Training run k (same ID,
same noise); per variable, mean |faulty - normal| over samples 21-40 in std units of the normal run; top 3 per
run. The paired difference is a diagnostic here, never a model input. The expected tags are hypotheses from the
IDV descriptions (src/signatures.py: IDV_EXPECTED); they are not tuned to pass.
Writes results/phase-1/idv_check.csv. Usage: .venv/Scripts/python scripts/check_idv.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, prepare, signatures, split  # noqa: E402

NAME, NORMAL = "TEP_Faulty_Training", "TEP_FaultFree_Training"
OUT = ROOT / "results" / "phase-1" / "idv_check.csv"


def main():
    t0 = time.perf_counter()
    onset = data.onset_for(NAME)
    dev = split.ids("dev")
    frame = data.load_runs(NAME, dev, faults=list(signatures.IDV_EXPECTED))
    runs = {f: {r: g for r, g in sub.groupby("simulationRun")} for f, sub in frame.groupby("faultNumber")}
    assert all(len(r) == 20 for r in runs.values()), "expected 20 dev runs per fault"
    normal = data.load_runs(NORMAL, dev)
    normal_runs = {r: g for r, g in normal.groupby("simulationRun")}
    out = signatures.idv_check(runs, normal_runs, onset)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    for row in out.itertuples():
        print(f"fault {row.fault}: {row.verdict} (share {row.share_runs_in_top3:.2f}, top-1 {row.most_common_top1})")
    print(f"runtime {time.perf_counter() - t0:.1f} s, peak RAM {prepare.peak_ram_gb():.2f} GB")


if __name__ == "__main__":
    main()
