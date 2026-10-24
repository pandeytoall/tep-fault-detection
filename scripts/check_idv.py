"""Check that faultNumber 4-7 in Faulty_Training behave like IDV(4)-(7).

On the 20 `dev` runs of TEP_Faulty_Training, rank the standardised mean shifts (after onset vs samples 1-20)
and record how often the expected variable is in the run's top 3. The expectations are hypotheses from the
IDV descriptions (src/signatures.py: IDV_EXPECTED); they are not tuned to pass.
Writes results/phase-1/idv_check.csv. Usage: .venv/Scripts/python scripts/check_idv.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, prepare, signatures, split  # noqa: E402

NAME = "TEP_Faulty_Training"
OUT = ROOT / "results" / "phase-1" / "idv_check.csv"


def main():
    t0 = time.perf_counter()
    onset = data.onset_for(NAME)
    frame = data.load_runs(NAME, split.ids("dev"), faults=list(signatures.IDV_EXPECTED))
    runs = {f: [g for _, g in sub.groupby("simulationRun")] for f, sub in frame.groupby("faultNumber")}
    assert all(len(r) == 20 for r in runs.values()), "expected 20 dev runs per fault"
    out = signatures.idv_check(runs, onset)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    for row in out.itertuples():
        verdict = "PASS" if row.share_runs_in_top3 >= signatures.IDV_PASS_SHARE else "FAIL"
        print(f"fault {row.fault}: {verdict} (share {row.share_runs_in_top3:.2f}, top-1 {row.most_common_top1})")
    print(f"runtime {time.perf_counter() - t0:.1f} s, peak RAM {prepare.peak_ram_gb():.2f} GB")


if __name__ == "__main__":
    main()
