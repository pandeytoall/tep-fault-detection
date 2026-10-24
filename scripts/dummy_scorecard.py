"""End-to-end proof of src/evaluate.py with a dummy model (training family only, no testing files).

Model `dummy-maxz`: per-variable mean and std from the `fit` normal runs; score = max |z| over the variables
that have a non-zero std on those runs. Limits come from the `limit` normal runs at 1% and 5% FAR. The `dev`
runs (all 20 faults, plus the normal runs with the same IDs) are scored.
Writes results/phase-1/scorecard-dummy.csv (never results/scorecard.csv).
Usage: .venv/Scripts/python scripts/dummy_scorecard.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, evaluate, prepare, split  # noqa: E402

NORMAL, FAULTY = "TEP_FaultFree_Training", "TEP_Faulty_Training"
OUT = ROOT / "results" / "phase-1" / "scorecard-dummy.csv"
MODEL = "dummy-maxz"
FAR_TARGETS = {"1%": 0.01, "5%": 0.05}


def fit_maxz(frame):
    """Mean and std per variable; variables with zero std are dropped (listed in `dropped`)."""
    x = frame[data.VARIABLES].astype(np.float64)
    mean, std = x.mean(), x.std(ddof=0)
    keep = std > 0
    return {"mean": mean[keep], "std": std[keep], "variables": list(mean[keep].index),
            "dropped": list(mean[~keep].index)}


def score_maxz(model, frame):
    """Max |z| over the kept variables, one value per row."""
    z = (frame[model["variables"]].to_numpy(dtype=np.float64) - model["mean"].to_numpy()) / model["std"].to_numpy()
    return np.abs(z).max(axis=1)


def scored(model, name, runs, faults=None):
    frame = data.load_runs(name, runs, faults)
    out = frame[data.ID_COLUMNS].copy()
    out["score"] = score_maxz(model, frame)
    return out


def main():
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    onset = data.onset_for(FAULTY)

    t_fit = time.perf_counter()
    model = fit_maxz(data.load_runs(NORMAL, split.ids("fit")))
    train_time = time.perf_counter() - t_fit
    print(f"dropped zero-std variables: {model['dropped'] or 'none'}")

    limit_scores = scored(model, NORMAL, split.ids("limit"))["score"]
    dev_faulty = scored(model, FAULTY, split.ids("dev"))
    dev_normal = scored(model, NORMAL, split.ids("dev"))
    dev_faulty = dev_faulty[dev_faulty["faultNumber"] > 0]
    peak = prepare.peak_ram_gb()

    rows = []
    for limit_name, target in FAR_TARGETS.items():
        limit = evaluate.limit_for_far(limit_scores, target)
        rows += evaluate.model_rows(MODEL, "unsupervised", limit_name, limit, dev_faulty, dev_normal, onset,
                                    machine=machine, split_file="results/phase-1/run_split.json",
                                    train_time_s=train_time, peak_ram_gb=peak)
    evaluate.write_scorecard(rows, OUT)

    for limit_name in FAR_TARGETS:
        sel = [r for r in rows if r["limit_name"] == limit_name]
        get = lambda metric, fault=None: next(r["value"] for r in sel if r["metric"] == metric and r["fault"] == fault)  # noqa: E731
        fdrs = {f: get("fdr", f) for f in (3, 6, 9, 15)}
        print(f"limit {limit_name}: FAR {get('far'):.4f}, FDR "
              + ", ".join(f"f{f} {v:.3f}" for f, v in fdrs.items())
              + f", mean FDR {get('mean_fdr'):.3f}, excl 3/9/15 {get('mean_fdr_excl_3_9_15'):.3f}"
              + f", sustained rate {get('sustained_rate'):.2f}")
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
