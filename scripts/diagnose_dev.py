"""Phase 4 development: per-fault blame and the before/after study on the `dev` runs (training family; no testing file is read).

Steps: (1) the RBC references (mean RBC per reading on the 330 `fit` normal runs; the stored SPE and T2 references in pca.npz are
checked against the same runs); (2) permutation importance of the saved hgb-l2 on the dev faulty runs' post-onset samples (pooled,
thinned to every PERM_STRIDE-th sample for the 3 h budget; one vector per fault); (3) per fault: relative blame of SPE, T2, RBC-SPE and
RBC-T2 over the post-onset window and the first hour, the four blame metrics against the expected scoring unit, the top readings,
and the within-run before/after statistics, next to a fault-free control; (4) the dev sanity items. Nothing is fitted or chosen.
Writes results/phase-4/{rbc_reference.npz, dev_blame.csv, dev_blame_by_group.csv, dev_hidden.csv, dev_perm_importance.csv,
top_readings_<fault>.csv}. Run scripts/diagnose_final.py afterwards.
Usage: .venv/Scripts/python scripts/diagnose_dev.py [--perm-stride N]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")  # HistGradientBoosting ignores n_jobs; cap its threads

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import contributions, data, diagnosis, mspc, models, prepare, split  # noqa: E402

NORMAL, FAULTY = "TEP_FaultFree_Training", "TEP_Faulty_Training"
OUT = data.RESULTS_4
PERM_STRIDE = 5          # every 5th post-onset sample of the dev runs: 192k pooled rows take about 5 h of predictions, 38k about 1 h
PERM_REPEATS, SEED = 5, 42
A_FEED = ("xmeas_1", "xmv_3")
REACTOR_COOLING = ("xmeas_21", "xmv_10")


def permutation_vectors(hgb, dev_faulty, onset, stride):
    """({fault: importance per reading}, number of pooled rows): permutation importance of hgb-l2 on the dev faulty runs."""
    ids, x = hgb._rows(dev_faulty)
    keep = (ids["sample"].to_numpy() > onset) & (ids["sample"].to_numpy() % stride == 0)
    x, y = x[keep], ids["faultNumber"].to_numpy()[keep]
    print(f"permutation importance: {len(y)} pooled post-onset rows, {x.shape[1]} inputs x {PERM_REPEATS} repeats", flush=True)
    t = time.perf_counter()
    vectors = diagnosis.permutation_blame(hgb.estimator, x, y, sorted(set(y.tolist())), n_lags=hgb.lags + 1,
                                          n_repeats=PERM_REPEATS, seed=SEED, n_jobs=1)
    print(f"permutation importance done in {time.perf_counter() - t:.0f} s", flush=True)
    return vectors, len(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perm-stride", type=int, default=PERM_STRIDE)
    args = parser.parse_args()
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    groups = split.load_split()
    ids = lambda g: split.ids(g, groups)  # noqa: E731
    onset = data.onset_for(FAULTY)
    OUT.mkdir(parents=True, exist_ok=True)
    monitor = mspc.PCAMonitor.load(data.PCA_MODEL)
    assert monitor.lags == 0 and monitor.n_pc == 31, (monitor.lags, monitor.n_pc)
    tags = monitor.variables

    # --- references: RBC from the `fit` runs; SPE and T2 as stored in pca.npz ----------------------------------------------
    fit_runs = data.load_runs(NORMAL, ids("fit"))
    assert (fit_runs["faultNumber"] == 0).all() and fit_runs["simulationRun"].nunique() == 330
    fit_refs = diagnosis.normal_references(monitor, fit_runs)
    del fit_runs
    stored = contributions.load_reference(data.PCA_MODEL)
    gap = max(float(np.max(np.abs(fit_refs[m] / stored[m] - 1))) for m in ("spe", "t2"))
    print(f"stored SPE and T2 references match the `fit` runs (largest relative gap {gap:.2e})")
    assert gap < 1e-6, "pca.npz reference was not computed on these fit runs"
    diagnosis.save_rbc_reference(OUT / "rbc_reference.npz", fit_refs, machine, 330)
    refs = diagnosis.load_references(data.PCA_MODEL, OUT / "rbc_reference.npz")

    dev_faulty, dev_normal = data.load_runs(FAULTY, ids("dev")), data.load_runs(NORMAL, ids("dev"))
    assert sorted(dev_faulty["faultNumber"].unique()) == list(range(1, 21)) and dev_normal["simulationRun"].nunique() == 20

    # --- permutation importance (the supervised view) -----------------------------------------------------------------------
    hgb = models.load(data.MODELS_DIR / "hgb-l2.joblib")
    perm, n_pooled = permutation_vectors(hgb, dev_faulty, onset, args.perm_stride)
    perm_table = pd.DataFrame(perm, index=tags).T.rename_axis("fault").reset_index()
    files = {"dev_perm_importance.csv": perm_table.assign(pooled_rows=n_pooled, stride=args.perm_stride)}  # written at the end

    # --- per fault: blame, top readings, before/after ----------------------------------------------------------------------
    blame, change, medians, vectors = [], {}, {}, {}
    for fault in range(1, 21):
        frame = dev_faulty[dev_faulty["faultNumber"] == fault]
        result = diagnosis.analyse_fault(monitor, refs, frame, fault, onset)
        assert len(result["runs"]) == 20
        blame += result["blame"]
        # the supervised view has one vector per fault: it counts as one "run" in the post-onset window
        blame += diagnosis.blame_rows(fault, {"post": {diagnosis.PERM: perm[fault][None, :]}}, tags)
        change[fault], vectors[fault] = result["change"], result["vectors"]
        medians[fault] = {m: np.median(result["vectors"]["post"][m], axis=0) for m in diagnosis.PCA_METHODS}
        files[f"top_readings_{fault}.csv"] = diagnosis.top_readings(fault, medians[fault], tags, extra={diagnosis.PERM: perm[fault]})
    # the before/after control: fault-free runs with IDs of the `limit` group, so it shares no noise stream with the faulty dev runs
    control_ids = diagnosis.dev_control_ids(ids("limit"))
    control = diagnosis.analyse_fault(monitor, refs, data.load_runs(NORMAL, control_ids), 0, onset)
    same_ids = diagnosis.analyse_fault(monitor, refs, dev_normal, 0, onset)   # fault-free dev runs: the relative-contribution check
    blame = pd.DataFrame(blame)
    post = blame[blame["window"] == "post"]
    summary = diagnosis.group_summary(post)
    hidden = diagnosis.hidden_table(change, control["change"])
    files.update({"dev_blame.csv": blame, "dev_blame_by_group.csv": summary, "dev_hidden.csv": hidden})

    # --- print: blame -----------------------------------------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print("\nBlame by fault group, post-onset window (dev, 20 runs per fault; baseline = a method that picks readings at random)")
    print(summary.round(3).to_string(index=False))
    print("\nBlame per fault (post-onset window), methods side by side")
    baselines = post.drop_duplicates("fault").set_index("fault")
    for metric, base in (("top1_hit", "baseline_unit_share"), ("top5_hit", "baseline_top5_hit"),
                         ("unit_mass", "baseline_unit_share"), ("smear", None)):
        wide = post.pivot(index="fault", columns="method", values=metric)[list(diagnosis.METHODS)]
        if base:
            wide["random"] = baselines[base]
        print(f"-- {metric}")
        print(wide.round(2).to_string())
    print("\nTop 3 SPE readings per fault (median over runs)")
    for fault in range(1, 21):
        note = diagnosis.fault_note(fault)
        top3 = post[(post.fault == fault) & (post.method == "spe")]["top3"].iloc[0]
        print(f"  fault {fault:2d}: {top3}" + (f"   [{note}]" if note else ""))

    # --- print: before/after ------------------------------------------------------------------------------------------------------
    cols = ["fault", "group", "median_max_abs_shift", "median_n_shift_gt_1sd", "median_max_spread_ratio", "median_max_abs_dacf",
            "effect_vs_control", "verdict"]
    print("\nWithin-run before/after (dev: 20 samples before, 480 after the onset); fault 0 = fault-free control")
    print(hidden[cols].round(2).to_string(index=False))

    # --- the dev sanity items (PASS or FAIL on the post-onset window, the headline; the first hour is shown beside it) ---------------
    failed = []

    def check(ok, text):
        print(("PASS  " if ok else "FAIL  ") + text)
        if not ok:
            failed.append(text)

    def share(fault, wanted, k, window):
        return diagnosis.top_share(vectors[fault][window]["spe"], tags, wanted, k=k)

    f6, f6h = share(6, A_FEED, 1, "post"), share(6, A_FEED, 1, "first_hour")
    check(f6 >= 0.8, f"fault 6: an A-feed reading is the top SPE reading in {f6:.0%} of runs, post-onset window (need at least 80%); "
                     f"first hour {f6h:.0%}")
    f4, f4h = share(4, REACTOR_COOLING, 5, "post"), share(4, REACTOR_COOLING, 5, "first_hour")
    check(f4 >= 0.5, f"fault 4: a reactor cooling reading is in the top 5 SPE readings in {f4:.0%} of runs, post-onset window "
                     f"(need at least 50%); first hour {f4h:.0%}")
    spe_rows = blame[blame["method"] == "spe"].set_index(["fault", "window"])
    for fault in (1, 2, 7):
        row, first = spe_rows.loc[(fault, "post")], spe_rows.loc[(fault, "first_hour")]
        check(row["top5_hit"] > row["baseline_top5_hit"],
              f"fault {fault}: SPE top5_hit {row['top5_hit']:.2f} above the random baseline {row['baseline_top5_hit']:.2f}, "
              f"post-onset window; first hour {first['top5_hit']:.2f}")
    for method in diagnosis.PCA_METHODS:
        median = float(np.median(same_ids["vectors"]["post"][method].mean(axis=0)))
        check(0.5 <= median <= 2.0, f"fault-free dev runs, {method}: median relative contribution over readings {median:.2f} (0.5 to 2)")
    stops = diagnosis.hidden_stop_messages(hidden)
    paired = float(same_ids["change"]["max_abs_shift"].median())
    for text in stops:
        failed.append(text)
    check(not stops, "faults 3, 9, 15: largest median |shift| at most 3x the fault-free control's (control "
                     f"{float(hidden.loc[hidden.fault == 0, 'median_max_abs_shift'].iloc[0]):.2f} sd on {len(control_ids)} unpaired runs; "
                     f"{paired:.2f} sd on the 20 dev IDs, which share noise streams with the faulty dev runs)" + (f" {stops}" if stops else ""))
    print("every sanity item passed" if not failed else f"{len(failed)} sanity item(s) failed")
    print(prepare.runtime_line(t0))
    info = {"machine": machine, "runtime_s": round(time.perf_counter() - t0, 1), "peak_ram_gb": round(prepare.peak_ram_gb(), 2)}
    for name, table in files.items():
        table.assign(**info).to_csv(OUT / name, index=False)
    if failed:
        sys.exit(3)


if __name__ == "__main__":
    main()
