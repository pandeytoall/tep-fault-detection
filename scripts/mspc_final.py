"""Phase 2 final scoring: the testing files, once. Nothing is fitted or chosen here.

Loads results/phase-2/{pca,dpca}.npz and the limits from results/phase-2/dev_scores.csv (set by mspc_dev.py on
the training family), scores all 500 FaultFree_Testing runs and all 500 runs x 20 faults of Faulty_Testing (one fault
at a time), and writes the 4 models x 3 limits to results/scorecard.csv. Also writes results/phase-2/fdr_by_fault.csv,
vs_literature.csv and figures/fdr-by-fault.png. Exits with code 3 if a stop condition (leak bound, FAR range) fails.
Usage: .venv/Scripts/python scripts/mspc_final.py
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, evaluate, mspc, prepare, style  # noqa: E402

NORMAL, FAULTY = "TEP_FaultFree_Testing", "TEP_Faulty_Testing"
OUT = ROOT / "results" / "phase-2"
SPLIT_FILE = "results/phase-1/run_split.json"
KINDS = ("pca", "dpca")
MODELS = [f"{k}-{s}" for k in KINDS for s in mspc.STATS]

# Rato & Reis (2013), Table 2: FDR at a 1% FAR on the original Braatz data (research/pca-anchors.md, Anchor C).
# Columns: PCA T2, PCA Q, DPCA T2, DPCA Q. Their DPCA uses 3 lags and 29 PCs, their PCA 17 PCs (parallel analysis).
RATO_REIS = {
    1: (0.991, 0.995, 0.990, 0.994), 2: (0.985, 0.984, 0.984, 0.981), 3: (0.036, 0.006, 0.035, 0.010),
    4: (0.218, 0.980, 0.165, 0.999), 5: (0.257, 0.217, 0.293, 0.228), 6: (0.989, 0.999, 0.989, 0.999),
    7: (0.999, 0.999, 0.986, 0.999), 8: (0.974, 0.968, 0.973, 0.974), 9: (0.034, 0.010, 0.030, 0.002),
    10: (0.367, 0.154, 0.439, 0.172), 11: (0.414, 0.638, 0.340, 0.829), 12: (0.985, 0.925, 0.990, 0.964),
    13: (0.943, 0.950, 0.943, 0.950), 14: (0.988, 0.999, 0.990, 0.999), 15: (0.035, 0.007, 0.059, 0.009),
    16: (0.174, 0.137, 0.217, 0.145), 17: (0.787, 0.905, 0.790, 0.953), 18: (0.893, 0.901, 0.890, 0.898),
    19: (0.115, 0.059, 0.046, 0.298), 20: (0.340, 0.423, 0.408, 0.493),
}
LITERATURE_COLUMNS = ["pca-t2", "pca-spe", "dpca-t2", "dpca-spe"]  # order of the tuples above


def saved_limits():
    """({kind: {stat: {limit name: value}}}, {kind: train time s}, dev peak RAM GB).

    Limits come from the models' .npz files (written by mspc_dev.py); train time and RAM from dev_scores.csv.
    """
    dev = pd.read_csv(OUT / "dev_scores.csv")
    limits = {k: mspc.PCAMonitor.load(OUT / f"{k}.npz").limits for k in KINDS}
    assert all(v is not None for v in limits.values()), "no limits in the .npz; rerun scripts/mspc_dev.py"
    train = {k: float(dev[(dev.model == f"{k}-t2") & (dev.metric == "train_time_s")]["value"].iloc[0]) for k in KINDS}
    return limits, train, float(dev[dev.metric == "peak_ram_gb"]["value"].max())


def plot_fdr_by_fault(fdr1, path):
    """Dot plot, one row per fault, PCA T2 and PCA SPE at the 1% limit."""
    t2, spe = fdr1["pca-t2"], fdr1["pca-spe"]
    title = (f"PCA SPE catches fault 4 ({spe[4]:.2f}) where T² does not ({t2[4]:.2f});\n"
             f"faults 3, 9 and 15 stay unseen by both (1% FAR, test runs)")
    colours = {"pca-spe": style.OKABE_ITO["vermillion"], "pca-t2": style.OKABE_ITO["blue"]}
    fig, ax = style.new_figure(figsize=(8, 9), font_size=14)
    faults = list(fdr1.index)
    for i, f in enumerate(faults):
        ax.plot([t2[f], spe[f]], [i, i], color="0.75", lw=2, zorder=1)
    ax.scatter(t2.values, range(len(faults)), color=colours["pca-t2"], s=70, zorder=2)
    ax.scatter(spe.values, range(len(faults)), color=colours["pca-spe"], s=70, zorder=2)
    ax.set_yticks(range(len(faults)), [f"Fault {f}" for f in faults])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Fault detection rate (share of post-onset samples flagged)")
    ax.text(1.03, 0, "SPE", color=colours["pca-spe"], fontweight="bold", va="center")
    ax.text(1.03, 1, "T²", color=colours["pca-t2"], fontweight="bold", va="center")
    style.finish_figure(fig, ax, title, path, title_size=14)


def main():
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    limits, train_time, dev_peak = saved_limits()
    monitors = {k: mspc.PCAMonitor.load(OUT / f"{k}.npz") for k in KINDS}
    skip = monitors["dpca"].lags
    onset = data.onset_for(FAULTY)
    print(f"L = {skip}; PCs: " + ", ".join(f"{k} {m.n_pc}" for k, m in monitors.items()))

    normal = {k: m.score(data.load_runs(NORMAL, range(1, 501))) for k, m in monitors.items()}
    assert normal["pca"]["simulationRun"].nunique() == 500
    faulty = {k: [] for k in KINDS}
    for fault in range(1, 21):
        frame = data.load_runs(FAULTY, range(1, 501), faults=[fault])
        assert frame["simulationRun"].nunique() == 500 and (frame["faultNumber"] == fault).all()
        for k, m in monitors.items():
            faulty[k].append(m.score(frame))
        del frame
    faulty = {k: pd.concat(v, ignore_index=True) for k, v in faulty.items()}
    scoring_peak = prepare.peak_ram_gb()

    peak = max(dev_peak, scoring_peak)
    rows = []
    for k in KINDS:
        rows += mspc.scorecard_rows(k, faulty[k], normal[k], limits[k], onset, skip, after=onset, machine=machine,
                                    split_file=SPLIT_FILE, train_time_s=train_time[k], peak_ram_gb=peak)
    evaluate.write_scorecard(rows, evaluate.DEFAULT_SCORECARD)
    df = pd.DataFrame(rows)

    # --- derived files -------------------------------------------------------------------
    one = df[df.limit_name == "1%"]
    fdr1 = one[one.metric == "fdr"].pivot(index="fault", columns="model", values="value")[MODELS]
    fdr1.index = fdr1.index.astype(int)
    far1 = one[one.metric == "far"].set_index("model")["value"]
    wide = fdr1.reset_index().astype({"fault": str})
    wide.loc[len(wide)] = ["far"] + [far1[m] for m in MODELS]
    wide.to_csv(OUT / "fdr_by_fault.csv", index=False)
    lit = pd.DataFrame.from_dict(RATO_REIS, orient="index", columns=[f"rr-{c}" for c in LITERATURE_COLUMNS])
    lit.index.name = "fault"
    lit = pd.concat([fdr1[LITERATURE_COLUMNS].add_prefix("ours-"), lit], axis=1)
    for c in LITERATURE_COLUMNS:
        lit[f"diff-{c}"] = lit[f"ours-{c}"] - lit[f"rr-{c}"]
    lit.round(4).to_csv(OUT / "vs_literature.csv")

    # --- checks and print ------------------------------------------------------------------
    print("\nTest table at the 1% limit")
    summary = one[one.metric.isin(["far", "sustained_rate", "mean_fdr", "mean_fdr_excl_3_9_15"])]
    print(summary.pivot(index="metric", columns="model", values="value")[MODELS].round(4).to_string())
    shown = fdr1.loc[[1, 2, 4, 6, 7, 3, 9, 15]]
    print(shown.round(3).to_string())
    delay = one[one.metric.isin(["delay_median_min", "never_detected_share"]) & one.fault.isin([1, 4])]
    print(delay.pivot(index=["metric", "fault"], columns="model", values="value")[MODELS].round(3).to_string())
    for limit_name in ("1%", "5%", mspc.TEXTBOOK):
        sel = df[df.limit_name == limit_name]
        far = sel[sel.metric == "far"].set_index("model")["value"]
        print(f"test FAR at {limit_name}: " + ", ".join(f"{m} {far[m]:.4f}" for m in MODELS))
    big = lit[[c for c in lit.columns if c.startswith("diff-")]]
    print("\nFaults where ours differs from Rato & Reis by more than 0.1 (1% limit):")
    for c in LITERATURE_COLUMNS:
        d = big[f"diff-{c}"]
        print(f"  {c}: " + (", ".join(f"f{f} {lit.loc[f, f'ours-{c}']:.2f} vs {lit.loc[f, f'rr-{c}']:.2f}"
                                     for f in d.index[d.abs() > 0.1]) or "none"))
    print(f"train time s: {train_time}; peak RAM GB: dev {dev_peak:.2f}, final scoring {scoring_peak:.2f}")

    failures = mspc.final_check_failures(df, MODELS)
    for f in failures:
        print("CHECK FAILED:", f)
    if not failures:  # the chart title states the findings the checks confirm
        plot_fdr_by_fault(fdr1, OUT / "figures" / "fdr-by-fault.png")
    print(prepare.runtime_line(t0))
    if failures:
        sys.exit(3)


if __name__ == "__main__":
    main()
