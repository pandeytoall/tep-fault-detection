"""Phase 2 development: PCA and DPCA monitors on the training family only (no testing file is read).

Steps: choose L with Ku et al. (1995) on the `fit` normal runs (90% rule decides; parallel analysis is shown) and
stop after lag_choice.csv if the rules disagree, L = 0 or nothing stops by l = 5; fit PCA (no lags) and DPCA (L lags)
on `fit`; set 1%, 5% and textbook 99% limits (limits from the `limit` runs); score the `dev` runs (20 faults, plus the
normal runs with the same IDs). Run scripts/mspc_final.py afterwards.
Writes results/phase-2/{lag_choice.csv, dev_scores.csv, pca.npz, dpca.npz, figures/pc-sweep-dev.png}.
Usage: .venv/Scripts/python scripts/mspc_dev.py
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, evaluate, mspc, prepare, split, style  # noqa: E402

NORMAL, FAULTY = "TEP_FaultFree_Training", "TEP_Faulty_Training"
OUT = ROOT / "results" / "phase-2"
SPLIT_FILE = "results/phase-1/run_split.json"
SWEEP = range(5, 31)
SHOWN_FAULTS = (1, 2, 4, 6, 7, 3, 9, 15)
# Explicit override of the stop in mspc.lag_problem (spec Comments, 2026-10-25): the user accepts the 90% rule's L
# although parallel analysis disagreed. It only applies while the 90% rule still gives exactly this L.
LAG_OVERRIDE = {"chosen_L": 2, "note": "user decision 2026-10-25: 90% rule; parallel analysis disagreed"}


def alarm_limits(monitor, limit_runs, skip):
    """{stat: {limit name: value}}: 1% and 5% FAR from the `limit` normal runs (samples after `skip`), plus textbook."""
    scores = monitor.score(limit_runs)
    scores = scores[scores["sample"] > skip]
    textbook = monitor.textbook_limits(0.99)
    return {stat: {**{name: evaluate.limit_for_far(scores[stat], far) for name, far in mspc.FAR_LIMITS.items()},
                   mspc.TEXTBOOK: textbook[stat]} for stat in mspc.STATS}


def pc_sweep(pca, limit_runs, dev_faulty, onset):
    """Dev mean FDR at the 1% limit for PCA with 5-30 PCs. Side chart only; it never picks the PC count."""
    rows = []
    for k in SWEEP:
        monitor = pca.with_n_pc(k)
        limit_scores = monitor.score(limit_runs)
        dev_scores = monitor.score(dev_faulty)
        for stat in mspc.STATS:
            limit = evaluate.limit_for_far(limit_scores[stat], 0.01)
            f = dev_scores[data.ID_COLUMNS].assign(score=dev_scores[stat].to_numpy())
            rows.append({"n_pc": k, "stat": stat, "mean_fdr": evaluate.fdr(f, limit, onset).mean()})
    return pd.DataFrame(rows)


def plot_sweep(sweep, chosen, path):
    colours = {"spe": style.OKABE_ITO["vermillion"], "t2": style.OKABE_ITO["blue"]}
    names = {"spe": "PCA SPE", "t2": "PCA T²"}
    spe = sweep[sweep.stat == "spe"].set_index("n_pc")["mean_fdr"]
    t2 = sweep[sweep.stat == "t2"].set_index("n_pc")["mean_fdr"]
    assert (spe > t2).all(), "title claims SPE beats T2 at every PC count"
    title = (f"PCA SPE beats T² at every PC count from 5 to 30\n"
             f"(dev mean FDR at 1% FAR; the 90% rule picks {chosen} PCs)")
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for stat, series in (("spe", spe), ("t2", t2)):
        ax.plot(series.index, series.values, color=colours[stat], lw=2.2)
        ax.text(series.index[-1] + 0.4, series.values[-1], names[stat], color=colours[stat], va="center",
                fontweight="bold")
    ax.axvline(chosen, color="0.35", ls="--", lw=1.4)
    ax.text(chosen + 0.4, ax.get_ylim()[0] + 0.02, f"90% rule: {chosen} PCs", color="0.25", va="bottom")
    ax.set_xlim(5, 36)
    ax.set_xlabel("Retained principal components")
    ax.set_ylabel("Mean FDR over 20 faults (dev runs)")
    ax.set_title(title, fontsize=15, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    groups = split.load_split()
    ids = lambda g: split.ids(g, groups)  # noqa: E731
    onset = data.onset_for(FAULTY)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- lag choice: 90% rule decides, parallel analysis is a diagnostic -----------
    fit_runs = data.load_runs(NORMAL, ids("fit"))
    lag90, table90 = mspc.ku_lag(fit_runs, max_lag=5, rule="cpv", cpv=0.90)
    lag_pa, table_pa = mspc.ku_lag(fit_runs, max_lag=5, rule="parallel")
    problem = mspc.lag_problem(lag90, lag_pa)
    override = problem is not None and lag90 == LAG_OVERRIDE["chosen_L"]
    lag_table = pd.concat([table90, table_pa], ignore_index=True)
    if override:
        lag_table = lag_table.assign(**LAG_OVERRIDE)
    lag_table.to_csv(OUT / "lag_choice.csv", index=False)
    print(lag_table.to_string(index=False))
    print(f"L from the 90% rule: {lag90}; L from parallel analysis: {lag_pa}")
    if override:
        print(f"OVERRIDE of '{problem}': {LAG_OVERRIDE['note']}")
    elif problem:
        print(f"STOP after lag_choice.csv, nothing fitted: {problem}")
        print(prepare.runtime_line(t0))
        sys.exit(2)
    skip = lag90

    # --- fit -----------------------------------------------------------------------
    monitors, train_time = {}, {}
    for kind, lags in (("pca", 0), ("dpca", skip)):
        t_fit = time.perf_counter()
        monitors[kind] = mspc.PCAMonitor(lags=lags, cpv=0.90).fit(fit_runs)
        train_time[kind] = time.perf_counter() - t_fit
        monitors[kind].save(OUT / f"{kind}.npz")
        print(f"{kind}: {monitors[kind].n_pc} PCs of {len(monitors[kind].eigvals)}, lags {lags}, "
              f"{monitors[kind].n_rows} fit rows, fit {train_time[kind]:.2f} s")
    del fit_runs

    # --- limits and dev scoring -------------------------------------------------------
    limit_runs = data.load_runs(NORMAL, ids("limit"))
    dev_faulty = data.load_runs(FAULTY, ids("dev"))
    dev_normal = data.load_runs(NORMAL, ids("dev"))
    assert sorted(dev_faulty["faultNumber"].unique()) == list(range(1, 21))
    rows = []
    for kind, monitor in monitors.items():
        limits = alarm_limits(monitor, limit_runs, skip)
        rows += mspc.scorecard_rows(kind, monitor.score(dev_faulty), monitor.score(dev_normal), limits, onset, skip,
                                    machine=machine, split_file=SPLIT_FILE, train_time_s=train_time[kind],
                                    peak_ram_gb=prepare.peak_ram_gb())
    (OUT / "dev_scores.csv").unlink(missing_ok=True)
    evaluate.write_scorecard(rows, OUT / "dev_scores.csv")

    # --- print ------------------------------------------------------------------------
    df = pd.DataFrame(rows)
    for limit_name in ("1%", "5%", mspc.TEXTBOOK):
        sel = df[df.limit_name == limit_name]
        far = sel[sel.metric == "far"].set_index("model")["value"]
        print(f"dev FAR at {limit_name}: " + ", ".join(f"{m} {v:.4f}" for m, v in far.items()))
    fdr = df[(df.limit_name == "1%") & (df.metric == "fdr")].pivot(index="fault", columns="model", values="value")
    print("dev FDR at the 1% limit")
    print(fdr.loc[list(SHOWN_FAULTS)].round(3).to_string())
    mean = df[(df.limit_name == "1%") & df.metric.isin(["mean_fdr", "mean_fdr_excl_3_9_15"])]
    print(mean.pivot(index="metric", columns="model", values="value").round(3).to_string())

    # --- side chart --------------------------------------------------------------------
    sweep = pc_sweep(monitors["pca"], limit_runs, dev_faulty, onset)
    print(sweep.pivot(index="stat", columns="n_pc", values="mean_fdr").round(3).to_string())
    plot_sweep(sweep, monitors["pca"].n_pc, OUT / "figures" / "pc-sweep-dev.png")
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
