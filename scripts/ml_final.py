"""Phase 3 final scoring: the testing files, once. Nothing is fitted or chosen here.

Loads the models saved by scripts/ml_dev.py (their SHA-256 must match results/phase-3/models.json), scores all 500
FaultFree_Testing runs and all 500 runs x 20 faults of Faulty_Testing (one fault at a time) and writes every model at its
1% and 5% limit to results/scorecard.csv beside PCA and DPCA. Also writes results/phase-3/{fdr_by_fault.csv, unseen_fault.csv}
and figures/{fdr-by-fault-unsupervised.png, unseen-fault.png}. Exits with code 3 if a stop condition fails
(models.final_check_failures); the scorecard is written first so the numbers can be read.
--charts-only rewrites the two tables and redraws the two charts from results/scorecard.csv and dev_scores.csv without scoring anything.
Usage: .venv/Scripts/python scripts/ml_final.py [--charts-only]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")  # HistGradientBoosting ignores n_jobs; cap its threads

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, evaluate, models, prepare, split, style  # noqa: E402

NORMAL, FAULTY = "TEP_FaultFree_Testing", "TEP_Faulty_Testing"
OUT = data.RESULTS_3
UNSUPERVISED = ["iforest-l0", "iforest-l2", "ae-l0", "ae-l2"]
SUPERVISED, UNSEEN = "hgb-l2", "hgb-l2-unseen"
WIDE_ORDER = ["pca-t2", "pca-spe", "dpca-t2", "dpca-spe"] + UNSUPERVISED + [SUPERVISED, UNSEEN]
GATE_FAULTS = (1, 2, 6, 7)   # autoencoder sanity gate
N_RUNS = 500                 # runs per testing file (all of them)
COLOURS = {"pca-spe": style.OKABE_ITO["vermillion"], "if": style.OKABE_ITO["blue"], "ae": style.OKABE_ITO["bluish_green"],
           "hgb-l2": style.OKABE_ITO["black"], "hgb-l2-unseen": style.OKABE_ITO["orange"]}


def load_saved_models():
    """({name: monitor}, models.json record). Stops if a model file does not match its recorded SHA-256."""
    record = json.loads((OUT / "models.json").read_text())
    loaded = {}
    for entry in record["models"]:
        path = ROOT / entry["file"]
        if models.sha256_file(path) != entry["sha256"]:
            sys.exit(f"{entry['file']} does not match its SHA-256 in models.json; rerun scripts/ml_dev.py")
        loaded[entry["name"]] = models.load(path)
    return loaded, record


def best_by_dev(dev, prefix):
    """Name of the `prefix` model (iforest-l0/l2 or ae-l0/l2) with the best dev mean FDR excluding 3/9/15 at the 1% limit.
    Chosen on the dev runs, so the test scores play no part in which model the chart shows."""
    sel = dev[(dev.limit_name == "1%") & (dev.metric == "mean_fdr_excl_3_9_15") & dev.model.str.startswith(prefix)]
    return sel.sort_values("value", ascending=False)["model"].iloc[0]


def wide_table(scorecard):
    """Test FDR per fault at the 1% limit for every model on the scorecard, plus track, FAR and the two means."""
    one = scorecard[scorecard.limit_name == "1%"]
    cols = [m for m in WIDE_ORDER if m in set(one.model)]
    fdr = one[one.metric == "fdr"].pivot(index="fault", columns="model", values="value")[cols]
    track = one.drop_duplicates("model").set_index("model")["track"]
    overall = one[one.metric.isin(["far", "mean_fdr", "mean_fdr_excl_3_9_15"])].pivot(index="metric", columns="model",
                                                                                      values="value")[cols]
    table = pd.concat([pd.DataFrame([track[cols]], index=["track"]), fdr.rename(index=str),
                       overall.loc[["far", "mean_fdr", "mean_fdr_excl_3_9_15"]]])
    table.index.name = "fault"
    fdr.index = fdr.index.astype(int)
    return table, fdr


def write_tables(scorecard):
    """fdr_by_fault.csv and unseen_fault.csv from the scorecard, each with a `machine` column. Returns (FDR table, unseen table)."""
    table, fdr = wide_table(scorecard)
    machine = "; ".join(sorted(scorecard[scorecard.model.isin(table.columns)]["machine"].unique()))
    table.assign(machine=machine).to_csv(OUT / "fdr_by_fault.csv")
    unseen_table = fdr.loc[list(models.UNSEEN_FAULTS), [UNSEEN, SUPERVISED, "pca-spe"]]
    unseen_table.assign(machine=machine).to_csv(OUT / "unseen_fault.csv")
    return fdr, unseen_table


def dot_plot(fdr, series, title, path, figsize, colours, title_size=14):
    """One row per fault, one dot per model (`series` = [(column, label)]), a grey bar from the lowest to the highest."""
    faults = list(fdr.index)
    fig, ax = style.new_figure(figsize=figsize, font_size=14)
    low, high = fdr[[c for c, _, _ in series]].min(axis=1), fdr[[c for c, _, _ in series]].max(axis=1)
    for i, f in enumerate(faults):
        ax.plot([low[f], high[f]], [i, i], color="0.8", lw=2, zorder=1)
    for col, label, marker in series:
        ax.scatter(fdr[col].values, range(len(faults)), color=colours[col], s=60, marker=marker, zorder=2, label=label)
    ax.set_yticks(range(len(faults)), [f"Fault {f}" for f in faults])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Fault detection rate (share of post-onset samples flagged)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32 if len(faults) < 10 else -0.12), ncol=len(series),
              frameon=False, fontsize=11, handletextpad=0.2, columnspacing=1.0)
    style.finish_figure(fig, ax, title, path, title_size=title_size)


def plot_unsupervised(fdr, if_name, ae_name, path):
    top = fdr[["pca-spe", if_name, ae_name]]
    wins = (top.eq(top.max(axis=1), axis=0)).sum()
    title = (f"PCA SPE has the highest FDR on {wins['pca-spe']} of 20 faults, {if_name} on {wins[if_name]},\n"
             f"{ae_name} on {wins[ae_name]} (1% FAR, test runs; ties count for each)")
    colours = {"pca-spe": COLOURS["pca-spe"], if_name: COLOURS["if"], ae_name: COLOURS["ae"]}
    dot_plot(fdr, [("pca-spe", "PCA SPE", "o"), (if_name, if_name.replace("iforest", "Isolation Forest"), "s"),
                   (ae_name, ae_name.replace("ae", "Autoencoder"), "D")], title, path, (8, 9.5), colours, title_size=13)


def plot_unseen(unseen_table, path):
    full, left_out, pca = unseen_table["hgb-l2"].mean(), unseen_table[UNSEEN].mean(), unseen_table["pca-spe"].mean()
    verb = "falls" if left_out < full else "does not fall"
    title = (f"Without training on them, gradient boosting {verb}\n"
             f"from {full:.2f} to {left_out:.2f} mean FDR on faults {', '.join(str(f) for f in unseen_table.index)}\n"
             f"(PCA SPE: {pca:.2f}; 1% FAR, test runs)")
    colours = {"hgb-l2": COLOURS["hgb-l2"], UNSEEN: COLOURS["hgb-l2-unseen"], "pca-spe": COLOURS["pca-spe"]}
    dot_plot(unseen_table, [("hgb-l2", "Boosting, all faults seen", "s"), (UNSEEN, "Boosting, fault left out", "D"),
                            ("pca-spe", "PCA SPE", "o")], title, path, (8, 6), colours, title_size=13)


def charts_only():
    """Rewrite the two tables and redraw the charts from the saved scorecard (no model is loaded or scored)."""
    scorecard = pd.read_csv(evaluate.DEFAULT_SCORECARD, dtype={"model": str, "limit_name": str})
    fdr, unseen_table = write_tables(scorecard)
    dev = pd.read_csv(OUT / "dev_scores.csv")
    plot_unsupervised(fdr, best_by_dev(dev, "iforest"), best_by_dev(dev, "ae"),
                      OUT / "figures" / "fdr-by-fault-unsupervised.png")
    plot_unseen(unseen_table, OUT / "figures" / "unseen-fault.png")


def main():
    if "--charts-only" in sys.argv[1:]:
        return charts_only()
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    monitors, record = load_saved_models()
    names = UNSUPERVISED + [SUPERVISED, UNSEEN]
    assert sorted(monitors) == sorted(names), sorted(monitors)
    dev = pd.read_csv(OUT / "dev_scores.csv")
    dev_peak = float(dev[dev.metric == "peak_ram_gb"]["value"].max())
    onset = data.onset_for(FAULTY)
    grid = pd.read_csv(OUT / "hgb_tuning.csv")
    print("models: " + ", ".join(f"{n} (L={monitors[n].lags})" for n in names))
    print(f"hgb grid time (8 fits, logged apart from the model rows): {grid['fit_time_s'].sum():.0f} s")

    normal_frame = data.load_runs(NORMAL, range(1, N_RUNS + 1))
    assert normal_frame["simulationRun"].nunique() == N_RUNS and (normal_frame["faultNumber"] == 0).all()
    normal = {n: monitors[n].score(normal_frame) for n in names}
    del normal_frame
    faulty = {n: [] for n in names}
    for fault in range(1, 21):
        frame = data.load_runs(FAULTY, range(1, N_RUNS + 1), faults=[fault])
        assert frame["simulationRun"].nunique() == N_RUNS and (frame["faultNumber"] == fault).all()
        for n in names:
            faulty[n].append(monitors[n].score(frame))
        del frame
        print(f"scored fault {fault} ({time.perf_counter() - t0:.0f} s)", flush=True)
    faulty = {n: pd.concat(v, ignore_index=True) for n, v in faulty.items()}
    peak = max(dev_peak, prepare.peak_ram_gb())

    rows = []
    for n in names:
        rows += models.scorecard_rows(monitors[n], faulty[n], normal[n], onset, after=onset, machine=machine,
                                      split_file=split.SPLIT_FILE, train_time_s=monitors[n].meta["train_time_s"],
                                      peak_ram_gb=peak)
    evaluate.write_scorecard(rows, evaluate.DEFAULT_SCORECARD)
    ours = pd.DataFrame(rows)

    # --- derived files ----------------------------------------------------------------------------------------------------
    scorecard = pd.read_csv(evaluate.DEFAULT_SCORECARD, dtype={"model": str, "limit_name": str})
    fdr, unseen_table = write_tables(scorecard)

    # --- checks and print ---------------------------------------------------------------------------------------------------------
    one = ours[ours.limit_name == "1%"]
    print("\nTest table at the 1% limit")
    summary = one[one.metric.isin(["far", "sustained_rate", "mean_fdr", "mean_fdr_excl_3_9_15"])]
    print(summary.pivot(index="metric", columns="model", values="value")[names].round(4).to_string())
    print("test FAR at 5%: " + ", ".join(
        f"{n} {ours[(ours.model == n) & (ours.limit_name == '5%') & (ours.metric == 'far')]['value'].iloc[0]:.4f}"
        for n in names))
    ours_fdr = one[one.metric == "fdr"].pivot(index="fault", columns="model", values="value")[names]
    print("\nFDR on faults 3, 9, 15 at 1%")
    print(ours_fdr.loc[list(evaluate.HARD_FAULTS)].round(3).to_string())
    print("\nAutoencoder sanity gate: FDR on faults 1, 2, 6, 7 at 1% next to PCA SPE")
    gate = fdr.loc[list(GATE_FAULTS), ["pca-spe", "ae-l0", "ae-l2"]]
    print(gate.round(3).to_string())
    for n in ("ae-l0", "ae-l2"):
        m = monitors[n].meta
        print(f"{n}: {len(m['val_mse_history'])} epochs, best {m['best_epoch'] + 1}, final train MSE "
              f"{m['final_train_mse']:.4f}, validation MSE {m['final_val_mse']:.4f}")
    chosen = grid[grid.chosen].iloc[0]
    print(f"\nhgb tuning choice: lr {chosen.learning_rate}, leaves {int(chosen.max_leaf_nodes)}, iter "
          f"{int(chosen.max_iter)} (dev mean FDR excl 3/9/15 {chosen.dev_mean_fdr_excl_3_9_15:.4f})")
    print("\nUnseen-fault test (test FDR at 1%)")
    print(unseen_table.round(3).to_string())
    print(f"mean: {unseen_table.mean().round(3).to_dict()}")
    print(f"train time s: " + ", ".join(f"{n} {monitors[n].meta['train_time_s']:.0f}" for n in names)
          + f"; peak RAM GB: dev {dev_peak:.2f}, final scoring {prepare.peak_ram_gb():.2f}")

    stops, notes = models.final_check_failures(ours, UNSUPERVISED, [SUPERVISED], unseen=UNSEEN)
    for line in stops:
        print("CHECK FAILED:", line)
    for line in notes:
        print("NOTE:", line)
    if_best, ae_best = best_by_dev(dev, "iforest"), best_by_dev(dev, "ae")
    print(f"chart models (best on dev): {if_best}, {ae_best}")
    plot_unsupervised(fdr, if_best, ae_best, OUT / "figures" / "fdr-by-fault-unsupervised.png")
    plot_unseen(unseen_table, OUT / "figures" / "unseen-fault.png")
    print(prepare.runtime_line(t0))
    if stops:
        sys.exit(3)


if __name__ == "__main__":
    main()
