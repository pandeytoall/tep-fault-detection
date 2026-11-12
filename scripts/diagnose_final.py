"""Phase 4 final pass: the same blame and before/after analysis on the testing files, once. Nothing is fitted or chosen here.

Reads the saved PCA model (results/phase-2/pca.npz, with its SPE and T2 references) and the RBC reference written by
scripts/diagnose_dev.py, then for each fault 1-20 reads the same 100 runs (the first 100 of a seed-42 shuffle of the 500, saved in
test_runs.json) of Faulty_Testing and their post-onset window (samples after 160) and first hour, and 100 other runs (the next
100 of the shuffle, sharing no noise stream with them) of FaultFree_Testing as the before/after control. The permutation-importance rows in the blame chart come from the dev runs (dev only, no testing data).
Writes results/phase-4/{test_blame.csv, test_blame_by_group.csv, test_blame_median_spe.csv, test_hidden.csv, test_runs.json} and
figures/{blame-heatmap.png, blame-methods.png, hidden-faults.png}. Exits with code 3 if a fault in 3/9/15 shows a clear shift.
Usage: .venv/Scripts/python scripts/diagnose_final.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import psutil
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, diagnosis, mspc, prepare, split, style  # noqa: E402
from src.variables import VARIABLE_INFO, scoring_unit  # noqa: E402

NORMAL, FAULTY = "TEP_FaultFree_Testing", "TEP_Faulty_Testing"
OUT = data.RESULTS_4
FIGURES = OUT / "figures"
GROUP_LABELS = {"condenser": "cond.", "separator": "sep.", "compressor": "comp."}   # short, so neighbouring groups do not overlap
PLANT_UNIT_ORDER = ["feed", "reactor", "condenser", "separator", "compressor", "purge", "stripper", "product"]
CHART_METHODS = ("spe", "t2", "rbc_spe", "perm")
COLOURS = {"spe": style.OKABE_ITO["vermillion"], "t2": style.OKABE_ITO["blue"], "rbc_spe": style.OKABE_ITO["bluish_green"],
           "perm": style.OKABE_ITO["black"], "random": "0.55"}
OFFSETS = {"spe": -0.18, "t2": -0.06, "rbc_spe": 0.06, "perm": 0.18}   # a small vertical offset keeps overlapping dots visible
MARKERS = {"spe": "o", "t2": "s", "rbc_spe": "D", "perm": "^"}


def short_name(tag):
    """D&V name, with the analyzer wording shortened ('Reactor feed analysis, component A' -> 'Reactor feed A')."""
    return VARIABLE_INFO[tag].name.replace(" analysis, component", "")


def column_order(tags):
    """Indices of `tags` sorted by plant unit (PLANT_UNIT_ORDER), then by tag number; scoring units stay contiguous."""
    return sorted(range(len(tags)), key=lambda j: (PLANT_UNIT_ORDER.index(VARIABLE_INFO[tags[j]].plant_unit), j))


# --- charts --------------------------------------------------------------------------------------------------------------------
def plot_heatmap(median_spe, tags, path):
    """Faults 1-20 x 52 readings, colour = median relative SPE contribution (log scale); the expected unit's block is outlined."""
    order = column_order(tags)
    values = median_spe.loc[:, [tags[j] for j in order]]
    units = [scoring_unit(tags[j]) for j in order]
    detectable = [f for f in values.index if diagnosis.expected_unit(f) and f not in diagnosis.HIDDEN_FAULTS]
    hits = sum(scoring_unit(values.columns[int(np.argmax(values.loc[f].to_numpy()))]) == diagnosis.expected_unit(f) for f in detectable)
    title = (f"SPE's top reading lies in the faulted unit for {hits} of {len(detectable)} detectable faults with a known unit\n"
             f"(median relative SPE contribution, test runs; outlined = expected unit)")
    fig, ax = style.new_figure(figsize=(16, 9.5), font_size=11)
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap="viridis", norm=LogNorm(vmin=1.0, vmax=max(float(values.to_numpy().max()), 10.0)))
    for i, fault in enumerate(values.index):
        unit = diagnosis.expected_unit(fault)
        cols = [k for k, u in enumerate(units) if u == unit]
        if cols:
            ax.add_patch(Rectangle((cols[0] - 0.5, i - 0.5), len(cols), 1, fill=False, ec="white", lw=2.2))
    labels = [f"Fault {f}" + (" (hidden)" if f in diagnosis.HIDDEN_FAULTS else " (cause unknown)" if f in diagnosis.UNKNOWN_FAULTS else "")
              for f in values.index]
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.set_xticks(range(len(order)), [short_name(t) for t in values.columns], rotation=90, fontsize=8)
    start = 0                                      # plant unit names above the columns
    for unit in PLANT_UNIT_ORDER:
        n = sum(VARIABLE_INFO[tags[j]].plant_unit == unit for j in order)
        ax.text(start + n / 2 - 0.5, -0.8, GROUP_LABELS.get(unit, unit), ha="center", va="bottom", fontsize=10, fontweight="bold")
        if start:
            ax.axvline(start - 0.5, color="0.85", lw=0.8)
        start += n
    colourbar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.025)
    colourbar.set_label("Relative SPE contribution (1 = normal), log scale")
    style.finish_figure(fig, ax, title, path, title_size=14, pad=34)


def plot_methods(test_blame, dev_blame, path):
    """top5_hit per fault (faults with an expected unit) for SPE, T2, RBC-SPE (test runs) and permutation importance (dev runs)."""
    faults = [f for f in range(1, 21) if diagnosis.expected_unit(f)]
    post = test_blame[test_blame["window"] == "post"]
    perm = dev_blame[(dev_blame["window"] == "post") & (dev_blame["method"] == "perm")].set_index("fault")
    hit = {m: post[post["method"] == m].set_index("fault")["top5_hit"] for m in ("spe", "t2", "rbc_spe")}
    hit["perm"] = perm["top5_hit"]
    baseline = post[post["method"] == "spe"].set_index("fault")["baseline_top5_hit"]
    detectable = [f for f in faults if f not in diagnosis.HIDDEN_FAULTS]
    spe_mean, base_mean = hit["spe"][detectable].mean(), baseline[detectable].mean()
    title = (f"SPE's top-5 readings hit the faulted unit in {spe_mean:.0%} of runs,\n"
             f"random picks in {base_mean:.0%} (mean over {len(detectable)} detectable faults, test runs)")
    fig, ax = style.new_figure(figsize=(8, 9), font_size=13)
    for i, f in enumerate(faults):
        values = [hit[m][f] for m in CHART_METHODS]
        ax.plot([min(values), max(values)], [i, i], color="0.85", lw=2, zorder=1)
    ax.scatter([baseline[f] for f in faults], range(len(faults)), marker="|", s=260, color=COLOURS["random"], lw=2.5, zorder=2,
               label="Random picks")
    labels = {"spe": "SPE (test)", "t2": "T² (test)", "rbc_spe": "RBC-SPE (test)", "perm": "Permutation importance (dev)"}
    for m in CHART_METHODS:
        ax.scatter([hit[m][f] for f in faults], np.arange(len(faults)) + OFFSETS[m], color=COLOURS[m], marker=MARKERS[m], s=55, zorder=3, label=labels[m])
    ax.set_yticks(range(len(faults)), [f"Fault {f}" + (" (hidden)" if f in diagnosis.HIDDEN_FAULTS else "") for f in faults])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Share of runs with a top-5 reading in the faulted unit")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.24), ncol=3, frameon=False, fontsize=10, handletextpad=0.2, columnspacing=0.9)
    style.finish_figure(fig, ax, title, path, title_size=13)


def plot_hidden(hidden, path):
    """Largest median |shift| per fault (dev), faults 3, 9, 15 and the fault-free control highlighted."""
    table = hidden.set_index("fault")
    shifts = table["median_max_abs_shift"]
    control = float(shifts[0])
    order = list(range(1, 21)) + [0]
    colour = lambda f: COLOURS["perm"] if f == 0 else style.OKABE_ITO["vermillion"] if f in diagnosis.HIDDEN_FAULTS else style.OKABE_ITO["sky_blue"]  # noqa: E731
    h = [shifts[f] for f in diagnosis.HIDDEN_FAULTS]
    title = (f"Faults 3, 9 and 15 move a reading by at most {max(h):.2f} sd, the same\n"
             f"as a fault-free run ({control:.2f} sd; median over dev runs)")
    fig, ax = style.new_figure(figsize=(8, 9), font_size=13)
    ax.axvline(control, color="0.5", ls="--", lw=1.3, zorder=1)
    ax.scatter([shifts[f] for f in order], range(len(order)), c=[colour(f) for f in order], s=70, zorder=2)
    for i, f in enumerate(order):
        ax.plot([0, shifts[f]], [i, i], color=colour(f), lw=1.6, alpha=0.6, zorder=1)
    ax.set_yticks(range(len(order)), ["No fault (control)" if f == 0 else f"Fault {f}" + (" (hidden)" if f in diagnosis.HIDDEN_FAULTS else "")
                                      for f in order])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.3, max(float(shifts.max()) * 1.3, 2))
    ax.set_xlabel("Largest change in any reading (sd of normal runs, log scale)")
    ax.text(control * 1.06, len(order) - 0.45, "fault-free level", color="0.35", fontsize=11, va="bottom")
    style.finish_figure(fig, ax, title, path, title_size=13)


# --- the pass --------------------------------------------------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    onset = data.onset_for(FAULTY)
    run_ids = diagnosis.test_run_ids()
    monitor = mspc.PCAMonitor.load(data.PCA_MODEL)
    assert monitor.lags == 0 and monitor.n_pc == 31, (monitor.lags, monitor.n_pc)
    tags = monitor.variables
    refs = diagnosis.load_references(data.PCA_MODEL, OUT / "rbc_reference.npz")

    blame, change, medians = [], {}, {}
    for fault in range(1, 21):
        frame = data.load_runs(FAULTY, run_ids, faults=[fault])
        assert frame["simulationRun"].nunique() == len(run_ids) and (frame["faultNumber"] == fault).all()
        result = diagnosis.analyse_fault(monitor, refs, frame, fault, onset)
        blame += result["blame"]
        change[fault] = result["change"]
        medians[fault] = pd.Series(np.median(result["vectors"]["post"]["spe"], axis=0), index=tags)
        del frame
        print(f"fault {fault} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    control_ids = diagnosis.test_run_ids(offset=100)                  # fault-free runs sharing no ID (no noise stream) with the faulty ones
    normal = data.load_runs(NORMAL, control_ids)
    assert normal["simulationRun"].nunique() == len(control_ids) and (normal["faultNumber"] == 0).all()
    control = diagnosis.analyse_fault(monitor, refs, normal, 0, onset)
    del normal

    blame = pd.DataFrame(blame)
    post = blame[blame["window"] == "post"]
    summary = diagnosis.group_summary(post)
    hidden = diagnosis.hidden_table(change, control["change"])
    median_spe = pd.DataFrame(medians).T.rename_axis("fault")
    info = {"machine": machine, "runtime_s": round(time.perf_counter() - t0, 1), "peak_ram_gb": round(prepare.peak_ram_gb(), 2)}

    # --- print --------------------------------------------------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\nBlame by fault group, post-onset window (test, {len(run_ids)} runs per fault; baseline = random picks)")
    print(summary.round(3).to_string(index=False))
    baselines = post.drop_duplicates("fault").set_index("fault")
    for metric, base in (("top1_hit", "baseline_unit_share"), ("top5_hit", "baseline_top5_hit"),
                         ("unit_mass", "baseline_unit_share"), ("smear", None)):
        wide = post.pivot(index="fault", columns="method", values=metric)[list(diagnosis.PCA_METHODS)]
        if base:
            wide["random"] = baselines[base]
        print(f"-- {metric}")
        print(wide.round(2).to_string())
    print("\nTop 3 SPE readings per fault (median over runs)")
    for fault in range(1, 21):
        note = diagnosis.fault_note(fault)
        print(f"  fault {fault:2d}: {post[(post.fault == fault) & (post.method == 'spe')]['top3'].iloc[0]}"
              + (f"   [{note}]" if note else ""))
    cols = ["fault", "group", "median_max_abs_shift", "median_n_shift_gt_1sd", "median_max_spread_ratio", "median_max_abs_dacf",
            "effect_vs_control", "verdict"]
    print(f"\nWithin-run before/after (test: 160 samples before, 800 after the onset); fault 0 = fault-free control")
    print(hidden[cols].round(2).to_string(index=False))
    stops = diagnosis.hidden_stop_messages(hidden)
    for text in stops:
        print("CHECK FAILED:", text)
    print(prepare.runtime_line(t0))

    # --- write ----------------------------------------------------------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    blame.assign(**info).to_csv(OUT / "test_blame.csv", index=False)
    summary.assign(**info).to_csv(OUT / "test_blame_by_group.csv", index=False)
    median_spe.reset_index().assign(**info).to_csv(OUT / "test_blame_median_spe.csv", index=False)
    hidden.assign(**info).to_csv(OUT / "test_hidden.csv", index=False)
    (OUT / "test_runs.json").write_text(json.dumps({"seed": 42, "rule": "first 100 of a seed-42 shuffle of 1..500, the same for every fault",
                                                    "n_runs": len(run_ids), "run_ids": run_ids, "control_run_ids": control_ids, "onset": onset,
                                                    "split_file": split.SPLIT_FILE, **info}, indent=1))
    if stops:
        sys.exit(3)
    dev_blame = pd.read_csv(OUT / "dev_blame.csv")
    dev_hidden = pd.read_csv(OUT / "dev_hidden.csv")
    plot_heatmap(median_spe, tags, FIGURES / "blame-heatmap.png")
    plot_methods(blame, dev_blame, FIGURES / "blame-methods.png")
    plot_hidden(dev_hidden, FIGURES / "hidden-faults.png")
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
