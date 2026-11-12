"""Phase 5: build results/report/report.md (comparison), operator-fault6.md (shift handover for one fault 6 testing run),
README.md and the new charts, from the saved results files.

Reads results/ (CSV, JSON, the saved PCA models) and one run of Faulty_Testing (fault 6, the first run ID in
results/phase-4/test_runs.json). Fits and tunes nothing. Overwrites everything in results/report/.
Exits with code 3 if a link in any of the three files does not exist.
Usage: .venv/Scripts/python scripts/make_report.py
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import contributions, data, operator_report, prepare, report, report_text, style  # noqa: E402

OUT = ROOT / "results" / "report"
FIGURES = OUT / "figures"
TESTING, FAULT, ONSET = "TEP_Faulty_Testing", 6, data.ONSET["Testing"]
O = style.OKABE_ITO
FAMILY_STYLE = {"PCA / DPCA": (O["vermillion"], "o"), "Isolation Forest": (O["blue"], "s"), "Autoencoder": (O["bluish_green"], "D"),
                "Gradient boosting": (O["black"], "^")}


# --- chart: the best model of each family, per fault ------------------------------------------------------------------------
def plot_detection_by_family(rows, path, spread=0.25):
    best = report_text.best_of_family(rows)
    faults = list(range(1, 21))
    frame = pd.DataFrame({family: rows[name]["fdr"].reindex(faults) for family, name in best.items()})
    hard = list(report.HARD_FAULTS)
    wide = [f for f in faults if f not in hard and frame.loc[f].max() - frame.loc[f].min() >= spread]
    title = (f"All four model families flag at most {frame.loc[hard].to_numpy().max():.2f} of faults 3, 9\n"
             f"and 15 (1% limit). They differ by {spread:.2f} or more\non faults {', '.join(map(str, wide))}")
    fig, ax = style.new_figure(figsize=(8, 9), font_size=13)
    for i, f in enumerate(faults):
        ax.plot([frame.loc[f].min(), frame.loc[f].max()], [i, i], color="0.85", lw=2, zorder=1)
    for family, name in best.items():
        colour, marker = FAMILY_STYLE[family]
        label = report.MODELS[name][0] + ("  (supervised)" if report.MODELS[name][1] == "supervised" else "")
        ax.scatter(frame[family], range(len(faults)), color=colour, marker=marker, s=60, zorder=3, label=label)
    ax.set_yticks(range(len(faults)), [f"Fault {f}" + (" (hard)" if f in hard else "") for f in faults])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Fault detection rate (share of post-onset samples flagged)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, -0.2), ncol=2, frameon=False, fontsize=10, handletextpad=0.2, columnspacing=0.9)
    style.finish_figure(fig, ax, title, path, title_size=13)


# --- charts: the operator report --------------------------------------------------------------------------------------------
def plot_run(facts, path):
    """SPE over the whole run (log scale), the 1% limit, the fault start and the alarm."""
    t, spe = facts["series"]["t_hours"], facts["series"]["spe"]
    start, alarm = facts["start_h"], facts["alarm"]
    fig, ax = style.new_figure(figsize=(8, 6), font_size=13)
    before = t < start
    ax.plot(t[before], spe[before], color=style.NORMAL, lw=1.4)
    ax.plot(t[~before], spe[~before], color=style.FAULT, lw=1.4)
    ax.axhline(facts["limit"], color="0.4", ls="--", lw=1.3)
    ax.text(t.max(), facts["limit"] * 0.8, f"1% alarm limit ({facts['limit']:.1f})", ha="right", va="top", color="0.3", fontsize=11)
    ax.axvline(start, color=style.FAULT, ls=":", lw=1.3)
    at = float(spe[np.isclose(t, alarm["t_hours"])][0])
    ax.scatter([alarm["t_hours"]], [at], color="black", s=60, zorder=5)
    ax.annotate(f"Fault starts {start:.1f} h, alarm {alarm['t_hours']:.1f} h\n({round(facts['delay_h'] * 60)} min later)",
                xy=(alarm["t_hours"], at), xytext=(start + 4, at * 0.12), fontsize=11, arrowprops={"arrowstyle": "-", "color": "0.3"})
    ax.text(start / 2, float(spe[before].max()) * 1.5, "normal", color=style.NORMAL, ha="center", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlabel("Run clock (hours)")
    ax.set_ylabel("PCA SPE (log scale)")
    style.finish_figure(fig, ax, operator_report.title_finding(facts).replace(" after the A feed", "\nafter the A feed"), path, title_size=13)


SHORT = {"xmeas_1": "A feed", "xmv_3": "A feed valve", "xmv_11": "Condenser water valve", "xmv_5": "Recycle valve"}
LINE_COLOURS = [O["vermillion"], O["orange"], O["blue"], O["bluish_green"]]


def plot_readings(facts, path):
    """The A feed, its valve and the two downstream readings as distance from normal (sd of normal runs)."""
    z = facts["series"]["z"]
    t = operator_report.run_clock(z.index.to_numpy())
    start, rows = facts["start_h"], facts["readings"]
    fig, ax = style.new_figure(figsize=(8, 6), font_size=13)
    for r, colour in zip(rows, LINE_COLOURS):
        ax.plot(t, z[r["tag"]].to_numpy(), color=colour, lw=1.8)
        ax.text(t.max() + 0.6, float(z[r["tag"]].iloc[-1]), f"{SHORT.get(r['tag'], r['name'])} ({r['tag']})", color=colour, va="center", fontsize=10)
    ax.axvline(start, color="0.4", ls=":", lw=1.3)
    ax.text(start - 0.5, ax.get_ylim()[1], f"fault starts\n{start:.1f} h", ha="right", va="top", fontsize=10, color="0.3")
    if facts["freeze"]:
        ax.axvline(facts["freeze"]["t_hours"], color="0.7", ls="--", lw=1.2)
        ax.text(facts["freeze"]["t_hours"] + 0.5, ax.get_ylim()[0] * 0.55,
                f"{facts['freeze']['n_frozen']} readings stop\nchanging ({facts['freeze']['t_hours']:.1f} h)", fontsize=10, color="0.35", va="center")
    ax.axhline(0, color="0.8", lw=1)
    ax.set_xlim(0, t.max() + 17)
    ax.set_xlabel("Run clock (hours)")
    ax.set_ylabel("Distance from normal (sd of normal runs)")
    down = [r["departure_after_h"] for r in rows if r["tag"] in facts["downstream"]]
    title = (f"The A feed drops and its valve opens at the fault start;\n"
             f"downstream readings leave normal {min(down):.1f} to {max(down):.1f} h later")
    style.finish_figure(fig, ax, title, path, title_size=13)


# --- the pass ---------------------------------------------------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    inp = report.load_inputs(ROOT)
    top = pd.read_csv(ROOT / "results" / "phase-4" / "top_readings_6.csv")
    run_id = int(inp["test_runs"]["run_ids"][0])           # the first run ID of the 100 fixed testing runs; not chosen by how it looks
    frame = data.load_runs(TESTING, [run_id], faults=[FAULT])
    assert len(frame) == 960 and (frame["faultNumber"] == FAULT).all(), frame.shape
    monitor = inp["pca"]
    assert monitor.lags == 0, monitor.lags
    reference = contributions.load_reference(data.PCA_MODEL)
    facts = operator_report.compute_facts(frame, monitor, reference, run_id, FAULT, ONSET, top, inp["test_blame"], inp["scorecard"])
    del frame

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_run(facts, FIGURES / "operator-fault6.png")
    plot_readings(facts, FIGURES / "operator-fault6-readings.png")
    rows = {name: report.model_row(inp["scorecard"], name) for name in report.MODELS}
    plot_detection_by_family(rows, FIGURES / "detection-by-family.png")

    def how_made():
        return (f"Built by `scripts/make_report.py` on {machine}. Runtime {time.perf_counter() - t0:.1f} s, peak RAM {prepare.peak_ram_gb():.2f} GB "
                f"(the whole script, including the one fault 6 testing run it reads).")

    (OUT / "operator-fault6.md").write_text(operator_report.operator_markdown(facts, how_made()), encoding="utf8")
    (OUT / "report.md").write_text(report_text.build_report(inp, how_made()), encoding="utf8")
    (OUT / "README.md").write_text(report_text.readme_text(), encoding="utf8")

    broken = {name: report.broken_links((OUT / name).read_text(encoding="utf8"), OUT) for name in ("report.md", "operator-fault6.md", "README.md")}
    broken = {k: v for k, v in broken.items() if v}
    if broken:
        print("BROKEN LINKS:", broken)
        sys.exit(3)
    print(f"operator run: fault {FAULT}, run {run_id}; alarm at {facts['alarm']['t_hours']:.1f} h, {facts['delay_h']:.1f} h after the start")
    print("findings:")
    for text in report_text.findings(inp):
        print(" -", text)
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
