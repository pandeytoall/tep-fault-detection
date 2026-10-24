"""Chart for post 1: one normal run against one fault 1 run on the most-shifted variable.

Uses training files only: FaultFree_Training run 1 and Faulty_Training fault 1 run 1.
Usage: .venv/Scripts/python scripts/plot_fault1.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, signatures, style  # noqa: E402
from src.variables import VARIABLE_INFO  # noqa: E402

OUT = ROOT / "results" / "phase-0" / "figures" / "normal-vs-fault1.png"


def main():
    onset = data.onset_for("TEP_Faulty_Training")
    n_samples = data.FILES["TEP_Faulty_Training"]["samples"]
    normal = data.load_run("TEP_FaultFree_Training", 0, 1)
    faulty = data.load_run("TEP_Faulty_Training", 1, 1)
    assert len(normal) == len(faulty) == n_samples

    shifts = signatures.rank_shifts(faulty, data.VARIABLES, onset)
    print(f"Top 3 standardised mean shifts, fault 1 run 1 (after sample {onset} vs samples 1-{onset}):")
    for var, s in shifts.head(3).items():
        name, unit = VARIABLE_INFO[var].name, VARIABLE_INFO[var].unit
        print(f"  {var}: {name} ({unit})  shift = {s:.1f} pre-onset standard deviations")

    var = shifts.index[0]
    name, unit = VARIABLE_INFO[var].name, VARIABLE_INFO[var].unit
    pre = faulty.loc[faulty["sample"] <= onset, var].mean()
    peak_sample, peak = signatures.peak_after_onset(faulty, var, onset)
    hours = round((peak_sample - onset) * data.HOURS_PER_SAMPLE * 2) / 2  # nearest half hour
    verb = "drives" if peak > pre else "pulls"
    title = (f"Fault 1 {verb} the {name.lower()}\n"
             f"from {pre:.0f} {unit} to about {peak:.0f} {unit} within {hours:g} hours of the fault")
    onset_h = onset * data.HOURS_PER_SAMPLE

    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for df, colour in ((normal, style.NORMAL), (faulty, style.FAULT)):
        ax.plot(df["sample"] * data.HOURS_PER_SAMPLE, df[var], color=colour, lw=1.6)
    ax.axvline(onset_h, color="0.35", ls="--", lw=1.4)
    top = max(normal[var].max(), faulty[var].max())
    bottom = min(normal[var].min(), faulty[var].min())
    pad = 0.12 * (top - bottom)
    ax.set_ylim(bottom - pad, top + pad)
    ax.annotate(f"Fault starts (hour {onset_h:g})", (onset_h, 0.0), xycoords=("data", "axes fraction"),
                xytext=(6, 6), textcoords="offset points", va="bottom", color="0.25")
    ax.annotate("Fault 1 run", (peak_sample * data.HOURS_PER_SAMPLE, peak), xytext=(40, 6),
                textcoords="offset points", ha="left", color=style.FAULT, fontweight="bold")
    x_n = 13.0
    y_n = normal.loc[(normal["sample"] * data.HOURS_PER_SAMPLE - x_n).abs().idxmin(), var]
    ax.annotate("Normal run", (x_n, y_n), xytext=(0, -10), textcoords="offset points", ha="center", va="top",
                color=style.NORMAL, fontweight="bold")
    last = n_samples * data.HOURS_PER_SAMPLE
    ax.set_xlabel("Hours since start of run")
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_xlim(0, last)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"saved {OUT}\ntitle: {title}")


if __name__ == "__main__":
    main()
