"""Chart for post 1: one normal run against one fault 1 run on the most-shifted variable.

Uses training files only: FaultFree_Training run 1 and Faulty_Training fault 1 run 1.
Usage: .venv/Scripts/python scripts/plot_fault1.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase-0" / "figures" / "normal-vs-fault1.png"
ONSET = 20            # Faulty_Training: fault starts after sample 20 (1 h)
HOURS_PER_SAMPLE = 0.05  # 3-minute sampling
NORMAL, FAULT = "#0072B2", "#D55E00"  # Okabe-Ito blue and vermillion


def load_run(file, fault, run):
    cols = ["sample"] + data.VARIABLES
    return pq.read_table(PROCESSED / f"{file}.parquet", columns=cols,
                         filters=[("faultNumber", "=", fault), ("simulationRun", "=", run)]).to_pandas()


def main():
    normal = load_run("TEP_FaultFree_Training", 0, 1)
    faulty = load_run("TEP_Faulty_Training", 1, 1)
    assert len(normal) == 500 and len(faulty) == 500

    shifts = data.rank_shifts(faulty, data.VARIABLES, ONSET)
    print("Top 3 standardised mean shifts, fault 1 run 1 (after sample 20 vs samples 1-20):")
    for var, s in shifts.head(3).items():
        name, unit = data.VARIABLE_INFO[var]
        print(f"  {var}: {name} ({unit})  shift = {s:.1f} pre-onset standard deviations")

    var = shifts.index[0]
    name, unit = data.VARIABLE_INFO[var]
    pre = faulty.loc[faulty["sample"] <= ONSET, var].mean()
    after = faulty[faulty["sample"] > ONSET]
    peak_row = after.loc[after[var].abs().idxmax()] if abs(after[var].max() - pre) >= abs(after[var].min() - pre)         else after.loc[after[var].idxmin()]
    peak = peak_row[var]
    peak_row = after.loc[(after[var] - peak).abs() <= 0.005 * abs(peak)].iloc[0]  # first time at the peak (0.5 %)
    hours_to_peak = peak_row["sample"] * HOURS_PER_SAMPLE - ONSET * HOURS_PER_SAMPLE
    verb = "drives" if peak > pre else "pulls"
    title = (f"Fault 1 {verb} the {name.lower()} from {pre:.0f} to {peak:.0f} {unit} "
             f"within {int(np.ceil(hours_to_peak))} hours")

    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for df, colour in ((normal, NORMAL), (faulty, FAULT)):
        ax.plot(df["sample"] * HOURS_PER_SAMPLE, df[var], color=colour, lw=1.6)
    ax.axvline(ONSET * HOURS_PER_SAMPLE, color="0.35", ls="--", lw=1.4)
    top = max(normal[var].max(), faulty[var].max())
    bottom = min(normal[var].min(), faulty[var].min())
    pad = 0.12 * (top - bottom)
    ax.set_ylim(bottom - pad, top + pad)
    ax.annotate("Fault starts (hour 1)", (ONSET * HOURS_PER_SAMPLE, 0.0), xycoords=("data", "axes fraction"),
                xytext=(6, 6), textcoords="offset points", va="bottom", color="0.25")
    ax.annotate("Fault 1 run", (peak_row["sample"] * HOURS_PER_SAMPLE, peak), xytext=(40, 6),
                textcoords="offset points", ha="left", color=FAULT, fontweight="bold")
    x_n = 13.0
    y_n = normal.loc[(normal["sample"] * HOURS_PER_SAMPLE - x_n).abs().idxmin(), var]
    ax.annotate("Normal run", (x_n, y_n), xytext=(0, -10), textcoords="offset points", ha="center", va="top",
                color=NORMAL, fontweight="bold")
    last = 500 * HOURS_PER_SAMPLE
    ax.set_xlabel("Hours since start of run")
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_xlim(0, last)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", wrap=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"saved {OUT}\ntitle: {title}")


if __name__ == "__main__":
    main()
