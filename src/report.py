"""Phase 5: the comparison report, built from the saved results files only (no model is run here).

Every number in results/report/report.md goes through this module: the table builders read a results file, compute
nothing the files do not already hold (means are recomputed from per-fault values and checked against the stored
means), and the section writers put them into short prose. Rules: .scratch/phase-5/spec.md.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import HOURS_PER_SAMPLE
from src.evaluate import HARD_FAULTS

DEFAULT_LIMIT = "1%"
BENCHMARK_FAULTS = (1, 2, 4, 6, 7)
TRACKS = ("unsupervised", "supervised")

# scorecard model name -> (label, track)
MODELS = {
    "pca-t2": ("PCA T²", "unsupervised"), "pca-spe": ("PCA SPE", "unsupervised"),
    "dpca-t2": ("DPCA T² (L=2)", "unsupervised"), "dpca-spe": ("DPCA SPE (L=2)", "unsupervised"),
    "iforest-l0": ("Isolation Forest L=0", "unsupervised"), "iforest-l2": ("Isolation Forest L=2", "unsupervised"),
    "ae-l0": ("Autoencoder L=0", "unsupervised"), "ae-l2": ("Autoencoder L=2", "unsupervised"),
    "hgb-l2": ("Gradient boosting (L=2)", "supervised"),
}


# --- formatting --------------------------------------------------------------------------------------------------------
def _missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def fmt_fdr(value):
    """FDR or FAR with three decimals; n/a when missing."""
    return "n/a" if _missing(value) else f"{value:.3f}"


def fmt_share(value):
    """A share as a whole percent."""
    return "n/a" if _missing(value) else f"{value * 100:.0f}%"


def fmt_delay(median_min, never_share):
    """'24 min (0% never)': median delay over detected runs and the share of runs never detected. When 95% or more of the runs
    are never detected the shares get one decimal (so 99.6% is not shown as 100%) and the delay is labelled as over the few detected."""
    if _missing(median_min):
        return f"none detected ({fmt_share(never_share)} never)"
    if 0.95 <= never_share < 1:
        return f"{median_min:.0f} min over the {(1 - never_share) * 100:.1f}% detected ({never_share * 100:.1f}% never)"
    return f"{median_min:.0f} min ({fmt_share(never_share)} never)"


def fmt_seconds(value):
    """Seconds with one decimal below 10 s, whole seconds above."""
    if _missing(value):
        return "n/a"
    return f"{value:.1f} s" if value < 10 else f"{value:.0f} s"


def fmt_gb(value):
    return "n/a" if _missing(value) else f"{value:.1f} GB"


def md_table(frame):
    """A frame of strings as a Markdown pipe table."""
    cells = [[str(v) for v in row] for row in frame.itertuples(index=False)]
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines)


# --- the scorecard ---------------------------------------------------------------------------------------------------
def load_scorecard(path):
    return pd.read_csv(path)


def means_from_fdr(fdr, hard=HARD_FAULTS):
    """(mean over all faults, mean without the hard faults) of a Series of FDR indexed by fault."""
    return float(fdr.mean()), float(fdr.drop(list(hard), errors="ignore").mean())


def _select(scorecard, model, limit):
    rows = scorecard[(scorecard["model"] == model) & (scorecard["limit_name"] == limit)]
    if rows.empty:
        raise KeyError(f"no scorecard rows for model {model!r} at limit {limit!r}")
    return rows


def model_row(scorecard, model, limit=DEFAULT_LIMIT, tol=1e-9):
    """Everything the report needs about one model at one limit, as a dict.

    Keys: track, far, sustained_rate, train_time_s, peak_ram_gb, mean_fdr, mean_fdr_excl_3_9_15 (recomputed from the
    per-fault values), and fdr, delay, never (Series indexed by fault). Raises ValueError if a recomputed mean differs
    from the mean stored in the scorecard.
    """
    rows = _select(scorecard, model, limit)
    overall = rows[rows["fault"].isna()].set_index("metric")["value"]
    per_fault = rows[rows["fault"].notna()].assign(fault=lambda d: d["fault"].astype(int))

    def series(metric):
        return per_fault[per_fault["metric"] == metric].set_index("fault")["value"].sort_index()

    fdr = series("fdr")
    mean_all, mean_excl = means_from_fdr(fdr)
    for name, value in (("mean_fdr", mean_all), ("mean_fdr_excl_3_9_15", mean_excl)):
        if abs(overall[name] - value) > tol:
            raise ValueError(f"{model} at {limit}: stored {name} {overall[name]:.9f} != recomputed {value:.9f}")
    return {"track": rows["track"].iloc[0], "far": float(overall["far"]), "sustained_rate": float(overall["sustained_rate"]),
            "train_time_s": float(overall["train_time_s"]), "peak_ram_gb": float(overall["peak_ram_gb"]),
            "mean_fdr": mean_all, "mean_fdr_excl_3_9_15": mean_excl, "fdr": fdr, "delay": series("delay_median_min"),
            "never": series("never_detected_share")}


def detection_tables(scorecard, models=MODELS, limit=DEFAULT_LIMIT, faults=BENCHMARK_FAULTS, hard=HARD_FAULTS):
    """{track: {'summary', 'fdr', 'delay'}} of formatted frames, one set per track; a track never sees the other's models.

    `models` is {scorecard name: (label, track)}. The track must agree with the scorecard's track column.
    summary: FAR, fault-free 3-in-a-row rate, mean FDR with and without the hard faults, train time, peak RAM.
    fdr and delay: the asked `faults` then the `hard` faults as columns F<n>.
    """
    rows = {}
    for name, (label, track) in models.items():
        row = model_row(scorecard, name, limit)
        if row["track"] != track:
            raise ValueError(f"{name} is in the {row['track']} track in the scorecard, not {track}")
        rows[name] = (label, track, row)
    columns = [f"F{f}" for f in (*faults, *hard)]
    out = {}
    for track in TRACKS:
        chosen = [(label, row) for label, t, row in rows.values() if t == track]
        if not chosen:
            continue
        summary = pd.DataFrame([{
            "Model": label, "FAR": fmt_fdr(r["far"]), "3-in-a-row rate, fault-free": fmt_share(r["sustained_rate"]),
            "Mean FDR, all 20 faults": fmt_fdr(r["mean_fdr"]), "Mean FDR without 3, 9, 15": fmt_fdr(r["mean_fdr_excl_3_9_15"]),
            "Train time": fmt_seconds(r["train_time_s"]), "Peak RAM": fmt_gb(r["peak_ram_gb"])} for label, r in chosen])
        fdr = pd.DataFrame([{"Model": label, **{c: fmt_fdr(r["fdr"][int(c[1:])]) for c in columns}} for label, r in chosen])
        delay = pd.DataFrame([{"Model": label, **{c: fmt_delay(r["delay"][int(c[1:])], r["never"][int(c[1:])]) for c in columns}}
                              for label, r in chosen])
        out[track] = {"summary": summary, "fdr": fdr, "delay": delay}
    return out


# --- against the literature -------------------------------------------------------------------------------------------
LITERATURE = {"pca-t2": "PCA T²", "pca-spe": "PCA SPE", "dpca-t2": "DPCA T²", "dpca-spe": "DPCA SPE"}
ROUNDING = 5.1e-5  # vs_literature.csv holds our FDR rounded to 4 decimals


def literature_frame(vs_lit, scorecard, limit=DEFAULT_LIMIT):
    """Numeric frame indexed by fault: ours-<model> (full precision from the scorecard), rr-<model> (Rato & Reis 2013 as
    saved in vs_literature.csv) and diff-<model> = ours - rr. Raises ValueError if the saved comparison's own numbers
    disagree with the scorecard beyond rounding."""
    vs_lit = vs_lit.set_index("fault")
    out = pd.DataFrame(index=vs_lit.index)
    for model in LITERATURE:
        rows = _select(scorecard, model, limit)
        ours = rows[(rows["metric"] == "fdr") & rows["fault"].notna()].assign(fault=lambda d: d["fault"].astype(int))
        ours = ours.set_index("fault")["value"].reindex(vs_lit.index)
        worst = (ours - vs_lit[f"ours-{model}"]).abs().max()
        if not worst <= ROUNDING:
            raise ValueError(f"vs_literature.csv and the scorecard disagree for {model} (largest difference {worst:.5f})")
        out[f"ours-{model}"] = ours
        out[f"rr-{model}"] = vs_lit[f"rr-{model}"]
        out[f"diff-{model}"] = ours - vs_lit[f"rr-{model}"]
    return out


def literature_table(frame, hard=HARD_FAULTS):
    """Formatted table: fault, then ours and Rato & Reis per statistic, then a mean row over all faults and one without `hard`."""
    columns = {"Fault": []}
    for model, label in LITERATURE.items():
        columns[f"{label} ours"], columns[f"{label} R&R"] = [], []
    for fault, row in frame.iterrows():
        columns["Fault"].append(str(fault))
        for model, label in LITERATURE.items():
            columns[f"{label} ours"].append(fmt_fdr(row[f"ours-{model}"]))
            columns[f"{label} R&R"].append(fmt_fdr(row[f"rr-{model}"]))
    kept = frame.drop(list(hard), errors="ignore")
    for name, part in ((f"Mean, {len(frame)} faults", frame), (f"Mean without {', '.join(map(str, hard))}", kept)):
        columns["Fault"].append(name)
        for model, label in LITERATURE.items():
            columns[f"{label} ours"].append(fmt_fdr(part[f"ours-{model}"].mean()))
            columns[f"{label} R&R"].append(fmt_fdr(part[f"rr-{model}"].mean()))
    return pd.DataFrame(columns)


def literature_gaps(frame, threshold=0.10):
    """[{fault, model, ours, rr, diff}] where |ours - rr| >= threshold, by fault then model."""
    gaps = []
    for fault, row in frame.iterrows():
        for model in LITERATURE:
            if abs(row[f"diff-{model}"]) >= threshold - 1e-12:
                gaps.append({"fault": int(fault), "model": model, "ours": float(row[f"ours-{model}"]),
                             "rr": float(row[f"rr-{model}"]), "diff": float(row[f"diff-{model}"])})
    return gaps


# --- the unseen-fault test --------------------------------------------------------------------------------------------
UNSEEN_COLUMNS = ["hgb-l2-unseen", "hgb-l2", "pca-spe"]


def unseen_summary(unseen, carrier=16):
    """Means over the unseen faults, and how much of the gap (unseen supervised minus PCA SPE) one fault carries.

    Keys: mean and mean_without_carrier ({column: value}), gap_total (sum of the gaps), carrier_share (the carrier's gap over
    the total) and max_gap_others (largest absolute gap on the other faults).
    """
    unseen = unseen.set_index("fault")[UNSEEN_COLUMNS]
    gap = unseen["hgb-l2-unseen"] - unseen["pca-spe"]
    others = unseen.drop(index=carrier)
    return {"mean": unseen.mean().to_dict(), "mean_without_carrier": others.mean().to_dict(),
            "gap_total": float(gap.sum()), "carrier_share": float(gap[carrier] / gap.sum()),
            "max_gap_others": float(gap.drop(index=carrier).abs().max())}


def unseen_table(unseen):
    """Formatted table by fault with a mean row and the gap between the unseen-fault model and PCA SPE."""
    unseen = unseen.set_index("fault").sort_index()[UNSEEN_COLUMNS]
    gap = unseen["hgb-l2-unseen"] - unseen["pca-spe"]

    def line(name, row, g):
        return {"Fault": name, "Supervised, fault unseen": fmt_fdr(row["hgb-l2-unseen"]),
                "Supervised, all faults": fmt_fdr(row["hgb-l2"]), "PCA SPE (no labels)": fmt_fdr(row["pca-spe"]),
                "Unseen minus PCA SPE": f"{g:+.3f}"}

    rows = [line(str(f), unseen.loc[f], gap[f]) for f in unseen.index]
    rows.append(line(f"Mean, {len(unseen)} faults", unseen.mean(), gap.mean()))
    return pd.DataFrame(rows)


# --- diagnosis ---------------------------------------------------------------------------------------------------------
BLAME_WINDOW = "post"          # every sample after the onset
HIDDEN_GROUP = "hidden"        # faults 3, 9, 15 in the phase 4 tables


def _detectable(blame, window=BLAME_WINDOW):
    """Rows of the window for faults with a known expected unit, without the hidden faults."""
    rows = blame[(blame["window"] == window) & blame["expected_unit"].notna() & (blame["group"] != HIDDEN_GROUP)]
    return rows[rows["expected_unit"] != ""]


def detectable_faults(blame):
    """Sorted faults that enter the blame means: a known expected unit and not a hidden fault."""
    return sorted(int(f) for f in _detectable(blame)["fault"].unique())


def blame_summary(blame, perm_blame=None):
    """One row per method: mean top1_hit, top5_hit, unit_mass over the detectable faults with a known unit (post window),
    the random-picking baselines and the number of faults. `perm_blame` (a dev-run frame) adds only the methods `blame` lacks
    (permutation importance); its other rows are dev-run copies and are ignored."""
    rows = _detectable(blame)
    if perm_blame is not None:
        extra = _detectable(perm_blame)
        rows = pd.concat([rows, extra[~extra["method"].isin(set(rows["method"]))]], ignore_index=True)
    order = list(dict.fromkeys(rows["method"]))
    grouped = rows.groupby("method")
    out = grouped[["top1_hit", "top5_hit", "unit_mass"]].mean()
    out["random_top1_and_mass"] = grouped["baseline_unit_share"].mean()
    out["random_top5"] = grouped["baseline_top5_hit"].mean()
    out["n_faults"] = grouped["fault"].nunique()
    return out.loc[order].reset_index()


def blame_misses(blame, cutoff=0.05, method="spe"):
    """Detectable faults whose `method` top-5 hit rate (post window) is at or below `cutoff`, in fault order."""
    rows = _detectable(blame)
    rows = rows[(rows["method"] == method) & (rows["top5_hit"] <= cutoff + 1e-12)]
    return sorted(int(f) for f in rows["fault"])


def window_contrast(blame, fault, method="spe"):
    """top1 and top5 hit rates of one fault in the first-hour window and in the whole post-onset window."""
    rows = blame[(blame["fault"] == fault) & (blame["method"] == method)].set_index("window")
    return {"first_hour_top1": float(rows.loc["first_hour", "top1_hit"]), "post_top1": float(rows.loc["post", "top1_hit"]),
            "first_hour_top5": float(rows.loc["first_hour", "top5_hit"]), "post_top5": float(rows.loc["post", "top5_hit"])}


def hidden_table(hidden, faults):
    """Within-run before/after statistics of the chosen faults (0 = the fault-free control), two decimals."""
    table = hidden.set_index("fault").loc[list(faults)]
    return pd.DataFrame({
        "Fault": ["Fault-free control" if f == 0 else str(f) for f in table.index],
        "Largest shift (sd)": [f"{v:.2f}" for v in table["median_max_abs_shift"]],
        "Largest spread ratio": [f"{v:.2f}" for v in table["median_max_spread_ratio"]],
        "Largest change in autocorrelation": [f"{v:.2f}" for v in table["median_max_abs_dacf"]]})


def looks_like_control(hidden):
    """Faults whose mean-shift verdict is 'looks like the fault-free control' (the control itself is not listed)."""
    rows = hidden[(hidden["fault"] != 0) & hidden["verdict"].str.startswith("looks like")]
    return sorted(int(f) for f in rows["fault"])


# --- the assistant -----------------------------------------------------------------------------------------------------
ASSISTANT_V1, ASSISTANT_V11 = "v1 (PCA tools only)", "v1.1 (adds a trained classifier)"
ASSISTANT_ROWS = [  # (metric, label, with an interval)
    ("headline", "Unit accuracy (fault cases other than 3/9/15 and 16-20)", True),
    ("idv_top1_all", "IDV top-1 (all fault cases)", True),
    ("idv_top3_all", "IDV top-3 (all fault cases)", True),
    ("normal", "Normal cases called normal", True),
    ("hard", "Faults 3, 9, 15 answered 'normal' or 'can't tell' (not a detection)", True),
    ("baseline", "No-LLM baseline (unit of the top SPE reading)", False),
    ("chance", "Chance (always the most common unit)", False),
]


def _count_text(k, n, interval):
    from src.assistant.score import wilson
    k, n = int(k), int(n)
    text = f"{k} / {n} = {k / n:.2f}"
    if interval:
        lo, hi = wilson(k, n)
        text += f" [{lo:.2f}, {hi:.2f}]"
    return text


def assistant_table(comparison):
    """The v1 vs v1.1 table from assistant_v1_vs_v1.1.csv: counts with Wilson 95% intervals, and the cases that flipped."""
    table = comparison.set_index("metric")
    rows = []
    for metric, label, interval in ASSISTANT_ROWS:
        r = table.loc[metric]
        flips = "" if _missing(r["wrong_to_right"]) else f"{int(r['wrong_to_right'])} / {int(r['right_to_wrong'])}"
        rows.append({"Measure": label, ASSISTANT_V1: _count_text(r["v1_k"], r["v1_n"], interval),
                     ASSISTANT_V11: _count_text(r["v1.1_k"], r["v1.1_n"], interval), "Cases fixed / broken": flips})
    return pd.DataFrame(rows)


def repeat_agreement(diagnosis):
    """(k, n): of the n cases run more than once, k where every run gave the same status and unit."""
    from src.assistant.score import agreement
    result = agreement(diagnosis)
    return result["k"], result["n"]


def session_cost(folder):
    """(total cost in USD, number of sessions) from the result lines of the *.jsonl transcripts in a folder, or None if it is missing."""
    import json
    folder = Path(folder)
    if not folder.is_dir():
        return None
    total, sessions = 0.0, 0
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text(encoding="utf8").splitlines():
            record = json.loads(line)
            if record.get("type") == "result":
                total += float(record["total_cost_usd"])
                sessions += 1
    return total, sessions


# --- links -------------------------------------------------------------------------------------------------------------
_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def broken_links(markdown, base_dir):
    """Link and image targets in `markdown` that do not exist relative to `base_dir`, in order. URLs and #anchors are skipped."""
    base = Path(base_dir)
    missing = []
    for target in _LINK.findall(markdown):
        if re.match(r"[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
            continue
        if not (base / target.split("#")[0]).exists():
            missing.append(target)
    return missing


# --- loading ---------------------------------------------------------------------------------------------------------
def load_inputs(root):
    """Every results file the comparison report reads, as a dict (frames, dicts, the two saved PCA models)."""
    import json

    from src.mspc import PCAMonitor
    results = Path(root) / "results"
    csv = lambda *p: pd.read_csv(results.joinpath(*p))  # noqa: E731
    js = lambda *p: json.loads(results.joinpath(*p).read_text(encoding="utf8"))  # noqa: E731
    return {
        "scorecard": load_scorecard(results / "scorecard.csv"), "vs_lit": csv("phase-2", "vs_literature.csv"),
        "lag_choice": csv("phase-2", "lag_choice.csv"), "unseen": csv("phase-3", "unseen_fault.csv"),
        "hgb_tuning": csv("phase-3", "hgb_tuning.csv"), "assistant_cmp": csv("phase-3", "assistant_v1_vs_v1.1.csv"),
        "test_blame": csv("phase-4", "test_blame.csv"), "dev_blame": csv("phase-4", "dev_blame.csv"),
        "test_hidden": csv("phase-4", "test_hidden.csv"), "test_runs": js("phase-4", "test_runs.json"),
        "run_split": js("phase-1", "run_split.json"),
        "diag_v1": csv("phase-2b", "diagnosis_test.csv"), "diag_v11": csv("phase-2b", "diagnosis_test_v1.1.csv"),
        "cost_v1": session_cost(results / "phase-2b" / "transcripts_test"),
        "cost_v11": session_cost(results / "phase-2b" / "transcripts_test_v1.1"),
        "pca": PCAMonitor.load(results / "phase-2" / "pca.npz"), "dpca": PCAMonitor.load(results / "phase-2" / "dpca.npz"),
    }


# --- the pressure-limit tail --------------------------------------------------------------------------------------------
PRESSURE_TAG = "xmeas_7"          # reactor pressure, kPa gauge
PRESSURE_LEVEL_KPA = 2990.0       # close to 3000, where the readings sit flat in the fault 6 run; the cause is not verified here


def pressure_limit_table(frame, onset, threshold=PRESSURE_LEVEL_KPA, tag=PRESSURE_TAG):
    """Per fault: runs, share of runs whose `tag` reaches `threshold` after the onset, and the median first sample (and hours
    after the onset) over the runs that do. `frame` has faultNumber, simulationRun, sample and the tag."""
    post = frame[frame["sample"] > onset]
    hit = post[post[tag] >= threshold].groupby(["faultNumber", "simulationRun"])["sample"].min()
    runs = frame.groupby("faultNumber")["simulationRun"].nunique()
    first = hit.groupby(level=0).median().reindex(runs.index)
    n_hit = hit.groupby(level=0).size().reindex(runs.index).fillna(0)
    out = pd.DataFrame({"n_runs": runs, "share_reached": n_hit / runs, "median_first_sample": first})
    out["median_first_hours_after_onset"] = (out["median_first_sample"] - onset) * HOURS_PER_SAMPLE
    out.index.name = "fault"
    return out


def window_comparison(blame, pressure, method="spe", affected_share=0.5):
    """(table, summary): SPE top-5 hit in the first-hour and whole post-onset windows for the detectable faults, with the share of
    runs whose reactor pressure reaches the level after the onset (`pressure` from pressure_limit_table). Faults with a share at
    or above `affected_share` are 'affected'; summary holds the window means with and without them."""
    rows = _detectable(blame)
    rows = rows[rows["method"] == method]
    first = _detectable(blame, "first_hour")
    first = first[first["method"] == method].set_index("fault")["top5_hit"]
    whole = rows.set_index("fault")["top5_hit"]
    faults = sorted(whole.index)
    share = pressure["share_reached"].reindex(faults)
    affected = [int(f) for f in faults if share[f] >= affected_share]
    keep = [f for f in faults if f not in affected]
    lines = [{"Fault": str(f), "Top-5 hit, first hour": f"{first[f]:.2f}", "Top-5 hit, whole window": f"{whole[f]:.2f}",
              "Runs reaching 2990 kPa": fmt_share(share[f])} for f in faults]
    lines.append({"Fault": f"Mean, {len(faults)} faults", "Top-5 hit, first hour": f"{first[faults].mean():.2f}",
                  "Top-5 hit, whole window": f"{whole[faults].mean():.2f}", "Runs reaching 2990 kPa": ""})
    summary = {"mean_first_hour": float(first[faults].mean()), "mean_whole": float(whole[faults].mean()), "affected": affected,
               "mean_first_hour_without_affected": float(first[keep].mean()), "mean_whole_without_affected": float(whole[keep].mean())}
    return pd.DataFrame(lines), summary
