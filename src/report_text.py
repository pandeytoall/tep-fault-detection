"""Phase 5: the words of results/report/report.md. Each section function takes the loaded results (report.load_inputs) and
returns Markdown; every number in it comes from one of those files through src/report.py (or is a count of rows in them).

Style: short sentences, a one-line meaning for each term, and the caveat that belongs to a number written next to it.
"""
import pandas as pd

from src import report
from src.operator_report import SHUTDOWN
from src.report import fmt_fdr, fmt_gb, fmt_seconds, fmt_share, md_table

FIGURES = "figures/detection-by-family.png"
PHASE_FIGURES = {
    "fdr-by-fault": "../phase-2/figures/fdr-by-fault.png",
    "fdr-unsupervised": "../phase-3/figures/fdr-by-fault-unsupervised.png",
    "unseen": "../phase-3/figures/unseen-fault.png",
    "blame-methods": "../phase-4/figures/blame-methods.png",
    "blame-heatmap": "../phase-4/figures/blame-heatmap.png",
    "hidden": "../phase-4/figures/hidden-faults.png",
}
FAULT_NAMES = {1: "A/C feed ratio step", 2: "B composition step", 4: "reactor cooling water inlet temperature step",
               6: "A feed loss", 7: "C header pressure loss"}

# the caveat that must sit inside each section (not only in the closing list): section heading prefix -> phrases
CAVEATS = {
    "What this is": ["one seed-42 split", "tuned with all 20 faults", "36 headline cases", "may remember", "shutdown limit", "first-hour numbers are the fairer view"],
    "Data and rules": ["Rieth", "one seed-42 split"],
    "Detection": ["every 3rd row", "false-alarm streaks", "one seed-42 split", "grid fits are degenerate"],
    "Against the literature": ["one run per fault", "Rieth"],
    "The unseen-fault test": ["tuned with all 20 faults", "every 3rd row", "one seed-42 split"],
    "Diagnosis": ["every 5th", "one dev vector per fault", "mean shift", "14 and 19", "shutdown limit", "pressure tail", "first hour"],
    "What we did not do": ["shutdown limit", "pressure tail"],
    "The assistant": ["36 headline cases", "may remember", "all 90", "not a detection"],
}


def src(*names):
    """'Source: [`name`](../name), ...' for files given relative to results/."""
    links = ", ".join(f"[`{n}`](../{n})" for n in names)
    return f"_Source: {links}._"


def _f(value, digits=3):
    return f"{value:.{digits}f}"


def _rows(inp):
    return {name: report.model_row(inp["scorecard"], name) for name in report.MODELS}


def _label(name):
    return report.MODELS[name][0]


def _hard_max(row):
    return max(row["fdr"][f] for f in report.HARD_FAULTS)


# --- the opening: what to take from it ---------------------------------------------------------------------------------------
def findings(inp):
    """The five bullets, as a list of strings (kept separate so the build report can quote them)."""
    rows = _rows(inp)
    unsup = {n: r for n, r in rows.items() if report.MODELS[n][1] == "unsupervised"}
    excl = {n: r["mean_fdr_excl_3_9_15"] for n, r in unsup.items()}
    best = max(excl, key=excl.get)
    hgb = rows["hgb-l2"]
    worst_hard = max(_hard_max(r) for r in rows.values())
    far_lo, far_hi = min(r["far"] for r in rows.values()), max(r["far"] for r in rows.values())
    us = report.unseen_summary(inp["unseen"])
    blame = report.blame_summary(inp["test_blame"], inp["dev_blame"]).set_index("method")
    contrast = report.window_contrast(inp["test_blame"], fault=6)
    _, wc = report.window_comparison(inp["test_blame"], inp["pressure"])
    cmp = inp["assistant_cmp"].set_index("metric")
    k1, n1, k11, n11 = (int(cmp.loc["headline", c]) for c in ("v1_k", "v1_n", "v1.1_k", "v1.1_n"))
    lo1, hi1 = _wilson(k1, n1)
    lo11, hi11 = _wilson(k11, n11)
    hard_k, hard_n = int(cmp.loc["hard", "v1.1_k"]), int(cmp.loc["hard", "v1.1_n"])
    n_v11 = int(inp["diag_v11"]["used_classify"].sum())
    return [
        (f"**Detection at a 1% false-alarm limit.** The unsupervised models (they learn from normal runs only) flag "
         f"{_f(min(excl.values()))} to {_f(max(excl.values()))} of post-onset samples on average without faults 3, 9, 15; "
         f"{_label(best)} is highest ({_f(excl[best])}) and PCA SPE, the reference, reaches {_f(excl['pca-spe'])}. Gradient boosting, "
         f"the supervised model (it also learns from labelled faults), reaches {_f(hgb['mean_fdr_excl_3_9_15'])}. The two tracks have "
         f"separate tables and are not ranked against each other. The top four unsupervised models sit within "
         f"{_f(sorted(excl.values())[-1] - sorted(excl.values())[-4])} of each other on one seed-42 split, so we call no winner among them."),
        (f"**Faults 3, 9 and 15 are not detected by any model.** Every model flags at most {_f(worst_hard)} of their post-onset samples, "
         f"against a false-alarm rate of {_f(far_lo)} to {_f(far_hi)}. Inside one run the readings after the fault start look like the "
         f"readings before it, exactly as a fault-free run does. This is a result about these data, and the three faults sit at "
         f"the false-alarm floor, as the literature reports."),
        (f"**Not seeing a fault costs the supervised model most on fault 16.** Retrained without faults 1, 8, 13, 14 and 16, it "
         f"flags {_f(us['mean']['hgb-l2-unseen'])} of samples on those five, against {_f(us['mean']['pca-spe'])} for PCA SPE, which uses no labels. "
         f"Fault 16 carries {fmt_share(us['carrier_share'])} of that gap; on the other four the two are within {_f(us['max_gap_others'])}. "
         f"The model's settings were tuned with all 20 faults, including these five, so the unseen-fault result is mildly optimistic."),
        (f"**Contribution plots point at symptoms, not causes.** SPE's top five readings include the faulted plant area in "
         f"{fmt_share(wc['mean_whole'])} of runs over the whole post-onset window and {fmt_share(wc['mean_first_hour'])} in the first hour, random picking in "
         f"{fmt_share(blame.loc['spe', 'random_top5'])}; T² is no better than random. "
         f"For fault 6 (A feed loss) a feed reading (the A feed or its valve) is the top SPE reading in {fmt_share(contrast['first_hour_top1'])} of runs in the first hour "
         f"and {fmt_share(contrast['post_top1'])} over the whole window. Most of that whole window comes after reactor pressure reaches about 3000 kPa, "
         f"{SHUTDOWN}, so the first-hour numbers are the fairer view and whole-window blame can be inflated by that tail."),
        (f"**The assistant scored higher with a trained classifier, which does not show better reasoning.** On the frozen cases it named the right plant "
         f"area in {k1} of {n1} fault cases as v1 (95% interval {_f(lo1, 2)} to {_f(hi1, 2)}) and {k11} of {n11} as v1.1 ({_f(lo11, 2)} to {_f(hi11, 2)}). "
         f"v1.1 adds a supervised classifier trained on faults 1-20, and all {n_v11} v1.1 sessions used it, so the gain is \"a trained classifier helps\" "
         f"and the two cannot be separated. With only {n1} headline cases the "
         f"intervals are wide, and Claude may remember the published TEP papers. Answering 'normal' or 'can't tell' on faults 3, 9, 15 "
         f"({hard_k} of {hard_n}) is not a detection."),
    ]


def _wilson(k, n):
    from src.assistant.score import wilson
    return wilson(k, n)


def section_summary(inp):
    return "## What this is and what to take from it\n\n" \
        "We compared fault-detection models on the public Tennessee Eastman Process (TEP) simulation data, on one scorecard. " \
        "Everything below was generated from saved results files by `scripts/make_report.py`; every table says where its numbers come from. " \
        "Five things to take from it:\n\n" + "\n".join(f"- {b}" for b in findings(inp)) + "\n"


# --- data and rules -------------------------------------------------------------------------------------------------------------
def groups_table(split):
    """Small table of the run groups from run_split.json plus the testing role."""
    sizes = split["sizes"]
    return pd.DataFrame([
        {"Run group (training family)": "fit", "Runs": sizes["fit"], "Job": "Unsupervised models learn normal behaviour. The lag choice and autoencoder validation use these too, split by run."},
        {"Run group (training family)": "limit", "Runs": sizes["limit"], "Job": "Set the alarm limits at 1% and 5% false-alarm rate. Nothing else."},
        {"Run group (training family)": "limit_tune", "Runs": sizes["limit_tune"], "Job": "Half of `limit`; also tunes the supervised model (with the dev runs)."},
        {"Run group (training family)": "supervised", "Runs": sizes["supervised"], "Job": "Supervised training: faulty runs plus the normal runs with these IDs."},
        {"Run group (training family)": "dev", "Runs": sizes["dev"], "Job": "Development runs: tuning and sanity checks. Never headline results."},
        {"Run group (training family)": "testing files", "Runs": 500, "Job": "All 500 runs per fault, once, for the scorecard. They share no noise stream with any training run."},
    ])


def _lag_sentence(lag_choice):
    cpv = lag_choice[lag_choice["rule"] == "cpv"].set_index("lag")
    chosen = int(lag_choice["chosen_L"].iloc[0])
    parallel = lag_choice[lag_choice["rule"] == "parallel"].sort_values("lag")
    stop = parallel[parallel["r_new"] <= 0]["lag"]
    parallel_l = max(int(stop.iloc[0]) - 1, 0) if len(stop) else None
    return (f"The lag window is L = {chosen}, one value for DPCA and the lagged models, chosen on normal runs with the Ku et al. (1995) rule. "
            f"The choice is borderline: the last lag that added a new relation added {int(cpv.loc[chosen, 'r_new'])}, the next lag {int(cpv.loc[chosen + 1, 'r_new'])}. "
            f"Parallel analysis would have given L = {parallel_l}; published TEP work uses 2 to 3 lags.")


def section_data(inp):
    split, pca = inp["run_split"], inp["pca"]
    return f"""## Data and rules

**Data.** Rieth, Amsel, Tran and Cook (2017), simulated TEP runs, public. Each of 20 faults has 500 runs in a training file (500 samples per run, one sample every 3 minutes, fault starts after sample 20) and 500 in a testing file (960 samples, fault starts after sample 160). Fault-free files have 500 runs each. There are 52 readings per sample. A sample is labelled faulty only after the fault start (the **onset**); the samples before it in a faulty run are normal operation. The data are Rieth's, not the Braatz data most papers use.

**Why the run numbers are split into disjoint groups.** Run k of the fault-free file and of every fault in one file family share a noise stream. They are identical until the onset, and faults 3, 9 and 15 stay almost identical after it. Reusing a run number for two jobs would let a model memorise noise and make limits and tuning optimistic. So one seed-42 partition of the numbers 1-500 serves every training file, no run number serves two jobs, and a paired difference (faulty run k minus normal run k) is never a model input. All results here come from one seed-42 split; we did not repeat the split.

{md_table(groups_table(split))}

**Two tracks, never ranked together.** The *unsupervised* track (PCA, DPCA, Isolation Forest, autoencoder) learns from normal runs only. The *supervised* track (gradient boosting, 21 classes) also learns from labelled faulty runs. A supervised model is expected to look better on the 20 known faults and a real plant rarely has labels for its next fault, so each track has its own tables and the unseen-fault test below tries to show what labels buy.

**Alarm limits.** Each model's alarm limit is set on the 100 `limit` normal runs so that it false-alarms on 1% of normal samples (the headline) or 5% (secondary). Tables below use 1%. Test data never set a limit. PCA uses {pca.n_pc} PCs (90% of the variance) and no lags. {_lag_sentence(inp['lag_choice'])}

**Terms.**
- *FDR* (fault detection rate): share of post-onset samples of faulty runs that a model flags, counted per sample.
- *FAR* (false alarm rate): share of samples of fault-free test runs that a model flags.
- *T²*: how far a sample sits from normal along the patterns normal operation uses. *SPE*: how much of a sample falls outside those patterns.
- *Sustained alarm*: 3 samples in a row (9 minutes) above the limit. *Detection delay*: minutes from the onset to the third sample of the first sustained alarm after it.
- *Lag window L*: how many past samples a model sees besides the current one.

{src('phase-1/run_split.json', 'phase-2/lag_choice.csv', 'phase-2/pca.npz')}
"""


# --- detection --------------------------------------------------------------------------------------------------------------------
def best_of_family(rows):
    """{family: scorecard name} with the highest mean FDR without 3, 9, 15 inside each model family."""
    families = {"PCA / DPCA": ("pca-t2", "pca-spe", "dpca-t2", "dpca-spe"), "Isolation Forest": ("iforest-l0", "iforest-l2"),
                "Autoencoder": ("ae-l0", "ae-l2"), "Gradient boosting": ("hgb-l2",)}
    return {family: max(names, key=lambda n: rows[n]["mean_fdr_excl_3_9_15"]) for family, names in families.items()}


def section_detection(inp):
    rows = _rows(inp)
    tables = report.detection_tables(inp["scorecard"])
    tuning = inp["hgb_tuning"]
    degenerate = int((tuning["limit_1pct_limit_tune"] >= 0.9999).sum())
    stride = int(tuning["row_stride"].iloc[0])
    pca_t2, pca_spe = rows["pca-t2"], rows["pca-spe"]
    textbook_far = report.model_row(inp["scorecard"], "dpca-spe", "99% textbook")["far"]
    streak = {n: rows[n]["sustained_rate"] for n in ("dpca-t2", "iforest-l0", "iforest-l2", "hgb-l2")}
    unsup = {n: r["mean_fdr_excl_3_9_15"] for n, r in rows.items() if report.MODELS[n][1] == "unsupervised"}
    top4 = sorted(unsup, key=unsup.get, reverse=True)[:4]
    block = []
    for track, title in (("unsupervised", "Unsupervised track (normal runs only)"), ("supervised", "Supervised track (also uses labelled faults)")):
        t = tables[track]
        block.append(f"### {title}\n\n**Summary at the 1% limit.**\n\n{md_table(t['summary'])}\n\n"
                     f"**FDR on the five benchmark faults (1, 2, 4, 6, 7) and on the three hard ones (3, 9, 15).**\n\n{md_table(t['fdr'])}\n\n"
                     f"**Detection delay: median minutes over detected runs (share of runs never detected).**\n\n{md_table(t['delay'])}\n")
    hgb = rows["hgb-l2"]
    return f"""## Detection

All numbers are from the 500 testing runs per fault and the 500 fault-free testing runs, at the 1% alarm limit set on held-back normal training runs. Benchmark faults: 1 ({FAULT_NAMES[1]}), 2 ({FAULT_NAMES[2]}), 4 ({FAULT_NAMES[4]}), 6 ({FAULT_NAMES[6]}), 7 ({FAULT_NAMES[7]}); 3, 9, 15 are the near-undetectable ones. Everything is one seed-42 split of the run numbers, so no table carries a run-to-run interval.

{chr(10).join(block)}
Peak RAM is for the whole script that scored the model, not for the model alone. Machine for every row: {', '.join(inp['scorecard']['machine'].unique())}.

![Per-fault detection rate for the best model of each family](figures/detection-by-family.png)

**How to read it.**
- Every FAR is between {_f(min(r['far'] for r in rows.values()))} and {_f(max(r['far'] for r in rows.values()))}, so detection rates are compared at about equal false-alarm rate.
- PCA T² misses fault 4 ({fmt_fdr(pca_t2['fdr'][4])}) while PCA SPE catches it ({fmt_fdr(pca_spe['fdr'][4])}). T² looks at the size of normal-looking movement, SPE at broken relationships between readings.
- The top four unsupervised models ({', '.join(_label(n) for n in top4)}) are within {_f(unsup[top4[0]] - unsup[top4[-1]])} of each other in mean FDR without 3, 9, 15. With one seed-42 split we do not call a winner among them. Adding lags (L = 2) changes the autoencoder by {_f(unsup['ae-l2'] - unsup['ae-l0'])} and Isolation Forest by {_f(unsup['iforest-l2'] - unsup['iforest-l0'])}.
- The 5% limit (secondary) and the textbook 99% limit are in `scorecard.csv`. The textbook limit can miss its target: DPCA SPE with it has a FAR of {fmt_fdr(textbook_far)}, not 0.010, which is why the tables use limits set on held-back normal runs.
- Isolation Forest is the weakest unsupervised model; it misses fault 4 ({fmt_fdr(rows['iforest-l0']['fdr'][4])}) and runs it never detects reach {fmt_share(rows['iforest-l0']['never'][4])}.
- False-alarm streaks: the per-sample FAR is about 1% for every model, but alarms come in runs. The share of fault-free runs with a 3-in-a-row alarm after sample 160 is {fmt_share(streak['dpca-t2'])} for DPCA T², {fmt_share(streak['iforest-l0'])} to {fmt_share(streak['iforest-l2'])} for Isolation Forest and {fmt_share(streak['hgb-l2'])} for gradient boosting, against {fmt_share(pca_spe['sustained_rate'])} for PCA SPE. A detection delay from a model with a high streak rate is therefore less trustworthy, and a run is easily "detected" by chance.
- Faults 3, 9, 15 are not detected. The highest FDR on any of them is {fmt_fdr(max(_hard_max(r) for r in rows.values()))}, against FARs near 0.01. For PCA T², {fmt_share(1 - pca_t2['never'][3])} of fault-3 runs show a sustained alarm after the onset, and {fmt_share(pca_t2['sustained_rate'])} of fault-free runs do. Gradient boosting's FDR on them ({fmt_fdr(hgb['fdr'][3])}, {fmt_fdr(hgb['fdr'][9])}, {fmt_fdr(hgb['fdr'][15])}) is a little above its FAR ({fmt_fdr(hgb['far'])}) and still nowhere near a detection; the phase 3 leak audit found no leak.

**Caveats for the supervised row.** Gradient boosting was fitted on every 3rd row (row stride {stride}, all 50 supervised runs kept) to keep the tuning grid short; scoring, limits and tuning scores use every row. Its tuning used the development runs of all 20 faults. {degenerate} of the {len(tuning)} grid fits are degenerate (their 1% limit saturates at 1.0, so FDR reads about 0) and the selection rule discarded them. The train time ({fmt_seconds(hgb['train_time_s'])}) is one fit; the whole grid took {_f(tuning['fit_time_s'].sum(), 0)} s. A published unsupervised DPCA variant reaches a mean of 0.951 on 18 faults without 3, 9, 15 on the Braatz data (computed from `research/pca-anchors.md`), so {_f(hgb['mean_fdr_excl_3_9_15'], 2)} is not extraordinary.

Earlier figures that already show this: [PCA and DPCA per fault](../phase-2/figures/fdr-by-fault.png), [PCA SPE, Isolation Forest and autoencoder per fault](../phase-3/figures/fdr-by-fault-unsupervised.png).

{src('scorecard.csv', 'phase-3/hgb_tuning.csv', 'phase-3/models.json')} Plus `research/pca-anchors.md` for the 0.951.
"""


# --- literature -------------------------------------------------------------------------------------------------------------------
def section_literature(inp):
    frame = report.literature_frame(inp["vs_lit"], inp["scorecard"])
    table = report.literature_table(frame)
    means = {m: (frame[f"ours-{m}"].mean(), frame[f"rr-{m}"].mean()) for m in report.LITERATURE}
    worst = max(abs(a - b) for a, b in means.values())
    gaps = report.literature_gaps(frame, 0.10)
    faults = sorted({g["fault"] for g in gaps})
    pca, dpca = inp["pca"], inp["dpca"]
    return f"""## Against the literature

Rato and Reis (2013) report FDR per fault for PCA and DPCA at a 1% false-alarm limit. Ours are the same models at our 1% limit.

{md_table(table)}

**What differs, so read the gaps with care.**
- Their data are the original Braatz data: one run per fault, 960 samples, fault from sample 160. Ours are Rieth's: 500 runs per fault, so our numbers average 500 runs. Theirs carry single-run noise; a gap of a few points is not a finding.
- Their limits were set by trial to a 1% false-alarm rate on a second fault-free data set. Ours come from 100 held-back normal runs.
- Their PCA keeps 17 PCs (parallel analysis) and DPCA 3 lags and 29 PCs. Ours keeps {pca.n_pc} PCs (90% of the variance) and DPCA {dpca.lags} lags and {dpca.n_pc} PCs. Their FDR definition is not fully stated in the paper.
- Their table has 21 faults; Rieth's data have 20, so means are over the 20 shared faults.

**What agrees.** The means over the 20 faults are within {_f(worst)} of theirs for every statistic. T² misses fault 4 in both and SPE catches it; faults 3, 9, 15 sit near the false-alarm floor in both; faults 5, 10, 16, 19, 20 are weak for static PCA in both. Differences of 0.10 or more fall on faults {', '.join(map(str, faults))}, the faults where these monitors are weak or T² and SPE disagree. With one run per fault on their side and a different limit procedure on ours, we do not read them as a disagreement.

{src('phase-2/vs_literature.csv', 'scorecard.csv')} Published numbers: Rato and Reis (2013), Chemometrics and Intelligent Laboratory Systems 125:101-108, Table 2, as saved in `vs_literature.csv`; their PC and lag counts are from `research/pca-anchors.md`.
"""


# --- unseen -----------------------------------------------------------------------------------------------------------------------
def section_unseen(inp):
    us = report.unseen_summary(inp["unseen"])
    table = report.unseen_table(inp["unseen"])
    others = us["mean_without_carrier"]
    return f"""## The unseen-fault test

The supervised model was retrained once without faults 1, 8, 13, 14 and 16 (the lowest-numbered fault of each Downs and Vogel type, fixed before any results) and scored on those faults, next to the full supervised model and to PCA SPE, which never sees labels. Same 1% limit, same testing runs.

{md_table(table)}

- Without labels for them the supervised model flags {_f(us['mean']['hgb-l2-unseen'])} of samples on these five, against {_f(us['mean']['hgb-l2'])} when it saw them and {_f(us['mean']['pca-spe'])} for PCA SPE.
- **Fault 16 carries the gap.** It holds {fmt_share(us['carrier_share'])} of the unseen model's lead over PCA SPE. On faults 1, 8, 13 and 14 the two are within {_f(us['max_gap_others'])} of each other. Without fault 16 the means are {_f(others['hgb-l2-unseen'])} (unseen) and {_f(others['pca-spe'])} (PCA SPE). Fault 14 is the other one labels clearly help: the unseen model drops from {_f(inp['unseen'].set_index('fault').loc[14, 'hgb-l2'])} to {_f(inp['unseen'].set_index('fault').loc[14, 'hgb-l2-unseen'])}, no better than PCA SPE.
- Caveats. The unseen model reuses the full model's settings, which were tuned with all 20 faults, including these five, so this test is mildly optimistic for the unseen model. Both supervised models are fitted on every 3rd row. It is one seed-42 split with one retrain, so there is no interval on these numbers. Fault 16's cause is unknown (Downs and Vogel give no description).

![Unseen-fault test: supervised model with and without the five faults, and PCA SPE](../phase-3/figures/unseen-fault.png)

{src('phase-3/unseen_fault.csv', 'scorecard.csv')}
"""


# --- diagnosis --------------------------------------------------------------------------------------------------------------------
BLAME_LABELS = {"spe": "SPE (test runs)", "t2": "T² (test runs)", "rbc_spe": "RBC-SPE (test runs)", "rbc_t2": "RBC-T² (test runs)",
                "perm": "Permutation importance (dev runs, one vector per fault)"}


def blame_table(summary):
    """Formatted blame table by method, with the random-picking row."""
    rows = [{"Method": BLAME_LABELS[r.method], "Top reading in faulted area": _f(r.top1_hit, 2), "Top-5 includes faulted area": _f(r.top5_hit, 2),
             "Share of blame in faulted area": _f(r.unit_mass, 2)} for r in summary.itertuples()]
    first = summary.iloc[0]
    rows.append({"Method": "Random picking", "Top reading in faulted area": _f(first.random_top1_and_mass, 2),
                 "Top-5 includes faulted area": _f(first.random_top5, 2), "Share of blame in faulted area": _f(first.random_top1_and_mass, 2)})
    return pd.DataFrame(rows)


def _top3(blame, fault, window="post", method="spe"):
    row = blame[(blame["fault"] == fault) & (blame["method"] == method) & (blame["window"] == window)]
    return row["top3"].iloc[0]


def section_diagnosis(inp):
    summary = report.blame_summary(inp["test_blame"], inp["dev_blame"])
    n_faults = int(summary["n_faults"].iloc[0])
    spe = summary.set_index("method").loc["spe"]
    misses = report.blame_misses(inp["test_blame"])
    tb = inp["test_blame"]
    post_spe = tb[(tb["window"] == "post") & (tb["method"] == "spe")].set_index("fault")
    contrast = report.window_contrast(tb, fault=6)
    hidden = inp["test_hidden"].set_index("fault")
    ctrl = hidden.loc[0]
    hard = list(report.HARD_FAULTS)
    pressure = inp["pressure"]
    p6 = pressure.loc[6]
    reached = pressure[pressure["share_reached"] > 0]
    pressure_view = pd.DataFrame([{"Fault": str(f), "Runs reaching 2990 kPa": fmt_share(r.share_reached), "Median first sample": f"{r.median_first_sample:.0f}",
                                   "Hours after the onset": f"{r.median_first_hours_after_onset:.1f}"} for f, r in reached.iterrows()])
    window_view, wc = report.window_comparison(tb, pressure)
    like = [f for f in report.looks_like_control(inp["test_hidden"]) if f not in hard]
    spread = hidden.loc[hard, "median_max_spread_ratio"]
    dacf = hidden.loc[hard, "median_max_abs_dacf"]
    hb = tb[(tb["method"] == "spe") & (tb["window"] == "post") & tb["fault"].isin(hard)].set_index("fault")
    perm_seconds = inp["dev_blame"]["runtime_s"].iloc[0]
    miss_lines = "\n".join(f"  - Fault {f}: top-5 hit rate {_f(post_spe.loc[f, 'top5_hit'], 2)} against {_f(post_spe.loc[f, 'baseline_top5_hit'], 2)} for random picking. "
                           f"SPE's top readings are {_top3(tb, f)}." for f in misses)
    return f"""## Diagnosis

**What blame is.** An alarm says something is wrong; blame says which readings the alarm statistic points at. For each reading we take its contribution to SPE (or T²) over a window and divide it by what it contributes on normal runs. About 1 is normal; a high value marks a reading to look at. It shows where the symptoms are, not where the cause is, because contributions smear across correlated readings. We score it as *top-5 hit*: a top-5 reading sits in the plant area the fault belongs to (a fixed fault-to-area map), next to what random picking scores.

**Which method blames the right area.** Means over the {n_faults} detectable faults that have a known area ({', '.join(map(str, report.detectable_faults(inp['test_blame'])))}), whole post-onset window (these whole-window numbers can be inflated by the pressure tail described below; first-hour numbers follow it):

{md_table(blame_table(summary))}

- SPE blames best of the PCA views: its top five include the faulted area in {fmt_share(spe['top5_hit'])} of runs, {_f((spe['top5_hit'] - spe['random_top5']) * 100, 0)} points above random picking ({fmt_share(spe['random_top5'])}). T² is no better than random on that measure.
- RBC (reconstruction-based contribution) for SPE gives the same ranking as plain SPE contributions in the relative form, by algebra, so it adds nothing; they differ only where a 1% floor applies.
- The permutation row is coarse: it comes from the development runs only, with one dev vector per fault, so each fault scores 0 or 1. The 8-hour permutation pass used every 5th post-onset sample ({_f(perm_seconds, 0)} s on the laptop, over the 3-hour guide) and was not rerun.
- The gain over chance is modest. Only {fmt_share(spe['unit_mass'])} of SPE's blame lands in the right area.

![Blame hit rate by method and fault](../phase-4/figures/blame-methods.png)

_Footnote: the permutation-importance dots are dev runs, one importance vector per fault (each is 0 or 1); the PCA dots are testing runs. The chart title says "test runs" for the PCA views only._

**The pressure tail: why whole-window blame can mislead.** In some faults reactor pressure (xmeas_7) climbs to about 3000 kPa gauge after the onset and the readings then sit flat. In fault 6 run 1 (see [the operator report](operator-fault6.md)) 24 of 52 readings stop changing from 13.8 h on. 3000 kPa is {SHUTDOWN}, so this is probably the simulator's pressure shutdown regime, but we have not checked the cause. A whole-window average then mostly measures that frozen stretch. Share of the 100 testing runs per fault whose reactor pressure reaches 2990 kPa or more after the onset, and the median first sample (onset at sample 160):

{md_table(pressure_view)}

The other {20 - len(reached)} faults have no run that reaches it. Among the 12 detectable faults only fault 6 is affected in most runs, faults 12 and 13 in a minority, and fault 18 has no expected area, so it is not in the blame means. SPE top-5 hit, first hour beside whole window, for the 12 detectable faults:

{md_table(window_view)}

- Over all 12 faults the means are {_f(wc['mean_first_hour'], 2)} (first hour) and {_f(wc['mean_whole'], 2)} (whole window), against {_f(spe['random_top5'], 2)} for random picking, so the lead over random does not depend on the window. Without fault 6 the means are {_f(wc['mean_first_hour_without_affected'], 2)} and {_f(wc['mean_whole_without_affected'], 2)}.
- Top-1 hit, the share of blame in the faulted area, and the heatmap below (median whole-window contribution) are more exposed to the tail than top-5 hit. Treat their whole-window values for fault 6 (and fault 18) as inflated by it.
- The window gap also goes both ways without the tail: fault 1 scores better over the whole window, faults 2, 8 and 10 worse, and none of them reaches the pressure level. We do not explain that here.

**The fault 6 lesson: first hour against whole window.** For fault 6 (A feed loss) the top SPE reading is a feed reading in {fmt_share(contrast['first_hour_top1'])} of the 100 testing runs in the first hour after the onset (the median top three are {_top3(tb, 6, 'first_hour')}), and in {fmt_share(contrast['post_top1'])} over the whole post-onset window (median top three: {_top3(tb, 6)}). The main reason the two differ is the pressure tail: reactor pressure reaches the level in {fmt_share(p6['share_reached'])} of the fault 6 runs, at a median of sample {p6['median_first_sample']:.0f} ({p6['median_first_hours_after_onset']:.1f} h after the onset), so most of the 800 post-onset samples come after it. Downstream readings also move before that point, so "the symptoms spread downstream" is part of the story, not all of it. Both windows have a feed reading among the top five in {fmt_share(contrast['post_top5'])} of runs. The window, not the method, decides what the plot says.

**Where blame misses.** SPE's top-5 hit rate is at or below 0.05 on faults {', '.join(map(str, misses))}:

{miss_lines}

These are results about contribution plots (symptoms, not causes), not a fault in the method.

**Why faults 3, 9 and 15 hide.** Inside one run, we compare the 160 samples before the fault start with the 800 after, on three statistics: the largest shift of any reading (in standard deviations of normal runs), the largest change in spread and the largest change in autocorrelation. The control is 100 fault-free testing runs that share no noise stream with the faulty ones.

{md_table(report.hidden_table(inp['test_hidden'], [0, 3, 9, 15, 14, 19]))}

- Faults 3, 9, 15 sit at the control on all three: shift {_f(hidden.loc[hard, 'median_max_abs_shift'].min(), 2)} to {_f(hidden.loc[hard, 'median_max_abs_shift'].max(), 2)} against {_f(ctrl['median_max_abs_shift'], 2)}; spread {_f(spread.min(), 2)} to {_f(spread.max(), 2)} against {_f(ctrl['median_max_spread_ratio'], 2)}; autocorrelation {_f(dacf.min(), 2)} to {_f(dacf.max(), 2)} against {_f(ctrl['median_max_abs_dacf'], 2)}. There is no change inside the run for any monitor to find, and their blame hit rates ({_f(hb.loc[3, 'top5_hit'], 2)}, {_f(hb.loc[9, 'top5_hit'], 2)}, {_f(hb.loc[15, 'top5_hit'], 2)}) are near chance and mean nothing.
- Caution: the headline check is mean shift, and faults {', '.join(map(str, like))} also match the control on it. Faults 14 and 19 are the clearest example: fault 14 is easy to detect and shows no mean shift ({_f(hidden.loc[14, 'median_max_abs_shift'], 2)} sd), but its spread ratio is {_f(hidden.loc[14, 'median_max_spread_ratio'], 2)} against {_f(ctrl['median_max_spread_ratio'], 2)}. So mean shift alone proves nothing; for 3, 9, 15 all three statistics are at the control level, and that is what supports "hidden".

![Largest change in any reading per fault](../phase-4/figures/hidden-faults.png)

See also [the blame heatmap](../phase-4/figures/blame-heatmap.png) (median whole-window contribution; the fault 6 and fault 18 rows are inflated by the pressure tail).

{src('phase-4/test_blame.csv', 'phase-4/dev_blame.csv', 'phase-4/test_hidden.csv', 'phase-4/README.md')} Pressure table: [`pressure_tail.csv`](pressure_tail.csv), computed from the 100 fixed testing runs of each fault.
"""


# --- the assistant ----------------------------------------------------------------------------------------------------------------
def section_assistant(inp):
    cmp = inp["assistant_cmp"].set_index("metric")
    table = report.assistant_table(inp["assistant_cmp"])
    k1, n1 = report.repeat_agreement(inp["diag_v1"])
    k11, n11 = report.repeat_agreement(inp["diag_v11"])
    sessions = len(inp["diag_v11"])
    used = int(inp["diag_v11"]["used_classify"].sum())
    cases = int(inp["diag_v1"]["case_id"].nunique())
    costs = []
    for label, cost in (("v1", inp["cost_v1"]), ("v1.1", inp["cost_v11"])):
        costs.append(f"{label} ${cost[0]:.2f} for {cost[1]} sessions" if cost else f"{label} not available (transcripts are not in git)")
    head = cmp.loc["headline"]
    lo1, hi1 = _wilson(int(head["v1_k"]), int(head["v1_n"]))
    lo11, hi11 = _wilson(int(head["v1.1_k"]), int(head["v1.1_n"]))
    overlap = "do not overlap" if lo11 > hi1 or lo1 > hi11 else "overlap"
    repeats = int(inp["diag_v1"]["repeat"].max())
    return f"""## The assistant

The assistant is Claude with one tool server that reads a replayed plant history for one case. It sees the 52 readings, a PCA alarm log and public plant notes. It answers with a fixed block: status (fault, normal or can't tell), plant area, up to three disturbance guesses (IDV numbers) and evidence. It is scored on {cases} frozen cases from the testing runs (3 runs per fault and 10 normal, neutral IDs, so the fault, run number and onset are hidden), each in a fresh session with no file or web access. *Unit accuracy* is the share of fault cases (not 3, 9, 15 or 16-20) where it says "fault" and names the right plant area.

{md_table(table)}

- **Read v1.1 for what it is.** v1.1 adds `classify_fault`, a supervised closed-set classifier (gradient boosting trained on faults 1-20 plus normal). All {used} of {sessions} v1.1 test sessions used it, so there is no with-and-without comparison. The gain from {int(head['v1_k'])} to {int(head['v1.1_k'])} of {int(head['v1_n'])} is "a trained classifier helps", not better reasoning, and v1.1 is not a reasoning-only score.
- **Wide intervals.** With {int(head['v1_n'])} headline cases the 95% intervals above are wide. The intervals for v1 and v1.1 {overlap}, and each fault has only three cases.
- **Baselines.** The no-LLM baseline ({int(cmp.loc['baseline', 'v1_k'])} of {int(cmp.loc['baseline', 'v1_n'])}: unit of the top SPE reading) is below chance ({int(cmp.loc['chance', 'v1_k'])} of {int(cmp.loc['chance', 'v1_n'])}: always the most common unit), so the SPE reading alone is a poor guide to the plant area.
- **Faults 3, 9, 15.** The assistant answered "normal" or "can't tell" on all {int(cmp.loc['hard', 'v1_n'])}. That is the right answer for a fault with no visible change, and it is not a detection.
- **Repeats.** {n1} cases were run {repeats} times. v1: {k1} of {n1} cases agree on status and plant area; v1.1: {k11} of {n11}. Headline numbers use the first run of each case.
- **Cost.** {'; '.join(costs)} (API-equivalent cost reported by the CLI).
- **Memory of the papers.** Claude may remember the published TEP papers and their fault descriptions; no setting controls that, so a result here is not proof of reasoning from the data. The frozen prompt also mildly pushes toward finding a fault, which can flatter the fault cases.

{src('phase-3/assistant_v1_vs_v1.1.csv', 'phase-2b/diagnosis_test.csv', 'phase-2b/diagnosis_test_v1.1.csv')} Cost: result lines of `results/phase-2b/transcripts_test*/` (local files, not in git).
"""


# --- what we did not do -----------------------------------------------------------------------------------------------------------
def section_limits(inp):
    rows = _rows(inp)
    tuning = inp["hgb_tuning"]
    stride = int(tuning["row_stride"].iloc[0])
    degenerate = int((tuning["limit_1pct_limit_tune"] >= 0.9999).sum())
    perm_h = inp["dev_blame"]["runtime_s"].iloc[0] / 3600
    cmp = inp["assistant_cmp"].set_index("metric")
    lo, hi = _wilson(int(cmp.loc["headline", "v1_k"]), int(cmp.loc["headline", "v1_n"]))
    return f"""## What we did not do and what is uncertain

- **One seed-42 split.** One partition of the run numbers, one fit per model, no repeat over splits. No detection number has an interval; differences of a few points between models are not findings.
- **Rieth data, not Braatz data.** Most papers use the original Braatz data (one run per fault). Same faults, different noise and run counts, so compare patterns, not exact percentages.
- **Gradient boosting stride.** Fitted on every 3rd row (stride {stride}) to keep the grid short. {degenerate} of {len(tuning)} grid fits are degenerate (limit saturates at 1.0) and discarded by the rule. It was tuned with all 20 faults, so the unseen-fault test is mildly optimistic.
- **Few assistant cases.** 36 headline cases give wide intervals (v1: {_f(lo, 2)} to {_f(hi, 2)}). Claude may remember the TEP papers. v1.1 includes a supervised classifier.
- **Pressure tail.** Whole-window blame for fault 6 is dominated by samples after reactor pressure reaches about 3000 kPa ({SHUTDOWN}); first-hour numbers are the fairer view. Faults 12, 13 and 18 reach it too (see the Diagnosis section).
- **Permutation importance.** One dev vector per fault, every 5th post-onset sample, one process; the pass took about {perm_h:.0f} hours, over the guide, and was not rerun.
- **False-alarm streaks.** DPCA T² has a 3-in-a-row rate of {fmt_share(rows['dpca-t2']['sustained_rate'])} on fault-free runs at the 1% limit, so its delays are less trustworthy.
- **Hidden-fault check.** Its headline is mean shift, which does not separate faults 14 and 19 from the control; spread and autocorrelation do.
- **Not done.** No new models or testing-file runs in this phase; no fault 21 (not in Rieth's data); no repeated splits or confidence intervals on the detection tables; no deep learning beyond a small scikit-learn autoencoder; no cloud runs.
"""


def section_reproduce():
    return """## Reproduce it

Run from the project folder, in this order. Python is the project venv.

```
.venv/Scripts/python scripts/prepare_data.py          # download and convert the four files to parquet
.venv/Scripts/python -m src.split                     # the seed-42 run groups
.venv/Scripts/python scripts/mspc_dev.py              # PCA / DPCA on development runs (lag choice)
.venv/Scripts/python scripts/mspc_final.py            # PCA / DPCA on the testing files, scorecard rows
.venv/Scripts/python scripts/ml_dev.py                # Isolation Forest, autoencoder, gradient boosting tuning
.venv/Scripts/python scripts/ml_final.py              # the same on the testing files, unseen-fault test
.venv/Scripts/python scripts/assistant_build.py       # case sets and historians (the test set is frozen, it refuses a rebuild)
.venv/Scripts/python scripts/assistant_eval.py --version v1.1 --set test   # assistant sessions (already run; needs the Claude CLI and costs money)
.venv/Scripts/python scripts/assistant_score.py --set test --version v1.1
.venv/Scripts/python scripts/diagnose_dev.py          # permutation importance on dev runs (about 8 hours)
.venv/Scripts/python scripts/diagnose_final.py        # blame and hidden-fault tables on the testing files
.venv/Scripts/python scripts/make_report.py           # these two reports and their charts
.venv/Scripts/python -m pytest -q                     # the tests
```

`make_report.py` only reads saved results files, except for one fault 6 testing run it reads from the parquet file and scores with the saved PCA model. It overwrites everything in `results/report/`.
"""


def readme_text():
    return """# Phase 5: reports

Four files and a folder, all generated by one script from saved results files. Do not edit them by hand.

| File | What it is |
|---|---|
| `report.md` | The comparison report: findings, data and rules, detection, literature, unseen-fault test, diagnosis, the assistant, limits, reproduce. |
| `operator-fault6.md` | An operator-style shift handover for one fault 6 (A feed loss) testing run: the first run ID in `phase-4/test_runs.json`. |
| `pressure_tail.csv` | Per fault: share of the 100 fixed testing runs whose reactor pressure reaches 2990 kPa after the onset, and the median first sample. |
| `figures/` | `detection-by-family.png`, `operator-fault6.png`, `operator-fault6-readings.png`. The report also links the figures of phases 2-4 where they already say it. |

## Rebuild

```
.venv/Scripts/python scripts/make_report.py
```

One command. It overwrites this folder, and records runtime, peak RAM and machine at the end of each report. It reads only saved results files, except for one fault 6 run it reads from `data/processed/TEP_Faulty_Testing.parquet` and scores with `phase-2/pca.npz`. Tests: `.venv/Scripts/python -m pytest -q`.

## Which results files feed each section

| Section | Files (under `results/`) |
|---|---|
| What this is and what to take from it | all of the files below |
| Data and rules | `phase-1/run_split.json`, `phase-2/lag_choice.csv`, `phase-2/pca.npz` |
| Detection | `scorecard.csv`, `phase-3/hgb_tuning.csv`, `phase-3/models.json`; `research/pca-anchors.md` for one published mean |
| Against the literature | `phase-2/vs_literature.csv`, `scorecard.csv`, `phase-2/pca.npz`, `phase-2/dpca.npz` |
| The unseen-fault test | `phase-3/unseen_fault.csv` |
| Diagnosis | `phase-4/test_blame.csv`, `phase-4/dev_blame.csv`, `phase-4/test_hidden.csv`, `report/pressure_tail.csv` (from the testing parquet) |
| The assistant | `phase-3/assistant_v1_vs_v1.1.csv`, `phase-2b/diagnosis_test.csv`, `phase-2b/diagnosis_test_v1.1.csv`, `phase-2b/transcripts_test*/` (cost; local, not in git) |
| What we did not do | `phase-3/hgb_tuning.csv`, `phase-4/dev_blame.csv`, `phase-3/assistant_v1_vs_v1.1.csv`, `scorecard.csv` |
| operator-fault6.md | `phase-4/test_runs.json`, `phase-4/top_readings_6.csv`, `phase-4/test_blame.csv`, `scorecard.csv`, `phase-2/pca.npz`, the fault 6 testing run |

Terms and rules: `CONTEXT.md`, `docs/adr/`. Sources for the numbers are also listed under each section of the report.
"""


def build_report(inp, how_made):
    """The whole comparison report as one Markdown string; `how_made` is the closing 'How this was made' line."""
    parts = ["# TEP fault detection: comparison report\n",
             section_summary(inp), section_data(inp), section_detection(inp), section_literature(inp), section_unseen(inp),
             section_diagnosis(inp), section_assistant(inp), section_limits(inp), section_reproduce(), f"## How this was made\n\n{how_made}\n"]
    return "\n".join(parts)
