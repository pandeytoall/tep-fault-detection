"""Phase 4 diagnosis: which readings the PCA monitor and the supervised model blame, and why faults 3, 9 and 15 hide.

Rules: .scratch/phase-4/spec.md. Everything here is offline analysis of whole runs: the onset is known, the assistant never
sees it. Blame is measured in the scaled space of the saved PCA model (31 PCs, L = 0).
- Methods (per run and window, relative to the `fit` normal runs, see src/contributions.py): `spe`, `t2` (complete
  decompositions), `rbc_spe`, `rbc_t2` (reconstruction-based contributions) and `perm` (permutation importance of the
  supervised model, one vector per fault).
- Windows: `post` = every post-onset sample of a run, `first_hour` = its first 20 post-onset samples.
- Metrics against the expected scoring unit (src/cases.py): top1_hit, top5_hit, unit_mass, smear; plus the random baseline.
- Before/after: standardised mean shift, spread ratio and change in lag-1 autocorrelation of each reading inside one run.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from src import cases, contributions
from src.variables import VARIABLE_INFO, scoring_unit

PCA_METHODS = ("spe", "t2", "rbc_spe", "rbc_t2")
PERM = "perm"
METHODS = PCA_METHODS + (PERM,)
METHOD_LABELS = {"spe": "SPE", "t2": "T²", "rbc_spe": "RBC-SPE", "rbc_t2": "RBC-T²", "perm": "Permutation importance"}
WINDOWS = ("post", "first_hour")
FIRST_HOUR = 20                    # samples after the onset in the first-hour window
RBC_MIN_DENOMINATOR = 1e-9
TOP_K = 5
BENCHMARK_FAULTS = (1, 2, 4, 6, 7)
HIDDEN_FAULTS = (3, 9, 15)
UNKNOWN_FAULTS = (16, 17, 18, 19, 20)
HIDDEN_NOTE = "near the false-alarm level; blame is mostly noise"
UNKNOWN_NOTE = "unknown cause"
SHIFT_STOP_FACTOR = 3.0            # faults 3/9/15: stop if the largest median |shift| is above this x the fault-free control's
SAME_AS_CONTROL = 1.5              # verdict: at most this x the control's effect size = "looks like the control"


# --- reconstruction-based contributions --------------------------------------------------------------------------
def _rbc(matrix, z):
    """rbc_j = (xi_j' M x)^2 / (xi_j' M xi_j) for every row x of `z`, M symmetric; 0 where the denominator is below 1e-9."""
    denominator = np.diag(matrix)
    ok = denominator >= RBC_MIN_DENOMINATOR
    out = np.zeros(z.shape)
    out[:, ok] = (z @ matrix)[:, ok] ** 2 / denominator[ok]
    return out


def rbc_spe(monitor, z):
    """(rows x variables) reconstruction-based contributions to SPE of scaled rows `z` (Alcala & Qin 2011).

    With C~ = I - P P', rbc_j = (xi_j' C~ x)^2 / (xi_j' C~ xi_j): how much SPE falls if reading j alone is reconstructed.
    A reading whose denominator is below 1e-9 (fully explained by the PCs) gets 0.
    Unlike the complete decomposition in contributions.spe_contributions, RBC values do NOT sum to the statistic.
    """
    p = monitor.loadings[:, :monitor.n_pc]
    return _rbc(np.eye(p.shape[0]) - p @ p.T, np.asarray(z, dtype=float))


def rbc_t2(monitor, z):
    """(rows x variables) reconstruction-based contributions to T2: D = P Lambda^-1 P', rbc_j = (xi_j' D x)^2 / (xi_j' D xi_j).

    Same guard as rbc_spe. These values do NOT sum to T2, unlike contributions.t2_contributions.
    """
    p = monitor.loadings[:, :monitor.n_pc]
    return _rbc(p @ np.diag(1.0 / monitor.eigvals[:monitor.n_pc]) @ p.T, np.asarray(z, dtype=float))


ROW_CONTRIBUTIONS = {"spe": contributions.spe_contributions, "t2": contributions.t2_contributions,
                     "rbc_spe": rbc_spe, "rbc_t2": rbc_t2}


def normal_references(monitor, frame):
    """{method: mean contribution per variable over the rows of a normal frame} for the four PCA methods."""
    z = contributions.scale(monitor, frame)
    return {m: fn(monitor, z).mean(axis=0) for m, fn in ROW_CONTRIBUTIONS.items()}


def save_rbc_reference(path, references, machine, n_runs):
    """Write the two RBC references (the `fit` normal-run means) to an .npz with the machine and run count."""
    np.savez(path, rbc_spe=references["rbc_spe"], rbc_t2=references["rbc_t2"], machine=np.array(machine),
             n_runs=int(n_runs))


def load_references(pca_path, rbc_path):
    """All four references: spe and t2 from the PCA model's .npz, rbc_spe and rbc_t2 from the RBC reference .npz."""
    stored = contributions.load_reference(pca_path)
    with np.load(rbc_path) as f:
        return {"spe": stored["spe"], "t2": stored["t2"], "rbc_spe": f["rbc_spe"], "rbc_t2": f["rbc_t2"]}


# --- per-run blame ---------------------------------------------------------------------------------------------------
def window_masks(sample, onset):
    """{'post': samples after the onset, 'first_hour': the first FIRST_HOUR of them} as boolean arrays."""
    sample = np.asarray(sample)
    post = sample > onset
    return {"post": post, "first_hour": post & (sample <= onset + FIRST_HOUR)}


def blame_vectors(monitor, references, frame, onset):
    """(run IDs, {window: {method: (runs x variables) relative contribution}}) for a frame of whole runs.

    Per run and window: the mean contribution of every reading over the window's samples, divided by the `fit` normal
    mean of that reading (contributions.relative, floor 1% of the mean over readings).
    """
    frame = frame.sort_values(["simulationRun", "sample"], kind="stable")
    z = contributions.scale(monitor, frame)
    rows = {m: fn(monitor, z) for m, fn in ROW_CONTRIBUTIONS.items()}
    run_ids = frame["simulationRun"].to_numpy()
    out = {}
    for window, mask in window_masks(frame["sample"].to_numpy(), onset).items():
        out[window] = {}
        for m, per_row in rows.items():
            mean = pd.DataFrame(per_row[mask]).groupby(run_ids[mask]).mean()
            out[window][m] = contributions.relative(mean.to_numpy(), references[m])
    return sorted(int(r) for r in np.unique(run_ids)), out


# --- the expected unit ------------------------------------------------------------------------------------------------
def expected_unit(fault):
    """Scoring unit a fault belongs to (src/cases.py), or None for faults without one (16-20, normal)."""
    unit = cases.case_unit(fault)
    return None if unit in ("unknown", "none") else unit


def unit_mask(fault, tags):
    """Boolean array over `tags`: the reading is in the fault's expected scoring unit (None when the fault has none)."""
    unit = expected_unit(fault)
    return None if unit is None else np.array([scoring_unit(t) == unit for t in tags])


def fault_group(fault):
    """'benchmark' (1, 2, 4, 6, 7), 'hidden' (3, 9, 15), 'unknown cause' (16-20) or 'the rest'."""
    if fault in BENCHMARK_FAULTS:
        return "benchmark"
    if fault in HIDDEN_FAULTS:
        return "hidden"
    return "unknown cause" if fault in UNKNOWN_FAULTS else "the rest"


def fault_note(fault):
    """Label printed beside a fault's blame: near-the-false-alarm-level for 3/9/15, unknown cause for 16-20, else ''."""
    return HIDDEN_NOTE if fault in HIDDEN_FAULTS else UNKNOWN_NOTE if fault in UNKNOWN_FAULTS else ""


# --- blame metrics ----------------------------------------------------------------------------------------------------
def random_baseline(in_unit, k=TOP_K):
    """(share of readings in the expected unit, 1 - (1 - share)^k): what picking readings at random scores on unit_mass and
    top5_hit (the with-replacement approximation of the spec; it slightly overstates chance, so it is the harder bar)."""
    share = float(np.mean(in_unit))
    return share, 1.0 - (1.0 - share) ** k


def blame_metrics(vectors, in_unit=None, k=TOP_K):
    """Blame metrics of a (runs x readings) block of blame vectors (negative values count as zero).

    - top1_hit: share of runs whose top reading is in the expected unit;
    - top5_hit: share of runs with at least one top-5 reading in the expected unit;
    - unit_mass: mean over runs of the share of the clipped blame that falls in the expected unit;
    - smear: mean over runs of the share of the clipped blame held by the top 5 readings (high = concentrated).
    Without `in_unit` (faults 16-20) only smear is returned. A run with no positive blame counts 0 for unit_mass and smear.
    """
    v = np.clip(np.atleast_2d(np.asarray(vectors, dtype=float)), 0.0, None)
    order = np.argsort(-v, axis=1, kind="stable")[:, :k]
    total = v.sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    out = {"smear": float(np.mean(np.take_along_axis(v, order, axis=1).sum(axis=1) / safe))}
    if in_unit is not None:
        in_unit = np.asarray(in_unit, dtype=bool)
        out.update(top1_hit=float(np.mean(in_unit[order[:, 0]])), top5_hit=float(np.mean(in_unit[order].any(axis=1))),
                   unit_mass=float(np.mean(v[:, in_unit].sum(axis=1) / safe)))
    return out


def top_share(vectors, tags, wanted, k=1):
    """Share of runs (rows of `vectors`) whose top `k` readings include at least one of the tags in `wanted`."""
    order = np.argsort(-np.atleast_2d(np.asarray(vectors, dtype=float)), axis=1, kind="stable")[:, :k]
    return float(np.mean(np.isin(order, [list(tags).index(t) for t in wanted]).any(axis=1)))


def test_run_ids(n=100, offset=0, n_runs=500, seed=42):
    """Sorted run IDs of the final pass: runs `offset` to `offset + n` of a seed-42 shuffle of 1..n_runs (the same IDs for every
    fault). offset 0 gives the faulty runs; offset 100 gives the fault-free control runs, which share no ID (so no noise stream)
    with them."""
    return sorted(int(r) for r in np.random.default_rng(seed).permutation(np.arange(1, n_runs + 1))[offset:offset + n])


def dev_control_ids(limit_ids, n=20, seed=42):
    """Sorted IDs of `n` fault-free training runs drawn (seed 42) from the `limit` group, for the before/after control. They share no
    noise stream with the dev runs, so the control is not paired with the faulty runs it is compared with."""
    return sorted(int(r) for r in np.random.default_rng(seed).choice(np.asarray(limit_ids), size=n, replace=False))


def top_names(vector, tags, n=3):
    """Names (D&V, with the tag) of the `n` largest entries of one blame vector, largest first."""
    return [f"{VARIABLE_INFO[tags[j]].name} ({tags[j]})" for j in np.argsort(-np.asarray(vector), kind="stable")[:n]]


def blame_rows(fault, vectors_by_method, tags, windows=WINDOWS):
    """List of dict rows (one per method and window) with the metrics, the baseline and the top 3 names of a fault.

    `vectors_by_method` is {window: {method: (runs x readings)}}; methods missing from a window are skipped.
    """
    in_unit = unit_mask(fault, tags)
    unit = expected_unit(fault)
    share, base_top5 = random_baseline(in_unit) if in_unit is not None else (np.nan, np.nan)
    rows = []
    for window in windows:
        for method, block in vectors_by_method.get(window, {}).items():
            block = np.atleast_2d(block)
            metrics = blame_metrics(block, in_unit)
            rows.append({"fault": fault, "group": fault_group(fault), "note": fault_note(fault), "method": method,
                         "window": window, "expected_unit": unit or "", "n_runs": len(block),
                         "top1_hit": metrics.get("top1_hit", np.nan), "top5_hit": metrics.get("top5_hit", np.nan),
                         "unit_mass": metrics.get("unit_mass", np.nan), "smear": metrics["smear"],
                         "baseline_unit_share": share, "baseline_top5_hit": base_top5,
                         "top3": "; ".join(top_names(np.median(block, axis=0), tags, 3))})
    return rows


def group_summary(blame):
    """Mean of the four metrics and the baselines per method and fault group for one window of a blame table."""
    columns = ["top1_hit", "top5_hit", "unit_mass", "smear", "baseline_unit_share", "baseline_top5_hit"]
    out = blame.groupby(["group", "method"], sort=False)[columns].mean()
    out["n_faults"] = blame.groupby(["group", "method"], sort=False)["fault"].nunique()
    return out.reset_index()


def top_readings(fault, medians, tags, extra=None, n=10):
    """Frame of the top `n` readings by the SPE median relative contribution, with the other methods' values and ranks.

    `medians` is {method: vector over readings}; `extra` (optional) holds more vectors, shown as given (e.g. permutation
    importance). Columns: rank, tag, name, unit, plant_unit, scoring_unit, one value and one rank column per method, note.
    """
    vectors = {**medians, **(extra or {})}
    ranks = {m: pd.Series(-np.asarray(v, dtype=float)).rank(method="first").astype(int).to_numpy() for m, v in vectors.items()}
    rows = []
    for j in np.argsort(-np.asarray(medians["spe"]), kind="stable")[:n]:
        info = VARIABLE_INFO[tags[j]]
        row = {"fault": fault, "rank": len(rows) + 1, "tag": tags[j], "name": info.name, "unit": info.unit,
               "plant_unit": info.plant_unit, "scoring_unit": scoring_unit(tags[j])}
        row.update({f"{m}": float(v[j]) for m, v in vectors.items()})
        row.update({f"rank_{m}": int(r[j]) for m, r in ranks.items()})
        row["note"] = fault_note(fault)
        rows.append(row)
    return pd.DataFrame(rows)


def analyse_fault(monitor, references, frame, fault, onset):
    """Everything one fault's runs give: {'runs', 'vectors' ({window: {method: runs x readings}}), 'blame' (blame_rows for the
    four PCA methods), 'change' (run_change_stats)}. `frame` holds whole runs of one fault (or of fault-free runs, fault 0)."""
    runs, vectors = blame_vectors(monitor, references, frame, onset)
    return {"runs": runs, "vectors": vectors, "blame": blame_rows(fault, vectors, monitor.variables),
            "change": run_change_stats(frame, onset, monitor)}


# --- permutation importance (the supervised view) -------------------------------------------------------------------
def sum_over_lags(importances, n_lags):
    """Importance per reading from per-input importances laid out as mspc.lagged_rows does: [x(k), x(k-1), ...], so input
    c is reading c % n_readings at lag c // n_readings. Sums the `n_lags` lags of each reading."""
    importances = np.asarray(importances, dtype=float)
    return importances.reshape(n_lags, -1).sum(axis=0)


def _accuracy_on(fault):
    def accuracy(y_true, y_pred):
        sel = np.asarray(y_true) == fault
        return float(np.mean(np.asarray(y_pred)[sel] == fault))
    return make_scorer(accuracy)


def permutation_blame(estimator, x, y, faults, n_lags, n_repeats=5, seed=42, n_jobs=1):
    """{fault: importance per reading}: sklearn permutation importance on the pooled rows `x` (lagged inputs) with the 21-class
    labels `y`; each column is permuted over all rows, then only that fault's rows are scored (accuracy of the class
    prediction). The importance of an input is the mean accuracy drop over `n_repeats`; the lags of a reading are summed."""
    scoring = {str(int(f)): _accuracy_on(int(f)) for f in faults}
    result = permutation_importance(estimator, x, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs)
    return {int(f): sum_over_lags(result[str(int(f))]["importances_mean"], n_lags) for f in faults}


# --- before / after inside one run ------------------------------------------------------------------------------------
def lag1_autocorrelation(values):
    """Lag-1 autocorrelation of every column of an (n x p) block: sum of consecutive centred products over the sum of squares."""
    centred = np.asarray(values, dtype=float) - np.mean(values, axis=0)
    denominator = (centred ** 2).sum(axis=0)
    numerator = (centred[1:] * centred[:-1]).sum(axis=0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def before_after(values, sample, onset, sd):
    """Per reading, how the post-onset window of one run differs from its pre-onset window.

    Returns {'shift': (mean_post - mean_pre) / sd, 'spread': sd_post / sd_pre, 'dacf': acf1_post - acf1_pre}. `sd` is the reading's
    standard deviation on the `fit` normal runs. Pre = samples up to the onset, post = samples after it. The statistics need
    both windows to hold at least 3 samples; a constant pre window gives spread inf (or nan if post is constant too).
    """
    values, sample = np.asarray(values, dtype=float), np.asarray(sample)
    pre, post = values[sample <= onset], values[sample > onset]
    if min(len(pre), len(post)) < 3:
        raise ValueError("before/after needs at least 3 samples in each window")
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = post.std(axis=0, ddof=1) / pre.std(axis=0, ddof=1)
    return {"shift": (post.mean(axis=0) - pre.mean(axis=0)) / sd, "spread": spread,
            "dacf": lag1_autocorrelation(post) - lag1_autocorrelation(pre)}


def run_change_stats(frame, onset, monitor):
    """One row per run of a frame of whole runs: the largest |shift| over readings and which reading, how many readings
    shift by more than 1 sd, the largest spread ratio (in either direction) and the largest |change| of the lag-1 autocorrelation."""
    rows = []
    for run, one in frame.sort_values(["simulationRun", "sample"], kind="stable").groupby("simulationRun"):
        change = before_after(one[monitor.variables].to_numpy(dtype=float), one["sample"].to_numpy(), onset, monitor.std)
        shift = np.abs(change["shift"])
        spread = np.where(change["spread"] >= 1, change["spread"], 1.0 / change["spread"])
        rows.append({"run": int(run), "max_abs_shift": float(shift.max()), "top_reading": monitor.variables[int(shift.argmax())],
                     "n_shift_gt_1sd": int((shift > 1.0).sum()), "max_spread_ratio": float(np.nanmax(spread)),
                     "max_abs_dacf": float(np.abs(change["dacf"]).max())})
    return pd.DataFrame(rows)


def verdict(effect_ratio):
    """Plain verdict from an effect size relative to the fault-free control's."""
    if effect_ratio <= SAME_AS_CONTROL:
        return "looks like the fault-free control"
    if effect_ratio <= SHIFT_STOP_FACTOR:
        return "somewhat above the control"
    return "clear shift, well above the control"


def hidden_table(per_run, control):
    """One row per fault (and fault 0 for the control) from per-run frames: {fault: run_change_stats frame}, control frame.

    Columns: fault, group, n_runs, medians of the four statistics, the most common top reading, effect_vs_control (the fault's
    median largest |shift| over the control's) and a plain verdict. Fault 0 is the fault-free control.
    """
    def row(fault, stats):
        return {"fault": fault, "group": "control" if fault == 0 else fault_group(fault), "n_runs": len(stats),
                "median_max_abs_shift": float(stats["max_abs_shift"].median()),
                "median_n_shift_gt_1sd": float(stats["n_shift_gt_1sd"].median()),
                "median_max_spread_ratio": float(stats["max_spread_ratio"].median()),
                "median_max_abs_dacf": float(stats["max_abs_dacf"].median()),
                "most_common_top_reading": stats["top_reading"].mode().iloc[0]}
    rows = [row(0, control)] + [row(f, s) for f, s in sorted(per_run.items())]
    table = pd.DataFrame(rows)
    floor = float(table.loc[table["fault"] == 0, "median_max_abs_shift"].iloc[0])
    table["effect_vs_control"] = table["median_max_abs_shift"] / floor
    table["verdict"] = ["control (the noise floor)" if f == 0 else verdict(r) for f, r in
                        zip(table["fault"], table["effect_vs_control"])]
    table["note"] = [fault_note(f) for f in table["fault"]]
    return table


def hidden_stop_messages(table):
    """Messages for faults 3/9/15 whose effect size is above SHIFT_STOP_FACTOR x the control's (empty = nothing to report)."""
    bad = table[table["fault"].isin(HIDDEN_FAULTS) & (table["effect_vs_control"] > SHIFT_STOP_FACTOR)]
    return [f"fault {int(r.fault)}: largest median |shift| {r.median_max_abs_shift:.2f} sd is {r.effect_vs_control:.1f}x the control's"
            for r in bad.itertuples()]
