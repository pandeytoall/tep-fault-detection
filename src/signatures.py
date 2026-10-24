"""Fault signatures: which variables a fault moves, and the paired IDV numbering check."""
from collections import Counter

import pandas as pd

from src.data import VARIABLES
from src.variables import VARIABLE_INFO

# Hypotheses from the IDV descriptions (spec phase 1): fault -> variable that should be among the top 3 shifts.
# Kept unchanged when the method moved from before/after shifts to the paired first-hour difference.
IDV_EXPECTED = {
    4: "xmv_10",   # reactor cooling water inlet temperature step -> reactor cooling water flow
    5: "xmv_11",   # condenser cooling water inlet temperature step -> condenser cooling water flow
    6: "xmeas_1",  # A feed loss -> A feed
    7: "xmeas_4",  # C header pressure loss -> A and C feed
}
IDV_PASS_SHARE = 0.8
IDV_WINDOW = 20  # first 20 post-onset samples = the first hour
# Plant units whose top-1 variable counts as a "unit match" when the expected tag itself fails. Added AFTER seeing
# the data (fault 5 and 7 did not show the expected tag): a post-hoc softening, reported as its own verdict.
IDV_ACCEPTABLE_UNITS = {4: {"reactor"}, 5: {"condenser", "separator"}, 6: {"feed"}, 7: {"feed", "reactor"}}


def rank_shifts(run, variables, onset):
    """|mean after onset - mean before| / std before, sorted descending.

    'before' = samples <= onset of the same run (ddof=1); variables with a constant
    pre-onset stretch are left out (shift undefined).
    """
    pre = run.loc[run["sample"] <= onset, variables]
    post = run.loc[run["sample"] > onset, variables]
    std = pre.std(ddof=1)
    shift = ((post.mean() - pre.mean()).abs() / std).where(std > 0).dropna()
    return shift.sort_values(ascending=False)


def peak_after_onset(run, var, onset, tol=0.005):
    """(sample, value) of the first post-onset sample within tol (relative) of the extreme value.

    The extreme is the post-onset value furthest from the pre-onset mean.
    """
    pre_mean = run.loc[run["sample"] <= onset, var].mean()
    after = run[run["sample"] > onset]
    extreme = after.loc[(after[var] - pre_mean).abs().idxmax(), var]
    near = after[(after[var] - extreme).abs() <= tol * abs(extreme)]
    first = near.iloc[0]
    return int(first["sample"]), float(first[var])


def paired_shifts(faulty, normal, onset, window):
    """Mean |faulty - normal| over the first `window` samples after onset, in std units of the normal run.

    Both runs are the same simulation (same noise) until onset, so the difference is the fault's effect.
    A paired difference is allowed here because this is a diagnostic, not a model input.
    Variables with zero std in the normal run are left out. Sorted descending (stable).
    """
    f = faulty.set_index("sample").loc[onset + 1:onset + window, VARIABLES]
    n = normal.set_index("sample").loc[onset + 1:onset + window, VARIABLES]
    std = normal[VARIABLES].std(ddof=1)
    shift = ((f - n).abs().mean() / std).where(std > 0).dropna()
    return shift.sort_values(ascending=False, kind="stable")


def idv_check(faulty_runs, normal_runs, onset, expected=IDV_EXPECTED, window=IDV_WINDOW, top=3):
    """Paired first-hour check that fault k behaves like IDV(k).

    faulty_runs: {fault: {run id: run frame}}; normal_runs: {run id: run frame} (same run IDs, so the same noise).
    Per run, rank paired_shifts and take the top `top`. One row per fault: fault, expected_tag, expected_name,
    share_runs_in_top3, most_common_top1, top3_most_common ("tag:count;..." over runs), plant_unit_of_top1, verdict.
    """
    rows = []
    for fault, tag in expected.items():
        tops = [list(paired_shifts(run, normal_runs[rid], onset, window).index[:top])
                for rid, run in faulty_runs[fault].items()]
        share = sum(tag in t for t in tops) / len(tops)
        top1 = Counter(t[0] for t in tops)
        top1_tag = sorted(top1.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        counts = sorted(Counter(v for t in tops for v in t).items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        unit = VARIABLE_INFO[top1_tag].plant_unit
        if share >= IDV_PASS_SHARE:
            verdict = "confirmed"
        elif unit in IDV_ACCEPTABLE_UNITS[fault]:
            verdict = "unit_match_post_hoc"
        else:
            verdict = "fail"
        rows.append({"fault": fault, "expected_tag": tag, "expected_name": VARIABLE_INFO[tag].name,
                     "share_runs_in_top3": share, "most_common_top1": top1_tag,
                     "top3_most_common": ";".join(f"{v}:{c}" for v, c in counts),
                     "plant_unit_of_top1": unit, "verdict": verdict})
    return pd.DataFrame(rows)
