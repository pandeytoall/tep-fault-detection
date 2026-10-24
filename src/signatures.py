"""Fault signatures: which variables a fault moves, and the IDV numbering check."""
import pandas as pd

from src.data import VARIABLES
from src.variables import VARIABLE_INFO

# Hypotheses from the IDV descriptions (spec phase 1): fault -> variable that should be among the top 3 shifts.
IDV_EXPECTED = {
    4: "xmv_10",   # reactor cooling water inlet temperature step -> reactor cooling water flow
    5: "xmv_11",   # condenser cooling water inlet temperature step -> condenser cooling water flow
    6: "xmeas_1",  # A feed loss -> A feed
    7: "xmeas_4",  # C header pressure loss -> A and C feed
}
IDV_PASS_SHARE = 0.8


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


def idv_check(runs_by_fault, onset, expected=IDV_EXPECTED, top=3):
    """Does the expected variable of each fault rank in the top `top` shifts of the run?

    runs_by_fault: {fault: [run frame with sample + the 52 variables, ...]}. Shifts: rank_shifts, after onset
    vs samples <= onset of the same run. Returns one row per fault: fault, expected_tag, expected_name,
    share_runs_in_top3 and most_common_top1 (the variable most often ranked first).
    """
    rows = []
    for fault, tag in expected.items():
        ranked = [rank_shifts(run, VARIABLES, onset) for run in runs_by_fault[fault]]
        in_top = [tag in r.index[:top] for r in ranked]
        top1 = pd.Series([r.index[0] for r in ranked])
        rows.append({"fault": fault, "expected_tag": tag, "expected_name": VARIABLE_INFO[tag].name,
                     "share_runs_in_top3": sum(in_top) / len(in_top), "most_common_top1": top1.mode().iloc[0]})
    return pd.DataFrame(rows)
