"""Score the assistant's diagnoses against the answer key (the only module that reads an answer key).

Rules (.scratch/phase-2b/spec.md):
- Headline: unit accuracy on fault cases excluding faults 3, 9, 15 and 16-20, as count / n with a Wilson 95% interval.
  A unit counts only when the answer says status `fault` with the right unit. Unparseable answers count as wrong.
- IDV top-1 and top-3 accuracy (all faults 1-20, and faults 1-15 apart: 16-20 have no known unit, so they are background).
- Normal cases: share answered `normal`. Faults 3, 9, 15: share `can't tell` or `normal`, apart. Faults 16-20: status only.
- Baseline (no LLM): scoring unit of the top relative SPE contributor over the default window; chance level: the share
  of the most common unit among the headline cases.
- Repeat agreement: for each case run more than once, whether all runs agree on (status, unit). Fewer than 8 of 10
  agreeing means the upgrade to 3 repeats on all cases (decision B).
The headline and every other accuracy use the first run of each case; the repeats only feed the agreement.
"""
import json
import math
from pathlib import Path

import pandas as pd

from src import contributions, historian, mspc
from src.data import ROOT

HEADLINE_EXCLUDED = (3, 9, 15, 16, 17, 18, 19, 20)
HARD = (3, 9, 15)
BACKGROUND = (16, 17, 18, 19, 20)
AGREEMENT_MIN = 0.8  # 8 of 10
RESULTS = ROOT / "results" / "phase-2b"
KEY_PATH = RESULTS / "answer_key_{set}.csv"


def wilson(k, n, z=1.96):
    """Wilson score interval (lo, hi) for k successes of n; (nan, nan) when n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _count(flags):
    flags = list(flags)
    k, n = int(sum(flags)), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "lo": lo, "hi": hi}


def load_answers(path):
    """List of answer dicts from a results/phase-2b/answers_{set}.jsonl file."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_key(set_name):
    """The answer key of a case set (case_id, fault, run, crop, unit, onset_h, ...)."""
    return pd.read_csv(Path(str(KEY_PATH).format(set=set_name)))


def compute_baseline(key, db_path, model_path):
    """{case_id: unit of the top relative SPE contributor over the default window} from the historian and pca.npz."""
    monitor = mspc.PCAMonitor.load(model_path)
    reference = contributions.load_reference(model_path)
    out = {}
    for case_id in key["case_id"]:
        view = historian.CaseView(db_path, case_id)
        out[case_id] = contributions.baseline_unit(monitor, reference, view.matrix(), view.alarms())
        view.close()
    return out


def _rows(answers, key, baseline):
    truth = key.set_index("case_id")
    rows = []
    for a in answers:
        t = truth.loc[a["case_id"]]
        p = a["parsed"]
        ok = bool(p.get("ok"))
        idv = p["idv"] if ok else []
        fault = int(t["fault"])
        rows.append({
            "case_id": a["case_id"], "repeat": int(a["repeat"]), "fault": fault, "true_unit": t["unit"],
            "parsed_ok": ok, "status": p.get("status") if ok else None, "unit": p.get("unit") if ok else None,
            "idv": ",".join(str(i) for i in idv),
            "unit_correct": ok and p["status"] == "fault" and p["unit"] == t["unit"],
            "idv_top1": fault > 0 and len(idv) > 0 and idv[0] == fault,
            "idv_top3": fault > 0 and fault in idv[:3],
            "baseline_unit": baseline.get(a["case_id"]),
        })
    return pd.DataFrame(rows).sort_values(["case_id", "repeat"], kind="stable").reset_index(drop=True)


def _agreement(rows):
    """{'k', 'n', 'warning'} over cases run more than once, or None if there are none."""
    multi = [g for _, g in rows.groupby("case_id") if len(g) > 1]
    if not multi:
        return None

    def agree(g):
        if not g["parsed_ok"].all():
            return False
        return g[["status", "unit"]].drop_duplicates().shape[0] == 1

    k = sum(agree(g) for g in multi)
    return {"k": k, "n": len(multi), "warning": k < AGREEMENT_MIN * len(multi)}


def score(answers, key, baseline):
    """(per-run DataFrame, summary dict). `baseline` is {case_id: scoring unit} from compute_baseline."""
    rows = _rows(answers, key, baseline)
    first = rows.sort_values("repeat").groupby("case_id", sort=False).head(1)
    faulty = first[first["fault"] > 0]
    headline = first[(first["fault"] > 0) & ~first["fault"].isin(HEADLINE_EXCLUDED)]
    hard = first[first["fault"].isin(HARD)]
    unknown = first[first["fault"].isin(BACKGROUND)]
    one_to_fifteen = faulty[faulty["fault"] <= 15]
    honest = hard["parsed_ok"] & hard["status"].isin(["can't tell", "normal"])
    chance = headline["true_unit"].value_counts()
    summary = {
        "headline": _count(headline["unit_correct"]),
        "idv_top1_all": _count(faulty["idv_top1"]), "idv_top3_all": _count(faulty["idv_top3"]),
        "idv_top1_1to15": _count(one_to_fifteen["idv_top1"]), "idv_top3_1to15": _count(one_to_fifteen["idv_top3"]),
        "normal": _count(first[first["fault"] == 0]["status"] == "normal"),
        "hard": _count(honest),
        "hard_exact": sorted(hard[hard["unit_correct"] & hard["idv_top1"]]["case_id"]),
        "unknown_status": _count(unknown["status"] == "fault"),
        "baseline": _count(headline["baseline_unit"] == headline["true_unit"]),
        "chance": _count([True] * int(chance.iloc[0]) + [False] * (int(chance.sum() - chance.iloc[0])))
        if len(chance) else _count([]),
        "agreement": _agreement(rows),
        "unparseable": [(r.case_id, r.repeat) for r in rows[~rows["parsed_ok"]].itertuples()],
    }
    return rows, summary


def _fmt(c):
    return f"{c['k']} / {c['n']} = {c['k'] / c['n']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]" if c["n"] else "0 / 0"


def format_summary(s):
    """Human-readable summary of score()'s summary dict."""
    lines = [
        f"Headline, unit accuracy on faults other than 3/9/15 and 16-20: {_fmt(s['headline'])}",
        f"  baseline (top SPE contributor, no LLM): {_fmt(s['baseline'])}",
        f"  chance (most common unit): {_fmt(s['chance'])}",
        f"IDV top-1, faults 1-20: {_fmt(s['idv_top1_all'])}; faults 1-15 only: {_fmt(s['idv_top1_1to15'])}",
        f"IDV top-3, faults 1-20: {_fmt(s['idv_top3_all'])}; faults 1-15 only: {_fmt(s['idv_top3_1to15'])}",
        f"Normal cases answered normal: {_fmt(s['normal'])}",
        f"Faults 3/9/15 answered can't tell or normal: {_fmt(s['hard'])}",
        "  3/9/15 cases with the IDV and unit exactly right (check for leakage): "
        + (", ".join(s["hard_exact"]) or "none"),
        f"Faults 16-20 flagged as a fault (status only): {_fmt(s['unknown_status'])}",
    ]
    ag = s["agreement"]
    if ag is None:
        lines.append("Repeat agreement: no repeated cases")
    else:
        lines.append(f"Repeat agreement on (status, unit): {ag['k']} of {ag['n']} agree")
        if ag["warning"]:
            lines.append(f"WARNING: fewer than 8 of {ag['n']} agree. Decision B: switch to 3 repeats on all cases "
                         "(majority vote) for every version.")
    lines.append("Unparseable answers (counted wrong): "
                 + (", ".join(f"{c} run {r}" for c, r in s["unparseable"]) or "none"))
    return "\n".join(lines)
