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
Versions (phase 3): v1 keeps the original file names; v1.1 adds a `_v1.1` suffix (answers_test_v1.1.jsonl,
diagnosis_test_v1.1.csv). `compare` puts v1 and v1.1 side by side case by case, with the flip counts, and splits the
headline by use of `classify_fault`.
"""
import json
import math
from pathlib import Path

import pandas as pd

from src import contributions, evaluate, historian, mspc
from src.data import RESULTS_2B

HEADLINE_EXCLUDED = (3, 9, 15, 16, 17, 18, 19, 20)
HARD = evaluate.HARD_FAULTS
BACKGROUND = (16, 17, 18, 19, 20)
AGREEMENT_MIN = 0.8  # 8 of 10
RESULTS = RESULTS_2B
KEY_PATH = RESULTS / "answer_key_{set}.csv"
VERSIONS = ("v1", "v1.1")
CLASSIFY_TOOL = "classify_fault"


def version_suffix(version):
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
    return "" if version == "v1" else f"_{version}"


def answers_path(set_name, version="v1"):
    """results/phase-2b/answers_{set}[_v1.1].jsonl"""
    return RESULTS / f"answers_{set_name}{version_suffix(version)}.jsonl"


def diagnosis_path(set_name, version="v1"):
    """results/phase-2b/diagnosis_{set}[_v1.1].csv"""
    return RESULTS / f"diagnosis_{set_name}{version_suffix(version)}.csv"


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


def compute_classify_tops(key, db_path):
    """{case_id: top-3 class labels} from the classify_fault tool (default window) on every case of a set."""
    from src.assistant import server
    out = {}
    for case_id in key["case_id"]:
        tools = server.TepTools(db_path, case_id)
        out[case_id] = [c["class"] for c in json.loads(tools.classify_fault())["top"]]
        tools.view.close()
    return out


def classify_accuracy(key, tops):
    """Top-1 and top-3 accuracy of the classifier over the fault cases (faults 1-20), how often normal cases come out as
    'normal', and its top-3 lists on faults 3, 9, 15. `tops` is {case_id: top-3 labels} from compute_classify_tops."""
    truth = key.set_index("case_id")["fault"]
    faulty = [c for c in truth.index if truth[c] > 0]
    normal = [c for c in truth.index if truth[c] == 0]
    return {"top1": _count(tops[c][0] == f"IDV {truth[c]}" for c in faulty),
            "top3": _count(f"IDV {truth[c]}" in tops[c] for c in faulty),
            "normal": _count(tops[c][0] == "normal" for c in normal),
            "hard": {c: tops[c] for c in faulty if truth[c] in HARD}}


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
            "used_classify": any(c["name"].endswith(CLASSIFY_TOOL) for c in a.get("tool_calls", [])),
        })
    return pd.DataFrame(rows).sort_values(["case_id", "repeat"], kind="stable").reset_index(drop=True)


def agreement(rows):
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


def _first_runs(rows):
    """The first run of each case, indexed by case_id."""
    return rows.sort_values("repeat").groupby("case_id", sort=False).head(1).set_index("case_id")


def case_flags(rows):
    """{metric: bool Series indexed by case_id} over the first run of each case: the per-case right/wrong behind every count."""
    first = _first_runs(rows)
    faulty = first[first["fault"] > 0]
    headline = first[(first["fault"] > 0) & ~first["fault"].isin(HEADLINE_EXCLUDED)]
    hard = first[first["fault"].isin(HARD)]
    one_to_fifteen = faulty[faulty["fault"] <= 15]
    return {
        "headline": headline["unit_correct"], "idv_top1_all": faulty["idv_top1"], "idv_top3_all": faulty["idv_top3"],
        "idv_top1_1to15": one_to_fifteen["idv_top1"], "idv_top3_1to15": one_to_fifteen["idv_top3"],
        "normal": first[first["fault"] == 0]["status"] == "normal",
        "hard": hard["parsed_ok"] & hard["status"].isin(["can't tell", "normal"]),
        "unknown_status": first[first["fault"].isin(BACKGROUND)]["status"] == "fault",
        "baseline": headline["baseline_unit"] == headline["true_unit"],
    }


def _chance(units):
    """Share of the most common unit among the headline cases, as a count."""
    counts = units.value_counts()
    return _count([True] * int(counts.iloc[0]) + [False] * int(counts.sum() - counts.iloc[0])) if len(counts) else _count([])


def score(answers, key, baseline):
    """(per-run DataFrame, summary dict). `baseline` is {case_id: scoring unit} from compute_baseline."""
    rows = _rows(answers, key, baseline)
    first = _first_runs(rows)
    hard = first[first["fault"].isin(HARD)]
    headline = first[(first["fault"] > 0) & ~first["fault"].isin(HEADLINE_EXCLUDED)]
    summary = {name: _count(flag) for name, flag in case_flags(rows).items()}
    summary.update({
        "hard_exact": sorted(hard[hard["unit_correct"] & hard["idv_top1"]].index),
        "chance": _chance(headline["true_unit"]),
        "agreement": agreement(rows),
        "unparseable": [(r.case_id, r.repeat) for r in rows[~rows["parsed_ok"]].itertuples()],
    })
    return rows, summary


def classify_split(rows):
    """Scores of the first sessions that called classify_fault vs those that did not.

    {'used'|'not_used': {'sessions', 'headline', 'idv_top1_all', 'idv_top3_all'}}; the HGB-assisted number is a separate claim.
    """
    first = _first_runs(rows)
    flags = case_flags(rows)
    out = {}
    for name, used in (("used", True), ("not_used", False)):
        cases = first.index[first["used_classify"] == used]
        out[name] = {"sessions": int(len(cases)),
                     **{m: _count(flags[m][flags[m].index.isin(cases)]) for m in ("headline", "idv_top1_all", "idv_top3_all")}}
    return out


COMPARED = ("headline", "idv_top1_all", "idv_top3_all", "idv_top1_1to15", "idv_top3_1to15", "normal", "hard",
            "unknown_status", "baseline")
COMPARISON_COLUMNS = ["metric", "v1_k", "v1_n", "v1.1_k", "v1.1_n", "wrong_to_right", "right_to_wrong"]
NAN = float("nan")


def compare(v1, v11):
    """v1 vs v1.1 table (DataFrame COMPARISON_COLUMNS) from two score() results (rows, summary).

    One row per metric with both versions' counts and, for the per-case metrics, how many cases flipped from wrong to right
    and from right to wrong (cases in both versions). The last rows score the headline cases split by whether the first v1.1
    session called classify_fault; v1 is scored on the same cases.
    """
    (rows1, sum1), (rows11, sum11) = v1, v11
    flags1, flags11 = case_flags(rows1), case_flags(rows11)

    def line(metric, a, b):
        both = a.index.intersection(b.index)
        return {"metric": metric, "v1_k": int(a.sum()), "v1_n": len(a), "v1.1_k": int(b.sum()), "v1.1_n": len(b),
                "wrong_to_right": int((~a[both] & b[both]).sum()), "right_to_wrong": int((a[both] & ~b[both]).sum())}

    table = []
    for m in COMPARED:
        row = line(m, flags1[m], flags11[m])
        if m == "baseline":      # no LLM: the same in both versions, nothing flips
            row["wrong_to_right"] = row["right_to_wrong"] = NAN
        table.append(row)
    table.append({"metric": "chance", "v1_k": sum1["chance"]["k"], "v1_n": sum1["chance"]["n"], "v1.1_k": sum11["chance"]["k"],
                  "v1.1_n": sum11["chance"]["n"], "wrong_to_right": NAN, "right_to_wrong": NAN})
    used = _first_runs(rows11)["used_classify"]
    head1, head11 = flags1["headline"], flags11["headline"]
    for label, cases in (("used", used.index[used]), ("not used", used.index[~used])):
        table.append(line(f"headline | classify_fault {label}", head1[head1.index.isin(cases)], head11[head11.index.isin(cases)]))
    return pd.DataFrame(table, columns=COMPARISON_COLUMNS)


def _kn(k, n):
    return f"{int(k)} / {int(n)} = {k / n:.2f}" if n else "0 / 0"


def format_comparison(table):
    """Readable v1 vs v1.1 table with the flip counts, and the caution about what the HGB-assisted gain means."""
    lines = ["v1 vs v1.1 (first run of each case; k / n)"]
    for _, r in table.iterrows():
        text = f"  {r['metric']}: v1 {_kn(r['v1_k'], r['v1_n'])}, v1.1 {_kn(r['v1.1_k'], r['v1.1_n'])}"
        if r["wrong_to_right"] == r["wrong_to_right"]:
            text += f"; flips wrong->right {int(r['wrong_to_right'])}, right->wrong {int(r['right_to_wrong'])}"
        lines.append(text)
    lines.append("Caution: hgb-l2 is a supervised classifier trained on labelled training-family runs, so a v1.1 gain from "
                 "classify_fault comes from labelled plant history, not from reasoning alone. It is a separate claim from v1. "
                 "The classifier only knows normal and faults 1-20.")
    return "\n".join(lines)


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
