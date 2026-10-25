"""Phase 2b tests for src/assistant/score.py on a hand-made answers file; every metric matches a hand count.

Cases (fault: true unit) and first answers (status, unit, idv):
 c01  1: feeds     fault feeds [1, 2]       unit ok, top1, top3
 c02  2: feeds     fault feeds [6, 2, 1]    unit ok, top3 only
 c03  4: reactor   fault feeds [1]          wrong
 c04  5: cond/sep  can't tell unknown []     wrong
 c05  6: feeds     fault feeds [6]          unit ok, top1, top3
 c06  7: feeds     unparseable              wrong
 c07 13: reactor   fault reactor [14,4,13]  unit ok, top3 only
 c08  3: feeds     can't tell               honest
 c09  9: feeds     fault feeds [9]          not honest; IDV and unit right -> listed for the leak check
 c10 15: cond/sep  normal                   honest
 c11 16: unknown   fault unknown [16, 17]   status ok, top1, top3
 c12 18: unknown   normal                   status wrong
 c13  0: none      normal                   ok
 c14  0: none      fault feeds [1]          wrong
 c15  0: none      unparseable              wrong
Headline = c01-c07: 4 of 7. IDV over faults 1-20 (12 cases): top1 4, top3 6; faults 1-15 (10 cases): top1 3, top3 5.
Normal: 1 of 3. Faults 3/9/15: 2 of 3 honest. Faults 16-20: 1 of 2 flagged as fault.
"""
import json
import math

import pandas as pd
import pytest

from src.assistant import score

KEY = pd.DataFrame({
    "case_id": [f"c{i:02d}" for i in range(1, 16)],
    "fault": [1, 2, 4, 5, 6, 7, 13, 3, 9, 15, 16, 18, 0, 0, 0],
    "unit": ["feeds", "feeds", "reactor", "condenser/separator", "feeds", "feeds", "reactor", "feeds", "feeds",
             "condenser/separator", "unknown", "unknown", "none", "none", "none"],
})


def _ans(case, status, unit, idv, repeat=1):
    return {"case_id": case, "repeat": repeat, "parsed": {"ok": True, "status": status, "unit": unit, "idv": idv,
                                                           "evidence": "e"}}


def _bad(case, repeat=1):
    return {"case_id": case, "repeat": repeat, "parsed": {"ok": False, "error": "no <answer> block"}}


FIRST = [
    _ans("c01", "fault", "feeds", [1, 2]), _ans("c02", "fault", "feeds", [6, 2, 1]), _ans("c03", "fault", "feeds", [1]),
    _ans("c04", "can't tell", "unknown", []), _ans("c05", "fault", "feeds", [6]), _bad("c06"),
    _ans("c07", "fault", "reactor", [14, 4, 13]), _ans("c08", "can't tell", "unknown", []),
    _ans("c09", "fault", "feeds", [9]), _ans("c10", "normal", "unknown", []),
    _ans("c11", "fault", "unknown", [16, 17]), _ans("c12", "normal", "unknown", []),
    _ans("c13", "normal", "unknown", []), _ans("c14", "fault", "feeds", [1]), _bad("c15"),
]
BASELINE = {"c01": "feeds", "c02": "reactor", "c03": "reactor", "c04": "feeds", "c05": "feeds", "c06": "feeds",
            "c07": "stripper"}


def _repeats(agree_c03):
    """Cases c01-c10 repeated twice more. c03, c06, c10 disagree (c03 only when agree_c03 is False)."""
    out = []
    for first in FIRST[:10]:
        for rep in (2, 3):
            out.append({**first, "repeat": rep})
    for a in out:
        if a["case_id"] == "c03" and a["repeat"] == 2 and not agree_c03:
            a["parsed"] = {**a["parsed"], "unit": "reactor"}
        if a["case_id"] == "c10" and a["repeat"] == 3:
            a["parsed"] = {**a["parsed"], "status": "fault"}
        if a["case_id"] == "c06":
            a["parsed"] = {"ok": True, "status": "fault", "unit": "feeds", "idv": [7], "evidence": "e"}
    return out


def _run(agree_c03=False):
    return score.score(FIRST + _repeats(agree_c03), KEY, BASELINE)


# --- Wilson interval -------------------------------------------------------------------------
def test_wilson_interval_matches_known_values():
    lo, hi = score.wilson(4, 7)
    assert (round(lo, 3), round(hi, 3)) == (0.250, 0.842)
    lo, hi = score.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and round(hi, 4) == 0.2775
    lo, hi = score.wilson(10, 10)
    assert round(lo, 4) == 0.7225 and hi == pytest.approx(1.0)
    assert all(math.isnan(v) for v in score.wilson(0, 0))


# --- metrics -----------------------------------------------------------------------------------
def test_headline_is_unit_accuracy_on_faults_other_than_3_9_15_and_16_to_20():
    _, s = _run()
    h = s["headline"]
    assert (h["k"], h["n"]) == (4, 7)
    assert (round(h["lo"], 3), round(h["hi"], 3)) == (0.250, 0.842)


def test_a_unit_counts_only_when_the_answer_says_fault_with_that_unit():
    rows = [_ans("c01", "can't tell", "feeds", [1])] + [a for a in FIRST if a["case_id"] != "c01"]
    _, s = score.score(rows, KEY, BASELINE)
    assert s["headline"]["k"] == 3


def test_idv_top1_and_top3_over_all_faults_and_over_1_to_15():
    _, s = _run()
    assert (s["idv_top1_all"]["k"], s["idv_top1_all"]["n"]) == (4, 12)
    assert (s["idv_top3_all"]["k"], s["idv_top3_all"]["n"]) == (6, 12)
    assert (s["idv_top1_1to15"]["k"], s["idv_top1_1to15"]["n"]) == (3, 10)
    assert (s["idv_top3_1to15"]["k"], s["idv_top3_1to15"]["n"]) == (5, 10)


def test_normal_cases_are_right_only_when_answered_normal():
    _, s = _run()
    assert (s["normal"]["k"], s["normal"]["n"]) == (1, 3)


def test_hard_faults_are_reported_apart_with_honest_answers_counted():
    _, s = _run()
    assert (s["hard"]["k"], s["hard"]["n"]) == (2, 3)          # can't tell or normal
    assert s["hard_exact"] == ["c09"]                          # IDV and unit both right: check for leakage


def test_faults_16_to_20_are_scored_on_status_only():
    _, s = _run()
    assert (s["unknown_status"]["k"], s["unknown_status"]["n"]) == (1, 2)


def test_baseline_and_chance_level_on_the_headline_cases():
    _, s = _run()
    assert (s["baseline"]["k"], s["baseline"]["n"]) == (4, 7)   # c01, c03, c05, c06
    assert (s["chance"]["k"], s["chance"]["n"]) == (4, 7)       # feeds is the most common unit: c01, c02, c05, c06


def test_unparseable_answers_are_listed_and_counted_wrong():
    rows, s = _run()
    assert s["unparseable"] == [("c06", 1), ("c15", 1)]
    row = rows[(rows.case_id == "c06") & (rows.repeat == 1)].iloc[0]
    assert not row["parsed_ok"] and not row["unit_correct"] and not row["idv_top3"]


def test_headline_uses_the_first_run_of_each_case_not_the_repeats():
    only_first = score.score(FIRST, KEY, BASELINE)[1]
    with_repeats = _run()[1]
    assert only_first["headline"] == with_repeats["headline"]
    assert only_first["agreement"] is None


# --- repeat agreement -------------------------------------------------------------------------------
def test_repeat_agreement_counts_cases_where_all_runs_agree_on_status_and_unit():
    _, s = _run(agree_c03=False)
    # c03 (unit differs), c06 (unparseable first), c10 (status differs) disagree; 7 of 10 agree
    assert (s["agreement"]["k"], s["agreement"]["n"]) == (7, 10)
    assert s["agreement"]["warning"] is True


def test_no_upgrade_warning_at_8_of_10():
    _, s = _run(agree_c03=True)
    assert (s["agreement"]["k"], s["agreement"]["n"]) == (8, 10)
    assert s["agreement"]["warning"] is False


def test_summary_text_prints_k_of_n_and_the_upgrade_warning_only_when_needed():
    low = score.format_summary(_run(agree_c03=False)[1])
    assert "7 of 10 agree" in low and "3 repeats on all cases" in low
    ok = score.format_summary(_run(agree_c03=True)[1])
    assert "8 of 10 agree" in ok and "3 repeats on all cases" not in ok


def test_summary_text_reports_counts_with_intervals_and_lists_unparseable():
    text = score.format_summary(_run()[1])
    assert "4 / 7" in text and "[0.25, 0.84]" in text
    assert "c06" in text and "c15" in text
    assert "c09" in text   # the exact-IDV hard-fault case is listed for the orchestrator


# --- files --------------------------------------------------------------------------------------------
def test_load_answers_reads_one_json_object_per_line(tmp_path):
    path = tmp_path / "answers_dev.jsonl"
    path.write_text("\n".join(json.dumps(a) for a in FIRST[:3]) + "\n")
    assert [a["case_id"] for a in score.load_answers(path)] == ["c01", "c02", "c03"]


def test_per_row_table_has_the_columns_the_diagnosis_csv_promises():
    rows, _ = _run()
    assert {"case_id", "repeat", "fault", "true_unit", "status", "unit", "idv", "parsed_ok", "unit_correct",
            "idv_top1", "idv_top3", "baseline_unit"} <= set(rows.columns)
    first = rows[(rows.case_id == "c02") & (rows.repeat == 1)].iloc[0]
    assert first["idv"] == "6,2,1" and first["unit_correct"] and not first["idv_top1"] and first["idv_top3"]
