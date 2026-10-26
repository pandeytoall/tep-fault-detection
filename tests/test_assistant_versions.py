"""Phase 3 tests for the assistant's version handling in src/assistant/score.py and src/assistant/session.py.

Hand-made v1 answers are the FIRST list of test_assistant_score.py (headline 4 of 7). The v1.1 answers change five cases:
 c01  right -> wrong (headline)     c03  wrong -> right (headline)    c04  wrong, stays wrong
 c07  right, stays right            c10  honest (normal) -> fault (not honest)   c13  normal -> fault (wrong)
 c14  fault -> normal (right)       c02, c03, c10 called classify_fault
"""
from pathlib import Path

import pandas as pd
import pytest
from test_assistant_score import BASELINE, FIRST, KEY, _ans

from src.assistant import score, session

ROOT = Path(__file__).resolve().parent.parent
CLASSIFY = {"name": "mcp__tep__classify_fault", "input": {}}
OTHER = {"name": "mcp__tep__get_alarms", "input": {}}


def _v11():
    changes = {
        "c01": _ans("c01", "fault", "reactor", [4]),                    # headline right -> wrong
        "c03": _ans("c03", "fault", "reactor", [4, 1]),                 # headline wrong -> right, top1 and top3 too
        "c10": _ans("c10", "fault", "condenser/separator", [5]),        # honest -> not honest
        "c13": _ans("c13", "fault", "feeds", [1]),                      # normal case answered fault: stays wrong? (was right)
        "c14": _ans("c14", "normal", "unknown", []),                    # normal case now right
    }
    out = []
    for a in FIRST:
        a = dict(changes.get(a["case_id"], a))
        a["tool_calls"] = [OTHER] + ([CLASSIFY] if a["case_id"] in ("c02", "c03", "c10") else [])
        out.append(a)
    return out


def _scored(answers):
    return score.score(answers, KEY, BASELINE)


# --- file names by version --------------------------------------------------------------------------------------
def test_v1_keeps_the_original_file_names_and_v1_1_adds_a_suffix():
    assert score.answers_path("test", "v1") == score.RESULTS / "answers_test.jsonl"
    assert score.answers_path("test", "v1.1") == score.RESULTS / "answers_test_v1.1.jsonl"
    assert score.answers_path("dev", "v1.1") == score.RESULTS / "answers_dev_v1.1.jsonl"
    assert score.diagnosis_path("test", "v1") == score.RESULTS / "diagnosis_test.csv"
    assert score.diagnosis_path("test", "v1.1") == score.RESULTS / "diagnosis_test_v1.1.csv"


def test_an_unknown_version_is_refused():
    with pytest.raises(ValueError, match="v1.2"):
        score.answers_path("test", "v1.2")


def test_each_version_has_its_prompt_its_tools_and_its_own_frozen_hash_file():
    assert session.PROMPTS["v1"].name == "prompt.md" and session.PROMPTS["v1.1"].name == "prompt_v1.1.md"
    assert session.TOOLSETS["v1"] == session.TOOLS and len(session.TOOLSETS["v1"]) == 6
    assert session.TOOLSETS["v1.1"] == session.TOOLS + ["score_model", "classify_fault"]
    assert session.frozen_path("v1") == score.RESULTS / "prompt_frozen.sha256"
    assert session.frozen_path("v1.1") == score.RESULTS / "prompt_frozen_v1.1.sha256"


def test_the_v1_1_command_allows_exactly_its_eight_tools():
    cfg = session.mcp_config("py.exe", "server.py", "dev_01", "dev")
    cmd = session.build_command("claude.exe", "Diagnose.", session.PROMPTS["v1.1"], cfg, tools=session.TOOLSETS["v1.1"])
    allowed = cmd[cmd.index("--allowedTools") + 1].split(",")
    assert sorted(allowed) == sorted(f"mcp__tep__{t}" for t in session.TOOLSETS["v1.1"]) and len(allowed) == 8
    assert cmd[cmd.index("--system-prompt-file") + 1].endswith("prompt_v1.1.md")


@pytest.mark.skipif(not (ROOT / "src" / "assistant" / "prompt_v1.1.md").exists(), reason="prompt_v1.1.md is written after the dev check")
def test_the_v1_prompt_is_untouched_and_the_v1_1_prompt_only_adds_the_new_tools():
    v1 = (ROOT / "src" / "assistant" / "prompt.md").read_text(encoding="utf-8")
    v11 = (ROOT / "src" / "assistant" / "prompt_v1.1.md").read_text(encoding="utf-8")
    assert "score_model" in v11 and "classify_fault" in v11 and "score_model" not in v1
    assert v11.split("## How to answer")[1] == v1.split("## How to answer")[1]       # the answer block is unchanged


# --- used_classify and the split ----------------------------------------------------------------------------------------
def test_a_session_counts_as_using_classify_fault_when_it_called_the_tool():
    rows, _ = _scored(_v11())
    used = rows[rows.used_classify].case_id.tolist()
    assert used == ["c02", "c03", "c10"]
    rows_v1, _ = _scored(FIRST)                                            # v1 answers carry no tool list in this fixture
    assert not rows_v1.used_classify.any()


def test_classify_split_scores_the_cases_whose_first_session_used_the_tool_apart():
    rows, summary = _scored(_v11())
    split = score.classify_split(rows)
    assert split["used"]["sessions"] == 3 and split["not_used"]["sessions"] == 12
    # headline cases that called the tool: c02 and c03 (c10 is a fault 15 case, outside the headline)
    assert (split["used"]["headline"]["k"], split["used"]["headline"]["n"]) == (2, 2)
    assert (split["not_used"]["headline"]["k"], split["not_used"]["headline"]["n"]) == (2, 5)   # c05, c07 right
    assert split["used"]["headline"]["n"] + split["not_used"]["headline"]["n"] == summary["headline"]["n"]


# --- v1 vs v1.1 -----------------------------------------------------------------------------------------------------------------
def _compare():
    return score.compare(_scored(FIRST), _scored(_v11()))


def _row(table, metric):
    return table[table.metric == metric].iloc[0]


def test_the_comparison_lists_both_versions_counts_for_every_metric():
    t = _compare()
    h = _row(t, "headline")
    assert (h["v1_k"], h["v1_n"], h["v1.1_k"], h["v1.1_n"]) == (4, 7, 4, 7)       # -c01 +c03
    assert (_row(t, "idv_top1_all")["v1_k"], _row(t, "idv_top1_all")["v1.1_k"]) == (4, 4)   # c01 lost, c03 gained
    assert (_row(t, "idv_top3_all")["v1_k"], _row(t, "idv_top3_all")["v1.1_k"]) == (6, 6)
    n = _row(t, "normal")
    assert (n["v1_k"], n["v1.1_k"], n["v1_n"]) == (1, 1, 3)                        # c13 right -> wrong, c14 wrong -> right
    hard = _row(t, "hard")
    assert (hard["v1_k"], hard["v1.1_k"]) == (2, 1)                                # c10 lost
    assert {"baseline", "chance", "idv_top1_1to15", "idv_top3_1to15", "unknown_status"} <= set(t.metric)
    assert (_row(t, "baseline")["v1_k"], _row(t, "baseline")["v1.1_k"]) == (4, 4)  # no LLM: identical


def test_flip_counts_are_wrong_to_right_and_right_to_wrong_case_by_case():
    t = _compare()
    h = _row(t, "headline")
    assert (h["wrong_to_right"], h["right_to_wrong"]) == (1, 1)                    # c03 gained, c01 lost
    n = _row(t, "normal")
    assert (n["wrong_to_right"], n["right_to_wrong"]) == (1, 1)                    # c14 gained, c13 lost
    top1 = _row(t, "idv_top1_all")
    assert (top1["wrong_to_right"], top1["right_to_wrong"]) == (1, 1)
    assert pd.isna(_row(t, "baseline")["wrong_to_right"]) and pd.isna(_row(t, "chance")["wrong_to_right"])


def test_the_comparison_also_scores_the_classify_cases_in_both_versions():
    t = _compare()
    used = _row(t, "headline | classify_fault used")
    assert (used["v1_k"], used["v1_n"], used["v1.1_k"], used["v1.1_n"]) == (1, 2, 2, 2)    # c02 right in v1, c03 wrong in v1
    assert (used["wrong_to_right"], used["right_to_wrong"]) == (1, 0)
    unused = _row(t, "headline | classify_fault not used")
    assert (unused["v1_n"], unused["v1.1_n"]) == (5, 5) and (unused["v1_k"], unused["v1.1_k"]) == (3, 2)


def test_the_comparison_table_has_the_columns_the_csv_promises():
    assert list(_compare().columns) == ["metric", "v1_k", "v1_n", "v1.1_k", "v1.1_n", "wrong_to_right", "right_to_wrong"]


def test_format_comparison_prints_the_flip_counts_and_the_leak_caution():
    text = score.format_comparison(_compare())
    assert "headline" in text and "4 / 7" in text and "wrong->right 1" in text and "right->wrong 1" in text
    assert "supervised classifier" in text


# --- the dev check of the classifier ------------------------------------------------------------------------------------------
def test_classify_accuracy_counts_top1_and_top3_over_fault_cases_and_lists_what_it_says_about_3_9_15():
    key = pd.DataFrame({"case_id": ["d1", "d2", "d3", "d4", "d5", "d6"], "fault": [1, 2, 6, 3, 9, 0]})
    tops = {"d1": ["IDV 1", "IDV 2", "normal"], "d2": ["IDV 6", "IDV 2", "IDV 1"], "d3": ["normal", "IDV 7", "IDV 8"],
            "d4": ["normal", "IDV 3", "IDV 9"], "d5": ["IDV 9", "normal", "IDV 3"], "d6": ["normal", "IDV 1", "IDV 2"]}
    out = score.classify_accuracy(key, tops)
    assert (out["top1"]["k"], out["top1"]["n"]) == (2, 5)            # d1, d5 (a hard fault counts as a fault case)
    assert (out["top3"]["k"], out["top3"]["n"]) == (4, 5)            # d1, d2, d4 (IDV 3 listed), d5
    assert (out["normal"]["k"], out["normal"]["n"]) == (1, 1)
    assert out["hard"] == {"d4": ["normal", "IDV 3", "IDV 9"], "d5": ["IDV 9", "normal", "IDV 3"]}
