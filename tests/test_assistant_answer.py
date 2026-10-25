"""Phase 2b tests for src/assistant/answer.py: parsing the assistant's final answer block."""
import pytest

from src.assistant import answer

GOOD = """I looked at the alarms.

<answer>
status: fault
unit: reactor
idv: 4, 11, 13
evidence: Reactor cooling water flow moved first and SPE alarmed at hour 3.
</answer>
"""


def test_a_well_formed_block_is_parsed():
    out = answer.parse_answer(GOOD)
    assert out == {"ok": True, "status": "fault", "unit": "reactor", "idv": [4, 11, 13],
                   "evidence": "Reactor cooling water flow moved first and SPE alarmed at hour 3."}


def test_the_last_block_wins_and_text_around_it_is_ignored():
    text = GOOD.replace("reactor", "stripper", 1) + "\nActually, on reflection:\n" + GOOD
    assert answer.parse_answer(text)["unit"] == "reactor"


@pytest.mark.parametrize("raw, status", [("fault", "fault"), ("Normal", "normal"), ("can't tell", "can't tell"),
                                         ("can’t tell", "can't tell"), ("cannot tell", "can't tell"),
                                         ("  FAULT ", "fault")])
def test_status_spellings(raw, status):
    block = f"<answer>\nstatus: {raw}\nunit: unknown\nidv: none\nevidence: x\n</answer>"
    assert answer.parse_answer(block)["status"] == status


def test_normal_answers_may_have_no_idv_and_an_unknown_unit():
    out = answer.parse_answer("<answer>\nstatus: normal\nunit: unknown\nidv: none\nevidence: quiet\n</answer>")
    assert out["ok"] and out["idv"] == [] and out["unit"] == "unknown"


def test_units_are_the_five_scoring_units_or_unknown():
    for unit in ("feeds", "reactor", "condenser/separator", "compressor/recycle", "stripper", "unknown"):
        block = f"<answer>\nstatus: fault\nunit: {unit}\nidv: 1\nevidence: e\n</answer>"
        assert answer.parse_answer(block)["unit"] == unit


def test_evidence_may_run_over_several_lines():
    block = "<answer>\nstatus: fault\nunit: feeds\nidv: 1\nevidence: first line\nsecond line\n</answer>"
    assert answer.parse_answer(block)["evidence"] == "first line second line"


def test_missing_evidence_is_allowed():
    out = answer.parse_answer("<answer>\nstatus: fault\nunit: feeds\nidv: 6\n</answer>")
    assert out["ok"] and out["evidence"] == ""


@pytest.mark.parametrize("text, reason", [
    ("no block at all", "no <answer> block"),
    ("<answer>\nstatus: fault\nunit: feeds\nidv: 1\n", "no <answer> block"),                      # not closed
    ("<answer>\nunit: feeds\nidv: 1\nevidence: e\n</answer>", "status"),                          # missing status
    ("<answer>\nstatus: maybe\nunit: feeds\nidv: 1\nevidence: e\n</answer>", "status"),           # bad status
    ("<answer>\nstatus: fault\nidv: 1\nevidence: e\n</answer>", "unit"),                          # missing unit
    ("<answer>\nstatus: fault\nunit: boiler\nidv: 1\nevidence: e\n</answer>", "unit"),            # bad unit
    ("<answer>\nstatus: fault\nunit: feeds\nevidence: e\n</answer>", "idv"),                      # missing idv
    ("<answer>\nstatus: fault\nunit: feeds\nidv: one\nevidence: e\n</answer>", "idv"),            # not numbers
    ("<answer>\nstatus: fault\nunit: feeds\nidv: 1, 2, 3, 4\nevidence: e\n</answer>", "idv"),     # more than 3
    ("<answer>\nstatus: fault\nunit: feeds\nidv: 21\nevidence: e\n</answer>", "idv"),             # out of range
    ("<answer>\nstatus: fault\nunit: feeds\nidv: 0\nevidence: e\n</answer>", "idv"),              # out of range
    ("<answer>\nstatus: fault\nunit: feeds\nidv: 2, 2\nevidence: e\n</answer>", "idv"),           # repeated
    ("", "no <answer> block"),
    (None, "no <answer> block"),
])
def test_malformed_blocks_are_reported_not_guessed(text, reason):
    out = answer.parse_answer(text)
    assert out["ok"] is False and reason in out["error"]


def test_a_fault_with_no_idv_numbers_still_parses():
    out = answer.parse_answer("<answer>\nstatus: fault\nunit: feeds\nidv: none\nevidence: e\n</answer>")
    assert out["ok"] is True and out["idv"] == []   # still parsed; the scorer counts it wrong for IDV
