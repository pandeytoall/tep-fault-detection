"""Phase 2b tests for src/assistant/server.py: the tools over a small synthetic historian, the output cap, the
one-case view, the notes folder and the isolation rules (no answer key, no parquet)."""
import asyncio
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import contributions, data, historian
from src.assistant import server

ROOT = Path(__file__).resolve().parent.parent
T = 0.05


def _tools(world, case="dev_01"):
    return server.TepTools(world["db"], case, world["npz"], world["notes"])


# --- query_historian -------------------------------------------------------------------------
def test_query_returns_the_stored_values_for_the_window_and_step(world):
    out = json.loads(_tools(world).query_historian(["xmeas_1", "xmv_3"], 0.5, 1.0, 5))
    assert out["t_hours"] == [0.5, 0.75, 1.0] and out["truncated"] is False
    frame = world["frames"]["dev_01"]
    assert np.allclose(out["values"]["xmeas_1"], frame["xmeas_1"].iloc[[10, 15, 20]], rtol=1e-4)
    assert set(out["values"]) == {"xmeas_1", "xmv_3"}


def test_query_output_is_capped_at_2000_values_and_says_so(world):
    out = json.loads(_tools(world).query_historian(data.VARIABLES, 0.0, 100.0))   # 52 tags x 240 samples = 12480
    n_values = sum(len(v) for v in out["values"].values())
    assert out["truncated"] is True and n_values <= server.MAX_VALUES == 2000
    assert len(out["t_hours"]) == 2000 // 52 == 38
    assert "2000-value cap" in out["note"] and "38 of 240" in out["note"]
    assert out["t_hours"][0] == 0.0     # the first samples are kept


def test_query_under_the_cap_is_not_truncated(world):
    out = json.loads(_tools(world).query_historian(["xmeas_1"] * 1 + ["xmeas_2"], 0.0, 100.0))   # 2 x 240
    assert out["truncated"] is False and len(out["t_hours"]) == 240 and "note" not in out


def test_query_rejects_unknown_tags_and_bad_windows(world):
    with pytest.raises(ValueError, match="unknown tag"):
        _tools(world).query_historian(["temperature"], 0, 1)
    with pytest.raises(ValueError, match="no samples"):
        _tools(world).query_historian(["xmeas_1"], 500.0, 600.0)
    with pytest.raises(ValueError, match="step"):
        _tools(world).query_historian(["xmeas_1"], 0, 1, 0)


# --- get_alarms --------------------------------------------------------------------------------
def test_get_alarms_returns_the_alarm_log_and_the_case_span(world):
    out = json.loads(_tools(world).get_alarms())
    view = historian.CaseView(world["db"], "dev_01")
    expected = view.alarms()
    assert len(out["alarms"]) == len(expected) >= 1
    assert out["alarms"][0].keys() == {"t_hours", "statistic", "event", "value", "limit"}
    assert out["case_span_hours"] == [0.0, 11.95]
    assert "1%" in out["note"]
    late = json.loads(_tools(world).get_alarms(start_h=expected["t_hours"].iloc[0] + 0.01))
    assert len(late["alarms"]) < len(out["alarms"]) or len(expected) == 1


def test_a_quiet_case_returns_an_empty_log_with_a_note(world):
    out = json.loads(_tools(world, "dev_03").get_alarms())
    assert out["alarms"] == [] and "no alarm events" in out["note"]


# --- score_pca ------------------------------------------------------------------------------------
def test_score_pca_returns_t2_spe_and_the_limits(world):
    out = json.loads(_tools(world).score_pca(0.0, 1.0, 5))
    scores = historian.score_case(world["monitor"], pd.DataFrame(world["frames"]["dev_01"]))
    assert out["t_hours"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert np.allclose(out["t2"], scores["t2"].iloc[[0, 5, 10, 15, 20]], rtol=1e-3)
    assert np.allclose(out["spe"], scores["spe"].iloc[[0, 5, 10, 15, 20]], rtol=1e-3)
    assert out["limits"] == {"t2": world["monitor"].limits["t2"]["1%"], "spe": world["monitor"].limits["spe"]["1%"]}


def test_score_pca_cap_counts_both_statistics(world):
    out = json.loads(_tools(world).score_pca(0.0, 100.0))
    assert out["truncated"] is False and len(out["t_hours"]) == 240          # 480 values
    big = server.TepTools(world["db"], "dev_01", world["npz"], world["notes"], max_values=100)
    out = json.loads(big.score_pca(0.0, 100.0))
    assert out["truncated"] is True and len(out["t_hours"]) == 50 and "100-value cap" in out["note"]


# --- top_contributors -----------------------------------------------------------------------------------
def test_top_contributors_default_window_starts_at_the_first_raise(world):
    out = json.loads(_tools(world).top_contributors("spe"))
    first_raise = contributions.window_start(historian.CaseView(world["db"], "dev_01").alarms(), "spe")
    assert out["window_hours"][0] == first_raise and first_raise >= 100 * T
    assert out["window"] == "default: from the first SPE raise to the end of the case"
    assert len(out["top"]) == 5 and out["top"][0]["tag"] == "xmeas_1"
    assert out["top"][0].keys() == {"tag", "name", "unit", "scoring_unit", "ratio"}
    assert out["top"][0]["scoring_unit"] == "feeds"
    ratios = [t["ratio"] for t in out["top"]]
    assert ratios == sorted(ratios, reverse=True)


def test_top_contributors_with_no_raise_uses_the_whole_case(world):
    out = json.loads(_tools(world, "dev_03").top_contributors("t2", n=3))
    assert out["window_hours"] == [0.0, 11.95] and "whole case" in out["window"] and len(out["top"]) == 3


def test_top_contributors_explicit_window_and_statistic_spelling(world):
    out = json.loads(_tools(world).top_contributors("SPE", start_h=2.0, end_h=3.0))
    assert out["window_hours"] == [2.0, 3.0] and out["window"] == "as requested"
    with pytest.raises(ValueError, match="statistic"):
        _tools(world).top_contributors("q")
    with pytest.raises(ValueError, match="no samples"):
        _tools(world).top_contributors("spe", start_h=50.0, end_h=60.0)


# --- notes -------------------------------------------------------------------------------------------------
def test_notes_are_listed_and_read_by_name(world):
    listing = json.loads(_tools(world).list_notes())
    assert [n["name"] for n in listing["notes"]] == ["monitor-guide", "process-description"]
    assert listing["notes"][1]["title"] == "Process description"
    assert "five units" in _tools(world).read_note("process-description")
    assert "T2 and SPE" in _tools(world).read_note("monitor-guide.md")


@pytest.mark.parametrize("name", ["../secret", "..\\secret", "secret", str(Path("x") / ".." / "secret"), "/etc/passwd",
                                  "C:\\Windows\\win.ini", "", "process-description\x00"])
def test_read_note_only_reads_listed_notes(world, name):
    with pytest.raises(ValueError, match="no such note"):
        _tools(world).read_note(name)


# --- one case only ---------------------------------------------------------------------------------------------
def test_each_server_sees_only_its_own_case(world):
    a = json.loads(_tools(world, "dev_01").query_historian(["xmeas_3"], 0.0, 100.0))
    b = json.loads(_tools(world, "dev_02").query_historian(["xmeas_3"], 0.0, 100.0))
    assert np.allclose(a["values"]["xmeas_3"], world["frames"]["dev_01"]["xmeas_3"], rtol=1e-4)
    assert np.allclose(b["values"]["xmeas_3"], world["frames"]["dev_02"]["xmeas_3"], rtol=1e-4)
    assert not np.allclose(a["values"]["xmeas_3"], b["values"]["xmeas_3"])
    alarms_b = json.loads(_tools(world, "dev_02").get_alarms())["alarms"]
    assert not any(row.keys() & {"case_id"} for row in alarms_b)


def test_a_server_for_an_unknown_case_does_not_start(world):
    with pytest.raises(ValueError, match="no readings"):
        _tools(world, "dev_99")


def test_no_tool_takes_a_case_or_file_argument(world):
    for name, fn in inspect.getmembers(server.TepTools, inspect.isfunction):
        if name.startswith("_"):
            continue
        params = set(inspect.signature(fn).parameters) - {"self"}
        assert not params & {"case", "case_id", "path", "file", "db", "run", "fault"}, name


# --- the MCP layer ----------------------------------------------------------------------------------------------------
def test_the_mcp_server_exposes_exactly_the_six_tools_and_reports_errors_readably(world):
    from mcp import Client

    mcp_server = server.build_server(_tools(world))

    async def run():
        async with Client(mcp_server) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            ok = await client.call_tool("get_alarms", {})
            bad = await client.call_tool("query_historian", {"tags": ["nope"], "start_h": 0, "end_h": 1})
            return names, ok, bad

    names, ok, bad = asyncio.run(run())
    assert names == sorted(["query_historian", "get_alarms", "score_pca", "top_contributors", "list_notes", "read_note"])
    assert not ok.is_error and "alarms" in ok.content[0].text
    assert bad.is_error and "unknown tag" in bad.content[0].text


# --- isolation: no answer key, no parquet, no test files ------------------------------------------------------------------
ASSISTANT_DIR = ROOT / "src" / "assistant"


def test_nothing_in_the_assistant_package_reads_parquet_or_testing_files():
    for path in ASSISTANT_DIR.glob("*"):
        if path.suffix not in (".py", ".md"):
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"parquet|load_runs?\(|_Testing|_Training|run_split|simulationRun|faultNumber", text), path.name


def test_only_score_py_mentions_the_answer_key():
    for path in ASSISTANT_DIR.glob("*"):
        if path.suffix in (".py", ".md") and path.name != "score.py":
            assert "answer_key" not in path.read_text(encoding="utf-8"), path.name


def test_the_server_never_imports_the_case_builder():
    code = ("import sys; import src.assistant.server; "
            "bad = [m for m in ('src.cases', 'src.assistant.score', 'src.split') if m in sys.modules]; "
            "assert not bad, bad")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
