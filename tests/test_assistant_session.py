"""Phase 2b tests for src/assistant/session.py: the isolated claude command, reading its stream, rate-limit stops."""
import json
from pathlib import Path

import pytest

from src.assistant import session

ROOT = Path(__file__).resolve().parent.parent


def _cmd(**kw):
    cfg = session.mcp_config("py.exe", "server.py", "dev_01", "dev")
    return session.build_command("claude.exe", "Diagnose.", ROOT / "src" / "assistant" / "prompt.md", cfg, **kw)


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_command_disables_every_built_in_tool_and_loads_only_the_tep_server():
    cmd = _cmd()
    assert cmd[0] == "claude.exe" and "-p" in cmd
    assert _flag(cmd, "--tools") == ""                       # no Bash, file, web or agent tools
    assert "--strict-mcp-config" in cmd
    cfg = json.loads(_flag(cmd, "--mcp-config"))
    assert list(cfg["mcpServers"]) == ["tep"]
    assert cfg["mcpServers"]["tep"]["args"] == ["server.py", "--case", "dev_01", "--set", "dev"]
    allowed = _flag(cmd, "--allowedTools").split(",")
    assert sorted(allowed) == sorted(f"mcp__tep__{t}" for t in session.TOOLS) and len(allowed) == 6


def test_command_drops_user_project_and_local_settings_slash_commands_and_session_files():
    cmd = _cmd()
    assert _flag(cmd, "--setting-sources") == ""
    assert "--disable-slash-commands" in cmd and "--no-session-persistence" in cmd
    assert _flag(cmd, "--permission-prompts") == "none"       # anything outside the allowed tools is denied


def test_command_pins_the_model_the_prompt_file_and_the_output_format():
    cmd = _cmd()
    assert _flag(cmd, "--model") == "claude-sonnet-5-5" == session.MODEL
    assert Path(_flag(cmd, "--system-prompt-file")).name == "prompt.md"
    assert _flag(cmd, "--output-format") == "stream-json" and "--verbose" in cmd
    assert _flag(cmd, "--max-budget-usd") == str(session.MAX_BUDGET_USD)


def test_the_user_message_is_the_prompt_argument_and_names_no_case():
    cmd = _cmd()
    assert cmd[cmd.index("-p") + 1] == "Diagnose."


def test_session_env_turns_auto_memory_off_without_touching_the_rest():
    env = session.session_env({"PATH": "x", "ANTHROPIC_API_KEY": "k"})
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1" and env["PATH"] == "x"


# --- reading the stream ------------------------------------------------------------------------
def _stream(final_text="done", error=None, extra=()):
    events = [
        {"type": "system", "subtype": "init", "model": "claude-sonnet-5-5", "tools": ["mcp__tep__get_alarms"],
         "mcp_servers": [{"name": "tep", "status": "connected"}], "session_id": "s1", "claude_code_version": "2.1.310"},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "t1", "name": "mcp__tep__get_alarms", "input": {}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t2", "name": "mcp__tep__query_historian",
             "input": {"tags": ["xmeas_1"], "start_h": 0, "end_h": 1}}]}},
        *extra,
        {"type": "result", "subtype": "success", "is_error": error is not None, "result": final_text,
         "total_cost_usd": 0.1234, "duration_ms": 4321, "num_turns": 3,
         "usage": {"input_tokens": 100, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 3000,
                   "output_tokens": 250},
         "modelUsage": {"claude-sonnet-5-5": {"costUSD": 0.1234}}, "api_error_status": error},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_parse_stream_extracts_tools_tokens_cost_and_final_text():
    out = session.parse_stream(_stream("final <answer> text"))
    assert out["text"] == "final <answer> text"
    assert [c["name"] for c in out["tool_calls"]] == ["mcp__tep__get_alarms", "mcp__tep__query_historian"]
    assert out["tool_calls"][1]["input"] == {"tags": ["xmeas_1"], "start_h": 0, "end_h": 1}
    assert out["tokens"] == {"input": 100, "cache_creation": 2000, "cache_read": 3000, "output": 250, "total_in": 5100}
    assert out["cost_usd"] == pytest.approx(0.1234) and out["duration_s"] == pytest.approx(4.321)
    assert out["num_turns"] == 3 and out["model"] == "claude-sonnet-5-5" and out["models_used"] == ["claude-sonnet-5-5"]
    assert out["init"]["tools"] == ["mcp__tep__get_alarms"] and out["session_id"] == "s1"


def test_a_rate_limited_stream_raises_instead_of_recording_an_answer():
    extra = [{"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": 1792882200}}]
    with pytest.raises(session.RateLimited, match="resets"):
        session.parse_stream(_stream("You've hit your session limit", error=429, extra=extra))


def test_other_api_errors_raise_a_session_error():
    with pytest.raises(session.SessionError, match="500"):
        session.parse_stream(_stream("boom", error=500))


def test_a_stream_without_a_result_is_an_error():
    with pytest.raises(session.SessionError, match="no result"):
        session.parse_stream("")


def test_non_json_lines_are_ignored():
    out = session.parse_stream("warning: something\n" + _stream())
    assert out["text"] == "done"


def test_only_tep_tools_are_allowed_in_a_clean_transcript():
    out = session.parse_stream(_stream())
    assert session.outside_tools(out) == []
    bad = dict(out, tool_calls=out["tool_calls"] + [{"name": "Bash", "input": {"command": "ls"}}])
    assert session.outside_tools(bad) == ["Bash"]
    bad_init = dict(out, init={"tools": ["mcp__tep__get_alarms", "Read"], "mcp_servers": []})
    assert session.outside_tools(bad_init) == ["Read"]


def test_the_saved_sandbox_probe_shows_no_file_shell_or_web_access():
    path = ROOT / "results" / "phase-2b" / "sandbox_probe.json"
    if not path.exists():
        pytest.skip("run scripts/assistant_eval.py --probe first")
    probe = json.loads(path.read_text())
    assert probe["verdict"]["no_file_shell_or_web_access"] is True
    assert sorted(probe["verdict"]["tools_available_to_the_model"]) == sorted(session.ALLOWED_TOOLS)
    assert probe["verdict"]["tool_calls_made"] == [] and probe["machine"]


# --- the prompt hash survives line-ending changes ------------------------------------------------------
def test_prompt_hash_ignores_crlf_and_matches_a_hand_computed_sha256(tmp_path):
    import hashlib
    lf, crlf = tmp_path / "lf.md", tmp_path / "crlf.md"
    lf.write_bytes(b"line one\nline two\n")
    crlf.write_bytes(b"line one\r\nline two\r\n")
    expected = hashlib.sha256(b"line one\nline two\n").hexdigest()
    assert session.prompt_sha256(lf) == session.prompt_sha256(crlf) == expected


def test_the_frozen_prompt_hash_matches_the_prompt_file_in_any_line_ending(tmp_path):
    prompt = ROOT / "src" / "assistant" / "prompt.md"
    frozen = (ROOT / "results" / "phase-2b" / "prompt_frozen.sha256").read_text().strip()
    assert session.prompt_sha256(prompt) == frozen
    as_crlf = tmp_path / "prompt.md"
    as_crlf.write_bytes(prompt.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert session.prompt_sha256(as_crlf) == frozen


def test_gitattributes_keeps_the_prompt_and_notes_on_lf_line_endings():
    lines = (ROOT / ".gitattributes").read_text().splitlines()
    assert "src/assistant/prompt.md text eol=lf" in lines
    assert "plant-notes/*.md text eol=lf" in lines
