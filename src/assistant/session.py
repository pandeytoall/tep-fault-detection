"""One isolated headless `claude -p` session against the `tep` MCP server, and how its output is read.

Isolation (checked with claude 2.1.310; `--bare` is not usable because it needs an API key and this machine logs in
with OAuth, so the same effect comes from these flags):
- `--tools ""`            no built-in tools at all (no file, shell, web or agent tools);
- `--strict-mcp-config --mcp-config <json>`  only the `tep` server, started with `--case <id> --set <set>`;
- `--allowedTools mcp__tep__*`  the six tep tools only; `--permission-prompts none` denies everything else;
- `--setting-sources ""`  no user, project or local settings (so no user hooks, plugins or enabled plugins);
- `--disable-slash-commands`  no skills or commands;
- `--system-prompt-file src/assistant/prompt.md`  replaces the default Claude Code system prompt;
- `--no-session-persistence`, a fresh empty temp folder as cwd, CLAUDE_CODE_DISABLE_AUTO_MEMORY=1.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

MODEL = "claude-sonnet-5-5"
TOOLS = ["query_historian", "get_alarms", "score_pca", "top_contributors", "list_notes", "read_note"]
ALLOWED_TOOLS = [f"mcp__tep__{t}" for t in TOOLS]
MAX_BUDGET_USD = 2
TIMEOUT_S = 900
USER_MESSAGE = "Diagnose the plant case in front of you."


class SessionError(RuntimeError):
    """The session did not produce a usable result."""


class RateLimited(SessionError):
    """The account's usage limit was hit; no answer was produced and nothing should be recorded."""


def prompt_sha256(path):
    """SHA-256 of a text file with CRLF read as LF, so a checkout with other line endings keeps the same hash."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def find_claude():
    """Path of the claude executable (CLAUDE_BIN, then PATH, then the npm global install next to claude.cmd)."""
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    found = shutil.which("claude.exe")
    if found:
        return found
    shim = shutil.which("claude.cmd") or shutil.which("claude")
    if shim:
        candidate = Path(shim).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if candidate.exists():
            return str(candidate)
        return shim
    raise SessionError("claude CLI not found; set CLAUDE_BIN")


def mcp_config(python, server_script, case_id, set_name):
    """JSON string for --mcp-config: one stdio server `tep`, bound to one case."""
    return json.dumps({"mcpServers": {"tep": {"command": str(python), "args": [
        str(server_script), "--case", case_id, "--set", set_name]}}})


def build_command(claude_bin, user_message, prompt_path, mcp_json, model=MODEL):
    """argv of the isolated session (see the module docstring for what each flag does)."""
    return [str(claude_bin), "-p", user_message, "--model", model, "--output-format", "stream-json", "--verbose",
            "--tools", "", "--strict-mcp-config", "--mcp-config", mcp_json,
            "--allowedTools", ",".join(ALLOWED_TOOLS), "--permission-prompts", "none",
            "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
            "--system-prompt-file", str(prompt_path), "--max-budget-usd", str(MAX_BUDGET_USD)]


def session_env(base):
    """Environment for the session: the caller's, with auto-memory off."""
    return {**base, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}


def parse_stream(text):
    """Read claude's stream-json output.

    Returns dict: text (final reply), tool_calls [{name, input}], tokens {input, cache_creation, cache_read, output,
    total_in}, cost_usd, duration_s, num_turns, model, models_used, init (the system init event), session_id.
    Raises RateLimited when the account limit was hit, SessionError on any other failure.
    """
    init, result, calls, rejected = None, None, [], None
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            init = event
        elif kind == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    calls.append({"name": block["name"], "input": block.get("input", {})})
        elif kind == "rate_limit_event" and event.get("rate_limit_info", {}).get("status") == "rejected":
            rejected = event["rate_limit_info"]
        elif kind == "result":
            result = event
    if result is None:
        raise SessionError("no result event in the claude output")
    status = result.get("api_error_status")
    if result.get("is_error") and (status == 429 or rejected):
        raise RateLimited(f"{result.get('result', 'rate limited')} (resetsAt {(rejected or {}).get('resetsAt')})")
    if result.get("is_error"):
        raise SessionError(f"claude returned an error (api status {status}): {result.get('result')}")
    usage = result.get("usage", {})
    tokens = {"input": usage.get("input_tokens", 0), "cache_creation": usage.get("cache_creation_input_tokens", 0),
              "cache_read": usage.get("cache_read_input_tokens", 0), "output": usage.get("output_tokens", 0)}
    tokens["total_in"] = tokens["input"] + tokens["cache_creation"] + tokens["cache_read"]
    models = sorted(result.get("modelUsage", {}))
    return {"text": result.get("result", ""), "tool_calls": calls, "tokens": tokens,
            "cost_usd": result.get("total_cost_usd"), "duration_s": result.get("duration_ms", 0) / 1000,
            "num_turns": result.get("num_turns"), "model": (init or {}).get("model"), "models_used": models,
            "init": init or {}, "session_id": (init or {}).get("session_id")}


def outside_tools(parsed):
    """Names of tools in the session's init list or calls that are not tep tools (should be empty)."""
    names = list(parsed.get("init", {}).get("tools", [])) + [c["name"] for c in parsed["tool_calls"]]
    return [n for n in names if not n.startswith("mcp__tep__")]


def run_session(claude_bin, python, server_script, prompt_path, case_id, set_name, user_message=USER_MESSAGE,
                timeout=TIMEOUT_S):
    """Run one session in a fresh empty temp folder. Returns (parsed stream dict, raw stdout, argv used)."""
    cmd = build_command(claude_bin, user_message, prompt_path, mcp_config(python, server_script, case_id, set_name))
    with tempfile.TemporaryDirectory(prefix="tep_session_") as cwd:
        if any(Path(cwd).iterdir()):
            raise SessionError("session folder is not empty")
        proc = subprocess.run(cmd, cwd=cwd, env=session_env(os.environ), stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=timeout)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    try:
        parsed = parse_stream(stdout)
    except SessionError as e:
        if isinstance(e, RateLimited):
            raise
        raise SessionError(f"{e}; exit code {proc.returncode}; stderr: {proc.stderr.decode('utf-8', 'replace')[:500]}")
    return parsed, stdout, cmd
