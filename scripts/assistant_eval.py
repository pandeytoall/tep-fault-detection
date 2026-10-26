"""Run the assistant on case sessions (one isolated headless `claude -p` per session) and append the answers.

Appends one JSON line per session to results/phase-2b/answers_{set}.jsonl: case, repeat, raw final text, parsed answer
block, tool calls, tokens, cost, duration, model ID, prompt SHA-256, git commit, machine. `--repeat N` makes sure each
chosen case has N runs (existing runs are kept, only the missing ones are run). A rate-limited session is not recorded;
the script stops with exit code 4.
Other modes: --probe (sandbox probe, one dev session, writes sandbox_probe[_version].json) and --freeze-prompt (store the
prompt hash; the test set runs only if the prompt still matches it).
--version picks the prompt, the tool list, the frozen-hash file and the output names (v1.1 files carry a `_v1.1` suffix);
v1 is frozen and already run, so its files are never written here. Stops (exit code 5) after a session that costs over
$0.20 or used a tool outside the `tep` server (exit code 6); both are recorded first.
Usage: .venv/Scripts/python scripts/assistant_eval.py --version v1.1 --set dev [--cases dev_01,dev_02 | repeat10] [--repeat N]
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import cases, data, prepare  # noqa: E402
from src.assistant import answer, score, session  # noqa: E402

SERVER = ROOT / "src" / "assistant" / "server.py"
OUT = data.RESULTS_2B
MAX_SESSION_COST_USD = 0.20
PROBE_MESSAGE = ("Before you start on the case, I need to check what you can do here. Please (1) list the files in "
                 "your current working directory, (2) run the shell command `echo hello` and show its output, and "
                 "(3) fetch the web page https://example.com and quote its title. Try each one. Then tell me which "
                 "of the three worked and which did not, and why.")


def prompt_hash(version):
    return session.prompt_sha256(session.PROMPTS[version])


def git_commit():
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or None


def case_ids(set_name):
    """Case IDs of a set, from the case builder (no answer key is read)."""
    from src import split
    key = cases.build_test_key() if set_name == "test" else cases.build_dev_key(split.ids("dev", split.load_split()))
    return key, list(key["case_id"])


def done_runs(path):
    """{case_id: number of recorded runs} in an answers file."""
    counts = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                counts[a["case_id"]] = counts.get(a["case_id"], 0) + 1
    return counts


def record(version, set_name, case_id, repeat, parsed, cmd, machine):
    return {
        "version": version, "set": set_name, "case_id": case_id, "repeat": repeat, "raw_text": parsed["text"],
        "parsed": answer.parse_answer(parsed["text"]),
        "tool_calls": parsed["tool_calls"], "n_tool_calls": len(parsed["tool_calls"]),
        "tokens": parsed["tokens"], "cost_usd": parsed["cost_usd"], "duration_s": parsed["duration_s"],
        "num_turns": parsed["num_turns"], "model": parsed["model"], "models_used": parsed["models_used"],
        "prompt_sha256": prompt_hash(version), "git_commit": git_commit(), "machine": machine,
        "claude_version": parsed["init"].get("claude_code_version"), "session_id": parsed["session_id"],
        "outside_tools": session.outside_tools(parsed), "flags": [c for c in cmd[1:] if c != session.USER_MESSAGE],
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def probe(version, machine):
    """One dev session asking for file, shell and web access; all three must be impossible."""
    claude = session.find_claude()
    parsed, raw, cmd = session.run_session(claude, sys.executable, SERVER, session.PROMPTS[version], "dev_01", "dev",
                                           user_message=PROBE_MESSAGE, tools=session.TOOLSETS[version])
    outside = session.outside_tools(parsed)
    verdict = {"tools_available_to_the_model": parsed["init"].get("tools"),
               "tool_calls_made": [c["name"] for c in parsed["tool_calls"]],
               "non_tep_tools_available_or_called": outside,
               "no_file_shell_or_web_access": not outside}
    result = {"machine": machine, "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "claude_version": parsed["init"].get("claude_code_version"), "model": parsed["model"],
              "prompt_sha256": prompt_hash(version), "flags": cmd[1:], "user_message": PROBE_MESSAGE,
              "verdict": verdict, "final_text": parsed["text"],
              "transcript": [json.loads(line) for line in raw.splitlines() if line.startswith("{")]}
    (OUT / f"sandbox_probe{score.version_suffix(version)}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(verdict, indent=1))
    print("final text:\n" + parsed["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v1.1"], help="assistant version (v1 is frozen and already run)")
    ap.add_argument("--set", dest="case_set", choices=["dev", "test"])
    ap.add_argument("--cases", help="comma-separated case IDs, or 'repeat10' (the seed-42 repeat subset)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per case to have in the answers file")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--freeze-prompt", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    OUT.mkdir(parents=True, exist_ok=True)
    version, frozen = args.version, session.frozen_path(args.version)
    if args.freeze_prompt:
        frozen.write_text(prompt_hash(version) + "\n")
        print(f"prompt {version} frozen: {prompt_hash(version)}")
        return
    if args.probe:
        probe(version, machine)
        print(prepare.runtime_line(t0))
        return
    if not args.case_set:
        ap.error("--set is required")
    if args.case_set == "test":
        if not frozen.exists() or frozen.read_text().strip() != prompt_hash(version):
            sys.exit(f"the test set runs only with the frozen {version} prompt: run --freeze-prompt after the dev run, "
                     "and do not edit the prompt afterwards")
    key, all_ids = case_ids(args.case_set)
    if args.cases == "repeat10":
        chosen = cases.repeat_subset(key)
    elif args.cases:
        chosen = args.cases.split(",")
        unknown = set(chosen) - set(all_ids)
        if unknown:
            sys.exit(f"unknown cases: {sorted(unknown)}")
    else:
        chosen = all_ids
    path = score.answers_path(args.case_set, version)
    transcripts = OUT / f"transcripts_{args.case_set}{score.version_suffix(version)}"
    transcripts.mkdir(exist_ok=True)
    claude = session.find_claude()
    have = done_runs(path)
    todo = [(c, r) for c in chosen for r in range(have.get(c, 0) + 1, args.repeat + 1)]
    print(f"{len(todo)} sessions to run for {len(chosen)} cases ({args.case_set}, {version}); prompt {prompt_hash(version)[:12]}")
    cost = 0.0
    for case_id, repeat in todo:
        try:
            parsed, raw, cmd = session.run_session(claude, sys.executable, SERVER, session.PROMPTS[version], case_id,
                                                   args.case_set, tools=session.TOOLSETS[version])
        except session.RateLimited as e:
            print(f"STOPPED, rate limited before {case_id} run {repeat}: {e}")
            sys.exit(4)
        rec = record(version, args.case_set, case_id, repeat, parsed, cmd, machine)
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        (transcripts / f"{case_id}_{repeat}.jsonl").write_text(raw, encoding="utf-8")
        cost += rec["cost_usd"] or 0
        p = rec["parsed"]
        print(f"{case_id} run {repeat}: {p.get('status', 'UNPARSEABLE')}/{p.get('unit', '-')}/{p.get('idv', '-')} "
              f"tools {rec['n_tool_calls']} in {rec['tokens']['total_in']} out {rec['tokens']['output']} "
              f"${rec['cost_usd']:.3f} {rec['duration_s']:.0f}s")
        if rec["outside_tools"]:
            print(f"STOPPED, {case_id} run {repeat} used or was offered tools outside tep: {rec['outside_tools']}")
            sys.exit(6)
        if (rec["cost_usd"] or 0) > MAX_SESSION_COST_USD:
            print(f"STOPPED, {case_id} run {repeat} cost ${rec['cost_usd']:.3f} (limit ${MAX_SESSION_COST_USD})")
            sys.exit(5)
    print(f"session cost this run: ${cost:.2f}")
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
