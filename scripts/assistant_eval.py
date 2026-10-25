"""Run the assistant on case sessions (one isolated headless `claude -p` per session) and append the answers.

Appends one JSON line per session to results/phase-2b/answers_{set}.jsonl: case, repeat, raw final text, parsed answer
block, tool calls, tokens, cost, duration, model ID, prompt SHA-256, git commit, machine. `--repeat N` makes sure each
chosen case has N runs (existing runs are kept, only the missing ones are run). A rate-limited session is not recorded;
the script stops with exit code 4.
Other modes: --probe (sandbox probe, one dev session, writes sandbox_probe.json) and --freeze-prompt (store the prompt
hash; the test set runs only if the prompt still matches it).
Usage: .venv/Scripts/python scripts/assistant_eval.py --set dev [--cases dev_01,dev_02 | repeat10] [--repeat N]
"""
import argparse
import hashlib
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
from src import cases, prepare  # noqa: E402
from src.assistant import answer, session  # noqa: E402

PROMPT = ROOT / "src" / "assistant" / "prompt.md"
SERVER = ROOT / "src" / "assistant" / "server.py"
OUT = ROOT / "results" / "phase-2b"
FROZEN = OUT / "prompt_frozen.sha256"
PROBE_MESSAGE = ("Before you start on the case, I need to check what you can do here. Please (1) list the files in "
                 "your current working directory, (2) run the shell command `echo hello` and show its output, and "
                 "(3) fetch the web page https://example.com and quote its title. Try each one. Then tell me which "
                 "of the three worked and which did not, and why.")


def prompt_hash():
    return hashlib.sha256(PROMPT.read_bytes()).hexdigest()


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


def record(set_name, case_id, repeat, parsed, cmd, machine):
    return {
        "set": set_name, "case_id": case_id, "repeat": repeat, "raw_text": parsed["text"],
        "parsed": answer.parse_answer(parsed["text"]),
        "tool_calls": parsed["tool_calls"], "n_tool_calls": len(parsed["tool_calls"]),
        "tokens": parsed["tokens"], "cost_usd": parsed["cost_usd"], "duration_s": parsed["duration_s"],
        "num_turns": parsed["num_turns"], "model": parsed["model"], "models_used": parsed["models_used"],
        "prompt_sha256": prompt_hash(), "git_commit": git_commit(), "machine": machine,
        "claude_version": parsed["init"].get("claude_code_version"), "session_id": parsed["session_id"],
        "outside_tools": session.outside_tools(parsed), "flags": [c for c in cmd[1:] if c != session.USER_MESSAGE],
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def probe(machine):
    """One dev session asking for file, shell and web access; all three must be impossible."""
    claude = session.find_claude()
    parsed, raw, cmd = session.run_session(claude, sys.executable, SERVER, PROMPT, "dev_01", "dev",
                                           user_message=PROBE_MESSAGE)
    outside = session.outside_tools(parsed)
    verdict = {"tools_available_to_the_model": parsed["init"].get("tools"),
               "tool_calls_made": [c["name"] for c in parsed["tool_calls"]],
               "non_tep_tools_available_or_called": outside,
               "no_file_shell_or_web_access": not outside}
    result = {"machine": machine, "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "claude_version": parsed["init"].get("claude_code_version"), "model": parsed["model"],
              "prompt_sha256": prompt_hash(), "flags": cmd[1:], "user_message": PROBE_MESSAGE,
              "verdict": verdict, "final_text": parsed["text"],
              "transcript": [json.loads(line) for line in raw.splitlines() if line.startswith("{")]}
    (OUT / "sandbox_probe.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(verdict, indent=1))
    print("final text:\n" + parsed["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="case_set", choices=["dev", "test"])
    ap.add_argument("--cases", help="comma-separated case IDs, or 'repeat10' (the seed-42 repeat subset)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per case to have in the answers file")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--freeze-prompt", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.freeze_prompt:
        FROZEN.write_text(prompt_hash() + "\n")
        print(f"prompt frozen: {prompt_hash()}")
        return
    if args.probe:
        probe(machine)
        print(prepare.runtime_line(t0))
        return
    if not args.case_set:
        ap.error("--set is required")
    if args.case_set == "test":
        if not FROZEN.exists() or FROZEN.read_text().strip() != prompt_hash():
            sys.exit("the test set runs only with the frozen prompt: run --freeze-prompt after the dev run, "
                     "and do not edit prompt.md afterwards")
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
    path = OUT / f"answers_{args.case_set}.jsonl"
    transcripts = OUT / f"transcripts_{args.case_set}"
    transcripts.mkdir(exist_ok=True)
    claude = session.find_claude()
    have = done_runs(path)
    todo = [(c, r) for c in chosen for r in range(have.get(c, 0) + 1, args.repeat + 1)]
    print(f"{len(todo)} sessions to run for {len(chosen)} cases ({args.case_set}); prompt {prompt_hash()[:12]}")
    cost = 0.0
    for case_id, repeat in todo:
        try:
            parsed, raw, cmd = session.run_session(claude, sys.executable, SERVER, PROMPT, case_id, args.case_set)
        except session.RateLimited as e:
            print(f"STOPPED, rate limited before {case_id} run {repeat}: {e}")
            sys.exit(4)
        rec = record(args.case_set, case_id, repeat, parsed, cmd, machine)
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        (transcripts / f"{case_id}_{repeat}.jsonl").write_text(raw, encoding="utf-8")
        cost += rec["cost_usd"] or 0
        p = rec["parsed"]
        print(f"{case_id} run {repeat}: {p.get('status', 'UNPARSEABLE')}/{p.get('unit', '-')}/{p.get('idv', '-')} "
              f"tools {rec['n_tool_calls']} in {rec['tokens']['total_in']} out {rec['tokens']['output']} "
              f"${rec['cost_usd']:.3f} {rec['duration_s']:.0f}s")
    print(f"session cost this run: ${cost:.2f}")
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
