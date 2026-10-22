"""Where the TEP project is and the one next step, derived from git phase tags and files.

Used by the status line (`--line`) and the SessionStart hook (no arguments).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES = [
    ("R", "research & scope"),
    ("0", "setup & data"),
    ("1", "labels & split"),
    ("2", "MSPC"),
    ("3", "ML models"),
    ("4", "diagnosis"),
    ("5", "report"),
]


def phase_tags(root=ROOT):
    out = subprocess.run(
        ["git", "-C", str(root), "tag", "--list", "phase-*"],
        capture_output=True, text=True,
    ).stdout
    return set(out.split())


def next_step(root, tags):
    """Return (where, next step) for the first phase without a `phase-<key>` tag."""
    if not (root / "docs/agents/issue-tracker.md").exists():
        return "Setup incomplete", 'type "/setup-matt-pocock-skills" (choose local markdown)'
    for key, name in PHASES:
        if f"phase-{key}" not in tags:
            break
    else:
        return "All phases done", 'say "next" to review the report'
    where = f"Phase {key}: {name}"
    if key == "R" and any((root / "research").glob("*.md")):
        return where, 'type "/grill-with-docs" to settle the scope'
    return where, 'say "next"'


if __name__ == "__main__":
    where, step = next_step(ROOT, phase_tags())
    if "--line" in sys.argv:
        print(f"TEP | {where} | next: {step}")
    else:
        print(f"TEP project status (from git tags and files): {where}. Next step for the user: {step}.")
