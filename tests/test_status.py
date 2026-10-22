"""Tests for .claude/status.py, the where-are-we / next-step helper."""
import importlib.util
import tempfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "status", Path(__file__).resolve().parent.parent / ".claude" / "status.py"
)
status = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(status)


def _root(setup=True, research=False):
    root = Path(tempfile.mkdtemp())
    if setup:
        (root / "docs/agents").mkdir(parents=True)
        (root / "docs/agents/issue-tracker.md").write_text("x")
    if research:
        (root / "research").mkdir()
        (root / "research/notes.md").write_text("x")
    return root


def test_missing_setup_asks_for_setup_command():
    assert "/setup-matt-pocock-skills" in status.next_step(_root(setup=False), set())[1]


def test_fresh_project_starts_phase_r():
    assert status.next_step(_root(), set()) == ("Phase R: research & scope", 'say "next"')


def test_research_notes_ready_asks_for_grilling():
    assert "/grill-with-docs" in status.next_step(_root(research=True), set())[1]


def test_first_untagged_phase_is_current():
    where, _ = status.next_step(_root(), {"phase-R", "phase-0", "phase-1"})
    assert where == "Phase 2: MSPC"


def test_all_phases_tagged_is_done():
    tags = {f"phase-{key}" for key, _ in status.PHASES}
    assert status.next_step(_root(), tags)[0] == "All phases done"
