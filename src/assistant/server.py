"""The `tep` MCP server (stdio): tools over one case of a replayed historian, the PCA monitor and the plant notes.

Started per session as `python src/assistant/server.py --case <id> --set <test|dev>`. It sees only that case. It reads
only its set's SQLite file (readings and alarm log), the PCA model file (loadings, 1% limits, normal reference) and the
plant-notes folder; it never opens an answer key or a data file.
Tools: query_historian, get_alarms, score_pca, top_contributors, list_notes, read_note.
"""
import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:  # run as a script from any working directory
    sys.path.insert(0, str(ROOT))

from src import contributions, historian, mspc  # noqa: E402

MAX_VALUES = 2000
DEFAULT_MODEL = ROOT / "results" / "phase-2" / "pca.npz"
DEFAULT_NOTES = ROOT / "plant-notes"
TOLERANCE_H = 1e-6


def _round(values, digits=5):
    """Plain floats with `digits` significant digits (keeps the output short)."""
    return [float(f"{v:.{digits}g}") for v in values]


def _dump(obj):
    return json.dumps(obj, separators=(",", ":"))


class TepTools:
    """The tool logic for one case, independent of MCP (the MCP layer in build_server only wraps it)."""

    def __init__(self, db_path, case_id, model_path=DEFAULT_MODEL, notes_dir=DEFAULT_NOTES, max_values=MAX_VALUES):
        self.view = historian.CaseView(db_path, case_id)
        self.monitor = mspc.PCAMonitor.load(model_path)
        self.reference = contributions.load_reference(model_path)
        if self.monitor.limits is None or self.reference is None:
            raise ValueError("the model file has no limits or normal reference; run scripts/assistant_build.py")
        self.limits = {s: float(self.monitor.limits[s][historian.LIMIT_NAME]) for s in historian.STATISTICS}
        self.notes_dir = Path(notes_dir)
        self.max_values = int(max_values)
        self._matrix = self.view.matrix()
        self._scores = historian.score_case(self.monitor, self._matrix)
        self.span = [float(self._matrix["t_hours"].iloc[0]), float(self._matrix["t_hours"].iloc[-1])]

    # --- helpers ---------------------------------------------------------------------------------
    def _window(self, frame, start_h, end_h, step):
        """Rows of a frame with a t_hours column inside [start_h, end_h], every step-th one."""
        if int(step) < 1:
            raise ValueError("step must be a whole number of samples, 1 or more")
        t = frame["t_hours"].to_numpy()
        keep = (t >= float(start_h) - TOLERANCE_H) & (t <= float(end_h) + TOLERANCE_H)
        rows = frame[keep].iloc[::int(step)]
        if rows.empty:
            raise ValueError(f"no samples between {start_h} and {end_h} h; this case runs from "
                             f"{self.span[0]} to {self.span[1]} h")
        return rows

    def _capped(self, rows, columns):
        """(rows kept, note or None): at most max_values values, counting every column of every sample."""
        per_sample = len(columns)
        keep = max(1, self.max_values // per_sample)
        if len(rows) <= keep:
            return rows, None
        note = (f"Output truncated to the first {keep} of {len(rows)} samples ({self.max_values}-value cap). "
                "Use a larger step, a narrower window or fewer tags to see the rest.")
        return rows.iloc[:keep], note

    # --- tools --------------------------------------------------------------------------------------
    def query_historian(self, tags, start_h, end_h, step=1):
        """Readings of the given tags between start_h and end_h hours (inclusive), every step-th sample."""
        rows = self._window(self.view.readings(tags, None, None, 1), start_h, end_h, step)
        rows, note = self._capped(rows, list(tags))
        out = {"tags": list(tags), "t_hours": [float(round(t, 2)) for t in rows["t_hours"]],
               "values": {tag: _round(rows[tag]) for tag in tags}, "truncated": note is not None}
        if note:
            out["note"] = note
        return _dump(out)

    def get_alarms(self, start_h=None, end_h=None):
        """PCA alarm log (T2 and SPE against their 1% limits) with an optional hour window."""
        log = self.view.alarms(start_h, end_h)
        out = {"alarms": [{"t_hours": float(r.t_hours), "statistic": r.statistic, "event": r.event,
                           "value": float(f"{r.value:.5g}"), "limit": float(f"{r.limit:.5g}")}
                          for r in log.itertuples()],
               "case_span_hours": self.span,
               "note": "raise = 3rd of 3 consecutive samples above the 1% limit; clear = 3rd of 3 consecutive "
                       "samples not above it after a raise."}
        if not out["alarms"]:
            out["note"] = "no alarm events in this window. " + out["note"]
        return _dump(out)

    def score_pca(self, start_h, end_h, step=1):
        """PCA T2 and SPE per sample between start_h and end_h (inclusive), every step-th sample, plus the 1% limits."""
        rows = self._window(self._scores, start_h, end_h, step)
        rows, note = self._capped(rows, ["t2", "spe"])
        out = {"t_hours": [float(round(t, 2)) for t in rows["t_hours"]], "t2": _round(rows["t2"]),
               "spe": _round(rows["spe"]), "limits": self.limits, "truncated": note is not None}
        if note:
            out["note"] = note
        return _dump(out)

    def top_contributors(self, statistic, start_h=None, end_h=None, n=5):
        """Readings contributing most to T2 or SPE over a window, as a ratio to their normal contribution."""
        stat = str(statistic).lower()
        if stat not in contributions.STATISTICS:
            raise ValueError(f"statistic must be 't2' or 'spe', got {statistic!r}")
        label = {"t2": "T2", "spe": "SPE"}[stat]
        if start_h is None and end_h is None:
            first = contributions.window_start(self.view.alarms(), stat)
            if first is None:
                window = f"default: whole case (no {label} raise in the alarm log)"
            else:
                window = f"default: from the first {label} raise to the end of the case"
            start = first
        else:
            window, start = "as requested", start_h
        rows = contributions.window_rows(self._matrix, start, end_h)
        top = contributions.top_contributors(self.monitor, self.reference, rows, stat, max(1, min(int(n), 10)))
        return _dump({"statistic": stat, "window": window,
                      "window_hours": [float(rows["t_hours"].iloc[0]), float(rows["t_hours"].iloc[-1])],
                      "n_samples": int(len(rows)),
                      "note": "ratio = mean contribution in the window / mean contribution in normal operation",
                      "top": top})

    def _notes(self):
        return {p.stem: p for p in sorted(self.notes_dir.glob("*.md"))}

    def list_notes(self):
        """Names and titles of the plant notes."""
        notes = []
        for name, path in self._notes().items():
            first = next((ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()), name)
            notes.append({"name": name, "title": first.lstrip("# ").strip()})
        return _dump({"notes": notes})

    def read_note(self, name):
        """The text of one plant note, by name from list_notes."""
        notes = self._notes()
        key = str(name).strip()
        key = key[:-3] if key.endswith(".md") else key
        if key not in notes:
            raise ValueError(f"no such note {name!r}; available: {sorted(notes)}")
        return notes[key].read_text(encoding="utf-8")


def build_server(tools):
    """Wrap a TepTools in an MCP server named `tep` (errors are returned as readable tool errors)."""
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = MCPServer("tep", instructions="Tools over one replayed TEP case: historian, PCA alarm log, PCA scores, "
                                        "top contributing readings and plant notes.")

    def guarded(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ValueError as e:
                raise ToolError(str(e)) from e
        return run

    @mcp.tool()
    @guarded
    def query_historian(tags: list[str], start_h: float, end_h: float, step: int = 1) -> str:
        """Readings of tags (xmeas_1..xmeas_41, xmv_1..xmv_11) between start_h and end_h hours on the case clock
        (inclusive), every step-th sample (3 minutes apart). At most 2000 values come back; if the request is
        larger the output is truncated and says so."""
        return tools.query_historian(tags, start_h, end_h, step)

    @mcp.tool()
    @guarded
    def get_alarms(start_h: float | None = None, end_h: float | None = None) -> str:
        """The PCA alarm log: raise and clear events for T2 and SPE against their 1% limits, optionally between
        start_h and end_h hours. Also gives the span of the case in hours."""
        return tools.get_alarms(start_h, end_h)

    @mcp.tool()
    @guarded
    def score_pca(start_h: float, end_h: float, step: int = 1) -> str:
        """PCA T2 and SPE per sample between start_h and end_h hours (inclusive), every step-th sample, with the
        two 1% alarm limits. At most 2000 values come back (T2 and SPE each count)."""
        return tools.score_pca(start_h, end_h, step)

    @mcp.tool()
    @guarded
    def top_contributors(statistic: str, start_h: float | None = None, end_h: float | None = None, n: int = 5) -> str:
        """The n readings (default 5) that contribute most to T2 or SPE ('t2' or 'spe') over a window, with name,
        unit, plant area and ratio to their normal contribution. Default window: from the first raise of that
        statistic to the end of the case, or the whole case if it never raised."""
        return tools.top_contributors(statistic, start_h, end_h, n)

    @mcp.tool()
    @guarded
    def list_notes() -> str:
        """Names and titles of the plant notes (process description, tag list, disturbance list, monitor guide)."""
        return tools.list_notes()

    @mcp.tool()
    @guarded
    def read_note(name: str) -> str:
        """The full text of one plant note, by the name that list_notes gives."""
        return tools.read_note(name)

    return mcp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", required=True)
    parser.add_argument("--set", required=True, choices=["test", "dev"], dest="case_set")
    parser.add_argument("--db", help="override the historian file (tests)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--notes", default=str(DEFAULT_NOTES))
    args = parser.parse_args(argv)
    db = args.db or ROOT / "data" / "processed" / f"historian_{args.case_set}.sqlite"
    build_server(TepTools(db, args.case, args.model, args.notes)).run()


if __name__ == "__main__":
    main()
