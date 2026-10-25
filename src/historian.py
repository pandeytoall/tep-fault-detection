"""The replayed historian (phase 2b): one SQLite file per case set, an alarm log, and the per-case read helpers.

Tables (nothing else; no fault number, run ID, crop or label anywhere in the file):
- readings(case_id, t_hours, tag, value): tag is xmeas_N or xmv_N; index on (case_id, tag, t_hours).
- alarms(case_id, t_hours, statistic, event, value, limit): PCA T2 and SPE against their 1% limits. A `raise` is the
  3rd of 3 consecutive samples above the limit; a `clear` is the 3rd of 3 consecutive samples not above it after a raise.
The writer runs at build time (it needs the PCA monitor); the MCP server only uses CaseView.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ID_COLUMNS, VARIABLES

IN_A_ROW = 3
LIMIT_NAME = "1%"
STATISTICS = ("t2", "spe")


def alarm_events(t_hours, values, limit, n=IN_A_ROW):
    """[(t_hours, 'raise'|'clear', value)] for one statistic. Above means value > limit."""
    events, above_run, below_run, alarmed = [], 0, 0, False
    for t, v in zip(t_hours, values):
        if v > limit:
            above_run, below_run = above_run + 1, 0
        else:
            above_run, below_run = 0, below_run + 1
        if not alarmed and above_run >= n:
            alarmed = True
            events.append((float(t), "raise", float(v)))
        elif alarmed and below_run >= n:
            alarmed = False
            events.append((float(t), "clear", float(v)))
    return events


def score_case(monitor, matrix):
    """DataFrame t_hours, t2, spe: the (lag-free) PCA scores of one case matrix (t_hours + the 52 variables)."""
    if monitor.lags != 0:
        raise ValueError("the assistant uses PCA without lags")
    frame = matrix[VARIABLES].copy()
    frame.insert(0, "sample", np.arange(1, len(frame) + 1))
    frame.insert(0, "simulationRun", 1)
    frame.insert(0, "faultNumber", 0)
    scores = monitor.score(frame[ID_COLUMNS + VARIABLES])
    return pd.DataFrame({"t_hours": matrix["t_hours"].to_numpy(), "t2": scores["t2"].to_numpy(),
                         "spe": scores["spe"].to_numpy()})


def write_historian(path, frames, monitor):
    """Write readings and the alarm log for {case_id: frame (t_hours + 52 variables)}. Replaces an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute("pragma journal_mode = off")
    con.execute("pragma synchronous = off")
    con.execute("create table readings (case_id text, t_hours real, tag text, value real)")
    con.execute('create table alarms (case_id text, t_hours real, statistic text, event text, value real, "limit" real)')
    for case_id, frame in frames.items():
        t = frame["t_hours"].to_numpy(dtype=float)
        rows = ((case_id, float(t[i]), tag, float(v))
                for tag in VARIABLES for i, v in enumerate(frame[tag].to_numpy(dtype=np.float64)))
        con.executemany("insert into readings values (?, ?, ?, ?)", rows)
    con.execute("create index readings_case_tag_time on readings (case_id, tag, t_hours)")
    con.commit()
    con.close()
    # alarms come from the values as read back, the same path the server scores
    con = sqlite3.connect(path)
    for case_id in frames:
        scores = score_case(monitor, CaseView._matrix_from(con, case_id))
        for stat in STATISTICS:
            limit = float(monitor.limits[stat][LIMIT_NAME])
            con.executemany("insert into alarms values (?, ?, ?, ?, ?, ?)",
                            [(case_id, t, stat, event, v, limit) for t, event, v in
                             alarm_events(scores["t_hours"].to_numpy(), scores[stat].to_numpy(), limit)])
    con.commit()
    con.close()


class CaseView:
    """Read-only access to one case of a historian file. Every query is bound to that case_id."""

    def __init__(self, path, case_id):
        self.case_id = str(case_id)
        self._con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        n = self._con.execute("select count(*) from readings where case_id = ?", (self.case_id,)).fetchone()[0]
        if n == 0:
            raise ValueError(f"no readings for case {case_id!r}")

    def close(self):
        self._con.close()

    @staticmethod
    def _matrix_from(con, case_id):
        long = pd.read_sql_query("select t_hours, tag, value from readings where case_id = ?", con, params=(case_id,))
        wide = long.pivot(index="t_hours", columns="tag", values="value").sort_index()[VARIABLES]
        return wide.reset_index()

    def matrix(self):
        """The whole case as DataFrame t_hours + the 52 variables (float64)."""
        return self._matrix_from(self._con, self.case_id)

    def readings(self, tags, start_h=None, end_h=None, step=1):
        """DataFrame t_hours + one column per tag (in the order given). Window inclusive; every step-th sample."""
        tags = list(tags)
        bad = [t for t in tags if t not in VARIABLES]
        if bad:
            raise ValueError(f"unknown tag(s): {bad}; use xmeas_1..xmeas_41 and xmv_1..xmv_11")
        if not tags:
            raise ValueError("give at least one tag")
        if int(step) < 1:
            raise ValueError("step must be a whole number of samples, 1 or more")
        lo = -np.inf if start_h is None else float(start_h) - 1e-6
        hi = np.inf if end_h is None else float(end_h) + 1e-6
        marks = ",".join("?" * len(tags))
        long = pd.read_sql_query(
            f"select t_hours, tag, value from readings where case_id = ? and tag in ({marks}) "
            f"and t_hours >= ? and t_hours <= ?", self._con, params=[self.case_id, *tags, lo, hi])
        wide = long.pivot(index="t_hours", columns="tag", values="value").sort_index()
        wide = wide.reindex(columns=tags).iloc[::int(step)]
        return wide.reset_index()

    def alarms(self, start_h=None, end_h=None):
        """The alarm log of this case as DataFrame t_hours, statistic, event, value, limit."""
        lo = -np.inf if start_h is None else float(start_h) - 1e-6
        hi = np.inf if end_h is None else float(end_h) + 1e-6
        return pd.read_sql_query(
            'select t_hours, statistic, event, value, "limit" from alarms where case_id = ? '
            "and t_hours >= ? and t_hours <= ? order by t_hours, statistic", self._con,
            params=[self.case_id, lo, hi])
