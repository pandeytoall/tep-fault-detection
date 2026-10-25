"""Phase 2b tests for src/historian.py: alarm events, the SQLite schema and the per-case read helpers."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import data, historian, mspc

T = 0.05  # hours per sample


# --- alarm events (hand-made series) ------------------------------------------------------
def _events(values, limit=1.0):
    t = np.round(np.arange(len(values)) * T, 2)
    return historian.alarm_events(t, np.asarray(values, dtype=float), limit)


def test_raise_at_the_third_of_three_samples_above_and_clear_at_the_third_below():
    #        0    1    2    3    4    5    6    7    8    9    10   11   12   13   14
    values = [2.0, 2.0, 0.5, 2.0, 2.0, 2.5, 3.0, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5, 2.0, 2.0]
    assert _events(values) == [(0.25, "raise", 2.5), (0.6, "clear", 0.5)]


def test_two_in_a_row_is_not_enough_and_an_equal_value_is_not_above():
    assert _events([2.0, 2.0, 0.5, 2.0, 2.0, 0.5]) == []
    assert _events([2.0, 2.0, 1.0, 2.0]) == []            # 1.0 == limit: not above
    assert _events([2.0, 2.0, 2.0]) == [(0.1, "raise", 2.0)]


def test_a_clear_needs_a_raise_first_and_three_samples_not_above():
    assert _events([0.1] * 10) == []                        # never raised, never cleared
    # raised at idx 2; below only twice, above again, so no clear; no second raise while still in alarm
    assert _events([2, 2, 2, 0.5, 0.5, 2, 2, 2, 2.0]) == [(0.1, "raise", 2.0)]


def test_alarm_can_raise_again_after_a_clear():
    values = [2, 2, 2, 0, 0, 0, 2, 2, 2.0]
    assert _events(values) == [(0.1, "raise", 2.0), (0.25, "clear", 0.0), (0.4, "raise", 2.0)]


# --- a small historian ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def monitor():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3000, 52)) @ np.diag(np.linspace(0.5, 2, 52)) + np.linspace(10, 20, 52)
    frame = pd.DataFrame(x, columns=data.VARIABLES)
    frame.insert(0, "sample", np.tile(np.arange(1, 501), 6))
    frame.insert(0, "simulationRun", np.repeat(np.arange(1, 7), 500))
    frame.insert(0, "faultNumber", 0)
    m = mspc.PCAMonitor(lags=0, cpv=0.9).fit(frame)
    scores = m.score(frame)
    m.limits = {s: {"1%": float(np.quantile(scores[s], 0.99))} for s in mspc.STATS}
    return m


def _case_frame(seed, n=120, offset=0.0, shift_from=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 52)) @ np.diag(np.linspace(0.5, 2, 52)) + np.linspace(10, 20, 52) + offset
    if shift_from is not None:
        x[shift_from:, 0] += 30.0  # a large step on xmeas_1
    frame = pd.DataFrame(x.astype(np.float32), columns=data.VARIABLES)
    frame.insert(0, "t_hours", np.round(np.arange(n) * T, 2))
    return frame


@pytest.fixture(scope="module")
def db(tmp_path_factory, monitor):
    path = tmp_path_factory.mktemp("hist") / "historian_dev.sqlite"
    frames = {"dev_01": _case_frame(1, shift_from=60), "dev_02": _case_frame(2, offset=100.0)}
    historian.write_historian(path, frames, monitor)
    return path, frames


def test_schema_has_only_readings_and_alarms_and_no_label_columns(db):
    path, _ = db
    con = sqlite3.connect(path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type = 'table'")}
    assert tables == {"readings", "alarms"}
    assert [r[1] for r in con.execute("pragma table_info(readings)")] == ["case_id", "t_hours", "tag", "value"]
    assert [r[1] for r in con.execute("pragma table_info(alarms)")] == [
        "case_id", "t_hours", "statistic", "event", "value", "limit"]
    forbidden = ("fault", "run", "crop", "label", "onset", "sample", "idv")
    columns = [r[1] for t in tables for r in con.execute(f"pragma table_info({t})")]
    assert not [c for c in columns if any(f in c.lower() for f in forbidden)]
    con.close()


def test_readings_have_an_index_on_case_tag_and_time(db):
    path, _ = db
    con = sqlite3.connect(path)
    cols = [[r[2] for r in con.execute(f"pragma index_info({idx[1]})")]
            for idx in con.execute("pragma index_list(readings)")]
    con.close()
    assert ["case_id", "tag", "t_hours"] in cols


def test_every_reading_is_stored_once_with_its_exact_value(db):
    path, frames = db
    con = sqlite3.connect(path)
    assert con.execute("select count(*) from readings").fetchone()[0] == 2 * 120 * 52
    got = con.execute("select value from readings where case_id='dev_01' and tag='xmv_5' order by t_hours").fetchall()
    tags = {r[0] for r in con.execute("select distinct tag from readings")}
    con.close()
    assert np.array_equal(np.array(got)[:, 0], frames["dev_01"]["xmv_5"].to_numpy(dtype=np.float64))
    assert tags == set(data.VARIABLES)


def test_alarm_log_matches_the_pca_scores_and_the_event_rule(db, monitor):
    path, frames = db
    view = historian.CaseView(path, "dev_01")
    log = view.alarms()
    scores = historian.score_case(monitor, view.matrix())
    assert set(log["statistic"]) <= {"t2", "spe"} and set(log["event"]) <= {"raise", "clear"}
    for stat in ("t2", "spe"):
        expected = historian.alarm_events(scores["t_hours"].to_numpy(), scores[stat].to_numpy(),
                                          monitor.limits[stat]["1%"])
        got = log[log["statistic"] == stat]
        assert [(r.t_hours, r.event, r.value) for r in got.itertuples()] == expected
        assert (got["limit"] == monitor.limits[stat]["1%"]).all()
    spe_raises = log[(log["statistic"] == "spe") & (log["event"] == "raise")]
    assert len(spe_raises) >= 1 and spe_raises["t_hours"].iloc[0] >= 60 * T  # the step starts at sample 60


def test_a_view_sees_only_its_own_case(db):
    path, frames = db
    a, b = historian.CaseView(path, "dev_01"), historian.CaseView(path, "dev_02")
    assert np.allclose(a.matrix()["xmeas_3"], frames["dev_01"]["xmeas_3"])
    assert np.allclose(b.matrix()["xmeas_3"], frames["dev_02"]["xmeas_3"])
    assert not np.allclose(a.matrix()["xmeas_3"], b.matrix()["xmeas_3"])
    assert len(a.readings(["xmeas_3"])) == 120


def test_readings_window_is_inclusive_and_step_takes_every_nth_sample(db):
    path, frames = db
    view = historian.CaseView(path, "dev_01")
    out = view.readings(["xmeas_2", "xmv_1"], start_h=0.5, end_h=1.0)
    assert list(out.columns) == ["t_hours", "xmeas_2", "xmv_1"]
    assert out["t_hours"].iloc[0] == 0.5 and out["t_hours"].iloc[-1] == 1.0 and len(out) == 11
    stepped = view.readings(["xmeas_2"], start_h=0.5, end_h=1.0, step=5)
    assert list(stepped["t_hours"]) == [0.5, 0.75, 1.0]
    assert np.allclose(stepped["xmeas_2"], frames["dev_01"]["xmeas_2"].iloc[[10, 15, 20]])


def test_unknown_tags_and_bad_steps_are_rejected(db):
    path, _ = db
    view = historian.CaseView(path, "dev_01")
    with pytest.raises(ValueError, match="unknown tag"):
        view.readings(["xmeas_99"])
    with pytest.raises(ValueError, match="step"):
        view.readings(["xmeas_1"], step=0)
    with pytest.raises(ValueError, match="no readings"):
        historian.CaseView(path, "dev_99")


def test_the_view_is_read_only(db):
    path, _ = db
    view = historian.CaseView(path, "dev_01")
    with pytest.raises(sqlite3.OperationalError):
        view._con.execute("delete from readings")


def test_alarms_window_filter(db):
    path, _ = db
    view = historian.CaseView(path, "dev_01")
    full = view.alarms()
    late = view.alarms(start_h=60 * T)
    assert len(late) <= len(full) and (late["t_hours"] >= 60 * T - 1e-9).all()
    assert view.alarms(end_h=-1.0).empty
