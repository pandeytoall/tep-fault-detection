"""Phase 1 tests for the paired IDV numbering check in src/signatures.py (synthetic runs, no data files)."""
import numpy as np
import pandas as pd
import pytest

from src import data, signatures

ONSET = 20
WINDOW = 20


def _normal(seed, n=60):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, 52)), columns=data.VARIABLES)
    frame.insert(0, "sample", np.arange(1, n + 1))
    return frame


def _faulty(normal, shifts):
    """Same noise as `normal` (plus a tiny perturbation so unmoved variables do not tie at exactly 0);
    after onset the given variables move by `size` normal-run std units."""
    frame = normal.copy()
    frame[data.VARIABLES] += np.random.default_rng(int(normal["xmeas_1"].iloc[0] * 1e6) % 2**32).normal(
        scale=0.01, size=(len(normal), 52)) * normal[data.VARIABLES].std().to_numpy()
    for tag, size in shifts.items():
        frame.loc[frame["sample"] > ONSET, tag] += size * normal[tag].std()
    return frame


def _check(fault, shifts_by_run, expected=None, n_runs=5):
    normals = {r: _normal(r) for r in range(1, n_runs + 1)}
    faulty = {fault: {r: _faulty(normals[r], shifts_by_run(r)) for r in normals}}
    kwargs = {"expected": {fault: signatures.IDV_EXPECTED[fault]} if expected is None else expected}
    return signatures.idv_check(faulty, normals, ONSET, window=WINDOW, **kwargs).set_index("fault").loc[fault]


def test_expected_signatures_are_the_four_hypotheses_from_the_spec():
    assert signatures.IDV_EXPECTED == {4: "xmv_10", 5: "xmv_11", 6: "xmeas_1", 7: "xmeas_4"}


def test_idv_check_columns():
    normals = {1: _normal(1)}
    out = signatures.idv_check({4: {1: _faulty(normals[1], {"xmv_10": 5})}}, normals, ONSET, window=WINDOW,
                               expected={4: "xmv_10"})
    assert list(out.columns) == ["fault", "expected_tag", "expected_name", "share_runs_in_top3", "most_common_top1",
                                 "top3_most_common", "plant_unit_of_top1", "verdict"]


def test_only_the_expected_variable_moves_is_confirmed():
    row = _check(4, lambda r: {"xmv_10": 8})
    assert row.expected_tag == "xmv_10" and row.expected_name == "Reactor cooling water flow"
    assert row.share_runs_in_top3 == 1 and row.most_common_top1 == "xmv_10"
    assert row.plant_unit_of_top1 == "reactor" and row.verdict == "confirmed"
    assert "xmv_10:5" in row.top3_most_common.split(";")  # tag:count over the 5 runs


def test_other_unit_variable_moves_is_fail():
    row = _check(4, lambda r: {"xmeas_1": 8})  # A feed (feed unit) moves, the reactor cooling water flow does not
    assert row.share_runs_in_top3 < 0.8 and row.most_common_top1 == "xmeas_1"  # chance slots 2-3 may hit xmv_10
    assert row.plant_unit_of_top1 == "feed" and row.verdict == "fail"


def test_other_variable_in_an_acceptable_unit_is_unit_match_post_hoc():
    row = _check(4, lambda r: {"xmeas_9": 8})  # reactor temperature moves, not the expected xmv_10
    assert row.verdict == "unit_match_post_hoc" and row.plant_unit_of_top1 == "reactor"
    row5 = _check(5, lambda r: {"xmeas_11": 8})  # separator temperature: acceptable for fault 5 only
    assert row5.verdict == "unit_match_post_hoc"
    row5_fail = _check(5, lambda r: {"xmeas_9": 8})
    assert row5_fail.verdict == "fail"
    row7 = _check(7, lambda r: {"xmeas_7": 8})  # reactor pressure: acceptable for fault 7 only
    assert row7.verdict == "unit_match_post_hoc"
    assert _check(6, lambda r: {"xmeas_7": 8}).verdict == "fail"  # fault 6 accepts the feed unit only


def test_acceptable_units_are_explicit_per_fault():
    assert signatures.IDV_ACCEPTABLE_UNITS == {4: {"reactor"}, 5: {"condenser", "separator"}, 6: {"feed"},
                                               7: {"feed", "reactor"}}


def test_share_threshold_is_0_8():
    # expected variable in the top 3 in 4 of 5 runs (0.8) passes; 3 of 5 (0.6) does not
    def shifts(hits):
        return lambda r: {"xmv_10": 8} if r <= hits else {"xmeas_1": 8, "xmeas_2": 8, "xmeas_3": 8, "xmeas_4": 8}
    assert _check(4, shifts(4)).verdict == "confirmed"
    row = _check(4, shifts(3))
    assert row.share_runs_in_top3 == pytest.approx(0.6) and row.verdict != "confirmed"


def test_window_is_the_first_samples_after_onset():
    # a shift that starts after the window must not count
    normals = {r: _normal(r) for r in (1, 2)}
    faulty = {}
    for r, n in normals.items():
        f = n.copy()
        f.loc[f["sample"] > ONSET + WINDOW, "xmv_10"] += 10 * n["xmv_10"].std()
        faulty[r] = f
    out = signatures.idv_check({4: faulty}, normals, ONSET, window=WINDOW, expected={4: "xmv_10"})
    assert out["share_runs_in_top3"].iloc[0] < 1  # not seen in 21-40 (ties at noise level at best)


def test_full_default_expectations_run():
    normals = {r: _normal(r) for r in (1, 2, 3)}
    faulty = {f: {r: _faulty(normals[r], {tag: 8}) for r in normals} for f, tag in signatures.IDV_EXPECTED.items()}
    out = signatures.idv_check(faulty, normals, ONSET, window=WINDOW)
    assert list(out.fault) == [4, 5, 6, 7] and (out.verdict == "confirmed").all()
    assert list(out.most_common_top1) == list(signatures.IDV_EXPECTED.values())
