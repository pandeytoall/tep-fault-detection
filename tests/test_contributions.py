"""Phase 2b tests for src/contributions.py: SPE and T2 contributions, the normal reference and the relative top-N.

Hand example (3 variables, 2 retained PCs, scaled space): p1 = (1, 1, 0)/sqrt2, p2 = (0, 0, 1), p3 = (1, -1, 0)/sqrt2,
eigenvalues 4, 1, 0.25. For x = (2, 0, 3): t = (sqrt2, 3), reconstruction (1, 1, 3), residual e = (1, -1, 0), so
SPE contributions are (1, 1, 0), SPE = 2; T2 = 2/4 + 9 = 9.5 and the complete decomposition gives
P L^(-1/2) P'x = (0.5, 0.5, 3), contributions (0.25, 0.25, 9), which sum to 9.5.
"""
import numpy as np
import pandas as pd
import pytest

from src import contributions, mspc

TAGS = ["xmeas_1", "xmeas_7", "xmv_5"]  # feeds, reactor, compressor/recycle


def _monitor(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
    s = np.sqrt(2)
    m = mspc.PCAMonitor(lags=0, cpv=0.9, variables=TAGS)
    m.mean, m.std = np.array(mean), np.array(std)
    m.loadings = np.array([[1 / s, 0, 1 / s], [1 / s, 0, -1 / s], [0, 1, 0]])
    m.eigvals = np.array([4.0, 1.0, 0.25])
    m.n_pc, m.n_rows = 2, 1000
    return m


def test_spe_contributions_are_the_squared_residuals_and_sum_to_spe():
    m = _monitor()
    z = np.array([[2.0, 0.0, 3.0]])
    c = contributions.spe_contributions(m, z)
    assert np.allclose(c, [[1.0, 1.0, 0.0]])
    assert c.sum() == pytest.approx(2.0)


def test_t2_contributions_use_the_complete_decomposition_and_sum_to_t2():
    m = _monitor()
    z = np.array([[2.0, 0.0, 3.0]])
    c = contributions.t2_contributions(m, z)
    assert np.allclose(c, [[0.25, 0.25, 9.0]])
    assert c.sum() == pytest.approx(9.5)


def test_contributions_sum_to_the_monitor_scores_on_random_rows():
    rng = np.random.default_rng(0)
    m = _monitor(mean=(1.0, -2.0, 0.5), std=(2.0, 0.5, 3.0))
    raw = rng.normal(size=(30, 3)) * 3
    frame = pd.DataFrame(raw, columns=TAGS)
    frame.insert(0, "sample", np.arange(1, 31))
    frame.insert(0, "simulationRun", 1)
    frame.insert(0, "faultNumber", 0)
    scores = m.score(frame)
    z = contributions.scale(m, raw)
    assert np.allclose(contributions.spe_contributions(m, z).sum(axis=1), scores["spe"])
    assert np.allclose(contributions.t2_contributions(m, z).sum(axis=1), scores["t2"])


def test_scaling_uses_the_monitor_mean_and_std():
    m = _monitor(mean=(1.0, 0.0, 0.0), std=(2.0, 1.0, 1.0))
    assert np.allclose(contributions.scale(m, np.array([[5.0, 0.0, 3.0]])), [[2.0, 0.0, 3.0]])


def test_normal_reference_is_the_mean_contribution_per_variable():
    m = _monitor()
    normal = pd.DataFrame([[2.0, 0.0, 3.0], [0.0, 0.0, 0.0]], columns=TAGS)
    ref = contributions.normal_reference(m, normal)
    assert np.allclose(ref["spe"], [0.5, 0.5, 0.0])
    assert np.allclose(ref["t2"], [0.125, 0.125, 4.5])


def test_relative_contribution_divides_by_the_reference_with_a_floor():
    window = np.array([4.0, 1.0, 8.0])
    ref = np.array([2.0, 0.0, 4.0])             # mean 2, so floor = 0.02
    assert contributions.FLOOR_SHARE == 0.01
    assert np.allclose(contributions.relative(window, ref), [2.0, 50.0, 2.0])


def test_top_contributors_rank_by_ratio_and_carry_names_units_and_scoring_unit():
    m = _monitor()
    ref = {"spe": np.array([1.0, 2.0, 1.0]), "t2": np.array([1.0, 0.5, 1.0])}
    window = pd.DataFrame([[2.0, 0.0, 3.0], [2.0, 0.0, 3.0]], columns=TAGS)
    top = contributions.top_contributors(m, ref, window, "spe", n=2)
    assert [(t["tag"], t["ratio"]) for t in top] == [("xmeas_1", 1.0), ("xmeas_7", 0.5)]   # residual^2 (1, 1, 0) / ref
    top_t2 = contributions.top_contributors(m, ref, window, "t2", n=3)
    assert [t["tag"] for t in top_t2] == ["xmv_5", "xmeas_7", "xmeas_1"]   # ratios 9, 0.5, 0.25
    assert top_t2[0] == {"tag": "xmv_5", "name": "Compressor recycle valve", "unit": "%",
                         "scoring_unit": "compressor/recycle", "ratio": 9.0}
    assert top_t2[2]["scoring_unit"] == "feeds" and top_t2[1]["scoring_unit"] == "reactor"


def test_ratio_is_rounded_to_one_decimal():
    m = _monitor()
    ref = {"spe": np.array([3.0, 3.0, 3.0]), "t2": np.array([3.0, 3.0, 3.0])}
    window = pd.DataFrame([[2.0, 0.0, 3.0]], columns=TAGS)
    top = contributions.top_contributors(m, ref, window, "t2", n=1)
    assert top[0]["ratio"] == 3.0
    ref = {"spe": np.array([7.0, 7.0, 7.0]), "t2": np.array([7.0, 7.0, 7.0])}
    assert contributions.top_contributors(m, ref, window, "t2", n=1)[0]["ratio"] == round(9.0 / 7.0, 1) == 1.3


def test_unknown_statistic_is_rejected():
    m = _monitor()
    with pytest.raises(ValueError, match="statistic"):
        contributions.top_contributors(m, {"spe": np.ones(3), "t2": np.ones(3)}, pd.DataFrame([[0.0] * 3], columns=TAGS),
                                       "q")


# --- the default window -----------------------------------------------------------------------
def _alarms(rows):
    return pd.DataFrame(rows, columns=["t_hours", "statistic", "event", "value", "limit"])


def test_default_window_starts_at_the_first_raise_of_that_statistic():
    log = _alarms([[1.0, "t2", "raise", 9.0, 5.0], [2.0, "spe", "raise", 9.0, 5.0], [3.0, "spe", "clear", 1.0, 5.0],
                   [4.0, "spe", "raise", 9.0, 5.0]])
    assert contributions.window_start(log, "spe") == 2.0
    assert contributions.window_start(log, "t2") == 1.0


def test_default_window_is_the_whole_case_without_a_raise():
    log = _alarms([[3.0, "spe", "clear", 1.0, 5.0]])
    assert contributions.window_start(log, "spe") is None
    assert contributions.window_start(_alarms([]), "t2") is None


def test_window_rows_take_the_case_from_the_start_to_the_end():
    matrix = pd.DataFrame({"t_hours": [0.0, 0.05, 0.1, 0.15], "xmeas_1": [1, 2, 3, 4]})
    assert list(contributions.window_rows(matrix, None, None)["xmeas_1"]) == [1, 2, 3, 4]
    assert list(contributions.window_rows(matrix, 0.1, None)["xmeas_1"]) == [3, 4]
    assert list(contributions.window_rows(matrix, 0.05, 0.1)["xmeas_1"]) == [2, 3]


# --- the reference lives in pca.npz -------------------------------------------------------------------
def test_reference_is_added_to_the_npz_and_the_model_still_loads(tmp_path):
    m = _monitor()
    limits = {"t2": {"1%": 5.0}, "spe": {"1%": 1.0}}
    m.save(tmp_path / "pca.npz", limits=limits)
    ref = {"spe": np.array([0.5, 0.5, 0.0]), "t2": np.array([0.125, 0.125, 4.5])}
    contributions.add_reference(tmp_path / "pca.npz", ref)
    again = contributions.load_reference(tmp_path / "pca.npz")
    assert np.array_equal(again["spe"], ref["spe"]) and np.array_equal(again["t2"], ref["t2"])
    loaded = mspc.PCAMonitor.load(tmp_path / "pca.npz")
    assert loaded.limits == limits and loaded.n_pc == 2


def test_load_reference_says_so_when_the_npz_has_none(tmp_path):
    _monitor().save(tmp_path / "pca.npz")
    assert contributions.load_reference(tmp_path / "pca.npz") is None


# --- baseline (no LLM) --------------------------------------------------------------------------------
def test_baseline_unit_is_the_scoring_unit_of_the_top_spe_contributor_over_the_default_window():
    m = _monitor()
    # from t = 0.1 on, x = (2, 0, 3) -> SPE contributions (1, 1, 0); before that x = 0 -> nothing
    matrix = pd.DataFrame({"t_hours": [0.0, 0.05, 0.1, 0.15], "xmeas_1": [0, 0, 2.0, 2.0],
                           "xmeas_7": [0, 0, 0.0, 0.0], "xmv_5": [0, 0, 3.0, 3.0]})
    log = _alarms([[0.1, "spe", "raise", 9.0, 5.0]])
    ref = {"spe": np.array([1.0, 2.0, 1.0]), "t2": np.ones(3)}
    assert contributions.baseline_unit(m, ref, matrix, log) == "feeds"
    ref = {"spe": np.array([2.0, 1.0, 1.0]), "t2": np.ones(3)}
    assert contributions.baseline_unit(m, ref, matrix, log) == "reactor"
