"""Phase 2 tests for src/mspc.py on synthetic data (no parquet).

Samples are 1-based; frames carry faultNumber, simulationRun, sample and the variable columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import mspc

VARS3 = ["v1", "v2", "v3"]


def _frame(values, runs=1, fault=0, variables=VARS3):
    """Rows of `values` (n x k) split evenly into `runs` runs, samples 1..n/runs."""
    values = np.asarray(values, dtype=float)
    per = len(values) // runs
    out = pd.DataFrame(values, columns=variables)
    out.insert(0, "sample", np.tile(np.arange(1, per + 1), runs))
    out.insert(0, "simulationRun", np.repeat(np.arange(1, runs + 1), per))
    out.insert(0, "faultNumber", fault)
    return out


# --- PC-count rule -------------------------------------------------------------
def test_n_components_is_the_fewest_pcs_reaching_the_variance_share():
    eig = np.array([5.0, 3.0, 1.0, 1.0])  # cumulative share 0.5, 0.8, 0.9, 1.0
    assert mspc.n_components(eig, 0.80) == 2   # exactly on the boundary counts
    assert mspc.n_components(eig, 0.81) == 3
    assert mspc.n_components(eig, 0.90) == 3
    assert mspc.n_components(eig, 0.91) == 4
    assert mspc.n_components(eig, 0.10) == 1


# --- lagged rows ---------------------------------------------------------------
def _two_runs():
    vals = np.array([[100 * r + s, -(100 * r + s), 0.0] for r in (1, 2) for s in range(1, 6)])
    return _frame(vals, runs=2)


def test_lagged_rows_never_cross_runs_and_drop_each_runs_first_L_samples():
    ids, x = mspc.lagged_rows(_two_runs(), VARS3, lags=2)
    assert list(ids["simulationRun"]) == [1, 1, 1, 2, 2, 2]
    assert list(ids["sample"]) == [3, 4, 5, 3, 4, 5]
    assert x.shape == (6, 3 * 3)
    # columns: x(k), x(k-1), x(k-2) for each of the 3 variables, current sample first
    assert list(x[0, :3]) == [103, -103, 0] and list(x[0, 3:6]) == [102, -102, 0] and list(x[0, 6:]) == [101, -101, 0]
    assert list(x[3, 6:]) == [201, -201, 0]  # run 2's first row looks back inside run 2, not into run 1


def test_zero_lags_keeps_every_row():
    ids, x = mspc.lagged_rows(_two_runs(), VARS3, lags=0)
    assert len(ids) == 10 and x.shape == (10, 3)


def test_lagged_rows_follow_sample_order_even_if_the_frame_is_shuffled():
    shuffled = _two_runs().sample(frac=1, random_state=0)
    ids, x = mspc.lagged_rows(shuffled, VARS3, lags=1)
    assert list(ids["sample"]) == [2, 3, 4, 5] * 2
    assert list(x[0, :3]) == [102, -102, 0] and list(x[0, 3:]) == [101, -101, 0]


# --- scores against hand-computed values ----------------------------------------
def _hand_monitor(tmp_path, n_pc):
    """Scaler mean (1, 2, 3), std (2, 1, 0.5); rotated loadings; eigenvalues (4, 1, 0.25)."""
    loadings = np.array([[0.6, -0.8, 0.0],
                         [0.8, 0.6, 0.0],
                         [0.0, 0.0, 1.0]])  # columns are the PCs
    path = tmp_path / "hand.npz"
    np.savez(path, mean=[1.0, 2.0, 3.0], std=[2.0, 1.0, 0.5], loadings=loadings, eigvals=[4.0, 1.0, 0.25],
             lags=0, n_pc=n_pc, n_rows=100, cpv=0.9, variables=VARS3)
    return mspc.PCAMonitor.load(path)


@pytest.mark.parametrize("n_pc, t2, spe", [(2, 2.0, 1.0), (1, 1.0, 2.0)])
def test_t2_and_spe_match_a_hand_computed_row(tmp_path, n_pc, t2, spe):
    # x = (5, 3, 3.5) -> z = (2, 1, 1) -> t = (0.6*2 + 0.8*1, -0.8*2 + 0.6*1, 1) = (2, -1, 1)
    # n_pc=2: T2 = 2^2/4 + (-1)^2/1 = 2, SPE = t3^2 = 1.  n_pc=1: T2 = 2^2/4 = 1, SPE = 1 + 1 = 2
    monitor = _hand_monitor(tmp_path, n_pc)
    out = monitor.score(_frame([[5.0, 3.0, 3.5]], variables=VARS3))
    assert list(out.columns) == ["faultNumber", "simulationRun", "sample", "t2", "spe"]
    assert out["t2"].iloc[0] == pytest.approx(t2)
    assert out["spe"].iloc[0] == pytest.approx(spe)


def _plane_data():
    """v1 == v2 (so PC1 = (1, 1, 0)/sqrt 2 with eigenvalue 2), v3 orthogonal to them: eigenvalues 2, 1, 0."""
    return _frame([[1, 1, 1], [1, 1, -1], [-1, -1, -1], [-1, -1, 1]])


def test_fit_recovers_known_eigenvalues_and_pc_count():
    monitor = mspc.PCAMonitor(lags=0, cpv=0.90, variables=VARS3).fit(_plane_data())
    assert monitor.eigvals == pytest.approx([2.0, 1.0, 0.0], abs=1e-9)
    assert monitor.n_pc == 2  # cumulative share 2/3, then 1
    assert np.abs(monitor.loadings[:, 0]) == pytest.approx([2 ** -0.5, 2 ** -0.5, 0.0], abs=1e-9)
    assert monitor.n_rows == 4


def test_fit_scores_on_training_rows_follow_the_algebra():
    data = _plane_data()
    monitor = mspc.PCAMonitor(variables=VARS3).fit(data)
    out = monitor.score(data)
    assert out["spe"].to_numpy() == pytest.approx(0.0, abs=1e-12)
    # sum over rows of T2 = (n - 1) * n_pc (each PC score has variance equal to its eigenvalue)
    assert out["t2"].sum() == pytest.approx(3 * 2)
    std = np.sqrt(4 / 3)
    off_plane = _frame([[std, -std, 0.0]])  # z = (1, -1, 0): only the (1, -1, 0)/sqrt 2 direction is left over
    assert monitor.score(off_plane)["spe"].iloc[0] == pytest.approx(2.0)
    assert monitor.score(off_plane)["t2"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_fit_rejects_faulty_rows():
    data = _plane_data().assign(faultNumber=1)
    with pytest.raises(ValueError, match="normal"):
        mspc.PCAMonitor(variables=VARS3).fit(data)


def test_dpca_scores_only_lagged_rows_and_stacks_the_lagged_variables():
    rng = np.random.default_rng(0)
    data = _frame(rng.normal(size=(60, 3)), runs=3)
    monitor = mspc.PCAMonitor(lags=2, variables=VARS3).fit(data)
    assert monitor.eigvals.shape == (9,) and monitor.loadings.shape == (9, 9)
    assert monitor.n_rows == 3 * (20 - 2)
    out = monitor.score(data)
    assert len(out) == 3 * 18 and out.groupby("simulationRun")["sample"].min().tolist() == [3, 3, 3]


# --- textbook limits -------------------------------------------------------------
def _spectrum_monitor(tmp_path, eigvals, n_pc, n_rows):
    k = len(eigvals)
    names = [f"v{i}" for i in range(k)]
    path = tmp_path / "spectrum.npz"
    np.savez(path, mean=np.zeros(k), std=np.ones(k), loadings=np.eye(k), eigvals=eigvals, lags=0, n_pc=n_pc,
             n_rows=n_rows, cpv=0.9, variables=names)
    return mspc.PCAMonitor.load(path)


def test_textbook_t2_limit_matches_the_f_distribution_formula(tmp_path):
    # a = 2 PCs, n = 101 rows: 2 * (101^2 - 1) / (101 * 99) * F(2, 99; 0.99), F quantile 4.826188891906277 (scipy)
    limit = _spectrum_monitor(tmp_path, [4.0, 2.0, 1.0], n_pc=2, n_rows=101).textbook_limits(0.99)["t2"]
    assert limit == pytest.approx(2 * 10200 / (101 * 99) * 4.826188891906277)
    assert limit == pytest.approx(9.846409980486854)
    assert limit == pytest.approx(2 * 10200 / (101 * 99) * stats.f.ppf(0.99, 2, 99))


def test_textbook_spe_limit_matches_a_hand_computed_jackson_mudholkar_example(tmp_path):
    # discarded eigenvalues 0.5, 0.3, 0.2: theta = (1.0, 0.38, 0.16), h0 = 1 - 2*1.0*0.16/(3*0.38^2) = 0.261311,
    # c = 2.326348 (0.99 normal quantile): Q = 1.0 * [c*sqrt(2*0.38*h0^2) + 1 + 0.38*h0*(h0-1)]^(1/h0) = 4.2178
    limit = _spectrum_monitor(tmp_path, [3.0, 0.5, 0.3, 0.2], n_pc=1, n_rows=500).textbook_limits(0.99)["spe"]
    assert limit == pytest.approx(4.2178, abs=5e-4)
    assert limit == pytest.approx(4.217794597705633)


def test_textbook_limits_rise_with_the_confidence_level(tmp_path):
    m = _spectrum_monitor(tmp_path, [3.0, 0.5, 0.3, 0.2], n_pc=1, n_rows=500)
    assert m.textbook_limits(0.95)["t2"] < m.textbook_limits(0.99)["t2"]
    assert m.textbook_limits(0.95)["spe"] < m.textbook_limits(0.99)["spe"]


# --- persistence -----------------------------------------------------------------
def test_save_and_load_give_identical_scores(tmp_path):
    rng = np.random.default_rng(1)
    train = _frame(rng.normal(size=(120, 3)) @ np.array([[1, .5, 0], [0, 1, .3], [0, 0, 1]]), runs=4)
    test = _frame(rng.normal(size=(40, 3)), runs=2, fault=7)
    monitor = mspc.PCAMonitor(lags=1, cpv=0.8, variables=VARS3).fit(train)
    monitor.save(tmp_path / "m.npz")
    again = mspc.PCAMonitor.load(tmp_path / "m.npz")
    assert (again.lags, again.cpv, again.n_pc, again.n_rows) == (1, 0.8, monitor.n_pc, monitor.n_rows)
    pd.testing.assert_frame_equal(monitor.score(test), again.score(test), check_exact=True)
    assert again.textbook_limits() == monitor.textbook_limits()


def test_scores_do_not_depend_on_how_the_runs_are_chunked(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    train = _frame(rng.normal(size=(120, 3)), runs=4)
    test = _frame(rng.normal(size=(90, 3)), runs=3, fault=2)
    monitor = mspc.PCAMonitor(lags=2, variables=VARS3).fit(train)
    whole = monitor.score(test)
    monkeypatch.setattr(mspc, "CHUNK_ROWS", 1)  # one run per chunk
    pd.testing.assert_frame_equal(monitor.score(test), whole)


# --- Ku et al. lag choice ----------------------------------------------------------
VARS4 = ["v1", "v2", "v3", "v4"]


def _dynamic_system(n=20_000, phi=0.5, noise=1e-3, seed=0):
    """AR(1) series s; v1 = s(k), v2 = s(k-1) (one dynamic relation v2(k) = v1(k-1)); v4 = v3 (one static relation).

    Ku's counts with the relations as the near-zero PCs: r(0) = 1, r(1) = 3, r(2) = 5, so
    r_new = 1, 3 - 2*1 = 1, 5 - 3*1 - 2*1 = 0 -> stop at l = 2, use L = 1.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + 1)
    s = np.zeros(n + 1)
    for k in range(1, n + 1):
        s[k] = phi * s[k - 1] + e[k]
    v3 = rng.normal(size=n)
    values = np.c_[s[1:], s[:-1] + noise * rng.normal(size=n), v3, v3 + noise * rng.normal(size=n)]
    return _frame(values, runs=4, variables=VARS4)


def test_ku_lag_finds_the_known_lag_of_a_dynamic_system():
    lag, table = mspc.ku_lag(_dynamic_system(), max_lag=5, cpv=0.99, variables=VARS4)
    assert lag == 1
    assert list(table["lag"]) == [0, 1, 2]  # stops at the first l with r_new <= 0
    assert list(table["n_vars"]) == [4, 8, 12]
    assert list(table["a"]) == [3, 5, 7]
    assert list(table["r"]) == [1, 3, 5]
    assert list(table["r_new"]) == [1, 1, 0]


def test_ku_lag_is_none_when_nothing_stops_by_the_maximum_lag():
    lag, table = mspc.ku_lag(_dynamic_system(), max_lag=1, cpv=0.99, variables=VARS4)
    assert lag is None and list(table["lag"]) == [0, 1]


def test_ku_lag_is_zero_for_a_static_system():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(8000, 2))
    values = np.c_[base, base[:, 0] + 1e-3 * rng.normal(size=8000), rng.normal(size=8000)]  # one static relation
    lag, table = mspc.ku_lag(_frame(values, runs=4, variables=VARS4), max_lag=5, cpv=0.99, variables=VARS4)
    assert lag == 0 and list(table["r_new"])[:2] == [1, 0]


def test_ku_lag_labels_its_table_with_the_rule_used():
    _, table = mspc.ku_lag(_dynamic_system(), max_lag=1, cpv=0.99, variables=VARS4)
    assert set(table["rule"]) == {"cpv"}
    assert list(table.columns) == ["rule", "lag", "n_vars", "a", "r", "r_new"]


def test_parallel_analysis_keeps_only_the_components_above_noise():
    rng = np.random.default_rng(4)
    n = 5000
    f1, f2 = rng.normal(size=(2, n))
    z = np.c_[f1, f1, f1, f2, f2, f2] + 0.5 * rng.normal(size=(n, 6))  # two clear factors, eigenvalues ~ 2.5, 2.5, 0.2...
    z = (z - z.mean(0)) / z.std(0, ddof=1)
    eigvals = np.sort(np.linalg.eigvalsh(z.T @ z / (n - 1)))[::-1]
    assert mspc.parallel_n_components(eigvals, n_rows=n) == 2


def test_ku_lag_parallel_rule_runs_and_labels_the_table():
    lag, table = mspc.ku_lag(_dynamic_system(n=4000), max_lag=2, rule="parallel", variables=VARS4)
    assert set(table["rule"]) == {"parallel"} and table["r"].iloc[0] >= 1  # the static relation is found


# --- lag verdict (hard stop before any fit) ------------------------------------------
@pytest.mark.parametrize("cpv_lag, pa_lag, stops", [
    (2, 2, False),
    (1, 1, False),
    (0, 0, True),      # L = 0: nothing for DPCA to add
    (2, 3, True),      # the two rules disagree
    (None, None, True),  # no stop by the maximum lag
    (None, 2, True),
    (2, None, True),
])
def test_lag_problem_says_when_to_stop_before_fitting(cpv_lag, pa_lag, stops):
    assert (mspc.lag_problem(cpv_lag, pa_lag) is not None) == stops


# --- scorecard rows for the four models ----------------------------------------------
def test_scorecard_rows_cover_four_models_at_three_limits():
    onset, skip = 20, 1
    rows_ids = []
    for fault, run, n in [(0, 1, 40), (0, 2, 40), (1, 1, 40)]:
        rows_ids.append(pd.DataFrame({"faultNumber": fault, "simulationRun": run, "sample": np.arange(1, n + 1)}))
    ids = pd.concat(rows_ids, ignore_index=True)
    scores = ids.assign(t2=np.where((ids.faultNumber == 1) & (ids["sample"] > onset), 10.0, 0.1), spe=0.2)
    limits = {"t2": {"1%": 5.0, "5%": 4.0, "99% textbook": 6.0}, "spe": {"1%": 1.0, "5%": 0.9, "99% textbook": 1.5}}
    faulty, normal = scores[scores.faultNumber > 0], scores[scores.faultNumber == 0]
    out = mspc.scorecard_rows("pca", faulty, normal, limits, onset, skip, machine="m", split_file="s",
                              train_time_s=1.0, peak_ram_gb=0.5)
    df = pd.DataFrame(out)
    assert sorted(df["model"].unique()) == ["pca-spe", "pca-t2"]
    assert sorted(df["limit_name"].unique()) == ["1%", "5%", "99% textbook"]
    assert set(df["track"]) == {"unsupervised"}
    t2_1 = df[(df.model == "pca-t2") & (df.limit_name == "1%") & (df.metric == "fdr")]["value"].iloc[0]
    far_spe = df[(df.model == "pca-spe") & (df.limit_name == "1%") & (df.metric == "far")]["value"].iloc[0]
    assert t2_1 == 1.0 and far_spe == 0.0
    assert df[df.model == "pca-spe"].query("limit_name == '5%'")["limit_value"].iloc[0] == 0.9


# --- phase 2b cleanup: limits stored with the model ------------------------------------
LIMITS = {"t2": {"1%": 5.5, "5%": 4.25, "99% textbook": 6.0}, "spe": {"1%": 1.5, "5%": 0.75, "99% textbook": 2.0}}


def _small_monitor():
    rng = np.random.default_rng(5)
    train = _frame(rng.normal(size=(60, 3)), runs=3)
    return mspc.PCAMonitor(lags=0, cpv=0.8, variables=VARS3).fit(train)


def test_limits_saved_with_the_model_come_back_exactly(tmp_path):
    monitor = _small_monitor()
    monitor.save(tmp_path / "m.npz", limits=LIMITS)
    assert mspc.PCAMonitor.load(tmp_path / "m.npz").limits == LIMITS


def test_a_model_saved_without_limits_loads_with_none(tmp_path):
    monitor = _small_monitor()
    monitor.save(tmp_path / "m.npz")
    assert mspc.PCAMonitor.load(tmp_path / "m.npz").limits is None


def test_stored_limits_equal_the_phase_2_dev_scores():
    """The .npz limits (written by mspc_dev.py) are the ones in dev_scores.csv; phase 2b reads only the .npz."""
    root = Path(__file__).resolve().parent.parent / "results" / "phase-2"
    dev = pd.read_csv(root / "dev_scores.csv")
    dev = dev[dev.metric == "far"]
    for kind in ("pca", "dpca"):
        stored = mspc.PCAMonitor.load(root / f"{kind}.npz").limits
        assert stored is not None, f"{kind}.npz has no limits; rerun scripts/mspc_dev.py"
        for stat in mspc.STATS:
            table = dev[dev.model == f"{kind}-{stat}"].set_index("limit_name")["limit_value"]
            assert set(stored[stat]) == set(table.index) == {"1%", "5%", mspc.TEXTBOOK}
            for name, value in table.items():
                assert stored[stat][name] == pytest.approx(value, rel=1e-12)


# --- phase 2b cleanup: final-run plausibility, FAR and leak checks ----------------------
MODELS4 = ["pca-t2", "pca-spe", "dpca-t2", "dpca-spe"]
GOOD_FDR = {"pca-t2": {4: 0.3, 3: 0.02, 9: 0.02, 15: 0.03}, "pca-spe": {4: 0.98, 3: 0.01, 9: 0.01, 15: 0.02},
            "dpca-t2": {4: 0.2, 3: 0.03, 9: 0.03, 15: 0.05}, "dpca-spe": {4: 0.99, 3: 0.01, 9: 0.01, 15: 0.01}}


def _final_rows(fdr=None, far_by_limit=None):
    fdr = fdr or GOOD_FDR
    far_by_limit = far_by_limit or {"1%": 0.01, "5%": 0.05, "99% textbook": 0.02}
    rows = []
    for limit_name, far in far_by_limit.items():
        for m in MODELS4:
            rows.append({"model": m, "limit_name": limit_name, "metric": "far", "fault": np.nan, "value": far})
            for fault, v in fdr[m].items():
                rows.append({"model": m, "limit_name": limit_name, "metric": "fdr", "fault": fault, "value": v})
    return pd.DataFrame(rows)


def test_final_checks_pass_on_a_plausible_scorecard():
    assert mspc.final_check_failures(_final_rows(), MODELS4) == []


def test_final_checks_flag_a_far_outside_its_window():
    out = mspc.final_check_failures(_final_rows(far_by_limit={"1%": 0.04, "5%": 0.05, "99% textbook": 0.02}), MODELS4)
    assert len(out) == 4 and all("FAR out of range" in f and "at 1%" in f for f in out)


def test_final_checks_flag_a_leak_on_the_hard_faults():
    fdr = {**GOOD_FDR, "dpca-t2": {**GOOD_FDR["dpca-t2"], 9: 0.30}}
    out = mspc.final_check_failures(_final_rows(fdr), MODELS4)
    assert len(out) >= 1 and all("LEAK?" in f and "dpca-t2" in f for f in out)


def test_the_leak_bound_at_other_limits_is_far_plus_five_points():
    fdr = {**GOOD_FDR, "pca-t2": {**GOOD_FDR["pca-t2"], 15: 0.11}}
    rows = _final_rows(fdr, far_by_limit={"1%": 0.01, "5%": 0.05, "99% textbook": 0.07})
    out = mspc.final_check_failures(rows, MODELS4)
    # 0.11 > 0.10 at the 1% limit, but 0.11 <= 0.07 + 0.05 at the textbook limit
    assert [f for f in out if "LEAK?" in f] and not [f for f in out if "textbook" in f and "LEAK?" in f]


@pytest.mark.parametrize("m, value", [("pca-t2", 0.6), ("pca-spe", 0.85)])
def test_final_checks_flag_an_implausible_fault_4(m, value):
    fdr = {**GOOD_FDR, m: {**GOOD_FDR[m], 4: value}}
    out = mspc.final_check_failures(_final_rows(fdr), MODELS4)
    assert any("fault 4" in f and m in f for f in out)


# --- phase 2b cleanup: parallel analysis takes only what varies ------------------------------
def test_parallel_analysis_needs_only_the_eigenvalues_and_row_count():
    import inspect
    assert list(inspect.signature(mspc.parallel_n_components).parameters) == ["eigvals", "n_rows"]
