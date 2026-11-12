"""Phase 4 tests for src/diagnosis.py on synthetic data (no parquet): RBC, blame metrics, before/after, permutation importance."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src import cases, data, diagnosis, mspc

# --- RBC against a hand-computed example --------------------------------------------------------------------------------
P1 = np.array([1.0, 2.0, 2.0]) / 3
P2 = np.array([2.0, 1.0, -2.0]) / 3


def monitor3(columns, eigvals):
    """A 3-variable PCA 'monitor' with the given orthonormal loading columns."""
    return SimpleNamespace(loadings=np.column_stack(columns), eigvals=np.array(eigvals, dtype=float), n_pc=len(columns))


def test_rbc_spe_matches_the_hand_computed_one_pc_example():
    # C~ = I - p p', p = (1, 2, 2)/3: diagonal (8/9, 5/9, 5/9). x = (3, 0, 0): C~x = (8/3, -2/3, -2/3).
    # rbc = (8/3)^2 / (8/9) = 8 and (2/3)^2 / (5/9) = 0.8 twice. x = (0, 3, 0): C~x = (-2/3, 5/3, -4/3) -> 0.5, 5, 3.2.
    out = diagnosis.rbc_spe(monitor3([P1], [4.0]), np.array([[3.0, 0, 0], [0, 3.0, 0]]))
    assert out == pytest.approx(np.array([[8.0, 0.8, 0.8], [0.5, 5.0, 3.2]]))


def test_rbc_t2_matches_the_hand_computed_two_pc_example():
    # lambda = (4, 1), x = (3, 0, 0): Dx = (17/12, 10/12, -14/12), diag D = (17/36, 8/36, 20/36)
    # rbc = (17/12)^2 / (17/36) = 17/4, (10/12)^2 / (8/36) = 25/8, (14/12)^2 / (20/36) = 49/20
    out = diagnosis.rbc_t2(monitor3([P1, P2], [4.0, 1.0]), np.array([[3.0, 0.0, 0.0]]))
    assert out == pytest.approx(np.array([[4.25, 3.125, 2.45]]))


def test_rbc_values_do_not_sum_to_the_statistic():
    monitor = monitor3([P1, P2], [4.0, 1.0])
    x = np.array([[3.0, 0.0, 0.0]])
    t2 = float((x @ monitor.loadings @ np.diag([0.25, 1.0]) @ monitor.loadings.T @ x.T)[0, 0])   # 0.25 + 4 = 4.25
    assert t2 == pytest.approx(4.25)
    assert diagnosis.rbc_t2(monitor, x).sum() != pytest.approx(t2)


def test_rbc_skips_a_reading_whose_denominator_is_zero_instead_of_dividing_by_it():
    # all 3 PCs kept: C~ = 0, nothing is left to blame, and no nan or inf
    full = monitor3([np.eye(3)[:, 0], np.eye(3)[:, 1], np.eye(3)[:, 2]], [1.0, 1.0, 1.0])
    out = diagnosis.rbc_spe(full, np.array([[1.0, 2.0, 3.0]]))
    assert np.array_equal(out, np.zeros((1, 3)))
    # 2 PCs kept (first two axes): C~ = diag(0, 0, 1), only reading 3 can be blamed: 3^2 / 1
    partial = monitor3([np.eye(3)[:, 0], np.eye(3)[:, 1]], [1.0, 1.0])
    assert diagnosis.rbc_spe(partial, np.array([[1.0, 2.0, 3.0]])) == pytest.approx(np.array([[0.0, 0.0, 9.0]]))


# --- the expected unit agrees with cases.py ---------------------------------------------------------------------------
def test_expected_unit_agrees_with_the_case_table_and_has_none_for_16_to_20():
    for fault in range(1, 16):
        assert diagnosis.expected_unit(fault) == cases.case_unit(fault)
    for fault in (0, 16, 17, 18, 19, 20):
        assert diagnosis.expected_unit(fault) is None and diagnosis.unit_mask(fault, data.VARIABLES) is None


def test_unit_mask_picks_the_readings_of_the_scoring_unit():
    tags = data.VARIABLES
    feeds, reactor, separator = (diagnosis.unit_mask(f, tags) for f in (6, 4, 5))
    # feeds: 4 + 4 readings; reactor: 5 measured + 6 feed analyzers + the cooling water valve; condenser/separator: 5 + 2
    assert (feeds.sum(), reactor.sum(), separator.sum()) == (8, 12, 7)
    assert {tags[i] for i in np.flatnonzero(feeds)} >= {"xmeas_1", "xmv_3"}            # the A feed readings, fault 6
    assert {"xmeas_21", "xmv_10"} <= {tags[i] for i in np.flatnonzero(reactor)}         # the reactor cooling readings, fault 4


def test_fault_groups_and_notes_follow_the_spec():
    assert [diagnosis.fault_group(f) for f in (1, 7, 3, 15, 5, 14, 16, 20)] == \
        ["benchmark", "benchmark", "hidden", "hidden", "the rest", "the rest", "unknown cause", "unknown cause"]
    assert diagnosis.fault_note(9) == "near the false-alarm level; blame is mostly noise"
    assert diagnosis.fault_note(18) == "unknown cause" and diagnosis.fault_note(1) == ""


# --- blame metrics on a hand-made case ----------------------------------------------------------------------------------
IN_UNIT = np.array([True, True] + [False] * 6)       # readings 0 and 1 are in the expected unit (a quarter of 8)
BLAME = np.array([
    [5.0, 1.0, 0, 0, 0, 0, 0, 0],                    # top reading in the unit; all blame (6 of 6) in the unit; top 5 hold all of it
    [0.5, 0.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.2],        # top reading outside; top 5 = readings 2-6, none in the unit; mass 0.5 / 11.5
    [1.0, 4.0, 3.0, 2.0, 0.5, 0.2, 0.1, -3.0],       # the -3 counts as 0; top reading in the unit; mass 5 / 10.8; top 5 hold 10.5 / 10.8
])


def test_blame_metrics_match_the_hand_counts():
    m = diagnosis.blame_metrics(BLAME, IN_UNIT)
    assert m["top1_hit"] == pytest.approx(2 / 3)
    assert m["top5_hit"] == pytest.approx(2 / 3)
    assert m["unit_mass"] == pytest.approx((1 + 1 / 23 + 25 / 54) / 3)
    assert m["smear"] == pytest.approx((1 + 108 / 115 + 35 / 36) / 3)


def test_random_baseline_is_the_unit_share_and_one_minus_one_minus_share_to_the_fifth():
    share, top5 = diagnosis.random_baseline(IN_UNIT)
    assert share == 0.25 and top5 == pytest.approx(1 - 0.75 ** 5)           # 0.7626953125


def test_blame_metrics_without_an_expected_unit_report_only_smear():
    m = diagnosis.blame_metrics(BLAME[:1])
    assert set(m) == {"smear"} and m["smear"] == 1.0


def test_blame_metrics_accept_a_single_vector_and_an_all_zero_vector():
    one = diagnosis.blame_metrics(BLAME[1], IN_UNIT)                          # one permutation-importance vector
    assert (one["top1_hit"], one["top5_hit"]) == (0.0, 0.0)
    zero = diagnosis.blame_metrics(np.zeros(8), IN_UNIT)
    assert zero["unit_mass"] == 0.0 and zero["smear"] == 0.0


def test_top_share_counts_runs_whose_top_k_include_a_wanted_reading():
    tags = ["a", "b", "c", "d"]
    v = np.array([[9, 1, 2, 3], [1, 9, 2, 3], [1, 2, 3, 9]], dtype=float)
    assert diagnosis.top_share(v, tags, ["a"], k=1) == pytest.approx(1 / 3)
    assert diagnosis.top_share(v, tags, ["a", "b"], k=1) == pytest.approx(2 / 3)
    assert diagnosis.top_share(v, tags, ["d"], k=1) == pytest.approx(1 / 3)      # only run 3 has d on top
    assert diagnosis.top_share(v, tags, ["d"], k=2) == 1.0                       # d is second in runs 1 and 2


def test_final_pass_run_ids_are_100_fixed_runs_of_the_500():
    ids = diagnosis.test_run_ids()
    assert len(ids) == len(set(ids)) == 100 and ids == sorted(ids) and 1 <= ids[0] and ids[-1] <= 500
    assert ids[:8] == [1, 21, 34, 41, 45, 46, 47, 52] and ids[-3:] == [490, 495, 497]       # pinned: seed 42
    assert diagnosis.test_run_ids() == ids and diagnosis.test_run_ids(n=10) != ids[:10]


def test_the_control_runs_share_no_run_id_with_the_runs_they_are_compared_with():
    faulty, control = diagnosis.test_run_ids(), diagnosis.test_run_ids(offset=100)
    assert len(control) == 100 and not set(faulty) & set(control)
    limit = list(range(100, 200))
    dev_control = diagnosis.dev_control_ids(limit)
    assert len(dev_control) == len(set(dev_control)) == 20 and set(dev_control) <= set(limit)
    assert diagnosis.dev_control_ids(limit) == dev_control and dev_control == sorted(dev_control)


def test_blame_rows_carry_the_baseline_notes_and_top_names():
    tags = data.VARIABLES
    vec = np.full((4, 52), 0.1)
    vec[:, 0] = 9.0                                                           # xmeas_1 (A feed) blamed in every run
    rows = diagnosis.blame_rows(6, {"post": {"spe": vec}, "first_hour": {"spe": vec}}, tags)
    assert [(r["method"], r["window"]) for r in rows] == [("spe", "post"), ("spe", "first_hour")]
    r = rows[0]
    assert r["top1_hit"] == 1.0 and r["top5_hit"] == 1.0 and r["n_runs"] == 4 and r["expected_unit"] == "feeds"
    assert r["baseline_unit_share"] == pytest.approx(8 / 52) and r["baseline_top5_hit"] == pytest.approx(1 - (44 / 52) ** 5)
    assert r["top3"].startswith("A feed (xmeas_1)") and r["note"] == "" and r["group"] == "benchmark"
    unknown = diagnosis.blame_rows(16, {"post": {"spe": vec}}, tags)[0]
    assert unknown["expected_unit"] == "" and np.isnan(unknown["top1_hit"]) and unknown["smear"] > 0
    assert unknown["note"] == "unknown cause"
    hidden = diagnosis.blame_rows(3, {"post": {"spe": vec}}, tags)[0]
    assert hidden["note"] == "near the false-alarm level; blame is mostly noise"


def test_group_summary_averages_the_faults_of_each_group_per_method():
    rows = []
    for fault, hit in ((1, 1.0), (2, 0.0), (3, 0.5)):
        vec = np.zeros((2, 52)) + (0.1 if hit < 1 else 0.0)
        rows += [{**r, "top5_hit": hit} for r in diagnosis.blame_rows(fault, {"post": {"spe": vec + 0.01}}, data.VARIABLES)]
    summary = diagnosis.group_summary(pd.DataFrame(rows)).set_index("group")
    assert summary.loc["benchmark", "top5_hit"] == 0.5 and summary.loc["benchmark", "n_faults"] == 2
    assert summary.loc["hidden", "top5_hit"] == 0.5


# --- from frames of runs to per-run blame vectors ------------------------------------------------------------------------
def normal_frame(n_runs, seed, n=300):
    rng = np.random.default_rng(seed)
    scale, base = np.linspace(0.5, 2, 52), np.linspace(10, 20, 52)
    parts = []
    for r in range(1, n_runs + 1):
        frame = pd.DataFrame(rng.normal(size=(n, 52)) * scale + base, columns=data.VARIABLES)
        frame.insert(0, "sample", np.arange(1, n + 1))
        frame.insert(0, "simulationRun", r)
        frame.insert(0, "faultNumber", 0)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def fitted():
    fit = normal_frame(6, seed=1)
    monitor = mspc.PCAMonitor(lags=0, cpv=0.9).fit(fit)
    return monitor, diagnosis.normal_references(monitor, fit)


def test_window_masks_are_the_post_onset_samples_and_the_first_20_of_them():
    masks = diagnosis.window_masks(np.arange(1, 61), onset=20)
    assert masks["post"].sum() == 40 and masks["first_hour"].sum() == 20
    assert np.flatnonzero(masks["first_hour"])[[0, -1]].tolist() == [20, 39]       # samples 21 .. 40
    assert diagnosis.window_masks(np.arange(1, 161), 160)["post"].sum() == 0


def test_blame_vectors_point_at_the_shifted_reading_in_each_run_and_window(fitted):
    monitor, refs = fitted
    frame = normal_frame(3, seed=7, n=100)
    frame["faultNumber"] = 6
    late = (frame["sample"] > 20)
    frame.loc[late, "xmeas_1"] += 8.0                                                 # a big step on the first reading
    runs, vec = diagnosis.blame_vectors(monitor, refs, frame, onset=20)
    assert runs == [1, 2, 3]
    for window in diagnosis.WINDOWS:
        for method in diagnosis.PCA_METHODS:
            block = vec[window][method]
            assert block.shape == (3, 52) and (block.argmax(axis=1) == 0).all(), (window, method)
    # a step that starts after the first hour is invisible to the first-hour window but not to the post-onset window
    later = normal_frame(3, seed=7, n=100)
    later.loc[later["sample"] > 60, "xmeas_1"] += 8.0
    _, vec_later = diagnosis.blame_vectors(monitor, refs, later, onset=20)
    assert vec_later["post"]["spe"][:, 0].min() > 5 * vec_later["first_hour"]["spe"][:, 0].max()


def test_relative_rbc_spe_equals_relative_spe_because_the_reconstruction_denominator_is_a_constant_per_reading(fitted):
    # rbc_j = e_j^2 / C~_jj with C~_jj fixed per reading, so dividing by the normal mean of rbc_j cancels it: the two relative
    # forms agree wherever the 1% floor is not applied (the floors differ because the references have different means)
    monitor, refs = fitted
    frame = normal_frame(3, seed=7, n=100)
    frame.loc[frame["sample"] > 20, ["xmeas_3", "xmeas_9"]] += 6.0
    _, vec = diagnosis.blame_vectors(monitor, refs, frame, onset=20)
    free = (refs["spe"] >= 0.01 * refs["spe"].mean()) & (refs["rbc_spe"] >= 0.01 * refs["rbc_spe"].mean())
    assert free.sum() > 40
    assert np.allclose(vec["post"]["rbc_spe"][:, free], vec["post"]["spe"][:, free], rtol=1e-9)
    assert not np.allclose(vec["post"]["rbc_t2"], vec["post"]["t2"], rtol=1e-3)         # for T2 the two forms differ


def test_relative_blame_of_fault_free_runs_is_about_one_for_every_pca_method(fitted):
    monitor, refs = fitted
    frame = normal_frame(5, seed=99)
    _, vec = diagnosis.blame_vectors(monitor, refs, frame, onset=20)
    for method in diagnosis.PCA_METHODS:
        assert 0.5 < np.median(vec["post"][method].mean(axis=0)) < 2.0, method


def test_rbc_reference_round_trips_through_its_file(tmp_path, fitted):
    monitor, refs = fitted
    path = tmp_path / "rbc_reference.npz"
    diagnosis.save_rbc_reference(path, refs, "test machine", 6)
    npz = tmp_path / "pca.npz"
    monitor.save(npz)
    from src import contributions
    contributions.add_reference(npz, {"spe": refs["spe"], "t2": refs["t2"]})
    loaded = diagnosis.load_references(npz, path)
    assert set(loaded) == set(diagnosis.PCA_METHODS)
    for m in diagnosis.PCA_METHODS:
        assert np.array_equal(loaded[m], refs[m])
    with np.load(path) as f:
        assert str(f["machine"]) == "test machine" and int(f["n_runs"]) == 6


def test_top_readings_lists_the_ten_largest_spe_readings_with_names_and_every_methods_value_and_rank():
    tags = data.VARIABLES
    spe = np.linspace(1, 2, 52)
    spe[5] = 50.0
    out = diagnosis.top_readings(3, {"spe": spe, "t2": spe[::-1].copy()}, tags, extra={"perm": np.arange(52.0)})
    assert len(out) == 10 and out["tag"].iloc[0] == "xmeas_6" and out["name"].iloc[0] == "Reactor feed rate"
    assert out["plant_unit"].iloc[0] == "reactor" and out["scoring_unit"].iloc[0] == "reactor"
    assert out["rank_spe"].tolist() == list(range(1, 11))
    assert {"spe", "t2", "perm", "rank_t2", "rank_perm", "unit", "note"} <= set(out.columns)
    assert (out["note"] == "near the false-alarm level; blame is mostly noise").all()


# --- permutation importance sums the 3 lags of a reading ------------------------------------------------------------------
def test_sum_over_lags_adds_the_three_lags_of_each_reading():
    # lagged_rows layout: columns [x(k) (2 readings), x(k-1), x(k-2)] -> reading 0 owns columns 0, 2, 4 and reading 1 columns 1, 3, 5
    assert diagnosis.sum_over_lags(np.arange(6.0), n_lags=3).tolist() == [0 + 2 + 4, 1 + 3 + 5]


def test_the_lag_layout_matches_mspc_lagged_rows():
    frame = pd.DataFrame({"faultNumber": 0, "simulationRun": 1, "sample": [1, 2, 3, 4], "a": [10, 20, 30, 40],
                          "b": [1, 2, 3, 4]})
    _, x = mspc.lagged_rows(frame, ["a", "b"], 2)
    assert x[0].tolist() == [30, 3, 20, 2, 10, 1]                     # a(k), b(k), a(k-1), b(k-1), a(k-2), b(k-2)


def test_permutation_blame_names_the_reading_each_fault_depends_on_and_sums_its_lags():
    rng = np.random.default_rng(0)
    n, n_readings, n_lags = 600, 3, 2
    y = rng.integers(0, 3, size=n)                                    # classes 0 (normal), 1, 2
    x = rng.normal(scale=0.1, size=(n, n_readings * n_lags))
    x[y == 1, 1] += 5.0                                               # fault 1 shows in reading 1 at lag 0 (column 1)
    x[y == 2, 3] += 5.0                                               # fault 2 shows in reading 0 at lag 1 (column 3)
    tree = DecisionTreeClassifier(random_state=0).fit(x, y)
    out = diagnosis.permutation_blame(tree, x, y, faults=[1, 2], n_lags=n_lags, n_repeats=5, seed=42, n_jobs=1)
    assert set(out) == {1, 2} and all(v.shape == (n_readings,) for v in out.values())
    assert out[1].argmax() == 1 and out[2].argmax() == 0
    assert out[1][2] < 0.02 and out[2][2] < 0.02                      # the noise reading carries nothing
    again = diagnosis.permutation_blame(tree, x, y, faults=[1, 2], n_lags=n_lags, n_repeats=5, seed=42, n_jobs=1)
    assert np.array_equal(out[1], again[1])                           # random_state 42: repeatable


# --- before / after inside a run ------------------------------------------------------------------------------------------
# sample 1-4 = before (onset 4), 5-8 = after. Three readings with the fault-free sd (2, 1, 1):
# a: [0,2,0,2] -> [4,6,4,6]   mean +4 = 2 sd, same spread, acf1 -0.75 both
# b: [0,2,0,2] -> [-2,4,-2,4] same mean, spread x3
# c: [1,2,3,4] (acf1 = 1.25 / 5 = 0.25) -> [4,1,3,2] (acf1 = -3.25 / 5 = -0.65), same mean 2.5 and spread
VALUES = np.array([[0, 0, 1], [2, 2, 2], [0, 0, 3], [2, 2, 4], [4, -2, 4], [6, 4, 1], [4, -2, 3], [6, 4, 2]], dtype=float)
SD = np.array([2.0, 1.0, 1.0])


def test_lag1_autocorrelation_matches_the_hand_value():
    assert diagnosis.lag1_autocorrelation(np.array([[1.0], [2.0], [3.0], [4.0]])) == pytest.approx([0.25])


def test_before_after_recovers_a_known_shift_spread_and_autocorrelation_change():
    out = diagnosis.before_after(VALUES, np.arange(1, 9), onset=4, sd=SD)
    assert out["shift"] == pytest.approx([2.0, 0.0, 0.0])
    assert out["spread"] == pytest.approx([1.0, 3.0, 1.0])
    assert out["dacf"] == pytest.approx([0.0, 0.0, -0.9])


def test_before_after_needs_three_samples_in_each_window():
    with pytest.raises(ValueError):
        diagnosis.before_after(VALUES, np.arange(1, 9), onset=2, sd=SD)


def test_run_change_stats_report_the_largest_shift_its_reading_and_the_counts():
    frame = pd.DataFrame(VALUES, columns=["xmeas_1", "xmeas_2", "xmeas_3"])
    frame.insert(0, "sample", np.arange(1, 9))
    frame.insert(0, "simulationRun", 5)
    monitor = SimpleNamespace(variables=["xmeas_1", "xmeas_2", "xmeas_3"], std=SD)
    stats = diagnosis.run_change_stats(frame, 4, monitor)
    row = stats.iloc[0]
    assert (row["run"], row["max_abs_shift"], row["top_reading"], row["n_shift_gt_1sd"]) == (5, pytest.approx(2.0), "xmeas_1", 1)
    assert row["max_spread_ratio"] == pytest.approx(3.0) and row["max_abs_dacf"] == pytest.approx(0.9)


def stats_frame(shifts):
    return pd.DataFrame({"run": range(len(shifts)), "max_abs_shift": shifts, "top_reading": "xmeas_1",
                         "n_shift_gt_1sd": 0, "max_spread_ratio": 1.2, "max_abs_dacf": 0.1})


def test_hidden_table_compares_every_fault_with_the_control_and_flags_a_clear_shift_on_3_9_15():
    table = diagnosis.hidden_table({1: stats_frame([5.0, 5.0, 6.0]), 3: stats_frame([0.6, 0.6, 0.7]),
                                    9: stats_frame([2.0, 2.0, 2.0])}, control=stats_frame([0.5, 0.5, 0.5])).set_index("fault")
    assert table.loc[0, "verdict"] == "control (the noise floor)" and table.loc[0, "group"] == "control"
    assert table.loc[3, "effect_vs_control"] == pytest.approx(1.2)
    assert table.loc[3, "verdict"] == "looks like the fault-free control"
    assert table.loc[1, "verdict"] == "clear shift, well above the control"
    assert table.loc[9, "effect_vs_control"] == pytest.approx(4.0)
    messages = diagnosis.hidden_stop_messages(table.reset_index())
    assert len(messages) == 1 and messages[0].startswith("fault 9")                    # fault 1 is not a hidden fault
    assert diagnosis.hidden_stop_messages(diagnosis.hidden_table({3: stats_frame([1.4, 1.4])}, stats_frame([0.5, 0.5]))) == []
