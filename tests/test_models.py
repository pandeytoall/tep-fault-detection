"""Phase 3 tests for src/models.py (Isolation Forest, autoencoder, gradient boosting) on small synthetic runs.

Synthetic plant: 6 variables driven by 2 latent factors, so normal rows are correlated; a fault shifts the variables
after the onset (training-family onset 20). The scaler is the identity (mean 0, std 1) unless a test says otherwise.
"""
import numpy as np
import pandas as pd
import pytest

from src import evaluate, models

VARS = [f"xmeas_{i}" for i in range(1, 7)]
MEAN, STD = np.zeros(6), np.ones(6)
ONSET = 20
_W = np.random.default_rng(0).normal(size=(2, 6))


def runs(run_ids, fault=0, n=60, seed=1, shift=None, flip=False):
    """Frame of simulated runs: faultNumber, simulationRun, sample + 6 variables. `shift` is added after the onset."""
    rng = np.random.default_rng(seed)
    parts = []
    for r in run_ids:
        z = rng.normal(size=(n, 2))
        x = z @ _W + 0.1 * rng.normal(size=(n, 6))
        if flip:
            x[:, 3:] *= -1                      # breaks the correlation structure
        if shift is not None:
            x[ONSET:] += shift
        frame = pd.DataFrame(x, columns=VARS)
        frame.insert(0, "sample", np.arange(1, n + 1))
        frame.insert(0, "simulationRun", r)
        frame.insert(0, "faultNumber", fault)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def shifted(fault):
    v = np.zeros(6)
    v[(fault - 1) % 6] = 6.0
    return v


def supervised_data(faults=(1, 2, 3), run_ids=range(1, 7), n=60):
    faulty = pd.concat([runs(run_ids, f, n, seed=10 + f, shift=shifted(f)) for f in faults], ignore_index=True)
    return faulty, runs(run_ids, 0, n, seed=5)


# --- the autoencoder's run-level validation split and early stopping -------------------------------------------------
def test_validation_runs_are_split_by_run_with_no_run_on_both_sides():
    ids = list(range(1, 331))
    train, val = models.split_runs(ids)
    assert len(val) == 30 and len(train) == 300
    assert not set(train) & set(val) and sorted(train + val) == ids
    assert models.split_runs(ids) == (train, val)                   # seed 42: the same split every time
    assert models.split_runs(ids[::-1]) == (train, val)             # whatever the order they come in


def test_early_stopper_stops_after_patience_epochs_without_improvement_and_remembers_the_best():
    # best is epoch 2 (0.7); 5 epochs without improvement after it end at epoch 7, so the 0.2 at epoch 8 is never seen
    series = [1.0, 0.8, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.2]
    stopper = models.EarlyStopper(patience=5)
    stops = []
    for v in series:
        stops.append(stopper.update(v))
        if stops[-1]:
            break
    assert stops == [False] * 7 + [True]
    assert stopper.best_epoch == 2 and stopper.best == 0.7


def test_early_stopper_counts_equal_values_as_no_improvement_and_resets_on_improvement():
    stopper = models.EarlyStopper(patience=2)
    assert [stopper.update(v) for v in [1.0, 1.0, 0.9, 0.95]] == [False, False, False, False]
    assert stopper.update(0.95) is True
    assert stopper.best_epoch == 2


# --- fitting: autoencoder ---------------------------------------------------------------------------------------------
def fit_ae(lags=0, patience=3, max_epochs=40, n_runs=14, n_val=4, bottleneck=2):
    ae = models.AEMonitor(MEAN, STD, lags=lags, variables=VARS)
    return ae.fit(runs(range(1, n_runs + 1), n=100, seed=2), bottleneck=bottleneck, max_epochs=max_epochs,
                  patience=patience, n_val_runs=n_val)


def test_ae_trains_and_validates_on_different_runs_of_the_fit_group_and_records_both():
    ae = fit_ae()
    train, val = ae.meta["train_runs"], ae.meta["val_runs"]
    assert len(val) == 4 and len(train) == 10 and not set(train) & set(val)
    assert sorted(train + val) == list(range(1, 15))


def test_ae_has_the_agreed_shape_and_keeps_the_best_epoch_weights():
    ae = fit_ae()
    assert ae.estimator.hidden_layer_sizes == (4, 2, 4)             # (2b, b, 2b) with b = bottleneck 2
    hp = ae.meta["hyperparameters"]
    assert hp["bottleneck"] == 2 and hp["batch_size"] == 256 and hp["activation"] == "relu" and hp["solver"] == "adam"
    history, best = ae.meta["val_mse_history"], ae.meta["best_epoch"]
    assert best == int(np.argmin(history))
    assert len(history) == min(40, best + 1 + 3)                    # stopped `patience` epochs after the best, or ran out
    assert ae.meta["final_val_mse"] == pytest.approx(min(history), rel=1e-4)   # best weights restored, not the last ones
    assert ae.meta["final_train_mse"] > 0


def test_ae_stops_when_validation_stalls_and_returns_the_weights_of_the_best_epoch(monkeypatch):
    # scripted validation MSE per epoch: best at epoch 2, then 3 epochs without improvement (patience 3) -> stop at epoch 5
    series = iter([1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.1, 0.1])
    monkeypatch.setattr(models, "reconstruction_mse", lambda est, x: np.full(len(x), next(series, 0.5)))
    seen = []
    real = models.MLPRegressor.partial_fit

    def spy(self, x, y):
        out = real(self, x, y)
        seen.append(self.coefs_[0].copy())
        return out

    monkeypatch.setattr(models.MLPRegressor, "partial_fit", spy)
    ae = fit_ae(patience=3, max_epochs=100)
    assert ae.meta["val_mse_history"] == pytest.approx([1.0, 0.9, 0.8, 0.85, 0.86, 0.87]) and ae.meta["best_epoch"] == 2
    assert len(seen) == 6 and np.array_equal(ae.estimator.coefs_[0], seen[2])
    assert not np.array_equal(seen[2], seen[5])


def test_ae_with_lags_sees_stacked_samples_and_scores_no_first_rows():
    ae = fit_ae(lags=2)
    assert ae.estimator.hidden_layer_sizes == (4, 2, 4)
    assert ae.estimator.coefs_[0].shape[0] == 6 * 3
    scores = ae.score(runs([50], n=30, seed=9))
    assert scores["sample"].min() == 3 and len(scores) == 28


# --- score direction: anomalies score higher ---------------------------------------------------------------------------
def test_isolation_forest_scores_anomalies_higher_than_normal_rows():
    iforest = models.IFMonitor(MEAN, STD, lags=0, variables=VARS).fit(runs(range(1, 31), n=100, seed=3))
    normal = iforest.score(runs([101, 102], n=100, seed=4))["alarm"]
    odd = iforest.score(runs([103, 104], n=100, seed=4, shift=np.full(6, 8.0)))["alarm"]
    assert odd.iloc[ONSET + 5:100].mean() > normal.mean() + 0.1
    assert iforest.estimator.n_estimators == 200 and iforest.estimator.max_samples == 256
    assert iforest.estimator.random_state == 42


def test_isolation_forest_fits_on_at_most_100000_rows_drawn_from_the_fit_runs():
    iforest = models.IFMonitor(MEAN, STD, lags=0, variables=VARS)
    iforest.fit(runs(range(1, 21), n=100, seed=3), max_rows=500)
    assert iforest.meta["n_train_rows"] == 500
    assert set(iforest.meta["train_runs"]) <= set(range(1, 21))
    assert models.IF_ROWS == 100_000


def test_autoencoder_scores_rows_that_break_the_learned_correlations_higher():
    ae = fit_ae(max_epochs=80, patience=8, n_runs=30, n_val=5)
    normal = ae.score(runs([101, 102], n=100, seed=4))["alarm"]
    odd = ae.score(runs([103, 104], n=100, seed=4, flip=True))["alarm"]
    assert odd.mean() > 2 * normal.mean()


# --- gradient boosting -------------------------------------------------------------------------------------------------
def test_hgb_classes_follow_the_onset_rule():
    ids = pd.DataFrame({"faultNumber": [0, 3, 3, 3, 0], "simulationRun": 1, "sample": [25, 19, 20, 21, 30]})
    assert list(models.hgb_classes(ids, ONSET)) == [0, 0, 0, 3, 0]


def fit_hgb(faults=(1, 2, 3), lags=2, **kw):
    faulty, normal = supervised_data(faults)
    frame = models.supervised_frame(faulty, normal)
    return models.HGBMonitor(MEAN, STD, lags=lags, variables=VARS).fit(frame, onset=ONSET, max_iter=20, **kw), frame


def test_hgb_trains_on_pre_onset_rows_as_normal_and_post_onset_rows_as_the_fault_class():
    hgb, _ = fit_hgb()
    counts = hgb.meta["class_counts"]
    # 6 normal runs x 58 rows, plus 18 pre-onset rows (samples 3-20) in each of 3 faults x 6 faulty runs
    assert counts["0"] == 6 * 58 + 3 * 6 * 18
    assert counts["1"] == counts["2"] == counts["3"] == 6 * 40
    assert list(hgb.estimator.classes_) == [0, 1, 2, 3]


def test_hgb_is_a_21_class_style_model_without_built_in_early_stopping():
    hgb, _ = fit_hgb()
    est = hgb.estimator
    assert est.early_stopping is False and est.random_state == 42
    assert hgb.meta["hyperparameters"] == {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 20}


def test_hgb_alarm_is_one_minus_the_probability_of_normal():
    hgb, _ = fit_hgb()
    test = runs([20, 21], fault=2, n=60, seed=33, shift=shifted(2))
    ids, proba = hgb.proba(test)
    alarm = hgb.score(test)
    assert np.allclose(alarm["alarm"], 1 - proba[:, 0], atol=1e-9)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(alarm[["simulationRun", "sample"]].itertuples(index=False)) == \
        list(ids[["simulationRun", "sample"]].itertuples(index=False))
    post = alarm[alarm["sample"] > ONSET + 5]["alarm"]
    assert post.mean() > 0.9                                          # a shifted fault run is flagged


def test_hgb_probability_columns_follow_its_classes():
    hgb, _ = fit_hgb()
    ids, proba = hgb.proba(runs([30], fault=3, n=60, seed=34, shift=shifted(3)))
    late = proba[ids["sample"].to_numpy() > ONSET + 5]
    assert list(hgb.estimator.classes_[late.mean(axis=0).argmax(keepdims=True)]) == [3]


def test_hgb_row_stride_thins_the_rows_but_keeps_every_run_and_the_onset_labels():
    faulty, normal = supervised_data()
    frame = models.supervised_frame(faulty, normal)
    hgb = models.HGBMonitor(MEAN, STD, lags=2, variables=VARS).fit(frame, onset=ONSET, max_iter=5, stride=3)
    # lagged rows are samples 3-60; every third one (3, 6, ..., 60) = 20 per run, 24 runs
    assert hgb.meta["row_stride"] == 3 and hgb.meta["n_train_rows"] == 24 * 20
    assert hgb.meta["train_runs"] == list(range(1, 7))
    counts = hgb.meta["class_counts"]
    assert counts["0"] == 6 * 20 + 3 * 6 * 6          # normal runs + samples 3-18 (6 rows) before the onset of each faulty run
    assert counts["1"] == 6 * 14                       # samples 21-60 (14 rows)


def test_the_unseen_fault_training_set_has_no_rows_of_the_left_out_faults():
    faulty, normal = supervised_data(faults=(1, 2, 3, 8))
    frame = models.supervised_frame(faulty, normal, exclude=(1, 8))
    assert sorted(frame["faultNumber"].unique()) == [0, 2, 3]
    assert len(frame) == len(normal) + 2 * len(runs(range(1, 7), n=60))
    hgb = models.HGBMonitor(MEAN, STD, lags=0, variables=VARS, name="hgb-l2-unseen").fit(frame, onset=ONSET, max_iter=5)
    assert list(hgb.estimator.classes_) == [0, 2, 3] and hgb.name == "hgb-l2-unseen"
    assert hgb.meta["faults_trained"] == [2, 3]


def test_supervised_frame_needs_faulty_runs_with_faults_and_normal_runs_without():
    faulty, normal = supervised_data()
    with pytest.raises(ValueError):
        models.supervised_frame(normal, faulty)


def test_hgb_choice_takes_the_best_dev_fdr_then_the_fewest_trees_within_the_tolerance():
    grid = pd.DataFrame({
        "learning_rate": [0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1],
        "max_leaf_nodes": [31, 31, 63, 63, 31, 31, 63, 63],
        "max_iter": [200, 400, 200, 400, 200, 400, 200, 400],
        "dev_mean_fdr_excl_3_9_15": [0.80, 0.83, 0.82, 0.84, 0.835, 0.838, 0.84, 0.842],
    })
    # best 0.842; within 0.005: 0.84 (63 leaves, 400), 0.84 (63, 200 trees), 0.838 (400), 0.84 ... fewest iterations wins
    chosen = models.pick_hgb(grid)
    assert chosen == {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 200}
    only_best = grid.assign(dev_mean_fdr_excl_3_9_15=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    assert models.pick_hgb(only_best) == {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 400}
    # an exact tolerance edge counts as within (0.005 below the best)
    edge = grid.assign(dev_mean_fdr_excl_3_9_15=[0.895, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.9])
    assert models.pick_hgb(edge) == {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 200}
    # equal trees: the higher dev FDR wins
    tie = grid.assign(dev_mean_fdr_excl_3_9_15=[0.7, 0.7, 0.8, 0.7, 0.7, 0.7, 0.81, 0.7])
    assert models.pick_hgb(tie) == {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 200}


# --- limits ---------------------------------------------------------------------------------------------------------------
def test_limits_come_from_the_normal_limit_runs_and_drop_each_runs_first_skip_samples():
    ae = fit_ae(lags=2)
    limit_runs = runs(range(201, 206), n=40, seed=8)
    ae.calibrate(limit_runs)
    scores = ae.score(limit_runs)
    assert scores["sample"].min() == 3 and len(scores) == 5 * 38           # first L = 2 samples of every run dropped
    assert ae.limits["1%"] == evaluate.limit_for_far(scores["alarm"], 0.01)
    assert ae.limits["5%"] == evaluate.limit_for_far(scores["alarm"], 0.05)
    assert ae.limits["5%"] <= ae.limits["1%"]
    assert ae.meta["limit_runs"] == list(range(201, 206))
    assert evaluate.far(scores.rename(columns={"alarm": "score"}), ae.limits["1%"]) <= 0.01


def test_calibrating_on_faulty_runs_is_refused():
    ae = fit_ae()
    with pytest.raises(ValueError, match="normal"):
        ae.calibrate(runs([300], fault=1, n=40))


def test_fitting_the_unsupervised_models_on_faulty_runs_is_refused():
    with pytest.raises(ValueError, match="normal"):
        models.IFMonitor(MEAN, STD, variables=VARS).fit(runs([1], fault=2))
    with pytest.raises(ValueError, match="normal"):
        models.AEMonitor(MEAN, STD, variables=VARS).fit(runs(range(1, 8), fault=2), bottleneck=2)


# --- save / load ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def one_of_each():
    iforest = models.IFMonitor(MEAN, STD, lags=2, variables=VARS).fit(runs(range(1, 11), n=80, seed=3))
    ae = fit_ae(lags=0)
    hgb, _ = fit_hgb()
    limit_runs = runs(range(201, 205), n=50, seed=8)
    for m in (iforest, ae, hgb):
        m.calibrate(limit_runs)
    hgb.meta["tuning_limit_runs"] = [201, 202]
    return {"iforest": iforest, "ae": ae, "hgb": hgb}


@pytest.mark.parametrize("kind", ["iforest", "ae", "hgb"])
def test_save_and_load_round_trip_gives_identical_scores_limits_and_record(tmp_path, one_of_each, kind):
    model = one_of_each[kind]
    path = tmp_path / f"{model.name}.joblib"
    model.save(path)
    again = models.load(path)
    assert type(again) is type(model) and again.name == model.name and again.lags == model.lags
    frame = runs([60, 61], fault=2, n=60, seed=21, shift=shifted(2))
    assert np.array_equal(again.score(frame)["alarm"].to_numpy(), model.score(frame)["alarm"].to_numpy())
    assert again.limits == model.limits and again.meta == model.meta
    assert np.array_equal(again.mean, model.mean) and np.array_equal(again.std, model.std)


def test_model_names_follow_kind_and_lag(one_of_each):
    assert one_of_each["iforest"].name == "iforest-l2" and one_of_each["ae"].name == "ae-l0"
    assert one_of_each["hgb"].name == "hgb-l2"
    assert [one_of_each[k].track for k in ("iforest", "ae", "hgb")] == ["unsupervised", "unsupervised", "supervised"]


def test_a_model_is_callable_like_its_score_method(one_of_each):
    frame = runs([70], n=40, seed=3)
    model = one_of_each["iforest"]
    assert model(frame).equals(model.score(frame))


def test_scoring_in_chunks_equals_scoring_in_one_go(one_of_each, monkeypatch):
    from src import mspc
    frame = runs(range(300, 312), n=60, seed=6)
    whole = one_of_each["hgb"].score(frame)["alarm"].to_numpy()
    monkeypatch.setattr(mspc, "CHUNK_ROWS", 100)                  # a few runs per chunk
    assert np.allclose(one_of_each["hgb"].score(frame)["alarm"].to_numpy(), whole, atol=1e-12)


# --- run groups of the saved models --------------------------------------------------------------------------------------------
GROUPS = {"fit": list(range(1, 15)), "limit": list(range(201, 205)), "limit_tune": [201, 202], "supervised": list(range(1, 7)),
          "dev": [300]}


def test_every_run_a_model_used_lies_in_the_group_of_its_job(one_of_each):
    groups = dict(GROUPS)
    assert models.run_group_problems(one_of_each["ae"].meta, groups) == []
    assert models.run_group_problems(one_of_each["iforest"].meta, groups) == []
    assert models.run_group_problems(one_of_each["hgb"].meta, groups) == []


def test_a_model_trained_or_limited_on_runs_of_another_group_is_reported(one_of_each):
    groups = {**GROUPS, "fit": list(range(1, 12))}                  # AE validation runs now partly outside `fit`
    problems = models.run_group_problems(one_of_each["ae"].meta, groups)
    assert problems and all("fit" in p for p in problems)
    meta = dict(one_of_each["hgb"].meta, train_runs=[1, 2, 201], tuning_limit_runs=[203])
    problems = models.run_group_problems(meta, GROUPS)
    assert any("train_runs" in p and "supervised" in p for p in problems)
    assert any("tuning_limit_runs" in p and "limit_tune" in p for p in problems)
    meta = dict(one_of_each["iforest"].meta, limit_runs=[1])
    assert any("limit_runs" in p and "limit" in p for p in models.run_group_problems(meta, GROUPS))


# --- scorecard rows -------------------------------------------------------------------------------------------------------------
def test_scorecard_rows_cover_both_limits_with_the_models_name_and_track(one_of_each):
    hgb = one_of_each["hgb"]
    faulty = hgb.score(runs(range(40, 44), fault=1, n=60, seed=1, shift=shifted(1)))
    normal = hgb.score(runs(range(40, 44), fault=0, n=60, seed=2))
    rows = models.scorecard_rows(hgb, faulty, normal, onset=ONSET, machine="m", split_file="s", train_time_s=1.5,
                                 peak_ram_gb=0.5)
    df = pd.DataFrame(rows)
    assert set(df["model"]) == {"hgb-l2"} and set(df["track"]) == {"supervised"}
    assert set(df["limit_name"]) == {"1%", "5%"}
    assert df[(df.limit_name == "1%") & (df.metric == "train_time_s")]["value"].iloc[0] == 1.5
    far5 = df[(df.limit_name == "5%") & (df.metric == "far")]["value"].iloc[0]
    assert far5 == evaluate.far(normal.rename(columns={"alarm": "score"}), hgb.limits["5%"], skip=2)


# --- the final-run checks ----------------------------------------------------------------------------------------------------------
def _scorecard(fdr_by_model, far_by_model, limit_name="1%"):
    rows = []
    for m, fdrs in fdr_by_model.items():
        rows += [{"model": m, "limit_name": limit_name, "metric": "fdr", "fault": f, "value": v} for f, v in fdrs.items()]
        rows.append({"model": m, "limit_name": limit_name, "metric": "far", "fault": None, "value": far_by_model[m]})
    return pd.DataFrame(rows)


def _fdrs(**over):
    base = {f: 0.95 for f in range(1, 21)}
    base.update({3: 0.02, 9: 0.02, 15: 0.02})
    base.update({int(k[1:]): v for k, v in over.items()})
    return base


def test_final_checks_pass_on_a_plausible_scorecard():
    df = _scorecard({"iforest-l0": _fdrs(), "hgb-l2": _fdrs(), "hgb-l2-unseen": _fdrs(f1=0.1, f8=0.1, f13=0.1)},
                    {"iforest-l0": 0.01, "hgb-l2": 0.012, "hgb-l2-unseen": 0.01})
    stops, notes = models.final_check_failures(df, ["iforest-l0"], ["hgb-l2"], unseen="hgb-l2-unseen")
    assert stops == [] and notes == []


def test_final_checks_stop_on_unsupervised_leaks_weak_hgb_faults_and_a_hgb_above_half_on_3_9_15():
    df = _scorecard({"ae-l2": _fdrs(f9=0.2), "hgb-l2": _fdrs(f1=0.5, f3=0.6)}, {"ae-l2": 0.01, "hgb-l2": 0.01})
    stops, notes = models.final_check_failures(df, ["ae-l2"], ["hgb-l2"])
    text = "\n".join(stops)
    assert "ae-l2" in text and "3/9/15" in text            # unsupervised above 0.10
    assert "hgb-l2" in text and "fault 1" in text          # plausibility: faults 1, 2, 4, 6, 7 at least 0.9
    assert "above 0.5" in text                             # supervised above 0.5 on 3/9/15


def test_final_checks_list_hgb_fdr_between_a_tenth_and_a_half_on_3_9_15_as_a_note_and_flag_far_out_of_range():
    df = _scorecard({"hgb-l2": _fdrs(f9=0.3)}, {"hgb-l2": 0.03})
    stops, notes = models.final_check_failures(df, [], ["hgb-l2"])
    assert stops == [] and any("fault 9" in n and "0.30" in n for n in notes)
    assert any("FAR" in n and "hgb-l2" in n for n in notes)


def test_final_checks_want_the_unseen_model_worse_on_most_unseen_faults():
    good = _fdrs(f1=0.1, f8=0.1, f13=0.1, f14=0.95, f16=0.95)
    df = _scorecard({"hgb-l2": _fdrs(f16=0.9), "hgb-l2-unseen": good}, {"hgb-l2": 0.01, "hgb-l2-unseen": 0.01})
    assert models.final_check_failures(df, [], ["hgb-l2"], unseen="hgb-l2-unseen")[1] == []
    bad = _fdrs(f1=0.99, f8=0.99, f13=0.99, f14=0.5, f16=0.5)
    df = _scorecard({"hgb-l2": _fdrs(), "hgb-l2-unseen": bad}, {"hgb-l2": 0.01, "hgb-l2-unseen": 0.01})
    notes = models.final_check_failures(df, [], ["hgb-l2"], unseen="hgb-l2-unseen")[1]
    assert any("unseen" in n for n in notes)


# --- the supervised tuning grid --------------------------------------------------------------------------------------
TINY_GRID = {"learning_rate": (0.1,), "max_leaf_nodes": (4,), "max_iter": (3, 6)}


def tuning_frames(dev_shift=True):
    faulty, normal = supervised_data(run_ids=range(1, 7))                       # `supervised` runs
    tune_normal = runs(range(101, 104), seed=21, flip=True)                                  # `limit_tune` runs
    dev_faulty = pd.concat([runs(range(201, 204), f, seed=40 + f, shift=shifted(f) if dev_shift else None)
                            for f in (1, 2, 3)], ignore_index=True)               # `dev` runs
    dev_normal = runs(range(201, 204), seed=22)
    return models.supervised_frame(faulty, normal), tune_normal, dev_faulty, dev_normal


def test_tuning_takes_each_limit_from_the_limit_tune_runs_and_scores_only_the_dev_runs():
    train, tune_normal, dev_faulty, dev_normal = tuning_frames()
    grid, fits = models.tune_hgb(train, MEAN, STD, 2, tune_normal, dev_faulty, dev_normal, ONSET, "test machine",
                                 grid=TINY_GRID, stride=1, variables=VARS)
    assert list(grid["max_iter"]) == [3, 6] and sorted(fits) == [0, 1]
    assert list(grid["total_trees"]) == [3 * 21, 6 * 21] and set(grid["machine"]) == {"test machine"}
    for i, row in grid.iterrows():
        model = fits[i]
        limit = evaluate.limit_for_far(model.score(tune_normal)["alarm"], 0.01)
        assert row["limit_1pct_limit_tune"] == limit                              # not from dev_normal or the training runs
        assert limit != evaluate.limit_for_far(model.score(dev_normal)["alarm"], 0.01)
        dev_scores = model.score(dev_faulty)
        f = dev_scores[["faultNumber", "simulationRun", "sample"]].assign(score=dev_scores["alarm"].to_numpy())
        per_fault = evaluate.fdr(f, limit, ONSET, model.lags)
        assert row["dev_mean_fdr"] == pytest.approx(per_fault.mean())
        assert row["dev_mean_fdr_excl_3_9_15"] == pytest.approx(per_fault.drop([3]).mean())   # 3/9/15 left out of the choice
        assert row["dev_far"] == pytest.approx((model.score(dev_normal)["alarm"] > limit).mean())
        assert model.meta["train_runs"] == list(range(1, 7))                       # no limit_tune or dev run in the fit


def test_tuning_scores_change_with_the_dev_runs_only():
    train, tune_normal, dev_faulty, dev_normal = tuning_frames()
    same_fit = dict(grid=TINY_GRID, stride=1, variables=VARS)
    shifted_dev, _ = models.tune_hgb(train, MEAN, STD, 2, tune_normal, dev_faulty, dev_normal, ONSET, "m", **same_fit)
    _, _, flat_dev_faulty, _ = tuning_frames(dev_shift=False)
    flat_dev, _ = models.tune_hgb(train, MEAN, STD, 2, tune_normal, flat_dev_faulty, dev_normal, ONSET, "m", **same_fit)
    assert (shifted_dev["limit_1pct_limit_tune"] == flat_dev["limit_1pct_limit_tune"]).all()
    assert (shifted_dev["dev_mean_fdr_excl_3_9_15"] > flat_dev["dev_mean_fdr_excl_3_9_15"] + 0.3).all()
