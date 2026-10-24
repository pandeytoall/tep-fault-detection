"""Phase 1 tests for src/evaluate.py, on synthetic dummy models (no data files).

Samples are 1-based. Score 1.0 means "alarm" (limit 0.5) unless a test says otherwise.
"""
import numpy as np
import pandas as pd
import pytest

from src import evaluate

LIMIT = 0.5
ONSET = 20
N = 100            # samples per synthetic run
POST = N - ONSET   # post-onset samples per faulty run
FAULTS = (1, 2, 3)
RUNS = (1, 2, 3, 4)


def _runs(faults, runs, score_fn, n=N):
    """Frame with one block per (fault, run); score_fn(fault, run, samples) -> scores."""
    parts = []
    for f in faults:
        for r in runs:
            s = np.arange(1, n + 1)
            parts.append(pd.DataFrame({"faultNumber": f, "simulationRun": r, "sample": s, "score": score_fn(f, r, s)}))
    return pd.concat(parts, ignore_index=True)


def _always(f, r, s):
    return np.ones(len(s))


def _never(f, r, s):
    return np.zeros(len(s))


def _oracle(delay=0):
    def fn(f, r, s):
        return ((f > 0) & (s > ONSET + delay)).astype(float)
    return fn


def _at(alarm_samples):
    def fn(f, r, s):
        return np.isin(s, list(alarm_samples)).astype(float)
    return fn


def _faulty(fn, runs=RUNS, n=N):
    return _runs(FAULTS, runs, fn, n)


def _normal(fn, runs=RUNS, n=N):
    return _runs((0,), runs, fn, n)


# --- always / never / oracle ------------------------------------------------
def test_always_alarm():
    faulty, normal = _faulty(_always), _normal(_always)
    assert (evaluate.fdr(faulty, LIMIT, ONSET) == 1).all()
    assert evaluate.far(normal, LIMIT) == 1
    d = evaluate.detection_delay(faulty, LIMIT, ONSET)
    assert (d["delay_median_min"] == 9).all() and (d["never_detected_share"] == 0).all()
    assert evaluate.sustained_rate(normal, LIMIT, after=ONSET) == 1


def test_never_alarm():
    faulty, normal = _faulty(_never), _normal(_never)
    assert (evaluate.fdr(faulty, LIMIT, ONSET) == 0).all()
    assert evaluate.far(normal, LIMIT) == 0
    d = evaluate.detection_delay(faulty, LIMIT, ONSET)
    assert d["delay_median_min"].isna().all() and (d["never_detected_share"] == 1).all()
    assert evaluate.sustained_rate(normal, LIMIT, after=ONSET) == 0


def test_oracle():
    faulty, normal = _faulty(_oracle()), _normal(_oracle())
    assert (evaluate.fdr(faulty, LIMIT, ONSET) == 1).all()
    assert evaluate.far(normal, LIMIT) == 0
    d = evaluate.detection_delay(faulty, LIMIT, ONSET)
    assert (d["delay_median_min"] == 9).all() and (d["never_detected_share"] == 0).all()
    assert list(d.index) == list(FAULTS) and (d["n_runs"] == len(RUNS)).all()


@pytest.mark.parametrize("k", [1, 2, 5, 17])
def test_oracle_delayed_by_k_samples(k):
    faulty = _faulty(_oracle(k))
    assert evaluate.fdr(faulty, LIMIT, ONSET).tolist() == pytest.approx([(POST - k) / POST] * 3)
    assert (evaluate.detection_delay(faulty, LIMIT, ONSET)["delay_median_min"] == (k + 3) * 3).all()


def test_fdr_is_pooled_per_fault_and_indexed_by_fault():
    def fn(f, r, s):  # fault 1: run 1 always alarms, fault 2: nothing, fault 3: run 2 alarms for 40 post-onset samples
        if f == 1 and r == 1:
            return np.ones(len(s))
        if f == 3 and r == 2:
            return ((s > ONSET) & (s <= ONSET + 40)).astype(float)
        return np.zeros(len(s))
    out = evaluate.fdr(_faulty(fn), LIMIT, ONSET)
    assert out.to_dict() == pytest.approx({1: 0.25, 2: 0.0, 3: 40 / 80 / 4})


def test_alarm_is_strictly_above_the_limit():
    frame = _faulty(lambda f, r, s: np.full(len(s), 0.5))
    assert (evaluate.fdr(frame, 0.5, ONSET) == 0).all()
    assert (evaluate.fdr(frame, 0.49, ONSET) == 1).all()


# --- onset rules ------------------------------------------------------------
def test_pre_onset_alarms_do_not_change_fdr():
    base = _faulty(_at(range(ONSET + 1, N + 1)))
    noisy = _faulty(_at(list(range(1, ONSET + 1)) + list(range(ONSET + 1, N + 1))))
    assert evaluate.fdr(noisy, LIMIT, ONSET).tolist() == evaluate.fdr(base, LIMIT, ONSET).tolist() == [1, 1, 1]


def test_three_in_a_row_starting_at_or_before_onset_does_not_count():
    # alarms on onset-1 .. onset+2: the run of 4 straddles onset; the only 3-window entirely after
    # onset would need onset+1..onset+3, which is not alarmed
    straddle = _faulty(_at(range(ONSET - 1, ONSET + 3)))
    d = evaluate.detection_delay(straddle, LIMIT, ONSET)
    assert (d["never_detected_share"] == 1).all() and d["delay_median_min"].isna().all()
    starts_at_onset = _faulty(_at([ONSET, ONSET + 1, ONSET + 2]))
    assert (evaluate.detection_delay(starts_at_onset, LIMIT, ONSET)["never_detected_share"] == 1).all()
    first_valid = _faulty(_at([ONSET + 1, ONSET + 2, ONSET + 3]))
    assert (evaluate.detection_delay(first_valid, LIMIT, ONSET)["delay_median_min"] == 9).all()


def test_delay_uses_the_first_sustained_alarm_not_the_first_alarm():
    # isolated alarms at onset+1 and onset+3, then a real run at onset+10..onset+12
    frame = _faulty(_at([ONSET + 1, ONSET + 3, ONSET + 10, ONSET + 11, ONSET + 12]))
    assert (evaluate.detection_delay(frame, LIMIT, ONSET)["delay_median_min"] == 12 * 3).all()


def test_delay_median_over_detected_runs_and_never_detected_share():
    def fn(f, r, s):  # runs 1,2: detected at 3 and 5 samples after onset; runs 3,4 never alarm
        if r == 1:
            return _at([ONSET + 1, ONSET + 2, ONSET + 3])(f, r, s)
        if r == 2:
            return _at([ONSET + 3, ONSET + 4, ONSET + 5])(f, r, s)
        return np.zeros(len(s))
    d = evaluate.detection_delay(_faulty(fn), LIMIT, ONSET)
    assert (d["delay_median_min"] == 12).all()          # median of 9 and 15 minutes
    assert (d["never_detected_share"] == 0.5).all() and (d["n_runs"] == 4).all()


# --- skip -------------------------------------------------------------------
def test_skip_drops_the_first_samples_of_every_run():
    normal = _normal(_at(range(1, 6)))                  # alarms only on samples 1-5
    assert evaluate.far(normal, LIMIT) == pytest.approx(5 / N)
    assert evaluate.far(normal, LIMIT, skip=5) == 0
    assert evaluate.far(normal, LIMIT, skip=3) == pytest.approx(2 / (N - 3))
    assert evaluate.sustained_rate(normal, LIMIT, after=0) == 1
    assert evaluate.sustained_rate(normal, LIMIT, after=0, skip=3) == 0   # only 2 alarms left


def test_skip_applies_to_fdr_and_delay():
    onset = 3
    faulty = _runs(FAULTS, RUNS, _at([4, 5, 6]), n=10)  # alarms on samples 4-6; 7 post-onset samples
    assert evaluate.fdr(faulty, LIMIT, onset).tolist() == pytest.approx([3 / 7] * 3)
    assert evaluate.fdr(faulty, LIMIT, onset, skip=5).tolist() == pytest.approx([1 / 5] * 3)  # samples 6-10
    assert (evaluate.detection_delay(faulty, LIMIT, onset)["delay_median_min"] == 9).all()
    assert (evaluate.detection_delay(faulty, LIMIT, onset, skip=4)["never_detected_share"] == 1).all()


# --- run boundaries ---------------------------------------------------------
def test_alarms_across_two_runs_are_not_sustained():
    # run 1 alarms on its last 2 samples, run 2 on its first sample: 3 in a row only across the boundary
    def fn(f, r, s):
        return np.isin(s, [N - 1, N]).astype(float) if r == 1 else (s == 1).astype(float)
    normal = _runs((0,), (1, 2), fn)
    assert evaluate.sustained_rate(normal, LIMIT, after=0) == 0
    faulty = _runs((1,), (1, 2), fn)
    assert evaluate.detection_delay(faulty, LIMIT, onset=0)["never_detected_share"].iloc[0] == 1


def test_sample_numbers_that_line_up_across_runs_are_not_sustained():
    # run 1 alarms on samples 5 and 6, run 2 on sample 7: consecutive sample numbers, but two runs
    def fn(f, r, s):
        return np.isin(s, [5, 6] if r == 1 else [7]).astype(float)
    assert evaluate.sustained_rate(_runs((0,), (1, 2), fn), LIMIT, after=0) == 0
    assert evaluate.detection_delay(_runs((1,), (1, 2), fn), LIMIT, onset=0)["never_detected_share"].iloc[0] == 1


def test_results_do_not_depend_on_row_order():
    frame = _faulty(_oracle(2))
    shuffled = frame.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(evaluate.detection_delay(frame, LIMIT, ONSET),
                                  evaluate.detection_delay(shuffled, LIMIT, ONSET))


# --- sustained rate on fixed fault-free runs --------------------------------
def test_sustained_rate_fixed_runs():
    after, n = 10, 30
    alarms = {1: [15, 16, 17],     # 3 in a row fully after `after`
              2: [9, 10, 11],      # straddles `after`: samples 9 and 10 are not > 10
              3: [20, 21]}         # only 2 in a row
    frame = _runs((0,), (1, 2, 3), lambda f, r, s: np.isin(s, alarms[r]).astype(float), n=n)
    assert evaluate.sustained_rate(frame, LIMIT, after=after) == pytest.approx(1 / 3)


def test_each_metric_only_reads_its_own_rows():
    mixed = pd.concat([_faulty(_always), _normal(_never)], ignore_index=True)
    assert list(evaluate.fdr(mixed, LIMIT, ONSET).index) == list(FAULTS)
    assert list(evaluate.detection_delay(mixed, LIMIT, ONSET).index) == list(FAULTS)
    assert evaluate.far(mixed, LIMIT) == 0
    assert evaluate.sustained_rate(mixed, LIMIT, after=ONSET) == 0


# --- random scores ----------------------------------------------------------
def test_random_scores_at_five_percent_alarm_probability():
    rng = np.random.default_rng(0)
    # uniform scores, limit 0.95: alarm probability 0.05
    normal = _runs((0,), range(1, 201), lambda f, r, s: rng.random(len(s)), n=960)       # 192k samples
    faulty = _runs(range(1, 6), range(1, 21), lambda f, r, s: rng.random(len(s)), n=960)  # 96k samples
    assert len(normal) + len(faulty) > 190_000
    assert evaluate.far(normal, 0.95) == pytest.approx(0.05, abs=0.005)
    pooled = evaluate.fdr(faulty, 0.95, 160).mean()  # equal post-onset size per fault, so the mean is the pooled rate
    assert pooled == pytest.approx(0.05, abs=0.005)


# --- limit_for_far ----------------------------------------------------------
def test_limit_for_far_exact_small_case():
    scores = np.arange(100.0)  # 0..99
    assert evaluate.limit_for_far(scores, 0.05) == 95.0  # 'higher' quantile at 0.95; 96..99 alarm = 4% <= 5%
    assert evaluate.limit_for_far(pd.Series(scores), 0.01) == 99.0


def test_limit_for_far_on_uniform_scores_and_fresh_far():
    rng = np.random.default_rng(1)
    limit = evaluate.limit_for_far(rng.random(300_000), 0.01)
    assert limit == pytest.approx(0.99, abs=0.002)
    fresh = _runs((0,), range(1, 301), lambda f, r, s: rng.random(len(s)), n=1000)
    assert evaluate.far(fresh, limit) == pytest.approx(0.01, abs=0.003)


def test_limit_for_far_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        evaluate.limit_for_far(np.array([0.1, np.nan, 0.3]), 0.01)


# --- scorecard --------------------------------------------------------------
META = dict(machine="laptop, Intel i5-1135G7, 7.7 GB RAM", split_file="results/phase-1/run_split.json")


def _rows(model, limit_name="1%", limit=0.5):
    faulty, normal = _faulty(_oracle()), _normal(_oracle())
    return evaluate.model_rows(model, "unsupervised", limit_name, limit, faulty, normal, onset=ONSET,
                               train_time_s=1.5, peak_ram_gb=0.25, **META)


def test_model_rows_long_format_and_metric_names():
    rows = pd.DataFrame(_rows("m"))
    assert set(rows.columns) == set(evaluate.COLUMNS) - {"written_at"}
    per_fault = rows[rows.fault.notna()]
    assert set(per_fault.metric) == {"fdr", "delay_median_min", "never_detected_share"}
    assert sorted(per_fault.fault.unique()) == [1, 2, 3]
    overall = rows[rows.fault.isna()].set_index("metric")
    assert set(overall.index) == {"mean_fdr", "mean_fdr_excl_3_9_15", "far", "sustained_rate", "train_time_s",
                                  "peak_ram_gb"}
    assert overall.loc["mean_fdr", "value"] == 1 and overall.loc["far", "value"] == 0
    assert overall.loc["train_time_s", "value"] == 1.5 and overall.loc["peak_ram_gb", "value"] == 0.25
    assert overall.loc["far", "n_runs"] == len(RUNS)
    assert (rows.limit_name == "1%").all() and (rows.limit_value == 0.5).all()


def test_mean_fdr_excluding_3_9_15():
    def fn(f, r, s):  # fault 1 is perfect, faults 2 and 3 never alarm
        return _oracle()(f, r, s) if f == 1 else np.zeros(len(s))
    faulty, normal = _faulty(fn), _normal(_never)
    rows = pd.DataFrame(evaluate.model_rows("m", "unsupervised", "1%", 0.5, faulty, normal, onset=ONSET, **META))
    overall = rows[rows.fault.isna()].set_index("metric")["value"]
    assert overall["mean_fdr"] == pytest.approx(1 / 3)
    assert overall["mean_fdr_excl_3_9_15"] == pytest.approx(1 / 2)  # faults 1 and 2 only


def test_write_scorecard_replaces_only_the_matching_model_and_limit(tmp_path):
    path = tmp_path / "scorecard.csv"
    evaluate.write_scorecard(_rows("A", "1%") + _rows("A", "5%") + _rows("B", "1%"), path)
    first = pd.read_csv(path)
    assert list(first.columns) == evaluate.COLUMNS and first.written_at.notna().all()
    n_a1 = ((first.model == "A") & (first.limit_name == "1%")).sum()

    evaluate.write_scorecard(_rows("A", "1%", limit=0.75), path)  # rerun model A at 1% only
    second = pd.read_csv(path)
    a1 = second[(second.model == "A") & (second.limit_name == "1%")]
    assert len(a1) == n_a1 and (a1.limit_value == 0.75).all()          # replaced, not duplicated
    for model, limit_name in (("A", "5%"), ("B", "1%")):               # untouched
        before = first[(first.model == model) & (first.limit_name == limit_name)].reset_index(drop=True)
        after = second[(second.model == model) & (second.limit_name == limit_name)].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)
    assert len(second) == len(first)


def test_write_scorecard_creates_parent_folder_and_leaves_overall_fault_empty(tmp_path):
    path = tmp_path / "new" / "scorecard.csv"
    evaluate.write_scorecard(_rows("A"), path)
    text = path.read_text().splitlines()
    assert text[0] == ",".join(evaluate.COLUMNS)
    assert any(",1%,0.5,,mean_fdr," in line for line in text)  # empty fault field on overall rows
    assert any(",far,0.0,4," in line for line in text)        # run counts are written as integers
