"""Phase 5 tests for src/operator_report.py: the facts about one run and the operator text, on hand-made arrays."""
import numpy as np
import pandas as pd
import pytest

from src import mspc, operator_report as op

TAGS = ["xmeas_1", "xmeas_7", "xmv_5"]  # feeds, reactor, compressor/recycle


def test_run_clock_starts_at_zero_and_steps_three_minutes():
    assert op.run_clock(1) == 0.0
    assert op.run_clock(161) == pytest.approx(8.0)
    assert op.run_clock(np.array([1, 21])) == pytest.approx([0.0, 1.0])


def test_fault_start_is_the_run_clock_time_of_the_first_faulty_sample():
    assert op.fault_start_h(160) == pytest.approx(8.0)       # sample 161 is the first faulty one, stamped 8.0 h
    assert op.fault_start_h(20) == pytest.approx(1.0)


def test_first_alarm_is_the_third_of_three_samples_in_a_row_above_the_limit():
    samples = np.arange(1, 13)
    scores = np.array([0, 0, 9, 0, 0, 0, 9, 9, 9, 9, 0, 0], dtype=float)    # a stray alarm at 3, then 7-10
    alarm = op.first_alarm(samples, scores, limit=5.0)
    assert alarm == {"sample": 9, "t_hours": pytest.approx(0.4)}


def test_first_alarm_is_none_when_there_is_no_three_in_a_row():
    assert op.first_alarm(np.arange(1, 7), np.array([9, 9, 0, 9, 9, 0], dtype=float), limit=5.0) is None


def test_hours_after_gives_the_difference_on_the_run_clock():
    assert op.hours_after(8.1, 8.0) == pytest.approx(0.1)


def test_first_departure_needs_three_samples_in_a_row_beyond_the_band_after_the_start():
    z = pd.Series([0, 5, 0, 0, 4, 4, 0, 4, 4, 4, 4], index=np.arange(1, 12), dtype=float)
    assert op.first_departure(z, after_sample=0) == 8            # samples 8-10 are the first three in a row
    assert op.first_departure(z, after_sample=8) == 9
    assert op.first_departure(z, after_sample=9) is None


def test_first_departure_counts_negative_departures_too():
    z = pd.Series([0, -4, -4, -4, 0], index=np.arange(1, 6), dtype=float)
    assert op.first_departure(z, after_sample=0) == 2


def test_pick_downstream_takes_the_top_readings_outside_the_feeds_that_really_move():
    top = pd.DataFrame({"tag": ["xmeas_1", "xmv_11", "xmeas_17", "xmv_3", "xmv_5"],
                        "scoring_unit": ["feeds", "condenser/separator", "stripper", "feeds", "compressor/recycle"]})
    departures = {"xmeas_1": 3, "xmv_11": 99, "xmeas_17": None, "xmv_3": 3, "xmv_5": 9}
    assert op.pick_downstream(top, departures, n=2) == ["xmv_11", "xmv_5"]
    assert op.pick_downstream(top, departures, n=1) == ["xmv_11"]


def test_frozen_from_is_the_last_sample_at_which_the_reading_changed():
    values = pd.Series([1.0, 2.0, 3.0, 3.0, 3.0], index=np.arange(1, 6))
    assert op.frozen_from(values) == 3
    assert op.frozen_from(pd.Series([1.0, 2.0, 3.0], index=np.arange(1, 4))) is None   # still moving at the end


def _toy_monitor():
    s = np.sqrt(2)
    m = mspc.PCAMonitor(lags=0, cpv=0.9, variables=TAGS)
    m.mean, m.std = np.zeros(3), np.ones(3)
    m.loadings = np.array([[1 / s, 0, 1 / s], [1 / s, 0, -1 / s], [0, 1, 0]])
    m.eigvals = np.array([4.0, 1.0, 0.25])
    m.n_pc, m.n_rows = 2, 1000
    return m


def test_alarm_windows_split_the_first_hour_after_the_alarm_from_the_whole_window():
    # a step on xmeas_1 leaves a residual on xmeas_1 and xmeas_7 (the first PC ties them); it doubles after 1.5 h
    t = np.arange(60) * 0.05
    frame = pd.DataFrame({"t_hours": t, "xmeas_1": 0.0, "xmeas_7": 0.0, "xmv_5": 0.0})
    frame.loc[t >= 0.5, "xmeas_1"] = 3.0
    frame.loc[t >= 1.55, "xmeas_1"] = 6.0
    reference = {"spe": np.array([1.0, 1.0, 1.0]), "t2": np.array([1.0, 1.0, 1.0])}
    windows = op.alarm_windows(_toy_monitor(), reference, frame, alarm_h=0.5, n=2)
    assert set(windows) == {"first_hour", "whole"}
    assert windows["first_hour"][0]["tag"] in ("xmeas_1", "xmeas_7") and len(windows["whole"]) == 2
    assert windows["first_hour"][0]["ratio"] == pytest.approx(2.2, abs=0.1)         # (1.5**2 = 2.25) over a reference of 1
    assert windows["whole"][0]["ratio"] > windows["first_hour"][0]["ratio"]        # the later, bigger step is in the whole window only


# --- compute_facts on a synthetic run (52 readings, a fitted monitor from the shared `world` fixture) -------------------------
def _fault_run(freeze_from=None):
    """960 normal-looking samples of all 52 readings; after sample 160 the A feed (xmeas_1) drops by a lot, the compressor recycle
    valve (xmv_5) shifts at once and the condenser cooling water valve (xmv_11) ramps up later. xmeas_17 never moves."""
    from src import data
    rng = np.random.default_rng(5)
    scale, base = np.linspace(0.5, 2, 52), np.linspace(10, 20, 52)
    values = rng.normal(size=(960, 52)) * scale + base
    cols = {t: i for i, t in enumerate(data.VARIABLES)}
    values[160:, cols["xmeas_1"]] -= 30.0
    values[166:, cols["xmv_5"]] += 40.0
    values[250:, cols["xmv_11"]] += np.linspace(0, 80, 710)
    if freeze_from is not None:
        values[freeze_from - 1:, cols["xmeas_7"]] = values[freeze_from - 1, cols["xmeas_7"]]
    frame = pd.DataFrame(values, columns=data.VARIABLES)
    frame.insert(0, "sample", np.arange(1, 961))
    frame.insert(0, "simulationRun", 1)
    frame.insert(0, "faultNumber", 6)
    return frame


def _top_readings():
    return pd.DataFrame({"tag": ["xmeas_1", "xmv_11", "xmeas_17", "xmv_3", "xmv_5"],
                         "scoring_unit": ["feeds", "condenser/separator", "stripper", "feeds", "compressor/recycle"]})


def _blame_for_facts():
    rows = []
    for fault, top in ((6, "A feed flow (xmv_3); A feed (xmeas_1); Compressor work (xmeas_20)"), (7, "A and C feed (xmeas_4); A feed (xmeas_1)"),
                       (2, "Condenser cooling water flow (xmv_11)")):
        rows.append({"fault": fault, "method": "spe", "window": "first_hour", "top1_hit": 1.0 if fault == 6 else 0.5, "top5_hit": 1.0, "top3": top})
        rows.append({"fault": fault, "method": "spe", "window": "post", "top1_hit": 0.22 if fault == 6 else 0.5, "top5_hit": 1.0, "top3": top})
    return pd.DataFrame(rows)


def _scorecard_for_facts():
    rows = [{"model": "pca-spe", "track": "unsupervised", "limit_name": "1%", "fault": 6, "metric": m, "value": v}
            for m, v in (("fdr", 1.0), ("delay_median_min", 9.0), ("never_detected_share", 0.0))]
    rows += [{"model": "pca-spe", "track": "unsupervised", "limit_name": "1%", "fault": np.nan, "metric": m, "value": v}
             for m, v in (("mean_fdr", 1.0), ("mean_fdr_excl_3_9_15", 1.0), ("far", 0.01), ("sustained_rate", 0.05), ("train_time_s", 1.0), ("peak_ram_gb", 2.0))]
    return pd.DataFrame(rows)


def test_compute_facts_alarm_time_delay_and_the_readings_that_move_later(world):
    from src import contributions
    monitor = world["monitor"]
    reference = contributions.load_reference(world["npz"])
    facts = op.compute_facts(_fault_run(), monitor, reference, run_id=1, fault=6, onset=160, top_readings=_top_readings(),
                             test_blame=_blame_for_facts(), scorecard=_scorecard_for_facts())
    assert facts["start_h"] == pytest.approx(8.0)
    assert facts["alarm"]["sample"] == 163 and facts["alarm"]["t_hours"] == pytest.approx(8.1)
    assert facts["delay_h"] == pytest.approx(0.1) and facts["delay_min_scorecard_rule"] == 9
    assert facts["cleared"] is False and facts["share_above_after"] == pytest.approx(1.0)
    assert facts["downstream"] == ["xmv_11", "xmv_5"]                       # xmeas_17 never moves, so it is skipped
    assert [r["tag"] for r in facts["skipped"]] == ["xmeas_17"]
    by_tag = {r["tag"]: r for r in facts["readings"]}
    assert by_tag["xmeas_1"]["departure_after_h"] == pytest.approx(0.0)
    assert by_tag["xmv_5"]["departure_after_h"] == pytest.approx(0.3)       # sample 167 is the start of 3 in a row beyond 3 sd
    assert by_tag["xmv_11"]["departure_after_h"] > by_tag["xmv_5"]["departure_after_h"]
    assert facts["windows"]["first_hour"][0]["tag"] == "xmeas_1"
    assert facts["phase4"]["faults_with_a_feed_in_top3"] == [7]
    assert facts["phase4"]["first_hour_top1"] == 1.0 and facts["phase4"]["post_top1"] == 0.22
    assert facts["scorecard"] == {"median_delay_min": 9.0, "never_share": 0.0}
    assert facts["freeze"] is None


def test_compute_facts_reports_where_the_measurements_stop_moving(world):
    from src import contributions
    reference = contributions.load_reference(world["npz"])
    facts = op.compute_facts(_fault_run(freeze_from=277), world["monitor"], reference, run_id=1, fault=6, onset=160,
                             top_readings=_top_readings(), test_blame=_blame_for_facts(), scorecard=_scorecard_for_facts())
    assert facts["freeze"]["sample"] == 277 and facts["freeze"]["t_hours"] == pytest.approx(13.8)
    assert facts["freeze"]["n_frozen"] == 1
    assert facts["freeze"]["share_of_window"] == pytest.approx((960 - 277) / (960 - 163))


def _facts(world, freeze_from=None):
    from src import contributions
    return op.compute_facts(_fault_run(freeze_from), world["monitor"], contributions.load_reference(world["npz"]), run_id=1, fault=6,
                            onset=160, top_readings=_top_readings(), test_blame=_blame_for_facts(), scorecard=_scorecard_for_facts())


def test_operator_markdown_has_the_five_sections_in_order_and_the_alarm_numbers(world):
    text = op.operator_markdown(_facts(world), "how it was made")
    order = ["## Situation", "## What the readings show", "## Where to look first", "## What this does and doesn't tell you", "## Evidence",
             "## How this was made"]
    positions = [text.index(h) for h in order]
    assert positions == sorted(positions)
    assert "**8.1 h**" in text and "**8.0 h**" in text and "6 minutes" in text
    assert "how it was made" in text.split("## How this was made")[1]
    assert "figures/operator-fault6.png" in text and "figures/operator-fault6-readings.png" in text


def test_operator_markdown_names_the_three_things_it_cannot_say(world):
    text = op.operator_markdown(_facts(world, freeze_from=277), "x")
    cannot = text.split("Three things we cannot say:")[1].split("- This is one run")[0]
    assert cannot.count("**") == 6                       # three bold headings
    assert "13.8 h" in cannot and "faults 7" in cannot


def test_operator_markdown_skips_the_freeze_sentences_when_nothing_freezes(world):
    text = op.operator_markdown(_facts(world), "x")
    assert "stay at one fixed value" not in text


def test_title_finding_states_minutes_and_that_the_alarm_never_cleared(world):
    assert op.title_finding(_facts(world)) == "PCA SPE crossed its 1% limit 6 minutes after the A feed was lost, and never came back below it"
