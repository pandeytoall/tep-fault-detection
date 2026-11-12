"""Phase 5 tests for src/report.py: the table builders on small hand-made frames (no results files, no data files)."""
import numpy as np
import pandas as pd
import pytest

from src import report


# --- formatting ----------------------------------------------------------------------------------------------------
def test_fmt_fdr_has_three_decimals_and_nan_is_a_dash():
    assert report.fmt_fdr(0.99432) == "0.994"
    assert report.fmt_fdr(0.0) == "0.000"
    assert report.fmt_fdr(float("nan")) == "n/a"


def test_fmt_share_is_a_whole_percent():
    assert report.fmt_share(0.174) == "17%"
    assert report.fmt_share(0.0) == "0%"
    assert report.fmt_share(1.0) == "100%"


def test_fmt_delay_shows_median_minutes_and_never_detected_share():
    assert report.fmt_delay(24.0, 0.0) == "24 min (0% never)"
    assert report.fmt_delay(1161.0, 0.832) == "1161 min (83% never)"
    assert report.fmt_delay(float("nan"), 1.0) == "none detected (100% never)"


def test_fmt_seconds_keeps_small_values_readable():
    assert report.fmt_seconds(0.678) == "0.7 s"
    assert report.fmt_seconds(903.878) == "904 s"
    assert report.fmt_seconds(float("nan")) == "n/a"


def test_md_table_is_a_pipe_table_with_header_and_rule():
    frame = pd.DataFrame({"Model": ["a", "b"], "FDR": ["0.1", "0.2"]})
    assert report.md_table(frame).splitlines() == ["| Model | FDR |", "|---|---|", "| a | 0.1 |", "| b | 0.2 |"]


# --- the scorecard: mean FDR with and without faults 3, 9, 15 --------------------------------------------------------
def _scorecard():
    """Two unsupervised models and one supervised one, faults 1, 2, 3, 9, 15 only, two limits for model 'm1'."""
    rows = []

    def add(model, track, limit, fault, metric, value):
        rows.append({"model": model, "track": track, "limit_name": limit, "fault": fault, "metric": metric, "value": value})

    fdrs = {"m1": {1: 0.9, 2: 0.5, 3: 0.02, 9: 0.01, 15: 0.03}, "m2": {1: 0.4, 2: 0.2, 3: 0.0, 9: 0.0, 15: 0.0},
            "s1": {1: 1.0, 2: 0.9, 3: 0.01, 9: 0.02, 15: 0.0}}
    tracks = {"m1": "unsupervised", "m2": "unsupervised", "s1": "supervised"}
    for model, by_fault in fdrs.items():
        for limit in ("1%", "5%"):
            scale = 1.0 if limit == "1%" else 0.5            # the 5% rows get different numbers, so a mix-up shows
            for fault, v in by_fault.items():
                add(model, tracks[model], limit, fault, "fdr", v * scale)
                add(model, tracks[model], limit, fault, "delay_median_min", 30.0 + fault)
                add(model, tracks[model], limit, fault, "never_detected_share", 0.1 * scale)
            vals = np.array(list(by_fault.values())) * scale
            add(model, tracks[model], limit, np.nan, "mean_fdr", vals.mean())
            add(model, tracks[model], limit, np.nan, "mean_fdr_excl_3_9_15", vals[:2].mean())
            add(model, tracks[model], limit, np.nan, "far", 0.01)
            add(model, tracks[model], limit, np.nan, "sustained_rate", 0.05)
            add(model, tracks[model], limit, np.nan, "train_time_s", 12.0)
            add(model, tracks[model], limit, np.nan, "peak_ram_gb", 2.5)
    return pd.DataFrame(rows)


def test_means_from_fdr_with_and_without_the_hard_faults():
    fdr = pd.Series({1: 0.9, 2: 0.5, 3: 0.02, 9: 0.01, 15: 0.03})
    with_all, without = report.means_from_fdr(fdr)
    assert with_all == pytest.approx((0.9 + 0.5 + 0.02 + 0.01 + 0.03) / 5)
    assert without == pytest.approx((0.9 + 0.5) / 2)


def test_model_row_reads_the_1pct_limit_only():
    row = report.model_row(_scorecard(), "m1", "1%")
    assert row["fdr"][1] == pytest.approx(0.9)            # the 5% row says 0.45
    assert row["mean_fdr"] == pytest.approx(0.292)
    assert row["mean_fdr_excl_3_9_15"] == pytest.approx(0.7)
    assert row["far"] == 0.01 and row["sustained_rate"] == 0.05
    assert row["train_time_s"] == 12.0 and row["peak_ram_gb"] == 2.5
    assert row["delay"][2] == 32.0 and row["never"][2] == pytest.approx(0.1)


def test_model_row_stops_when_the_stored_mean_disagrees_with_the_per_fault_values():
    card = _scorecard()
    bad = (card["model"] == "m1") & (card["limit_name"] == "1%") & (card["metric"] == "mean_fdr_excl_3_9_15")
    card.loc[bad, "value"] = 0.123
    with pytest.raises(ValueError, match="m1"):
        report.model_row(card, "m1", "1%")


def test_model_row_unknown_model_or_limit_is_an_error():
    with pytest.raises(KeyError):
        report.model_row(_scorecard(), "nope", "1%")
    with pytest.raises(KeyError):
        report.model_row(_scorecard(), "m1", "7%")


# --- the two tracks stay in separate tables ----------------------------------------------------------------------------
MODELS = {"m1": ("Model one", "unsupervised"), "m2": ("Model two", "unsupervised"), "s1": ("Supervised one", "supervised")}


def test_detection_tables_split_the_tracks_and_never_mix_them():
    tables = report.detection_tables(_scorecard(), MODELS, faults=(1, 2), hard=(3, 9, 15))
    assert set(tables) == {"unsupervised", "supervised"}
    assert list(tables["unsupervised"]["summary"]["Model"]) == ["Model one", "Model two"]
    assert list(tables["supervised"]["summary"]["Model"]) == ["Supervised one"]


def test_detection_tables_wrong_track_in_the_scorecard_is_an_error():
    bad = {"m1": ("Model one", "supervised")}                # the scorecard says m1 is unsupervised
    with pytest.raises(ValueError, match="m1"):
        report.detection_tables(_scorecard(), bad, faults=(1, 2), hard=(3, 9, 15))


def test_summary_table_columns_and_formatting():
    summary = report.detection_tables(_scorecard(), MODELS, faults=(1, 2), hard=(3, 9, 15))["unsupervised"]["summary"]
    assert list(summary.columns) == ["Model", "FAR", "3-in-a-row rate, fault-free", "Mean FDR, all 20 faults",
                                     "Mean FDR without 3, 9, 15", "Train time", "Peak RAM"]
    first = summary.iloc[0]
    assert first["FAR"] == "0.010" and first["3-in-a-row rate, fault-free"] == "5%"
    assert first["Mean FDR, all 20 faults"] == "0.292" and first["Mean FDR without 3, 9, 15"] == "0.700"
    assert first["Train time"] == "12 s" and first["Peak RAM"] == "2.5 GB"


def test_fdr_and_delay_tables_use_the_asked_faults_in_order():
    track = report.detection_tables(_scorecard(), MODELS, faults=(2, 1), hard=(15, 3))["unsupervised"]
    assert list(track["fdr"].columns) == ["Model", "F2", "F1", "F15", "F3"]
    assert list(track["fdr"].iloc[0]) == ["Model one", "0.500", "0.900", "0.030", "0.020"]
    assert list(track["delay"].columns) == ["Model", "F2", "F1", "F15", "F3"]
    assert track["delay"].iloc[0]["F2"] == "32 min (10% never)"


# --- against the literature --------------------------------------------------------------------------------------------
def _vs_lit():
    """vs_literature.csv layout: ours at 4 decimals, Rato & Reis at 3, diff = ours - rr."""
    return pd.DataFrame({
        "fault": [1, 2], "ours-pca-t2": [0.9, 0.5], "ours-pca-spe": [0.95, 0.6], "ours-dpca-t2": [0.9, 0.5], "ours-dpca-spe": [0.99, 0.7],
        "rr-pca-t2": [0.8, 0.5], "rr-pca-spe": [0.9, 0.7], "rr-dpca-t2": [0.9, 0.4], "rr-dpca-spe": [0.99, 0.9],
        "diff-pca-t2": [0.1, 0.0], "diff-pca-spe": [0.05, -0.1], "diff-dpca-t2": [0.0, 0.1], "diff-dpca-spe": [0.0, -0.2]})


def _lit_scorecard():
    rows = []
    for model, ours in {"pca-t2": [0.9, 0.5], "pca-spe": [0.95, 0.6], "dpca-t2": [0.9, 0.5], "dpca-spe": [0.99, 0.7]}.items():
        for fault, v in zip((1, 2), ours):
            rows.append({"model": model, "limit_name": "1%", "fault": fault, "metric": "fdr", "value": v + 0.00002})
            rows.append({"model": model, "limit_name": "5%", "fault": fault, "metric": "fdr", "value": 0.123})
    return pd.DataFrame(rows)


def test_literature_frame_joins_our_scorecard_fdr_with_the_published_numbers():
    frame = report.literature_frame(_vs_lit(), _lit_scorecard())
    assert list(frame.index) == [1, 2]
    assert frame.loc[1, "ours-pca-t2"] == pytest.approx(0.90002)     # full precision from the scorecard (1% rows)
    assert frame.loc[2, "rr-dpca-spe"] == 0.9
    assert frame.loc[2, "diff-dpca-spe"] == pytest.approx(0.70002 - 0.9)


def test_literature_frame_stops_when_the_saved_comparison_disagrees_with_the_scorecard():
    card = _lit_scorecard()
    card.loc[(card["model"] == "pca-t2") & (card["fault"] == 1) & (card["limit_name"] == "1%"), "value"] = 0.8
    with pytest.raises(ValueError, match="pca-t2"):
        report.literature_frame(_vs_lit(), card)


def test_literature_table_has_ours_and_published_per_statistic_and_a_mean_row():
    frame = report.literature_frame(_vs_lit(), _lit_scorecard())
    table = report.literature_table(frame, hard=(2,))
    assert list(table.columns)[:3] == ["Fault", "PCA T² ours", "PCA T² R&R"]
    assert len(table.columns) == 9
    assert list(table["Fault"]) == ["1", "2", "Mean, 2 faults", "Mean without 2"]
    assert table.loc[0, "PCA T² ours"] == "0.900" and table.loc[0, "PCA T² R&R"] == "0.800"
    assert table.loc[2, "PCA T² ours"] == "0.700" and table.loc[2, "PCA T² R&R"] == "0.650"
    assert table.loc[3, "PCA T² ours"] == "0.900" and table.loc[3, "PCA T² R&R"] == "0.800"


def test_literature_gaps_lists_faults_where_we_differ_by_at_least_the_threshold():
    frame = report.literature_frame(_vs_lit(), _lit_scorecard())
    gaps = report.literature_gaps(frame, threshold=0.15)
    assert [(g["fault"], g["model"]) for g in gaps] == [(2, "dpca-spe")]
    assert gaps[0]["ours"] == pytest.approx(0.70002) and gaps[0]["rr"] == 0.9


# --- the unseen-fault test ---------------------------------------------------------------------------------------------
def _unseen():
    return pd.DataFrame({"fault": [1, 16, 14], "hgb-l2-unseen": [0.99, 0.5, 0.92], "hgb-l2": [1.0, 0.85, 1.0],
                         "pca-spe": [0.98, 0.25, 0.92], "machine": "laptop"})


def test_unseen_summary_says_how_much_of_the_gap_one_fault_carries():
    s = report.unseen_summary(_unseen(), carrier=16)
    assert s["mean"]["hgb-l2-unseen"] == pytest.approx((0.99 + 0.5 + 0.92) / 3)
    assert s["mean"]["pca-spe"] == pytest.approx((0.98 + 0.25 + 0.92) / 3)
    assert s["gap_total"] == pytest.approx(0.01 + 0.25 + 0.0)           # unseen minus PCA SPE, summed over faults
    assert s["carrier_share"] == pytest.approx(0.25 / 0.26)
    assert s["mean_without_carrier"]["hgb-l2-unseen"] == pytest.approx((0.99 + 0.92) / 2)
    assert s["max_gap_others"] == pytest.approx(0.01)


def test_unseen_table_has_a_mean_row_and_the_gap_column():
    table = report.unseen_table(_unseen())
    assert list(table.columns) == ["Fault", "Supervised, fault unseen", "Supervised, all faults", "PCA SPE (no labels)",
                                   "Unseen minus PCA SPE"]
    assert list(table["Fault"]) == ["1", "14", "16", "Mean, 3 faults"]
    assert list(table.iloc[2])[1:] == ["0.500", "0.850", "0.250", "+0.250"]
    assert table.iloc[3]["Unseen minus PCA SPE"] == "+0.087"


# --- diagnosis: blame by method ----------------------------------------------------------------------------------------
def _blame_row(fault, method, window, unit, top1, top5, mass, base_share=0.2, base5=0.6, group="the rest"):
    return {"fault": fault, "group": group, "method": method, "window": window, "expected_unit": unit, "top1_hit": top1,
            "top5_hit": top5, "unit_mass": mass, "baseline_unit_share": base_share, "baseline_top5_hit": base5}


def _blame():
    rows = []
    for fault, unit, spe, t2 in ((1, "feeds", (1.0, 1.0, 0.4), (0.0, 0.5, 0.2)), (2, "feeds", (0.0, 0.0, 0.1), (0.2, 0.3, 0.3)),
                                 (4, "reactor", (0.5, 1.0, 0.7), (0.5, 0.5, 0.3))):
        rows.append(_blame_row(fault, "spe", "post", unit, *spe))
        rows.append(_blame_row(fault, "t2", "post", unit, *t2))
        rows.append(_blame_row(fault, "spe", "first_hour", unit, 0.0, 0.0, 0.0))   # must not enter the post-window means
    rows.append(_blame_row(3, "spe", "post", "feeds", 0.9, 0.9, 0.9, group="hidden"))     # hidden faults are left out
    rows.append(_blame_row(16, "spe", "post", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, group="unknown cause"))
    return pd.DataFrame(rows)


def test_blame_summary_is_the_post_window_mean_over_detectable_faults_with_a_known_unit():
    s = report.blame_summary(_blame()).set_index("method")
    assert s.loc["spe", "top1_hit"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert s.loc["spe", "top5_hit"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert s.loc["spe", "unit_mass"] == pytest.approx((0.4 + 0.1 + 0.7) / 3)
    assert s.loc["t2", "top5_hit"] == pytest.approx((0.5 + 0.3 + 0.5) / 3)
    assert s.loc["spe", "random_top1_and_mass"] == pytest.approx(0.2) and s.loc["spe", "random_top5"] == pytest.approx(0.6)
    assert s.loc["spe", "n_faults"] == 3


def test_blame_summary_can_take_a_separate_frame_for_permutation_importance():
    dev = _blame().assign(top1_hit=0.0, top5_hit=0.0)                 # a dev frame: its spe/t2 rows must not leak into the test means
    dev.loc[dev["method"] == "t2", "method"] = "perm"
    dev.loc[dev["method"] == "spe", "top1_hit"] = 0.77
    s = report.blame_summary(_blame(), dev).set_index("method")
    assert list(s.index) == ["spe", "t2", "perm"]
    assert s.loc["spe", "top1_hit"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)       # test runs only
    assert s.loc["perm", "top1_hit"] == 0.0                                      # perm rows come from the dev frame


def test_blame_misses_lists_detectable_faults_whose_top5_is_at_or_below_the_cutoff():
    assert report.blame_misses(_blame(), cutoff=0.05) == [2]


def test_window_contrast_gives_first_hour_and_whole_window_for_one_fault():
    frame = pd.DataFrame([_blame_row(6, "spe", "post", "feeds", 0.22, 1.0, 0.17), _blame_row(6, "spe", "first_hour", "feeds", 1.0, 1.0, 0.98),
                          _blame_row(6, "t2", "post", "feeds", 0.0, 0.0, 0.0)])
    c = report.window_contrast(frame, fault=6, method="spe")
    assert c == {"first_hour_top1": 1.0, "post_top1": 0.22, "first_hour_top5": 1.0, "post_top5": 1.0}


# --- diagnosis: why 3, 9, 15 hide ---------------------------------------------------------------------------------------
def _hidden():
    return pd.DataFrame({
        "fault": [0, 3, 14, 6], "group": ["control", "hidden", "the rest", "benchmark"],
        "median_max_abs_shift": [0.47, 0.46, 0.47, 80.0], "median_max_spread_ratio": [2.1, 2.0, 14.0, 60.0],
        "median_max_abs_dacf": [0.25, 0.24, 1.2, 1.1],
        "verdict": ["control (the noise floor)", "looks like the fault-free control", "looks like the fault-free control",
                    "clear shift, well above the control"]})


def test_hidden_table_shows_the_control_first_and_the_three_statistics():
    table = report.hidden_table(_hidden(), faults=(0, 3, 14))
    assert list(table.columns) == ["Fault", "Largest shift (sd)", "Largest spread ratio", "Largest change in autocorrelation"]
    assert list(table["Fault"]) == ["Fault-free control", "3", "14"]
    assert list(table.iloc[1])[1:] == ["0.46", "2.00", "0.24"]
    assert list(table.iloc[2])[1:] == ["0.47", "14.00", "1.20"]


def test_looks_like_control_on_shift_lists_faults_with_that_verdict():
    assert report.looks_like_control(_hidden()) == [3, 14]


# --- the assistant ------------------------------------------------------------------------------------------------------
def _comparison():
    return pd.DataFrame({
        "metric": ["headline", "idv_top1_all", "idv_top3_all", "normal", "hard", "baseline", "chance"],
        "v1_k": [27, 25, 35, 10, 9, 15, 18], "v1_n": [36, 60, 60, 10, 9, 36, 36],
        "v1.1_k": [36, 51, 51, 10, 9, 15, 18], "v1.1_n": [36, 60, 60, 10, 9, 36, 36],
        "wrong_to_right": [9.0, 26.0, 16.0, 0.0, 0.0, np.nan, np.nan], "right_to_wrong": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan]})


def test_assistant_table_has_counts_wilson_intervals_and_flips():
    table = report.assistant_table(_comparison())
    assert list(table.columns) == ["Measure", "v1 (PCA tools only)", "v1.1 (adds a trained classifier)", "Cases fixed / broken"]
    head = table.iloc[0]
    assert head["v1 (PCA tools only)"] == "27 / 36 = 0.75 [0.59, 0.86]"
    assert head["v1.1 (adds a trained classifier)"].startswith("36 / 36 = 1.00 [0.90, 1.00]")
    assert head["Cases fixed / broken"] == "9 / 0"


def test_assistant_table_baseline_and_chance_have_no_interval_and_no_flips():
    table = report.assistant_table(_comparison()).set_index("Measure")
    base = [i for i in table.index if i.startswith("No-LLM baseline")][0]
    chance = [i for i in table.index if i.startswith("Chance")][0]
    assert table.loc[base, "v1 (PCA tools only)"] == "15 / 36 = 0.42"
    assert table.loc[chance, "Cases fixed / broken"] == ""


def test_assistant_table_marks_the_hard_faults_row_as_not_a_detection():
    table = report.assistant_table(_comparison())
    hard = [m for m in table["Measure"] if "3, 9, 15" in m]
    assert len(hard) == 1 and "not a detection" in hard[0]


def test_repeat_agreement_counts_repeated_cases_that_agree_on_status_and_unit():
    diag = pd.DataFrame({
        "case_id": ["a", "a", "a", "b", "b", "c"], "repeat": [1, 2, 3, 1, 2, 1], "parsed_ok": True,
        "status": ["fault", "fault", "fault", "fault", "normal", "normal"], "unit": ["feeds", "feeds", "feeds", "feeds", "unknown", "unknown"]})
    assert report.repeat_agreement(diag) == (1, 2)


def test_session_cost_sums_the_result_lines_and_is_none_without_transcripts(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "a.jsonl").write_text('{"type": "system"}\n{"type": "result", "total_cost_usd": 0.25}\n', encoding="utf8")
    (tmp_path / "t" / "b.jsonl").write_text('{"type": "result", "total_cost_usd": 0.5}\n', encoding="utf8")
    assert report.session_cost(tmp_path / "t") == (pytest.approx(0.75), 2)
    assert report.session_cost(tmp_path / "missing") is None


# --- links --------------------------------------------------------------------------------------------------------------
def test_broken_links_finds_missing_files_and_ignores_urls_and_anchors(tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "a.png").write_bytes(b"x")
    (tmp_path / "other.md").write_text("x")
    text = ("![chart](figures/a.png) and [other](other.md#part) and [web](https://example.org/x) and [top](#intro)\n"
            "[gone](missing.csv) and ![nothing](figures/none.png)")
    assert report.broken_links(text, tmp_path) == ["missing.csv", "figures/none.png"]


def test_broken_links_resolves_parent_folders_from_the_base_directory(tmp_path):
    (tmp_path / "phase-2").mkdir()
    (tmp_path / "phase-2" / "x.csv").write_text("x")
    (tmp_path / "report").mkdir()
    assert report.broken_links("[x](../phase-2/x.csv)", tmp_path / "report") == []


def test_detectable_faults_are_those_with_a_known_unit_outside_the_hidden_group():
    assert report.detectable_faults(_blame()) == [1, 2, 4]
