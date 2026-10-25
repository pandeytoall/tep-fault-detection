"""Phase 2b tests for src/cases.py: the frozen test case set, the dev case set and their answer keys."""
import numpy as np
import pandas as pd
import pytest

from src import cases, data, split

# From the spec (IDV -> scoring unit), typed out independently of the module under test.
EXPECTED_UNIT = {**{f: "feeds" for f in (1, 2, 3, 6, 7, 8, 9, 10)}, **{f: "reactor" for f in (4, 11, 13, 14)},
                 **{f: "condenser/separator" for f in (5, 12, 15)}, **{f: "unknown" for f in range(16, 21)}, 0: "none"}


def test_idv_to_unit_table_matches_the_spec():
    assert {f: cases.case_unit(f) for f in range(0, 21)} == EXPECTED_UNIT


@pytest.fixture(scope="module")
def test_key():
    return cases.build_test_key()


@pytest.fixture(scope="module")
def dev_key():
    return cases.build_dev_key(split.ids("dev", split.make_split()))


def test_test_set_has_three_runs_per_fault_and_ten_normal(test_key):
    assert len(test_key) == 70
    counts = test_key["fault"].value_counts()
    assert all(counts[f] == 3 for f in range(1, 21)) and counts[0] == 10


def test_test_set_uses_70_distinct_run_ids_from_1_to_500(test_key):
    assert test_key["run"].nunique() == 70
    assert test_key["run"].between(1, 500).all()


def test_test_set_crops_are_in_0_to_80_and_spread(test_key):
    assert test_key["crop"].between(0, 80).all()
    assert test_key["crop"].max() > 60 and test_key["crop"].min() < 20  # uniform, not constant


def test_every_test_case_has_the_same_length_so_length_says_nothing_about_onset(test_key):
    assert (test_key["crop"] + test_key["end_crop"] == 80).all()
    assert test_key["end_crop"].between(0, 80).all()
    assert (960 - test_key["crop"] - test_key["end_crop"]).nunique() == 1
    assert (960 - test_key["crop"] - test_key["end_crop"]).iloc[0] == 880      # 44 h
    faulty = test_key[test_key["fault"] > 0]
    assert faulty["onset_h"].nunique() > 20                                    # onsets still differ


def test_every_dev_case_has_the_same_length(dev_key):
    assert (dev_key["crop"] + dev_key["end_crop"] == 10).all()
    assert (500 - dev_key["crop"] - dev_key["end_crop"]).unique().tolist() == [490]
    assert dev_key.loc[dev_key["fault"] > 0, "onset_h"].nunique() > 5


def test_test_set_ids_are_case_01_to_case_70_in_a_shuffled_order(test_key):
    assert list(test_key["case_id"]) == [f"case_{i:02d}" for i in range(1, 71)]
    faults = list(test_key["fault"])
    assert faults != sorted(faults)                      # not sorted by fault
    assert len({tuple(faults[i:i + 3]) for i in range(0, 60, 3)}) > 10  # not blocks of three


def test_test_set_is_a_pure_function_of_the_seed(test_key):
    pd.testing.assert_frame_equal(test_key, cases.build_test_key())
    assert not cases.build_test_key(seed=7)["run"].equals(test_key["run"])


def test_test_set_is_the_testing_family_with_onset_in_historian_hours(test_key):
    assert set(test_key["family"]) == {"testing"} and set(test_key["set"]) == {"test"}
    faulty = test_key[test_key["fault"] > 0]
    # first faulty sample is 161 - crop, the clock starts at 0 at sample crop + 1, 3 minutes per sample
    assert np.allclose(faulty["onset_h"], (160 - faulty["crop"]) * 0.05)
    assert faulty["onset_h"].between(4, 8).all()
    assert test_key.loc[test_key["fault"] == 0, "onset_h"].isna().all()


def test_answer_key_carries_the_scoring_unit(test_key):
    assert all(row.unit == EXPECTED_UNIT[row.fault] for row in test_key.itertuples())
    assert list(test_key.columns) == ["case_id", "set", "family", "fault", "run", "crop", "end_crop", "unit", "onset_h"]


def test_dev_set_uses_each_dev_run_id_once(dev_key):
    dev_ids = split.ids("dev", split.make_split())
    assert len(dev_key) == 20 and sorted(dev_key["run"]) == sorted(dev_ids)


def test_dev_set_has_16_different_faults_and_4_normal(dev_key):
    faulty = dev_key[dev_key["fault"] > 0]
    assert len(faulty) == 16 and faulty["fault"].nunique() == 16 and faulty["fault"].between(1, 20).all()
    assert (dev_key["fault"] == 0).sum() == 4


def test_dev_set_is_the_training_family_with_short_crops(dev_key):
    assert set(dev_key["family"]) == {"training"} and set(dev_key["set"]) == {"dev"}
    assert dev_key["crop"].between(0, 10).all() and dev_key["end_crop"].between(0, 10).all()
    assert list(dev_key["case_id"]) == [f"dev_{i:02d}" for i in range(1, 21)]
    faulty = dev_key[dev_key["fault"] > 0]
    assert np.allclose(faulty["onset_h"], (20 - faulty["crop"]) * 0.05) and faulty["onset_h"].between(0.5, 1).all()


def test_dev_set_does_not_share_run_ids_with_the_test_set_logic():
    """Dev IDs come only from the argument: with other IDs the key follows them."""
    key = cases.build_dev_key(list(range(101, 121)))
    assert sorted(key["run"]) == list(range(101, 121))


# --- cropping and the historian clock -----------------------------------------------------
def _run_frame(n=30):
    frame = pd.DataFrame({"sample": np.arange(1, n + 1)})
    for v in data.VARIABLES:
        frame[v] = frame["sample"].astype(float)
    return frame


def test_crop_drops_the_first_samples_and_restarts_the_clock_at_zero():
    out = cases.crop_run(_run_frame(30), crop=3)
    assert len(out) == 27 and out["xmeas_1"].iloc[0] == 4.0   # sample 4 is the first kept
    assert out["t_hours"].iloc[0] == 0.0 and out["t_hours"].iloc[1] == 0.05
    assert out["t_hours"].iloc[-1] == pytest.approx(26 * 0.05)
    assert list(out.columns) == ["t_hours"] + data.VARIABLES


def test_end_crop_drops_the_last_samples_and_leaves_the_clock_alone():
    out = cases.crop_run(_run_frame(30), crop=3, end_crop=4)
    assert len(out) == 23 and out["xmeas_1"].iloc[0] == 4.0 and out["xmeas_1"].iloc[-1] == 26.0
    assert out["t_hours"].iloc[0] == 0.0 and out["t_hours"].iloc[-1] == pytest.approx(22 * 0.05)


def test_the_first_faulty_sample_sits_at_the_key_onset_hour():
    onset, crop = 20, 5
    out = cases.crop_run(_run_frame(60), crop)
    first_faulty_t = out.loc[out["xmeas_1"] == onset + 1, "t_hours"].iloc[0]
    assert first_faulty_t == pytest.approx(cases.onset_hours(onset, crop)) == pytest.approx(0.75)


# --- the test set is frozen --------------------------------------------------------------------
def test_the_test_set_refuses_to_rebuild(tmp_path):
    existing = tmp_path / "answer_key_test.csv"
    existing.write_text("x")
    with pytest.raises(FileExistsError, match="frozen"):
        cases.guard_frozen("test", [existing])


def test_the_test_set_builds_when_nothing_exists_and_dev_may_be_rebuilt(tmp_path):
    cases.guard_frozen("test", [tmp_path / "missing.csv"])
    existing = tmp_path / "answer_key_dev.csv"
    existing.write_text("x")
    cases.guard_frozen("dev", [existing])  # no error


def test_write_answer_key_adds_the_machine_and_refuses_to_overwrite_the_test_key(tmp_path, test_key):
    path = tmp_path / "answer_key_test.csv"
    cases.write_answer_key(test_key, path, machine="m1")
    again = pd.read_csv(path)
    assert (again["machine"] == "m1").all() and len(again) == 70
    with pytest.raises(FileExistsError):
        cases.write_answer_key(test_key, path, machine="m1")


# --- loading case runs from parquet --------------------------------------------------------------
def _write_parquet(path, runs_by_fault, n=40):
    frames = []
    for fault, runs in runs_by_fault.items():
        for run in runs:
            f = _run_frame(n)
            f.insert(0, "simulationRun", np.int16(run))
            f.insert(0, "faultNumber", np.int16(fault))
            for v in data.VARIABLES:
                f[v] = (f[v] + 1000 * fault + 10_000 * run).astype(np.float32)  # identifies the source run
            frames.append(f)
    pd.concat(frames)[data.ID_COLUMNS + data.VARIABLES].astype({"sample": "int16"}).to_parquet(path)


def test_load_case_frames_picks_each_case_run_from_the_right_file_and_crops_it(tmp_path):
    _write_parquet(tmp_path / "TEP_Faulty_Training.parquet", {2: [5, 6], 7: [5, 8]})
    _write_parquet(tmp_path / "TEP_FaultFree_Training.parquet", {0: [5, 6, 8]})
    key = pd.DataFrame({"case_id": ["dev_01", "dev_02", "dev_03"], "set": "dev", "family": "training",
                        "fault": [2, 7, 0], "run": [6, 8, 8], "crop": [0, 4, 2],
                        "end_crop": [0, 5, 1]})
    out = cases.load_case_frames(key, tmp_path)
    assert list(out) == ["dev_01", "dev_02", "dev_03"]
    assert len(out["dev_01"]) == 40 and out["dev_01"]["xmeas_1"].iloc[0] == 1 + 2000 + 60_000
    assert len(out["dev_02"]) == 31 and out["dev_02"]["xmeas_1"].iloc[0] == 5 + 7000 + 80_000
    assert len(out["dev_03"]) == 37 and out["dev_03"]["xmeas_1"].iloc[0] == 3 + 0 + 80_000
    assert out["dev_03"]["t_hours"].iloc[0] == 0.0


# --- build-time sanity checks (acceptance criteria) --------------------------------------------------
def test_sanity_checks_on_a_small_historian(world):
    from src import historian, mspc

    key = pd.DataFrame({"case_id": ["dev_01", "dev_02", "dev_03"], "set": "dev", "family": "training",
                        "fault": [6, 1, 0], "run": [1, 2, 3], "crop": [0, 0, 0], "unit": ["feeds", "feeds", "none"],
                        "onset_h": [0.5, 0.9, np.nan]})
    out = cases.sanity_checks(key, world["db"], world["npz"])
    monitor = mspc.PCAMonitor.load(world["npz"])
    scores = historian.score_case(monitor, historian.CaseView(world["db"], "dev_03").matrix())
    for stat in ("t2", "spe"):
        expected = float((scores[stat] > monitor.limits[stat]["1%"]).mean())
        assert out["normal_share_above"][stat] == pytest.approx(expected)
    assert out["normal_share_above"]["n_cases"] == 1
    assert out["fault6"]["dev_01"]["a_feed_in_top5"] is True          # the step is on xmeas_1, the A feed
    assert "xmeas_1" in out["fault6"]["dev_01"]["top5"]
    assert out["onset_in_range"] is True


def test_sanity_checks_flag_an_onset_outside_the_set_range(world):
    key = pd.DataFrame({"case_id": ["dev_01"], "set": "dev", "family": "training", "fault": [6], "run": [1],
                        "crop": [0], "unit": ["feeds"], "onset_h": [1.5]})
    assert cases.sanity_checks(key, world["db"], world["npz"])["onset_in_range"] is False


def test_repeat_subset_is_ten_distinct_test_cases_chosen_by_the_seed(test_key):
    chosen = cases.repeat_subset(test_key)
    assert len(chosen) == len(set(chosen)) == 10 and set(chosen) <= set(test_key["case_id"])
    assert chosen == cases.repeat_subset(test_key) == sorted(chosen)
