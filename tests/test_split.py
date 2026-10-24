"""Phase 1 tests for src/split.py: the one seed-42 partition of run IDs 1-500."""
import json

import numpy as np
import pandas as pd
import pytest

from src import data, split

GROUPS = ("fit", "limit", "supervised", "dev")
SIZES = {"fit": 330, "limit": 100, "supervised": 50, "dev": 20}


def test_group_sizes_disjoint_and_cover_all_ids():
    s = split.make_split()
    for g, n in SIZES.items():
        assert len(split.ids(g, s)) == n
    seen = np.concatenate([split.ids(g, s) for g in GROUPS])
    assert len(seen) == 500 and len(set(seen)) == 500  # pairwise disjoint
    assert sorted(seen) == list(range(1, 501))          # union = 1..500


def test_limit_tune_is_a_50_subset_of_limit():
    s = split.make_split()
    tune = set(split.ids("limit_tune", s))
    assert len(tune) == 50
    assert tune < set(split.ids("limit", s))


def test_ids_are_sorted_python_ints():
    for g in (*GROUPS, "limit_tune"):
        got = split.ids(g)
        assert got == sorted(got) and all(type(i) is int for i in got)


def test_make_split_is_deterministic_and_pinned():
    assert split.make_split() == split.make_split()
    assert split.SEED == 42
    # pinned literal: a different seed or draw order would change the dev runs of every later phase
    assert split.ids("dev") == [15, 23, 37, 42, 98, 114, 137, 142, 164, 195, 238, 276, 304, 321, 367, 397, 438,
                                450, 462, 492]


def test_unknown_group_raises():
    with pytest.raises(KeyError):
        split.ids("test")


def test_json_roundtrip_stores_seed_sizes_and_ids(tmp_path):
    path = tmp_path / "sub" / "run_split.json"
    split.save_split(path)
    raw = json.loads(path.read_text())
    assert raw["seed"] == 42
    assert raw["sizes"] == {**SIZES, "limit_tune": 50}
    assert raw["groups"]["dev"] == sorted(raw["groups"]["dev"])
    assert split.load_split(path) == split.make_split()
    assert split.ids("fit", split.load_split(path)) == split.ids("fit")


# --- real data: run k shares its noise inside one file family ---------------
def _have(*names):
    return all((data.PROCESSED / f"{n}.parquet").exists() for n in names)


@pytest.mark.skipif(not _have("TEP_FaultFree_Training", "TEP_Faulty_Training"), reason="parquet not converted")
def test_faulty_training_run_equals_normal_run_before_onset():
    rng = np.random.default_rng(42)
    runs = [int(r) for r in rng.choice(np.arange(1, 501), size=3, replace=False)]
    onset = data.onset_for("TEP_Faulty_Training")
    normal = data.load_runs("TEP_FaultFree_Training", runs)
    faulty = data.load_runs("TEP_Faulty_Training", runs, faults=[1, 3, 9])
    for run in runs:
        ref = normal[(normal.simulationRun == run) & (normal["sample"] <= onset)][data.VARIABLES].to_numpy()
        assert ref.shape == (onset, 52)
        for fault in (1, 3, 9):
            got = faulty[(faulty.simulationRun == run) & (faulty.faultNumber == fault)
                         & (faulty["sample"] <= onset)][data.VARIABLES].to_numpy()
            assert np.array_equal(got, ref), (fault, run)


@pytest.mark.skipif(not _have("TEP_FaultFree_Training", "TEP_FaultFree_Testing"), reason="parquet not converted")
def test_no_testing_run_matches_any_training_run_on_samples_1_to_20():
    # Reads samples 1-20 of FaultFree_Testing only (spec: the one allowed read of a testing file).
    import pyarrow.parquet as pq

    def first_blocks(name):
        t = pq.read_table(data.PROCESSED / f"{name}.parquet", columns=["simulationRun", "sample"] + data.VARIABLES,
                          filters=[("sample", "<=", 20)]).to_pandas().sort_values(["simulationRun", "sample"])
        assert t.simulationRun.nunique() == 500 and len(t) == 500 * 20
        return t[data.VARIABLES].to_numpy(dtype=np.float32).reshape(500, 20, 52)

    train, test = first_blocks("TEP_FaultFree_Training"), first_blocks("TEP_FaultFree_Testing")
    rows = lambda a: {r.tobytes() for r in a.reshape(-1, 52)}  # noqa: E731
    blocks = lambda a: {b.tobytes() for b in a}  # noqa: E731
    assert not rows(train) & rows(test)      # no shared sample
    assert not blocks(train) & blocks(test)  # no shared 20-sample block (all 500 x 500 pairs)
    assert len(blocks(train)) == 500         # sanity: training blocks are all different from each other
