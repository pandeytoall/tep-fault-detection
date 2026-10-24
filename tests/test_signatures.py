"""Phase 1 tests for the IDV numbering check in src/signatures.py (synthetic runs, no data files)."""
import numpy as np
import pandas as pd
import pytest

from src import data, signatures

ONSET = 20


def _run(shifts, seed):
    """40-sample run: unit-variance noise on all 52 variables, then a mean shift (in std units) after ONSET."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(40, 52)), columns=data.VARIABLES)
    for tag, size in shifts.items():
        frame.loc[ONSET:, tag] += size
    frame.insert(0, "sample", np.arange(1, 41))
    return frame


def test_expected_signatures_are_the_four_hypotheses_from_the_spec():
    assert signatures.IDV_EXPECTED == {4: "xmv_10", 5: "xmv_11", 6: "xmeas_1", 7: "xmeas_4"}


def test_idv_check_reports_top3_share_and_most_common_top1():
    big = {"xmv_10": 60, "xmeas_9": 30}
    # fault 4: expected variable is the top-1 in 3 runs and only 4th in the last run
    crowded = {"xmeas_9": 300, "xmeas_7": 200, "xmeas_8": 150, "xmv_10": 60}
    # fault 5: expected variable never in the top 3 (xmeas_9 always first)
    other = {"xmeas_9": 300, "xmeas_7": 200, "xmeas_8": 150, "xmv_11": 5}
    runs = {4: [_run(big, 0), _run(big, 1), _run(big, 2), _run(crowded, 3)],
            5: [_run(other, 4), _run(other, 5)]}
    out = signatures.idv_check(runs, ONSET, expected={4: "xmv_10", 5: "xmv_11"})
    assert list(out.columns) == ["fault", "expected_tag", "expected_name", "share_runs_in_top3", "most_common_top1"]
    f4, f5 = out.set_index("fault").loc[4], out.set_index("fault").loc[5]
    assert f4.expected_tag == "xmv_10" and f4.expected_name == "Reactor cooling water flow"
    assert f4.share_runs_in_top3 == pytest.approx(0.75)
    assert f4.most_common_top1 == "xmv_10"
    assert f5.share_runs_in_top3 == 0 and f5.most_common_top1 == "xmeas_9"


def test_idv_check_passes_all_four_faults_in_one_call():
    runs = {f: [_run({tag: 80}, f * 10 + i) for i in range(3)] for f, tag in signatures.IDV_EXPECTED.items()}
    out = signatures.idv_check(runs, ONSET)
    assert list(out.fault) == [4, 5, 6, 7]
    assert (out.share_runs_in_top3 == 1).all()
    assert list(out.most_common_top1) == list(signatures.IDV_EXPECTED.values())
