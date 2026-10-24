"""Phase 1 test for the dummy model in scripts/dummy_scorecard.py (synthetic data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import dummy_scorecard  # noqa: E402

from src import data  # noqa: E402


def _normal_frame(n=200):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(10, 2, size=(n, 52)).astype(np.float32), columns=data.VARIABLES)
    frame["xmv_2"] = np.float32(5.0)  # constant on the fit runs: zero std
    return frame


def test_fit_drops_zero_std_variables_and_lists_them():
    model = dummy_scorecard.fit_maxz(_normal_frame())
    assert model["dropped"] == ["xmv_2"]
    assert "xmv_2" not in model["variables"] and len(model["variables"]) == 51


def test_score_is_max_abs_z_over_kept_variables():
    model = dummy_scorecard.fit_maxz(_normal_frame())
    row = pd.DataFrame([{**{v: 0.0 for v in data.VARIABLES}, **model["mean"].to_dict()}]).astype(np.float32)
    row["xmeas_5"] += np.float32(3 * model["std"]["xmeas_5"])   # 3 std up on one variable
    row["xmv_2"] = np.float32(999.0)                            # dropped variable must not matter
    assert dummy_scorecard.score_maxz(model, row)[0] == pytest.approx(3.0, abs=1e-3)
