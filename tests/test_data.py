"""Phase 0 tests for src/data.py. No network."""
import hashlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyreadr
import pytest

from src import data

HELLO_MD5 = "5d41402abc4b2a76b9719d911017c592"  # md5 of b"hello"


def _synthetic(n=6):
    cols = {"faultNumber": np.zeros(n), "simulationRun": np.ones(n), "sample": np.arange(1, n + 1, dtype=float)}
    for name in data.VARIABLES:
        cols[name] = np.linspace(0.5, 1.5, n)
    return pd.DataFrame(cols)


# --- dtype conversion -------------------------------------------------------
def test_cast_frame_dtypes():
    out = data.cast_frame(_synthetic())
    assert list(out.columns) == data.ID_COLUMNS + data.VARIABLES
    assert len(data.VARIABLES) == 52
    assert all(out[c].dtype == np.int16 for c in data.ID_COLUMNS)
    assert all(out[c].dtype == np.float32 for c in data.VARIABLES)


def test_convert_roundtrip(tmp_path):
    src = tmp_path / "t.RData"
    pyreadr.write_rdata(str(src), _synthetic(), df_name="fault_free_training")
    dst = tmp_path / "t.parquet"
    rows = data.convert(src, dst)
    assert rows == 6
    schema = pq.read_schema(dst)
    assert str(schema.field("sample").type) == "int16"
    assert str(schema.field("xmeas_1").type) == "float"  # arrow name for float32
    assert not (tmp_path / "t.parquet.tmp").exists()


# --- md5 skip ---------------------------------------------------------------
def _fake_fetch(content, calls):
    def fetch(url, dest):
        calls.append(url)
        dest.write_bytes(content)
    return fetch


def test_download_skips_matching_file(tmp_path):
    dest = tmp_path / "f.RData"
    dest.write_bytes(b"hello")
    calls = []
    status = data.download("TEP_FaultFree_Training", dest, HELLO_MD5, fetch=_fake_fetch(b"x", calls))
    assert status == "skipped" and calls == []
    assert dest.read_bytes() == b"hello"


def test_download_replaces_mismatching_file(tmp_path):
    dest = tmp_path / "f.RData"
    dest.write_bytes(b"corrupt")
    calls = []
    status = data.download("TEP_FaultFree_Training", dest, HELLO_MD5, fetch=_fake_fetch(b"hello", calls))
    assert status == "downloaded" and len(calls) == 1
    assert dest.read_bytes() == b"hello"


def test_download_rejects_bad_checksum(tmp_path):
    dest = tmp_path / "f.RData"
    with pytest.raises(ValueError, match="md5"):
        data.download("TEP_FaultFree_Training", dest, HELLO_MD5, fetch=_fake_fetch(b"wrong", []))
    assert not dest.exists()


# --- memory guard -----------------------------------------------------------
def test_over_limit_decision():
    gb = 1024 ** 3
    assert not data.over_limit(5.99 * gb, 6.0)
    assert data.over_limit(6.01 * gb, 6.0)
    assert not data.over_limit(6.0 * gb, 6.0)


def _alloc_and_wait(mb):
    x = np.ones(mb * 1024 * 1024 // 8)
    x.sum()
    import time
    time.sleep(30)


def test_guard_kills_child_over_limit():
    status, peak = data.run_guarded(_alloc_and_wait, (400,), limit_gb=0.3)
    assert status == "over_limit"
    assert peak > 0.3


def test_guard_lets_small_child_finish():
    status, peak = data.run_guarded(_alloc_and_wait_short, (), limit_gb=2.0)
    assert status == "ok"
    assert 0 < peak < 2.0


def _alloc_and_wait_short():
    pass


# --- shift ranking ----------------------------------------------------------
def test_rank_shifts_orders_by_standardised_shift():
    pre = np.tile([1.0, -1.0], 10)  # mean 0, std (ddof=1) = sqrt(20/19)
    n_post = 30
    run = pd.DataFrame({
        "sample": np.arange(1, 51),
        "xmeas_1": np.concatenate([pre, np.zeros(n_post)]),        # no shift
        "xmeas_2": np.concatenate([pre, np.full(n_post, 5.0)]),    # shift 5
        "xmeas_3": np.concatenate([pre, np.full(n_post, -2.0)]),   # shift -2
    })
    ranked = data.rank_shifts(run, ["xmeas_1", "xmeas_2", "xmeas_3"], onset=20)
    assert list(ranked.index) == ["xmeas_2", "xmeas_3", "xmeas_1"]
    assert ranked["xmeas_2"] == pytest.approx(5 / np.sqrt(20 / 19), rel=1e-6)
    assert ranked["xmeas_3"] == pytest.approx(2 / np.sqrt(20 / 19), rel=1e-6)  # absolute value


def test_rank_shifts_ignores_constant_pre_stretch():
    run = pd.DataFrame({"sample": np.arange(1, 41), "xmeas_1": np.r_[np.ones(20), np.full(20, 9.0)],
                        "xmeas_2": np.r_[np.tile([1.0, -1.0], 10), np.full(20, 1.0)]})
    ranked = data.rank_shifts(run, ["xmeas_1", "xmeas_2"], onset=20)
    assert list(ranked.index) == ["xmeas_2"]
