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


# --- review fixes -----------------------------------------------------------
def test_onset_lookup():
    assert data.onset_for("TEP_FaultFree_Training") == 20
    assert data.onset_for("TEP_Faulty_Training") == 20
    assert data.onset_for("TEP_FaultFree_Testing") == 160
    assert data.onset_for("TEP_Faulty_Testing") == 160


def test_samples_per_run_and_interval():
    assert data.FILES["TEP_Faulty_Training"]["samples"] == 500
    assert data.FILES["TEP_Faulty_Testing"]["samples"] == 960
    assert data.HOURS_PER_SAMPLE == pytest.approx(3 / 60)


def test_load_run_selects_one_run(tmp_path):
    df = pd.concat([_synthetic(4).assign(faultNumber=f, simulationRun=r) for f in (0, 1) for r in (1, 2)])
    df["xmeas_1"] = np.arange(len(df), dtype=float)
    cast = data.cast_frame(df.copy())
    cast.to_parquet(tmp_path / "TEP_Faulty_Training.parquet")
    run = data.load_run("TEP_Faulty_Training", 1, 2, processed_dir=tmp_path)
    assert list(run["sample"]) == [1, 2, 3, 4]
    assert list(run["xmeas_1"]) == [12.0, 13.0, 14.0, 15.0]  # 4th block of 4 rows: indices 12..15


def test_row_status_flags_mismatch():
    assert data.row_status("ok", "TEP_FaultFree_Training", 250_000) == "ok"
    assert data.row_status("ok", "TEP_FaultFree_Training", 249_999) == "row_mismatch"
    assert data.row_status("over_limit", "TEP_FaultFree_Training", "") == "over_limit"
    assert data.row_status("error", "TEP_FaultFree_Training", "") == "error"


def test_machine_label_has_no_identifiers():
    label = data.machine_label("11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz", 7.73 * 1024 ** 3)
    assert label == "laptop, Intel i5-1135G7, 7.7 GB RAM"
    assert data.machine_label("", 8 * 1024 ** 3) == "laptop, unknown CPU, 8.0 GB RAM"


def test_peak_after_onset_finds_first_sample_near_extreme():
    samples = np.arange(1, 11)
    values = [10, 10, 10, 20, 50, 99.8, 100, 60, 100, 30]  # onset 3: pre mean 10, extreme 100 (up)
    run = pd.DataFrame({"sample": samples, "v": values})
    sample, value = data.peak_after_onset(run, "v", onset=3)
    assert sample == 6 and value == 99.8  # first within 0.5 % of 100


def test_peak_after_onset_handles_downward_fault():
    run = pd.DataFrame({"sample": np.arange(1, 9), "v": [10, 10, 10, 5, 0.0, 1, 0.0, 3]})
    sample, value = data.peak_after_onset(run, "v", onset=3)
    assert sample == 5 and value == 0.0


def test_is_up_to_date(tmp_path):
    files = {"a": {"rows": 6}}
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir(), proc.mkdir()
    assert not data.is_up_to_date(raw_dir=raw, processed_dir=proc, files=files)       # nothing there
    (raw / "a.RData").write_bytes(b"x")
    assert not data.is_up_to_date(raw_dir=raw, processed_dir=proc, files=files)       # no parquet
    data.cast_frame(_synthetic(5)).to_parquet(proc / "a.parquet")
    assert not data.is_up_to_date(raw_dir=raw, processed_dir=proc, files=files)       # wrong row count
    data.cast_frame(_synthetic(6)).to_parquet(proc / "a.parquet")
    assert data.is_up_to_date(raw_dir=raw, processed_dir=proc, files=files)
