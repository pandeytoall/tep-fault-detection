"""TEP data preparation: download, convert to parquet, memory guard, machine label.

Dataset: Rieth et al. (2017), doi:10.7910/DVN/6C3JR1.
"""
import gc
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadr

from src.data import FILES, ID_COLUMNS, PROCESSED, RAW, VARIABLES

DOI = "doi:10.7910/DVN/6C3JR1"
LIST_URL = f"https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId={DOI}"
FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{id}"

# --- download ---------------------------------------------------------------
def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _open(url):
    # Dataverse returns 403 to urllib's default User-Agent
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tep-project/0.1"}))


def fetch_md5s():
    """{file name without .RData: md5} from the Dataverse API."""
    with _open(LIST_URL) as r:
        listing = json.load(r)["data"]
    return {f["dataFile"]["filename"].removesuffix(".RData"): f["dataFile"]["md5"] for f in listing}


def _fetch(url, dest):
    with _open(url) as r, open(dest, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)


def download(name, dest, md5, fetch=_fetch):
    """Download FILES[name] to dest unless dest already has the right md5. Returns 'skipped' or 'downloaded'."""
    dest = Path(dest)
    if dest.exists() and md5_of(dest) == md5:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    fetch(FILE_URL.format(id=FILES[name]["id"]), tmp)
    got = md5_of(tmp)
    if got != md5:
        tmp.unlink()
        raise ValueError(f"md5 mismatch for {name}: got {got}, expected {md5}")
    os.replace(tmp, dest)
    return "downloaded"


# --- conversion -------------------------------------------------------------
def cast_frame(df):
    """int16 id columns, float32 variables. Consumes columns of df to keep peak memory down."""
    out = {}
    for c in ID_COLUMNS:
        out[c] = df.pop(c).to_numpy().astype(np.int16)
    for c in VARIABLES:
        out[c] = df.pop(c).to_numpy().astype(np.float32)
    return pd.DataFrame(out)


def convert(rdata_path, parquet_path):
    """RData -> float32/int16 parquet. Returns the row count. Writes atomically."""
    result = pyreadr.read_r(str(rdata_path))
    (key,) = result.keys()
    df = result.pop(key)
    del result
    typed = cast_frame(df)
    del df
    gc.collect()
    rows = len(typed)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet_path.with_name(parquet_path.name + ".tmp")
    pq.write_table(pa.Table.from_pandas(typed, preserve_index=False), tmp)
    os.replace(tmp, parquet_path)
    del typed
    gc.collect()
    return rows


# --- memory guard -----------------------------------------------------------
GB = 1024 ** 3


def over_limit(peak_bytes, limit_gb):
    return peak_bytes > limit_gb * GB


def _tree_usage(own_peak=True):
    """(current RSS, peak working set) in bytes summed over this process and all descendants.

    The peak uses the OS peak working set where there is one (current RSS elsewhere). With own_peak=False
    this process counts at its current RSS only, so a poller's earlier peak does not inflate a child's reading.
    """
    me = psutil.Process()
    cur = peak = 0
    for p in [me] + me.children(recursive=True):
        try:
            mi = p.memory_info()
        except psutil.Error:
            continue
        cur += mi.rss
        peak += getattr(mi, "peak_wset", mi.rss) if own_peak or p is not me else mi.rss
    return cur, peak


def peak_ram_gb():
    """Peak working set of this process plus its children, in GB (current RSS where the OS has no peak)."""
    return _tree_usage()[1] / GB


def runtime_line(t0):
    """The runtime and peak-RAM line every script prints at the end; t0 is a time.perf_counter() start."""
    return f"runtime {time.perf_counter() - t0:.1f} s, peak RAM {peak_ram_gb():.2f} GB"


def _kill_tree(proc):
    try:
        parent = psutil.Process(proc.pid)
        for c in parent.children(recursive=True):
            c.kill()
        parent.kill()
    except psutil.Error:
        pass
    proc.join()


def run_guarded(target, args, limit_gb, poll_s=0.02):
    """Run target(*args) in a child process, polling the process tree's RAM.

    Kills the child if RSS passes limit_gb. Returns (status, peak_gb); status is
    'ok', 'over_limit' or 'error'.
    """
    proc = mp.get_context("spawn").Process(target=target, args=args)
    proc.start()
    peak = 0
    status = None
    while proc.is_alive():
        cur, pk = _tree_usage(own_peak=False)
        peak = max(peak, cur, pk)
        if over_limit(cur, limit_gb):
            status = "over_limit"
            _kill_tree(proc)
            break
        time.sleep(poll_s)
    proc.join()
    if status is None:
        status = "ok" if proc.exitcode == 0 else "error"
    return status, peak / GB



def row_status(status, name, rows):
    """Downgrade an 'ok' conversion to 'row_mismatch' if its row count differs from FILES[name]['rows']."""
    if status == "ok" and rows != FILES[name]["rows"]:
        return "row_mismatch"
    return status


def machine_label(cpu_name, ram_bytes):
    """Non-identifying machine label: CPU model and RAM only (no hostname, user or paths)."""
    m = re.search(r"i[3579]-[0-9]+[A-Z0-9]*", cpu_name)
    cpu = f"Intel {m.group(0)}" if m else "unknown CPU"
    return f"laptop, {cpu}, {ram_bytes / GB:.1f} GB RAM"


def cpu_name():
    """CPU model string from the Windows registry ('' if unavailable)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0]
    except (ImportError, OSError):
        return ""



def is_up_to_date(names=None, raw_dir=RAW, processed_dir=PROCESSED, files=FILES):
    """True if every raw file exists and every parquet exists with the expected row count (no network)."""
    for name in names or files:
        raw, parquet = Path(raw_dir) / f"{name}.RData", Path(processed_dir) / f"{name}.parquet"
        if not raw.exists() or not parquet.exists():
            return False
        if pq.read_metadata(parquet).num_rows != files[name]["rows"]:
            return False
    return True
