"""TEP data access: download, convert to parquet, memory guard, variable names.

Dataset: Rieth et al. (2017), doi:10.7910/DVN/6C3JR1. Variable names and units:
Downs & Vogel (1993) as compiled in research/tep-flowsheet.md.
"""
import gc
import hashlib
import json
import multiprocessing as mp
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadr

DOI = "doi:10.7910/DVN/6C3JR1"
LIST_URL = f"https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId={DOI}"
FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{id}"

# Smallest first. Rows from the tep-data skill.
FILES = {
    "TEP_FaultFree_Training": {"id": 3031241, "rows": 250_000},
    "TEP_FaultFree_Testing": {"id": 3031240, "rows": 480_000},
    "TEP_Faulty_Training": {"id": 3031242, "rows": 5_000_000},
    "TEP_Faulty_Testing": {"id": 3031243, "rows": 9_600_000},
}

ID_COLUMNS = ["faultNumber", "simulationRun", "sample"]
VARIABLES = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]

_XMEAS = [
    ("A feed", "kscmh"), ("D feed", "kg/h"), ("E feed", "kg/h"), ("A and C feed", "kscmh"),
    ("Recycle flow", "kscmh"), ("Reactor feed rate", "kscmh"), ("Reactor pressure", "kPa gauge"),
    ("Reactor level", "%"), ("Reactor temperature", "deg C"), ("Purge rate", "kscmh"),
    ("Product separator temperature", "deg C"), ("Product separator level", "%"),
    ("Product separator pressure", "kPa gauge"), ("Product separator underflow", "m3/h"),
    ("Stripper level", "%"), ("Stripper pressure", "kPa gauge"), ("Stripper underflow", "m3/h"),
    ("Stripper temperature", "deg C"), ("Stripper steam flow", "kg/h"), ("Compressor work", "kW"),
    ("Reactor cooling water outlet temperature", "deg C"),
    ("Separator cooling water outlet temperature", "deg C"),
]
_ANALYZERS = (
    [(f"Reactor feed analysis, component {c}", "mol %") for c in "ABCDEF"]
    + [(f"Purge gas analysis, component {c}", "mol %") for c in "ABCDEFGH"]
    + [(f"Product analysis, component {c}", "mol %") for c in "DEFGH"]
)
_XMV = [
    ("D feed flow", "%"), ("E feed flow", "%"), ("A feed flow", "%"), ("A and C feed flow", "%"),
    ("Compressor recycle valve", "%"), ("Purge valve", "%"), ("Separator pot liquid flow", "%"),
    ("Stripper liquid product flow", "%"), ("Stripper steam valve", "%"),
    ("Reactor cooling water flow", "%"), ("Condenser cooling water flow", "%"),
]
# xmv units: D&V give valve positions in % (not in tep-flowsheet.md, which lists none).
VARIABLE_INFO = dict(zip(VARIABLES, _XMEAS + _ANALYZERS + _XMV))


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


def _tree_usage():
    """(current RSS, peak working set) in bytes summed over this process and all descendants."""
    me = psutil.Process()
    cur = peak = 0
    for p in [me] + me.children(recursive=True):
        try:
            mi = p.memory_info()
        except psutil.Error:
            continue
        cur += mi.rss
        peak += getattr(mi, "peak_wset", mi.rss) if p is not me else mi.rss
    return cur, peak


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
        cur, pk = _tree_usage()
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


# --- fault shift ranking ----------------------------------------------------
def rank_shifts(run, variables, onset):
    """|mean after onset - mean before| / std before, sorted descending.

    'before' = samples <= onset of the same run (ddof=1); variables with a constant
    pre-onset stretch are left out (shift undefined).
    """
    pre = run.loc[run["sample"] <= onset, variables]
    post = run.loc[run["sample"] > onset, variables]
    std = pre.std(ddof=1)
    shift = ((post.mean() - pre.mean()).abs() / std).where(std > 0).dropna()
    return shift.sort_values(ascending=False)
