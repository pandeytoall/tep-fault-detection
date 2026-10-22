"""Download the 4 TEP files and convert them to parquet, smallest first.

Logs conversion runtime and peak RAM (process tree) per file to
results/phase-0/conversion_log.csv. Conversions run in a child process that is
killed if the tree passes LIMIT_GB.
Usage: .venv/Scripts/python scripts/prepare_data.py [file names]   (default: all four)
"""
import csv
import platform
import sys
import time
from pathlib import Path

import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data  # noqa: E402

LIMIT_GB = 6.0
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LOG = ROOT / "results" / "phase-0" / "conversion_log.csv"
COLUMNS = ["file", "rows", "runtime_s", "peak_ram_gb", "machine", "status"]


def machine():
    ram = psutil.virtual_memory().total / data.GB
    return f"{platform.node()} {platform.processor()} {ram:.1f}GB RAM"


def main():
    md5s = data.fetch_md5s()
    rows = []
    for name in sys.argv[1:] or data.FILES:  # dict order = smallest first
        rdata, parquet = RAW / f"{name}.RData", PROCESSED / f"{name}.parquet"
        print(f"{name}: download {data.download(name, rdata, md5s[name])}", flush=True)
        parquet.unlink(missing_ok=True)
        t0 = time.perf_counter()
        status, peak = data.run_guarded(data.convert, (rdata, parquet), LIMIT_GB)
        runtime = time.perf_counter() - t0
        n = pq.read_metadata(parquet).num_rows if status == "ok" else ""
        rows.append({"file": name, "rows": n, "runtime_s": round(runtime, 1), "peak_ram_gb": round(peak, 2),
                     "machine": machine(), "status": status})
        print(rows[-1], flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "w", newline="") as f:  # rewritten after every file so a crash keeps earlier rows
            w = csv.DictWriter(f, COLUMNS)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
