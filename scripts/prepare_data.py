"""Download the 4 TEP files and convert them to parquet, smallest first.

Logs conversion runtime and peak RAM (process tree) per file to
results/phase-0/conversion_log.csv. Conversions run in a child process that is
killed if the tree passes LIMIT_GB. If everything is already in place (raw files
present, parquet with the right row counts) it reports "up to date" and touches
neither the network nor the log.
Usage: .venv/Scripts/python scripts/prepare_data.py [file names]   (default: all four)
"""
import csv
import sys
import time
from pathlib import Path

import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, prepare  # noqa: E402

LIMIT_GB = 6.0
LOG = ROOT / "results" / "phase-0" / "conversion_log.csv"
COLUMNS = ["file", "rows", "runtime_s", "peak_ram_gb", "machine", "status"]


def main():
    names = sys.argv[1:] or list(data.FILES)  # dict order = smallest first
    if prepare.is_up_to_date(names):
        print("up to date")
        return
    md5s = prepare.fetch_md5s()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    rows = []
    for name in names:
        rdata, parquet = data.RAW / f"{name}.RData", data.PROCESSED / f"{name}.parquet"
        print(f"{name}: download {prepare.download(name, rdata, md5s[name])}", flush=True)
        parquet.unlink(missing_ok=True)
        t0 = time.perf_counter()
        status, peak = prepare.run_guarded(prepare.convert, (rdata, parquet), LIMIT_GB)
        runtime = time.perf_counter() - t0
        n = pq.read_metadata(parquet).num_rows if status == "ok" else ""
        rows.append({"file": name, "rows": n, "runtime_s": round(runtime, 1), "peak_ram_gb": round(peak, 2),
                     "machine": machine, "status": prepare.row_status(status, name, n)})
        print(rows[-1], flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "w", newline="") as f:  # rewritten after every file so a crash keeps earlier rows
            w = csv.DictWriter(f, COLUMNS)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
