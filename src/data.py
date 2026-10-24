"""TEP data access: paths, file facts, column names, run loading, labels.

Dataset: Rieth et al. (2017), doi:10.7910/DVN/6C3JR1.
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Smallest first. Rows and samples per run from the tep-data skill.
FILES = {
    "TEP_FaultFree_Training": {"id": 3031241, "rows": 250_000, "samples": 500},
    "TEP_FaultFree_Testing": {"id": 3031240, "rows": 480_000, "samples": 960},
    "TEP_Faulty_Training": {"id": 3031242, "rows": 5_000_000, "samples": 500},
    "TEP_Faulty_Testing": {"id": 3031243, "rows": 9_600_000, "samples": 960},
}
HOURS_PER_SAMPLE = 3 / 60  # one sample every 3 minutes
ONSET = {"Training": 20, "Testing": 160}  # fault starts after this sample (1 h / 8 h)


def onset_for(name):
    """Last normal sample of a faulty run: 20 for *_Training files, 160 for *_Testing files."""
    return ONSET[name.rsplit("_", 1)[1]]


ID_COLUMNS = ["faultNumber", "simulationRun", "sample"]
VARIABLES = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]


def load_run(name, fault, run, processed_dir=PROCESSED):
    """One simulation run (sample + 52 variables) from the processed parquet."""
    return pq.read_table(Path(processed_dir) / f"{name}.parquet", columns=["sample"] + VARIABLES,
                         filters=[("faultNumber", "=", fault), ("simulationRun", "=", run)]).to_pandas()


def load_runs(name, runs, faults=None, processed_dir=PROCESSED):
    """Rows of the given run IDs (and fault numbers, default all) from one parquet file.

    Columns: ID_COLUMNS + VARIABLES, variables float32. The filter is applied while reading,
    so a whole faulty file is never held in memory.
    """
    filters = [("simulationRun", "in", [int(r) for r in runs])]
    if faults is not None:
        filters.append(("faultNumber", "in", [int(f) for f in faults]))
    return pq.read_table(Path(processed_dir) / f"{name}.parquet", columns=ID_COLUMNS + VARIABLES,
                         filters=filters).to_pandas()


def labels(frame, name):
    """int8 label per row: 1 only if faultNumber > 0 and sample > onset of the file family, else 0."""
    return ((frame["faultNumber"].to_numpy() > 0) & (frame["sample"].to_numpy() > onset_for(name))).astype(np.int8)
