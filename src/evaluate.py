"""Shared scorer: alarm limit, FDR, FAR, detection delay, sustained-alarm rate and the scorecard file.

Inputs are plain per-sample frames with columns faultNumber, simulationRun, sample, score.
faultNumber 0 marks fault-free runs. An alarm is score > limit. `onset` is the last normal sample of a
faulty run (20 in training files, 160 in testing files). `skip` is the lag window L: samples with
sample <= skip are dropped from every count. "In a row" means consecutive `sample` values inside one run.
Rules: docs/adr/0002 (limits at 1% and 5% FAR; delay to the third alarm of the first 3-in-a-row after onset).
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import HOURS_PER_SAMPLE, ROOT

MINUTES_PER_SAMPLE = round(HOURS_PER_SAMPLE * 60)
IN_A_ROW = 3
HARD_FAULTS = (3, 9, 15)
DEFAULT_SCORECARD = ROOT / "results" / "scorecard.csv"
COLUMNS = ["model", "track", "limit_name", "limit_value", "fault", "metric", "value", "n_runs", "machine",
           "split_file", "written_at"]
_RUN = ["faultNumber", "simulationRun"]


def limit_for_far(normal_scores, far_target):
    """The (1 - far_target) quantile of held-back normal scores (method='higher'), so FAR <= target."""
    scores = np.asarray(normal_scores, dtype=float)
    if np.isnan(scores).any():
        raise ValueError("normal_scores contains NaN; fix the model output before setting a limit")
    if not 0 < far_target < 1:
        raise ValueError(f"far_target must be in (0, 1), got {far_target}")
    return float(np.quantile(scores, 1 - far_target, method="higher"))


def fdr(frame, limit, onset, skip=0):
    """Per fault: share of samples with sample > onset that alarm, pooled over runs. Series indexed by fault."""
    post = frame[(frame["faultNumber"] > 0) & (frame["sample"] > max(onset, skip))]
    return (post["score"] > limit).groupby(post["faultNumber"]).mean().rename("fdr")


def far(frame, limit, skip=0):
    """Share of all (non-skipped) samples of fault-free runs that alarm."""
    normal = frame[(frame["faultNumber"] == 0) & (frame["sample"] > skip)]
    return float((normal["score"] > limit).mean()) if len(normal) else float("nan")


def _first_sustained(frame, limit, after, skip, faulty):
    """Sample of the third alarm of each run's first 3-in-a-row lying entirely after `after`.

    Series indexed by (faultNumber, simulationRun) for every run of the selected kind; NaN = never.
    """
    rows = frame[(frame["faultNumber"] > 0) == faulty]
    runs = pd.MultiIndex.from_frame(rows[_RUN].drop_duplicates().sort_values(_RUN))
    hot = rows[(rows["sample"] > max(after, skip)) & (rows["score"] > limit)]
    hot = hot.sort_values(_RUN + ["sample"])
    sample = hot["sample"].astype(np.int64)
    # alarm rows only: two alarm rows ahead in the same run, exactly 2 samples later = 3 in a row
    ahead = sample.groupby([hot["faultNumber"], hot["simulationRun"]]).shift(-(IN_A_ROW - 1))
    third = ahead.where(ahead - sample == IN_A_ROW - 1)
    first = third.groupby([hot["faultNumber"], hot["simulationRun"]]).min()
    return first.reindex(runs)


def detection_delay(frame, limit, onset, skip=0):
    """Per fault: median delay in minutes over detected runs, share of runs never detected, run count.

    Delay = (sample of the third alarm of the first 3-in-a-row with all samples > onset - onset) x 3 minutes.
    DataFrame indexed by fault with columns delay_median_min, never_detected_share, n_runs.
    """
    third = _first_sustained(frame, limit, onset, skip, faulty=True)
    minutes = (third - onset) * MINUTES_PER_SAMPLE
    by_fault = minutes.groupby(level=0)
    return pd.DataFrame({
        "delay_median_min": by_fault.median(),
        "never_detected_share": minutes.isna().groupby(level=0).mean(),
        "n_runs": by_fault.size(),
    })


def sustained_rate(frame, limit, after, skip=0):
    """Share of fault-free runs with at least one 3-in-a-row whose samples are all > after."""
    third = _first_sustained(frame, limit, after, skip, faulty=False)
    return float(third.notna().mean()) if len(third) else float("nan")


def model_rows(model, track, limit_name, limit, faulty, normal, onset, skip=0, after=None, *,
               machine, split_file, train_time_s=float("nan"), peak_ram_gb=float("nan")):
    """Scorecard rows (long format, without written_at) for one model at one alarm limit.

    `faulty` holds the faulty runs (faults 1-20), `normal` the fault-free runs scored with the same model.
    `after` is the sample after which the fault-free sustained rate counts (default: onset).
    """
    after = onset if after is None else after
    per_fault_fdr = fdr(faulty, limit, onset, skip)
    delay = detection_delay(faulty, limit, onset, skip)
    n_normal = int(normal.loc[normal["faultNumber"] == 0, _RUN].drop_duplicates().shape[0])
    base = dict(model=model, track=track, limit_name=limit_name, limit_value=float(limit), machine=machine,
                split_file=split_file)

    def row(metric, value, fault=None, n_runs=None):
        return {**base, "fault": fault, "metric": metric, "value": float(value),
                "n_runs": np.nan if n_runs is None else int(n_runs)}

    rows = []
    for fault in per_fault_fdr.index:
        n = delay.loc[fault, "n_runs"]
        rows += [row("fdr", per_fault_fdr[fault], int(fault), n),
                 row("delay_median_min", delay.loc[fault, "delay_median_min"], int(fault), n),
                 row("never_detected_share", delay.loc[fault, "never_detected_share"], int(fault), n)]
    n_faulty = delay["n_runs"].sum()
    rows += [
        row("mean_fdr", per_fault_fdr.mean(), n_runs=n_faulty),
        row("mean_fdr_excl_3_9_15", per_fault_fdr.drop(list(HARD_FAULTS), errors="ignore").mean(), n_runs=n_faulty),
        row("far", far(normal, limit, skip), n_runs=n_normal),
        row("sustained_rate", sustained_rate(normal, limit, after, skip), n_runs=n_normal),
        row("train_time_s", train_time_s),
        row("peak_ram_gb", peak_ram_gb),
    ]
    return rows


def write_scorecard(rows, path=DEFAULT_SCORECARD):
    """Write long-format rows to a CSV. Rows of the same model and limit_name already in the file are replaced.

    Other models (and the same model at another limit) stay untouched. `fault` is empty for overall rows.
    """
    new = pd.DataFrame(rows)
    missing = set(COLUMNS) - {"written_at"} - set(new.columns)
    if missing:
        raise KeyError(f"scorecard rows lack columns: {sorted(missing)}")
    if "written_at" not in new:
        new["written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = new[COLUMNS]
    new["fault"] = new["fault"].astype("Int64")
    new["n_runs"] = new["n_runs"].astype("Int64")
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, dtype={"model": str, "limit_name": str, "fault": "Int64", "n_runs": "Int64"})
        replaced = set(zip(new["model"], new["limit_name"]))
        keep = [(m, l) not in replaced for m, l in zip(old["model"], old["limit_name"])]
        new = pd.concat([old[keep], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
