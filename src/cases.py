"""Case sets for the assistant (phase 2b): which runs, how they are cropped, and the answer keys.

A case is one TEP run with its first `crop` and its last `end_crop` samples removed; the historian clock restarts at
0 h at the first kept sample. The start crop (seed-42 draw) hides the fault start; the end crop is the rest of the
maximum crop (max_crop - crop), so every case of a set has the same length and the length gives nothing away
(test 880 samples = 44 h, dev 490 samples; onset = onset_sample - crop). Two sets:
- test: testing family, 3 runs per fault 1-20 plus 10 fault-free runs = 70 cases, crop uniform 0-80, end_crop 80 - crop.
  FROZEN: built once, reused unchanged by v1, v2 and v3 (guard_frozen refuses a rebuild).
- dev: training family, only the `dev` run IDs, 16 faults plus 4 normal, crop uniform 0-10, end_crop 10 - crop; for prompt iteration.
Every case has its own run ID (runs with one number share a noise stream within a family, ADR 0003).
The answer key (case, fault, run, crop, scoring unit, onset in historian hours) is read only by src/assistant/score.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import contributions, data, historian, mspc

SEED = 42
KEY_COLUMNS = ["case_id", "set", "family", "fault", "run", "crop", "end_crop", "unit", "onset_h"]

# IDV -> scoring unit, from Downs & Vogel (1993) as listed in research/tep-flowsheet.md, section 5.
_UNIT_BY_FAULT = {**{f: "feeds" for f in (1, 2, 3, 6, 7, 8, 9, 10)}, **{f: "reactor" for f in (4, 11, 13, 14)},
                  **{f: "condenser/separator" for f in (5, 12, 15)}, **{f: "unknown" for f in range(16, 21)}}


def case_unit(fault):
    """Scoring unit of a fault (IDV 1-20); 'unknown' for 16-20, 'none' for a normal run (fault 0)."""
    return "none" if fault == 0 else _UNIT_BY_FAULT[int(fault)]


def onset_hours(onset_sample, crop):
    """Historian hour of the first faulty sample (sample onset + 1) in a run cropped by `crop` samples."""
    return round((onset_sample - crop) * data.HOURS_PER_SAMPLE, 2)


def _key(set_name, family, faults, runs, crops, end_crops, rng, prefix):
    """Shuffle the (fault, run, crop) rows and number them prefix_01, prefix_02, ..."""
    onset = data.ONSET[family.title()]
    rows = pd.DataFrame({"fault": faults, "run": runs, "crop": crops, "end_crop": end_crops}).iloc[rng.permutation(len(faults))]
    rows = rows.reset_index(drop=True)
    rows.insert(0, "case_id", [f"{prefix}_{i:02d}" for i in range(1, len(rows) + 1)])
    rows["set"], rows["family"] = set_name, family
    rows["unit"] = [case_unit(f) for f in rows["fault"]]
    rows["onset_h"] = [onset_hours(onset, c) if f > 0 else np.nan for f, c in zip(rows["fault"], rows["crop"])]
    return rows[KEY_COLUMNS].astype({"fault": int, "run": int, "crop": int, "end_crop": int})


def build_test_key(seed=SEED):
    """Answer key of the frozen test set: 70 distinct testing-family runs, crop uniform on 0-80."""
    rng = np.random.default_rng(seed)
    faults = [f for f in range(1, 21) for _ in range(3)] + [0] * 10
    runs = rng.choice(np.arange(1, 501), size=len(faults), replace=False)
    crops = rng.integers(0, 81, size=len(faults))
    end_crops = 80 - crops
    return _key("test", "testing", faults, runs, crops, end_crops, rng, "case")


def build_dev_key(dev_ids, seed=SEED):
    """Answer key of the dev set: 16 faults (seed-42 choice of 1-20) plus 4 normal, one per `dev` run ID."""
    rng = np.random.default_rng(seed)
    if len(dev_ids) != 20:
        raise ValueError(f"expected 20 dev run IDs, got {len(dev_ids)}")
    faults = sorted(int(f) for f in rng.choice(np.arange(1, 21), size=16, replace=False)) + [0] * 4
    runs = rng.permutation(np.asarray(sorted(dev_ids)))
    crops = rng.integers(0, 11, size=len(faults))
    end_crops = 10 - crops
    return _key("dev", "training", faults, runs, crops, end_crops, rng, "dev")


def guard_frozen(set_name, paths):
    """Raise FileExistsError if the test set was already built (any of `paths` exists). Dev may be rebuilt."""
    if set_name != "test":
        return
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise FileExistsError("the test set is frozen and already built; never rebuild or edit it: "
                              + ", ".join(existing))


def write_answer_key(key, path, machine):
    """Write the answer key CSV with a machine column. The frozen test key is never overwritten."""
    path = Path(path)
    guard_frozen(str(key["set"].iloc[0]), [path])
    path.parent.mkdir(parents=True, exist_ok=True)
    key.assign(machine=machine).to_csv(path, index=False)


def crop_run(frame, crop, end_crop=0):
    """A run without its first `crop` and last `end_crop` samples, with t_hours (0 at the first kept sample) and the
    52 variables."""
    last = frame["sample"].max() - end_crop
    kept = frame[(frame["sample"] > crop) & (frame["sample"] <= last)].sort_values("sample")
    t = np.round((kept["sample"].to_numpy() - crop - 1) * data.HOURS_PER_SAMPLE, 2)
    out = kept[data.VARIABLES].reset_index(drop=True)
    out.insert(0, "t_hours", t)
    return out


def load_case_frames(key, processed_dir=data.PROCESSED):
    """{case_id: cropped run frame (t_hours + 52 variables)} for every row of an answer key, from the parquet files.

    Reads only the runs of the key (filtered while reading), one file at a time.
    """
    out = {}
    for (family, fault), group in key.groupby(["family", "fault"], sort=False):
        name = f"TEP_Faulty_{family.title()}" if fault > 0 else f"TEP_FaultFree_{family.title()}"
        frame = data.load_runs(name, group["run"], faults=[fault], processed_dir=processed_dir)
        for row in group.itertuples():
            run = frame[frame["simulationRun"] == row.run]
            if run.empty:
                raise ValueError(f"{name} has no run {row.run} for {row.case_id}")
            out[row.case_id] = crop_run(run, row.crop, row.end_crop)
    return {case_id: out[case_id] for case_id in key["case_id"]}


# --- build-time sanity checks ---------------------------------------------------------------------
ONSET_RANGE_H = {"test": (0.0, 8.0), "dev": (0.0, 1.0)}
A_FEED_TAGS = ("xmeas_1", "xmv_3")


def sanity_checks(key, db_path, model_path):
    """Acceptance checks on a built set: {normal_share_above, fault6, onset_in_range}.

    - normal_share_above: mean share of samples above the 1% limits (t2, spe) over the normal cases (expect 0.3-3%);
    - fault6: for each fault 6 case, the top 5 SPE contributors and whether an A-feed reading (xmeas_1, xmv_3) is there;
    - onset_in_range: every faulty case's onset_h lies in the set's range (test 0-8 h, dev 0-1 h).
    """
    monitor = mspc.PCAMonitor.load(model_path)
    reference = contributions.load_reference(model_path)
    shares, fault6 = {"t2": [], "spe": []}, {}
    for row in key.itertuples():
        if row.fault not in (0, 6):
            continue
        view = historian.CaseView(db_path, row.case_id)
        matrix = view.matrix()
        if row.fault == 0:
            scores = historian.score_case(monitor, matrix)
            for stat in shares:
                shares[stat].append(float((scores[stat] > monitor.limits[stat][historian.LIMIT_NAME]).mean()))
        else:
            window = contributions.window_rows(matrix, contributions.window_start(view.alarms(), "spe"), None)
            top = contributions.top_contributors(monitor, reference, window, "spe", 5)
            tags = [t["tag"] for t in top]
            fault6[row.case_id] = {"top5": tags, "a_feed_in_top5": any(t in A_FEED_TAGS for t in tags)}
        view.close()
    lo, hi = ONSET_RANGE_H[str(key["set"].iloc[0])]
    onsets = key.loc[key["fault"] > 0, "onset_h"]
    return {"normal_share_above": {**{s: float(np.mean(v)) if v else float("nan") for s, v in shares.items()},
                                   "n_cases": len(shares["t2"])},
            "fault6": fault6, "onset_in_range": bool(onsets.between(lo, hi).all())}


def repeat_subset(key, n=10, seed=SEED):
    """The `n` case IDs (seed-42 choice from the key, sorted) that get 2 extra runs for the repeat-agreement check."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(key["case_id"].to_numpy(), size=n, replace=False).tolist())
