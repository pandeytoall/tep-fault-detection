---
name: tep-data
description: Tennessee Eastman Process (TEP) dataset facts, labeling rules, evaluation rules, laptop limits and project folder conventions. Load before planning, writing, researching or reviewing anything that touches the TEP data.
---

# TEP project facts

## Dataset
Rieth, Amsel, Tran & Cook (2017), "Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation", Harvard Dataverse, doi:10.7910/DVN/6C3JR1.

| File | Runs | Samples/run | Rows | Fault onset |
|---|---|---|---|---|
| TEP_FaultFree_Training.RData | 500 | 500 (25 h) | 250k | none |
| TEP_FaultFree_Testing.RData | 500 | 960 (48 h) | 480k | none |
| TEP_Faulty_Training.RData | 500 x 20 faults | 500 (25 h) | 5.0M | after sample 20 (1 h) |
| TEP_Faulty_Testing.RData | 500 x 20 faults | 960 (48 h) | 9.6M | after sample 160 (8 h) |

- Sampling every 3 minutes. Columns: `faultNumber` (0 = normal, 1-20), `simulationRun`, `sample` (1-based), then 52 process variables `xmeas_1..xmeas_41`, `xmv_1..xmv_11`.
- Download via the Dataverse API: list files with `https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest/files?persistentId=doi:10.7910/DVN/6C3JR1`, fetch each with `https://dataverse.harvard.edu/api/access/datafile/<id>` (add `?format=original` if you get `.tab` instead of `.RData`).
- Load with `pyreadr.read_r(path)`; it returns a dict, so check the object name with `list(result.keys())`.
- Real variable names: Downs & Vogel (1993), "A plant-wide industrial process control problem". Source them; don't guess.

## Labeling rule (the #1 source of wrong results)
`label = 1` only if `faultNumber > 0` AND `sample > onset`, where onset = 20 for `*_Training` files and 160 for `*_Testing` files.
Samples before onset in faulty runs are normal operation.

## Evaluation rules
- Fit and tune only on `*_Training` files. `*_Testing` files are used only for final metrics.
- Validation splits are by `simulationRun`, never by random rows (samples within a run are autocorrelated).
- Fit scalers/PCA on training data only (MSPC: normal training data only).
- Compare models at equal false-alarm rate: set each model's alarm threshold on validation normal data so FAR is about 1% (MSPC: 99% control limits). The research phase may change this target; the user decides.
- Report per fault: FDR (share of post-onset samples flagged). Overall: mean FDR with and without faults 3, 9, 15; FAR on `TEP_FaultFree_Testing`; train time; peak RAM.
- Faults 3, 9, 15 are near-undetectable in the literature. High FDR on them means suspect leakage, not a breakthrough.

## Laptop limits
Windows 11, i5-1135G7 (4 cores / 8 threads), 7.7 GB RAM, Intel Iris Xe (no CUDA).
- Keep peak RAM for any script under about 5 GB. Full Faulty_Testing is about 4.2 GB as float64, so never hold it all in float64.
- Convert each `.RData` to parquet once (float32 for variables, int16 for id columns), free memory, and work from parquet. Subsample by `simulationRun`.
- If a file can't be converted within the RAM limit, stop and report options. Don't silently switch data sources.
- CPU-only: no deep learning unless the user approves. Use `n_jobs` of 6 or fewer.

## Project conventions
- Python: project venv at `.venv`. Always call `.venv/Scripts/python`, since shell state doesn't persist.
- Folders:
  - `data/raw/`: original downloads, never modified
  - `data/processed/`: parquet
  - `src/`: code
  - `tests/`: pytest (`.venv/Scripts/python -m pytest -q`)
  - `results/<phase>/`: metrics files plus `figures/`, overwritten on rerun (no `_v2` copies)
  - `research/`: researcher notes
- Random seed 42 everywhere. Log runtime and peak RAM for every data and training script.
- Charts: one message per chart, and the title states the finding. Readable at phone size, a single y-axis, colour-blind-safe colours, lines labelled directly where possible.
