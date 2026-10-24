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
- Run budget (fixed seed-42 run IDs, saved with the results):
  - Run k of the fault-free file and of every fault share one noise stream within a family (training or testing). See `.scratch/phase-1/run-pairing.md`. So one seed-42 partition of IDs 1-500 applies to every training file and fault, in `results/phase-1/run_split.json`: `fit` 330, `limit` 100 (`limit_tune` 50 inside it), `supervised` 50, `dev` 20.
  - No model uses one run ID (any file, any fault) in two jobs. A paired difference (faulty run k minus normal run k) is never a model input, in either family.
  - `fit`: unsupervised fit. The lag choice and autoencoder validation also use `fit` runs, split by run.
  - `limit`: alarm limits only. `supervised`: supervised training (faulty runs plus the normal runs with these IDs). Supervised tuning uses faulty `dev` runs plus normal `dev` and `limit_tune` runs. The unseen-fault model reuses the full model's hyperparameters (no separate tuning).
  - Testing files: all runs, once, for the final scorecard. No testing run matches any training run (checked over all 500 x 500 pairs).
- Alarm limits: set on the 100 `limit` normal runs at two FAR targets, 1% (headline, used for ranking) and 5% (secondary). MSPC also reports the textbook 99% limit.
- Two tracks, separate tables, never ranked together: unsupervised (PCA, DPCA, Isolation Forest, autoencoder; normal runs only) and supervised (HistGradientBoosting, 21 classes, alarm score = 1 - P(normal), built-in early stopping off).
- Lag window L: one value for DPCA and all lagged models, chosen on normal training runs only with the Ku et al. (1995) rule from L = 0-5. Lags look back within one run; drop each run's first L samples from all counts. PCA uses no lags.
- Unseen-fault test: retrain the supervised model without faults 1, 8, 13, 14, 16; compare with the full supervised model and PCA on those faults.
- Report per fault: FDR (share of post-onset samples flagged, per sample) and detection delay (onset to the third alarm of the first 3-in-a-row after onset; median over detected runs, plus the share never detected, next to the 3-in-a-row rate on fault-free test runs after sample 160). Overall: mean FDR with and without faults 3, 9, 15; FAR on `TEP_FaultFree_Testing`; train time; peak RAM. All rows go in `results/scorecard.csv`.
- Headline metrics for all models come from one machine (laptop or cloud); every results file records the machine. Subsets are a seed-42 choice of `simulationRun` IDs, saved with the results.
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
