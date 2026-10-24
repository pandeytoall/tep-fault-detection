# Phase 1: labels, run split, variable map and the shared scorer

Goal: the ground rules every later model uses: sample labels, one seed-42 split of run IDs into jobs, the 52 variables mapped to plant units, a check that faults 1-20 are IDV(1)-(20), and `src/evaluate.py` (FDR, FAR, detection delay, scorecard) proven on dummy models.

Rules: the `tep-data` skill, except where this spec changes the run budget (see `.scratch/phase-1/run-pairing.md`). Terms: `CONTEXT.md`. Delay and limit rules: `docs/adr/0002`. Flag any other conflict.

## Inputs
- `data/processed/*.parquet` (phase 0). Read only the runs you need with pyarrow filters; never load a whole faulty file.
- `src/data.py`, `scripts/prepare_data.py`, `scripts/plot_fault1.py`, `tests/test_data.py` (phase 0 code).
- `research/tep-flowsheet.md`: units, streams and the xmeas/xmv tables (Unit / stream columns), and the IDV table.
- `.scratch/phase-1/run-pairing.md`: why run IDs must not be reused across jobs.

## Deliverables

### 1. Reshape `src/` (refactors deferred from the phase 0 review)
Split `src/data.py` by job. Keep behaviour identical and the phase 0 tests passing (update their imports only).
- `src/data.py`: paths, `FILES`, `ONSET`, `onset_for`, `ID_COLUMNS`, `VARIABLES` (the 52 column names), `load_run`, plus:
  - `load_runs(name, runs, faults=None)`: those run IDs (and faults) from one parquet file, float32, via pyarrow filters.
  - `labels(frame, name)`: int8 array, 1 only if `faultNumber > 0` and `sample > onset_for(name)`.
- `src/prepare.py`: download, md5, convert, the memory guard, machine label, `is_up_to_date` (everything `scripts/prepare_data.py` uses).
- `src/variables.py`: `class Variable(NamedTuple): tag, name, unit, plant_unit, kind`. `kind` is `measured` (xmeas 1-22), `analyzer` (xmeas 23-41) or `manipulated` (xmv 1-11). `plant_unit` comes from the flowsheet tables, using one of: `feed`, `reactor`, `condenser`, `separator`, `compressor`, `stripper`, `purge`, `product`. Use the closest unit when a row names a stream. `VARIABLE_INFO: dict[str, Variable]` replaces the old tuple dict.
- `src/signatures.py`: `rank_shifts`, `peak_after_onset` (moved), plus the IDV check (deliverable 4).
- `src/style.py`: shared chart colours (Okabe-Ito: normal blue `#0072B2`, fault vermillion `#D55E00`, plus the rest of the palette). `scripts/plot_fault1.py` uses it, and its output chart must not change.

### 2. Run split: `src/split.py` -> `results/phase-1/run_split.json`
- One partition of run IDs 1-500 with `numpy.random.default_rng(42)`. The same IDs apply to every training file and every fault:
  - `fit` 330: unsupervised fit (normal runs)
  - `limit` 100: alarm limits (normal runs). `limit_tune` (50 of them, also chosen by the seed) also tunes the supervised model
  - `supervised` 50: supervised training (faulty runs per fault, plus normal runs with these IDs)
  - `dev` 20: development and tuning (faulty and normal runs)
- Testing files are not split: all 500 IDs, scored once, in later phases only.
- `make_split()`, `save_split(path)`, `load_split(path)`, `ids(group)`. The JSON stores the seed, the group sizes and the sorted IDs.
- A real-data test (skipped if the parquet is missing) proves the pairing rule. For 3 seeded run IDs and faults 1, 3 and 9, the faulty training run equals the normal training run on samples 1-20. Also, no FaultFree_Testing run equals any FaultFree_Training run on samples 1-20 (all 500 x 500 pairs, exact byte match).

### 3. Variable map
- `src/variables.py` as above. Test: 52 entries, tags match `VARIABLES`, every `plant_unit` is from the allowed list, and spot checks (`xmeas_7` reactor, `xmeas_15` stripper, `xmv_10` reactor). Cite the flowsheet rows in a module comment. Where the flowsheet marks a row UNVERIFIED, keep the unit and add a short comment.

### 4. IDV numbering check: `scripts/check_idv.py` -> `results/phase-1/idv_check.csv`
- On the 20 `dev` runs of `TEP_Faulty_Training`, run `rank_shifts` (after onset vs samples 1-20) and record whether the expected variable is in that run's top 3:
  - fault 4 (reactor cooling water inlet temperature step) -> `xmv_10` reactor cooling water flow
  - fault 5 (condenser cooling water inlet temperature step) -> `xmv_11` condenser cooling water flow
  - fault 6 (A feed loss) -> `xmeas_1` A feed
  - fault 7 (C header pressure loss) -> `xmeas_4` A and C feed
- CSV columns: `fault, expected_tag, expected_name, share_runs_in_top3, most_common_top1`. Pass if every share is at least 0.8.
- These expected signatures are hypotheses from the IDV descriptions. If one fails, report it with the top-1 variable. **Don't change the expectations to pass.**

### 5. Shared scorer: `src/evaluate.py`
Inputs are plain per-sample frames with columns `faultNumber, simulationRun, sample, score`, plus the alarm limit, the onset and `skip` (the lag L; samples with `sample <= skip` are dropped from every count). An alarm is `score > limit`. "In a row" means consecutive `sample` values within one run; it never spans two runs. Functions:
- `limit_for_far(normal_scores, far_target)`: the (1 - target) quantile of held-back normal scores (`np.quantile`, `method="higher"`). It raises an error on NaN.
- `fdr(frame, limit, onset, skip=0)`: per fault, the share of samples with `sample > onset` that alarm, pooled over runs.
- `far(frame, limit, skip=0)`: the share of all (non-skipped) samples of fault-free runs that alarm.
- `detection_delay(frame, limit, onset, skip=0)`: per fault, the first 3 alarms in a row lying entirely after onset (all three samples `> onset`). Delay in minutes = (sample of the third alarm - onset) x 3. It returns the median over detected runs and the share of runs never detected.
- `sustained_rate(frame, limit, after, skip=0)`: the share of fault-free runs with at least one 3-in-a-row whose three samples are all `> after` (the delay baseline).
- `write_scorecard(rows, path)`: long format with columns `model, track, limit_name, limit_value, fault, metric, value, n_runs, machine, split_file, written_at`. `fault` is empty for overall rows. Per-fault metrics are `fdr`, `delay_median_min` and `never_detected_share`. Overall metrics are `mean_fdr`, `mean_fdr_excl_3_9_15`, `far`, `sustained_rate`, `train_time_s` and `peak_ram_gb`. Rerunning a model replaces that model's rows (match on `model` and `limit_name`) and leaves other models untouched. Default path `results/scorecard.csv`.

Synthetic dummy-model tests (no data files):
- Always alarm: FDR 1, FAR 1, delay 9 min.
- Never alarm: FDR 0, FAR 0, never-detected share 1, delay NaN.
- Oracle (alarms exactly when label = 1): FDR 1, FAR 0, delay 9 min, never-detected 0.
- Oracle delayed by k samples: FDR = (post - k) / post, delay = (k + 3) x 3 min.
- A 3-in-a-row that starts at or before onset doesn't count. Pre-onset alarms don't change FDR.
- `skip=L` removes the first L samples of each run from FDR, FAR and the sustained rate.
- Alarms that are 3 in a row only across two runs (the last 2 samples of run A, then the first sample of run B) are not a sustained alarm.
- Random scores, alarm probability 0.05, about 200k samples: FAR, and FDR pooled over all faults, within 0.05 ± 0.005. Don't check this per fault.
- Sustained rate on three fixed fault-free runs: one with a 3-in-a-row fully after `after`, one straddling `after`, one with only 2 in a row. The rate is 1/3.
- `limit_for_far` on uniform scores gives the expected quantile. Then FAR on fresh uniform scores is within ±0.003.
- `write_scorecard` replaces only the matching model's rows.

### 6. End-to-end dummy run: `scripts/dummy_scorecard.py` -> `results/phase-1/scorecard-dummy.csv`
Training family only; the testing files are not touched in this phase.
- Model `dummy-maxz`: the per-variable mean and std from the `fit` normal runs. The score is the max |z| over the 52 variables, dropping any variable with zero std on the `fit` runs (list them in the build report). Limits come from the `limit` normal runs at 1% and 5% FAR.
- Score the `dev` runs: faulty (all 20 faults) and normal (FAR). Write the rows to `results/phase-1/scorecard-dummy.csv`, never to `results/scorecard.csv`.
- Log runtime and peak RAM (psutil, process tree) in the build report and the CSV rows.

## Acceptance criteria
- `.venv/Scripts/python -m pytest -q` passes: the 43 existing tests (imports updated only) plus the new ones.
- `run_split.json`: sizes 330/100/50/20, pairwise disjoint, union = 1..500, `limit_tune` is a subset of `limit` with 50 IDs, and `make_split()` is identical on two calls.
- `scripts/plot_fault1.py` still produces the same chart (same top-3 printout).
- `idv_check.csv` exists; each share is reported, and any below 0.8 is flagged in the build report with its top-1 variable.
- Dummy run at the 1% limit: FAR on dev normal runs between 0.3% and 3%. Fault 6 FDR above 0.9. Faults 3, 9 and 15 FDR below 0.10. **If 3/9/15 come out higher, stop and report; don't tune.**
- Peak RAM under 5 GB for every script.
- No hostnames, usernames or local paths in tracked files. `machine` uses the existing CPU + RAM label.
- One commit: `phase 1: labels, run split, variable map, shared scorer`.

## Out of scope
- Any real model (PCA, DPCA and so on), scalers beyond the dummy's mean/std, the lag window choice, and reading the testing files.
- The analyzer hold-value check (do xmeas 23-41 repeat between samples?). It goes in the backlog for phase 2.
- Changes to `status.py`, hooks, agent or skill files, `CONTEXT.md` and ADRs (the orchestrator updates those).

## Comments
- 2026-10-22, orchestrator: run budget changed from 400/100 normal + 50/20 faulty (reused IDs) to one disjoint partition 330/100/50/20, because run k shares its noise across all files in a family. User chose this option. Evidence: `run-pairing.md`.
- 2026-10-22, critic (opus, xhigh): AGREE WITH CHANGES. Applied: the rules in the `tep-data` skill (no ID in two jobs, paired-difference ban, which group each later job uses), a cross-family test over all pairs, `skip` in `sustained_rate`, the run-boundary test, pooled random-score checks, the fixed-run sustained test, metric names, NaN guard, zero-std drop. The cross-family check was run before the build: 0 of 500 x 500 shared.
