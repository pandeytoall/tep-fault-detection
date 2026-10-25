# Phase 3: ML models (Isolation Forest, autoencoder, gradient boosting, unseen-fault test) and assistant v1.1

Goal: three ML monitors, each fitted on training runs only. Their alarm limits are set on the `limit` runs. They're scored once on the testing files and written to `results/scorecard.csv` beside PCA and DPCA. The unseen-fault test is run, then the new models are added as assistant tools and the frozen 70 cases are re-scored once (v1.1).

Rules: the `tep-data` skill wins on any conflict; flag it if so. Terms: `CONTEXT.md`. ADRs: `docs/adr/0001`-`0004`. Phase 2 code to reuse: `src/mspc.py` (`lagged_rows`, `PCAMonitor` scaler, `scorecard_rows`, `final_check_failures`), `src/evaluate.py`, `src/split.py`, `src/data.py`, `src/style.py`, `src/prepare.py`. Phase 2b code: `src/assistant/`, `src/cases.py`, `src/historian.py`, `scripts/assistant_*.py`.

## User decisions (2026-10-25)
- Autoencoder: scikit-learn `MLPRegressor`, no PyTorch.
- Isolation Forest and the autoencoder each run at L = 0 and at L = 2 (the agreed L; same lag rule as DPCA).
- Assistant: one rerun of the frozen 70 test cases with all new tools (v1.1).

## Fixed choices (decided before any results; don't tune them unless stated)
- **Scaling:** the PCA scaler (mean and std from the `fit` normal runs) for every model.
- **Lags:** `lagged_rows` with L = 0 or 2, built inside runs only. `skip = L` in all counts. Gradient boosting uses L = 2 only (the agreed lag for lagged models).
- **Isolation Forest** (`iforest-l0`, `iforest-l2`; unsupervised track):
  - `IsolationForest(n_estimators=200, max_samples=256, random_state=42, n_jobs=6)`
  - fitted on 100,000 rows drawn with seed 42 from the `fit` normal runs
  - alarm score = `-score_samples`
  - no tuning
- **Autoencoder** (`ae-l0`, `ae-l2`; unsupervised track):
  - Network: `MLPRegressor` mapping X to X, hidden layers (2b, b, 2b), with b = the PCA PC count at that lag (31 at L = 0, 68 at L = 2). ReLU, Adam, `random_state=42`, batch 256. Train on float32 data.
  - Validation: 30 of the 330 `fit` runs (seed 42), by run.
  - Training: train on the other 300 with `partial_fit`, one epoch at a time. Stop when the validation MSE hasn't improved for 5 epochs (max 100), and keep the best epoch's weights.
  - sklearn's built-in `early_stopping` splits by rows, which breaks the run-split rule, so don't use it.
  - Alarm score: reconstruction MSE per row. If one full fit takes over 30 min or 5 GB, subsample rows by run and record it.
- **Gradient boosting** (`hgb-l2`; supervised track):
  - `HistGradientBoostingClassifier`, 21 classes, `early_stopping=False`, `random_state=42`.
  - Training data: the `supervised` runs of all 20 faults plus the `supervised` normal runs. Labels follow the onset rule: pre-onset samples of faulty runs are class 0.
  - Alarm score: 1 − P(normal).
  - **Tuning** over learning_rate {0.05, 0.1} × max_leaf_nodes {31, 63} × max_iter {200, 400}, 8 fits.
    - Each fit gets a 1% limit from the `limit_tune` normal runs.
    - Each fit is scored on the `dev` faulty runs plus the `dev` normal runs.
    - Pick the highest dev mean FDR excluding 3/9/15. On a tie within 0.005, pick the fewer total trees.
    - Save the grid to `results/phase-3/hgb_tuning.csv`.
  - The chosen fit is the final model, with no refit on more data.
- **Limits:** for every model, 1% and 5% from `limit_for_far` on all 100 `limit` normal runs. ML models have no textbook limit. Limits are stored with each model.
- **Unseen-fault test:**
  - Retrain `hgb-l2` without faults 1, 8, 13, 14 and 16 (their runs left out entirely), with the chosen hyperparameters. Model name: `hgb-l2-unseen`.
  - Its 1% limit comes from the `limit` runs.
  - Report test FDR on those 5 faults for `hgb-l2-unseen`, `hgb-l2` and `pca-spe`. Write it to `results/phase-3/unseen_fault.csv`. Add a scorecard row block for `hgb-l2-unseen`.
- **Model files:** saved under `data/models/` (git-ignored). Each file records its training run IDs, hyperparameters and limits. `results/phase-3/models.json` lists each file with its SHA-256, so a later phase can tell which model the assistant used.
- **Final scoring:** `scripts/ml_final.py` loads saved models only, so it never fits or picks. It reads `FaultFree_Testing` (500 runs) and `Faulty_Testing` one fault at a time. Each model gets scorecard rows at 1% and 5% via `write_scorecard`, with train time (fit only; for hgb the chosen fit only, with grid time logged separately), peak RAM and machine. It also writes `results/phase-3/fdr_by_fault.csv`: the wide view, all phase 2 and phase 3 models at 1%, plus FAR, with the track shown.
- **Charts** (`results/phase-3/figures/`, `tep-data` style, `src/style.py` helper):
  - `fdr-by-fault-unsupervised.png`: test FDR per fault at 1% for PCA SPE, the best IF and the best AE. The title states the finding.
  - `unseen-fault.png`: the 5 unseen faults, comparing `hgb-l2-unseen`, `hgb-l2` and `pca-spe`.

## Assistant v1.1
- New MCP tools, reading the same per-case historian:
  - `score_model(model, start_h, end_h, step=1)`: per-sample alarm score with its 1% limit, for model in {iforest-l0, iforest-l2, ae-l0, ae-l2, hgb-l2}. Same 2,000-value cap as `score_pca`.
  - `classify_fault(start_h=None, end_h=None)`: the `hgb-l2` class probabilities averaged over the window (default: the whole case), returned as the top 3 classes with their probabilities. Class 0 is shown as "normal".
  - The alarm log stays PCA-based. No new tables, and no case data beyond what v1 had.
- **Prompt v1.1:** `src/assistant/prompt.md` adds only a short description of the new tools; the answer block is unchanged. Develop it on the dev set (20 cases) only. Then freeze it, with its own hash, and keep the v1 hash file. Answers go to `answers_test_v1.1.jsonl` and `answers_dev_v1.1.jsonl`, so the v1 files stay untouched. The scorer takes a version argument.
- **Test rerun:** 70 cases × 1, plus the same 10 repeat cases × 3. The same < 8 of 10 upgrade rule applies.
- **Scoring:** a v1 vs v1.1 table, case by case: headline, IDV top-1 and top-3, normal, 3/9/15, 16-20, baseline and chance. Also count the cases that flipped from wrong to right and from right to wrong.
- **Leak caution, to state in the results:** `hgb-l2` is trained on labelled training-family runs. So v1.1's IDV gain comes from a supervised classifier trained on labelled plant history, which is a different claim from v1's reasoning-only diagnosis. The test cases are all testing-family runs, which share no noise with training runs (ADR 0003).

## Carry-over cleanup (phase 2b review minors), own commit `phase 3: phase 2b review cleanup`
- One hour-window filter and one tolerance constant, shared by `server.py`, `contributions.py` and `historian.py`.
- `score.py` reuses `evaluate.HARD_FAULTS`.
- One shared path constant each for `pca.npz` and `results/phase-2b`.
- Check whether the identical fault 6 top-5 SPE lists also appear on normal cases, and print one line about it.

## Deliverables
1. `src/models.py`: one class per model kind (IF, AE, HGB), with the same small interface as `PCAMonitor`: `fit`, `score` (returns ids plus an `alarm` column), `save`/`load`, and the limits stored with the model. One callable per model, so the assistant tools can use them.
2. `scripts/ml_dev.py`: fits all models on the training family, tunes hgb, sets limits, scores the `dev` runs, saves the models, and writes `results/phase-3/dev_scores.csv` and `hgb_tuning.csv`. It prints the dev FAR and the dev FDR for faults 1, 2, 4, 6, 7, 3, 9 and 15 per model.
3. `scripts/ml_final.py`: as above.
4. Assistant v1.1: tools, prompt, eval and scorer versioning, and `results/phase-2b/diagnosis_test_v1.1.csv` plus the v1-vs-v1.1 comparison in `results/phase-3/assistant_v1_vs_v1.1.csv`.
5. Tests (synthetic data where possible):
   - the run-split autoencoder validation (no run on both sides) and early-stop logic
   - the IF and AE score direction (anomalies score higher)
   - hgb labels follow the onset rule, and the alarm is 1 − P(normal)
   - the unseen-fault training set contains no rows of faults 1, 8, 13, 14, 16
   - save/load round-trips give identical scores
   - the new MCP tools see only their case and respect the cap
   - `classify_fault` returns 3 classes with probabilities summing to at most 1
   - the scorer's version handling, including a v1 vs v1.1 flip count on a hand-made file

## Run order and stops
1. Build and test. Run `ml_dev.py`, then `ml_final.py`, then the unseen-fault test.
2. Assistant v1.1: dev run, freeze the prompt, then the test run and the repeats, then score.
3. Only stop early for something critical:
   - a sanity check below fails
   - any model above 50% FDR on fault 3, 9 or 15 (leak audit)
   - an assistant session using a non-`tep` tool
   - cost per session over $0.20
   - peak RAM over 5 GB that subsampling can't fix
   - total runtime for any one script over 3 h

## Acceptance criteria
- `.venv/Scripts/python -m pytest -q` passes (283 existing tests plus the new ones).
- Grep: `_Testing` is read only in `mspc_final.py`, `ml_final.py`, `cases.py` and the phase 1 pairing test. Nothing in `ml_final.py` fits or picks anything.
- Test FAR at the 1% limit is 0.5-2% for every model, and at 5% it is 2.5-10%. Otherwise, explain it.
- Plausibility:
  - Faults 1, 2, 4, 6 and 7 have FDR of at least 0.9 for `hgb-l2`.
  - Faults 3, 9 and 15 have FDR of at most 0.10 at the 1% limit for every unsupervised model.
  - For `hgb-l2`, any FDR on 3/9/15 above 0.10 is listed and explained before it's reported. Supervised models can learn weak signatures, but more than 0.5 means stop.
- `hgb-l2-unseen` does worse than `hgb-l2` on most of the 5 unseen faults. If not, explain it.
- The assistant v1.1 test run uses the frozen v1.1 prompt hash in all sessions, with no non-`tep` tools.
- Peak RAM under 5 GB for every script. Machine recorded in every results file.

## Out of scope
- PyTorch or any GPU or deep-learning library. LSTM or CNN models. ICA. Random Forest.
- Changing the test case set, the v1 prompt or v1 results.
- RBC contributions and contribution plots (phase 4).
- Posts, pushing to GitHub.

## Comments
