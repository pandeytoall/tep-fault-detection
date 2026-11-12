# Phase 4: Diagnosis (contributions per fault, and why faults 3, 9 and 15 hide)

Goal: for each fault, find which readings the PCA monitor blames, with real names, and check the blame against the plant unit the fault belongs to. Compare three ways of assigning blame, and measure how little faults 3, 9 and 15 move the readings. Everything is offline analysis; no new alarm model and no assistant change.

Rules: the `tep-data` skill wins on any conflict; flag it if so. Terms: `CONTEXT.md`. ADRs: `docs/adr/0001`-`0004`. Code to reuse: `src/contributions.py` (SPE/T² contributions, relative form, `normal_reference`), `src/mspc.py` (`PCAMonitor`, `lagged_rows`), `src/models.py` (the saved `hgb-l2` for permutation importance), `src/variables.py` (name, unit, plant unit), `src/cases.py` (IDV → scoring unit table), `src/evaluate.py`, `src/split.py`, `src/data.py`, `src/style.py`, `src/prepare.py`. Models: `results/phase-2/pca.npz`, `data/models/hgb-l2.joblib`.

## User decisions (2026-11-12)
- Methods: PCA SPE and T² contributions (relative), **RBC** (reconstruction-based contribution), and the supervised model's permutation importance as a third view.
- Faults 3/9/15: **within-run before/after** comparison only. The paired-difference (faulty minus normal run k) is NOT used, even as a diagnostic.

## Fixed choices (decided before any results; don't tune them)
- **Scaling and model:** the saved PCA model (31 PCs, L = 0). Normal reference for relative contributions: mean contribution on the `fit` normal runs, as in `src/contributions.py`, with the same 1% floor. Reuse the stored reference in `pca.npz`.
- **RBC** (Alcala & Qin 2011; PCA, scaled space, x = scaled sample):
  - SPE: with C̃ = I − P Pᵀ, rbc_j = (ξ_jᵀ C̃ x)² / (ξ_jᵀ C̃ ξ_j), where ξ_j is the j-th unit vector. Skip a reading whose denominator is below 1e-9.
  - T²: with D = P Λ⁻¹ Pᵀ, rbc_j = (ξ_jᵀ D x)² / (ξ_jᵀ D ξ_j).
  - Same relative form as the others: window mean divided by the `fit` normal mean of that reading's RBC (floor 1% of the mean over readings).
  - Test it against a hand-computed 3-variable example. Note in the code docstring that RBC values do not sum to the statistic, unlike the complete decomposition.
- **Permutation importance (supervised view):** `sklearn.inspection.permutation_importance` on `hgb-l2`, scoring = accuracy of the 21-class prediction on the **dev** faulty runs' post-onset samples (`n_repeats=5`, `random_state=42`, `n_jobs` ≤ 6). It is per reading, over the 156 lagged inputs, summed over the 3 lags of each reading. One importance vector per fault (permute, then score only that fault's samples). Dev only; no testing data.
- **Windows:** the post-onset samples of a run (the offline analysis knows the onset; the assistant doesn't), plus the first 20 post-onset samples (the first hour) as a second window. `skip` is not needed at L = 0.
- **Runs:** development on the `dev` runs only (20 faulty runs per fault, training family). One final pass over the testing files reads, per fault, the first 100 runs of a seed-42 choice of the 500 (the same 100 IDs for every fault, saved with the results) and each run's post-onset window; nothing in it fits or picks.
- **Expected unit per fault:** the IDV → scoring unit table in `src/cases.py` (feeds, reactor, condenser/separator, compressor/recycle, stripper; 16-20 and normal have none). Map each reading to a scoring unit with the 8 → 5 mapping in `src/variables.py`.
- **Blame metrics, per fault and method** (SPE, T², RBC-SPE, RBC-T²; and permutation importance for the supervised view):
  - `top1_hit`: share of runs whose top reading is in the expected scoring unit.
  - `top5_hit`: share of runs with at least one top-5 reading in the expected unit.
  - `unit_mass`: share of the (clipped at zero, summed) relative contribution that falls in the expected unit.
  - `smear`: the share of total relative contribution held by the top 5 readings (high = concentrated, low = smeared).
  - Faults 16-20: no expected unit, so report only `smear` and the top 3 reading names, labelled "unknown cause".
  - Faults 3/9/15: report them with the others, labelled "near the false-alarm level".
  - Also report the baseline: the share of readings in the expected unit (so a method that picks readings at random would score `top5_hit` ≈ 1 − (1 − that share)^5 and `unit_mass` ≈ that share). Print it next to every method's numbers.
- **Why 3/9/15 hide (within-run before/after, dev runs and then the test runs):** per run, per reading, compare the post-onset window to the pre-onset window of the same run:
  - standardised mean shift = (mean_post − mean_pre) / sd of the reading on the `fit` normal runs;
  - spread ratio = sd_post / sd_pre;
  - change in the lag-1 autocorrelation.
  - Per fault: the median over runs of the largest |shift| over readings and how many readings exceed 1 sd; the same numbers for the benchmark faults 1, 2, 4, 6, 7, and for a "no fault" control: the same statistics on fault-free runs, pre/post the same sample number (20 on dev, 160 on test). The control gives the noise floor.
  - Add a plain verdict per fault: effect size (largest median |shift|) relative to the control's.
  - Faults 3/9/15 are expected to look like the control. If any of them shows a clear shift (more than 3× the control's largest), stop and report it: it would mean the monitors miss a real signal or the data differs from what we assumed.
- **Charts** (`results/phase-4/figures/`, `tep-data` style, helper in `src/style.py`; one finding per chart, the title states it):
  1. `blame-heatmap.png`: faults 1-20 (rows) × 52 readings (columns, grouped by plant unit and labelled with real names), colour = median relative SPE contribution (log scale), expected-unit readings outlined. A single colour scale.
  2. `blame-methods.png`: `top5_hit` per fault for SPE, T², RBC-SPE, permutation importance, with the random baseline. Lines or dots, labelled directly.
  3. `hidden-faults.png`: largest median |shift| per fault (dev), with 3/9/15 and the fault-free control highlighted.
- **Operator-style report for one fault** is phase 5; don't write it. But save per-fault tables `results/phase-4/top_readings_<fault>.csv` (top 10 readings with names, units, plant unit, relative contribution for the methods).
- Faults 3/9/15 must not appear as "detected" anywhere: if a top-reading table is printed for them, add the label "near the false-alarm level; blame is mostly noise".

## Deliverables
1. `src/diagnosis.py`: RBC (SPE, T²), per-run blame vectors, the four blame metrics, the permutation importance wrapper, and the before/after statistics. Keep it small; reuse `contributions.py`.
2. `scripts/diagnose_dev.py`: all analysis on the `dev` runs. Writes `results/phase-4/dev_blame.csv` (fault × method × metric), `dev_hidden.csv` (before/after stats) and the top-readings tables.
3. `scripts/diagnose_final.py`: the one pass over the 100 testing runs per fault. Writes `results/phase-4/test_blame.csv`, `test_hidden.csv`, `test_runs.json` (the run IDs), and the three charts. Records peak RAM, runtime and machine in every file.
4. Tests (synthetic data, no parquet):
   - RBC against a hand-computed example, and its zero-denominator guard;
   - the blame metrics on a hand-made case (hand counts, including the random baseline);
   - before/after statistics on a synthetic run with a known shift;
   - permutation importance sums 3 lags per reading;
   - the expected-unit lookup agrees with `cases.py`.
5. Carry-over cleanup, own commit `phase 4: phase 3 review cleanup`: a `machine` column in `fdr_by_fault.csv` and `unseen_fault.csv`; a test for `tune_hgb` (a tiny synthetic fit: the limit comes from `limit_tune` and the dev scoring uses dev runs only); share the core of the two `final_check_failures`; one `SPLIT_FILE`/`OUT` constant per module; move `MODELS_DIR` and `RESULTS_3` in `src/data.py` next to the other path constants. Keep results unchanged: rerun `ml_final.py --charts-only` or recompute only what the machine column needs, and confirm the scorecard values are identical.

## Run order and stops
1. Build and test. Run dev. Check the dev sanity items below before the final pass.
2. Run the final pass, then the charts. Commit after each step.
3. Stop early only for something critical: a sanity item failing, the 3/9/15 shift rule above, a script over 3 h or 5 GB that subsampling can't fix.

## Acceptance criteria
- `.venv/Scripts/python -m pytest -q` passes (349 existing plus the new ones).
- Grep: `_Testing` is read only in `mspc_final.py`, `ml_final.py`, `diagnose_final.py`, `cases.py` and the phase 1 pairing test. `diagnose_final.py` fits and picks nothing.
- Dev sanity (print them; if one fails, explain before going on):
  - Fault 6: an A-feed reading (`xmeas_1` or `xmv_3`) is the top SPE reading in at least 80% of runs.
  - Fault 4: the top readings include a reactor cooling reading (`xmeas_21` or `xmv_10`) in at least half the runs.
  - Faults 1, 2, 7: `top5_hit` for SPE is above the random baseline.
  - For every method, the relative contribution of each reading on fault-free dev runs is about 1 (median over readings between 0.5 and 2).
- Faults 3/9/15 (dev and test): the largest median |shift| is at most 3× the fault-free control's. Otherwise stop and report.
- Peak RAM under 5 GB for every script. Machine recorded in every results file.
- A short `results/phase-4/README.md`: what each file holds, the one-line meaning of each metric, the answer "which method blames the right unit best", and the sentence on why 3/9/15 hide, in plain words.

## Out of scope
- DPCA or other model contributions; SHAP; new alarm models; assistant changes; any test-set rebuild.
- The operator-style report and the comparison report (phase 5). Posts. Pushing (the orchestrator pushes after review).

## Comments
- 2026-11-12, orchestrator: /ml-review **PASS** (no blockers; 383 tests; scorecard byte-identical after the cleanup). Checked by me: tests, `_Testing` reads only in the allowed scripts, the hidden-fault table and two of the three charts. Reviewer verified: RBC formulas, the RBC-SPE = SPE algebra, blame metrics, control runs share no noise stream with the faulty runs, the same 100 test IDs per fault.
  - **Failed dev sanity items, accepted with this explanation (nothing was tuned):** (1) fault 6: an A-feed reading is the top SPE reading in only 45% of post-onset runs (needed 80%) but in 100% of runs in the first hour. After the A feed is lost, SPE grows from about 117 to about 25,000, so the whole-window mean is dominated by downstream readings; the window, not the method, is the cause. (2) fault 2: SPE `top5_hit` is 0.00 against a random baseline of 0.57; every method scores 0 on this fault, and the monitors blame downstream readings (`xmv_11`, `xmeas_17`, `xmeas_19`) instead of the stream 4 composition. Both are results about contribution plots (they show symptoms, not causes), not a contradiction of the plan.
  - **Read the results this way:** RBC-SPE is a duplicate of SPE in the relative form (the reconstruction denominator is constant per reading and cancels); only the 1% floor separates them. SPE beats random by 14 points on `top5_hit` (75% vs 61%), T² does not. The permutation-importance comparison is coarse: one dev vector per fault, every 5th post-onset row, so its bars are 0 or 1. The "mean shift" check does not separate faults 14 and 19 from fault-free (their signature is in spread and autocorrelation); for 3/9/15 all three statistics sit at the control level.
  - **Deviations accepted:** the permutation step took 8 h (estimate 1.7 h; over the 3 h guide) and was not rerun; the dev run had no time guard. Chart 2 mixes test runs (PCA views) and dev runs (permutation); labelled in the legend.
  - Carried to phase 5 (minor): add a footnote on `blame-methods.png` that permutation is one dev vector per fault, and drop "test runs" from its title; share the print blocks of `diagnose_dev.py` and `diagnose_final.py` in `src/`; drop the unused `windows` parameter in `blame_rows`; give long scripts a time guard; correct "expected plant unit" to "expected scoring unit" in the README; remove the unused "paired" statistic print in `diagnose_dev.py`.
