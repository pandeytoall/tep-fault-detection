# Phase 2: MSPC (PCA and DPCA)

Goal: PCA and DPCA monitors fitted on the `fit` normal runs. Each gives T² and SPE alarm scores, with alarm limits from the `limit` runs at 1% and 5% FAR, plus the textbook 99% limit. They are developed on `dev` runs, scored once on the testing files, and written to `results/scorecard.csv`.

Rules: the `tep-data` skill (run groups, limits, lags, delay, tracks). It overrides this spec on any conflict; flag it if so. Terms: `CONTEXT.md`. Decisions: `docs/adr/0001`-`0003`.

## Inputs
- `data/processed/*.parquet`; `results/phase-1/run_split.json` via `src/split.py`.
- `src/data.py` (`load_runs`, `labels`), `src/evaluate.py` (`limit_for_far`, `model_rows`, `write_scorecard`), `src/prepare.py` (machine label, peak RAM), `src/style.py`.
- `research/pca-anchors.md`: the Rato & Reis (2013) Table 2 anchors and the Ku et al. (1995) lag procedure.

## Fixed choices (decided before any results; don't tune them)
- **Scaling:** mean and std per variable from the `fit` normal runs only. The same scaler is used for every later scoring.
- **PC count:** the fewest PCs whose cumulative explained variance is at least 90% on the scaled `fit` runs (user decision, 2026-10-24). The same rule applies to PCA and to DPCA (on the lagged matrix).
- **Lag window L:** Ku et al. (1995) on the `fit` runs, L = 0-5, with a(l) from the 90% rule. Follow the steps in `research/pca-anchors.md`. Use r_new(l) = r(l) − Σ_{i<l} (l−i+1)·r_new(i). Stop at the first l with r_new(l) ≤ 0 and use L = l−1. Print a(l), r(l) and r_new(l) per l for both the 90% rule and parallel analysis (diagnostic only; Ku's own choice), and save them to `results/phase-2/lag_choice.csv`. **L comes from the 90% rule. If the two rules give different L, if the 90% rule gives L = 0, or if it never stops by l = 5, stop after `lag_choice.csv` and report it before fitting.** The PCA fit, the PC count and the lag choice use all fit rows a lagged matrix allows; `skip` applies only to counting. Lagged rows are built within one run only.
- **Skip:** the first L samples of every run are dropped from all counts, for PCA and for DPCA (`skip=L`), so both are counted on the same samples.
- **Statistics:** T² = Σ t_i²/λ_i over the retained PCs. SPE = squared residual norm. Each is its own scorecard model: `pca-t2`, `pca-spe`, `dpca-t2`, `dpca-spe`, all `track=unsupervised`.
- **Limits:**
  - `1%` and `5%`: `limit_for_far` on the `limit` normal runs (all 100; `skip` applied).
  - `99% textbook`: T² from the F-distribution, a(n²−1)/(n(n−a)) · F(a, n−a) at 0.99, with n = the number of rows in the (lagged) fit matrix. SPE from Jackson-Mudholkar at 0.99: θk = Σ over discarded PCs of λ_j^k (k = 1, 2, 3), h0 = 1 − 2θ1θ3/(3θ2²), and Q_lim = θ1·[c·√(2θ2h0²)/θ1 + 1 + θ2h0(h0−1)/θ1²]^(1/h0), where c is the 0.99 normal quantile. Test it against a hand-computed example.

## Deliverables
1. `src/mspc.py`: `class PCAMonitor(lags=0, cpv=0.90)`.
   - `fit(frame)`: normal runs.
   - `score(frame)`: returns a frame of `faultNumber, simulationRun, sample, t2, spe` (lagged rows only).
   - `textbook_limits(alpha=0.99)`.
   - `save(path)` / `load(path)`: an `.npz` of the scaler, loadings, eigenvalues, lags and the PC count. Phase 2b reuses it.
   - Plus `ku_lag(frame, max_lag=5)`, which returns L and the r tables.
2. `scripts/mspc_dev.py`: development on the training family only.
   - Choose L, then fit PCA and DPCA, then set limits from `limit`.
   - Score `dev` faulty runs (all 20 faults) and `dev` normal runs.
   - Write `results/phase-2/dev_scores.csv` (the `model_rows` long format) and `lag_choice.csv`.
   - Save the models to `results/phase-2/pca.npz` and `dpca.npz`.
   - Print the PC counts, L, the dev FAR, and the dev FDR for faults 1, 2, 4, 6, 7, 3, 9 and 15.
3. `scripts/mspc_final.py`: one final scoring of the testing files.
   - Load the saved models and limits; never refit here.
   - Score all 500 FaultFree_Testing runs (FAR, plus `sustained_rate` with `after=160`) and all 500 runs × 20 faults of Faulty_Testing, read one fault at a time.
   - Write the 4 models × 3 limits to `results/scorecard.csv` via `write_scorecard`, with train time (fit only, `time.perf_counter`), peak RAM and machine.
   - Write `results/phase-2/fdr_by_fault.csv` (a wide view: fault × model at the 1% limit, plus FAR) and `results/phase-2/vs_literature.csv` (our 1%-limit FDR next to the Rato & Reis numbers for PCA T²/Q and DPCA T²/Q, faults 1-20).
4. Two charts in `results/phase-2/figures/`, following the `tep-data` chart style with colours from `src/style.py`:
   - `fdr-by-fault.png`: test FDR per fault at the 1% limit, PCA T² vs PCA SPE. The title states the main finding.
   - `pc-sweep-dev.png`: dev mean FDR at the 1% limit against the PC count (5-30) for PCA SPE and T². It's a side chart; the 90% choice is marked, and it is never used to pick.
5. Tests (synthetic data, no parquet):
   - The PC-count rule.
   - T² and SPE against a hand-computed 3-variable example.
   - Lagged rows never cross runs, and the first L samples of each run are dropped.
   - `ku_lag` on a synthetic AR(1) system with a known answer.
   - The textbook limits against scipy reference values.
   - `save`/`load` round-trip giving identical scores.
6. Phase 1 cleanup (review minors), in its own commit `phase 2: phase 1 review cleanup`:
   - Derive `MINUTES_PER_SAMPLE` from `data.HOURS_PER_SAMPLE`.
   - Share the process-tree loop between `prepare.peak_ram_gb` and `_tree_usage`.
   - Give the scripts one small helper for the runtime/RAM line.

## Acceptance criteria
- `.venv/Scripts/python -m pytest -q` passes (102 existing tests plus the new ones).
- `mspc_dev.py` runs before `mspc_final.py`. The final script reads the testing files once, and nothing in it fits or picks anything. Grep shows no other code path reading `_Testing` beyond the phase 1 pairing test.
- Test FAR at the `1%` limit is between 0.5% and 2% for all four models, and at `5%` between 2.5% and 10%. Otherwise, explain it before calling it good.
- Plausibility against Rato & Reis at 1% FAR: faults 1, 2, 6 and 7 have FDR at least 0.95 for SPE. Fault 4 SPE is at least 0.9, and fault 4 T² is below 0.5 and below fault 4 SPE (literature 0.17-0.22). A miss caused by our larger PC count gets explained, not tuned.
- **At the `1%` limit, faults 3, 9 and 15 have FDR at most 0.10 for every model. At `5%` and `99% textbook`, their FDR is at most the measured test FAR + 0.05. If any is higher, stop and report; it suggests a leak.**
- Every scorecard row carries `machine`, `split_file`, train time and peak RAM. Peak RAM is under 5 GB for every script.
- Commits: the cleanup commit first, then `phase 2: PCA and DPCA monitors`.

## Out of scope
- The contribution plots (phase 4) and the MCP assistant (phase 2b).
- Combining T² and SPE into one alarm.
- Other PC rules, apart from the side sweep.
- Any change to the split, labels or scorer rules.
- Changes to `status.py`, hooks, agent or skill files, `CONTEXT.md` and ADRs.

## Comments
- 2026-10-24, user: PC count = 90% cumulative variance on the `fit` runs. Rato & Reis used parallel analysis (17 PCs; DPCA 29 PCs, 3 lags), so expect different PC counts. Their data is the original Braatz set, not Rieth's.
- 2026-10-24, critic (sonnet, medium): AGREE WITH CHANGES. Applied: the 3/9/15 leak bound depends on the limit; fault 4 T² < 0.5; stop when L = 0 or there is no stop by l = 5; n = lagged fit rows; the SPE formula is written out; the cleanup is its own commit.
- 2026-10-25, user: L = 2 from the 90% rule (pre-agreed), accepted although parallel analysis gave L = 0. Caveats, recorded here and in the results: the 90% stop is borderline (r_new(2) = +1, r_new(3) = −1); the iid-noise parallel analysis at 165k rows behaves like "eigenvalue > 1" and is inflated by autocorrelation. The literature uses 2-3 lags on TEP (Rato & Reis: 3). The stop rule (a) is satisfied by this decision; the rest of the spec is unchanged.
- 2026-10-25, orchestrator: /ml-review PASS (tagged phase-2). Trivial fix during review: `src/mspc.py` imports `ID_COLUMNS` from `src.data`. Carried to phase 2b as cleanup:
  - a shared figure helper in `src/style.py` (chart boilerplate is duplicated across the scripts);
  - move the final-script plausibility, FAR and leak checks into a tested function, replacing the plot asserts;
  - store the limits with the models (.npz), not read from `dev_scores.csv`;
  - a one-line docstring note on the uncentred second moment;
  - drop unused `parallel_n_components` params.
  Results findings: the textbook SPE limit for DPCA is loose (test FAR 4.9%; textbook 25.1 vs empirical 1% limit 30.7, heavy-tailed lagged residuals). DPCA T² alarms in runs, with a fault-free 3-in-a-row rate of 0.50 at the 1% limit.
