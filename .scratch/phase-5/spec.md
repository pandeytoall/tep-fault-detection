# Phase 5: Reports (one comparison report, one operator-style report)

Goal: pull every phase's results into one comparison report, and write one operator-style report for fault 6 (A feed loss) as a shift handover. Both are generated from the saved results files by a script, so the numbers can't drift from the data, and both are written for engineers: short sentences, a one-line meaning for each term, what is uncertain stated next to each claim.

Rules: the `tep-data` skill wins on any conflict. Terms: `CONTEXT.md`. ADRs: `docs/adr/0001`-`0004`. Public data only: no employer, company, plant or tag names from outside the public TEP papers. Nothing here is a post; posts are drafted later by the `editor`.

## User decisions (2026-11-13)
- Phase 5 comes before assistant v2/v3.
- Operator-style report: **fault 6**, A feed loss.
- Comparison report: **Markdown + charts**, in `results/report/`.

## Inputs (read with short summaries, never print whole files)
- `results/scorecard.csv`, `results/phase-2/` (`fdr_by_fault.csv`, `vs_literature.csv`, `lag_choice.csv`, figures), `results/phase-3/` (`fdr_by_fault.csv`, `unseen_fault.csv`, `hgb_tuning.csv`, `assistant_v1_vs_v1.1.csv`, `models.json`, figures), `results/phase-4/` (all CSVs, `README.md`, figures, `test_runs.json`), `results/phase-2b/` (`diagnosis_test.csv`, `diagnosis_test_v1.1.csv`, the logs).
- Decisions and caveats: the Comments at the bottom of `.scratch/phase-2/spec.md`, `.scratch/phase-3/spec.md`, `.scratch/phase-4/spec.md` and `.scratch/phase-2b/spec.md` (read the review notes: they list what each result may and may not be used to claim), `docs/adr/`, `CONTEXT.md`, `research/pca-anchors.md`.
- Models for the operator report: `results/phase-2/pca.npz`; data via `src/data.py`.

## Deliverables
1. `scripts/make_report.py`: builds `results/report/report.md` and the operator report from the results files, plus any new charts. One command reruns it; results overwrite `results/report/`. It records runtime, peak RAM and machine at the end of each report (a "How this was made" line).
2. `results/report/report.md`, the comparison report, in this order, each section short:
   - **What this is and what to take from it** (5 bullets, the findings in plain words).
   - **Data and rules**: the Rieth dataset, labels, the disjoint run groups and why (paired noise), two tracks never ranked together, 1% FAR headline, fit/limit/dev/supervised/testing roles. One small table.
   - **Detection**: a table of every model (PCA/DPCA T² and SPE, Isolation Forest L0/L2, autoencoder L0/L2, gradient boosting) at the 1% limit: FAR, mean FDR with and without faults 3/9/15, FDR on faults 1, 2, 4, 6, 7 and on 3, 9, 15, detection delay median and share never detected, train time, peak RAM. Unsupervised and supervised in separate tables. One chart: FDR per fault for the best of each family (finding in the title).
   - **Against the literature**: our 1%-limit PCA/DPCA numbers beside Rato & Reis, from `vs_literature.csv`, with the dataset difference stated.
   - **The unseen-fault test**: the table by fault, with the sentence that fault 16 carries the gap and the tuning-on-all-faults caveat.
   - **Diagnosis**: what blame is and is not; the blame table by method (SPE, T², RBC, permutation importance) against random; the fault 6 first-hour vs whole-window lesson; the three faults where blame misses (2, 8, 10); why faults 3/9/15 hide (before/after against the fault-free control, with the fault 14/19 caveat).
   - **The assistant**: v1 and v1.1 on the frozen 70 cases: unit accuracy, IDV top-1/top-3, normal, 3/9/15, with the no-LLM baseline and chance; the 9-of-10 repeat agreement; cost. State plainly: v1.1 adds a supervised closed-set classifier, all 90 sessions used it, so the gain is "a trained classifier helps", not better reasoning. 3/9/15 answered "normal or can't tell" is not a detection.
   - **What we did not do and what is uncertain**: one list (single seed-42 split, Rieth data not the Braatz data, the HGB every-3rd-row stride, degenerate tuning points, 36 headline cases so wide intervals, Claude may remember the TEP papers, 8-hour permutation step subsampled, DPCA T² false-alarm streaks).
   - **Reproduce it**: the commands in order.
3. `results/report/operator-fault6.md`, the operator-style report, written as a handover for one testing run of fault 6:
   - **Run choice:** the first fault 6 run ID in `results/phase-4/test_runs.json` (the 100 fixed testing runs). Don't pick by how good it looks.
   - **Sections:** (a) *Situation* in 3 lines: when the first PCA SPE alarm was raised (3 in a row above the 1% limit), and the time from fault start to alarm, stated in hours of the run clock; (b) *What the readings show*: a chart of the A feed (`xmeas_1`) and its valve (`xmv_3`) with the fault start marked, plus two downstream readings that move later (the top whole-window blame readings from `top_readings_6.csv`), with real names and units; (c) *Where to look first*: the top 5 relative SPE contributions for the first hour after the alarm and for the whole window, with one line on why they differ (the symptoms spread downstream); (d) *What this does and doesn't tell you*: contribution plots point at symptoms not causes; what a process engineer would check next, limited to what the public TEP description supports (the A feed stream, its valve, the feed controller); the three things we can't say; (e) *Evidence*: file paths and run IDs.
   - One figure of the whole run with the alarm limit and the alarm time marked, readable at phone size. Plain language. No claims beyond the data.
4. New charts in `results/report/figures/`, `tep-data` style (Okabe-Ito colours from `src/style.py`, one finding per chart, the title states it, single y-axis, labels on the lines): `detection-by-family.png` and `operator-fault6.png` (the run chart). Reuse the existing figures from phases 2-4 by link where they already say it; don't redraw them.
5. Tests: every number in `report.md` comes from a results file, so test the table builders on small hand-made frames (the with/without 3/9/15 means, the unsupervised/supervised split, the literature table join, formatting). A test that every `[...](...)` link in the report and a figure path inside it exists after a build.
6. A short `results/report/README.md`: the three files, how to rebuild, which results files feed each section.

## Run order and stops
1. Build and test; run `make_report.py`; read both reports end to end and fix anything that isn't backed by a results file.
2. Stop early only for something critical: a number in the reports that disagrees with its source file, or a result that can't be reproduced from the saved files.

## Acceptance criteria
- `.venv/Scripts/python -m pytest -q` passes (383 existing plus the new ones).
- Every number in the reports matches its source file (spot-check 10 and list them in your report).
- The reports state the caveats listed above wherever the relevant number is used, not only in the final list.
- The reports contain no claim that faults 3, 9 or 15 are detected, no ranking across the supervised and unsupervised tracks, and no statement that Claude reasoning alone reached the v1.1 scores.
- Peak RAM under 5 GB. Machine recorded.

## Out of scope
- New models or runs on the testing files; assistant changes; v2 or v3; posts; pushing (the orchestrator pushes after review); PDF or HTML output.

## Comments
- 2026-11-13, orchestrator: /ml-review **PASS** (no blockers; 447 tests). Reviewer re-derived 15 numbers from the source files: all match. Hard rules held: no claim that faults 3/9/15 are detected, no ranking across tracks, v1.1 credited to the classifier, no private or employer content. Fixes applied after review (commit `954d6b2`): the published-0.951 remark is reworded (hand-added, 21 faults vs our 17, no side-by-side number); overclaims softened ("none of the three statistics we tested", "not above chance, within noise at 36 cases", "gap" not "lead"); the training caveat added to summary finding 1; 99.6% shown for DPCA SPE faults 3/9/15 never-detected; small code tidy-ups.
  - **Found during review, added by the builder at my request (commit `533358a`):** reactor pressure (`xmeas_7`) reaches about 3000 kPa after onset in 100% of fault 6 runs (median sample 277, 5.9 h after onset), 96% of fault 18, 13% of fault 12 and 9% of fault 13; no other fault. This looks like the simulator's pressure shutdown regime (3000 kPa limit not verified in our notes), after which many readings freeze. It inflates whole-window blame for fault 6 (top-1 hit 1.00 in the first hour vs 0.22 over the whole window). The phase 4 headline holds in both windows: SPE top-5 hit 0.77 (first hour) and 0.75 (whole window) vs 0.61 random over the 12 detectable faults (0.75 and 0.73 without fault 6). `results/report/pressure_tail.csv`.
  - Carried forward (minor): the phase 4 cleanups listed in `.scratch/phase-4/spec.md` Comments; verify the 3000 kPa shutdown limit in Downs & Vogel (1993) before quoting it in a post.
