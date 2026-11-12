# Phase 4: diagnosis (who gets blamed, and why faults 3, 9 and 15 hide)

Machine: laptop, Intel i5-1135G7, 7.7 GB RAM. Model: the saved PCA (31 PCs, no lags). Windows: `post` = every sample after the onset;
`first_hour` = the first 20 of them. Headline numbers use `post`.

## Files
| File | Holds |
|---|---|
| `dev_blame.csv`, `test_blame.csv` | one row per fault, method and window: the four blame metrics, the random baseline, the top 3 readings, the note. Dev = 20 dev runs per fault; test = 100 testing runs per fault. Permutation importance (`perm`) is in the dev file only. |
| `dev_blame_by_group.csv`, `test_blame_by_group.csv` | the same metrics averaged over the fault groups (benchmark 1/2/4/6/7, the rest, hidden 3/9/15, unknown cause 16-20) |
| `dev_hidden.csv`, `test_hidden.csv` | within-run before/after statistics per fault, with the fault-free control (fault 0) and a plain verdict |
| `dev_perm_importance.csv` | permutation importance of `hgb-l2` per fault and reading (3 lags summed) |
| `test_blame_median_spe.csv` | median relative SPE contribution per fault and reading (feeds the heatmap) |
| `top_readings_<fault>.csv` | top 10 readings by median SPE contribution (dev runs), with names, units, plant unit, every method's value and rank |
| `rbc_reference.npz` | mean RBC per reading on the 330 `fit` normal runs (the normal reference for the RBC methods) |
| `test_runs.json` | the 100 testing run IDs (the same for every fault) and the 100 control run IDs |
| `diagnose_dev.log`, `diagnose_final.log` | what the scripts printed, including the sanity items |
| `figures/` | `blame-heatmap.png`, `blame-methods.png`, `hidden-faults.png` |

Every CSV has `machine`, `runtime_s` and `peak_ram_gb` columns.

## Metrics, one line each
- `top1_hit`: share of runs whose single top reading is in the expected plant unit.
- `top5_hit`: share of runs with at least one top-5 reading in the expected unit.
- `unit_mass`: share of the (clipped at zero) relative contribution that falls in the expected unit, averaged over runs.
- `smear`: share of the total relative contribution held by the top 5 readings (high = concentrated, low = spread out).
- `baseline_unit_share`: share of the 52 readings that sit in the expected unit; what random picking scores on `top1_hit` and `unit_mass`.
- `baseline_top5_hit`: 1 - (1 - share)^5, what random picking scores on `top5_hit` (the spec's approximation; it slightly overstates chance).
- Relative contribution: window mean over the `fit` normal mean of that reading; about 1 is normal.
- Permutation importance: drop in 21-class accuracy on that fault's samples when one reading (all 3 lags) is shuffled; one vector per fault, so its hits are 0 or 1 per fault.

## Which method blames the right unit best
Over the 12 detectable faults that have an expected unit (1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14), post-onset window, means over faults:

| Method | top1_hit | top5_hit | unit_mass | random (top1 and mass / top5) |
|---|---|---|---|---|
| SPE (test) | 0.51 | 0.75 | 0.34 | 0.18 / 0.61 |
| RBC-SPE (test) | 0.51 | 0.74 | 0.33 | |
| T² (test) | 0.43 | 0.56 | 0.31 | |
| RBC-T² (test) | 0.48 | 0.52 | 0.31 | |
| Permutation importance (dev, 1 vector per fault) | 0.50 | 0.67 | 0.52 | |

- SPE blames the right unit best of the four PCA views: its top 5 readings include the faulted unit in 75% of runs, 14 points above random picking (61%). RBC-SPE is the same to within a point (see below). T² and RBC-T² are no better than random on `top5_hit`.
- Permutation importance concentrates its blame (unit_mass 0.52, far above random 0.18) but it is one vector per fault, scored on dev runs only, so it is a coarse comparison.
- The gain over chance is modest. Unit mass is only about 0.33 for SPE: contributions smear across correlated readings, and they show where the symptoms are, not where the cause is.
- Works well (SPE top5_hit at least 0.96 on test): faults 1, 4, 5, 6, 7, 11, 12, 13, 14. Fails: fault 2 (0.00), fault 8 (0.05) and fault 10 (0.00): SPE points at downstream readings (condenser cooling flow, stripper underflow, stripper pressure) instead of the feeds.
- Faults 3, 9, 15 ("near the false-alarm level; blame is mostly noise"): their blame hits are at or near chance and mean nothing.
- RBC-SPE equals SPE here by arithmetic, not by accident: the reconstruction denominator is a constant per reading, so dividing by the normal mean of the same quantity cancels it. They differ only where the 1% floor applies.

## Why faults 3, 9 and 15 hide (plain words)
Inside a single run, the readings after the onset look like the readings before it: the largest change in any reading is 0.47 sd on the testing runs (0.70 sd on dev), exactly what a fault-free run gives, and the spread and autocorrelation changes are at the fault-free level too, so there is no signal for any monitor to find.

Caution: the check uses mean shift as its headline, and faults 14 and 19 also match the control on it (0.47 sd on test) while 14 is easily detected: its signature is in spread (14.1 vs 2.1) and autocorrelation (1.23 vs 0.25), not in the mean. For 3, 9 and 15 all three statistics sit at the control's level (spread 2.04 to 2.10 vs 2.09; autocorrelation 0.24 to 0.25 vs 0.25).

Control choice: the fault-free control uses run IDs that share no noise stream with the faulty runs (dev: 20 runs of the `limit` group; test: the 100 runs after the 100 faulty ones in the seed-42 shuffle). With the same IDs as the faulty runs the control would be paired with them. The dev number with the paired IDs is also 0.70 sd.

## Deviations and notes
- Permutation importance on dev used every 5th post-onset sample (38,400 pooled rows instead of 192,000), `n_repeats=5`, one process. The pass took 28,671 s (about 8 h) on the laptop, over the 3 h guide (the estimate was about 1.7 h; the machine was also under memory pressure). Not rerun. The final pass computes no permutation importance (dev only, by spec).
- Dev sanity: two of the items failed on the post-onset window (fault 6 A-feed on top in 45% of runs, 100% in the first hour; fault 2 SPE top5_hit 0.00 against baseline 0.57). Nothing was tuned; see the build report.
- Fault 14 and 19 style "looks like the control" verdicts are on mean shift only.
