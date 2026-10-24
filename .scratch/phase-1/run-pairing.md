# Finding: run IDs share their noise across files (2026-10-22)

Checked by the orchestrator with short parquet reads (`pyarrow` filters on `faultNumber`, `simulationRun`; 52 variables).

## Evidence
| Comparison | Samples | Max abs difference |
|---|---|---|
| FaultFree_Training run 1 vs Faulty_Training fault 1 run 1 | 1-20 (before onset) | 0.0 |
| Faulty_Training fault 1 run 1 vs fault 2 run 1 | 1-20 | 0.0 |
| Faulty_Training fault 1 run 1 vs fault 1 run 2 | 1-20 | 98.9 |
| FaultFree_Testing run 1 vs Faulty_Testing fault 1 run 1 | 1-160 (before onset) | 0.0 |
| FaultFree_Training run 1 vs FaultFree_Testing run 1 | 1-20 | 98.8 |

After onset, faulty run 1 vs normal run 1 (training files), median |difference| in units of the normal run's standard deviation:

| Fault | Median |diff|/std |
|---|---|
| 1 | 0.806 |
| 3 | 0.002 |
| 9 | 0.030 |
| 15 | 0.056 |

Cross-family check (all 500 FaultFree_Training runs vs all 500 FaultFree_Testing runs, samples 1-20, exact byte match per sample and per block): 0 shared. Samples 6-20 for 50 x 500 pairs: the closest pair still differs by max |diff|/std = 3.28. No testing run is a copy of any training run, under any ID pairing.

## Meaning
- Within one file family (training or testing), run k of every fault and run k of the fault-free file are the same simulation with the same random noise. Only the fault's effect differs after onset.
- The training and testing families do not share noise, so final test metrics are not leaked by this.
- Any reuse of a run ID across jobs inside the training family leaks noise. Example: fit on normal run k, then tune or set limits on faulty or normal run k.
- Faults 3, 9 and 15 are almost copies of the paired normal run. A model that has seen run k can "detect" them by memorising noise, which matches the known too-good-result trap.
- A paired difference (faulty run k minus normal run k) would make faults 3/9/15 trivially visible, so it is banned as a model input in both families.

## Decision (user, 2026-10-22)
One partition of run IDs 1-500, applied to every training file and every fault:
- fit 330 (unsupervised fit, normal runs)
- limit 100 (alarm limits, normal runs; 50 of them also tune the supervised model)
- supervised 50 (faulty runs per fault for supervised training, plus the normal runs with these IDs)
- dev 20 (faulty and normal development and tuning runs)

This replaces the earlier 400/100 normal and 50 + 20 faulty budget, which reused IDs. Testing files: all runs, once, as before.
Options considered: shrink limits to 30 (noisier 1% limit: about 150 samples above it) or keep the overlap (leaky tuning and limits). Both rejected.
