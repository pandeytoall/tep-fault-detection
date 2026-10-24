# Disjoint run groups, because paired runs share noise

In the Rieth (2017) files, run k of the fault-free file and of every fault in the same family is one simulation with one noise stream. They're identical before onset, and faults 3, 9 and 15 stay almost identical after it. Our first run budget reused run numbers across jobs: fit on normal runs 1-400, tune on faulty runs with the same numbers. That would let a model memorise noise, which makes alarm limits and tuning optimistic and can make faults 3/9/15 look detectable.

So one seed-42 partition of run numbers 1-500 applies to every training file and fault:
- fit 330
- limit 100 (limit_tune 50 inside it)
- supervised 50
- dev 20

No run number serves two jobs. A paired difference (faulty run k minus normal run k) is never a model input. Training and testing families share no noise (checked over all 500 x 500 pairs), so final test scores are unaffected.

**Considered options:**
- Keep 400/100 with overlap: leaky limits and tuning.
- Shrink limits to 30: a noisier 1% limit, with about 150 samples above it.

**Consequences:** the supervised model is capped at 50 runs per fault. Evidence: `.scratch/phase-1/run-pairing.md`.
