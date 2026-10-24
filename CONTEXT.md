# TEP fault detection

Comparing fault-detection models on the public Tennessee Eastman Process (TEP) simulation data, on one honest scorecard.

## Language

**Run**:
One simulation of the plant from start to end (500 samples in training files, 960 in testing files, one sample every 3 minutes). Train/test splits are made by run, never by sample.
_Avoid_: episode, trajectory

**Paired runs**:
Runs with the same number in one file family (training or testing). They share one noise stream, so they're identical up to **onset**; afterwards they differ only by the fault's effect. Training and testing families share no noise.
_Avoid_: matched runs, twin runs

**Run group**:
One of four fixed, disjoint sets of run numbers in the training family. Each group has one job, the same for every fault: **fit** (330, models learn normal behaviour), **limit** (100, set **alarm limits**; 50 of them, **limit_tune**, also tune the supervised model), **supervised** (50, supervised training), **dev** (20, **development runs**). A run number is never used in two groups.

**Development runs**:
The **dev** run group: faulty and normal training runs used for tuning and sanity checks. They never set alarm limits or headline results.
_Avoid_: validation set (ambiguous with **limit**)

**Alarm limit**:
The score above which a model raises an alarm on a sample. Set on held-back normal training runs so the model hits a target **FAR**; never on test data.
_Avoid_: cut-off, threshold (in posts)

**FAR (false alarm rate)**:
The share of samples in fault-free test runs that a model flags. Every model is scored at two targets: 1% (the headline, used for ranking) and 5% (secondary).
_Avoid_: false positive rate, specificity

**Unsupervised track**:
Models that learn only from normal runs (e.g. PCA). Reported in their own table.
_Avoid_: anomaly detection track, one-class

**Supervised track**:
Models that also learn from labelled faulty runs. Reported in their own table, never ranked against the unsupervised track.
_Avoid_: classification track

**Alarm score**:
A model's per-sample number where higher means "less normal". For PCA it is T² or SPE; for the supervised track it is 1 minus the predicted chance of normal.

**Unseen fault**:
A fault type deliberately left out of supervised training, used to test how a supervised model does on a fault it has never seen.

**Onset**:
The sample after which a faulty run is actually faulty: after sample 20 in training files, after sample 160 in testing files. Samples up to onset are normal operation.
_Avoid_: fault start time (ambiguous between files)

**FDR (fault detection rate)**:
The share of post-onset samples in faulty test runs that a model flags, per fault. Counted per sample, as in the literature.
_Avoid_: accuracy, recall (in posts)

**Sustained alarm**:
Three alarmed samples in a row (9 minutes). Only used for **detection delay**, never for **FDR** or **FAR**.

**Detection delay**:
Minutes from **onset** to the third sample of the first **sustained alarm** in a run. Only alarms after onset count. Reported per fault as the median over detected runs, with the share of runs never detected, next to the sustained-alarm rate on fault-free runs (the baseline it must beat).
_Avoid_: latency, time to detect

**Lag window (L)**:
How many past samples a model sees besides the current one. One L is shared by DPCA and the other lagged models, chosen from normal runs only; PCA uses none.

**T²**:
A PCA alarm score: how far a sample sits from normal along the patterns normal operation uses. It catches unusual size of normal-looking movement.
_Avoid_: Hotelling (in posts)

**SPE**:
A PCA alarm score: how much of a sample falls outside the patterns normal operation uses (squared prediction error). It catches broken relationships between readings.
_Avoid_: Q statistic (in posts)

**Textbook limit**:
An alarm limit computed from a statistical formula (F-distribution for T², Jackson-Mudholkar for SPE) instead of from held-back normal runs. Reported for MSPC only; it assumes Gaussian data and can miss its FAR target.
