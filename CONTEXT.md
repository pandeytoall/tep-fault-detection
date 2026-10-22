# TEP fault detection

Comparing fault-detection models on the public Tennessee Eastman Process (TEP) simulation data, on one honest scorecard.

## Language

**Run**:
One simulation of the plant from start to end (500 samples in training files, 960 in testing files, one sample every 3 minutes). Train/test splits are made by run, never by sample.
_Avoid_: episode, trajectory

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
