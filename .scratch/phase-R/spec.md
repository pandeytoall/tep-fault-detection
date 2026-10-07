# Phase R: research brief

Goal: gather the evidence needed to propose a scope for phases 2-3 (which ML models, which MSPC variant, which FAR target).

Owner: `researcher`. Output: `research/tep-benchmarks.md`. No code.

## Questions
1. **Benchmarks.** Which methods have been reported on TEP, with per-fault FDR and the FAR used? Prioritise the Rieth 2017 data; include the original Braatz data for context, labelled as such.
2. **Fast CPU methods.** Compare MSPC (PCA T²/SPE, DPCA, CVA, ICA, SFA) and classical ML (RF, gradient boosting, SVM, kNN, one-class / isolation forest). Note which ones fit the laptop limits on the Rieth data size.
3. **Framing.** Supervised classifiers trained on faulty runs versus unsupervised detectors trained on normal data only. How do papers compare them fairly with MSPC?
4. **Threshold and FAR.** Which FAR do papers use (1%, 5%, 99% limits)? How are thresholds set? What FDR is achievable at about 1% FAR?
5. **Faults 3, 9, 15.** Typical FDR, whether papers exclude them, and known leakage traps behind high reported numbers.
6. **Dynamics.** Effect of lagged or windowed features (window length) on FDR and detection delay.
7. **Compute.** Train time and subsampling practice (how many runs) reported on the Rieth data.

## Done when
- `research/tep-benchmarks.md` has the comparison table (method | FDR easy faults | FDR 3/9/15 | FAR | compute cost | dataset | source) and key findings.
- Every number has a source URL; unconfirmed items are marked UNVERIFIED.

## Comments
