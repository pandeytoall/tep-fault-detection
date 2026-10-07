# TEP fault-detection benchmarks — research notes (Phase R)

Author: research pass for rahulpandey.pgcmma01@iimtrichy.ac.in, 2026-09-25.
Scope: answer the 7 questions in `.scratch/phase-R/spec.md` to support scoping the ML phase (Phase 2)
and the MSPC phase (Phase 3). Primary sources only; every number has a URL. Items I could not
confirm from a primary source are marked **UNVERIFIED**.

IMPORTANT CAVEAT ON SOURCE ACCESS: several of the most relevant papers (ChemRxiv benchmarking paper,
Yin et al. 2012, ResearchGate PDFs, Springer chapters) are paywalled or blocked automated fetching
(HTTP 403) during this research pass. Where I could only get title/abstract-level info I say so
explicitly. The orchestrator or user may need to fetch these manually (e.g. via institutional access)
before finalizing scope.

---

## Comparison table

| Method | Dataset | FAR used | FDR "easy" faults (e.g. 1,4,6,7,8,12,13,14) | FDR faults 3/9/15 | Compute cost | Source |
|---|---|---|---|---|---|---|
| PCA + k-means clustering (unsupervised) | Rieth 2017 (TEPRieth, cropped: 500 normal runs / 5 runs per fault train) | FPR threshold target ≤0.05 (paper's usability bar); reported overall Detection FPR = 0.00 | Overall avg Detection TPR = 0.36 across 20 faults; 12/20 faults "remained undetected" | Not broken out per-fault in extractable text; PCA is explicitly one of the weak baselines that misses most faults | Not reported (deep methods in same paper: laptop inference ~0.01s/sample, ~2.4MB footprint for the DL method, not PCA) | SensorSCAN paper, arXiv:2208.08879, ar5iv rendering: https://ar5iv.labs.arxiv.org/html/2208.08879 |
| GRU (supervised, trained on 1 labeled run per fault) | Rieth 2017 (TEPRieth) | Not stated as FAR; semi-supervised single-run setting | Not extracted | Fault 3: TPR 0.15±0.03; Fault 9: TPR 0.03±0.01; Fault 15: TPR 0.04±0.01 | Not reported | Same as above, Table 8: https://ar5iv.labs.arxiv.org/html/2208.08879 |
| SensorSCAN (self-supervised deep clustering) — DL, context only | Rieth 2017 (TEPRieth) | Usability bar FPR ≤ 0.05 | Fault 3: TPR 0.95 (much better than PCA/GRU) | Fault 9: 0.00 TPR (undetected); Fault 15: 0.00 TPR (undetected) — "consistently undetected across unsupervised methods" | Inference ~0.01s/sample on a laptop, 2.38 MB model; training cost not reported | arXiv:2208.08879, https://ar5iv.labs.arxiv.org/html/2208.08879 |
| 27 unsupervised deep time-series AD methods (LSTM-AE, transformers, GANs, etc.) — DL, context only | Rieth 2017 ("modified Downs & Vogel 1993 dataset, referenced in Rieth et al. 2018"); 21 classes, 500 runs/class, train 25h/test 48h @3min sampling | Not FAR-based; F1 and AUPRC used, no per-fault breakdown extracted | Not extracted (no per-fault table found) | Not extracted | **Each of the 27 methods took ~24 hours to train/evaluate**, standardized grid search — explicitly NOT CPU-laptop feasible at that scale | Hartung et al. 2023, arXiv:2303.05904, https://ar5iv.labs.arxiv.org/html/2303.05904 (also published Chemie Ingenieur Technik, https://onlinelibrary.wiley.com/doi/10.1002/cite.202200238) |
| XGBoost-based feature selection + AVSSA-optimized KELM | Dataset version not confirmed (likely original Downs-Vogel style, single/few runs) — **UNVERIFIED which TEP version** | Not stated as FAR; supervised classification accuracy framing | Overall mean test FDR 91.00% | Fault 3: 76.77%; Fault 9: 72.40%; Fault 15: 61.67% | Not reported | Found via search snippet only, likely "Fault Diagnosis of Tennessee Eastman Process with XGB-AVSSA-KELM Algorithm", https://doi.org/10.3390/en15093198 — **numbers not independently confirmed from full text, mark UNVERIFIED**; unusually high FDR on 3/9/15 vs. rest of literature is a leakage red flag (see Q5) |
| XGBoost (multiclass, 21-way classification) | Dataset/split not confirmed — **UNVERIFIED** | Not FAR-based; F1-micro used for multiclass | Reported: "except faults 10 and 12, all other faults FDR > 0.90", avg FDR ≈ 0.9288 | Not broken out in snippet | Not reported | Found only via search snippet (title/venue not confirmed) — **UNVERIFIED, could not re-locate/fetch the source paper directly; treat this number with suspicion** |
| Random Forest (multiclass) | Same unconfirmed source as above | Not FAR-based | avg FDR ≈ 0.8986 | Not broken out | Not reported | Same as above — **UNVERIFIED** |
| PCA T² / SPE (classic MSPC) with 99% control limit (≈1% FAR by construction) | Not confirmed which TEP version (tutorial doesn't state; cites Chiang et al. 2000, which normally uses original Downs-Vogel/Braatz-style single-run data) | Control limit set so 99% of training SPE falls below it → ~1% FAR by construction | IDV(1): FDR > 99% | Not reported in accessible text (tutorial only shows faults 1 and 11) | Not reported | BibMon docs tutorial, https://bibmon.readthedocs.io/en/latest/tutorial_tep.html |
| PCA T²/SPE, DPCA, ICA, PLS, FDA (classic multivariate SPC comparison) | **Original Braatz/Downs-Vogel TEP** (this is the standard reference dataset for this paper, single run per fault, not Rieth 2017) | 99% control limit / ~1% FAR (standard for this benchmark family) | Not independently confirmed (paywalled) | Paper explicitly identifies faults 3, 9, 15 as hard to diagnose "due to close similarity in the responses of noisy measurements" — a widely cited qualitative finding | Not applicable (classical, cheap methods; single-run TEP is tiny) | Yin, Ding, Haghani, Hao, Zhang (2012), "A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process," J. Process Control 22(9):1567-1581. https://www.sciencedirect.com/science/article/abs/pii/S0959152412001503 — **paywalled; exact FDR numbers per fault NOT independently confirmed, mark UNVERIFIED for numeric values, but the "faults 3/9/15 are hard" finding is confirmed at abstract/citation level from multiple secondary mentions** |
| DPCA-DR (dynamic PCA on decorrelated residuals) | **Original Braatz-style TEP** (paper predates Rieth 2017; "21 faults" framing matches classic benchmark) | Not confirmed (paywalled) | DPCA-DR reported to have "highest detection rates on 19 out of 21 faults" vs. PCA/DPCA | Not extracted (paywalled) | Not reported | ScienceDirect, https://www.sciencedirect.com/science/article/abs/pii/S0169743913000592 — **UNVERIFIED numeric FDR, abstract-level claim only** |
| Rieth, Amsel, Tran, Cook (2017/2018) — dataset-defining paper; RF/SVM-style anomaly detection with ROC analysis | Rieth 2017 (the dataset itself; this paper introduces it) | Uses ROC curves explicitly to show FAR/FDR tradeoff is "masked by single-metric reporting" | Not extracted (paywalled Springer chapter) | Not extracted | Not reported | "Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems," Rieth et al., in AHFE 2017 proceedings, Springer, https://link.springer.com/chapter/10.1007/978-3-319-60384-1_6 (paywalled — **could not fetch full text**); dataset DOI: https://doi.org/10.7910/DVN/6C3JR1 |
| fddbenchmark framework (AIRI-Institute) | Supports `rieth_tep` (21,000 runs = 500 runs × 21 states × ... actually 500×(1+20)), `small_tep` (210 runs, a cropped Rieth subset), `reinartz_tep` (2,800 runs, Reinartz et al. 2021 dataset) | Framework reports TPR/FPR/Average Detection Delay, CDR — user sets thresholds, no fixed FAR baked in | Not a benchmark result itself, a tooling repo | Not a benchmark result itself | Not reported | https://github.com/AIRI-Institute/fddbenchmark — useful as a **labeling/metric reference implementation**, not a source of comparable numbers |
| Reinartz, Kulahci, Ravn (2021) extended TEP dataset paper | **New dataset, not Rieth 2017**: 28 faults × 6 operating modes, 500 runs each, 100h duration, 3-min sampling | Not confirmed | Not confirmed | Not confirmed | Not reported | Comput. Chem. Eng. 149:107281, https://www.sciencedirect.com/science/article/abs/pii/S0098135421000594 — paywalled, **flagging as a DIFFERENT dataset from Rieth 2017, do not mix numbers** |

---

## Key findings by question

### Q1. Benchmarks reported on TEP
Two distinct dataset lineages dominate the literature, and mixing their numbers is a common
comparability trap:
- **Original Braatz/Downs-Vogel TEP** (Chiang, Russell, Braatz-era data; single run per fault,
  480 test samples, fault introduced at sample 160/8h in a 48h test run). This is the dataset behind
  the classic PCA/DPCA/ICA/PLS comparison literature (Yin et al. 2012:
  https://www.sciencedirect.com/science/article/abs/pii/S0959152412001503; DPCA-DR:
  https://www.sciencedirect.com/science/article/abs/pii/S0169743913000592).
- **Rieth et al. 2017** (this project's dataset, 500 runs/fault, Harvard Dataverse
  doi:10.7910/DVN/6C3JR1) — designed explicitly to fix the single-run evaluation problem of the
  original data and enable statistically meaningful FDR/FAR estimates. Most recent deep-learning
  and self-supervised papers (SensorSCAN arXiv:2208.08879, Hartung et al. arXiv:2303.05904) use
  this version or note they derive from it.
- A third, separate lineage is **Reinartz et al. 2021** (28 faults, 6 operating modes,
  https://www.sciencedirect.com/science/article/abs/pii/S0098135421000594) — not the same as
  Rieth 2017; results from it are not directly comparable to this project.

I could not get numeric FDR/FAR tables out of the paywalled classic papers (Yin 2012, DPCA-DR,
Rieth's own AHFE 2017 chapter) with the tools available in this session — all returned HTTP 403 or
unreadable PDF streams. The orchestrator should treat any specific numbers attributed to those
papers elsewhere as **UNVERIFIED** unless re-checked with manual/institutional access.

### Q2. Fast CPU methods vs. laptop limits
No primary source in this pass gave a directly reported wall-clock training time for PCA/DPCA/CVA/
ICA/SFA or RF/XGBoost/SVM/kNN/IsolationForest **on the Rieth 2017 data specifically**. Reasoning
from what is confirmed:
- Classic MSPC (PCA T²/SPE, DPCA) are the cheapest methods by construction — closed-form fit,
  O(n·p²) — and were originally run on tiny single-run TEP data, so they trivially fit the laptop
  limits (CPU-only, 7.7GB RAM) even at full Rieth scale (250k normal rows × 52 vars) if done in
  batches/streaming or on a subsampled set of runs.
- The 27-method deep-learning benchmark (arXiv:2303.05904) explicitly needed ~24h/method — this is
  the clearest confirmed evidence that DL approaches are **not** laptop-feasible at full scale and
  supports scoping Phase 2 to classical ML only, consistent with the CLAUDE.md/skill rule of
  "no deep learning unless the user approves."
- RF/XGBoost/SVM/kNN/IsolationForest compute costs on Rieth-scale data (millions of rows) were not
  found with confirmed numbers; this is a gap — recommend a quick internal timing pilot in Phase 2
  rather than relying on literature, since no paper in this pass reported laptop-class compute cost
  for these methods on the full Rieth data size.

### Q3. Supervised vs. unsupervised framing
- SensorSCAN (arXiv:2208.08879) is the clearest primary source on this: it explicitly compares an
  **unsupervised** PCA+k-means baseline (overall TPR 0.36, FPR 0.00) against a **supervised** GRU
  baseline trained on a single labeled run per fault (Fault 3 TPR 0.15, Fault 9 TPR 0.03, Fault 15
  TPR 0.04) and its own self-supervised method (Fault 3 TPR 0.95, Faults 9/15 TPR 0.00). This shows
  the fair-comparison convention used: same test set (full 500 runs/state), same fault labeling
  (post-onset positive), but **different training data availability** (PCA/self-supervised: normal
  only; GRU: 1 labeled run per fault) — i.e., papers report training-data budget explicitly rather
  than pretending supervised and unsupervised methods are on equal footing. The key comparability
  variable is not just the method but how many labeled faulty runs (if any) went into fitting it.
  Source: https://ar5iv.labs.arxiv.org/html/2208.08879
- No confirmed primary source in this pass reported classical RF/XGBoost trained on the **full**
  faulty training set (500 runs × 20 faults) compared head-to-head against PCA T²/SPE at a matched
  FAR on Rieth data. This is a genuine literature gap for this specific project's exact setup —
  recommend the project generate this comparison itself in Phase 2/3 rather than expecting to find
  it pre-published.

### Q4. FAR / threshold conventions
Two conventions recur:
- **"99% control limit"** for MSPC (PCA T²/SPE), i.e., threshold set at the 99th percentile of the
  training statistic → FAR ≈ 1% by construction. Confirmed in the BibMon TEP tutorial:
  https://bibmon.readthedocs.io/en/latest/tutorial_tep.html
- **FPR ≤ 0.05 "usability bar"** used by SensorSCAN as the ceiling for a method to be considered
  practical: https://ar5iv.labs.arxiv.org/html/2208.08879 — this is looser than 1% and not
  directly comparable to the 99%-control-limit convention; papers using it are not on the same FAR
  footing as classic MSPC papers.
- Achievable FDR at ~1% FAR: the only confirmed near-1%-FAR-style number at Rieth scale is PCA
  (via k-means clustering variant) at overall TPR 0.36, FPR 0.00 (arXiv:2208.08879) — notably FPR
  0.00 suggests the threshold may be conservative (below the nominal 1% target), which would
  further suppress FDR; this is a caution flag, not a clean "PCA gets 36% FDR at 1% FAR" result to
  cite directly, since FPR is 0 not ~1%. UNVERIFIED whether this PCA is standard T²/SPE MSPC or the
  paper's own k-means-on-PCA-embedding variant (text explicitly says "PCA with k-means clustering",
  which is not the same as classic Hotelling T²/SPE monitoring) — treat as informative but not a
  direct match to the project's planned MSPC baseline.
- No confirmed source gives a full per-fault FDR table for classic PCA T²/SPE at a clean 1% FAR on
  the **Rieth 2017** data specifically. This is the single biggest gap found in this research pass.

### Q5. Faults 3, 9, 15 — typical FDR and leakage traps
- Multiple independent sources agree these are the hardest faults: "unobservable" or "difficult to
  diagnose due to close similarity in the responses of noisy measurements" (paraphrase found
  attributed to the Yin et al. 2012 lineage and repeated across several search-indexed sources
  discussing that paper).
- Confirmed near-zero detectability for unsupervised/lightly-supervised methods on Rieth data:
  GRU (1 labeled run): Fault 3 = 15%, Fault 9 = 3%, Fault 15 = 4% TPR; PCA+k-means and other
  unsupervised baselines: Faults 9 and 15 essentially undetected (0.00 TPR) across the board;
  even the strongest deep self-supervised method (SensorSCAN) gets Fault 3 to 0.95 but Faults 9/15
  remain at 0.00. Source: https://ar5iv.labs.arxiv.org/html/2208.08879
- **Leakage red flag identified**: one search-indexed result (XGB-AVSSA-KELM,
  https://doi.org/10.3390/en15093198, and a separate unconfirmed XGBoost/RF multiclass result)
  reports Faults 3/9/15 FDR in the 60-77% range and overall FDR ~91-93% — dramatically higher than
  every other source in this table. Per the project's own skill notes ("High FDR on 3/9/15 means
  suspect leakage, not a breakthrough") and given the consistent near-zero results elsewhere in the
  literature, this discrepancy is very likely explained by evaluation-set leakage (e.g., random
  row-level train/test splits instead of by-run splits, letting autocorrelated samples from the
  same run appear in both train and test) or threshold/feature selection tuned on test data. I
  could not fetch the full text of either source to confirm the split methodology (403/paywall), so
  I cannot pinpoint the exact leakage mechanism — but the numbers should not be used as a target or
  a sanity check for this project without independently verifying the train/test split was by
  `simulationRun`, not by row.
- Practical implication for this project: expect Faults 3/9/15 to score near-FAR-level (i.e.,
  close to 0% incremental FDR) for both MSPC and classical ML if the by-run split rule is honored.
  Any run in this project that shows high FDR (>50%) on these faults should trigger a leakage audit
  before being reported as a result.

### Q6. Effect of dynamics (lagged/windowed features)
- DPCA-DR (dynamic PCA on decorrelated residuals) claims highest detection rate on 19/21 faults
  vs. static PCA/DPCA on the classic Braatz-style TEP, i.e., adding dynamics/autocorrelation
  handling measurably improves FDR: https://www.sciencedirect.com/science/article/abs/pii/S0169743913000592
  (abstract-level claim; the underlying numeric table was paywalled and not confirmed).
- A search-indexed summary (not independently fetched/confirmed) states that for the original TEP,
  fixed-control-limit DPCA can produce high FAR, high missed-detection rate, and long detection
  delay, and that **adding lags of monitoring-variable residuals plus moving-window/adaptive
  thresholds** improves fault detection rate, using IDV(11) as the example case. I could not trace
  this to a single confirmed primary source/URL within this session — mark **UNVERIFIED**, flagged
  via search snippet only.
- No confirmed source quantifies FDR or detection delay as a function of window length (e.g., 5 vs
  20 vs 50 samples) specifically on Rieth 2017 data. This is a gap; if the project wants to justify
  a specific window length for DPCA/dynamic features, it will likely need its own small
  experiment rather than a literature citation.

### Q7. Compute cost and run subsampling
- Confirmed: the 27-deep-method benchmark on Rieth-derived data used ~24h/method
  (arXiv:2303.05904) — not laptop-feasible, supports excluding DL from Phase 2/3 scope by default.
- Confirmed run-subsampling practice: SensorSCAN deliberately **imbalances** its Rieth-based
  training set to 500 normal runs + only 5 runs per faulty state (out of the available 500), while
  testing on the full 500 runs/state (https://ar5iv.labs.arxiv.org/html/2208.08879). This is a
  useful precedent: even papers working directly with the full Rieth release don't train supervised
  models on all 500×20 faulty runs; a much smaller labeled-run budget (e.g., 5-25 runs/fault) is
  treated as reasonable and realistic.
- No confirmed source reports peak RAM or wall-clock time for classical ML (RF/XGBoost/SVM/kNN/
  IsolationForest) at Rieth's full scale (5-9.6M rows for faulty train/test). Given the project's
  documented laptop limits (7.7GB RAM, 4 cores, keep peak RAM <5GB, Faulty_Testing ~4.2GB as
  float64), and no literature precedent for full-scale classical ML on this exact data, Phase 2
  should plan to (a) work from the float32 parquet conversion already mandated by project
  conventions, and (b) subsample runs per fault (SensorSCAN's 5-25 runs/fault precedent is a
  reasonable starting point) rather than assume full-scale training is necessary or proven safe.

---

## Sources consulted (full list)

- Rieth et al. dataset, Harvard Dataverse: https://doi.org/10.7910/DVN/6C3JR1
- Rieth, Amsel, Tran, Cook, "Issues and Advances in Anomaly Detection Evaluation for Joint
  Human-Automated Systems," AHFE 2017 / Springer: https://link.springer.com/chapter/10.1007/978-3-319-60384-1_6
  (paywalled, not fetched)
- Yin, Ding, Haghani, Hao, Zhang (2012), J. Process Control 22(9):1567-1581:
  https://www.sciencedirect.com/science/article/abs/pii/S0959152412001503 (paywalled, abstract/
  citation-level info only)
- DPCA-DR paper, J. Process Control-family journal:
  https://www.sciencedirect.com/science/article/abs/pii/S0169743913000592 (paywalled, abstract only)
- Hartung et al. (2023), "Deep Anomaly Detection on Tennessee Eastman Process Data," arXiv:2303.05904,
  https://ar5iv.labs.arxiv.org/html/2303.05904 ; also https://onlinelibrary.wiley.com/doi/10.1002/cite.202200238
- SensorSCAN, arXiv:2208.08879, https://ar5iv.labs.arxiv.org/html/2208.08879
- BibMon PCA/TEP tutorial: https://bibmon.readthedocs.io/en/latest/tutorial_tep.html
- fddbenchmark (AIRI-Institute): https://github.com/AIRI-Institute/fddbenchmark
- Reinartz, Kulahci, Ravn (2021), Comput. Chem. Eng. 149:107281:
  https://www.sciencedirect.com/science/article/abs/pii/S0098135421000594 (paywalled, not fetched)
- "Benchmarking Machine Learning Fault Detection Methods on the Tennessee Eastman Process Dataset,"
  ChemRxiv: https://chemrxiv.org/doi/10.26434/chemrxiv.10001628 (HTTP 403, could not fetch — noted
  as a promising primary source to revisit with authenticated/manual access)
- XGB-AVSSA-KELM, https://doi.org/10.3390/en15093198 (not fetched directly, numbers via search
  snippet only — UNVERIFIED)
- "Fault Detection and Diagnosis in Tennessee Eastman Process with Deep Autoencoder" (PHM Society),
  https://papers.phmsociety.org/index.php/phmconf/article/download/3578/phmc_23_3578 (PDF text
  extraction failed in this session)
- "Run-Level Fault Detection and SHAP-Based Diagnosis of Persistent Classification Difficulty in the
  Tennessee Eastman Process," MDPI Processes 14(16):2569, https://www.mdpi.com/2227-9717/14/16/2569
  (HTTP 403, could not fetch — title strongly suggests direct relevance to Q5, revisit)

## Unresolved access issues (for the orchestrator)
Several likely-high-value sources returned HTTP 403 or corrupted PDF text via the tools available
in this session: ChemRxiv benchmarking paper, MDPI persistent-difficulty paper, Springer Rieth
chapter, ResearchGate PDFs, ScienceDirect abstract pages beyond the abstract. If more precise
numbers are needed before finalizing scope, these should be fetched via institutional/library access
or asked of the user directly rather than guessed.
