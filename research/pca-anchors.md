# PCA / DPCA sanity anchors for phase 2 (D8) and HistorianRCA lead

Researched 2026-10-22. Short pass. Every number below was read from an open web page; anything not confirmed is marked UNVERIFIED.

## Bottom line
- The original sources (Russell, Chiang & Braatz 2000, ChemoLab 51:81-93; Chiang, Russell & Braatz 2001 book) are paywalled. I could not get their tables. The ResearchGate page for the 2000 paper lists no full text.
- I found two open reproductions. Neither is a faithful copy of the 2000 setup. Use them as rough guides only.
- Several PDFs (Rato & Reis 2013 accepted manuscript at Coimbra, Braatz-group papers on web.mit.edu, Lu 2018 JPC) came back as unreadable compressed bytes or blocked. They may hold better tables. A human opening them would settle this. See "Open leads".
- UPDATE 2026-10-24: the Rato & Reis 2013 text was later extracted locally and is now read. See the section "Anchor C" below, which gives per-fault T2/Q rates at a 1% FAR on the original Braatz data. The "not read" remarks above and in "Open leads" are superseded for that paper.

## Anchor A: Sun, Paiva, Xu, Sundaram, Braatz, Bayesian RNN paper (Braatz co-author)
Source: https://arxiv.org/abs/1911.04386 (read via https://ar5iv.labs.arxiv.org/html/1911.04386), Table 1. Published in Computers & Chemical Engineering.
- Data: TEP data from Chiang et al. (2000b) (original Braatz data). 480 normal training and 960 normal validation samples. Each fault run is 160 normal samples, then 800 post-fault samples. Sampling 3 min.
- Threshold: set for a 5% FAR, validated on normal data. Not the textbook 99% limit.
- FAR on normal data: r-PCA 5.00, f-PCA 4.88, r-DPCA 5.00, f-DPCA 5.00 (%).
- Lag: stated only for r-DPCA (lag = 1). f-DPCA lag not given (UNVERIFIED).
- Statistics: r-PCA and r-DPCA use T² and Q together (alarm if either fires). f-PCA and f-DPCA use T² only. Column labels: r = reduced (parallel analysis picks the PCs), f = full. Components: r-PCA 12, f-PCA 52, r-DPCA 25, f-DPCA 104.
- Fault 21 included. Faults 3, 9, 15 included.
- The page text was read only to character 100000 of 116186; the setup was taken from that part. The reference list at the end was read separately and does not add settings.

Detection rate (FDR, %) at 5% FAR, per fault:

| Fault | r-PCA (T²+Q) | f-PCA (T²) | r-DPCA (T²+Q) | f-DPCA (T²) |
|---|---|---|---|---|
| 1 | 99.75 | 100.00 | 99.75 | 99.25 |
| 2 | 98.75 | 99.12 | 98.62 | 99.12 |
| 3 | 7.00 | 19.75 | 6.12 | 22.25 |
| 4 | 98.88 | 100.00 | 100.00 | 100.00 |
| 5 | 32.62 | 100.00 | 34.50 | 100.00 |
| 6 | 100.00 | 100.00 | 100.00 | 100.00 |
| 7 | 100.00 | 100.00 | 100.00 | 100.00 |
| 8 | 98.00 | 98.25 | 97.75 | 98.38 |
| 9 | 7.88 | 15.25 | 8.87 | 21.37 |
| 10 | 54.13 | 93.50 | 55.75 | 94.63 |
| 11 | 74.25 | 87.25 | 80.75 | 92.75 |
| 12 | 99.00 | 100.00 | 99.25 | 100.00 |
| 13 | 95.50 | 95.75 | 95.50 | 96.25 |
| 14 | 100.00 | 100.00 | 100.00 | 100.00 |
| 15 | 10.62 | 26.87 | 11.13 | 36.63 |
| 16 | 46.50 | 95.50 | 48.50 | 97.00 |
| 17 | 93.13 | 97.75 | 94.78 | 98.12 |
| 18 | 90.38 | 91.50 | 90.50 | 92.87 |
| 19 | 25.12 | 96.00 | 34.00 | 99.50 |
| 20 | 58.25 | 92.13 | 61.75 | 92.37 |
| 21 | 48.50 | 61.62 | 47.88 | 59.38 |

Reading it: f-PCA and f-DPCA keep all PCs, so T² is on 52 or 104 dimensions and it is near a Hotelling test on raw variables. Do not compare these columns with a 12-15 PC model. Note faults 3, 9, 15 sit at 6-37%, so even there the rates are low but above the 5% FAR floor.

## Anchor B: Agarwal et al., Hierarchical Deep RNN (also quoted in 2103.12222)
Source: https://arxiv.org/abs/2012.03861 (read via https://ar5iv.labs.arxiv.org/html/2012.03861), Table 4. The same table appears as Table 5 in https://arxiv.org/abs/2103.12222. The papers credit Lv et al. (2016), "Fault diagnosis based on deep learning", 2016 American Control Conference, pp. 6851-6856, for the PCA/DPCA numbers. I did not open Lv et al., and whether these numbers trace back to the Chiang book is UNVERIFIED.
- Data: 72 h per fault run, 1440 samples; first 480 for training, the rest for testing. Faults 3, 9, 15 and 21 are excluded. Onset sample not given in 2012.03861; 2103.12222 says the fault starts at sample 160 of 960.
- Threshold: NOT stated. Lag: "15 lags" for the best DPCA model per 2012.03861. FAR on normal data: not reported.
- Model sizes: PCA 15 components, DPCA 22 components. The DPCA column labels are garbled in both pages (2012.03861 shows two DPCA columns, "T²" and "I²"; the second is probably a Q-type statistic but this is UNVERIFIED). 2103.12222 shows one DPCA column.

| Fault | PCA T² | PCA SPE | DPCA first col (T²?) | DPCA second col (label garbled) |
|---|---|---|---|---|
| 1 | 99.2 | 99.8 | 99 | 100 |
| 2 | 98 | 98.6 | 98 | 98 |
| 4 | 4.4 | 96.2 | 26 | 61 |
| 5 | 22.5 | 25.4 | 36 | 100 |
| 6 | 98.9 | 100 | 100 | 100 |
| 7 | 91.5 | 100 | 100 | 99 |
| 8 | 96.6 | 97.6 | 98 | 97 |
| 10 | 33.4 | 34.1 | 55 | 78 |
| 11 | 20.6 | 64.4 | 48 | 52 |
| 12 | 97.1 | 97.5 | 99 | 99 |
| 13 | 94 | 95.5 | 94 | 94 |
| 14 | 84.2 | 100 | 100 | 100 |
| 16 | 16.6 | 24.5 | 49 | 71 |
| 17 | 74.1 | 89.2 | 82 | 89 |
| 18 | 88.7 | 89.9 | 90 | 90 |
| 19 | 0.4 | 12.7 | 3 | 69 |
| 20 | 29.9 | 45 | 53 | 87 |
| Average (17 faults) | 61.77 | 74.72 | 72.35 | 87.29 |

Caveat: the DPCA second column in 2012.03861 averages 87.29 but the text of both papers quotes 72.35 for DPCA. So treat only the first DPCA column and the PCA columns as usable, and the second DPCA column as UNVERIFIED.

The PCA T² and SPE columns have the familiar pattern: T² misses fault 4 (4.4%) while SPE catches it (96.2%); faults 5, 10, 16, 19, 20 are weak for static PCA. Those patterns are worth looking for in our results.

## Other open data points (single statistic, not full tables)
- Ji et al. 2022, Frontiers in Chemical Engineering 4:899964, https://www.frontiersin.org/journals/chemical-engineering/articles/10.3389/fceng.2022.899964/full. TEP with 500 training and 960 testing samples, onset at sample 160, PCA with 85% variance and T²/SPE at 99% confidence, 33 of 52 variables. Gives alarm delay (3 consecutive points) and FAR per fault, no FDR; faults 3, 9, 15, 21 excluded. Example: fault 1 alarm at sample 163 (FAR 1.25%), fault 19 at sample 346. Useful only for delay and 99%-limit FAR on normal data (0-3%). Which statistic the PCA row uses is not stated.
- Rato & Reis 2013, ChemoLab 125:101-108 (DPCA-DR). Abstract says the proposed statistics were best on 19 of 21 faults. Search snippet from a related 2012 AIChE abstract (https://aiche.org/conferences/aiche-annual-meeting/2012/proceeding/paper/599e-multivariate-spc-dynamic-processes-using-integrated-dynamic-principal-components-analysis-and) says limits were tuned to a 1% FAR and that all statistics failed on faults 3 and 9, with low power on 15 and 21. Per-fault numbers not read. (Superseded: see Anchor C.)

## How our setup differs (so the reviewer uses these as a guide)
| Aspect | Classic / anchor setup | Ours |
|---|---|---|
| Data | Original Braatz data: 1 normal training run (480 samples), one run per fault (960 test samples, onset at 160) | Rieth 2017: 500 runs per fault, onset at 160 in testing |
| Counting | FDR over the post-onset samples of a single run; with one run the FDR has granularity of 1/800 and large run-to-run noise | FDR over 500 runs x post-onset samples; FAR from 500 normal runs |
| Limit | Textbook 99% (theoretical F / chi-square) in the book-style tables; 5% empirical in Anchor A | Limits fitted on 100 held-out normal runs at 1% and 5% FAR; textbook 99% also reported |
| Lag | Anchor A: 1 (r-DPCA); Anchor B: 15 (probably an outlier relative to Ku et al.'s rule) | One L chosen with Ku et al. (1995), L = 0-5 |
| FAR | Often not reported for the classic tables | Reported on FaultFree_Testing |
| Faults | 21 (with 3, 9, 15, 21 sometimes dropped) | 20 (no fault 21 in Rieth) |

Consequences for the reviewer:
- Fault 21 is not in the Rieth data, so it has no anchor.
- Expect the same ordering of easy and hard faults, not the same percentages. Rieth faults differ from the Braatz faults in random seeds and run length, so exact FDRs will not match.
- Anchor A (5% limit) is the closer comparison for our 5% secondary FAR. For the 1% headline there is no open per-fault anchor that I could read. (Superseded: Anchor C is a 1%-FAR per-fault anchor, on the original Braatz data.)
- Faults 3, 9, 15: anchors give 6-37%, mostly near the FAR floor. An FDR above 50% still triggers the leakage audit.

## Open leads (not read)
- Rato & Reis accepted manuscript: https://baes.uc.pt/bitstream/10316/27393/1/Fault%20detection%20in%20the%20Tennessee%20Eastman%20benchmark%20process.pdf (host would not resolve for the fetch tool; handle https://estudogeral.uc.pt/handle/10316/27393 redirected in a loop). Likely has per-fault PCA/DPCA T² and Q at a 1% limit on the original data. Best target for a human to open. (Now read from the local extraction; see Anchor C.)
- Braatz-group PDFs on web.mit.edu (Jiang et al. 2015 CVA + FDA, DOI 10.1016/j.compchemeng.2015.03.001; Jiang JProCon 2017; Lu 2018) returned compressed bytes, so unread. They may repeat the Russell 2000 PCA/DPCA/CVA comparison.
- PHM Society deep autoencoder paper (https://papers.phmsociety.org/index.php/phmconf/article/download/3578/phmc_23_3578) reportedly compares T² and SPE over 21 faults in its Table 2; PDF unreadable.
- The PDFs fetched were kept only in the local tool cache (not in the repo); re-download them from the URLs above if needed. No PDF reader (pdftoppm, pypdf) is installed in this environment.

## Track B lead: HistorianRCA
Source: https://github.com/ScottDuncanAI/HistorianRCA (read once; author site https://scottduncanai.com/). Release date is not on the page: it shows Version 1.0 and a 2026 copyright, and the user's date (about 2026-10-21) is UNVERIFIED. An open issue asks about integrating with IBM AssetOpsBench.
- Licence: CC BY 4.0.
- Content: simulated 90 days (2024-03-01 06:00 to 2024-05-30 06:00 UTC) of a steam-heated hold tank TK-101; first 14 days are normal baseline; DCS scan period 10 s.
- Size: historian.parquet about 33 MB; historian_60s.parquet (regular 1 min) about 25 MB. Tags: 90 (30 on TK-101, 60 on neighbouring units sharing only the weather). Number of fault events: not disclosed. No alarms, logs or maintenance records, by design.
- Answer key: NOT public (withheld; available from the author on request or after submission). Scoring: +1 per fault detected, +2 more for correct cause, -1 per false positive; submissions go privately to the author.
- Use for a root-cause assistant: it is designed for agents and tests detection plus a free-text cause with supporting tags. But with a withheld key and manual scoring it cannot be scored automatically. We could run the assistant, then submit and ask for a score. It is one unit, one product and a single cause per fault, so it is a narrow test. No published assistant results.

---

# Added 2026-10-24 (phase 2 follow-up)

## Anchor C: Rato & Reis (2013), per-fault PCA and DPCA T² / Q at a 1% FAR, original Braatz data

Source: Rato, T.J. and Reis, M.S., ChemoLab 125:101-108, DOI 10.1016/j.chemolab.2013.04.002. Read from the accepted manuscript, extracted text at `research/papers/rato2013/rato2013.txt` (page numbers below are the manuscript's own numbering; the table is on manuscript page 15). Read only; nothing in that file was acted on.

### Table 2 as printed (FDR, fraction, not percent)

Columns in the extracted text: PCA T², PCA Q, DPCA T², DPCA Q, DPCA-DR T²_PREV, DPCA-DR T²_RES. Every row has exactly six numbers and the header has six labels, so the column order is not ambiguous (the T² and subscript symbols were garbled in the header text, but the order PCA, DPCA, DPCA-DR with T² before Q within PCA/DPCA is given by the "PCA DPCA DPCA-DR" line plus the "T2 Q T2 Q" line). The T²_PREV / T²_RES labels were recovered from the surrounding text on page 16.

| Fault | PCA T² | PCA Q | DPCA T² | DPCA Q | DPCA-DR T²_PREV | DPCA-DR T²_RES |
|---|---|---|---|---|---|---|
| 1 | 0.991 | 0.995 | 0.990 | 0.994 | 0.996 | 0.998 |
| 2 | 0.985 | 0.984 | 0.984 | 0.981 | 0.985 | 0.983 |
| 3 | 0.036 | 0.006 | 0.035 | 0.010 | 0.021 | 0.016 |
| 4 | 0.218 | 0.980 | 0.165 | 0.999 | 0.998 | 0.999 |
| 5 | 0.257 | 0.217 | 0.293 | 0.228 | 0.999 | 0.999 |
| 6 | 0.989 | 0.999 | 0.989 | 0.999 | 0.999 | 0.999 |
| 7 | 0.999 | 0.999 | 0.986 | 0.999 | 0.999 | 0.999 |
| 8 | 0.974 | 0.968 | 0.973 | 0.974 | 0.985 | 0.981 |
| 9 | 0.034 | 0.010 | 0.030 | 0.002 | 0.020 | 0.010 |
| 10 | 0.367 | 0.154 | 0.439 | 0.172 | 0.956 | 0.933 |
| 11 | 0.414 | 0.638 | 0.340 | 0.829 | 0.965 | 0.865 |
| 12 | 0.985 | 0.925 | 0.990 | 0.964 | 0.998 | 0.998 |
| 13 | 0.943 | 0.950 | 0.943 | 0.950 | 0.958 | 0.956 |
| 14 | 0.988 | 0.999 | 0.990 | 0.999 | 0.998 | 0.999 |
| 15 | 0.035 | 0.007 | 0.059 | 0.009 | 0.385 | 0.047 |
| 16 | 0.174 | 0.137 | 0.217 | 0.145 | 0.976 | 0.945 |
| 17 | 0.787 | 0.905 | 0.790 | 0.953 | 0.976 | 0.975 |
| 18 | 0.893 | 0.901 | 0.890 | 0.898 | 0.905 | 0.900 |
| 19 | 0.115 | 0.059 | 0.046 | 0.298 | 0.971 | 0.843 |
| 20 | 0.340 | 0.423 | 0.408 | 0.493 | 0.908 | 0.916 |
| 21 | 0.362 | 0.414 | 0.429 | 0.409 | 0.539 | 0.577 |

Boldface for the top scores in the original was lost in text extraction. The manuscript says T²_PREV was best on 14 of 21 faults and T²_RES on 9 (these overlap through ties, since 14 + 9 > 21); the abstract says 19 of 21. The text names faults 3 and 9 as the two where DPCA-DR fails, and fault 15 as hard but best for T²_PREV (0.385).

### Mean FDR, computed by me from Table 2 (not reported in the paper)

Unweighted mean of the 21 per-fault rates, and of the 18 left after dropping faults 3, 9 and 15. Hand arithmetic from the table above; a spot re-check of the sums gave the same values, but recompute in code before quoting.

| Statistic | Mean, all 21 | Mean, without 3, 9, 15 (18 faults) |
|---|---|---|
| PCA T² | 0.566 | 0.654 |
| PCA Q | 0.603 | 0.703 |
| DPCA T² | 0.571 | 0.659 |
| DPCA Q | 0.634 | 0.738 |
| DPCA-DR T²_PREV | 0.835 | 0.951 |
| DPCA-DR T²_RES | 0.807 | 0.937 |

Our project's scorecard uses 20 faults (no fault 21), so these 21-fault means are not directly comparable; use the per-fault rows for the faults in both sets.

### Limits and FAR basis
- The manuscript states the UCLs for all methods "were set to a false alarm rate of 1% under normal operation conditions, by trial and error, on a second data set with no faults" (page 14). They did this because theoretical limits give widely different observed FARs, and they wanted the methods compared at equal FAR. So the limits are empirical, tuned on a second fault-free set, not the textbook F / chi-square / Jackson-Mudholkar values. The T² (Eq. 8) and Q (Eq. 10-12) formulas are given as the starting point (theoretical), then adjusted.
- Not stated in the text I read: how the "second data set" was split from the first, its length, whether the achieved FAR on the actual fault runs was reported (no FAR column appears in Table 2), and the exact FDR definition (fraction of samples flagged after the fault, or from sample 1). The FDR definition is UNVERIFIED; with onset at sample 160 of 960 and values at 3 decimals, a per-sample fraction over 800 post-fault samples is consistent but not confirmed.

### Data
- Original Downs & Vogel simulator data set from Russell, Chiang & Braatz, "available at http://web.mit.edu/braatzgroup" (page 12): the Braatz-group data, not Rieth 2017. Controlled with the Lyman & Georgakis scheme.
- Each data set has 960 observations at 3 min; faults introduced 8 h after start (sample 160). 52 variables (41 XMEAS + 11 XMV; the constant agitator speed dropped). 21 faults (IDV 1-21; faults 16-20 listed as "Unknown").
- Training: "a data set with no faults" used to estimate the PCA, DPCA and DPCA-DR models (page 14). Its number of samples is not stated in the text (UNVERIFIED; the Braatz group's normal training file has 500 samples, my memory, not from this paper). A second fault-free set sets the limits. One test file per fault is implied (UNVERIFIED), since no replicate or averaging is mentioned.

### PCs and lags
- PCA: number of PCs by parallel analysis, giving 17 PCs. DPCA: lag count by "the approach proposed by Ku et al. [11]", PCs by parallel analysis, giving 3 lags and 29 PCs. Both said to be "in good accordance" with Russell et al. 2000 (page 14).
- Anchor point for our lag rule: Ku et al.'s rule gave L = 3 on the original TEP data in this paper (single value, all variables). Our L range 0-5 contains it.
- DPCA-DR: per-variable lags from the Rato & Reis [26] algorithm ("a succession of SVD problems and analysis of an auxiliary function"; the paper is cited as "Submitted (2012)" and not described). Table A.1 lags per variable: 3 to 17, most variables at 17 (e.g. XMEAS(1) 17, XMEAS(3) 8, XMEAS(14) 4, XMEAS(29) 3, XMEAS(39) 4, XMV(5) 15); this gives a model with 69 PCs. A summary of Table A.1: many variables hit 17, which is likely the maximum lag searched, so the lags are possibly capped (UNVERIFIED; the manuscript does not state a cap). The maximum lag allowed is not stated.
- Note for us: DPCA-DR is not planned for phase 2; the point is that its 69-PC / up-to-17-lag model is far larger than the DPCA with 3 lags, and it is the one that rescues faults 4, 5, 10, 16, 19, 20.

### What this anchor says for our results
- Hard faults at 1% FAR: faults 3, 9 and 15 are at or near the FAR floor (0.002-0.059) for PCA and DPCA in both statistics. Only DPCA-DR T²_PREV gets 0.385 on 15. So an FDR above about 0.1 on 3, 9, 15 for plain PCA/DPCA with a 1% limit would be unusual; above 0.5 stays a leakage flag.
- Pattern: for PCA/DPCA, T² misses fault 4 (0.218 / 0.165) while Q catches it (0.980 / 0.999); faults 5, 10, 16, 19, 20 are weak for all four (0.05-0.5, except Q on 5 and similar low values). Faults 1, 2, 6, 7, 8, 12, 13, 14 are above 0.92 in all four.
- Within PCA vs DPCA, differences are small except fault 11 (Q: 0.638 vs 0.829), fault 17 Q (0.905 vs 0.953), fault 19 Q (0.059 vs 0.298). The paper's own paired t-tests (Table 3, 5% level) found no significant difference between any two of PCA T², PCA Q, DPCA T² and DPCA Q except PCA Q vs DPCA Q (p = 0.046).
- Single run per fault in the original data means each cell carries run noise; expect our 500-run averages to differ by a few points on the mid-range faults.

## Ku, Storer & Georgakis (1995), lag-selection procedure (VERIFIED against the paper's text)

Source: Ku, Storer & Georgakis, ChemoLab 30:179-196 (1995), section 4 "The design procedure", extracted text `research/papers/ku1995/ku1995.txt` (PDF page 8-10 of the extraction; journal pages 186-188). Table 2 is on journal page 191. Updated 2026-10-24: the earlier reconstruction (from memory, plus search summaries) is replaced by what the paper says. Read only.

### The procedure, as printed (steps 1-10)
Verbatim from the paper (text extraction; equation 31 is garbled in the extraction but readable):
1. Set l = 0.
2. Form data matrix X = [X(k) X(k-1) ... X(k-l)].
3. Perform PCA and calculate all the principal scores.
4. Set j = n x (l+1) and r(l) = 0.
5. Determine if the jth component represents a linear relation. If yes proceed, if no go to step 7.
6. Set j = j - 1 and r(l) = r(l) + 1, repeat 5.
7. Calculate the number of new relationships: r_new(l) = r(l) - sum_{i=0}^{l-1} (l - i + 1) r_new(i)   (Eq. 31)
8. If r_new(l) <= 0, go to step 10, otherwise proceed.
9. Set l = l + 1, go to step 2.
10. Stop.

Programmer's reading: n is the number of original variables (m = 52 in our notation). Steps 4-6 walk from the last (smallest-variance) component upward and count how many consecutive trailing components are "relations"; so r(l) = n(l+1) - a(l), where a(l) is the number of retained PCs. Step 7 subtracts the relations already found at earlier lags, each repeated (l - i + 1) times at lag l. The text also notes: "At each l-level, r_new might have a negative value for some value of l" (page 187) and the toy example says "these static relations will be repeated (l + 1) times in X_l".

### Checks against my earlier reconstruction
- r_new formula: CONFIRMED (Eq. 31 matches my formula). At l = 0 the sum is empty, so r_new(0) = r(0).
- Stopping rule: CONFIRMED. Stop when r_new(l) <= 0 (step 8). Note "<= 0", so a zero or negative count stops it.
- l versus l - 1: CONFIRMED that the answer is the last lag at which new relations appeared, i.e. l - 1 at the stopping level. Evidence: the four-variable toy example finds r_new = 0 at l = 2 and says "the procedure should stop at level 2 and [X(k) X(k-1)] will be sufficient" (page 188), so the model uses l = 1. In the TE example (Table 2, below) r_new is -3 at l = 3 and the authors "choose l = 2". The sentence "The result shows that choice of l = 2 will cover all the linear relations" confirms it.
- Cap at l = 5 (my addition, not in the paper): the paper has no cap. It says l is usually 1 or 2 ("The number l usually is 1 or 2 which indicates the order of the dynamic system"), higher for nonlinear systems. The cap is our project rule, not Ku's.

### Worked check of Eq. 31 on Ku's own Table 2 (normal TE data, auto-scaled, parallel analysis for the PC count)
| l | No. of variables | No. of PCs | r(l) | r_new(l) |
|---|---|---|---|---|
| 0 | 53 | 11 | 42 | 42 |
| 1 | 106 | 15 | 91 | 7 |
| 2 | 159 | 17 | 142 | 2 |
| 3 | 212 | 22 | 190 | -3 |

Re-computed by me from the formula: r_new(1) = 91 - 2x42 = 7; r_new(2) = 142 - 3x42 - 2x7 = 2; r_new(3) = 190 - 4x42 - 3x7 - 2x2 = -3. All match the printed column, so the formula and my reading of the table are consistent. Ku's choice: l = 2 with 17 PCs. Note Ku used 53 variables (his TE data set), not our 52.

### How the number of PCs or relations is decided at each lag
- The procedure does not contain its own rule. The paper says: "This procedure assumes that a method for determining the number of the principal components or the linear relations has been chosen." (page 186-187).
- Suggested rule: "We suggest to use parallel analysis and the cross-correlation plots of the scores to determine the number of the principal components. The number of linear relations, r, will be the total number of the variables minus the number of the principal components." Also: "Another way is to set a threshold for small singular values. It is important to check if the remaining scores are completely independent using the auto- and cross-correlation plots."
- In the TE application they used parallel analysis for the PC count at each l and the correlation plots (99% limits, +/-30 lags, Fig. 11) to verify.
- Threshold for "small": only a symbol is given, sigma_k < sigma(E), "some threshold for small singular values" (Eq. 10, page 183). The paper's theory (Theorem 1, Weyl-type bound): singular values at or below ||E||_2 (largest singular value of the noise matrix) are noise, but ||E|| is unknown and "needs to be constructed"; no numeric default is given. So no numeric threshold is pinned down by the paper.
- Warning in the paper (page 182): "Our experience indicates that when autocorrelation and nonlinearities are present in the data ... the methods just discussed [eigenvalue > 1, scree, parallel analysis, cross-validation] can be unreliable. This seems especially true for cross-valida[tion]..." (the sentence runs onto the next page of the extraction; I did not read the rest). They still use parallel analysis in practice and add the correlation-plot check. Whether cross-validation is rejected outright is therefore not fully confirmed here.
- Cumulative-variance rules are not mentioned in the paper (the methods it lists are eigenvalue > 1, scree, parallel analysis, cross-validation).

### Is "a(l) = fewest PCs reaching 90% cumulative variance" compatible?
- Mechanically yes. The paper allows any method for the PC count ("assumes that a method ... has been chosen"), and step 4-6 reduce to r(l) = n(l+1) - a(l). So a 90%-variance a(l) can be plugged in at every l and Eq. 31 and the stopping rule applied unchanged.
- It is not the paper's recommended method, and not tested in it. Three cautions (my judgment, not from the paper):
  - Ku's logic needs the discarded components to be actual near-zero-variance relations. A fixed 90% variance cut leaves the discarded 10% of variance as "relations", which can include real but small process variation. Then r(l) is not a count of constraints, and r_new may not behave like the clean 42, 7, 2, -3 pattern.
  - Because 10% of the variance is dropped at every l and the number of columns grows by 52 per lag, a(l) and r(l) may grow steadily, so r_new(l) can stay positive until the l = 5 cap, or turn non-positive early because of noise. Either way the stopping lag then reflects the 90% rule, not the data's dynamics.
  - A consistent a(l) rule at all l is required (any rule), so 90% is fine in that respect.
- Recommendation: use the 90% rule for the final PC count if the project has fixed it, but when choosing L, print the r(l), a(l), r_new(l) table for the 90% rule next to parallel analysis (the method the paper used and Rato & Reis used on the same data: L = 3 and 29 PCs for DPCA; Ku's own run gave L = 2). If they disagree on L, report both and record which one the ADR used. This is untested; I have not run the 90% rule on any data.

### Remaining uncertainties
- The extraction garbles some symbols (Eq. 10, Eq. 31 sub/superscripts); the formula was cross-checked numerically against Table 2, which settles it.
- Sentence after "especially true for cross-valida" (page 182) was not read in full.
- The paper does not give a numeric threshold for a "small" singular value; common choices are in the earlier list (parallel analysis, cross-validated reconstruction error, cumulative variance, eigenvalue > 1, relative-to-largest or gap).

Sources used:
- Local: `research/papers/ku1995/ku1995.txt` (Ku, Storer & Georgakis 1995, ChemoLab 30:179-196, extracted copy of a court exhibit of the paper)
- Local: `research/papers/rato2013/rato2013.txt` (Rato & Reis 2013 accepted manuscript, DOI 10.1016/j.chemolab.2013.04.002)
