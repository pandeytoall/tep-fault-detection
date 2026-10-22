# PCA / DPCA sanity anchors for phase 2 (D8) and HistorianRCA lead

Researched 2026-10-22. Short pass. Every number below was read from an open web page; anything not confirmed is marked UNVERIFIED.

## Bottom line
- The original sources (Russell, Chiang & Braatz 2000, ChemoLab 51:81-93; Chiang, Russell & Braatz 2001 book) are paywalled. I could not get their tables. The ResearchGate page for the 2000 paper lists no full text.
- I found two open reproductions. Neither is a faithful copy of the 2000 setup. Use them as rough guides only.
- Several PDFs (Rato & Reis 2013 accepted manuscript at Coimbra, Braatz-group papers on web.mit.edu, Lu 2018 JPC) came back as unreadable compressed bytes or blocked. They may hold better tables. A human opening them would settle this. See "Open leads".

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
- Rato & Reis 2013, ChemoLab 125:101-108 (DPCA-DR). Abstract says the proposed statistics were best on 19 of 21 faults. Search snippet from a related 2012 AIChE abstract (https://aiche.org/conferences/aiche-annual-meeting/2012/proceeding/paper/599e-multivariate-spc-dynamic-processes-using-integrated-dynamic-principal-components-analysis-and) says limits were tuned to a 1% FAR and that all statistics failed on faults 3 and 9, with low power on 15 and 21. Per-fault numbers not read.

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
- Anchor A (5% limit) is the closer comparison for our 5% secondary FAR. For the 1% headline there is no open per-fault anchor that I could read.
- Faults 3, 9, 15: anchors give 6-37%, mostly near the FAR floor. An FDR above 50% still triggers the leakage audit.

## Open leads (not read)
- Rato & Reis accepted manuscript: https://baes.uc.pt/bitstream/10316/27393/1/Fault%20detection%20in%20the%20Tennessee%20Eastman%20benchmark%20process.pdf (host would not resolve for the fetch tool; handle https://estudogeral.uc.pt/handle/10316/27393 redirected in a loop). Likely has per-fault PCA/DPCA T² and Q at a 1% limit on the original data. Best target for a human to open.
- Braatz-group PDFs on web.mit.edu (Jiang et al. 2015 CVA + FDA, DOI 10.1016/j.compchemeng.2015.03.001; Jiang JProCon 2017; Lu 2018) returned compressed bytes, so unread. They may repeat the Russell 2000 PCA/DPCA/CVA comparison.
- PHM Society deep autoencoder paper (https://papers.phmsociety.org/index.php/phmconf/article/download/3578/phmc_23_3578) reportedly compares T² and SPE over 21 faults in its Table 2; PDF unreadable.
- The PDFs fetched were saved under C:\Users\pande\.claude\projects\D--ML-model-training\...\tool-results\ if someone with a PDF text extractor wants them. No PDF reader (pdftoppm, pypdf) is installed in this environment.

## Track B lead: HistorianRCA
Source: https://github.com/ScottDuncanAI/HistorianRCA (read once; author site https://scottduncanai.com/). Release date is not on the page: it shows Version 1.0 and a 2026 copyright, and the user's date (about 2026-10-21) is UNVERIFIED. An open issue asks about integrating with IBM AssetOpsBench.
- Licence: CC BY 4.0.
- Content: simulated 90 days (2024-03-01 06:00 to 2024-05-30 06:00 UTC) of a steam-heated hold tank TK-101; first 14 days are normal baseline; DCS scan period 10 s.
- Size: historian.parquet about 33 MB; historian_60s.parquet (regular 1 min) about 25 MB. Tags: 90 (30 on TK-101, 60 on neighbouring units sharing only the weather). Number of fault events: not disclosed. No alarms, logs or maintenance records, by design.
- Answer key: NOT public (withheld; available from the author on request or after submission). Scoring: +1 per fault detected, +2 more for correct cause, -1 per false positive; submissions go privately to the author.
- Use for a root-cause assistant: it is designed for agents and tests detection plus a free-text cause with supporting tags. But with a withheld key and manual scoring it cannot be scored automatically. We could run the assistant, then submit and ask for a score. It is one unit, one product and a single cause per fault, so it is a narrow test. No published assistant results.
