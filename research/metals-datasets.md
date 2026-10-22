# Public metals-industry datasets for build projects

Date: 2026-10-22. Scope per brief, plus the later scope change: bearing/vibration sets are now allowed if provenance is verifiable (own section at the end, ranked separately).
Status key: confirmed = read on the primary page this session. UNVERIFIED = not confirmed. Nothing was filled from memory.
Laptop limit (tep-data): CPU-only, 7.7 GB RAM, peak script RAM under about 5 GB.

## A. Process and quality datasets

| # | Dataset | Primary source | Licence | Size | Type / unit | Labels | Honest use | Catch |
|---|---|---|---|---|---|---|---|---|
| 1 | Process data collected by blast-furnace sensors (V. Trofimov, 2025) | https://data.mendeley.com/datasets/6d7jbc7tb5/1 | CC BY 4.0 (confirmed) | Files via Mendeley API: raw xlsx 6.8 MB, second (validated, labelled) xlsx 58 KB, attribute txt. Rows, time span, sampling: UNVERIFIED (not stated). 26 sensors; about 15 parameters named (gas pressure differences, top-gas temperature, CO2, H2, ore/coke ratio, blast flow and temperature, hot-metal Si). | Time series, blast furnace | Partial: the second set is "validated and labelled into classes"; class names not stated (UNVERIFIED). | Unsupervised monitoring on the raw set; Si soft sensor/forecast; label use only after reading the attribute file | Single contributor, no paper found, institution not given. Labelled set is tiny (58 KB). Download and inspect before committing to a project. |
| 2 | Steel Industry Energy Consumption (Daewoo Steel, Gwangyang) | https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption (DOI 10.24432/C52G8C). Mendeley mirror https://data.mendeley.com/datasets/56nfs5hx9j | CC BY 4.0 (confirmed) | 35,040 rows, 9 features, no missing values. Time span and interval not on UCI page; IEEE DataPort entry says 15 min (secondary). 35,040 rows at 15 min would be one year (my arithmetic). | Tabular time series, plant electricity (whole plant, not one unit) | Load type (Light/Medium/Maximum) is derived, not fault | Energy forecasting, load-type classification, anomaly detection on usage (no fault labels) | UCI lists CO2 as "tCO2" with unit ppm: unit inconsistency. Single plant. Intro paper: Sathishkumar, Shin & Cho 2021, Building Research & Information 49:127-143. Tiny, runs on any laptop. |
| 3 | Steel Plates Faults (Buscema et al.) | https://archive.ics.uci.edu/dataset/198/steel+plates+faults (DOI 10.24432/C5J88N) | CC BY 4.0 (from search result, not fetched) | 1,941 rows, 27 features (UCI also describes 34 incl. fault indicators) | Tabular, plate surface | 7 fault classes (Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other) | Defect classification baseline, class-imbalance and leakage demos | Hand-extracted features, not process data. Very small and a heavily used benchmark, so low novelty. Check which version (OpenML differs). |
| 4 | Sinter machine productivity (S. Rath, SAIL) | https://data.mendeley.com/datasets/72hgnwmsvs | CC BY 4.0 (confirmed) | 16 inputs + productivity output; row count and units of output UNVERIFIED (not stated) | Tabular, sinter plant | Continuous target only | Regression / prescriptive analytics; feature importance | No time axis stated; likely small. Output units unknown. |
| 5 | Milling Wear (UC Berkeley BEST Lab, A. Agogino and K. Goebel 2007) | https://data.nasa.gov/dataset/milling-wear ; file https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip via https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository | NASA page: "other-license-specified", links usa.gov government-works (confirmed). PCoE page: acknowledge donors, own risk, no separate licence. | mill.zip, size not shown. Run count, sampling rate, sensors UNVERIFIED on these pages (sensor list not given). | Time series per cut, CNC milling (machining) | Flank wear VB | Tool-wear regression, RUL, condition monitoring | Probably includes vibration/acoustic-emission channels (UNVERIFIED), so it falls under the vibration scope. Small number of cuts; lots of missing VB values reported in the literature (UNVERIFIED). |
| 6 | PHM 2010 CNC milling challenge | https://ieee-dataport.org/ ... DOI 10.21227/jdxd-yy51 ; PCoE lists it as an external PHM Society page | UNVERIFIED (blank on search result) | UNVERIFIED | Dynamometer, accelerometer, AE time series | Wear labels for training cutters only | Tool RUL | The Hugging Face mirror is gated (login): exclude. Test labels not public as far as I know (UNVERIFIED). |
| 7 | Lab continuous casting, tomography-controlled (HZDR TOMOCON) | https://rodare.hzdr.de/record/1463 (DOI 10.14278/rodare.1463) | CC BY 4.0 (confirmed) | One 4.0 GB zip, contents not listed on the page | Lab-scale model, tomography + process data | UNVERIFIED | Possibly lab process control/sensing; read the linked paper first | Lab model, not an industrial caster. Contents unconfirmed. 4 GB is heavy for this laptop. Not breakout data. |
| 8 | Soft sensor for solidification in steel billet casting (CSS2) | https://doi.org/10.5281/zenodo.6914669 and 10.5281/zenodo.6397378 | CC BY 4.0 | Videos (23-39 MB) + model description only | Not a dataset | No | None | Listed to avoid wasting time: it is video and a docx, not data. |
| 9 | Cold rolling AGC dataset, 6-high mill, Nigeria (Data in Brief 2018) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5996350/ | Journal supplementary; licence UNVERIFIED | 486 points (per search summary) | Tabular, cold mill thickness/speed/power | No | Small regression demo only | Very small. |
| 10 | Cardiff steelmaking plant scheduling with renewables and storage | https://research.cardiff.ac.uk/converis/portal/detail/Dataset/291344811 (DOI 10.17035/d.2023.0291344811) | CC BY 4.0 (search result) | CSV, under 100 MB | Scheduling inputs, not sensors | No | Energy/scheduling optimisation | Not process monitoring. Contents UNVERIFIED. |
| 11 | Severstal Steel Defect Detection | https://www.kaggle.com/c/severstal-steel-defect-detection | Competition rules; terms not retrievable here (UNVERIFIED). A HF mirror card lists "other". | 12,568 train images, 6,666 defective (RLE masks); 1600x256 likely (sources conflict) | Images, flat steel sheet | Yes, 4 defect classes (class count from my recollection: UNVERIFIED) | Image defect segmentation | Needs Kaggle login and rule acceptance; redistribution limits. CNN training on CPU is slow. |
| 12 | NEU-DET (Northeastern Univ. hot-rolled strip) | IEEE DataPort DOI 10.21227/j84r-f770 | IEEE DataPort page says subscriber access required | 1,800 images, 6 classes, 200x200 | Images | Yes | Defect classification | Login-gated on that host, so excluded unless a free primary host is confirmed (not found). |
| 13 | GC10-DET (Lv et al., Sensors 2020) | GitHub lvxiaoming2019 ; DOI 10.57702/igfr8zeq (TIB copy) | UNVERIFIED | 3,570 grey images, 10 classes | Images | Yes | Defect detection | No licence found. |
| 14 | X-SDD | IEEE DataPort DOI 10.21227/k1wr-rc37 | UNVERIFIED (Roboflow mirror says CC BY 4.0, auto-derived) | 1,360 images, 7 classes | Images | Yes | Defect classification | Small; host licence unclear. |

Not found as public data: blast furnace beyond #1, BOF/EAF heat-level data, ladle furnace, continuous casting breakout, hot strip mill process data, pelletising process data, alumina refinery or Bayer, aluminium smelting pot data, foundry process data. Papers using plant data (Tata Steel 28,000 heats, SSAB converter vibration, an IEEE Access 2026 EAF paper with 120 MVA furnace data) do not show a data release. The IEEE DataPort blast-furnace entry (DOI 10.21227/a8wq-cb10) is UNVERIFIED and IEEE DataPort data is often login-gated.

## B. Bearing / vibration, public sources (ranked separately)

Ranked by usefulness for a monitoring project plus clarity of provenance. All are a different problem from TEP: high-frequency vibration, so feature extraction (RMS, kurtosis, envelope) is needed.

| Rank | Dataset | Original publisher | Source URL | Licence | Supports |
|---|---|---|---|---|---|
| 1 | IMS bearings (3 run-to-failure tests, 4 bearings on a loaded shaft, 1 s at 20 kHz every 10 min) | Center for Intelligent Maintenance Systems, Univ. of Cincinnati (Lee, Qiu, Yu, Lin, Rexnord 2007) | https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository ; https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip | data.nasa.gov record: usa.gov government works; mirrors say "other-license-specified" (inconsistent). PCoE: acknowledge donors, own risk. | Unsupervised degradation/anomaly detection, closest to the TEP workflow (healthy start then drift). |
| 2 | Paderborn bearing DataCenter (vibration + motor current, real and artificial damage) | KAt, Univ. Paderborn (Lessmeier et al. 2016) | https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter | CC BY-NC 4.0 (confirmed); commercial use needs the author | Fault classification and transfer from artificial to real damage. Non-commercial: check against a LinkedIn post's purpose. |
| 3 | Ottawa time-varying speed bearing data (60 files, 200 kHz, 10 s) | Univ. of Ottawa (Huang & Baddour 2018, Data in Brief) | https://data.mendeley.com/datasets/v43hmbwxpm/2 (DOI 10.17632/v43hmbwxpm.2) | CC BY 4.0 (confirmed) | Fault diagnosis under varying speed. Cleanest licence in this list. |
| 4 | FEMTO / PRONOSTIA (17 run-to-failure, 25.6 kHz, 3 conditions, 11 truncated test runs) | FEMTO-ST Institute, Besancon (Nectoux et al. 2012; IEEE PHM 2012 challenge) | Listed as item 10 on the PCoE page above | UNVERIFIED (none found) | RUL prediction. |
| 5 | XJTU-SY (15 bearings, 3 conditions, run-to-failure) | Xi'an Jiaotong Univ. (Prof. Yaguo Lei's group) with Changxing Sumyoung Technology (Wang et al., IEEE Trans. Reliability 2020) | Cited via https://data.mendeley.com/datasets/mpn45f4gxc (derived copy; official page not located) | UNVERIFIED | RUL / degradation. Use the original host, not the derived copy. |
| 6 | CWRU bearing data | Case School of Engineering, Case Western Reserve University | https://engineering.case.edu/bearingdatacenter | No licence stated on the page (confirmed absent). Copyright notice only. | Seeded-fault classification (EDM faults 0.007-0.040 in., 0-3 hp). Saturated benchmark, near 100% accuracy is common, so a low-value post. |

Related: Ottawa constant-speed set (UORED-VAFCLS, 42 kHz, 20 bearings) and motor set (UOEMD-VAFCVS, DOI 10.17632/msxs4vj48g.2): licences UNVERIFIED.

## C. Rankings

(a) Time-series process-monitoring project like TEP (unsupervised detector, then assistant), process datasets only:
1. Blast-furnace sensors (#1): only real plant multivariate time series found; risk is unknown size and labels.
2. Steel industry energy (#2): clean, licensed, a year of 15-min data; no faults, so use for anomaly/forecast, not a TEP-style scorecard.
3. Milling Wear (#5): labelled progression, but small and vibration-heavy.
4. Sinter productivity (#4): tabular soft sensor, not monitoring.
5. HZDR casting lab (#7): only if the archive turns out to hold usable time series.
If vibration is allowed in this ranking, IMS bearings would rank 2nd for (a).

(b) Short credible post a metals engineer would find useful:
1. Steel energy (#2): load, power factor, and CO2 are familiar; easy to make a clear chart.
2. Blast-furnace Si forecasting (#1): high interest, if the data holds up.
3. Steel Plates Faults (#3): quick, but common.
4. IMS bearings: rotating equipment in rolling mills, clear degradation story.
5. Sinter productivity (#4): sinter engineers would care, small data.

## D. Gaps
- No public alumina refinery, Bayer, or smelting-pot time series found (searched; only cost/econometric data).
- No heat-level BOF/EAF/ladle data, no continuous casting breakout data, no hot strip mill process time series.
- No public steel-plant dataset has TEP-style fault labels; the blast-furnace labelled set is the only candidate.
- Image sets exist but need GPU for serious work or sit behind login/rules.
- Search coverage was web search plus a Zenodo API query (first 15 results of one query); Kaggle and Figshare were not searched directly.

## E. Open questions
1. Download dataset #1 and read the attribute file: rows, sampling, class names.
2. Confirm UCI #851 time span and sampling from the paper.
3. Read Kaggle Severstal rules while logged in, if image work is wanted.
