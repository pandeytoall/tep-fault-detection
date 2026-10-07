# Phase R: proposed scope (for user approval)

Evidence: `research/tep-benchmarks.md`. Approve or change via `/grill-with-docs`.

## What the research settled
- Faults 3, 9, 15 are near-undetectable in every source that splits correctly. The two 60-77% claims are unverified and look like leakage; ignore them.
- Deep learning is out: a Rieth-data benchmark needed about 24 h per method (arXiv:2303.05904).
- Rieth-data papers train supervised models on a few runs per fault and test on all 500 (SensorSCAN, arXiv:2208.08879).

## Gap
No source gives per-fault PCA T²/SPE FDR at 1% FAR on the Rieth data. The only Rieth PCA number (TPR 0.36) is PCA + k-means clustering, not T²/SPE (checked in the paper). The classic tables (Chiang/Russell/Braatz 2000, Yin 2012) are paywalled and use the original single-run data. So our phase 3 PCA result becomes the reference.

## Decisions
| # | Decision | Recommended | Alternatives |
|---|---|---|---|
| D1 | ML framing | Two tracks reported separately: supervised (trained on faulty runs) and unsupervised (normal only, same footing as PCA) | Supervised only (looks better than PCA, but only on the 20 known faults); unsupervised only |
| D2 | ML models | HistGradientBoosting (supervised) + Isolation Forest (unsupervised), both scikit-learn | Add Random Forest as a second supervised model |
| D3 | Supervised task | Binary detection (normal vs any fault) | 21-class classification (diagnosis, not what FDR/FAR measure) |
| D4 | MSPC | PCA T²/SPE + DPCA | PCA only; add CVA/ICA/SFA (more code, no Rieth evidence) |
| D5 | FAR target | 1%. Every method's threshold set on held-out normal validation runs; MSPC also reports the textbook 99% limit | 5% (matches some DL papers, not MSPC practice) |
| D6 | Dynamics | One lag window shared by DPCA and the ML features, picked on validation runs; add mean detection delay as a metric | No lags (static only) |
| D7 | Data budget | Train: all 500 normal runs + 50 runs per fault (raise it if phase 2 timing allows). Test: all 500 runs per fault, read in chunks | 5-25 runs per fault (SensorSCAN precedent) |
| D8 | Second research pass | Short, targeted: open-access copies of the classic per-fault PCA/DPCA tables, as sanity anchors for the phase 3 review | Skip; rely on known patterns and leakage checks |

Leakage guard for every phase: FDR above 50% on fault 3, 9 or 15 triggers an audit before it's reported.
