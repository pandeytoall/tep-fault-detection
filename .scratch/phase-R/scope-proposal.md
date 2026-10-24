# Phase R: proposed scope (for user approval)

Evidence: `research/tep-benchmarks.md`. Approve or change via `/grill-with-docs`.

## What the research settled
- Faults 3, 9, 15 are near-undetectable in every source that splits correctly. The two 60-77% claims are unverified and look like leakage; ignore them.
- Deep learning is out: a Rieth-data benchmark needed about 24 h per method (arXiv:2303.05904).
- Rieth-data papers train supervised models on a few runs per fault and test on all 500 (SensorSCAN, arXiv:2208.08879).

## Gap
No source gives per-fault PCA T²/SPE FDR at 1% FAR on the Rieth data. The only Rieth PCA number (TPR 0.36) is PCA + k-means clustering, not T²/SPE (checked in the paper). The classic tables (Chiang/Russell/Braatz 2000, Yin 2012) are paywalled and use the original single-run data. So our phase 2 PCA result becomes the reference.

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

## Grilling agenda (added 2026-10-22)
Open scope questions, alongside D1-D8. Private questions (posts, logistics) are in `private/grilling-agenda.md`.
| # | Question | Current scope | Alternative to weigh |
|---|---|---|---|
| G1 | Deep learning? | Out (D2) | A small dense autoencoder on subsampled runs (minutes on CPU); LSTM autoencoder or 1D-CNN (slower, needs torch) |
| G2 | Phase order | ML (phase 2) before MSPC (phase 3) | PCA first, as the baseline every model is compared against |
| G3 | Model list | HistGB, Isolation Forest, PCA, DPCA | + ICA, XGBoost or RF, the DL models in G1 |
| G4 | Scorecard | FDR, FAR, train time, peak RAM; delay optional (D6) | Detection delay to the first *sustained* alarm (define it, e.g. k consecutive samples); one `results/scorecard.csv` for all models |
| G5 | Missing steps | none | Map the 52 variables to plant units and plot normal vs fault 1; a shared `evaluate.py` tested on a dummy model; a study of faults 3/9/15; an operator-style report |
| G6 | Model interface | not considered | One callable function per model in `src/`, so a later tool-calling assistant can use them |

## G8 decided: where runs execute
Brief and critic review (AGREE WITH CHANGES): `.scratch/phase-R/decision-cloud.md`. See the `Where runs execute` section in `CLAUDE.md`.

## Agreed scope (2026-10-22, approved by the user after grilling and a critic review)
The rules (run budget, alarm limits, tracks, lags, delay, unseen-fault test) are in the `tep-data` skill; the phases are in `CLAUDE.md`; the terms are in `CONTEXT.md`; the reasons are in `docs/adr/0001` and `0002`. In short:
- D1 Two tracks, reported separately. D3 Supervised = one 21-class model (alarm score = 1 - P(normal)).
- D2/G1/G3 Unsupervised: PCA, DPCA, Isolation Forest, a small dense autoencoder. Supervised: HistGradientBoosting. ICA, LSTM autoencoder and 1D-CNN are in the backlog.
- D4/G2 MSPC first (phase 2), ML second (phase 3).
- D5 FAR 1% headline, 5% secondary; limits set on 100 held-back normal training runs; MSPC also at the textbook 99% limit.
- D6 One lag window L for all lagged models, Ku et al. (1995) rule on normal runs, L = 0-5.
- G4 Detection delay to the first 3-in-a-row alarm after onset; median over detected runs, share never detected, fault-free baseline.
- D7 400/100 normal training runs; 50 faulty training runs per fault for supervised training; 20 other faulty training runs per fault for development; final scores on all test runs. Testing files are touched once.
- G5 Extra steps folded into phases 1, 4 and 5. G6 One shared interface per model (fit, score a run, alarm limit).
- Unseen-fault test: faults 1, 8, 13, 14, 16 (the lowest of each Downs & Vogel type, skipping 3/9/15), checked in the second research pass.
- D8 Second research pass runs during phase 0 (classic per-fault PCA/DPCA tables; fault-type groupings).
- If behind: cut the autoencoder, then the unseen-fault test. Track B (the assistant) comes after phase 5.
- The plan changes when results or limits demand it: propose, record here under Comments, then continue.

## Comments
- 2026-10-22, orchestrator, user agreed: phase 0 adds the normal vs fault 1 chart (post 1's image). Cloud readiness (cross-platform paths, hooks, cloud smoke test) only happens if the phase 0 RAM log shows a file over the limit.
- 2026-10-22, user agreed: Track B moves forward. A new phase 2b "Assistant v1" (MCP server over a replayed historian, alarm log and plant documents, with PCA as its only model tool) comes right after phase 2. Later model phases add their model as a tool. The full scorecard arrives about 1-2 weeks later.
- 2026-10-22, user agreed (critic: agree with changes): D7 run budget replaced. Run k shares its noise across all files within a family, so one disjoint seed-42 partition of run IDs applies to every training file and fault: fit 330 / limit 100 (limit_tune 50) / supervised 50 / dev 20. The supervised budget is capped at 50 runs per fault. Rules are in the `tep-data` skill; evidence is in `.scratch/phase-1/run-pairing.md`.
- 2026-10-24, user agreed: the assistant grows in versions after phase 2b. v2 adds a knowledge graph of TEP (units, streams, readings, control loops; built from the variable map and flowsheet). v3 adds RAG over a large public document set. Each version is scored on diagnosis accuracy over hidden faults. Timing: after phase 4, before the launch.
