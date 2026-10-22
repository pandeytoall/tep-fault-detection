# Decision G8: where runs execute (laptop vs Claude Code cloud session)

## Proposal (orchestrator's recommendation)
Hybrid workflow:
1. **Laptop** (7.7 GB RAM, 4c/8t, no GPU): grilling, specs, reviews (`/ml-review`), post drafts, and building/testing each phase on a subset of runs (e.g. 20 runs per fault).
2. **Cloud session** (about 4 vCPU, 16 GB RAM, 30 GB disk, no GPU; files don't persist between sessions; the session clones the public GitHub repo): full-data runs, meaning conversion of all 4 `.RData` files, scoring on all 500 runs x 20 faults, long sweeps (e.g. DPCA lag windows), and independent model phases built in parallel sessions on separate branches.
3. Each cloud session re-downloads the data (1.4 GB in total: Faulty_Testing 837 MB, Faulty_Training 494 MB, fault-free files 47 and 25 MB) and pushes only small results (`results/`, figures, code). Data is never committed.
4. Phase 0 adds: `requirements.txt` (cloud setup script: `pip install -r requirements.txt`), one idempotent download+convert command that runs on both machines, and a `--runs-per-fault N` option so the same command does the laptop subset or the cloud full run.
5. User setup: a cloud environment with a Custom network allowlist of `dataverse.harvard.edu` and `dvn-cloud-iqss.s3.amazonaws.com`.

## Why
- The laptop's hard limit is converting Faulty_Testing: pyreadr loads it all as float64, 9.6M rows x 55 cols x 8 B = about 4.2 GB, estimated 6-8 GB peak with pandas copies (ESTIMATE, not measured). Training on float32 parquet subsets fits the laptop.
- The cloud VM costs nothing extra; sessions consume plan usage (tokens). So cloud time is cheap when the session runs a complete spec and waits on background jobs.

## Evidence
- Cloud facts: code.claude.com/docs/en/cloud-environments and /claude-code-on-the-web
- Dataset, labels, evaluation and laptop rules: the `tep-data` skill (preloaded)
- Workflow: `CLAUDE.md` (phase loop, builder, review gate)
- Scope agenda: `.scratch/phase-R/scope-proposal.md`

## Known unverified
- The 6-8 GB conversion peak (not measured).
- The S3 redirect working through a Custom allowlist (not tested in the cloud).
- Long jobs: foreground commands are capped at 10 min; background jobs at about 30 min more (from the docs, not tested).

## Alternatives
A. Laptop only: convert Faulty_Testing per fault or in chunks if pyreadr allows; otherwise subsample.
B. Cloud only, for all execution.
C. Hybrid (proposal).

## Comments
**Critic (opus, high), 2026-10-22: AGREE WITH CHANGES.** Strongest objection: the only laptop blocker is the unmeasured one-time conversion; the hybrid adds a second environment that the repo can't run in yet (Windows-only paths and hooks; the user-scope `mattpocock-skills` plugin isn't loaded in cloud sessions). Verified: setup-script output is cached for about 7 days when setup finishes in about 5 min, so the download can sit in the setup script; background limits can be raised with `BASH_*_TIMEOUT_MS`. Required changes:
1. Measure the laptop conversion first (peak RAM logged). Under about 5 GB means laptop for everything except parallel sessions.
2. Download in the setup script (cached), not every session.
3. Cross-platform paths and hooks; skills the builder needs available in the cloud.
4. Keep private-derived text out of tracked files before pushing (done: `private/grilling-agenda.md`).
5. Headline metrics (FDR, FAR, delay, train time, peak RAM) for all models come from one machine, and each result records its machine.
6. Pinned versions in `requirements.txt`; the laptop subset is a deterministic seed-42 choice of run IDs.

**User, 2026-10-22:** adopt the plan if it checks out. Adopted as C with changes 1-6.
