# TEP fault detection: orchestrator setup

## Who and why
Rahul builds industrial AI for process plants (metals and alumina) in his day job. This side project learns the main fault-detection models hands-on, on the public Tennessee Eastman data only, and feeds a LinkedIn post series.
- **Public data only.** Never use, ask for or guess employer or client data, company names, plant names, tag names or screenshots. This repo is public on GitHub.
- **Private material lives in `private/` (git-ignored):** `NOTES.md` (goals, tracks A-D, open questions), `PLAN.md` (model list, build steps, post series, backlog), `linkedin/` (post analysis, scrapers, raw feeds) and `posts/` (drafts). Read them when planning scope or drafting posts; quote nothing from them into tracked files.
- **Posts:** only what we build, how it works and the science behind it; no companies, vendors, industry news or opinion. The `editor` drafts; Rahul approves; publish only after a fresh "yes" for that post.
- **Browsers:** pages that need Rahul's logins (LinkedIn) are read in the main session with Claude in Chrome and saved to `private/linkedin/` for the editor. The `browser-use` skill runs only in its isolated, logged-out Chrome (`bu-chrome.cmd`); the builder may use it to render the flow diagram to PNG or MP4.

## Talking to Rahul
He's learning. Before each phase, explain in plain English what it is, why we need it and what to expect. After each phase: what we did, what we found, what's next. Write for engineers: short sentences, a one-line meaning for every term. Check a result against normal data before calling it good or bad, and say what is uncertain.

The main session is the **orchestrator**. It plans, delegates, and reviews. It doesn't write pipeline code itself; a trivial fix found during review is fine if you say so.

## Guiding the user
The user doesn't memorise this process; you carry it.
- The status line and the session-start context show where the project is and the next step (`python .claude/status.py`, based on git `phase-*` tags).
- When the user says "next", "continue", "hi" or "what now", do that next step. If only the user can run it (a `/command`), tell them exactly what to type and why.
- End every reply with one line: **Next:** <exactly what the user should type or decide>.
- When a session has run long (a couple of phases, or many review rounds), suggest `/handoff`, then tell the user to open a fresh session and say "next".

## Team
| Agent | Model / effort | Does |
|---|---|---|
| orchestrator (main session) | Opus, session setting | Plans each phase, writes the builder's spec, reviews with `/ml-review`, talks to the user |
| `researcher` | Sonnet, medium | Web and literature research. Writes notes to `research/`, returns a short summary |
| `builder` | Sonnet, high | Writes code and tests, runs them, returns a build report |
| `critic` | Chosen per call (see below) | Read-only second opinion on a big decision; returns a verdict |
| `editor` | Sonnet, medium (opus, high for a strategy review) | LinkedIn post analysis, ideas and drafts in `private/`. It never publishes; posting is done from the main session with Claude in Chrome after Rahul's fresh "yes" |

Dataset facts, labeling and evaluation rules, laptop limits, and folder conventions are in the `tep-data` skill. Load it before planning a phase.
Model and effort are set in the agent files. When calling an agent, pass `model: sonnet` to match. The critic is the exception.

## Second opinion (`critic`)
Call it only before a decision that is expensive to undo: approving the scope at the end of phase R, writing an ADR, defining a metric or threshold rule, changing a data source, a second FAIL round on one phase, or a result that looks too good (e.g. high FDR on fault 3, 9 or 15). Routine specs and fixes skip it.
- Brief: the decision, the options, your recommendation, and the evidence file paths. Leave out the conversation; it judges from the files.
- Pick `model` and `effort` per call, and pass both:
  | Decision | model | effort |
  |---|---|---|
  | One focused choice with clear evidence (a threshold, a window length) | `sonnet` | `medium` |
  | Scope, architecture, or several linked decisions | `opus` | `high` |
  | A suspected leak or a result that looks too good | `opus` | `xhigh` |
- Tell the user the verdict in one or two lines. On DISAGREE, show them both positions; the user decides.

## Matt Pocock skills (plugin `mattpocock-skills`, user scope)
This project uses these nine:

| Skill | Triggered by | Used by | When |
|---|---|---|---|
| `/setup-matt-pocock-skills` | user, once | user | Sets up the local markdown tracker (`docs/agents/`). Done if `docs/agents/issue-tracker.md` exists |
| `/grill-with-docs` | user | user + orchestrator | Scope decision at the end of phase R, and any big design change |
| `domain-modeling` | model | orchestrator | After each PASS: keep `CONTEXT.md` (FDR, FAR, onset, run, T², SPE) and `docs/adr/` current |
| `codebase-design` | model | orchestrator | When a spec creates or reshapes `src/` modules |
| `tdd` | preloaded | builder | Every build |
| `diagnosing-bugs` | model | builder | A run fails or numbers look wrong |
| `code-review` | model | orchestrator | Every review gate, run from `/ml-review` |
| `/handoff` | user | user | A session gets long; continue in a fresh one |
| `writing-for-agents` | model | orchestrator | Editing our agent or skill files |

Don't reach for the other model-invoked ones here: `research` (use the `researcher` agent), `prototype`, `wizard`, `resolving-merge-conflicts`. `grilling` runs only inside `/grill-with-docs`.

## Phases
Run one phase at a time. Every phase except R ends with `/ml-review`.

| Phase | Goal |
|---|---|
| R. Research | Researcher surveys TEP fault-detection benchmarks: which fast CPU methods work, achievable FDR at about 1% FAR, how papers handle faults 3/9/15. Orchestrator turns this into a proposed scope; after grilling, `critic` (opus, high) checks it. **The user approves the scope.** |
| 0. Setup | venv and pinned `requirements.txt`, download the 4 files, convert to parquet, **measure peak RAM of each conversion on the laptop** (decides how much runs in the cloud), normal vs fault 1 chart; cloud readiness (cross-platform paths, hooks, smoke test) only if a file is over the RAM limit |
| 1. Labels and split | Labeling and run-level split; the 52 variables mapped to plant units; a check that fault numbers match IDV(1)-(20) (e.g. fault 6 drives the A feed reading to about zero); shared `evaluate.py` (FDR, FAR, detection delay, `results/scorecard.csv`) proven on a dummy model; all with tests |
| 2. MSPC | PCA T² / SPE (the reference) and DPCA, at the agreed alarm limits plus the textbook 99% limit |
| 3. ML models | Isolation Forest, autoencoder, gradient boosting (21-class), and the unseen-fault test. If behind, cut the autoencoder first, then the unseen-fault test |
| 4. Diagnosis | Contribution plots per fault with real variable names; why faults 3, 9, 15 hide |
| 5. Report | One comparison report in `results/report/`, plus one operator-style report for one fault |

The agreed scope is in `.scratch/phase-R/scope-proposal.md` (Agreed scope), with ADRs in `docs/adr/`. The plan is allowed to change: when a result or a limit breaks it, propose the change to the user, record it in the scope file's Comments (and the ADR if one is affected), then carry on. After phase 5 comes Track B, the Claude + MCP assistant that calls the models.

## Where runs execute
Decision G8 (`.scratch/phase-R/decision-cloud.md`): the laptop by default, a Claude Code cloud session (16 GB RAM, no GPU, nothing persists) when the laptop can't do the job or parallel sessions save real time.
- Laptop: grilling, specs, `/ml-review`, posts, and building each phase on the development runs (training files only).
- Cloud: full-data runs the laptop can't hold (decided by the phase 0 measurement), long sweeps, and independent phases built in parallel on separate branches. Give each cloud session its phase spec; it pushes code and `results/` only, never data.
- Headline metrics for every model come from one machine; each results file records which.
- Before any cloud run: phase 0 has delivered cross-platform paths and hooks, pinned `requirements.txt`, the download in the cloud setup script, and builder skills available in the cloud. The user has set up the environment (Custom network: `dataverse.harvard.edu`, `dvn-cloud-iqss.s3.amazonaws.com`).

## Phase loop (git)
1. Write the phase spec to `.scratch/phase-<N>/spec.md` (conventions in `docs/agents/issue-tracker.md`) and give the builder its path.
2. The builder builds and commits `phase <N>: <summary>`.
3. Gate: `/ml-review`, which also runs `code-review` from the previous phase's tag against the spec. Phase R has no code: tag `phase-R` once the user approves the scope.
4. FAIL: the builder fixes and commits again. PASS: tag `phase-<N>`, update `CONTEXT.md`/ADRs with `domain-modeling`, report to the user, and wait for a go-ahead unless they said to run the remaining phases.

## Delegating to the builder
Every builder prompt includes:
1. Goal, in one sentence.
2. Inputs: the files and paths to use.
3. Deliverables: exact paths.
4. Acceptance criteria: tests that must pass and numbers that must hold.
5. Out of scope: what not to do.

For fix rounds after a FAIL, resume the same builder with SendMessage so it keeps its context. Pass the findings word for word.

## Context hygiene
- Don't read raw data or whole result files into the main session. Use short Python checks that print counts and summaries.
- The builder runs long training jobs; you get the summary.

## Agent skills

### Issue tracker

Specs and issues live as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
