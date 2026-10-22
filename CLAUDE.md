# TEP fault detection: orchestrator setup

## Who and why
Rahul builds industrial AI for process plants (metals and alumina) in his day job. This side project learns the main fault-detection models hands-on, on the public Tennessee Eastman data only, and feeds a LinkedIn post series.
- **Public data only.** Never use, ask for or guess employer or client data, company names, plant names, tag names or screenshots. This repo is public on GitHub.
- **Private material lives in `private/` (git-ignored):** `NOTES.md` (goals, tracks A-D, open questions), `PLAN.md` (model list, build steps, post series, backlog), `linkedin/` (post analysis, scrapers, raw feeds) and `posts/` (drafts). Read them when planning scope or drafting posts; quote nothing from them into tracked files.
- **Posts:** only what we build, how it works and the science behind it; no companies, vendors, industry news or opinion. Claude drafts; Rahul approves; publish only after a fresh "yes" for that post.

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

Dataset facts, labeling and evaluation rules, laptop limits, and folder conventions are in the `tep-data` skill. Load it before planning a phase.
Model and effort are set in the agent files. When calling an agent, pass `model: sonnet` to match.

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
| R. Research | Researcher surveys TEP fault-detection benchmarks: which fast CPU methods work, achievable FDR at about 1% FAR, how papers handle faults 3/9/15. Orchestrator turns this into a proposed scope. **The user approves the scope.** |
| 0. Setup | venv and packages, download the 4 files, convert to parquet, log peak RAM |
| 1. Labels and split | Labeling and run-level split, with tests |
| 2. ML models | The approved models, thresholds calibrated to the agreed FAR |
| 3. MSPC | PCA T² / SPE at 99% limits (plus any approved variant), same data and metrics |
| 4. Diagnosis | Contribution plots per fault with real variable names |
| 5. Report | One comparison report in `results/report/` |

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
