# Phase 2b: Assistant v1 (historian, alarm log, plant notes, MCP server)

Goal: Claude diagnoses hidden TEP fault cases. Its only tools are an MCP server over a replayed historian, a PCA alarm log, PCA scores, top contributing readings and plant notes. It is scored on diagnosis accuracy over a frozen test case set that later versions reuse.

Rules: the `tep-data` skill wins on any conflict; flag it if so. Design decisions, with the critic's changes and the user's choices: `.scratch/phase-2b/decisions.md`. Read it first. Terms: `CONTEXT.md`. ADRs: `docs/adr/0001`-`0003`.

## Inputs
- `data/processed/*.parquet` via `src/data.py` (`load_runs`, `ONSET`, `HOURS_PER_SAMPLE`, `VARIABLES`).
- `results/phase-1/run_split.json` via `src/split.py`.
- `results/phase-2/pca.npz` via `src/mspc.py` (`PCAMonitor.load`, `score`). PCA only (lags = 0), no DPCA.
- `src/evaluate.py` (`limit_for_far`, the 3-in-a-row rule), `src/variables.py` (`VARIABLE_INFO`: name, unit, plant unit), `src/prepare.py` (runtime, peak RAM, machine).
- Plant notes: written by the `researcher` into `plant-notes/` (4 files). They arrive while you build. Don't write them; for tests, use a temporary notes folder.

## Fixed choices (decided before any results; don't tune them)
- **Unit mapping (8 → 5)**, for scoring and in tool outputs: feed → `feeds`; reactor → `reactor`; condenser, separator → `condenser/separator`; compressor, purge → `compressor/recycle`; stripper, product → `stripper`.
- **Cases.** Every case uses a different run ID. A case is one run, with its first `c` samples cropped (seed 42). The historian clock starts at 0 h at the first kept sample, so the fault start is hidden.
  - **Test set** (testing family): 3 runs for each fault 1-20 plus 10 FaultFree_Testing runs = 70 cases. 70 distinct run IDs, drawn with seed 42 from 1-500. Crop `c` is uniform on 0-80. IDs `case_01`..`case_70`, numbered in a seed-42 shuffled order. **This set is frozen. Build it once and never rebuild or edit it.** v2 and v3 reuse it.
  - **Dev set** (training family, `dev` run IDs only): 16 faults (seed-42 choice from 1-20) plus 4 normal, which uses each of the 20 `dev` IDs once. Crop `c` is uniform on 0-10. IDs `dev_01`..`dev_20`. All prompt and tool iteration happens on this set.
  - **Answer key** per set: case ID, family, fault, run ID, crop, scoring unit, and the onset in historian hours. Saved to `results/phase-2b/answer_key_{test,dev}.csv`. Only the scorer reads it.
- **IDV → scoring unit** (answer key). From Downs & Vogel as listed in `research/tep-flowsheet.md`; check each row there, and report any row that doesn't match:
  - 1, 2, 3, 6, 7, 8, 9, 10: `feeds`
  - 4, 11, 13, 14: `reactor`
  - 5, 12, 15: `condenser/separator`
  - 16-20: `unknown`
  - normal: `none`
- **Historian:** one SQLite file per set, `data/processed/historian_{test,dev}.sqlite` (git-ignored). Table `readings(case_id, t_hours, tag, value)` with an index on (case_id, tag, t_hours). Tag = `xmeas_N` / `xmv_N`. No fault number, run ID, crop or label columns anywhere in the file.
- **Alarm log:** table `alarms(case_id, t_hours, statistic, event, value, limit)` in the same file. Limits: the PCA 1% limits for T² and SPE. Events:
  - `raise`: at the 3rd of 3 consecutive samples above the limit.
  - `clear`: at the 3rd of 3 consecutive samples below the limit, after a raise.
- **Limits stored with the model** (phase 2 carry-over). `mspc_dev.py` writes the 1%, 5% and textbook limits into `pca.npz` and `dpca.npz`. Rerun it. The stored limits must equal the values in the current `dev_scores.csv` (test it). Phase 2b reads limits only from the `.npz`.
- **Contributions** (scaled space, PCA model; P = retained loadings, Λ = their eigenvalues):
  - SPE: c_j = e_j², with e = x − P Pᵀ x. They sum to SPE.
  - T² (complete decomposition, Alcala & Qin 2011): c_j = (ξ_jᵀ P Λ^(-1/2) Pᵀ x)². They sum to T².
  - Normal reference: the mean c_j per variable over all `fit` normal rows. Store it in `pca.npz`. Floor = 1% of the mean over variables of that reference.
  - Relative contribution = (mean c_j over the window) / max(reference_j, floor).
  - Default window: from the first `raise` of that statistic to the end of the case. With no raise: the whole case.
  - Output: the top 5 per statistic, each with tag, D&V name, unit, scoring unit and the ratio (rounded to 1 decimal).
- **Assistant run:**
  - Headless `claude -p`, with cwd set to a fresh empty temp folder per session.
  - Model `claude-sonnet-5-5`; record the exact ID and any settings.
  - Only the `tep` MCP server is loaded. Only its tools are allowed. No built-in tools: no file, shell, web or agent tools.
  - No user or project CLAUDE.md, memory, hooks or plugins. Check `claude --help` for `--bare`, `--tools`, `--strict-mcp-config`, `--allowedTools` and `--setting-sources`, then choose the flags that achieve this. If a needed isolation can't be achieved, stop and report.
  - The MCP server is started per session with `--case <id> --set <test|dev>` and can only see that case.
  - Prompt: `src/assistant/prompt.md`. It gives the role, the tools, and the answer block format (`status`: fault / normal / can't tell; `unit`: one of the 5 or `unknown`; `idv`: up to 3 numbers, best first; `evidence`: a short text). It must not say how many cases exist or how many are normal or faulty. Freeze it after the dev run, and log its SHA-256 with every answer.

## Deliverables
1. `src/cases.py`: builds both case sets and answer keys from the rules above.
2. `src/historian.py`: writes the SQLite files (readings and alarms). Includes read helpers that the server uses.
3. `src/contributions.py`: SPE and T² contributions, the normal reference, and the relative top-N. The reference is added to `pca.npz`.
4. `src/assistant/server.py`: a stdio MCP server using the official `mcp` Python SDK (pin it in `requirements.txt`). Tools:
   - `query_historian(tags, start_h, end_h, step=1)`: returns values. Cap the output at 2,000 values and say so when truncated.
   - `get_alarms(start_h=None, end_h=None)`
   - `score_pca(start_h, end_h, step=1)`: T², SPE and both 1% limits per sample, with the same cap.
   - `top_contributors(statistic, start_h=None, end_h=None, n=5)`
   - `list_notes()` and `read_note(name)`, over `plant-notes/`. The docs folder needs a read tool, because the assistant has no file tools.
   - The server never opens an answer key or parquet file. It reads only its SQLite file, `pca.npz` and `plant-notes/`.
5. `scripts/assistant_build.py`: builds both sets once (refuses to overwrite an existing test set), with a runtime and peak RAM line.
6. `scripts/assistant_eval.py --set <dev|test> [--cases ...] [--repeat N]`: runs the sessions. Appends one JSON line per session to `results/phase-2b/answers_{set}.jsonl`, with:
   - the case and repeat
   - the raw final text and the parsed block
   - tool calls made, tokens in and out, cost, duration
   - the model ID, prompt hash and git commit
7. `src/assistant/score.py` plus `scripts/assistant_score.py`, which write `results/phase-2b/diagnosis_{set}.csv` and print a summary:
   - **Headline:** unit accuracy on fault cases excluding 3, 9, 15 and 16-20, as count / n with a Wilson 95% interval.
   - IDV top-1 and top-3 accuracy (all faults; 16-20 marked as background).
   - Normal cases: share answered `normal`.
   - Faults 3, 9, 15: share `can't tell` or `normal`, reported apart.
   - Faults 16-20: status accuracy only.
   - **Baseline (no LLM):** the scoring unit of the top relative SPE contributor over the default window. Chance level: the share of the most common unit among headline cases.
   - **Repeat agreement:** for each repeated case, whether all 3 runs agree on (status, unit). Print "k of 10 agree". **If k < 8, print the upgrade warning** (decision B: switch to 3 repeats on all cases).
   - Unparseable answers count as wrong and are listed.
8. Phase 2 carry-over cleanup, in its own commit `phase 2b: phase 2 review cleanup`:
   - a shared figure helper in `src/style.py`
   - the final-script plausibility, FAR and leak checks moved into a tested function
   - limits in the `.npz` (above)
   - a docstring note on the uncentred second moment
   - unused `parallel_n_components` params dropped
9. Tests, on synthetic data where possible:
   - Contributions sum to SPE and T² (hand-checked on a 3-variable example), and the floor works.
   - Alarm raise/clear on a hand-made series.
   - Case sets: distinct run IDs, crop ranges, shuffled IDs, the test set refusing to rebuild, and dev using only `dev` IDs.
   - The historian has no label, fault, run or crop columns (check the schema).
   - The server can see only its own case, and nothing in `src/assistant/` reads an answer key or parquet (grep test).
   - The output cap.
   - The scorer: on a hand-made answers file, every metric matches a hand count, including Wilson intervals and the k < 8 warning.
   - Answer-block parsing, including malformed blocks.
   - **Sandbox probe:** one dev session where the prompt asks the assistant to list files, run a shell command and fetch a web page. All three must be impossible. Save the transcript to `results/phase-2b/sandbox_probe.json`.

## Run order and stops
1. Build and test everything. Rerun `mspc_dev.py` for the stored limits. Build both sets.
2. Run the sandbox probe, then the **dev set, all 20 cases, one run each**. Iterate the prompt and tool output format only on dev.
3. **STOP and report:** dev scores, tokens and cost per session (mean and max), the sandbox probe result, and the frozen prompt hash. Don't touch the test set until the orchestrator says go.
4. On "go": the test set, 70 cases × 1 run. Then 10 cases (a seed-42 choice from the 70) for 2 more runs each, so they have 3 in total. Then score.

## Acceptance criteria
- `.venv/Scripts/python -m pytest -q` passes (133 existing tests plus the new ones).
- The sandbox probe shows no file, shell or web access.
- Grep: no `_Testing` read outside `mspc_final.py`, `cases.py` and the phase 1 pairing test. No answer key read outside `score.py`.
- Each case's fault onset is between 0 and 8 h on the historian clock (test set), or 0 and 1 h (dev set), as the key shows.
- Sanity on the test set:
  - Fault 6 cases: `top_contributors("spe")` includes an A-feed reading (xmeas_1 or xmv_3) in the top 5.
  - Normal cases: on average about 1% of samples are above the 1% limits (0.3-3%).
  - If any 3/9/15 case gets the IDV exactly right with `unit` matching, list it. That isn't a failure, but the orchestrator checks it for leakage.
- Peak RAM under 5 GB for every script. Machine recorded in every results file.

## Out of scope
- DPCA, Isolation Forest or any other model as a tool (later phases). RBC contributions (phase 4).
- Writing the plant notes (the researcher does it).
- Rebuilding or editing the test set after it's built.
- Any prompt change after the test run starts.
- Posts, pushing to GitHub.

## Comments
- 2026-10-25, user: end crop added so every case has the same length (test 880 samples, dev 490), and the test set was rebuilt before any test session. The test run went ahead without the step-3 stop (pre-approved).
- 2026-10-25, orchestrator: /ml-review **PASS**. Test set: unit accuracy 27/36 = 0.75 [0.59, 0.86] vs no-LLM baseline 15/36 and chance 18/36. IDV top-1 25/60, top-3 35/60. Normal 10/10 called normal; 3/9/15 9/9 "can't tell" or normal; repeat agreement 9/10 (no upgrade). 90 sessions, $4.19. Checked by me: 285 tests, headline recount 27/36, only `tep` tools in all 90 sessions, one model and one prompt hash, historian schema, 70 distinct runs, constant case length.
  - Trivial fix during review: removed the `--rebuild-unused-test` escape hatch and its two tests, now that the test set is in use (283 tests pass).
  - Deliberate exceptions: `answers_{set}.jsonl` is append-only (repeat top-ups), not overwritten on rerun. Headline and IDV metrics use repeat 1 of each case.
  - Noted for the results write-up: the frozen prompt mildly pushes toward finding a fault ("do not dismiss one because the alarms are quiet or late"). Faults 16-18 and 20 are flagged every time with a made-up unit; fault 19 never. The no-LLM baseline (0.42) is below chance (0.50).
  - Carried to phase 3 as cleanup (minor): one hour-window filter and tolerance constant shared by `server.py`, `contributions.py` and `historian.py`; reuse `evaluate.HARD_FAULTS` in `score.py`; one shared path for `pca.npz` and `results/phase-2b`; check the identical fault 6 top-5 lists against normal-case lists; optionally score faults 16-20 against their true status.
