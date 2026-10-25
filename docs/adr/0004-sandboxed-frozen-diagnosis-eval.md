# Assistant scored on a frozen, sandboxed case set by unit accuracy

The assistant (Claude with one MCP server) is scored on a frozen test set of 70 cases from the testing family: 3 runs per fault plus 10 normal, all with distinct run numbers, shuffled neutral IDs, and equal length with a random start crop, so neither the fault, the paired normal run nor the onset can be inferred. Each case runs in a fresh headless session whose only tools are the MCP server's; it has no file, shell or web access, because inside the repo it could simply read the answer key. The headline is unit accuracy (the right plant area), not the IDV number, because that is what an operator needs first and it doesn't punish a right area with a wrong IDV; IDV top-1/top-3 is the strict secondary score. Prompts are tuned only on a training-family dev set; every later version (new model tools, knowledge graph, RAG) reruns the same 70 cases so versions compare case by case.

**Considered options:**
- 25 cases × 1 run: about ±23 points, too wide to compare versions.
- 70 cases × 3 repeats: majority vote, but 2.3 times the cost for every later version. Instead 10 cases are repeated 3 times; if fewer than 8 of 10 agree, all cases move to 3 repeats.
- An LLM judge marking free-text answers: more flexible, less repeatable.

**Consequences:** the test set can't be rebuilt or edited once a session has run. Claude may remember TEP papers, which no setting controls; results and posts say so. Evidence: `.scratch/phase-2b/decisions.md`, `.scratch/phase-2b/spec.md`.
