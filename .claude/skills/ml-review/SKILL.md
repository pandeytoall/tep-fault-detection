---
name: ml-review
description: Review gate for a finished TEP pipeline phase. Run before approving any builder work or moving to the next phase. Produces a PASS/FAIL verdict with findings.
---

# ML review gate

You wrote the spec, so you share its blind spots. Review against this list and the `tep-data` rules, not against your memory of the spec. Prove things by running them; don't trust the build report.

## Reviewing without flooding context
1. Read the builder's build report first.
2. Use Grep to find the code behind each check (labels, split, thresholds, metrics). Read only those functions, not whole files.
3. Check claims with short `.venv/Scripts/python -c "..."` snippets that print counts and means, never whole dataframes.
4. Run the tests yourself: `.venv/Scripts/python -m pytest -q`.
5. Run the `mattpocock-skills:code-review` skill. Fixed point: the previous phase's tag. Spec: this phase's issue file. Coding standards: the rules in `.claude/agents/builder.md` and the `tep-data` conventions. Its blocker and major findings count toward your verdict.

## Checklist
Skip items that don't apply to the phase, and say which ones you skipped.

**Data and labels**
- [ ] Onset rule is `sample > 20` (training) and `sample > 160` (testing). Check: label mean for faulty training is about 0.960, faulty testing about 0.833, fault-free 0.
- [ ] Row counts match the `tep-data` table, or the subsample is stated and balanced across faults.
- [ ] Variables stored as float32. Peak RAM logged and under 5 GB.

**Leakage**
- [ ] Testing files never feed fitting, thresholds, early stopping, feature selection or "best model" choice. Grep for every place test data is loaded.
- [ ] Validation split is by `simulationRun`, with no run on both sides.
- [ ] Scalers/PCA fitted on training data only (MSPC: normal training data only).

**Metrics**
- [ ] Thresholds set on validation normal data for the agreed FAR target. If test FAR is far off the target (more than 2x or less than 0.5x), it needs an explanation.
- [ ] FDR per fault, plus overall mean with and without faults 3, 9, 15. FAR is shown next to every FDR.
- [ ] Numbers are plausible: easy faults (1, 2, 4, 6, 7) near 100% for decent methods, 3/9/15 near the FAR level. Much better on 3/9/15 means finding the leak.
- [ ] Train time measured the same way for every model.

**Code**
- [ ] Tests exist for the label rule, split and metric functions, and pass when you run them.
- [ ] One command reruns the phase. Seed fixed. Results overwrite `results/<phase>/`.
- [ ] No unrequested extras.

## Verdict format
```
VERDICT: PASS | FAIL
Checked: <items>    Skipped: <items + why>
Findings:
1. [blocker|major|minor] path:line: problem. Required fix.
```
- FAIL: send the findings to the builder word for word. After 2 FAIL rounds on the same phase, stop and ask the user.
- PASS: tell the user in 10 lines or fewer what was built, the key numbers, and anything surprising.
