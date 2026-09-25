# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`. Pipeline phases use `phase-<key>` as the slug (`phase-R`, `phase-0` ... `phase-5`).
- The spec is `.scratch/<feature-slug>/spec.md`.
- If a phase needs separate tickets, use one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
- Comments and conversation history go at the bottom of the file under a `## Comments` heading.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`, creating the directory if needed.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user or orchestrator will normally pass the path directly.
