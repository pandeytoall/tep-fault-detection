# Domain Docs

How the engineering skills should use this repo's domain documentation.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root: the project glossary.
- **`docs/adr/`**: decision records that touch the area you're about to work in.

If they don't exist yet, carry on without flagging it; don't create them upfront. `domain-modeling` (reached through `/grill-with-docs`) creates them when terms or decisions actually get settled.

## Layout (single-context)

```
/
├── CONTEXT.md
├── docs/adr/
└── src/
```

## Use the glossary's vocabulary

When you name a domain concept (spec title, hypothesis, test name), use the term exactly as `CONTEXT.md` defines it, not a synonym. If the term isn't there, either you're inventing language (reconsider) or there's a real gap (note it for `domain-modeling`).

## Flag ADR conflicts

If your output contradicts an ADR, say so explicitly instead of silently overriding it.
