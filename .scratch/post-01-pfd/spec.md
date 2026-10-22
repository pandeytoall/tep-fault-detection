# Post 1 visual: TEP process flow diagram (SVG + PNG)

Goal: one correct, clean, editable process flow diagram of the Tennessee Eastman Process, drawn by us, for LinkedIn post 1. It will be reused later for diagnosis (highlighting a faulty unit) and animation.

## Inputs
- Verified facts: `research/tep-flowsheet.md`, especially the section "Stream routing verified from the flowsheet figure". That stream table is the single source of truth for units and streams.
- Drawing rules: `tools/pfd-gem-prompt.md` (hard rules, SVG structure, layers, ids such as `unit-reactor` and `stream-7`, `data-from`/`data-to`, orthogonal routing, colours and dash patterns, fonts, legend, checklist). Follow it exactly, with these choices: equipment-level detail, wide 16:9 canvas, no control loops, utilities limited to the reactor and condenser cooling water and the stripper steam.

## Deliverables
- `docs/figures/tep-pfd.svg`: self-contained and valid, every unit and stream its own named element.
- `docs/figures/tep-pfd.png`: rendered from the SVG at 1600x900 or larger, e.g. headless Chrome: `"C:\Program Files\Google\Chrome\Application\chrome.exe" --headless --disable-gpu --screenshot=<out> --window-size=1600,900 <file-url-of-svg>`. Don't use the user's Chrome profile.
- `scripts/check_pfd.py` + `tests/test_pfd.py`: parse the SVG and assert every stream id in the stream table exists exactly once, every `data-from`/`data-to` names an existing unit or a battery-limit inlet/outlet, and there are no duplicate unit labels.

## Acceptance criteria
- The tests pass. Streams 1-13 and steam/condensate match the verified table, including: stream 4 goes to the stripper, stream 5 from the stripper top joins the recycle to the reactor feed, purge 9 leaves from the separator vapor line, product 11 leaves from the stripper bottom.
- Title: "Tennessee Eastman Process: 5 units, 52 readings, 20 faults" (the title states what the reader learns).
- Readable at phone size: unit labels at 18 px or larger at 1600 px width; colour-blind-safe, with dash patterns per stream type.
- No instrument tags, valve numbers or equipment IDs (not even the paper's), and no company names. Unit names are generic (Reactor, Condenser, Vapor-liquid separator, Recycle compressor, Stripper); feeds are labelled A, D, E, C.

## Out of scope
Animation, fault highlighting, variable labels on the diagram, any change to the pipeline phases.

## Comments
