# Phase 0: setup and data

Goal: a reproducible Python environment and the four Rieth (2017) TEP files as float32 parquet, with each conversion's peak RAM measured on the laptop, plus one chart of a normal run against a fault 1 run (post 1's image).

Rules: the `tep-data` skill (dataset, conventions, laptop limits). It overrides this spec on any conflict; flag it if so.

## Inputs
- Dataverse API (in `tep-data`). File ids: 3031240 FaultFree_Testing (47 MB), 3031241 FaultFree_Training (25 MB), 3031242 Faulty_Training (494 MB), 3031243 Faulty_Testing (837 MB). Use the API's md5 checksums.
- Variable names and units: `research/tep-flowsheet.md` (Downs & Vogel names). Chart style: `tep-data` Project conventions.

## Deliverables
1. `.venv` (Python 3.11) and a pinned `requirements.txt` (exact versions) with what phases 0-3 need: numpy, pandas, pyarrow, pyreadr, scikit-learn, matplotlib, psutil, pytest.
2. `src/data.py`: `download(file, dest)` (skips a file already present with a matching md5) and `convert(rdata_path, parquet_path)` (float32 for the 52 variables, int16 for `faultNumber`, `simulationRun`, `sample`; frees memory after each file).
3. `scripts/prepare_data.py`: one command that downloads all 4 files to `data/raw/` and converts them to `data/processed/`, smallest first. It logs runtime and peak RAM (whole process tree, psutil) per file to `results/phase-0/conversion_log.csv`, with columns `file, rows, runtime_s, peak_ram_gb, machine, status`.
   - **Memory guard:** if peak RAM passes 6.0 GB during a conversion, stop that conversion cleanly, write `status=over_limit`, and carry on. Never freeze the laptop.
4. `scripts/plot_fault1.py` -> `results/phase-0/figures/normal-vs-fault1.png`. Use FaultFree_Training run 1 and Faulty_Training fault 1 run 1 (training files only). Pick the variable with the largest standardised mean shift after onset (sample > 20) versus that run's own normal stretch (samples 1-20), and print the top 3. Plot that variable over time (hours) for both runs, with the onset marked. The title states the finding in plain words, using the Downs & Vogel name and unit. Follow the chart style.
5. Tests (`tests/`, pytest): dtype conversion on a tiny synthetic frame, md5-skip logic, the memory-guard decision function, the shift ranking on a synthetic run. No network in the tests.

## Acceptance criteria
- `.venv/Scripts/python -m pytest -q` passes, including the existing `tests/test_status.py` and `tests/test_pfd.py`.
- `conversion_log.csv` has 4 rows. Row counts match `tep-data` (250,000 / 480,000 / 5,000,000 / 9,600,000) for every converted file. Over-limit files say so.
- Parquet variable columns are float32 and id columns are int16. Quick check: label means are not part of this phase.
- The chart exists, the top 3 shifted variables are printed in the build report, and the title is a finding.
- Nothing in `data/` is committed (it's git-ignored; check `git status`).

## Out of scope
Labels and the train/validation split (phase 1), any model, cloud setup (only if the RAM log shows a file over the limit; report it and stop), changes to hooks or `status.py`.

## Comments
