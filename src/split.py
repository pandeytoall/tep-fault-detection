"""One seed-42 partition of run IDs 1-500 into jobs, shared by every training file and every fault.

Why one partition: run k of the fault-free file and of every fault share one noise stream within the
training family (.scratch/phase-1/run-pairing.md), so no run ID may serve two jobs.
Testing files are not split: all runs, scored once, in later phases.

Usage: .venv/Scripts/python -m src.split   (writes results/phase-1/run_split.json)
"""
import json
from pathlib import Path

import numpy as np

SEED = 42
N_RUNS = 500
SIZES = {"fit": 330, "limit": 100, "supervised": 50, "dev": 20}
N_LIMIT_TUNE = 50  # inside `limit`; also tunes the supervised model
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "results" / "phase-1" / "run_split.json"


def make_split():
    """{'seed', 'sizes', 'groups': {name: sorted run IDs}} from numpy default_rng(42)."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(np.arange(1, N_RUNS + 1))
    groups, start = {}, 0
    for name, size in SIZES.items():
        groups[name] = order[start:start + size]
        start += size
    groups["limit_tune"] = rng.choice(groups["limit"], size=N_LIMIT_TUNE, replace=False)
    return {
        "seed": SEED,
        "sizes": {name: len(v) for name, v in groups.items()},
        "groups": {name: sorted(int(i) for i in v) for name, v in groups.items()},
    }


def ids(group, split=None):
    """Sorted run IDs (Python ints) of one group: fit, limit, limit_tune, supervised or dev."""
    return (split or make_split())["groups"][group]


def save_split(path=DEFAULT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sp = make_split()
    groups = ",\n".join(f'  "{name}": {json.dumps(v)}' for name, v in sp["groups"].items())
    path.write_text(f'{{\n "seed": {sp["seed"]},\n "sizes": {json.dumps(sp["sizes"])},\n'
                    f' "groups": {{\n{groups}\n }}\n}}\n')


def load_split(path=DEFAULT_PATH):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    save_split()
    print(f"saved {DEFAULT_PATH.name}: {make_split()['sizes']}")
