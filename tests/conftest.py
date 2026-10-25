"""Shared test fixtures."""
import numpy as np
import pandas as pd
import pytest

from src import contributions, data, historian, mspc

T = 0.05


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """A tiny world: fitted PCA with limits and reference in an .npz, two cases in a historian, two notes."""
    tmp = tmp_path_factory.mktemp("world")
    rng = np.random.default_rng(3)
    scale = np.linspace(0.5, 2, 52)
    base = np.linspace(10, 20, 52)
    fit = pd.DataFrame(rng.normal(size=(3000, 52)) * scale + base, columns=data.VARIABLES)
    fit.insert(0, "sample", np.tile(np.arange(1, 501), 6))
    fit.insert(0, "simulationRun", np.repeat(np.arange(1, 7), 500))
    fit.insert(0, "faultNumber", 0)
    monitor = mspc.PCAMonitor(lags=0, cpv=0.9).fit(fit)
    scores = monitor.score(fit)
    limits = {s: {"1%": float(np.quantile(scores[s], 0.99)), "5%": 0.0, "99% textbook": 0.0} for s in mspc.STATS}
    monitor.limits = limits
    npz = tmp / "pca.npz"
    monitor.save(npz, limits=limits)
    contributions.add_reference(npz, contributions.normal_reference(monitor, fit))

    def case(seed, n=240, step_from=None, offset=0.0):
        r = np.random.default_rng(seed)
        x = r.normal(size=(n, 52)) * scale + base + offset
        if step_from is not None:
            x[step_from:, 0] += 40.0  # a big step on xmeas_1
        frame = pd.DataFrame(x.astype(np.float32), columns=data.VARIABLES)
        frame.insert(0, "t_hours", np.round(np.arange(n) * T, 2))
        return frame

    frames = {"dev_01": case(1, step_from=100), "dev_02": case(2, offset=50.0), "dev_03": case(4)}
    db = tmp / "historian_dev.sqlite"
    historian.write_historian(db, frames, monitor)
    notes = tmp / "plant-notes"
    notes.mkdir()
    (notes / "process-description.md").write_text("# Process description\nThe plant has five units.\n")
    (notes / "monitor-guide.md").write_text("# Monitor guide\nT2 and SPE.\n")
    (tmp / "secret.md").write_text("secret")
    return {"db": db, "npz": npz, "notes": notes, "frames": frames, "monitor": monitor, "tmp": tmp}
