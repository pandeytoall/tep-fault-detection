"""Per-reading contributions to the PCA statistics (phase 2b), in the scaled space of the PCA model.

With P the retained loadings and Lambda their eigenvalues (the uncentred second-moment eigenpairs of src/mspc.py):
- SPE: c_j = e_j^2 with e = x - P P' x. They sum to SPE.
- T2 (complete decomposition, Alcala & Qin 2011): c_j = (xi_j' P Lambda^(-1/2) P' x)^2, xi_j the j-th unit vector.
  They sum to T2 = x' P Lambda^(-1) P' x.
Normal reference: mean c_j per variable over the `fit` normal rows, stored in pca.npz (ref_spe, ref_t2). The relative
contribution of a window is its mean c_j divided by max(reference_j, FLOOR_SHARE x mean of the reference).
"""
import numpy as np
import pandas as pd

from src.variables import VARIABLE_INFO, scoring_unit

FLOOR_SHARE = 0.01
STATISTICS = ("spe", "t2")
TOLERANCE_H = 1e-6


def scale(monitor, x):
    """Rows of raw readings (array or frame with the monitor's variables) in the monitor's scaled space."""
    if isinstance(x, pd.DataFrame):
        x = x[monitor.variables].to_numpy(dtype=np.float64)
    return (np.asarray(x, dtype=np.float64) - monitor.mean) / monitor.std


def spe_contributions(monitor, z):
    """(rows x variables) squared residuals e_j^2 of scaled rows `z`."""
    p = monitor.loadings[:, :monitor.n_pc]
    return (z - z @ p @ p.T) ** 2


def t2_contributions(monitor, z):
    """(rows x variables) complete-decomposition contributions to T2 of scaled rows `z`."""
    p = monitor.loadings[:, :monitor.n_pc]
    root = p @ np.diag(monitor.eigvals[:monitor.n_pc] ** -0.5) @ p.T  # symmetric
    return (z @ root) ** 2


_CONTRIBUTIONS = {"spe": spe_contributions, "t2": t2_contributions}


def _check(statistic):
    if statistic not in _CONTRIBUTIONS:
        raise ValueError(f"statistic must be one of {list(_CONTRIBUTIONS)}, got {statistic!r}")


def normal_reference(monitor, frame):
    """{'spe': array, 't2': array}: mean contribution per variable over the rows of a normal frame."""
    z = scale(monitor, frame)
    return {stat: fn(monitor, z).mean(axis=0) for stat, fn in _CONTRIBUTIONS.items()}


def relative(window_mean, reference):
    """Window-mean contribution over the normal reference, with the reference floored at FLOOR_SHARE of its mean."""
    reference = np.asarray(reference, dtype=float)
    return np.asarray(window_mean, dtype=float) / np.maximum(reference, FLOOR_SHARE * reference.mean())


def top_contributors(monitor, reference, window, statistic, n=5):
    """Top `n` readings by relative contribution over the rows of `window` (frame with the monitor's variables).

    Each entry: tag, D&V name, unit, scoring unit and ratio (rounded to 1 decimal), largest first.
    """
    _check(statistic)
    if len(window) == 0:
        raise ValueError("no samples in the requested window")
    mean = _CONTRIBUTIONS[statistic](monitor, scale(monitor, window)).mean(axis=0)
    ratio = relative(mean, reference[statistic])
    out = []
    for j in np.argsort(-ratio, kind="stable")[:int(n)]:
        tag = monitor.variables[j]
        info = VARIABLE_INFO[tag]
        out.append({"tag": tag, "name": info.name, "unit": info.unit, "scoring_unit": scoring_unit(tag),
                    "ratio": round(float(ratio[j]), 1)})
    return out


# --- windows ---------------------------------------------------------------------------------------
def window_start(alarms, statistic):
    """Hour of the first `raise` of that statistic in an alarm log, or None (then the window is the whole case)."""
    raised = alarms[(alarms["statistic"] == statistic) & (alarms["event"] == "raise")]
    return float(raised["t_hours"].min()) if len(raised) else None


def window_rows(matrix, start_h, end_h):
    """Rows of a case matrix (t_hours column) with start_h <= t <= end_h; None means unbounded."""
    t = matrix["t_hours"].to_numpy()
    keep = np.ones(len(t), dtype=bool)
    if start_h is not None:
        keep &= t >= start_h - TOLERANCE_H
    if end_h is not None:
        keep &= t <= end_h + TOLERANCE_H
    return matrix[keep]


def baseline_unit(monitor, reference, matrix, alarms):
    """No-LLM baseline: scoring unit of the top relative SPE contributor over the default window."""
    window = window_rows(matrix, window_start(alarms, "spe"), None)
    return top_contributors(monitor, reference, window, "spe", n=1)[0]["scoring_unit"]


# --- the reference lives in pca.npz ------------------------------------------------------------------
def add_reference(path, reference):
    """Add ref_spe and ref_t2 to an existing model .npz, keeping everything else (including the limits)."""
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["ref_spe"], arrays["ref_t2"] = np.asarray(reference["spe"]), np.asarray(reference["t2"])
    np.savez(path, **arrays)


def load_reference(path):
    """{'spe', 't2'} arrays stored in the model .npz, or None if it has no reference yet."""
    with np.load(path) as f:
        if "ref_spe" not in f.files:
            return None
        return {"spe": f["ref_spe"], "t2": f["ref_t2"]}
