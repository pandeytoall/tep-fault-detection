"""PCA and DPCA monitors (T2 and SPE alarm scores), the Ku et al. lag choice and the textbook limits.

Rules: .scratch/phase-2/spec.md. Frames hold faultNumber, simulationRun, sample plus the variable columns.
Lagged rows are built inside one run only; a run's first L samples have no full history and are dropped.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import evaluate
from src.data import ID_COLUMNS, VARIABLES


def n_components(eigvals, cpv=0.90):
    """Fewest leading PCs whose cumulative share of the total variance is at least `cpv`."""
    eigvals = np.asarray(eigvals, dtype=float)
    cum = np.cumsum(eigvals)
    return int(np.argmax(cum >= cpv * cum[-1] * (1 - 1e-12))) + 1


_RUN = ["faultNumber", "simulationRun"]


def lagged_rows(frame, variables, lags):
    """Lagged data matrix [x(k), x(k-1), ..., x(k-lags)] per row, built inside each run.

    Rows are ordered by run and sample. A run's first `lags` samples are dropped (no full history).
    Returns (id frame with faultNumber, simulationRun, sample; float64 array of len(variables) * (lags + 1) columns).
    """
    frame = frame.sort_values(ID_COLUMNS, kind="stable")
    ids = frame[ID_COLUMNS].reset_index(drop=True)
    values = frame[list(variables)].to_numpy(dtype=np.float64)
    position = ids.groupby(_RUN).cumcount().to_numpy()  # 0-based place of each row inside its run
    keep = position >= lags
    rows = np.flatnonzero(keep)
    x = np.concatenate([values[rows - lag] for lag in range(lags + 1)], axis=1)
    return ids[keep].reset_index(drop=True), x


CHUNK_ROWS = 100_000  # scoring works on whole runs, about this many rows at a time


def _run_chunks(frame):
    """Sorted frame cut at run boundaries into pieces of roughly CHUNK_ROWS rows."""
    frame = frame.sort_values(ID_COLUMNS, kind="stable")
    new_run = ~frame.duplicated(_RUN).to_numpy()
    starts = np.flatnonzero(new_run)
    bounds = [0]
    for s in starts[1:]:
        if s - bounds[-1] >= CHUNK_ROWS:
            bounds.append(int(s))
    bounds.append(len(frame))
    return [frame.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


class PCAMonitor:
    """PCA (lags=0) or dynamic PCA (lags=L) monitor with Hotelling T2 and SPE alarm scores.

    Scaler: mean and std (ddof=1) per variable of the normal fit runs. The PCs come from the second-moment
    matrix of the scaled, lagged fit matrix (columns [x(k) ... x(k-L)], rows inside one run only), so no
    run-crossing rows and no re-centring. The PC count is the fewest reaching `cpv` of the total variance.
    """

    def __init__(self, lags=0, cpv=0.90, variables=None):
        self.lags = int(lags)
        self.cpv = float(cpv)
        self.variables = list(VARIABLES if variables is None else variables)

    # --- fitting ---------------------------------------------------------------
    def _scaled(self, frame):
        scaled = frame[ID_COLUMNS].copy()
        z = (frame[self.variables].to_numpy(dtype=np.float64) - self.mean) / self.std
        return pd.concat([scaled, pd.DataFrame(z, columns=self.variables, index=scaled.index)], axis=1)

    def fit(self, frame):
        """Fit on fault-free runs (faultNumber 0 only). Returns self."""
        if (frame["faultNumber"] != 0).any():
            raise ValueError("fit needs normal runs only (faultNumber 0)")
        x = frame[self.variables].to_numpy(dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)
        if (self.std == 0).any():
            raise ValueError("zero-variance variables: " + ", ".join(np.array(self.variables)[self.std == 0]))
        _, z = lagged_rows(self._scaled(frame), self.variables, self.lags)
        self.n_rows = len(z)
        moment = z.T @ z / (self.n_rows - 1)
        values, vectors = np.linalg.eigh(moment)
        order = np.argsort(values)[::-1]
        self.eigvals = np.clip(values[order], 0.0, None)
        self.loadings = vectors[:, order]
        self.n_pc = n_components(self.eigvals, self.cpv)
        return self

    def with_n_pc(self, n_pc):
        """A copy of the fitted monitor that keeps `n_pc` PCs (the side sweep only; never used to pick)."""
        other = copy.copy(self)
        other.n_pc = int(n_pc)
        return other

    # --- scoring ---------------------------------------------------------------
    def score(self, frame):
        """DataFrame faultNumber, simulationRun, sample, t2, spe for the lagged rows of every run in `frame`."""
        p = self.loadings[:, :self.n_pc]
        inv_lambda = 1.0 / self.eigvals[:self.n_pc]
        parts = []
        for chunk in _run_chunks(frame):
            ids, z = lagged_rows(self._scaled(chunk), self.variables, self.lags)
            t = z @ p
            ids["t2"] = (t ** 2) @ inv_lambda
            ids["spe"] = np.clip((z ** 2).sum(axis=1) - (t ** 2).sum(axis=1), 0.0, None)
            parts.append(ids)
        return pd.concat(parts, ignore_index=True)

    # --- textbook limits ---------------------------------------------------------
    def textbook_limits(self, alpha=0.99):
        """{'t2', 'spe'} limits at confidence `alpha` from theory.

        T2: a (n^2 - 1) / (n (n - a)) * F(a, n - a) with a retained PCs and n rows of the lagged fit matrix.
        SPE: Jackson-Mudholkar from the discarded eigenvalues, theta_k = sum lambda^k (k = 1, 2, 3).
        """
        a, n = self.n_pc, self.n_rows
        t2 = a * (n ** 2 - 1) / (n * (n - a)) * stats.f.ppf(alpha, a, n - a)
        rest = self.eigvals[a:]
        th1, th2, th3 = ((rest ** k).sum() for k in (1, 2, 3))
        h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
        c = stats.norm.ppf(alpha)
        spe = th1 * (c * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)
        return {"t2": float(t2), "spe": float(spe)}

    # --- persistence -----------------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, loadings=self.loadings, eigvals=self.eigvals,
                 lags=self.lags, n_pc=self.n_pc, n_rows=self.n_rows, cpv=self.cpv,
                 variables=np.array(self.variables))

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            monitor = cls(lags=int(f["lags"]), cpv=float(f["cpv"]), variables=[str(v) for v in f["variables"]])
            monitor.mean, monitor.std = f["mean"], f["std"]
            monitor.loadings, monitor.eigvals = f["loadings"], f["eigvals"]
            monitor.n_pc, monitor.n_rows = int(f["n_pc"]), int(f["n_rows"])
        return monitor


# --- Ku et al. (1995) lag choice ----------------------------------------------------
def parallel_n_components(eigvals, n_rows, n_vars, reps=5, seed=42):
    """Horn's parallel analysis: leading eigenvalues of the correlation matrix that beat the mean eigenvalues
    of `reps` matrices of independent normal noise with the same shape. Counting stops at the first failure."""
    rng = np.random.default_rng(seed)
    noise = np.zeros(n_vars)
    for _ in range(reps):
        z = rng.standard_normal((n_rows, n_vars))
        z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)
        noise += np.sort(np.linalg.eigvalsh(z.T @ z / (n_rows - 1)))[::-1]
    beats = np.asarray(eigvals) > noise / reps
    return int(np.argmin(beats)) if not beats.all() else int(len(beats))


def ku_lag(frame, max_lag=5, rule="cpv", cpv=0.90, variables=None):
    """Ku, Storer & Georgakis (1995) lag window from fault-free runs.

    For l = 0, 1, ...: stack [x(k) ... x(k-l)] (rows inside one run), count the retained PCs a(l) with `rule`
    ('cpv': fewest PCs reaching `cpv` of the variance; 'parallel': parallel analysis), take r(l) = n_vars - a(l)
    and r_new(l) = r(l) - sum_{i<l} (l - i + 1) r_new(i). Stop at the first l with r_new(l) <= 0; the lag window
    is L = l - 1 (0 if l = 0). Returns (L, table); L is None when there is no stop by `max_lag`. The table has
    columns rule, lag, n_vars, a, r, r_new for every l tried.
    """
    rows, r_new = [], []
    lag = None
    for l in range(max_lag + 1):
        monitor = PCAMonitor(lags=l, cpv=cpv, variables=variables).fit(frame)
        n_vars = len(monitor.eigvals)
        if rule == "cpv":
            a = monitor.n_pc
        elif rule == "parallel":
            a = parallel_n_components(monitor.eigvals, monitor.n_rows, n_vars)
        else:
            raise ValueError(f"unknown rule {rule!r}")
        r = n_vars - a
        r_new.append(r - sum((l - i + 1) * r_new[i] for i in range(l)))
        rows.append({"rule": rule, "lag": l, "n_vars": n_vars, "a": a, "r": r, "r_new": r_new[-1]})
        if r_new[-1] <= 0:
            lag = max(l - 1, 0)
            break
    return lag, pd.DataFrame(rows)


# --- shared by the dev and final scripts ---------------------------------------------
STATS = ("t2", "spe")
FAR_LIMITS = {"1%": 0.01, "5%": 0.05}
TEXTBOOK = "99% textbook"


def lag_problem(cpv_lag, parallel_lag):
    """Why the run must stop after lag_choice.csv, or None if L can be used.

    L comes from the 90% rule. Stop if it gives L = 0, never stops by the maximum lag, or differs from
    the parallel-analysis L (also when that one never stops).
    """
    if cpv_lag is None:
        return "the 90% rule never stopped by the maximum lag"
    if cpv_lag == 0:
        return "the 90% rule gives L = 0"
    if cpv_lag != parallel_lag:
        return f"the rules disagree: 90% rule L = {cpv_lag}, parallel analysis L = {parallel_lag}"
    return None


def scorecard_rows(kind, faulty, normal, limits, onset, skip, after=None, **meta):
    """model_rows for `kind` ('pca' or 'dpca'): the two statistics as models `<kind>-t2` and `<kind>-spe`, each at
    every limit in limits[stat][limit_name]. `faulty` and `normal` are score frames (ID columns plus t2, spe).
    `meta` goes to evaluate.model_rows (machine, split_file, train_time_s, peak_ram_gb).
    """
    rows = []
    for stat in STATS:
        f = faulty[ID_COLUMNS].assign(score=faulty[stat].to_numpy())
        n = normal[ID_COLUMNS].assign(score=normal[stat].to_numpy())
        for limit_name, limit in limits[stat].items():
            rows += evaluate.model_rows(f"{kind}-{stat}", "unsupervised", limit_name, limit, f, n, onset, skip,
                                        after, **meta)
    return rows
