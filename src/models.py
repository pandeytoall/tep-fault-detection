"""Phase 3 monitors: Isolation Forest, a scikit-learn MLP autoencoder and 21-class gradient boosting.

Same small interface as mspc.PCAMonitor: `fit`, `score` (ID columns plus an `alarm` column, higher = less normal),
`calibrate` (the 1% and 5% alarm limits, stored with the model), `save` / `load`. A monitor is also a callable
(`monitor(frame)` scores a frame). Rules: .scratch/phase-3/spec.md.
- Scaling: the PCA scaler (mean and std of the `fit` normal runs), passed in as `mean` and `std`.
- Lags: mspc.lagged_rows with `lags` = 0 or 2, built inside each run; a run's first `lags` samples are dropped.
- Frames hold faultNumber, simulationRun, sample plus the variable columns, as everywhere else.
"""
import hashlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPRegressor

from src import evaluate, historian, mspc
from src.data import ID_COLUMNS, ONSET, VARIABLES

SEED = 42
IF_ROWS = 100_000          # rows the Isolation Forest is fitted on (drawn from the `fit` normal runs)
IF_TREES, IF_SAMPLES = 200, 256
VAL_RUNS = 30              # autoencoder validation runs, out of the `fit` runs
PATIENCE, MAX_EPOCHS = 5, 100
BATCH_SIZE = 256
TRAINING_ONSET = ONSET["Training"]
UNSEEN_FAULTS = (1, 8, 13, 14, 16)
ASSISTANT_MODELS = ("iforest-l0", "iforest-l2", "ae-l0", "ae-l2", "hgb-l2")
N_JOBS = 6


def sha256_file(path, chunk=1 << 20):
    """SHA-256 hex digest of a file, so a later phase can tell which model file was used."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --- helpers shared by the fits ----------------------------------------------------------------------------
def split_runs(run_ids, n_val=VAL_RUNS, seed=SEED):
    """(train run IDs, validation run IDs), both sorted: `n_val` runs drawn with the seed, the rest for training."""
    ids = sorted(int(r) for r in run_ids)
    if not 0 < n_val < len(ids):
        raise ValueError(f"need 0 < n_val < {len(ids)} runs, got {n_val}")
    val = sorted(int(r) for r in np.random.default_rng(seed).choice(ids, size=n_val, replace=False))
    return [r for r in ids if r not in set(val)], val


class EarlyStopper:
    """Stop when the validation value has not improved (got strictly lower) for `patience` epochs in a row."""

    def __init__(self, patience=PATIENCE):
        self.patience = int(patience)
        self.best, self.best_epoch, self.epochs = np.inf, -1, 0

    def update(self, value):
        """Record one epoch's validation value; True means stop now."""
        if value < self.best:
            self.best, self.best_epoch = float(value), self.epochs
        self.epochs += 1
        return self.epochs - 1 - self.best_epoch >= self.patience


def hgb_classes(ids, onset=TRAINING_ONSET):
    """Class per row: the fault number once sample > onset, else 0 (pre-onset samples of faulty runs are normal)."""
    return (ids["faultNumber"].to_numpy() * (ids["sample"].to_numpy() > onset)).astype(int)


def supervised_frame(faulty, normal, exclude=()):
    """Training frame for the supervised model: faulty runs (minus the faults in `exclude`) plus the normal runs."""
    if (faulty["faultNumber"] <= 0).any() or (normal["faultNumber"] != 0).any():
        raise ValueError("supervised_frame wants faulty runs (faultNumber 1-20) first and fault-free runs second")
    return pd.concat([faulty[~faulty["faultNumber"].isin([int(f) for f in exclude])], normal], ignore_index=True)


def pick_hgb(grid, tol=0.005):
    """Chosen grid row as {learning_rate, max_leaf_nodes, max_iter}: highest dev mean FDR (excluding 3/9/15), then among
    fits within `tol` of the best the fewest trees (trees = max_iter x classes, the same classes for every fit); ties go
    to the higher FDR, then fewer leaves, then the lower learning rate."""
    score = grid["dev_mean_fdr_excl_3_9_15"]
    close = grid[score >= score.max() - tol - 1e-12]
    pick = close.sort_values(["max_iter", "dev_mean_fdr_excl_3_9_15", "max_leaf_nodes", "learning_rate"],
                             ascending=[True, False, True, True], kind="stable").iloc[0]
    return {"learning_rate": float(pick["learning_rate"]), "max_leaf_nodes": int(pick["max_leaf_nodes"]),
            "max_iter": int(pick["max_iter"])}


def _require_normal(frame, what):
    if (frame["faultNumber"] != 0).any():
        raise ValueError(f"{what} needs normal runs only (faultNumber 0)")


def _runs(ids):
    return sorted(int(r) for r in ids["simulationRun"].unique())


# --- monitors --------------------------------------------------------------------------------------------------
class Monitor:
    """Base: scaling, lags, chunked scoring, limits and persistence. Subclasses set kind, track, prefix and fit/_alarm."""

    kind = track = prefix = None

    def __init__(self, mean, std, lags=0, variables=None, name=None):
        self.mean, self.std = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
        self.lags = int(lags)
        self.variables = list(VARIABLES if variables is None else variables)
        self.name = name or f"{self.prefix}-l{self.lags}"
        self.estimator = None
        self.limits = None   # {"1%": value, "5%": value}, set by calibrate()
        self.meta = {"name": self.name, "kind": self.kind, "track": self.track, "lags": self.lags}

    # --- rows ---------------------------------------------------------------------------------------------
    def _rows(self, frame):
        """(ID frame, float32 matrix) of the scaled, lagged rows of `frame`."""
        ids, x = mspc.lagged_rows(mspc.scaled_frame(frame, self.variables, self.mean, self.std), self.variables,
                                  self.lags)
        return ids, x.astype(np.float32)

    def _chunks(self, frame):
        for chunk in mspc.run_chunks(frame):
            yield self._rows(chunk)

    # --- scoring -----------------------------------------------------------------------------------------
    def score(self, frame):
        """DataFrame faultNumber, simulationRun, sample, alarm (float64) for the lagged rows of every run in `frame`."""
        parts = []
        for ids, x in self._chunks(frame):
            ids["alarm"] = self._alarm(x)
            parts.append(ids)
        return pd.concat(parts, ignore_index=True)

    __call__ = score

    def _alarm(self, x):
        raise NotImplementedError

    def calibrate(self, normal_frame):
        """Set the 1% and 5% alarm limits from normal runs (the `limit` group): score quantiles with each run's first
        `lags` samples left out (they are never scored). Returns the limits."""
        _require_normal(normal_frame, "calibrate")
        scores = self.score(normal_frame)
        self.limits = {name: evaluate.limit_for_far(scores["alarm"], far) for name, far in mspc.FAR_LIMITS.items()}
        self.meta["limit_runs"] = _runs(scores)
        return self.limits

    # --- persistence -------------------------------------------------------------------------------------
    def save(self, path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": self.kind, "name": self.name, "mean": self.mean, "std": self.std,
                     "variables": self.variables, "lags": self.lags, "estimator": self.estimator,
                     "limits": self.limits, "meta": self.meta}, path)

    @classmethod
    def load(cls, path):
        return load(path)


class IFMonitor(Monitor):
    """Isolation Forest (200 trees, 256 samples each) on the lagged scaled rows; alarm = -score_samples."""

    kind, track, prefix = "iforest", "unsupervised", "iforest"

    def fit(self, frame, max_rows=IF_ROWS):
        """Fit on `max_rows` rows drawn with the seed from the fault-free `fit` runs. Returns self."""
        _require_normal(frame, "fit")
        t0 = time.perf_counter()
        ids, x = self._rows(frame)
        n = min(int(max_rows), len(x))
        pick = np.sort(np.random.default_rng(SEED).choice(len(x), size=n, replace=False))
        self.estimator = IsolationForest(n_estimators=IF_TREES, max_samples=IF_SAMPLES, random_state=SEED,
                                         n_jobs=N_JOBS).fit(x[pick])
        self.meta.update(hyperparameters={"n_estimators": IF_TREES, "max_samples": IF_SAMPLES, "random_state": SEED},
                         train_runs=_runs(ids.iloc[pick]), n_train_rows=n, train_time_s=time.perf_counter() - t0)
        return self

    def _alarm(self, x):
        return -self.estimator.score_samples(x)


def reconstruction_mse(estimator, x):
    """Mean squared reconstruction error per row (float64)."""
    return np.mean((estimator.predict(x) - x) ** 2, axis=1, dtype=np.float64)


class AEMonitor(Monitor):
    """MLPRegressor autoencoder X -> X with hidden layers (2b, b, 2b); alarm = reconstruction MSE per row."""

    kind, track, prefix = "ae", "unsupervised", "ae"

    def fit(self, frame, bottleneck, max_epochs=MAX_EPOCHS, patience=PATIENCE, n_val_runs=VAL_RUNS):
        """Train on the fit runs minus `n_val_runs` validation runs (split by run), one `partial_fit` epoch at a time.

        Stops when the validation MSE has not improved for `patience` epochs (at most `max_epochs`) and keeps the best
        epoch's weights. sklearn's own early stopping splits by rows, so it is not used. Returns self.
        """
        _require_normal(frame, "fit")
        t0 = time.perf_counter()
        ids, x = self._rows(frame)
        train_runs, val_runs = split_runs(_runs(ids), n_val_runs)
        in_val = ids["simulationRun"].isin(val_runs).to_numpy()
        x_train, x_val = x[~in_val], x[in_val]
        b = int(bottleneck)
        net = MLPRegressor(hidden_layer_sizes=(2 * b, b, 2 * b), activation="relu", solver="adam",
                           batch_size=BATCH_SIZE, random_state=SEED)
        stopper, history, best_weights = EarlyStopper(patience), [], None
        for epoch in range(int(max_epochs)):
            net.partial_fit(x_train, x_train)
            history.append(float(reconstruction_mse(net, x_val).mean()))
            stop = stopper.update(history[-1])
            if stopper.best_epoch == epoch:
                best_weights = ([w.copy() for w in net.coefs_], [w.copy() for w in net.intercepts_])
            if stop:
                break
        net.coefs_, net.intercepts_ = best_weights
        self.estimator = net
        self.meta.update(
            hyperparameters={"bottleneck": b, "hidden_layer_sizes": [2 * b, b, 2 * b], "activation": "relu",
                             "solver": "adam", "batch_size": BATCH_SIZE, "random_state": SEED,
                             "max_epochs": int(max_epochs), "patience": int(patience)},
            train_runs=train_runs, val_runs=val_runs, n_train_rows=int(len(x_train)), n_val_rows=int(len(x_val)),
            val_mse_history=history, best_epoch=stopper.best_epoch,
            final_train_mse=float(reconstruction_mse(net, x_train).mean()),
            final_val_mse=float(reconstruction_mse(net, x_val).mean()), train_time_s=time.perf_counter() - t0)
        return self

    def _alarm(self, x):
        return reconstruction_mse(self.estimator, x)


class HGBMonitor(Monitor):
    """HistGradientBoostingClassifier on the lagged scaled rows, one class per fault plus normal (class 0).

    Alarm = 1 - P(normal), computed as the summed probability of all other classes (the same number, without the
    rounding loss of 1 - P(normal) when P(normal) is very close to 1).
    """

    kind, track, prefix = "hgb", "supervised", "hgb"

    def fit(self, frame, onset=TRAINING_ONSET, learning_rate=0.1, max_leaf_nodes=31, max_iter=200, stride=1):
        """Fit on a supervised training frame (see supervised_frame); labels follow the onset rule. Returns self.

        `stride` > 1 keeps only the rows whose current sample number is a multiple of it (every run stays in). The lags are
        built from the full-rate data first, so the features do not change; neighbouring 3-minute samples are highly
        correlated, so this is the compute-budget lever (the fit time is proportional to the rows)."""
        if not ((frame["faultNumber"] == 0).any() and (frame["faultNumber"] > 0).any()):
            raise ValueError("fit needs both normal and faulty runs")
        t0 = time.perf_counter()
        ids, x = self._rows(frame)
        if stride > 1:
            keep = ids["sample"].to_numpy() % stride == 0
            ids, x = ids[keep].reset_index(drop=True), x[keep]
        y = hgb_classes(ids, onset)
        self.estimator = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, max_iter=max_iter, early_stopping=False,
            random_state=SEED).fit(x, y)
        values, counts = np.unique(y, return_counts=True)
        self.meta.update(
            hyperparameters={"learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes, "max_iter": max_iter},
            train_runs=_runs(ids), faults_trained=sorted(int(f) for f in ids["faultNumber"].unique() if f > 0),
            class_counts={str(int(v)): int(c) for v, c in zip(values, counts)}, n_train_rows=int(len(y)),
            onset=int(onset), row_stride=int(stride), train_time_s=time.perf_counter() - t0)
        return self

    def _normal_column(self):
        return int(list(self.estimator.classes_).index(0))

    def _alarm(self, x):
        proba = self.estimator.predict_proba(x)
        return np.delete(proba, self._normal_column(), axis=1).sum(axis=1)

    def proba(self, frame):
        """(ID frame, probabilities (rows x classes_)) for the lagged rows of every run in `frame`."""
        parts = [(ids, self.estimator.predict_proba(x)) for ids, x in self._chunks(frame)]
        return pd.concat([p[0] for p in parts], ignore_index=True), np.concatenate([p[1] for p in parts])


KINDS = {cls.kind: cls for cls in (IFMonitor, AEMonitor, HGBMonitor)}


def load(path):
    """A monitor of the right kind from a file written by Monitor.save."""
    payload = joblib.load(path)
    monitor = KINDS[payload["kind"]](payload["mean"], payload["std"], payload["lags"], payload["variables"],
                                     payload["name"])
    monitor.estimator, monitor.limits, monitor.meta = payload["estimator"], payload["limits"], payload["meta"]
    return monitor


# --- checks and the scorecard --------------------------------------------------------------------------------
_RUN_KEYS = {"iforest": (("train_runs", "fit"),), "ae": (("train_runs", "fit"), ("val_runs", "fit")),
             "hgb": (("train_runs", "supervised"),)}
_LIMIT_KEYS = (("limit_runs", "limit"), ("tuning_limit_runs", "limit_tune"))


def run_group_problems(meta, groups):
    """Messages for every run ID a model used outside the group of its job (empty list = all fine).

    IF and AE (train and validation runs) belong to `fit`, HGB to `supervised`; limits to `limit`; the HGB tuning limits
    to `limit_tune`. `groups` is {group name: run IDs} from the saved split.
    """
    problems = []
    for key, group in _RUN_KEYS[meta["kind"]] + _LIMIT_KEYS:
        used = meta.get(key)
        outside = sorted(set(used) - set(groups[group])) if used is not None else []
        if outside:
            problems.append(f"{meta['name']}: {key} not all in {group}: {outside[:5]}")
    if meta.get("val_runs") and set(meta["val_runs"]) & set(meta["train_runs"]):
        problems.append(f"{meta['name']}: runs in both train_runs and val_runs")
    return problems


def scorecard_rows(model, faulty, normal, onset, after=None, **meta):
    """evaluate.model_rows for a monitor at its 1% and 5% limits. `faulty` and `normal` are score frames (ID columns
    plus alarm); `meta` goes to evaluate.model_rows (machine, split_file, train_time_s, peak_ram_gb)."""
    f = faulty[ID_COLUMNS].assign(score=faulty["alarm"].to_numpy())
    n = normal[ID_COLUMNS].assign(score=normal["alarm"].to_numpy())
    rows = []
    for limit_name, limit in model.limits.items():
        rows += evaluate.model_rows(model.name, model.track, limit_name, limit, f, n, onset, model.lags, after, **meta)
    return rows


STRONG_FAULTS = (1, 2, 4, 6, 7)     # hgb must catch these at least 0.9 at the 1% limit
STRONG_MIN = 0.9
HARD_NOTE, HARD_STOP = 0.10, 0.5    # supervised FDR on 3/9/15: above the first is listed, above the second stops the run


def final_check_failures(df, unsupervised, supervised, unseen=None):
    """(stops, notes) for the final scorecard `df` (long format: model, limit_name, metric, fault, value).

    Stops (the run must be looked at before anything is reported):
    - an unsupervised model flags faults 3/9/15 above 0.10 at the 1% limit (FAR + 0.05 at the others);
    - a supervised model (the unseen one too) above 0.5 on faults 3/9/15 at the 1% limit;
    - a supervised model under 0.9 on faults 1, 2, 4, 6 or 7 at the 1% limit (not the unseen model).
    Notes (to explain in the report): FAR outside mspc.FAR_WINDOWS; supervised FDR above 0.10 on 3/9/15; the unseen model
    not worse than the full model on at least 3 of the 5 unseen faults.
    """
    stops, notes = [], []
    every = list(unsupervised) + list(supervised) + ([unseen] if unseen else [])
    for limit_name in df["limit_name"].unique():
        sel = df[df["limit_name"] == limit_name]
        far = sel[sel["metric"] == "far"].set_index("model")["value"]
        fdr = sel[sel["metric"] == "fdr"].pivot(index="fault", columns="model", values="value")
        window = mspc.FAR_WINDOWS.get(limit_name)
        for m in every:
            if window and not window[0] <= far[m] <= window[1]:
                notes.append(f"FAR out of range: {m} at {limit_name} = {far[m]:.4f}, expected {window}")
        for m in unsupervised:
            worst = fdr.loc[list(evaluate.HARD_FAULTS), m].max()
            bound = mspc.LEAK_BOUND_1PCT if limit_name == "1%" else far[m] + 0.05
            if worst > bound:
                stops.append(f"LEAK? {m} at {limit_name}: max FDR on 3/9/15 = {worst:.3f} > {bound:.3f}")
        if limit_name != "1%":
            continue
        for m in list(supervised) + ([unseen] if unseen else []):
            for f in evaluate.HARD_FAULTS:
                v = fdr.loc[f, m]
                if v > HARD_STOP:
                    stops.append(f"LEAK? {m}: fault {f} FDR {v:.2f} is above {HARD_STOP} on 3/9/15 at 1%")
                elif v > HARD_NOTE:
                    notes.append(f"{m}: fault {f} FDR {v:.2f} on 3/9/15 at 1% is above {HARD_NOTE:.2f}; explain it")
        for m in supervised:
            for f in STRONG_FAULTS:
                if fdr.loc[f, m] < STRONG_MIN:
                    stops.append(f"implausible: {m} fault {f} FDR = {fdr.loc[f, m]:.3f}, expected at least {STRONG_MIN}")
        if unseen and supervised:
            full = supervised[0]
            worse = [f for f in UNSEEN_FAULTS if fdr.loc[f, unseen] < fdr.loc[f, full]]
            if len(worse) < 3:
                notes.append(f"unseen: {unseen} is worse than {full} on only {len(worse)} of the 5 unseen faults {worse}")
    return stops, notes


# --- one case (the assistant tools) ------------------------------------------------------------------------------
def score_case(monitor, matrix):
    """DataFrame t_hours, alarm for a case matrix (t_hours + the 52 variables). Lagged monitors have no score for the
    first `lags` samples."""
    scores = monitor.score(historian.case_frame(matrix))
    t = matrix["t_hours"].to_numpy()[scores["sample"].to_numpy() - 1]
    return pd.DataFrame({"t_hours": t, "alarm": scores["alarm"].to_numpy()})


def class_probabilities_case(hgb, matrix):
    """DataFrame t_hours plus one probability column per class (column name = the class number) for a case matrix."""
    ids, proba = hgb.proba(historian.case_frame(matrix))
    frame = pd.DataFrame(proba, columns=[int(c) for c in hgb.estimator.classes_])
    frame.insert(0, "t_hours", matrix["t_hours"].to_numpy()[ids["sample"].to_numpy() - 1])
    return frame


def class_label(k):
    """'normal' for class 0, else 'IDV k'."""
    return "normal" if int(k) == 0 else f"IDV {int(k)}"
