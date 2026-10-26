"""Phase 3 development: fit Isolation Forest, autoencoder and gradient boosting on the training family only.

No testing file is read. Steps: (1) the PCA scaler and PC counts come from results/phase-2/{pca,dpca}.npz; (2) Isolation
Forest and autoencoder at L = 0 and L = 2 are fitted on the `fit` normal runs; (3) gradient boosting (L = 2) is tuned over
8 fits on the `supervised` runs, each scored on the `dev` runs with a 1% limit from the `limit_tune` runs, and the choice
rule in models.pick_hgb picks one; (4) the unseen-fault model is fitted with the chosen settings; (5) every model gets its
1% and 5% limits from the 100 `limit` normal runs; (6) the `dev` runs are scored. Models go to data/models/ with a
SHA-256 list in results/phase-3/models.json. Run scripts/ml_final.py afterwards.
Writes results/phase-3/{dev_scores.csv, hgb_tuning.csv, models.json}.
Usage: .venv/Scripts/python scripts/ml_dev.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")  # HistGradientBoosting ignores n_jobs; cap its threads

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import data, evaluate, models, mspc, prepare, split  # noqa: E402

NORMAL, FAULTY = "TEP_FaultFree_Training", "TEP_Faulty_Training"
OUT = data.RESULTS_3
SPLIT_FILE = "results/phase-1/run_split.json"
DPCA_MODEL = data.ROOT / "results" / "phase-2" / "dpca.npz"
GRID = {"learning_rate": (0.05, 0.1), "max_leaf_nodes": (31, 63), "max_iter": (200, 400)}
SHOWN_FAULTS = (1, 2, 4, 6, 7, 3, 9, 15)
N_CLASSES = 21
# Compute budget: a full-row HGB fit takes about 6 s per boosting iteration (21 classes, 156 features, 525k rows), which makes the
# 8-fit grid about 4.5 h. Every supervised run stays in; only every HGB_STRIDE-th sample of each run is used for fitting
# (the lag features are built before thinning). Scoring, limits and dev tuning scores use all rows.
HGB_STRIDE = 3


def dev_fdr_excl_hard(model, limit, dev_faulty, onset):
    """(mean FDR excluding 3/9/15, mean FDR over all 20) of a scored model at one limit, on the dev faulty runs."""
    scores = model.score(dev_faulty)
    f = scores[data.ID_COLUMNS].assign(score=scores["alarm"].to_numpy())
    fdr = evaluate.fdr(f, limit, onset, model.lags)
    return float(fdr.drop(list(evaluate.HARD_FAULTS)).mean()), float(fdr.mean())


def tune_hgb(train, mean, std, lags, tune_normal, dev_faulty, dev_normal, onset, machine):
    """Fit the 8 grid points. Returns (grid DataFrame, {row index: fitted model}). Each fit: 1% limit from `limit_tune`."""
    rows, fits = [], {}
    for lr, leaves, n_iter in itertools.product(*GRID.values()):
        model = models.HGBMonitor(mean, std, lags=lags)
        model.fit(train, onset=onset, learning_rate=lr, max_leaf_nodes=leaves, max_iter=n_iter, stride=HGB_STRIDE)
        limit = evaluate.limit_for_far(model.score(tune_normal)["alarm"], 0.01)
        excl, full = dev_fdr_excl_hard(model, limit, dev_faulty, onset)
        dev_far = float((model.score(dev_normal)["alarm"] > limit).mean())
        rows.append({"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": n_iter, "total_trees": n_iter * N_CLASSES,
                     "row_stride": HGB_STRIDE, "n_train_rows": model.meta["n_train_rows"],
                     "fit_time_s": model.meta["train_time_s"], "limit_1pct_limit_tune": limit,
                     "dev_mean_fdr_excl_3_9_15": excl, "dev_mean_fdr": full, "dev_far": dev_far, "machine": machine})
        fits[len(rows) - 1] = model
        print(f"  grid lr {lr} leaves {leaves} iter {n_iter}: dev mean FDR excl 3/9/15 {excl:.4f}, all {full:.4f}, "
              f"dev FAR {dev_far:.4f}, fit {model.meta['train_time_s']:.0f} s", flush=True)
    return pd.DataFrame(rows), fits


def main():
    t0 = time.perf_counter()
    machine = prepare.machine_label(prepare.cpu_name(), psutil.virtual_memory().total)
    groups = split.load_split()
    ids = lambda g: split.ids(g, groups)  # noqa: E731
    onset = data.onset_for(FAULTY)
    OUT.mkdir(parents=True, exist_ok=True)
    data.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- the scaler and the PC counts (bottleneck sizes) come from the phase 2 models -----------------------------
    pca, dpca = mspc.PCAMonitor.load(data.PCA_MODEL), mspc.PCAMonitor.load(DPCA_MODEL)
    mean, std, lag = pca.mean, pca.std, dpca.lags
    assert lag == 2 and (pca.n_pc, dpca.n_pc) == (31, 68), (lag, pca.n_pc, dpca.n_pc)
    bottleneck = {0: pca.n_pc, lag: dpca.n_pc}
    print(f"scaler from pca.npz; L = {lag}; bottleneck b = {bottleneck}")

    # --- unsupervised models on the `fit` normal runs ---------------------------------------------------------------
    fit_runs = data.load_runs(NORMAL, ids("fit"))
    assert (fit_runs["faultNumber"] == 0).all() and fit_runs["simulationRun"].nunique() == 330
    fitted = {}
    for lags in (0, lag):
        iforest = models.IFMonitor(mean, std, lags=lags).fit(fit_runs)
        ae = models.AEMonitor(mean, std, lags=lags).fit(fit_runs, bottleneck=bottleneck[lags])
        for m in (iforest, ae):
            fitted[m.name] = m
            print(f"{m.name}: fit {m.meta['train_time_s']:.0f} s", end="")
            if m.kind == "ae":
                print(f", {len(m.meta['val_mse_history'])} epochs (best {m.meta['best_epoch'] + 1}), train MSE "
                      f"{m.meta['final_train_mse']:.4f}, validation MSE {m.meta['final_val_mse']:.4f}", end="")
            print(flush=True)
    del fit_runs

    # --- gradient boosting: tune on the `supervised` runs, choose, then the unseen-fault model --------------------------
    train = models.supervised_frame(data.load_runs(FAULTY, ids("supervised")), data.load_runs(NORMAL, ids("supervised")))
    assert sorted(train["faultNumber"].unique()) == list(range(21))
    tune_normal = data.load_runs(NORMAL, ids("limit_tune"))
    dev_faulty, dev_normal = data.load_runs(FAULTY, ids("dev")), data.load_runs(NORMAL, ids("dev"))
    assert sorted(dev_faulty["faultNumber"].unique()) == list(range(1, 21))
    t_grid = time.perf_counter()
    grid, fits = tune_hgb(train, mean, std, lag, tune_normal, dev_faulty, dev_normal, onset, machine)
    grid_time = time.perf_counter() - t_grid
    chosen = models.pick_hgb(grid)
    index = grid.index[(grid[list(chosen)] == pd.Series(chosen)).all(axis=1)][0]
    grid["chosen"] = grid.index == index
    grid.to_csv(OUT / "hgb_tuning.csv", index=False)
    hgb = fits[index]
    del fits
    hgb.meta["tuning_limit_runs"] = ids("limit_tune")
    hgb.meta["grid_time_s"] = grid_time
    fitted[hgb.name] = hgb
    print(f"hgb chosen: {chosen} (grid {grid_time:.0f} s for 8 fits; chosen fit {hgb.meta['train_time_s']:.0f} s)", flush=True)
    unseen = models.HGBMonitor(mean, std, lags=lag, name="hgb-l2-unseen")
    unseen.fit(models.supervised_frame(train[train["faultNumber"] > 0], train[train["faultNumber"] == 0],
                                       exclude=models.UNSEEN_FAULTS), onset=onset, stride=HGB_STRIDE, **chosen)
    assert not set(models.UNSEEN_FAULTS) & set(unseen.estimator.classes_)
    fitted[unseen.name] = unseen
    print(f"{unseen.name}: classes {[int(c) for c in unseen.estimator.classes_]}, fit {unseen.meta['train_time_s']:.0f} s", flush=True)
    del train, tune_normal

    # --- limits from the 100 `limit` runs, run-group check, save ------------------------------------------------------------
    limit_runs = data.load_runs(NORMAL, ids("limit"))
    assert limit_runs["simulationRun"].nunique() == 100
    records = []
    for name, m in fitted.items():
        m.calibrate(limit_runs)
        problems = models.run_group_problems(m.meta, groups["groups"])
        assert not problems, problems
        path = data.MODELS_DIR / f"{name}.joblib"
        m.save(path)
        records.append({"name": name, "file": path.relative_to(data.ROOT).as_posix(), "sha256": models.sha256_file(path),
                        "kind": m.kind, "track": m.track, "lags": m.lags, "hyperparameters": m.meta["hyperparameters"],
                        "limits": m.limits, "n_train_runs": len(m.meta["train_runs"]),
                        "n_val_runs": len(m.meta.get("val_runs", [])), "train_time_s": m.meta["train_time_s"]})
    del limit_runs
    (OUT / "models.json").write_text(json.dumps({"machine": machine, "split_file": SPLIT_FILE, "models": records}, indent=1))

    # --- score the dev runs ---------------------------------------------------------------------------------------------------
    rows = []
    for name, m in fitted.items():
        rows += models.scorecard_rows(m, m.score(dev_faulty), m.score(dev_normal), onset, machine=machine,
                                      split_file=SPLIT_FILE, train_time_s=m.meta["train_time_s"],
                                      peak_ram_gb=prepare.peak_ram_gb())
    (OUT / "dev_scores.csv").unlink(missing_ok=True)
    evaluate.write_scorecard(rows, OUT / "dev_scores.csv")

    # --- print ----------------------------------------------------------------------------------------------------------------------
    df = pd.DataFrame(rows)
    for limit_name in ("1%", "5%"):
        far = df[(df.limit_name == limit_name) & (df.metric == "far")].set_index("model")["value"]
        print(f"dev FAR at {limit_name}: " + ", ".join(f"{m} {v:.4f}" for m, v in far.items()))
    one = df[df.limit_name == "1%"]
    print("dev FDR at the 1% limit")
    print(one[one.metric == "fdr"].pivot(index="fault", columns="model", values="value").loc[list(SHOWN_FAULTS)]
          .round(3).to_string())
    print(one[one.metric.isin(["mean_fdr", "mean_fdr_excl_3_9_15"])].pivot(index="metric", columns="model", values="value")
          .round(3).to_string())
    print(prepare.runtime_line(t0))


if __name__ == "__main__":
    main()
