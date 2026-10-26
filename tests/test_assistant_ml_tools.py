"""Phase 3 tests for the v1.1 MCP tools (score_model, classify_fault) in src/assistant/server.py.

A tiny model folder is built from the same synthetic plant as the `world` fixture: five small models under their
assistant names, fitted on normal runs (and three step faults for the classifier).
"""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from src import data, historian, models
from src.assistant import server

SCALE = np.linspace(0.5, 2, 52)
BASE = np.linspace(10, 20, 52)
ONSET = 20


def runs(run_ids, fault=0, n=60, seed=1):
    """Synthetic runs like the world fixture's; a fault adds a +40 step on reading `fault` after the onset."""
    rng = np.random.default_rng(seed)
    parts = []
    for r in run_ids:
        x = rng.normal(size=(n, 52)) * SCALE + BASE
        if fault:
            x[ONSET:, fault - 1] += 40.0
        frame = pd.DataFrame(x.astype(np.float32), columns=data.VARIABLES)
        frame.insert(0, "sample", np.arange(1, n + 1))
        frame.insert(0, "simulationRun", r)
        frame.insert(0, "faultNumber", fault)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def ml_dir(world, tmp_path_factory):
    folder = tmp_path_factory.mktemp("ml_models")
    mean, std = world["monitor"].mean, world["monitor"].std
    fit, limit = runs(range(1, 9), n=100, seed=2), runs(range(101, 105), n=100, seed=3)
    sup = models.supervised_frame(pd.concat([runs(range(1, 5), f, 100, seed=10 + f) for f in (1, 2, 3)]),
                                  runs(range(1, 5), 0, 100, seed=4))
    for lags in (0, 2):
        iforest = models.IFMonitor(mean, std, lags=lags).fit(fit)
        ae = models.AEMonitor(mean, std, lags=lags).fit(fit, bottleneck=6, max_epochs=3, n_val_runs=2)
        for m in (iforest, ae):
            m.calibrate(limit)
            m.save(folder / f"{m.name}.joblib")
    hgb = models.HGBMonitor(mean, std, lags=2).fit(sup, onset=ONSET, max_iter=10)
    hgb.calibrate(limit)
    hgb.save(folder / "hgb-l2.joblib")
    return folder


def tools(world, ml_dir, case="dev_01", **kw):
    return server.TepTools(world["db"], case, world["npz"], world["notes"], models_dir=ml_dir, **kw)


# --- score_model -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", models.ASSISTANT_MODELS)
def test_score_model_returns_the_models_alarm_scores_with_its_1pct_limit(world, ml_dir, name):
    out = json.loads(tools(world, ml_dir).score_model(name, 0.0, 100.0))
    monitor = models.load(ml_dir / f"{name}.joblib")
    expected = models.score_case(monitor, pd.DataFrame(world["frames"]["dev_01"]))
    assert out["model"] == name and out["limit"] == pytest.approx(monitor.limits["1%"], rel=1e-4)
    first = 0.0 if monitor.lags == 0 else 0.1
    assert out["t_hours"][0] == first and len(out["alarm"]) == len(expected) == len(out["t_hours"])
    assert np.allclose(out["alarm"], expected["alarm"], rtol=1e-4)
    assert out["truncated"] is False


def test_score_model_takes_a_window_and_a_step(world, ml_dir):
    out = json.loads(tools(world, ml_dir).score_model("iforest-l2", 1.0, 2.0, 5))
    assert out["t_hours"] == [1.0, 1.25, 1.5, 1.75, 2.0] and len(out["alarm"]) == 5


def test_score_model_output_is_capped_like_score_pca_and_says_so(world, ml_dir):
    out = json.loads(tools(world, ml_dir, max_values=50).score_model("ae-l0", 0.0, 100.0))
    assert out["truncated"] is True and len(out["alarm"]) == len(out["t_hours"]) == 50
    assert "50-value cap" in out["note"] and "of 240" in out["note"]
    assert server.MAX_VALUES == 2000
    full = json.loads(tools(world, ml_dir).score_model("ae-l0", 0.0, 100.0))
    assert full["truncated"] is False and len(full["alarm"]) == 240


def test_score_model_rejects_unknown_models_and_bad_windows(world, ml_dir):
    t = tools(world, ml_dir)
    with pytest.raises(ValueError, match="iforest-l0"):
        t.score_model("pca", 0, 1)
    with pytest.raises(ValueError, match="no samples"):
        t.score_model("ae-l0", 500.0, 600.0)
    with pytest.raises(ValueError, match="step"):
        t.score_model("ae-l0", 0, 1, 0)


def test_score_model_sees_only_its_own_case(world, ml_dir):
    a = json.loads(tools(world, ml_dir, "dev_01").score_model("iforest-l0", 0.0, 100.0))
    b = json.loads(tools(world, ml_dir, "dev_02").score_model("iforest-l0", 0.0, 100.0))
    assert a["t_hours"] == b["t_hours"] and a["alarm"] != b["alarm"]
    assert np.mean(b["alarm"]) > np.mean(a["alarm"])          # dev_02 is shifted by +50 on every reading


# --- classify_fault ----------------------------------------------------------------------------------------------------------
def test_classify_fault_returns_three_classes_with_probabilities_summing_to_at_most_one(world, ml_dir):
    out = json.loads(tools(world, ml_dir).classify_fault(0.0, 100.0))
    assert len(out["top"]) == 3
    probs = [c["probability"] for c in out["top"]]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1.0
    assert all(set(c) == {"class", "probability"} for c in out["top"])
    assert out["n_samples"] > 0


def test_class_zero_is_shown_as_normal_and_faults_as_idv_numbers(world, ml_dir):
    quiet = json.loads(tools(world, ml_dir, "dev_03").classify_fault())
    assert quiet["top"][0]["class"] == "normal"
    stepped = json.loads(tools(world, ml_dir, "dev_01").classify_fault())      # a +40 step on xmeas_1 = the fault-1 style
    assert stepped["top"][0]["class"] == "IDV 1"
    assert all(c["class"] == "normal" or c["class"].startswith("IDV ") for c in stepped["top"])


def test_classify_fault_default_window_starts_at_the_first_spe_raise(world, ml_dir):
    t = tools(world, ml_dir, "dev_01")
    raises = historian.CaseView(world["db"], "dev_01").alarms()
    first = float(raises[(raises.statistic == "spe") & (raises.event == "raise")]["t_hours"].min())
    default = json.loads(t.classify_fault())
    explicit = json.loads(t.classify_fault(first, None))
    assert default["window_hours"][0] == pytest.approx(first, abs=0.051) and "first SPE raise" in default["window"]
    assert default["top"] == explicit["top"] and default["n_samples"] == explicit["n_samples"]
    whole = json.loads(t.classify_fault(0.0, 100.0))
    assert whole["n_samples"] > default["n_samples"]


def test_classify_fault_defaults_to_the_whole_case_without_an_spe_raise(world, ml_dir):
    out = json.loads(tools(world, ml_dir, "dev_03").classify_fault())
    assert "whole case" in out["window"]
    assert out["n_samples"] == 240 - 2                                     # the first L = 2 samples have no history


def test_classify_fault_rejects_an_empty_window(world, ml_dir):
    with pytest.raises(ValueError, match="no samples"):
        tools(world, ml_dir).classify_fault(500.0, 600.0)


# --- the MCP layer ----------------------------------------------------------------------------------------------------------------
def test_the_mcp_server_exposes_the_six_v1_tools_plus_the_two_model_tools(world, ml_dir):
    from mcp import Client

    mcp_server = server.build_server(tools(world, ml_dir))

    async def run():
        async with Client(mcp_server) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            scored = await client.call_tool("score_model", {"model": "hgb-l2", "start_h": 0, "end_h": 1})
            classified = await client.call_tool("classify_fault", {})
            bad = await client.call_tool("score_model", {"model": "nope", "start_h": 0, "end_h": 1})
            return names, scored, classified, bad

    names, scored, classified, bad = asyncio.run(run())
    assert names == sorted(["query_historian", "get_alarms", "score_pca", "top_contributors", "list_notes", "read_note",
                            "score_model", "classify_fault"])
    assert not scored.is_error and "alarm" in scored.content[0].text
    assert not classified.is_error and "top" in classified.content[0].text
    assert bad.is_error and "iforest-l0" in bad.content[0].text
