"""Benchmark plumbing: the GluonTS adapter shared by every model, the GIFT-Eval task list and headline. No benchmark data
is read here; gluonts is optional (`pip install ".[bench]"`) and these tests skip without it."""
import csv
import math
import os

import numpy as np
import pytest
import torch

from benchmarks import gift
from benchmarks.adapter import LEVELS, GluonTSAdapter, nom_forecast_fn
from nom import NomConfig, NomForecaster, NomModel

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "config.json")


def _entries(lengths, freq="h"):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    return [{"target": rng.normal(size=n).astype(np.float32), "start": pd.Period("2024-01-01 00:00", freq=freq),
             "item_id": f"s{i}"} for i, n in enumerate(lengths)]


# ---- adapter ---------------------------------------------------------------------------------------------------------
def test_adapter_emits_quantile_forecasts():
    pytest.importorskip("gluonts")
    calls = []

    def fn(contexts, prediction_length):
        calls.append(len(contexts))
        return np.stack([np.tile(np.arange(9, dtype=np.float32), (prediction_length, 1)) + c[-1] for c in contexts])

    entries = _entries([30, 50, 70])
    out = list(GluonTSAdapter(fn, prediction_length=12, batch_size=2).predict(entries))
    assert calls == [2, 1]                                                  # batched
    assert len(out) == 3
    for e, f in zip(entries, out):
        assert f.start_date == e["start"] + len(e["target"])
        assert f.item_id == e["item_id"]
        np.testing.assert_array_equal(f.quantile(0.5), np.full(12, 4 + e["target"][-1], np.float32))
        assert f.forecast_keys == [str(q) for q in LEVELS]


@pytest.mark.parametrize("bad, match", [
    (lambda c, h: np.zeros((len(c), h, 8)), "shape"),
    (lambda c, h: np.full((len(c), h, 9), np.nan), "non-finite"),
    (lambda c, h: np.zeros((len(c), h + 1, 9)), "shape"),
])
def test_adapter_refuses_bad_outputs(bad, match):
    pytest.importorskip("gluonts")
    with pytest.raises(ValueError, match=match):
        list(GluonTSAdapter(bad, prediction_length=5).predict(_entries([20])))


def test_nom_adapter_equals_forecast():
    pytest.importorskip("gluonts")
    torch.manual_seed(0)
    f = NomForecaster(NomModel(NomConfig.from_json(CONFIG)))
    entries = _entries([40, 3000])
    out = list(GluonTSAdapter(nom_forecast_fn(f), prediction_length=24).predict(entries))
    ref = f.forecast([e["target"] for e in entries], 24)
    for k, fc in enumerate(out):
        np.testing.assert_array_equal(fc.forecast_array, ref.quantiles[k].T)


# ---- GIFT-Eval task list and headline --------------------------------------------------------------------------------
def test_97_official_tasks():
    tasks = gift.tasks()
    assert len(tasks) == 97 and len(set(tasks)) == 97
    assert sum(t == "short" for _, t in tasks) == 55
    assert sum(t == "medium" for _, t in tasks) == sum(t == "long" for _, t in tasks) == 21


@pytest.mark.parametrize("name, term, freq, key", [
    ("saugeenday/D", "short", None, "saugeen/D/short"),
    ("LOOP_SEATTLE/5T", "long", None, "loop_seattle/5T/long"),
    ("m4_yearly", "short", "A", "m4_yearly/A/short"),
    ("kdd_cup_2018_with_missing/H", "medium", None, "kdd_cup_2018/H/medium"),
    ("car_parts_with_missing", "short", "M", "car_parts/M/short"),
])
def test_task_key_matches_the_leaderboard(name, term, freq, key):
    props = {"m4_yearly": {"frequency": "A"}, "car_parts": {"frequency": "M"}}
    assert gift.task_key(name, term, props) == key


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gift.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in gift.COLUMNS})


def _rows(keys, mase, crps):
    return [{"dataset": k, "model": "m", gift.MASE_COL: m, gift.CRPS_COL: c} for k, m, c in zip(keys, mase, crps)]


def test_headline_known_answer(tmp_path):
    keys = [f"t{i}/H/short" for i in range(97)]
    rng = np.random.default_rng(0)
    nm, nc = rng.uniform(0.5, 3, 97), rng.uniform(0.1, 1, 97)
    _write(tmp_path / "naive.csv", _rows(keys, nm, nc))
    r = rng.uniform(0.3, 1.2, 97)
    _write(tmp_path / "model.csv", _rows(keys, nm * r, nc * r[::-1]))
    h = gift.headline(str(tmp_path / "model.csv"), str(tmp_path / "naive.csv"))
    expected = math.exp(np.mean(np.log(r)))
    assert h["n_tasks"] == 97
    assert abs(h["MASE"] - expected) < 1e-12 and abs(h["CRPS"] - expected) < 1e-12


def test_headline_refuses_a_missing_task(tmp_path):
    keys = [f"t{i}/H/short" for i in range(97)]
    _write(tmp_path / "naive.csv", _rows(keys, np.ones(97), np.ones(97)))
    _write(tmp_path / "model.csv", _rows(keys[:96], np.ones(96), np.ones(96)))
    with pytest.raises(ValueError, match="96 of 97 tasks"):
        gift.headline(str(tmp_path / "model.csv"), str(tmp_path / "naive.csv"))


def test_nom_adapter_forecasts_series_with_no_observed_value():
    """Benchmarks contain contexts with no observed value (GIFT-Eval: bitbrains, electricity/W) whose labels are
    scored. `forecast()` refuses them; the benchmark function runs the model on the empty context, as the reference
    implementation does, and leaves the other series unchanged."""
    torch.manual_seed(0)
    f = NomForecaster(NomModel(NomConfig.from_json(CONFIG)))
    full, empty = np.sin(np.arange(300, dtype=np.float32)), np.full(120, np.nan, np.float32)
    q = nom_forecast_fn(f)([full, empty, full], 16)
    assert q.shape == (3, 16, 9) and np.isfinite(q).all()
    np.testing.assert_array_equal(q[[0, 2]], f.forecast([full, full], 16).quantiles)
    np.testing.assert_array_equal(q[1], f._forecast_batch([empty], 16, True)[0])
    with pytest.raises(ValueError, match="no finite value"):
        f.forecast([empty], 16)
