"""IND-Eval harness: known answers on synthetic data (no IND-Eval data, no weights)."""
import json
import os

import numpy as np
import pytest

from benchmarks import indeval

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


def test_point_forecast_crps_equals_mae():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=64), rng.normal(size=64)
    sae, sql, n = indeval.pair_stats(np.repeat(p[:, None], 9, 1), y)
    assert n == 64 and abs(sae - sql) < 1e-9


def test_seasonal_naive_is_exact_on_a_pure_season():
    s = np.tile(np.random.default_rng(1).normal(size=24), 100)
    q = indeval.snaive_quantiles(s[:2376], 24, 24)
    np.testing.assert_allclose(q[:, 4], s[2376:])
    np.testing.assert_allclose(q[:, 0], q[:, 8])


def test_seasonal_naive_covers_80_percent_of_a_random_walk():
    rng = np.random.default_rng(2)
    cov = []
    for _ in range(300):
        x = np.cumsum(rng.normal(size=2048 + 64))
        q = indeval.snaive_quantiles(x[:2048], 64, 1)
        cov.append(((x[2048:] >= q[:, 0]) & (x[2048:] <= q[:, 8])).mean())
    assert abs(np.mean(cov) - 0.8) < 0.03


def test_mase_scale_ignores_missing_values():
    assert indeval.seasonal_error([0, 1, np.nan, 3, 5], 1) == pytest.approx(1.5)


def _task(path, role=False, H=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("instance_id", "channel", "window", "context", "target", "seasonality",
                            "prediction_length")}
    if role:
        rows["role"] = []
    for i in range(3):
        for ch in ("a", "cov") if role else ("a",):
            x = np.cumsum(rng.normal(size=200 + H))
            rows["instance_id"].append(f"i{i}"); rows["channel"].append(ch); rows["window"].append(0)
            rows["context"].append(x[:200].tolist()); rows["target"].append(x[200:].tolist())
            rows["seasonality"].append(1); rows["prediction_length"].append(H)
            if role:
                rows["role"].append("covariate" if ch == "cov" else "target")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(rows), path)


def test_known_answers_and_files_per_source(tmp_path):
    """Oracle -> 0, seasonal naive against itself -> 1; covariate channels are never scored; one file per source."""
    _task(str(tmp_path / "data" / "neats" / "t1.parquet"))
    _task(str(tmp_path / "data" / "pronto" / "t2.parquet"), role=True)
    oracle = {"oracle": lambda C, H, Y: [np.repeat(np.asarray(y, np.float64)[:, None], 9, 1) for y in Y]}
    res = indeval.run(str(tmp_path), ["neats", "pronto"], oracle, str(tmp_path / "out"), log=lambda *_: None)
    for s in ("neats", "pronto"):
        assert res["sources"][s]["oracle"] == [0.0, 0.0] and res["sources"][s]["seasonal naive"] == [1.0, 1.0]
        assert json.load(open(tmp_path / "out" / f"{s}.json"))["n_pairs"] == 3       # pronto: the covariate is not scored
    assert json.load(open(tmp_path / "out" / "headline.json"))["sources"] == ["neats", "pronto"]
