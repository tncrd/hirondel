"""Known-future covariates (`hirondel.covariates`): exactness without covariates, the ridge, the gate and the inputs.

Exactness is checked on the released architecture (random weights). The accuracy tests replace the network by a
naive forecaster (mean of the last 24 observed points), so that they check the covariate procedure alone, fast."""
import os

import numpy as np
import pytest
import torch

from hirondel import HirondelConfig, HirondelForecaster, HirondelModel
from hirondel import covariates as cov

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "config.json")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return HirondelModel(HirondelConfig.from_json(CONFIG))


@pytest.fixture(scope="module")
def forecaster(model):
    return HirondelForecaster(model)


class Naive(HirondelForecaster):
    """Median = mean of the last 24 observed points, quantiles spread by their deviation."""

    def _forecast_batch(self, series, prediction_length, flip):
        levels = np.array(self.cfg.quantile_levels)
        out = []
        for s in series:
            last = s[np.isfinite(s)][-24:]
            out.append(np.broadcast_to(last.mean() + last.std() * (levels - 0.5) * 2,
                                       (prediction_length, len(levels))))
        return np.stack(out).astype(np.float32)


@pytest.fixture(scope="module")
def naive(model):
    return Naive(model)


def _ar(n, seed, phi=0.8, scale=0.3):
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=scale, size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def _driven(T, H, seed, coef=3.0):
    """y = AR noise + coef * x, with x a known-future step signal; returns (context, truth, x over T + H)."""
    rng = np.random.default_rng(seed + 1000)
    x = np.repeat(rng.normal(size=(T + H) // 8 + 1), 8)[:T + H]
    y = 10 + _ar(T + H, seed) + coef * x
    return y[:T].astype(np.float32), y[T:], x


def test_no_covariates_and_off_are_bit_identical(forecaster):
    rng = np.random.default_rng(0)
    series = [rng.normal(size=n).astype(np.float32) for n in (40, 300, 2100)]
    base = forecaster.forecast(series, 24)
    assert base.covariate_used is None
    X = [{"x": rng.normal(size=len(s) + 24)} for s in series]
    for out in (forecaster.forecast(series, 24, future_covariates=None),
                forecaster.forecast(series, 24, future_covariates=X, covariate_mode="off")):
        np.testing.assert_array_equal(out.quantiles, base.quantiles)
    assert not forecaster.forecast(series, 24, future_covariates=X, covariate_mode="off").covariate_used.any()


def test_ridge_reaches_the_noise_floor(naive):
    T, H = 400, 48
    err_off, err_ridge, err_floor = [], [], []
    for seed in range(5):
        y, truth, x = _driven(T, H, seed)
        off = naive.forecast(y, H).median[0]
        out = naive.forecast(y, H, future_covariates={"x": x}, covariate_mode="always")
        assert out.covariate_used[0]
        err_off.append(np.abs(off - truth).mean())
        err_ridge.append(np.abs(out.median[0] - truth).mean())
        # Floor: the naive forecast of the AR noise itself, knowing the covariate effect exactly.
        noise = (y - 3.0 * x[:T]).astype(np.float32)
        err_floor.append(np.abs(naive.forecast(noise, H).median[0] + 3.0 * x[T:] - truth).mean())
    assert np.mean(err_ridge) < 1.25 * np.mean(err_floor)   # beta is estimated from an autocorrelated context
    assert np.mean(err_ridge) < 0.3 * np.mean(err_off)


def test_gate_keeps_a_useful_covariate(naive):
    y, truth, x = _driven(400, 48, 0)
    out = naive.forecast(y, 48, future_covariates={"x": x})
    assert out.covariate_used[0]


def test_gate_on_pure_noise_is_not_worse_than_off(naive):
    T, H = 300, 24
    off, gate = [], []
    for seed in range(20):
        y = (10 + _ar(T + H, seed)).astype(np.float32)
        noise = np.random.default_rng(seed + 50).normal(size=(T + H, 3))
        off.append(np.abs(naive.forecast(y[:T], H).median[0] - y[T:]).mean())
        gate.append(np.abs(naive.forecast(y[:T], H, future_covariates=[noise]).median[0] - y[T:]).mean())
    assert np.mean(gate) <= 1.02 * np.mean(off)


def test_jump_beyond_the_past_range_is_clipped(naive):
    T, H = 200, 24
    y, _, x = _driven(T, H, 3)
    flat = np.zeros(T + H)
    flat[T - 5:T] = 1e-3                          # almost constant in the past
    flat[T:] = 1e8                                # huge in the future (walmart MarkDown case)
    out = naive.forecast(y, H, future_covariates={"x": x, "markdown": flat}, covariate_mode="always")
    assert np.isfinite(out.quantiles).all()
    assert np.abs(out.median).max() < 100 * np.abs(y).max()


def test_categorical_column(naive):
    T, H = 300, 24
    day = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])[np.arange(T + H) % 7]
    y = 10 + _ar(T + H, 4) + 4.0 * np.isin(day, ["sat", "sun"])
    out = naive.forecast(y[:T].astype(np.float32), H, future_covariates={"day": day}, covariate_mode="always")
    off = naive.forecast(y[:T].astype(np.float32), H)
    assert out.covariate_used[0]
    assert np.abs(out.median[0] - y[T:]).mean() < 0.5 * np.abs(off.median[0] - y[T:]).mean()
    enc = cov.encode_column(day)
    assert enc.shape == (T + H, 6)                # one-hot, first level as reference
    many = np.array([f"level{i % 40}" for i in range(T + H)])
    assert cov.encode_column(many).shape == (T + H, 1)   # more than 16 levels: ordinal


def test_missing_values_in_target_and_covariates(naive):
    T, H = 300, 24
    y, truth, x = _driven(T, H, 5)
    y[::7] = np.nan
    x = x.copy()
    x[10:20] = np.nan
    out = naive.forecast(y, H, future_covariates={"x": x}, covariate_mode="always")
    assert np.isfinite(out.quantiles).all() and out.covariate_used[0]
    cat = np.array(["a", None, "b", "a"] * ((T + H) // 4))
    assert np.isfinite(cov.encode_column(cat)).all()
    pd = pytest.importorskip("pandas")
    nullable = pd.array([1, 2, None, 4], dtype="Int64").to_numpy(dtype=object)   # numeric with pd.NA: stays numeric
    np.testing.assert_array_equal(cov.encode_column(nullable)[:, 0], [1, 2, 2, 4])


def test_short_series_falls_back_to_the_plain_forecast(naive):
    y = np.arange(6, dtype=np.float32)
    x = np.random.default_rng(6).normal(size=(6 + 4, 5))
    out = naive.forecast(y, 4, future_covariates=[x], covariate_mode="always")
    assert not out.covariate_used[0]
    np.testing.assert_array_equal(out.quantiles, naive.forecast(y, 4).quantiles)


def test_series_without_covariates_in_a_batch(naive):
    y, _, x = _driven(300, 24, 7)
    out = naive.forecast([y, y], 24, future_covariates=[{"x": x}, None], covariate_mode="always")
    assert out.covariate_used.tolist() == [True, False]
    np.testing.assert_array_equal(out.quantiles[1], naive.forecast(y, 24).quantiles[0])


def test_gate_choice_reads_only_the_context(naive):
    """The gate decides on the context: changing the covariate over the horizon changes the forecast, not the choice."""
    T, H = 400, 48
    for seed in range(6):
        y, _, x = _driven(T, H, seed, coef=float(seed % 3))
        x2 = x.copy()
        x2[T:] = np.random.default_rng(seed).normal(size=H) * 5
        a = naive.forecast(y, H, future_covariates={"x": x})
        b = naive.forecast(y, H, future_covariates={"x": x2})
        assert a.covariate_used[0] == b.covariate_used[0]


def test_input_validation(naive):
    y = np.zeros(50, np.float32)
    with pytest.raises(ValueError, match="past-only"):
        naive.forecast(y, 10, future_covariates={"x": np.zeros(50)})
    with pytest.raises(ValueError, match="expected 60"):
        naive.forecast(y, 10, future_covariates={"x": np.zeros(55)})
    with pytest.raises(ValueError, match="entries for 2 series"):
        naive.forecast([y, y], 10, future_covariates=[{"x": np.zeros(60)}])
    with pytest.raises(ValueError, match="covariate_mode"):
        naive.forecast(y, 10, future_covariates={"x": np.zeros(60)}, covariate_mode="ridge")
    with pytest.raises(ValueError, match="covariate_model"):
        naive.forecast(y, 10, future_covariates={"x": np.zeros(60)}, covariate_model="lightgbm")


def test_forecast_df_with_known_covariates(naive):
    pd = pytest.importorskip("pandas")
    T, H = 300, 24
    rows, future = [], []
    for item in ("a", "b"):
        y, _, x = _driven(T, H, 8 if item == "a" else 9)
        ts = pd.date_range("2024-01-01", periods=T + H, freq="h")
        rows.append(pd.DataFrame({"item_id": item, "timestamp": ts[:T], "target": y, "x": x[:T]}))
        future.append(pd.DataFrame({"item_id": item, "timestamp": ts[T:], "x": x[T:]}))
    df, fdf = pd.concat(rows), pd.concat(future)
    out = naive.forecast_df(df, H, known_covariates=["x"], future_df=fdf, covariate_mode="always")
    direct = naive.forecast([g["target"].to_numpy(np.float32) for _, g in df.groupby("item_id")], H,
                            future_covariates=[{"x": np.concatenate([g["x"], f["x"]])}
                                               for (_, g), (_, f) in zip(df.groupby("item_id"), fdf.groupby("item_id"))],
                            covariate_mode="always")
    np.testing.assert_allclose(out["predictions"].to_numpy().reshape(2, H), direct.median, rtol=1e-6)
    with pytest.raises(ValueError, match="exactly the 24"):
        naive.forecast_df(df, H, known_covariates=["x"], future_df=fdf.iloc[:-1])
    with pytest.raises(ValueError, match="needs future_df"):
        naive.forecast_df(df, H, known_covariates=["x"])
