"""Shapes and contracts of `forecast()`, on a randomly initialised model with the released architecture."""
import os

import numpy as np
import pytest
import torch

from nom import NomConfig, NomForecaster, NomModel
from nom.forecaster import Forecast

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "config.json")


@pytest.fixture(scope="module")
def forecaster():
    torch.manual_seed(0)
    return NomForecaster(NomModel(NomConfig.from_json(CONFIG)))


def _series(n, seed=0):
    t = np.arange(n)
    return (np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(seed).normal(size=n)).astype(np.float32)


def test_shapes_and_median(forecaster):
    out = forecaster.forecast([_series(100), _series(700, 1)], 48, flip=False)
    assert out.quantiles.shape == (2, 48, 9)
    assert out.quantiles.dtype == np.float32
    np.testing.assert_array_equal(out.median, out.quantiles[..., 4])


def test_quantiles_never_cross(forecaster):
    q = forecaster.forecast(_series(300), 720, flip=False).quantiles
    assert np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all()


def test_input_kinds_agree(forecaster):
    s = _series(256)
    a = forecaster.forecast(s, 24, flip=False).quantiles
    b = forecaster.forecast(torch.from_numpy(s), 24, flip=False).quantiles
    c = forecaster.forecast([s.tolist()], 24, flip=False).quantiles
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_batching_does_not_change_a_series(forecaster):
    series = [_series(64 * (i + 1), i) for i in range(5)]
    whole = forecaster.forecast(series, 32, flip=False).quantiles
    small = NomForecaster(forecaster.model, batch_size=2).forecast(series, 32, flip=False).quantiles
    np.testing.assert_allclose(small, whole, rtol=1e-5, atol=1e-5)


def test_only_the_last_context_length_points_are_read(forecaster):
    s = _series(2051)
    np.testing.assert_array_equal(forecaster.forecast(s, 24).quantiles, forecaster.forecast(s[-2048:], 24).quantiles)


def test_flip_is_the_default(forecaster):
    s = _series(300)
    default = forecaster.forecast(s, 24).quantiles
    np.testing.assert_array_equal(default, forecaster.forecast(s, 24, flip=True).quantiles)
    assert not np.array_equal(default, forecaster.forecast(s, 24, flip=False).quantiles)


def test_horizon_beyond_native(forecaster):
    q = forecaster.forecast(_series(500), 1000).quantiles
    assert q.shape == (1, 1000, 9) and np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all()


def test_missing_values_are_accepted(forecaster):
    s = _series(400)
    s[:30] = np.nan
    s[200:240] = np.nan
    q = forecaster.forecast(s, 48).quantiles
    assert np.isfinite(q).all()


def test_flip_is_sign_equivariant(forecaster):
    s = _series(300)
    q, q_neg = forecaster.forecast(s, 24).quantiles, forecaster.forecast(-s, 24).quantiles
    np.testing.assert_allclose(q_neg, -q[..., ::-1], rtol=1e-5, atol=1e-5)


def test_older_history_enters_through_the_coarse_reading(forecaster):
    s = _series(6000)
    with_history = forecaster.forecast(s, 96).quantiles
    assert not np.array_equal(with_history, forecaster.forecast(s[-2048:], 96).quantiles)
    # Only the last 4 x 2048 points are read.
    long = _series(9000)
    np.testing.assert_array_equal(forecaster.forecast(long, 96).quantiles,
                                  forecaster.forecast(long[-4 * 2048:], 96).quantiles)


def test_a_short_series_ignores_long_neighbours(forecaster):
    short = _series(300, 3)
    alone = forecaster.forecast(short, 48).quantiles
    mixed = forecaster.forecast([short, _series(7000, 4)], 48).quantiles[:1]
    np.testing.assert_allclose(mixed, alone, rtol=1e-5, atol=1e-5)


def test_no_frequency_argument(forecaster):
    with pytest.raises(TypeError):
        forecaster.forecast(_series(100), 24, freq="H")


# ---- result object ---------------------------------------------------------------------------------------------------
def test_quantile_reads_the_right_level(forecaster):
    out = forecaster.forecast([_series(200), _series(300, 1)], 24)
    for i, q in enumerate(out.quantile_levels):
        np.testing.assert_array_equal(out.quantile(q), out.quantiles[..., i])
    np.testing.assert_array_equal(out.median, out.quantile(0.5))
    assert out.quantile(0.1).shape == (2, 24)


@pytest.mark.parametrize("q", [0.02, 0.25, 0.97, 1.5])
def test_quantile_refuses_a_level_not_predicted(forecaster, q):
    out = forecaster.forecast(_series(200), 24)
    with pytest.raises(ValueError, match="is not available; predicted levels: .*, extrapolated levels"):
        out.quantile(q)


# ---- extrapolated tail levels ------------------------------------------------------------------------------------------
def _expected_tails(Q):
    """The formula of the tail levels, written out (levels 0.1, 0.2, 0.8, 0.9 at indices 0, 1, 7, 8)."""
    a, b = (Q[..., 1] - Q[..., 0]) / np.log(2.0), (Q[..., 8] - Q[..., 7]) / np.log(2.0)
    return {0.01: Q[..., 0] + a * np.log(0.1), 0.05: Q[..., 0] + a * np.log(0.5),
            0.95: Q[..., 8] - b * np.log(0.5), 0.99: Q[..., 8] - b * np.log(0.1)}


@pytest.mark.parametrize("flip", [False, True])
def test_tail_levels_follow_the_exponential_formula(forecaster, flip):
    out = forecaster.forecast([_series(200), _series(3000, 1)], 48, flip=flip)
    for q, expected in _expected_tails(out.quantiles.astype(np.float64)).items():
        got = out.quantile(q)
        assert got.shape == (2, 48) and got.dtype == np.float32
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_tail_levels_are_extrapolated_from_the_flip_averaged_quantiles(forecaster):
    s = _series(400)
    out = forecaster.forecast(s, 24, flip=True)
    rebuilt = Forecast(out.quantiles.copy(), out.quantile_levels)       # only the averaged quantiles, nothing else
    for q in Forecast.EXTRAPOLATED_LEVELS:
        np.testing.assert_array_equal(out.quantile(q), rebuilt.quantile(q))


def test_tail_levels_recover_laplace_quantiles():
    """Laplace(mu, b) has exactly exponential tails: the extrapolation must return its true quantiles."""
    levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    mu = np.array([[0.0, 3.0], [-2.5, 1e3]])
    b = np.array([[1.0, 0.2], [4.0, 30.0]])

    def laplace(t):
        return mu + b * (np.log(2 * t) if t < 0.5 else -np.log(2 * (1 - t)))

    out = Forecast(np.stack([laplace(t) for t in levels], -1), levels)
    for t in (0.01, 0.05, 0.95, 0.99):
        np.testing.assert_allclose(out.quantile(t), laplace(t), rtol=0, atol=1e-9)


def test_tail_levels_never_cross(forecaster):
    out = forecaster.forecast([_series(300), _series(5000, 2)], 720)
    all13 = np.stack([out.quantile(q) for q in out.all_levels], -1)
    assert out.all_levels == (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
    assert (np.diff(all13, axis=-1) >= 0).all()


def test_predicted_levels_are_untouched_by_the_tails(forecaster):
    out = forecaster.forecast(_series(300), 24)
    before = out.quantiles.copy()
    for q in Forecast.EXTRAPOLATED_LEVELS:
        out.quantile(q)
    np.testing.assert_array_equal(out.quantiles, before)
    assert out.quantiles.shape[-1] == 9


def test_tails_need_the_outer_predicted_levels():
    out = Forecast(np.zeros((1, 4, 3)), (0.25, 0.5, 0.75))
    with pytest.raises(ValueError, match="needs the predicted levels 0.1, 0.2, 0.8 and 0.9"):
        out.quantile(0.99)


# ---- clear errors ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("context, match", [
    ([], "context is empty"),
    (np.zeros((0, 10), np.float32), "context is empty"),
    ([np.array([], np.float32)], "series 0 is empty"),
    ([_series(50), np.full(30, np.nan, np.float32)], "series 1 has no finite value"),
    (np.full((2, 40), np.nan), "series 0 has no finite value"),
    (np.zeros((2, 3, 4)), "1-D \\(one series\\) or 2-D"),
    (np.float32(3.0), "1-D \\(one series\\) or 2-D"),
    ([np.zeros((3, 4))], "series 0 must be 1-D"),
    ([["a", "b"]], "series 0 must be numeric"),
    (42, "context must be an array"),
])
def test_bad_context_raises_a_clear_error(forecaster, context, match):
    with pytest.raises(ValueError, match=match):
        forecaster.forecast(context, 12)


@pytest.mark.parametrize("pl", [0, -3, 2.5, True, "24"])
def test_bad_prediction_length(forecaster, pl):
    with pytest.raises(ValueError, match="prediction_length must be a positive integer"):
        forecaster.forecast(_series(100), pl)


def test_infinite_values_are_missing_values(forecaster):
    s = _series(300)
    s_inf = s.copy(); s_inf[100:110] = np.inf
    s_nan = s.copy(); s_nan[100:110] = np.nan
    np.testing.assert_array_equal(forecaster.forecast(s_inf, 24).quantiles, forecaster.forecast(s_nan, 24).quantiles)


def test_device_is_the_users_choice():
    torch.manual_seed(0)
    f = NomForecaster(NomModel(NomConfig.from_json(CONFIG)), device="cpu")
    assert f.device.type == "cpu" and all(p.device.type == "cpu" for p in f.model.parameters())


# ---- forecast_df (pandas optional) --------------------------------------------------------------------------------
def _long_df(pd, lengths=(150, 400), freq="h"):
    rows = []
    for k, n in enumerate(lengths):
        ts = pd.date_range("2024-01-01", periods=n, freq=freq)
        rows.append(pd.DataFrame({"item_id": f"s{k}", "timestamp": ts, "target": _series(n, k)}))
    return pd.concat(rows, ignore_index=True)


def test_forecast_df_equals_forecast(forecaster):
    pd = pytest.importorskip("pandas")
    df = _long_df(pd)
    out = forecaster.forecast_df(df, 24)
    ref = forecaster.forecast([_series(150, 0), _series(400, 1)], 24)
    assert list(out.columns) == ["item_id", "timestamp", "predictions"] + [str(q) for q in ref.quantile_levels]
    for k in range(2):
        part = out[out["item_id"] == f"s{k}"]
        np.testing.assert_array_equal(part[[str(q) for q in ref.quantile_levels]].to_numpy(np.float32), ref.quantiles[k])
        np.testing.assert_array_equal(part["predictions"].to_numpy(np.float32), ref.median[k])
    last = df[df["item_id"] == "s0"]["timestamp"].max()
    assert list(out[out["item_id"] == "s0"]["timestamp"])[:2] == [last + pd.Timedelta(hours=1), last + pd.Timedelta(hours=2)]


def test_forecast_df_tail_columns_on_request(forecaster):
    pd = pytest.importorskip("pandas")
    df = _long_df(pd)
    ref = forecaster.forecast([_series(150, 0), _series(400, 1)], 24)
    out = forecaster.forecast_df(df, 24, tail_quantiles=True)
    levels = ["0.01", "0.05"] + [str(q) for q in ref.quantile_levels] + ["0.95", "0.99"]
    assert list(out.columns) == ["item_id", "timestamp", "predictions"] + levels
    for k in range(2):
        part = out[out["item_id"] == f"s{k}"]
        for q in ref.all_levels:
            np.testing.assert_array_equal(part[str(q)].to_numpy(np.float32), ref.quantile(q)[k])


def test_forecast_df_missing_rows_are_missing_values(forecaster):
    pd = pytest.importorskip("pandas")
    df = _long_df(pd, lengths=(200,))
    gap = df.drop(index=range(50, 60))
    full = df.copy()
    full.loc[50:59, "target"] = np.nan
    np.testing.assert_array_equal(forecaster.forecast_df(gap, 12, freq="h").iloc[:, 2:].to_numpy(),
                                  forecaster.forecast_df(full, 12).iloc[:, 2:].to_numpy())


def test_forecast_df_unsorted_rows(forecaster):
    pd = pytest.importorskip("pandas")
    df = _long_df(pd)
    np.testing.assert_array_equal(forecaster.forecast_df(df.sample(frac=1.0, random_state=0), 12).iloc[:, 2:].to_numpy(),
                                  forecaster.forecast_df(df, 12).iloc[:, 2:].to_numpy())


@pytest.mark.parametrize("make, match", [
    (lambda df: df.rename(columns={"target": "y"}), "columns not found"),
    (lambda df: df.iloc[:0], "df is empty"),
    (lambda df: df.iloc[[0, 1, 1, 2]], "duplicate timestamps"),
    (lambda df: df.iloc[[0, 1]], "cannot infer the frequency"),
])
def test_forecast_df_errors(forecaster, make, match):
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match=match):
        forecaster.forecast_df(make(_long_df(pd, lengths=(100,))), 12)


def test_forecast_df_without_pandas_says_how_to_install(forecaster, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_pandas(name, *a, **k):
        if name == "pandas":
            raise ImportError("no pandas")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_pandas)
    with pytest.raises(ImportError, match='pip install "nom\\[pandas\\]"'):
        forecaster.forecast_df(None, 12)
