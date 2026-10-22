"""Shapes and contracts of `forecast()`, on a randomly initialised model with the released architecture."""
import os

import numpy as np
import pytest
import torch

from nom import NomConfig, NomForecaster, NomModel

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


@pytest.mark.parametrize("kwargs, match", [
    ({"prediction_length": 721, "flip": False}, "horizons beyond"),
    ({"prediction_length": 24, "flip": True}, "flip"),
    ({"prediction_length": 24, "flip": False, "freq": "H"}, "coarse view"),
])
def test_unported_branches_say_so(forecaster, kwargs, match):
    with pytest.raises(NotImplementedError, match=match):
        forecaster.forecast(_series(100), **kwargs)


def test_long_context_not_yet(forecaster):
    with pytest.raises(NotImplementedError, match="longer than 2048"):
        forecaster.forecast(_series(3000), 24, flip=False)
