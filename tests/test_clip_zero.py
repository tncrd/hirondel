"""Zero clipping measurement: the two variants, and that the package default reproduces `after_flip`."""
import os

import numpy as np
import pytest
import torch

from benchmarks.experiments import clip_zero
from nom import NomConfig, NomForecaster, NomModel

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "config.json")


@pytest.fixture(scope="module")
def f():
    torch.manual_seed(0)
    return NomForecaster(NomModel(NomConfig.from_json(CONFIG)))


def _ctx():
    t = np.arange(300, dtype=np.float32)
    pos = np.maximum(0, np.sin(2 * np.pi * t / 24)) * 3              # non-negative, many zeros
    neg = np.sin(2 * np.pi * t / 24)                                  # takes negative values
    gaps = pos.copy()
    gaps[:50] = np.nan
    return [pos, neg, gaps]


def test_after_flip_clips_only_non_negative_contexts(f):
    ctx = _ctx()
    ref = f.forecast(ctx, 24, clip_nonnegative=False).quantiles
    q = clip_zero.after_flip(f)(ctx, 24)
    assert (q[[0, 2]] >= 0).all()
    np.testing.assert_array_equal(q[1], ref[1])
    np.testing.assert_array_equal(q[[0, 2]], np.maximum(ref[[0, 2]], 0))


def test_per_pass_without_clipping_rebuilds_the_flip_forecast(f):
    ctx = _ctx()
    np.testing.assert_array_equal(clip_zero.per_pass(f, clip=False)(ctx, 24),
                                  f.forecast(ctx, 24, clip_nonnegative=False).quantiles)


def test_per_pass_clips_the_direct_pass_only(f):
    ctx = _ctx()
    up = f.forecast(ctx, 24, flip=False, clip_nonnegative=False).quantiles
    down = f.forecast([-c for c in ctx], 24, flip=False, clip_nonnegative=False).quantiles
    expected = 0.5 * (np.where(clip_zero.non_negative(ctx)[:, None, None], np.maximum(up, 0), up) - down[..., ::-1])
    np.testing.assert_array_equal(clip_zero.per_pass(f)(ctx, 24), expected.astype(np.float32))


def test_non_negative_rule():
    assert clip_zero.non_negative([np.array([0.0, 1.0]), np.array([np.nan, 2.0]), np.array([-1.0, 3.0]),
                                   np.array([np.nan, np.nan])]).tolist() == [True, True, False, False]


def test_variants_accept_contexts_with_no_observed_value(f):
    ctx = _ctx() + [np.full(80, np.nan, np.float32)]
    for fn in (clip_zero.after_flip(f), clip_zero.per_pass(f)):
        q = fn(ctx, 12)
        assert q.shape == (4, 12, 9) and np.isfinite(q).all()


def test_package_default_reproduces_the_after_flip_measurement(f):
    """The released default (clip_nonnegative) gives exactly the outputs that were measured as `after_flip`."""
    ctx = _ctx() + [np.full(80, np.nan, np.float32)]
    from benchmarks.adapter import nom_forecast_fn
    np.testing.assert_array_equal(nom_forecast_fn(f)(ctx, 24), clip_zero.after_flip(f)(ctx, 24))
