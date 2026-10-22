"""Measurement only, outside the published inference path: zero clipping of the quantiles of series whose observed context
is non-negative (Timestral's `clip_nonnegative` rule). Two variants, as forecast functions for the benchmark adapters:

- `after_flip`: the package's forecast (flip average included), then every quantile clipped at 0;
- `per_pass`: the two passes of the flip computed separately, the direct pass clipped before the average (the earlier
  test); the pass on -x is never clipped since its context is non-positive.
"""
from __future__ import annotations

import numpy as np

from benchmarks.adapter import nom_forecast_fn


def non_negative(contexts) -> np.ndarray:
    """Per series: at least one observed value, and every observed value >= 0."""
    out = []
    for c in contexts:
        c = np.asarray(c, np.float64)
        obs = np.isfinite(c)
        out.append(bool(obs.any() and (c[obs] >= 0).all()))
    return np.array(out, bool)


def after_flip(forecaster):
    def fn(contexts, prediction_length):
        q = nom_forecast_fn(forecaster)(contexts, prediction_length)
        keep = non_negative(contexts)[:, None, None]
        return np.where(keep, np.maximum(q, 0), q).astype(np.float32)
    return fn


def per_pass(forecaster, clip: bool = True):
    def fn(contexts, prediction_length):
        one_pass = nom_forecast_fn(forecaster, flip=False)              # handles contexts with no observed value
        up = one_pass(contexts, prediction_length)
        down = one_pass([-np.asarray(c) for c in contexts], prediction_length)
        if clip:
            up = np.where(non_negative(contexts)[:, None, None], np.maximum(up, 0), up)
        return (0.5 * (up - down[..., ::-1])).astype(np.float32)
    return fn
