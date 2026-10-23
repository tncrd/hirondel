"""One adapter for every model: a quantile function -> a GluonTS predictor that official harnesses can evaluate.

A model enters a benchmark through a single function

    forecast_fn(contexts: list[np.ndarray], prediction_length: int) -> np.ndarray [n_series, prediction_length, 9]

returning the quantiles 0.1, ..., 0.9 on the raw scale, one row per context (1-D float arrays, NaN = missing). The adapter
batches the series, checks the output, and wraps each row into a `gluonts` `QuantileForecast`. Other models plug in the
same way (see `nom_forecast_fn`).
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator

import numpy as np

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ForecastFn = Callable[[list, int], np.ndarray]


class GluonTSAdapter:
    def __init__(self, forecast_fn: ForecastFn, prediction_length: int, batch_size: int = 256,
                 quantile_levels: tuple[float, ...] = LEVELS):
        self.forecast_fn = forecast_fn
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = tuple(quantile_levels)

    def predict(self, test_data_input: Iterable[dict], **kwargs) -> Iterator:
        from gluonts.model.forecast import QuantileForecast
        entries = list(test_data_input)
        keys = [str(q) for q in self.quantile_levels]
        for i in range(0, len(entries), self.batch_size):
            chunk = entries[i:i + self.batch_size]
            contexts = [np.asarray(e["target"], dtype=np.float32) for e in chunk]
            q = np.asarray(self.forecast_fn(contexts, self.prediction_length))
            expected = (len(chunk), self.prediction_length, len(keys))
            if q.shape != expected:
                raise ValueError(f"forecast_fn returned shape {q.shape}, expected {expected}")
            if not np.isfinite(q).all():
                raise ValueError("forecast_fn returned non-finite quantiles")
            for e, qi in zip(chunk, q):
                yield QuantileForecast(forecast_arrays=qi.T, forecast_keys=keys,
                                       start_date=e["start"] + len(e["target"]), item_id=e.get("item_id"))


def nom_forecast_fn(forecaster, flip: bool | None = None, clip_nonnegative: bool | None = None) -> ForecastFn:
    """The released model: `NomForecaster.forecast` with its default inference procedure (flip and zero clipping on
    unless disabled).

    Benchmarks contain contexts with no observed value whose labels are still scored (GIFT-Eval: bitbrains,
    electricity/W). `forecast()` refuses such a series; here it is run through the model on its empty context, as the
    reference implementation does, so that every series gets the model's own forecast."""
    def fn(contexts, prediction_length):
        empty = np.array([not np.isfinite(c).any() for c in contexts], bool)
        out = np.empty((len(contexts), prediction_length, len(LEVELS)), np.float32)
        if (~empty).any():
            out[~empty] = forecaster.forecast([c for c, e in zip(contexts, empty) if not e], prediction_length,
                                              flip=flip, clip_nonnegative=clip_nonnegative).quantiles
        if empty.any():
            use_flip = forecaster.cfg.flip if flip is None else flip
            out[empty] = forecaster._forecast_batch([c for c, e in zip(contexts, empty) if e], prediction_length,
                                                    use_flip)
        return out
    return fn
