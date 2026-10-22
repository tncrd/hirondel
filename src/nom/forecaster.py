"""User-facing forecaster: loading, and the inference procedure that turns raw series into quantiles."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch

from .config import NomConfig
from .fusion import CoarseReadings, block_mean_view
from .model import NomModel

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


@dataclass
class Forecast:
    quantiles: np.ndarray                 # [n_series, prediction_length, n_quantiles], raw scale
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> np.ndarray:
        """[n_series, prediction_length]: the 0.5 quantile."""
        return self.quantile(0.5)

    def quantile(self, q: float) -> np.ndarray:
        """[n_series, prediction_length] for one of the predicted levels; other levels are refused, not interpolated."""
        for i, level in enumerate(self.quantile_levels):
            if abs(level - q) < 1e-9:
                return self.quantiles[..., i]
        raise ValueError(f"quantile level {q} is not predicted; available levels: {list(self.quantile_levels)}")


class NomForecaster:
    def __init__(self, model: NomModel, device: Union[str, torch.device] = "cpu", batch_size: int = 256):
        self.model = model.to(device).eval()
        self.cfg = model.cfg
        self.device = torch.device(device)
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str, device: Union[str, torch.device] = "cpu",
                        revision: str | None = None, **kwargs) -> "NomForecaster":
        """Load from a local folder holding `config.json` and `model.safetensors`, or from a Hugging Face repo."""
        from safetensors.torch import load_file
        if os.path.isdir(path_or_repo_id):
            cfg_path = os.path.join(path_or_repo_id, "config.json")
            weights_path = os.path.join(path_or_repo_id, "model.safetensors")
        else:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(path_or_repo_id, "config.json", revision=revision)
            weights_path = hf_hub_download(path_or_repo_id, "model.safetensors", revision=revision)
        model = NomModel(NomConfig.from_json(cfg_path))
        model.load_state_dict(load_file(weights_path), strict=True)
        return cls(model, device=device, **kwargs)

    # ---- inference procedure ---------------------------------------------------------------------
    @staticmethod
    def _to_float(x, what: str) -> np.ndarray:
        try:
            return np.asarray(x, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{what} must be numeric ({e})") from None

    def _as_series(self, context) -> list[np.ndarray]:
        """Accepts a 1-D array (one series), a 2-D array [n_series, length], or a list of 1-D series of any lengths."""
        if isinstance(context, torch.Tensor):
            context = context.detach().cpu().numpy()
        if isinstance(context, np.generic):
            context = np.asarray(context)
        if isinstance(context, np.ndarray) and context.dtype != object:
            if context.ndim == 0 or context.ndim > 2:
                raise ValueError(f"context must be 1-D (one series) or 2-D [n_series, length], got shape {context.shape}")
            context = self._to_float(context[None] if context.ndim == 1 else context, "context")
            series = list(context)
        else:
            try:
                items = list(context)
            except TypeError:
                raise ValueError(f"context must be an array, a tensor or a list of series, got {type(context).__name__}") \
                    from None
            series = []
            for i, c in enumerate(items):
                c = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else c
                c = self._to_float(c, f"series {i}")
                if c.ndim != 1:
                    raise ValueError(f"series {i} must be 1-D, got shape {c.shape}")
                series.append(c)
        if not series:
            raise ValueError("context is empty: pass at least one series")
        for i, c in enumerate(series):
            if c.size == 0:
                raise ValueError(f"series {i} is empty")
            if not np.isfinite(c).any():
                raise ValueError(f"series {i} has no finite value (all missing)")
        return series

    @staticmethod
    def _last(series: list[np.ndarray], length: int) -> np.ndarray:
        """The last `length` points of each series, left-padded with NaN (unobserved) when shorter."""
        out = np.full((len(series), length), np.nan, dtype=np.float32)
        for i, s in enumerate(series):
            s = s[-length:]
            out[i, length - len(s):] = s
        return out

    def _scale(self, ctx: torch.Tensor):
        """Mean / standard deviation over observed points, then asinh. Unobserved points become 0."""
        observed = torch.isfinite(ctx).to(ctx.dtype)
        ctx0 = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        count = observed.sum(1, keepdim=True).clamp_min(1.0)
        mu = (ctx0 * observed).sum(1, keepdim=True) / count
        sigma = ((((ctx0 - mu) * observed) ** 2).sum(1, keepdim=True) / count).sqrt().clamp_min(self.cfg.min_scale)
        return torch.asinh((ctx0 - mu) / sigma) * observed, observed, mu, sigma

    def _unscale(self, q: torch.Tensor, ctx: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Back to raw scale, bounded to the observed context range widened by range_margin x sigma."""
        clip = self.cfg.output_clip
        raw = torch.sinh(q.clamp(-clip, clip)) * sigma.unsqueeze(-1) + mu.unsqueeze(-1)
        finite = torch.isfinite(ctx)
        has_obs = finite.any(1, keepdim=True).unsqueeze(-1)
        lo = torch.where(finite, ctx, torch.full_like(ctx, float("inf"))).amin(1, keepdim=True)
        hi = torch.where(finite, ctx, torch.full_like(ctx, float("-inf"))).amax(1, keepdim=True)
        lo = (lo - self.cfg.range_margin * sigma).unsqueeze(-1)
        hi = (hi + self.cfg.range_margin * sigma).unsqueeze(-1)
        return torch.where(has_obs, raw.clamp(lo, hi), raw)

    def _predict(self, ctx: torch.Tensor, views: list | None, available: torch.Tensor | None,
                 prediction_length: int) -> torch.Tensor:
        """One pass: fine window [n, context_length] raw (NaN = unobserved), optional coarse views
        [n, context_length] raw -> [n, prediction_length, n_quantiles] raw.

        The model always decodes at least its native horizon, then the output is cut. Longer horizons
        are decoded in one pass with as many future tokens as needed."""
        x, observed, mu, sigma = self._scale(ctx)
        coarse = None
        if views is not None:
            scaled = [self._scale(v) for v in views]
            coarse = CoarseReadings(values=[s[0] for s in scaled], observed=[s[1] for s in scaled],
                                    mu=[s[2] for s in scaled], sigma=[s[3] for s in scaled],
                                    available=available, fine_mu=mu, fine_sigma=sigma)
        horizon = max(prediction_length, self.cfg.native_horizon)
        q = self.model(x, observed, horizon, coarse)[:, :prediction_length].float()
        return self._unscale(q, ctx, mu, sigma)

    @torch.no_grad()
    def _forecast_batch(self, series: list[np.ndarray], prediction_length: int, flip: bool) -> np.ndarray:
        L = self.cfg.context_length
        ctx = torch.from_numpy(self._last(series, L)).to(self.device)
        lengths = np.array([len(s) for s in series])
        # A coarse reading k adds history only when the series extends at least k points before the fine window.
        available = np.stack([(lengths >= L) & (lengths - L >= k) for k in self.cfg.coarse_factors], 1)
        views = avail = None
        if available.any():                       # otherwise the gate gives the coarse readings weight 0 exactly
            views = [block_mean_view(torch.from_numpy(self._last(series, k * L)).to(self.device), k, L)
                     for k in self.cfg.coarse_factors]
            avail = torch.from_numpy(available).to(self.device)
        q = self._predict(ctx, views, avail, prediction_length)
        if flip:
            # The q-th quantile of -x is minus the (1-q)-th quantile of x: reverse the levels, then average.
            neg_views = None if views is None else [-v for v in views]
            q = 0.5 * (q - self._predict(-ctx, neg_views, avail, prediction_length).flip(-1))
        return q.float().cpu().numpy()

    def forecast(self, context, prediction_length: int, flip: bool | None = None) -> Forecast:
        """Forecast each series of `context` (1-D array, 2-D array [n, length], or a list of 1-D series).

        NaN values are treated as missing. The last `context_length` points are read at full resolution;
        older history, when present, enters through the coarse readings.
        flip=True (default) averages the forecast of the series with the mirrored forecast of its negation;
        flip=False runs one pass instead of two."""
        flip = self.cfg.flip if flip is None else flip
        if isinstance(prediction_length, bool) or not isinstance(prediction_length, (int, np.integer)) \
                or prediction_length < 1:
            raise ValueError(f"prediction_length must be a positive integer, got {prediction_length!r}")
        prediction_length = int(prediction_length)
        series = self._as_series(context)
        out = [self._forecast_batch(series[i:i + self.batch_size], prediction_length, flip)
               for i in range(0, len(series), self.batch_size)]
        return Forecast(np.concatenate(out, 0), tuple(self.cfg.quantile_levels))

    def forecast_df(self, df, prediction_length: int, id_column: str = "item_id", timestamp_column: str = "timestamp",
                    target: str = "target", freq: str | None = None, flip: bool | None = None):
        """Forecast every series of a long DataFrame (one row per id and timestamp) and return a long DataFrame.

        Each series is sorted by timestamp and placed on a regular grid; missing rows and NaN targets count as
        missing values. `freq` (a pandas offset alias) fixes the grid and the future timestamps; when None it is
        inferred per series. Returns columns [id_column, timestamp_column, "predictions" (the median), "0.1", ...].
        Requires pandas (`pip install "nom[pandas]"`)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('forecast_df needs pandas: pip install "nom[pandas]"') from None
        missing = [c for c in (id_column, timestamp_column, target) if c not in df.columns]
        if missing:
            raise ValueError(f"columns not found in df: {missing}")
        if len(df) == 0:
            raise ValueError("df is empty: pass at least one series")
        ids, series, futures = [], [], []
        for item, g in df.groupby(id_column, sort=False):
            g = g.sort_values(timestamp_column)
            ts = pd.DatetimeIndex(pd.to_datetime(g[timestamp_column]))
            if ts.has_duplicates:
                raise ValueError(f"series {item!r}: duplicate timestamps")
            f = freq or (pd.infer_freq(ts) if len(ts) >= 3 else None)
            if f is None:
                raise ValueError(f"series {item!r}: cannot infer the frequency; pass freq=")
            grid = pd.date_range(ts[0], ts[-1], freq=f)
            if not ts.isin(grid).all():
                raise ValueError(f"series {item!r}: timestamps do not lie on a regular {f!r} grid")
            values = pd.Series(g[target].to_numpy(dtype=np.float64), index=ts).reindex(grid).to_numpy(np.float32)
            ids.append(item)
            series.append(values)
            futures.append(pd.date_range(grid[-1], periods=prediction_length + 1, freq=f)[1:])
        out = self.forecast(series, prediction_length, flip=flip)
        levels = [str(q) for q in out.quantile_levels]
        frames = []
        for i, item in enumerate(ids):
            frame = pd.DataFrame(out.quantiles[i], columns=levels)
            frame.insert(0, "predictions", out.median[i])
            frame.insert(0, timestamp_column, futures[i])
            frame.insert(0, id_column, item)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
