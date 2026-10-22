"""User-facing forecaster: loading, and the inference procedure that turns raw series into quantiles."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch

from .config import NomConfig
from .model import NomModel

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


@dataclass
class Forecast:
    quantiles: np.ndarray                 # [n_series, prediction_length, n_quantiles], raw scale
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> np.ndarray:
        return self.quantiles[..., self.quantile_levels.index(0.5)]


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
    def _as_series(self, context) -> list[np.ndarray]:
        if isinstance(context, torch.Tensor):
            context = context.detach().cpu().numpy()
        if isinstance(context, np.ndarray) and context.dtype != object:
            if context.ndim == 1:
                context = context[None]
            if context.ndim != 2:
                raise ValueError(f"context must be 1-D or 2-D, got shape {context.shape}")
            return [np.asarray(c, dtype=np.float32) for c in context]
        series = []
        for c in context:
            c = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c)
            if c.ndim != 1:
                raise ValueError(f"each series must be 1-D, got shape {c.shape}")
            series.append(c.astype(np.float32))
        return series

    def _left_pad(self, series: list[np.ndarray]) -> np.ndarray:
        L = self.cfg.context_length
        out = np.full((len(series), L), np.nan, dtype=np.float32)
        for i, s in enumerate(series):
            out[i, L - len(s):] = s
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

    @torch.no_grad()
    def _forecast_batch(self, ctx: np.ndarray, prediction_length: int) -> np.ndarray:
        ctx_t = torch.from_numpy(ctx).to(self.device)
        x, observed, mu, sigma = self._scale(ctx_t)
        horizon = max(prediction_length, self.cfg.native_horizon)
        q = self.model(x, observed, horizon)[:, :prediction_length].float()
        return self._unscale(q, ctx_t, mu, sigma).cpu().numpy()

    def forecast(self, context, prediction_length: int, freq: str | None = None,
                 flip: bool | None = None) -> Forecast:
        """Forecast each series of `context` (1-D array, 2-D array [n, length], or a list of 1-D series)."""
        flip = self.cfg.flip if flip is None else flip
        series = self._as_series(context)
        if flip:
            raise NotImplementedError("flip averaging is not available yet; pass flip=False")
        if freq is not None:
            raise NotImplementedError("the long coarse view (freq=...) is not available yet")
        if prediction_length < 1:
            raise ValueError("prediction_length must be >= 1")
        if prediction_length > self.cfg.native_horizon:
            raise NotImplementedError(f"horizons beyond {self.cfg.native_horizon} steps are not available yet")
        if any(len(s) > self.cfg.context_length for s in series):
            raise NotImplementedError(f"contexts longer than {self.cfg.context_length} points are not available yet")
        out = [self._forecast_batch(self._left_pad(series[i:i + self.batch_size]), prediction_length)
               for i in range(0, len(series), self.batch_size)]
        return Forecast(np.concatenate(out, 0), tuple(self.cfg.quantile_levels))
