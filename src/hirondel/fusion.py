"""Multi-resolution fusion.

The same network also reads each series at coarser resolutions: the last k x context_length points
averaged in blocks of k. Each coarse reading forecasts in its own normalized space; the forecast is
mapped back to the fine reading's space, interpolated to the fine time step, and blended with the
fine forecast by a learned gate, per series and per horizon step. A coarse reading is available only
when the series has history beyond the fine window; an unavailable reading gets weight exactly zero,
so a short series is forecast by the fine reading alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import HirondelConfig


def block_mean_view(raw: torch.Tensor, k: int, length: int) -> torch.Tensor:
    """raw [B, >= k * length] (NaN = missing) -> [B, length]: means over blocks of k ending at the last point,
    NaN where a block has no observed point."""
    seg = raw[:, -k * length:].float().reshape(raw.shape[0], length, k)
    finite = torch.isfinite(seg)
    count = finite.sum(-1)
    mean = torch.where(finite, seg, torch.zeros_like(seg)).sum(-1) / count.clamp_min(1)
    return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))


def spectral_bands(x: torch.Tensor, observed: torch.Tensor, n_bands: int) -> torch.Tensor:
    """Log share of spectral power (DC excluded) in n_bands log-spaced frequency bands. [B, L] -> [B, n_bands]."""
    x = x.float() * observed.float()
    power = torch.fft.rfft(x, dim=-1).abs().pow(2)[:, 1:]
    n = power.shape[1]
    f = torch.arange(1, n + 1, device=x.device, dtype=torch.float32)
    idx = (n_bands * torch.log(f) / math.log(n + 1)).floor().clamp(0, n_bands - 1).long()
    bands = torch.zeros(x.shape[0], n_bands, device=x.device).scatter_add_(1, idx[None].expand(x.shape[0], -1), power)
    return torch.log(bands / bands.sum(1, keepdim=True).clamp_min(1e-12) + 1e-6)


def to_fine_space(q: torch.Tensor, mu_k, sigma_k, mu_f, sigma_f, limit: float = 20.0) -> torch.Tensor:
    """Normalized quantiles of a coarse reading -> normalized space of the fine reading (monotone maps)."""
    q = q.float().clamp(-limit, limit)
    raw = torch.sinh(q) * sigma_k[..., None].float() + mu_k[..., None].float()
    z = (raw - mu_f[..., None].float()) / sigma_f[..., None].float()
    return torch.asinh(z).clamp(-limit, limit)


def upsample(qk: torch.Tensor, k: int, horizon: int) -> torch.Tensor:
    """[B, J, Q] at coarse steps (block j covers fine steps j*k .. j*k + k - 1) -> [B, horizon, Q], linear between
    block centres, constant beyond the first and last centre."""
    J = qk.shape[1]
    t = torch.arange(horizon, device=qk.device, dtype=torch.float32)
    p = (t - (k - 1) / 2) / k
    lo = p.floor().clamp(0, J - 1).long()
    hi = (lo + 1).clamp(max=J - 1)
    a = (p - lo.float()).clamp(0, 1)[None, :, None]
    return qk[:, lo] * (1 - a) + qk[:, hi] * a


@dataclass
class CoarseReadings:
    """Inputs of the coarse readings, one list entry per coarse factor."""
    values: list            # normalized [B, L], 0 where unobserved
    observed: list          # [B, L] 0/1
    mu: list                # [B, 1] raw-scale statistics of each reading
    sigma: list
    available: torch.Tensor  # [B, n_coarse] bool
    fine_mu: torch.Tensor    # [B, 1]
    fine_sigma: torch.Tensor


class FusionGate(nn.Module):
    """Per-series, per-step mixing weights between the fine reading and the coarse readings.

    Inputs per reading: its last context state, its log-band power spectrum and its observed fraction.
    The logits are expanded over smooth Gaussian bumps in log(1 + step); an unavailable reading gets
    weight exactly zero.
    """

    def __init__(self, cfg: HirondelConfig):
        super().__init__()
        f = cfg.fusion
        self.n_readings = 1 + len(cfg.coarse_factors)
        self.basis_size, self.n_bands = f.basis_size, f.spectral_bands
        self.state_proj = nn.ModuleList([nn.Linear(cfg.d_model, f.state_dim) for _ in range(self.n_readings)])
        self.mlp = nn.Sequential(nn.Linear(self.n_readings * (f.state_dim + f.spectral_bands + 1), f.hidden),
                                 nn.GELU(), nn.Linear(f.hidden, self.n_readings * f.basis_size))
        self.bias = nn.Parameter(torch.zeros(self.n_readings))
        centers = torch.linspace(0.0, math.log1p(f.max_horizon), f.basis_size)
        self.register_buffer("centers", centers, persistent=False)
        self.width = float(centers[1] - centers[0]) if f.basis_size > 1 else 1.0

    def forward(self, states, spectra, coverage, available: torch.Tensor, horizon: int) -> torch.Tensor:
        """Lists over readings (fine first) -> weights [B, horizon, n_readings]."""
        feats = ([p(s.float()) for p, s in zip(self.state_proj, states)] + [s.float() for s in spectra]
                 + [c.float()[:, None] for c in coverage])
        z = self.mlp(torch.cat(feats, 1)).reshape(-1, self.n_readings, self.basis_size)
        steps = torch.log1p(torch.arange(1, horizon + 1, device=z.device, dtype=torch.float32))
        phi = torch.exp(-0.5 * ((steps[:, None] - self.centers[None]) / self.width) ** 2)
        logits = torch.einsum("bnm,hm->bhn", z, phi) + self.bias
        logits = logits.masked_fill(~available[:, None, :], -1e4)
        return torch.softmax(logits, -1)
