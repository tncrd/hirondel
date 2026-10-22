"""The network: patch embedding, a looped stack of delta and attention blocks, a per-position quantile head.

The stack is [first block] -> [middle blocks] x loop_passes -> [last block]. The middle blocks are
shared between passes. Every pass after the first re-reads an anchored copy of the input embedding
and adds only the update the middle blocks made, scaled by a learned gate.

All inputs and outputs of `NomModel.forward` are in normalized units; scaling to and from the raw
series is done by the forecaster.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NomConfig
from .fusion import CoarseReadings, FusionGate, spectral_bands, to_fine_space, upsample
from .layers import AttentionBlock, DeltaBlock, PatchEmbedding, QuantileHead


class LoopState(nn.Module):
    def __init__(self, cfg: NomConfig):
        super().__init__()
        d = cfg.d_model
        self.pass_embedding = nn.Parameter(torch.zeros(cfg.loop_passes, d))
        self.reinject_scale = nn.Parameter(torch.ones(d))
        self.gate = nn.Parameter(torch.zeros(cfg.loop_passes - 1))
        self.anchor_down = nn.Linear(d, cfg.anchor_rank)
        self.anchor_up = nn.Linear(cfg.anchor_rank, d)


class NomModel(nn.Module):
    def __init__(self, cfg: NomConfig):
        super().__init__()
        self.cfg = cfg
        kinds = {"delta": DeltaBlock, "attention": AttentionBlock}
        self.embed = PatchEmbedding(cfg)
        self.blocks = nn.ModuleList([kinds[t](cfg) for t in cfg.layer_types])
        self.loop = LoopState(cfg)
        self.head = QuantileHead(cfg)
        # Coarse readings (multi-resolution fusion): a learned flag added to every token of a coarse
        # reading, and the gate that blends the readings.
        self.coarse_view_embedding = nn.Parameter(torch.zeros(len(cfg.coarse_factors), cfg.d_model))
        self.fusion = FusionGate(cfg)

    def _trunk(self, h: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        first, middle, last = self.blocks[0], self.blocks[1:-1], self.blocks[-1]
        loop = self.loop
        e0 = h
        h = first(h, observed)
        x = h + loop.pass_embedding[0]
        for blk in middle:
            x = blk(x, observed)
        h = x
        if self.cfg.loop_passes > 1:
            anchor = e0 + loop.anchor_up(F.gelu(loop.anchor_down(e0)))
            for k in range(1, self.cfg.loop_passes):
                x_in = h + loop.reinject_scale * anchor + loop.pass_embedding[k]
                x = x_in
                for blk in middle:
                    x = blk(x, observed)
                h = h + torch.tanh(loop.gate[k - 1]) * (x - x_in)
        return last(h, observed)

    def _decode(self, x: torch.Tensor, observed: torch.Tensor, horizon: int,
                view_embedding: torch.Tensor | None = None):
        """One reading -> (normalized quantiles [B, horizon, Q], trunk state of the last context token [B, d])."""
        B, L = x.shape
        p = self.cfg.patch_size
        if L % p:
            raise ValueError(f"context length {L} is not a multiple of the patch size {p}")
        n_past, n_future = L // p, -(-horizon // p)
        future = x.new_zeros(B, n_future, p)
        values = torch.cat([x.reshape(B, n_past, p), future], 1)
        mask = torch.cat([observed.reshape(B, n_past, p), torch.zeros_like(future)], 1)
        h = self.embed(values, mask, n_past)
        if view_embedding is not None:
            h = h + view_embedding
        h = self._trunk(h, mask.amax(-1))
        q = self.head(h)
        return q[:, n_past * p:n_past * p + horizon], h[:, n_past - 1]

    def _fuse(self, q_fine, h_fine, x, observed, coarse: CoarseReadings, horizon: int) -> torch.Tensor:
        bands = self.cfg.fusion.spectral_bands
        q_fine = q_fine.float()
        states, spectra, coverage, deltas = [h_fine], [spectral_bands(x, observed, bands)], [observed.float().mean(1)], []
        for i, k in enumerate(self.cfg.coarse_factors):
            q_k, h_k = self._decode(coarse.values[i], coarse.observed[i], -(-horizon // k), self.coarse_view_embedding[i])
            q_k = upsample(to_fine_space(q_k, coarse.mu[i], coarse.sigma[i], coarse.fine_mu, coarse.fine_sigma), k, horizon)
            deltas.append(torch.nan_to_num(q_k - q_fine))
            states.append(h_k)
            spectra.append(spectral_bands(coarse.values[i], coarse.observed[i], bands))
            coverage.append(coarse.observed[i].float().mean(1))
        available = torch.cat([torch.ones_like(coarse.available[:, :1]), coarse.available], 1).bool()
        w = self.fusion(states, spectra, coverage, available, horizon)
        q = q_fine
        for i, d in enumerate(deltas):
            q = q + w[..., 1 + i, None] * d
        return q

    def forward(self, x: torch.Tensor, observed: torch.Tensor, horizon: int,
                coarse: CoarseReadings | None = None) -> torch.Tensor:
        """x, observed: [B, L] normalized context (0 where unobserved) and its 0/1 mask, L a multiple of the
        patch size. coarse: the coarse readings, or None for the fine reading alone. Returns normalized
        quantiles [B, horizon, n_quantiles]."""
        q, h_last = self._decode(x, observed, horizon)
        return q if coarse is None else self._fuse(q, h_last, x, observed, coarse, horizon)
