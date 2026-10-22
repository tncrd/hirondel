"""Building blocks of the network: patch embedding, the two token mixers, and the quantile head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NomConfig
from .delta_rule import gated_delta_rule


class RMSNorm(nn.Module):
    """Root-mean-square normalization, always computed in float32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PatchEmbedding(nn.Module):
    """One token per patch: values, observed mask and a signed time index relative to the forecast origin."""

    def __init__(self, cfg: NomConfig):
        super().__init__()
        d, p = cfg.d_model, cfg.patch_size
        self.value_proj = nn.Linear(p, d)
        self.residual = SwiGLU(d, cfg.embed_hidden)
        self.mask_proj = nn.Linear(p, d)
        self.time_proj = nn.Linear(1, d)
        self.time_scale = cfg.time_scale

    def forward(self, values: torch.Tensor, observed: torch.Tensor, n_past: int) -> torch.Tensor:
        """values, observed: [B, N, patch]; the first n_past tokens are context, the rest are future."""
        h = self.value_proj(values)
        h = h + self.residual(h)
        h = h + self.mask_proj(observed)
        t = torch.arange(values.shape[1], device=values.device, dtype=h.dtype) - n_past
        return h + self.time_proj(torch.asinh(t / self.time_scale)[None, :, None])


class DeltaBlock(nn.Module):
    """Pre-norm residual block whose token mixer is a gated delta rule (linear recurrent memory).

    Tokens marked unobserved (padding, missing patches, future placeholders) are read but never
    written: their write strength is zero and their decay is one.
    """

    def __init__(self, cfg: NomConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.n_heads, self.head_dim = h, d // h
        self.conv_kernel, self.write_max, self.chunk = cfg.conv_kernel, cfg.delta_write_max, cfg.delta_chunk
        self.mix_norm = RMSNorm(d, cfg.norm_eps)
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.q_conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d)
        self.k_conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d)
        self.v_conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d)
        self.decay_proj = nn.Linear(d, h)
        self.decay_rate_log = nn.Parameter(torch.zeros(h))
        self.write_proj = nn.Linear(d, h)
        self.out_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.gate_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.ffn_norm = RMSNorm(d, cfg.norm_eps)
        self.ffn = SwiGLU(d, cfg.ffn_hidden)

    def _causal_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        x = F.pad(x.transpose(1, 2), (self.conv_kernel - 1, 0))
        return conv(x).transpose(1, 2)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def _mix(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        q = self._heads(F.silu(self._causal_conv(self.q_proj(x), self.q_conv)))
        k = self._heads(F.silu(self._causal_conv(self.k_proj(x), self.k_conv)))
        v = self._heads(F.silu(self._causal_conv(self.v_proj(x), self.v_conv)))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        alpha = torch.exp(-F.softplus(self.decay_proj(x)) * torch.exp(self.decay_rate_log))
        beta = self.write_max * torch.sigmoid(self.write_proj(x))
        m = observed[..., None]
        beta = beta * m
        alpha = alpha * m + (1.0 - m)
        o = gated_delta_rule(q.float(), k.float(), v.float(), alpha.transpose(1, 2).float(),
                             beta.transpose(1, 2).float(), chunk=self.chunk).to(x.dtype)
        o = self.out_norm(o).transpose(1, 2).reshape(B, N, d)
        return self.out_proj(o * F.silu(self.gate_proj(x)))

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        x = x + self._mix(self.mix_norm(x), observed)
        return x + self.ffn(self.ffn_norm(x))


def rotary(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotate-half rotary embedding of x [B, heads, N, head_dim] at absolute positions [N]."""
    angle = positions[:, None].float() * inv_freq[None, :]
    cos = torch.cat([angle.cos(), angle.cos()], dim=-1)[None, None]
    sin = torch.cat([angle.sin(), angle.sin()], dim=-1)[None, None]
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


class AttentionBlock(nn.Module):
    """Pre-norm residual block with multi-head self-attention, rotary positions and a per-head output gate.

    The gate is 2 * sigmoid(.), so each head can scale its output anywhere in (0, 2), including near
    zero when there is nothing worth retrieving. Positions are measured in samples (token index x patch).
    """

    def __init__(self, cfg: NomConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.n_heads, self.head_dim, self.stride = h, d // h, cfg.patch_size
        self.mix_norm = RMSNorm(d, cfg.norm_eps)
        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.head_gate = nn.Linear(d, h)
        self.ffn_norm = RMSNorm(d, cfg.norm_eps)
        self.ffn = SwiGLU(d, cfg.ffn_hidden)
        inv_freq = cfg.rope_base ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        pos = torch.arange(N, device=x.device) * self.stride
        q, k = rotary(q, pos, self.inv_freq), rotary(k, pos, self.inv_freq)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o * (2 * torch.sigmoid(self.head_gate(x))).transpose(1, 2)[..., None].to(o.dtype)
        return self.out_proj(o.transpose(1, 2).reshape(B, N, d))

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        x = x + self._mix(self.mix_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class GatedResidual(nn.Module):
    """skip(h) + out(sigmoid(hidden(h)))."""

    def __init__(self, dim: int):
        super().__init__()
        self.skip = nn.Linear(dim, dim)
        self.hidden = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.skip(h) + self.out(torch.sigmoid(self.hidden(h)))


class QuantileHead(nn.Module):
    """Decodes every token into patch_size x n_quantiles values that never cross.

    One channel is the median; the others are softplus increments accumulated outward from it.
    """

    def __init__(self, cfg: NomConfig):
        super().__init__()
        self.patch, self.n_q, self.median = cfg.patch_size, len(cfg.quantile_levels), cfg.median_index
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.residual = GatedResidual(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.patch_size * self.n_q)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, T, d] -> quantiles [B, T * patch, n_quantiles]."""
        B, T, _ = h.shape
        raw = self.proj(self.residual(self.norm(h))).reshape(B, T * self.patch, self.n_q)
        step = F.softplus(raw)
        out = torch.empty_like(raw)
        mid = raw[..., self.median]
        out[..., self.median] = mid
        acc = mid
        for i in range(self.median + 1, self.n_q):
            acc = acc + step[..., i]
            out[..., i] = acc
        acc = mid
        for i in range(self.median - 1, -1, -1):
            acc = acc - step[..., i]
            out[..., i] = acc
        return out
