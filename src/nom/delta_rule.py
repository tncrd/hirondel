"""Gated delta rule, evaluated chunk by chunk.

For each head, a matrix state S (d_v x d_k) is decayed and rewritten at every step t:

    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,      o_t = S_t q_t

(Yang, Kautz and Hatamizadeh, "Gated Delta Networks", arXiv:2412.06464). A step with beta = 0 and
alpha = 1 neither reads nor writes the state, which is how unobserved tokens are skipped.

Inside a chunk of C steps the recurrence is rewritten as dense matrix products (the WY / UT
representation of a product of Householder-like updates), so only one state hand-off per chunk is
sequential. The result equals the step-by-step recurrence up to floating-point rounding.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     alpha: torch.Tensor, beta: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """q, k: [B, H, N, dk]; v: [B, H, N, dv]; alpha, beta: [B, H, N]. Returns o: [B, H, N, dv]."""
    B, H, N, dk = q.shape
    dv = v.shape[-1]
    C = chunk
    pad = (-N) % C
    if pad:                                           # padded steps: alpha = 1, beta = 0 (inert)
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        alpha = F.pad(alpha, (0, pad), value=1.0)
        beta = F.pad(beta, (0, pad), value=0.0)
    n_chunks = (N + pad) // C
    q = q.reshape(B, H, n_chunks, C, dk)
    k = k.reshape(B, H, n_chunks, C, dk)
    v = v.reshape(B, H, n_chunks, C, dv)
    alpha = alpha.reshape(B, H, n_chunks, C)
    beta = beta.reshape(B, H, n_chunks, C)

    # Cumulative decay inside each chunk, in log space; every ratio used below is exp(<= 0).
    log_decay = torch.log(alpha.clamp_min(1e-38)).cumsum(dim=-1)
    decay = torch.exp(log_decay)                                  # gamma_r
    decay_end = torch.exp(log_decay[..., -1])                     # gamma_C
    decay_to_end = torch.exp(log_decay[..., -1:] - log_decay)     # gamma_C / gamma_r

    log_ratio = log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)
    causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), -1)
    ratio_causal = torch.exp(log_ratio.masked_fill(~causal, float("-inf")))
    ratio_strict = torch.exp(log_ratio.masked_fill(~strict, float("-inf")))

    kk = torch.matmul(k, k.transpose(-1, -2))
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    b = beta.unsqueeze(-1)
    # Keys pass through the plain delta transform; values also carry the decay between writes.
    w = torch.linalg.solve_triangular(b * (kk * strict) + eye, b * k, upper=False, unitriangular=True)
    u = torch.linalg.solve_triangular(b * (kk * ratio_strict) + eye, b * v, upper=False, unitriangular=True)
    scores = torch.matmul(q, k.transpose(-1, -2)) * ratio_causal

    state = q.new_zeros(B, H, dv, dk)
    outputs = []
    for c in range(n_chunks):
        g = decay[:, :, c].unsqueeze(-1)
        new_values = u[:, :, c] - g * torch.einsum("bhck,bhvk->bhcv", w[:, :, c], state)
        read = torch.einsum("bhck,bhvk->bhcv", q[:, :, c], state)
        outputs.append(g * read + torch.einsum("bhij,bhjv->bhiv", scores[:, :, c], new_values))
        k_end = decay_to_end[:, :, c].unsqueeze(-1) * k[:, :, c]
        state = (decay_end[:, :, c][..., None, None] * state
                 + torch.einsum("bhcv,bhck->bhvk", new_values, k_end))
    return torch.stack(outputs, dim=2).reshape(B, H, N + pad, dv)[:, :, :N]
