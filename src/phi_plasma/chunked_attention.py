"""Eval-time chunked attention — process queries in chunks to fit MPS memory at 16K+.

The standard attention forward materializes a (B, H, T, T) score matrix.
At T=16384 H=11 B=1, that's 11*16384²*4 = 11.5 GB just for scores per layer.
MPS allocator OOMs at this size.

This module monkey-patches the attention forward of both HeckeEigensheafAttention
and StandardSelfAttention to chunk along the query dimension. Per chunk we materialize
only (B, H, chunk, T) scores — 16-64× less memory.

Usage:
    from phi_plasma.chunked_attention import patch_for_long_context
    patch_for_long_context(model, query_chunk=512)
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _chunked_softmax_attention(q, k, v, causal: bool, query_chunk: int):
    """q,k,v: (B, H, T, D). Returns (B, H, T, D) using chunked softmax+matmul.

    Memory: O(B*H*chunk*T*4 bytes) instead of O(B*H*T*T*4).
    """
    B, H, T, D = q.shape
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    for cs in range(0, T, query_chunk):
        ce = min(cs + query_chunk, T)
        q_c = q[:, :, cs:ce, :]                          # (B, H, c, D)
        s = q_c @ k.transpose(-2, -1) * scale            # (B, H, c, T)
        if causal:
            # Each query position i (relative cs..ce) can attend to keys 0..i (absolute).
            # Build a (c, T) causal mask: position cs+i can see j ≤ cs+i, i.e. j < cs+i+1.
            row_ix = torch.arange(cs, ce, device=q.device).unsqueeze(1)   # (c, 1)
            col_ix = torch.arange(T, device=q.device).unsqueeze(0)        # (1, T)
            mask = col_ix > row_ix                                         # (c, T) — True where masked
            s = s.masked_fill(mask, float("-inf"))
        a = F.softmax(s, dim=-1)
        out[:, :, cs:ce, :] = a @ v
    return out


def patch_for_long_context(model, query_chunk: int = 512):
    """Replace attention forwards across the model with chunked versions.

    Works for both HeckeEigensheafAttention and StandardSelfAttention (vanilla).
    Idempotent — safe to call multiple times."""
    # Late imports to avoid circular dependency.
    from .hecke_attention import HeckeEigensheafAttention
    from .vanilla_baseline import StandardSelfAttention

    def hecke_forward_chunked(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        Dt = x.dtype

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        v_routed = _chunked_softmax_attention(q, k, v, causal=causal, query_chunk=query_chunk)

        word_weights = self.word_policy(x)
        soft_hecke = torch.einsum("btn,nhk->bthk", word_weights, self.hecke_generators.to(Dt))

        v_perm = v_routed.permute(0, 2, 1, 3).contiguous()
        v_hecke = torch.einsum("bthj,btjd->bthd", soft_hecke, v_perm)
        v_out = v_perm + self.alpha * v_hecke
        out = v_out.reshape(B, T, D)
        return self.out_proj(out)

    def std_forward_chunked(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = _chunked_softmax_attention(q, k, v, causal=True, query_chunk=query_chunk)
        return self.o(a.transpose(1, 2).reshape(B, T, D))

    n_patched = 0
    for mod in model.modules():
        if isinstance(mod, HeckeEigensheafAttention):
            mod.forward = hecke_forward_chunked.__get__(mod, HeckeEigensheafAttention)
            n_patched += 1
        elif isinstance(mod, StandardSelfAttention):
            mod.forward = std_forward_chunked.__get__(mod, StandardSelfAttention)
            n_patched += 1
    return n_patched
