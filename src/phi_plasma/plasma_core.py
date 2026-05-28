"""Composed model: embedding → N × HamiltonianBlock → tied output head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import (
    VOCAB_SIZE, D_HIDDEN, N_LAYERS, N_HEADS, HEAD_DIM, SEQ_LEN, PHI_INV,
)
from .hamiltonian_block import HamiltonianBlock


class PlasmaCore(nn.Module):
    """The MVP architecture."""

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE, d_hidden: int = D_HIDDEN,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 head_dim: int = HEAD_DIM, seq_len: int = SEQ_LEN,
                 h_step: float = 0.5, v_hidden_mult: int = 2,
                 tie_embeddings: bool = True):
        super().__init__()
        assert d_hidden % 2 == 0
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden
        self.d_half = d_hidden // 2
        self.seq_len = seq_len
        self.tie_embeddings = tie_embeddings

        self.token_embed = nn.Embedding(vocab_size, d_hidden)
        self.pos_embed = nn.Embedding(seq_len, d_hidden)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.01)
        # Damp the p-part of the embedding so we start near zero momentum.
        with torch.no_grad():
            self.token_embed.weight[:, self.d_half:].mul_(0.1)

        self.blocks = nn.ModuleList([
            HamiltonianBlock(d_hidden, n_heads, head_dim, h_step, v_hidden_mult)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_hidden)

        if not tie_embeddings:
            self.head = nn.Linear(d_hidden, vocab_size, bias=False)
            nn.init.normal_(self.head.weight, std=0.02 * PHI_INV)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) int64 → logits (B, T, vocab)."""
        B, T = idx.shape
        assert T <= self.seq_len, f"context {T} > seq_len {self.seq_len}"

        pos = torch.arange(T, device=idx.device).expand(B, T)
        x = self.token_embed(idx) + self.pos_embed(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm_out(x)
        if self.tie_embeddings:
            return x @ self.token_embed.weight.T
        return self.head(x)

    # ── Diagnostics ────────────────────────────────────────────────
    def eigensheaf_penalty(self) -> torch.Tensor:
        return torch.stack([b.eigensheaf_penalty() for b in self.blocks]).mean()

    def head_isotypic_distances(self) -> list[float]:
        return [b.head_isotypic_distance() for b in self.blocks]

    def energy_per_layer(self, idx: torch.Tensor) -> list[float]:
        """H(q, p, a) at each layer (mean over batch/tokens). Diagnostic only."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device).expand(B, T)
            x = self.token_embed(idx) + self.pos_embed(pos)
            energies = []
            for blk in self.blocks:
                H = blk.hamiltonian(x).mean().item()
                energies.append(H)
                x = blk(x)
        if was_training:
            self.train()
        return energies

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
