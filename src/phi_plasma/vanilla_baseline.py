"""Parameter-matched vanilla transformer (the control for ablations).

Same embedding sizes, same vocab, same number of layers as PlasmaCore — but
standard pre-LN transformer blocks instead of Hamiltonian flow. This is the
'is the geometric machinery doing anything' control."""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import VOCAB_SIZE, D_HIDDEN, N_LAYERS, N_HEADS, HEAD_DIM, SEQ_LEN, PHI_INV


class StandardSelfAttention(nn.Module):
    def __init__(self, d: int, h: int, dh: int):
        super().__init__()
        assert d == h * dh
        self.h, self.dh = h, dh
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02 * PHI_INV)
        nn.init.normal_(self.o.weight, std=0.02 * (PHI_INV ** 2))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        m = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        s = s.masked_fill(m, float("-inf"))
        a = F.softmax(s, dim=-1) @ v
        return self.o(a.transpose(1, 2).reshape(B, T, D))


class VanillaBlock(nn.Module):
    def __init__(self, d: int, h: int, dh: int, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = StandardSelfAttention(d, h, dh)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_mult, bias=False),
            nn.SiLU(),
            nn.Linear(d * ffn_mult, d, bias=False),
        )
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02 * PHI_INV)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class VanillaTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE, d_hidden: int = D_HIDDEN,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 head_dim: int = HEAD_DIM, seq_len: int = SEQ_LEN,
                 ffn_mult: int = 4, tie_embeddings: bool = True):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_hidden)
        self.pos_embed = nn.Embedding(seq_len, d_hidden)
        self.blocks = nn.ModuleList([VanillaBlock(d_hidden, n_heads, head_dim, ffn_mult)
                                     for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_hidden)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.01)
        self.tie = tie_embeddings
        self.seq_len = seq_len
        if not tie_embeddings:
            self.head = nn.Linear(d_hidden, vocab_size, bias=False)
            nn.init.normal_(self.head.weight, std=0.02 * PHI_INV)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).expand(B, T)
        x = self.token_embed(idx) + self.pos_embed(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        return x @ self.token_embed.weight.T if self.tie else self.head(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
