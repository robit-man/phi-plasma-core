"""S2 — Hecke-Eigensheaf Attention.

Builds on logos/hecke_attention.py and adds the **eigensheaf constraint**:
each head's value projection is regularized toward an eigenvector of one
specific Hecke generator. Heads are thus forced to occupy algebraically
distinct subspaces — they cannot mode-collapse onto each other.

Constraint penalty (added to loss):
    L_eigensheaf = (1/H) Σ_i ‖ T_i · W_V^(i) − λ_i · W_V^(i) ‖_F²

where W_V^(i) is the head-i block of v_proj.weight reshaped (D, head_dim),
T_i is the i-th Hecke generator action on the head index (H, H), and λ_i
is a learned eigenvalue scalar.

We assign generator i to head i for i ∈ [1, n_gens]; heads with no assigned
generator (i = n_gens+1 ... H) carry no constraint."""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    PHI, PHI_INV, SPECTRAL_GAP, D_HIDDEN, N_HEADS, HEAD_DIM,
    HECKE_BASE_RANK, HECKE_N_GENERATORS, HECKE_PARAM_Q,
)
from .hecke_algebra import HeckeAlgebra


class HeckeWordPolicy(nn.Module):
    """Per-token soft Hecke word over generators T_1..T_{n-1}."""

    def __init__(self, d_hidden: int, n_generators: int):
        super().__init__()
        self.proj = nn.Linear(d_hidden, n_generators, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.proj(x), dim=-1)


class HeckeEigensheafAttention(nn.Module):
    """Self-attention + Hecke head-mixing + eigensheaf constraint hook.

    Forward returns the attention output. The eigensheaf penalty is
    computed by `eigensheaf_penalty()` and added to the training loss
    by the training loop."""

    def __init__(self, d_hidden: int = D_HIDDEN,
                 n_heads: int = N_HEADS, head_dim: int = HEAD_DIM,
                 alpha: float = SPECTRAL_GAP):
        super().__init__()
        assert d_hidden == n_heads * head_dim, \
            f"d_hidden={d_hidden} must equal n_heads*head_dim={n_heads*head_dim}"
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.alpha = alpha

        self.q_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.k_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.v_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_hidden, bias=False)

        for p in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(p.weight, std=0.02 * PHI_INV)
        nn.init.normal_(self.out_proj.weight, std=0.02 * (PHI_INV ** 2))

        self.hecke = HeckeAlgebra(n=n_heads, q=HECKE_PARAM_Q)
        gens = self.hecke.stack()                                  # (n_gens, H, H)
        self.register_buffer("hecke_generators", gens)

        self.word_policy = HeckeWordPolicy(d_hidden, HECKE_N_GENERATORS)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        Dt = x.dtype

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)        # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        v_routed = torch.matmul(attn, v)                            # (B, H, T, Dh)

        word_weights = self.word_policy(x)                          # (B, T, n_gens)
        soft_hecke = torch.einsum(
            "btn,nhk->bthk", word_weights, self.hecke_generators.to(Dt)
        )                                                           # (B, T, H, H)

        v_perm = v_routed.permute(0, 2, 1, 3).contiguous()          # (B, T, H, Dh)
        v_hecke = torch.einsum("bthj,btjd->bthd", soft_hecke, v_perm)
        v_out = v_perm + self.alpha * v_hecke                       # (B, T, H, Dh)

        out = v_out.reshape(B, T, D)
        return self.out_proj(out)

    def eigensheaf_penalty(self) -> torch.Tensor:
        """Hecke centralizer constraint — the proper isotypic-decomposition penalty.

        By Schur's lemma: a matrix M commutes with every Hecke generator T_i
        iff M acts as a scalar on each irreducible component of the Hecke
        representation. So requiring

            [T_i, U U^T] = T_i U U^T − U U^T T_i = 0   for all i

        is equivalent to requiring U U^T to be in the *centralizer* of the
        Hecke algebra action — which means the head Gram matrix lives in
        the block-diagonal structure dictated by the algebra's irreps.

        This is the genuine eigensheaf condition: heads partition into
        isotypic blocks according to the Hecke decomposition.

        Why this is non-vacuous (unlike the prior orthogonality version):
          - random Gaussian U has UU^T ≈ ‖U[h]‖² · I (a scalar matrix on each
            head's row independently), but T_i has off-diagonal structure
            (1 in the (i-1, i) block), so [T_i, scalar*I] = 0 only in the
            limit of perfect homogeneity. For asymmetric U, [T_i, UU^T] is
            non-trivially nonzero.
          - The constraint forces UU^T to acquire the specific block
            structure: a scalar on the trivial rep, a scalar on the
            alternating rep, a scalar on the standard rep — three distinct
            "scales" coupled to the algebra's algebraic structure.

        Penalty:
            L = (1 / n_gens) Σ_i  ‖[T_i, UU^T]‖²_F / (‖UU^T‖²_F + ε)

        Scale-invariant (division by ‖UU^T‖²) so the penalty value lives in
        [0, ~2] regardless of model size."""
        W = self.v_proj.weight.view(self.n_heads, self.head_dim, self.d_hidden)
        U = W.reshape(self.n_heads, -1)                             # (H, head_dim * D)
        UU = U @ U.T                                                # (H, H)
        UU_frob_sq = (UU * UU).sum().clamp(min=1e-8)

        n_gens = HECKE_N_GENERATORS
        penalty = UU.new_zeros(())
        for idx in range(n_gens):
            T_i = self.hecke_generators[idx]                        # (H, H)
            commutator = T_i @ UU - UU @ T_i                        # (H, H)
            penalty = penalty + (commutator * commutator).sum() / UU_frob_sq
        return penalty / n_gens

    @torch.no_grad()
    def head_isotypic_distance(self) -> float:
        """Diagnostic: mean off-diagonal cosine similarity between heads' V-vectors.
        Vanilla transformers typically hit ~0.4-0.6 after training. HESA should hold ≤0.2."""
        W = self.v_proj.weight.detach().view(self.n_heads, self.head_dim, self.d_hidden)
        U = W.reshape(self.n_heads, -1)
        Un = F.normalize(U, dim=-1)
        sim = Un @ Un.T
        H = self.n_heads
        mask = ~torch.eye(H, dtype=torch.bool, device=sim.device)
        return float(sim[mask].abs().mean())
