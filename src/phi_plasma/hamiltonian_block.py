"""S3 — Symplectic-Hamiltonian Token Flow.

Each token embedding is split into (q, p) ∈ ℝ^{d/2} × ℝ^{d/2}. The block
performs one Störmer-Verlet leapfrog step under a learned Hamiltonian

    H(q, p, a) = T(p) + V(q, a)
    T(p) = ½ p^T M^{-1} p           (diagonal mass M = softplus(·) + ε)
    V(q, a) = MLP([q, a])           (small scalar MLP)

The "action" a is the Hecke-Eigensheaf Attention output. The block is
volume-preserving on the (q, p) state when V depends only on q (exact
symplectic flow), modulo the learned-V approximation that V depends on
a (which is computed from x = (q, p) — strictly speaking this couples q
and p, but the action is held fixed during the leapfrog step so within
one step the dynamics remain symplectic on (q, p)).

Step (one block = one Störmer-Verlet leapfrog):
    a       = attn(LN(x))                       # action from current state
    p₁/₂   = p − (h/2) · ∂V/∂q            (q held fixed)
    q'      = q + h · M⁻¹ · p₁/₂
    p'      = p₁/₂ − (h/2) · ∂V/∂q'

Two autograd evaluations of ∂V/∂q per block — that's the cost of exact
symplecticity. Gradient flows through them via create_graph."""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import PHI_INV, SPECTRAL_GAP, D_HIDDEN, N_HEADS, HEAD_DIM
from .hecke_attention import HeckeEigensheafAttention


class HamiltonianBlock(nn.Module):
    """One Störmer-Verlet leapfrog step on the (q, p) split of a token embedding."""

    def __init__(self, d_hidden: int = D_HIDDEN,
                 n_heads: int = N_HEADS, head_dim: int = HEAD_DIM,
                 h_step: float = 0.5, v_hidden_mult: int = 2):
        super().__init__()
        assert d_hidden % 2 == 0, f"d_hidden must be even, got {d_hidden}"
        self.d_hidden = d_hidden
        self.d_half = d_hidden // 2
        self.h = h_step

        self.norm = nn.LayerNorm(d_hidden)
        self.attn = HeckeEigensheafAttention(d_hidden, n_heads, head_dim)

        # Diagonal mass parameter; softplus → positive. Init at softplus⁻¹(1) so M ≈ 1.
        log_one = math.log(math.expm1(1.0))
        self.mass_log = nn.Parameter(torch.full((self.d_half,), log_one))

        # Potential V(q, a): inputs = (q, a) = (d_half + d_hidden) → scalar.
        v_hid = d_hidden * v_hidden_mult
        self.V_net = nn.Sequential(
            nn.Linear(self.d_half + d_hidden, v_hid, bias=True),
            nn.SiLU(),
            nn.Linear(v_hid, 1, bias=False),
        )
        for layer in self.V_net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02 * (PHI_INV ** 2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    @property
    def mass(self) -> torch.Tensor:
        return F.softplus(self.mass_log) + 1e-6                  # (d_half,)

    def potential(self, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """V(q, a) summed over tokens. q: (B, T, d_half), a: (B, T, D) → scalar."""
        inp = torch.cat([q, a], dim=-1)
        return self.V_net(inp).squeeze(-1)                       # (B, T)

    def grad_V_q(self, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """∂V/∂q via autograd. Differentiable through optimizer steps.

        Force-enable autograd so the gradient computation works even when
        the caller wraps the forward in torch.no_grad() (e.g. eval/diagnostic)."""
        with torch.enable_grad():
            q_in = q.detach().requires_grad_(True)
            a_in = a.detach()
            V = self.potential(q_in, a_in).sum()
            (grad,) = torch.autograd.grad(V, q_in, create_graph=self.training)
        return grad                                              # (B, T, d_half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One Störmer-Verlet leapfrog step. x: (B, T, D) → (B, T, D)."""
        x_n = self.norm(x)
        a = self.attn(x_n, causal=True)                          # (B, T, D)

        q, p = x.split(self.d_half, dim=-1)                      # (B, T, d_half) ×2

        dV_dq = self.grad_V_q(q, a)
        p_half = p - 0.5 * self.h * dV_dq

        m_inv = 1.0 / self.mass
        q_new = q + self.h * m_inv * p_half

        dV_dq_new = self.grad_V_q(q_new, a)
        p_new = p_half - 0.5 * self.h * dV_dq_new

        return torch.cat([q_new, p_new], dim=-1)

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """H(q, p, a) for inspection. Returns (B, T) scalar field."""
        x_n = self.norm(x)
        a = self.attn(x_n, causal=True)
        q, p = x.split(self.d_half, dim=-1)
        T = 0.5 * (p * p / self.mass).sum(dim=-1)
        V = self.potential(q, a)
        return T + V

    def eigensheaf_penalty(self) -> torch.Tensor:
        return self.attn.eigensheaf_penalty()

    def head_isotypic_distance(self) -> float:
        return self.attn.head_isotypic_distance()
