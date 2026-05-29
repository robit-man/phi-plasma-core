"""KoopmanProbe — EDMD with polynomial lift (ported from PHI_AEON L7).

Background
----------
Williams-Kevrekidis-Rowley (2015) prove that as the lift basis grows, the
linear approximation `K · φ(x) ≈ φ(F(x))` converges to the true Koopman
operator on the lifted space. We add the simplest informative lift:
per-coordinate squared monomials, with a learnable projection picking which
latent dims to square.

Lifted dim = latent + ⌈latent/φ⌉. K acts on the full lifted space.

Spectral diagnostics: we compute the actual eigenvalue spectrum of K via
`torch.linalg.eig` (CPU round-trip on MPS). Spectral gap = |λ_1| − |λ_2|.
Lyapunov estimate = log|λ_1|. Used both as diagnostic outputs and as
auxiliary training signals.

This is one of the two genuinely rigorous probes hiding inside the
retired PHI_AEON architecture. The implementation is verbatim except
for the import surface (plasma's `constants` instead of PHI_AEON's
`phi_constants`)."""

from __future__ import annotations
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .constants import PHI, PHI_INV


class KoopmanProbe(nn.Module):
    """EDMD lift + linear Koopman operator + spectral diagnostics.

    Args:
        d_model:    backbone hidden dimension
        latent_dim: dimension of the encoder's latent z
    """

    def __init__(self, d_model: int, latent_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        # EDMD polynomial lift dimension: ⌈latent/φ⌉ extra squared features
        self.lift_dim = max(1, int(round(latent_dim / PHI)))
        self.full_dim = latent_dim + self.lift_dim

        self.encoder = nn.Sequential(
            nn.Linear(d_model, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        # Learnable projection picking which latent dims to square
        self.lift_proj = nn.Linear(latent_dim, self.lift_dim, bias=False)
        # Koopman operator on the lifted state
        self.K = nn.Parameter(
            torch.eye(self.full_dim) * PHI_INV
            + torch.randn(self.full_dim, self.full_dim) * 0.01
        )
        self.decoder = nn.Linear(self.full_dim, d_model)

    def _edmd_lift(self, z: torch.Tensor) -> torch.Tensor:
        """Augment z with squared monomials: φ(z) = [z, (W z)²]."""
        squared = self.lift_proj(z).pow(2)
        return torch.cat([z, squared], dim=-1)

    def _spectral_diagnostics(self, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact eigenvalue spectral gap + Lyapunov estimate.

        Notes from the PHI_AEON v4.1 implementation:
          - linalg.eig is exact, not iterative
          - MPS lacks linalg.eig — CPU round-trip; cost ~10μs for 32x32
          - These outputs are DIAGNOSTICS (no_grad). K is trained via the
            residual loss, not via differentiation through eig
          - NaN guard against transient K corruption during training"""
        device = K.device
        needs_cpu = device.type == "mps"
        with torch.no_grad():
            K_for_eig = K.detach().cpu() if needs_cpu else K.detach()
            if not torch.isfinite(K_for_eig).all():
                # Safe defaults so training survives a transient numerical event
                spectral_gap = torch.tensor(1.0, device=device, dtype=K.dtype)
                lyapunov = torch.tensor(0.0, device=device, dtype=K.dtype)
                return spectral_gap, lyapunov
            eigvals = torch.linalg.eig(K_for_eig).eigenvalues
            mags = eigvals.abs()
            top_two = mags.topk(2).values
            top_mag = top_two[0].item()
            second_mag = top_two[1].item()
        spectral_gap = torch.tensor(top_mag - second_mag, device=device, dtype=K.dtype)
        lyapunov = torch.tensor(math.log(max(top_mag, 1e-8)), device=device, dtype=K.dtype)
        return spectral_gap, lyapunov

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """h: (B, T, D) → dict of probe outputs.

        Returns:
            latent:           (B, T, latent_dim) — encoder output
            lifted:           (B, T, full_dim)   — EDMD-lifted state
            koopman_residual: scalar             — ‖K·φ(z_t) − φ(z_{t+1})‖²
            spectral_gap:     scalar             — |λ_1| − |λ_2| of K
            lyapunov:         scalar             — log|λ_1| of K
            decoded:          (B, T, D)          — reconstruction
        """
        z = self.encoder(h)
        z_full = self._edmd_lift(z)
        z_pred = z_full[:, :-1] @ self.K.T
        z_true = z_full[:, 1:]
        koopman_residual = (z_pred - z_true).pow(2).mean()
        spectral_gap, lyapunov = self._spectral_diagnostics(self.K)
        return {
            "latent": z,
            "lifted": z_full,
            "koopman_residual": koopman_residual,
            "spectral_gap": spectral_gap,
            "lyapunov": lyapunov,
            "decoded": self.decoder(z_full),
        }
