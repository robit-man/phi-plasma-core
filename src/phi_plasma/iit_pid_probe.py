"""IITPhiProbe — Williams-Beer partial information decomposition (ported from
PHI_AEON L12).

Background
----------
Williams & Beer (2010) split mutual information I(X1, X2; Y) into:
    Unique(X1; Y), Unique(X2; Y)  — info each source contributes alone
    Redundant(X1, X2; Y)           — info both sources share
    Synergy(X1, X2; Y)             — info available only from both together

Synergy is the "integrated information" component that Tononi's IIT-Φ aims
to capture. For multivariate Gaussian distributions the synergy admits a
closed-form estimator (Bertschinger et al. MMI):

    Syn(X1, X2; Y) = I(X1, X2; Y) − max(I(X1; Y), I(X2; Y))

with the Gaussian MI formula

    I(A; B) = ½ · log(|Σ_A| · |Σ_B| / |Σ_{A,B}|)

We partition the model's hidden state into n_partitions chunks, fit a
Gaussian to each + the joint, compute logdet via Cholesky for stability,
and return the synergy as a differentiable scalar.

The information-theoretic content is rigorous. The connection to
consciousness theory (IIT) is philosophical and not load-bearing on the
math — drop the IIT framing if you don't want the baggage.

Ported verbatim from PHI_AEON v4.1 except for the import surface."""

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class IITPhiProbe(nn.Module):
    """Williams-Beer PID synergy estimator via Gaussian MMI.

    Args:
        d_model:        hidden dimension
        n_partitions:   number of feature-dim chunks treated as separate sources
    """

    def __init__(self, d_model: int, n_partitions: int = 4):
        super().__init__()
        assert d_model % n_partitions == 0, \
            f"d_model={d_model} must divide n_partitions={n_partitions}"
        self.d_model = d_model
        self.n_partitions = n_partitions
        self.d_part = d_model // n_partitions
        # Passive self-readout (diagnostic only; not used by any loss)
        self.self_encoder = nn.Linear(d_model, d_model // 4)

    @staticmethod
    def _gauss_entropy_logdet(x: torch.Tensor) -> torch.Tensor:
        """Gaussian differential entropy via logdet of the empirical covariance.

        Drops the constant (2πe)^d/2 offset (immaterial for MI differences).
        Uses slogdet via Cholesky for numerical stability on small matrices."""
        centered = x - x.mean(dim=0, keepdim=True)
        n = max(centered.size(0) - 1, 1)
        cov = centered.T @ centered / n
        # Ridge for stability
        cov = cov + 1e-4 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        _, logabsdet = torch.linalg.slogdet(cov)
        return logabsdet

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """h: (B, T, D) → dict of probe outputs.

        Returns:
            phi_surrogate:     PID synergy estimate (differentiable scalar, ≥ 0)
            phi_surrogate_v3:  legacy variance-based proxy
            total_correlation: I(parts; whole) — multi-information
            self_signal:       passive readout (diagnostic only)
        """
        B, T, D = h.shape
        flat = h.reshape(B * T, D)
        parts = torch.chunk(flat, self.n_partitions, dim=-1)

        H_joint = self._gauss_entropy_logdet(flat)
        H_marginals = [self._gauss_entropy_logdet(p) for p in parts]
        total_correlation = sum(H_marginals) - H_joint

        # MPS-safe reduction: amax() vs max() — max() uses evenly_distribute_backward
        # which has historically deadlocked on MPS at scale
        max_marginal = torch.stack(H_marginals).amax()
        phi_pid = (total_correlation - 0.5 * max_marginal).clamp(min=0.0) * 1e-2

        # Variance surrogate for back-compat with v3 code paths
        full_var = h.var(dim=-1).mean()
        part_var_sum = sum(p.var(dim=-1).mean() for p in parts)
        phi_v3 = full_var - part_var_sum / self.n_partitions

        self_signal = self.self_encoder(h.mean(dim=1)).mean()

        return {
            "phi_surrogate": phi_pid,
            "phi_surrogate_v3": phi_v3,
            "total_correlation": total_correlation,
            "self_signal": self_signal,
        }
