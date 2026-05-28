"""Loss components for MVP training:

  L_total = L_NLL + λ_eig(t) · L_eigensheaf

NLL is standard cross-entropy; the eigensheaf penalty is computed on
the model's weights (cheap — no forward pass needed).

λ_eig(t) ramps linearly from 0 to λ_eig_max over `warmup_steps`. Holds the
geometric constraint out of early training so NLL can take off."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nll_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V), targets (B, T) int64 → scalar mean NLL."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))


def eigensheaf_weight(step: int, warmup: int, lam_max: float) -> float:
    if step >= warmup:
        return lam_max
    return lam_max * (step / max(1, warmup))


def combined_loss(model, logits: torch.Tensor, targets: torch.Tensor,
                  step: int, warmup: int = 2000, lam_max: float = 1e-3,
                  ) -> dict[str, torch.Tensor]:
    """Returns dict with total, nll, eigensheaf (separate scalars)."""
    nll = nll_loss(logits, targets)
    eig = model.eigensheaf_penalty() if hasattr(model, "eigensheaf_penalty") else nll.new_zeros(())
    lam = eigensheaf_weight(step, warmup, lam_max)
    total = nll + lam * eig
    return {"total": total, "nll": nll, "eigensheaf": eig, "lambda": lam}
