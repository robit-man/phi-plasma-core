"""Φ-CONCENTRATE — plasma v0.2

Composes the 5 keepers from PRIME's phi-family audit (the 30% honest core)
into a single architecture:

  1. Hecke-Eigensheaf Attention      (already in plasma — from LOGOS)
  2. Symplectic-Hamiltonian Token Flow (already in plasma — the backbone)
  3. Sheaf Edge Consistency           (NEW — ported from world_model)
  4. Koopman EDMD Probe               (NEW — ported from PHI_AEON L7)
  5. IIT-PID Synergy Probe            (NEW — ported from PHI_AEON L12)

Plus side memory option:
  - P47 modular tape (TODO; included as a stub for now)

Plus eval infrastructure:
  - Chunked attention for 1K→64K context (already in plasma)
  - Symplectic AdamW (already in plasma)

The architecture is plasma's PlasmaCore as the backbone, with the three new
probes attached at the final hidden state. Each probe contributes:
  - Diagnostic outputs (for interpretability / monitoring)
  - Optional auxiliary loss terms (controllable via config)
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn

from .constants import VOCAB_SIZE, D_HIDDEN, N_LAYERS, N_HEADS, HEAD_DIM, SEQ_LEN
from .plasma_core import PlasmaCore
from .koopman_probe import KoopmanProbe
from .iit_pid_probe import IITPhiProbe
from .sheaf_consistency import SheafConsistencyHead


class ConcentrateModel(nn.Module):
    """Plasma backbone + 3 keeper probes attached to the final hidden state."""

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 d_hidden: int = D_HIDDEN,
                 n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS,
                 head_dim: int = HEAD_DIM,
                 seq_len: int = SEQ_LEN,
                 h_step: float = 0.5,
                 v_hidden_mult: int = 2,
                 tie_embeddings: bool = True,
                 # Probe controls
                 use_koopman: bool = True,
                 koopman_latent_dim: int = 32,
                 use_iit_pid: bool = True,
                 iit_n_partitions: int = 4,
                 use_sheaf: bool = True,
                 sheaf_n_edge_types: int = 8,
                 sheaf_max_skip: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden
        self.seq_len = seq_len

        # Plasma backbone (S2 + S3 + Symplectic AdamW substrate)
        self.backbone = PlasmaCore(
            vocab_size=vocab_size,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            h_step=h_step,
            v_hidden_mult=v_hidden_mult,
            tie_embeddings=tie_embeddings,
        )

        # Keeper probes (all attached at the final backbone hidden state,
        # post-norm — we hook in just before the LM head)
        self.use_koopman = use_koopman
        if use_koopman:
            self.koopman = KoopmanProbe(d_hidden, latent_dim=koopman_latent_dim)

        self.use_iit_pid = use_iit_pid
        if use_iit_pid:
            self.iit_pid = IITPhiProbe(d_hidden, n_partitions=iit_n_partitions)

        self.use_sheaf = use_sheaf
        if use_sheaf:
            self.sheaf = SheafConsistencyHead(
                seq_len=seq_len,
                stalk_dim=d_hidden,
                n_edge_types=sheaf_n_edge_types,
                max_skip=sheaf_max_skip,
            )

    def _final_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the backbone but stop at the final LayerNorm output (pre LM head).

        This duplicates PlasmaCore.forward up to the norm_out step. We do
        not have a clean hook in PlasmaCore for this, so we replicate the
        sequence here."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).expand(B, T)
        x = self.backbone.token_embed(idx) + self.backbone.pos_embed(pos)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm_out(x)
        return x

    def forward(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """idx: (B, T) int64 → dict with logits + probe outputs.

        Returns:
            logits:    (B, T, V)
            koopman:   dict from KoopmanProbe (if enabled)
            iit_pid:   dict from IITPhiProbe (if enabled)
            sheaf_loss: scalar (if enabled)
        """
        x = self._final_hidden(idx)
        if self.backbone.tie_embeddings:
            logits = x @ self.backbone.token_embed.weight.T
        else:
            logits = self.backbone.head(x)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if self.use_koopman:
            out["koopman"] = self.koopman(x)
        if self.use_iit_pid:
            out["iit_pid"] = self.iit_pid(x)
        if self.use_sheaf:
            out["sheaf_loss"] = self.sheaf(x)
        return out

    # ── Diagnostics passthrough ─────────────────────────────────────
    def eigensheaf_penalty(self) -> torch.Tensor:
        return self.backbone.eigensheaf_penalty()

    def head_isotypic_distances(self) -> list[float]:
        return self.backbone.head_isotypic_distances()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def combined_concentrate_loss(model, output, targets, step,
                                eig_warmup: int = 500, eig_max: float = 0.0,
                                koopman_weight: float = 1e-3,
                                iit_weight: float = 1e-4,
                                sheaf_weight: float = 1e-3,
                                koopman_warmup: int = 1000,
                                iit_warmup: int = 1000,
                                sheaf_warmup: int = 500) -> Dict[str, torch.Tensor]:
    """Composite loss combining NLL + the 3 keeper probes.

    Each probe's contribution is ramped linearly from 0 → weight over its
    warmup window. Set any weight to 0.0 to disable that probe's
    contribution while keeping the diagnostic outputs."""
    logits = output["logits"]
    B, T, V = logits.shape
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

    def ramp(s: int, warmup: int, target: float) -> float:
        if target == 0.0:
            return 0.0
        if s >= warmup:
            return target
        return target * (s / max(1, warmup))

    eig = model.eigensheaf_penalty() if hasattr(model, "eigensheaf_penalty") else nll.new_zeros(())
    eig_lam = ramp(step, eig_warmup, eig_max)

    koop_lam = ramp(step, koopman_warmup, koopman_weight)
    iit_lam = ramp(step, iit_warmup, iit_weight)
    sheaf_lam = ramp(step, sheaf_warmup, sheaf_weight)

    total = nll + eig_lam * eig

    koop_residual = output.get("koopman", {}).get("koopman_residual",
                                                   nll.new_zeros(()))
    if "koopman" in output:
        total = total + koop_lam * koop_residual

    iit_phi = output.get("iit_pid", {}).get("phi_surrogate",
                                              nll.new_zeros(()))
    if "iit_pid" in output:
        # Maximize synergy → subtract (we want phi LARGE)
        total = total - iit_lam * iit_phi

    sheaf_loss = output.get("sheaf_loss", nll.new_zeros(()))
    if "sheaf_loss" in output:
        total = total + sheaf_lam * sheaf_loss

    return {
        "total": total,
        "nll": nll,
        "eigensheaf": eig,
        "koopman_residual": koop_residual,
        "phi_surrogate": iit_phi,
        "sheaf_loss": sheaf_loss,
        "lam_eig": eig_lam,
        "lam_koopman": koop_lam,
        "lam_iit": iit_lam,
        "lam_sheaf": sheaf_lam,
    }
