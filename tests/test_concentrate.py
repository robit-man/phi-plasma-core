"""Sanity tests for Φ-CONCENTRATE components and the composed model."""

import torch

from phi_plasma.constants import D_HIDDEN
from phi_plasma.koopman_probe import KoopmanProbe
from phi_plasma.iit_pid_probe import IITPhiProbe
from phi_plasma.sheaf_consistency import (
    SheafConsistencyHead, build_edges, edge_types_from_edges,
)
from phi_plasma.concentrate_model import ConcentrateModel, combined_concentrate_loss


# ─── Koopman probe ───────────────────────────────────────────────
def test_koopman_forward_shapes():
    p = KoopmanProbe(D_HIDDEN, latent_dim=32)
    h = torch.randn(2, 16, D_HIDDEN)
    out = p(h)
    assert out["latent"].shape == (2, 16, 32)
    assert out["lifted"].shape == (2, 16, p.full_dim)
    assert out["koopman_residual"].dim() == 0
    assert out["spectral_gap"].dim() == 0
    assert out["lyapunov"].dim() == 0
    assert out["decoded"].shape == (2, 16, D_HIDDEN)


def test_koopman_residual_decreases_under_gradient():
    """Direct GD on koopman_residual should reduce it (sanity)."""
    torch.manual_seed(0)
    p = KoopmanProbe(D_HIDDEN, latent_dim=32)
    opt = torch.optim.SGD(p.parameters(), lr=1e-2)
    h = torch.randn(2, 16, D_HIDDEN)
    r0 = p(h)["koopman_residual"].item()
    for _ in range(20):
        opt.zero_grad()
        r = p(h)["koopman_residual"]
        r.backward()
        opt.step()
    r1 = p(h)["koopman_residual"].item()
    assert r1 < r0, f"koopman residual didn't decrease: {r0:.4e} → {r1:.4e}"


def test_koopman_spectral_diagnostics_finite():
    p = KoopmanProbe(D_HIDDEN, latent_dim=32)
    h = torch.randn(2, 16, D_HIDDEN)
    out = p(h)
    assert torch.isfinite(out["spectral_gap"])
    assert torch.isfinite(out["lyapunov"])


# ─── IIT-PID probe ────────────────────────────────────────────────
def test_iit_pid_forward_shapes():
    p = IITPhiProbe(D_HIDDEN, n_partitions=4)
    h = torch.randn(2, 16, D_HIDDEN)
    out = p(h)
    assert out["phi_surrogate"].dim() == 0
    assert out["phi_surrogate"].item() >= 0.0
    assert out["total_correlation"].dim() == 0


def test_iit_pid_synergy_nonzero_for_correlated_partitions():
    """Synergy should be > 0 when partitions are correlated (a meaningful joint)."""
    torch.manual_seed(0)
    p = IITPhiProbe(D_HIDDEN, n_partitions=4)
    # Build h such that partitions share structure: copy a common signal across
    common = torch.randn(2, 16, D_HIDDEN // 4)
    h = torch.cat([common + 0.1 * torch.randn_like(common) for _ in range(4)], dim=-1)
    out = p(h)
    assert out["phi_surrogate"].item() > 0.0, \
        f"synergy should be positive for correlated parts, got {out['phi_surrogate'].item()}"


# ─── Sheaf consistency ────────────────────────────────────────────
def test_sheaf_edges_have_fibonacci_skips():
    edges = build_edges(seq_len=64, max_skip=6)
    diffs = (edges[1] - edges[0]).tolist()
    assert 1 in diffs
    assert 2 in diffs        # F(3)
    assert 3 in diffs        # F(4)
    assert 5 in diffs        # F(5)
    assert 8 in diffs        # F(6)


def test_sheaf_consistency_head_finite():
    head = SheafConsistencyHead(seq_len=64, stalk_dim=D_HIDDEN, n_edge_types=8)
    x = torch.randn(2, 64, D_HIDDEN)
    loss = head(x)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_sheaf_loss_decreases_under_gradient():
    """Directly minimizing the consistency loss should reduce it."""
    torch.manual_seed(7)
    head = SheafConsistencyHead(seq_len=64, stalk_dim=D_HIDDEN, n_edge_types=8)
    x = torch.randn(2, 64, D_HIDDEN, requires_grad=True)
    opt = torch.optim.SGD([x] + list(head.parameters()), lr=1e-2)
    l0 = head(x).item()
    for _ in range(20):
        opt.zero_grad()
        l = head(x)
        l.backward()
        opt.step()
    l1 = head(x).item()
    assert l1 < l0, f"sheaf loss didn't decrease: {l0:.4e} → {l1:.4e}"


# ─── Composed ConcentrateModel ────────────────────────────────────
def test_concentrate_forward_backward():
    torch.manual_seed(0)
    m = ConcentrateModel(
        vocab_size=512, d_hidden=D_HIDDEN, n_layers=2,
        n_heads=11, head_dim=32, seq_len=32,
        koopman_latent_dim=16, iit_n_partitions=4,
        sheaf_max_skip=4,
    )
    idx = torch.randint(0, 512, (2, 32))
    out = m(idx)
    assert out["logits"].shape == (2, 32, 512)
    assert "koopman" in out
    assert "iit_pid" in out
    assert "sheaf_loss" in out

    # Combined loss + backward
    losses = combined_concentrate_loss(m, out, idx, step=1500,
                                        koopman_weight=1e-3,
                                        iit_weight=1e-4,
                                        sheaf_weight=1e-3)
    losses["total"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum().item() > 0 for g in grads), "no real gradient"


def test_concentrate_param_count_reasonable():
    m = ConcentrateModel(seq_len=1024)
    n = m.param_count()
    # plasma backbone ~15.7M + 3 probes ~2-5M → 17-22M total
    assert 12e6 < n < 30e6, f"concentrate param count: {n/1e6:.2f}M"
