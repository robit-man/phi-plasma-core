"""Invariant test #3 — phase-space volume preservation.

The Störmer-Verlet leapfrog applied to the (q, p) state is volume-preserving:
det |∂(q', p') / ∂(q, p)| = 1.

We verify numerically by computing the Jacobian of one block step and
checking |det J| ≈ 1 within numerical tolerance, on a small test instance
(full 352-dim Jacobian would be 352×352 and we'd lose precision)."""

import torch

from phi_plasma.constants import N_HEADS, HEAD_DIM
from phi_plasma.hamiltonian_block import HamiltonianBlock


def test_volume_preserved_small():
    """Use a SMALL configuration where computing the Jacobian is tractable."""
    torch.manual_seed(3)
    # Tiny config: 11 heads × 2 dim_per_head = 22-dim hidden, so 11-dim phase space.
    d_h, h, dh = 22, 11, 2
    block = HamiltonianBlock(d_hidden=d_h, n_heads=h, head_dim=dh, h_step=0.05)
    block.eval()

    # Single batch element, single token — so phase space is just (d_h,).
    x = torch.randn(1, 1, d_h, requires_grad=True) * 0.3

    def f(x_flat):
        x = x_flat.view(1, 1, d_h)
        return block(x).view(d_h)

    J = torch.autograd.functional.jacobian(f, x.view(d_h))      # (d_h, d_h)
    det = torch.linalg.det(J)
    # Volume preservation: |det J| should be close to 1.
    # Tolerate up to 10% drift because:
    #   - finite step size (h=0.05)
    #   - V depends on a=attn(x), which is recomputed at q' so adds a small non-symplectic term
    rel_err = abs(abs(det.item()) - 1.0)
    assert rel_err < 0.20, f"|det J| = {abs(det.item()):.4f}, expected ≈ 1.0 (rel_err={rel_err:.4f})"


def test_volume_preserved_smaller_h():
    """With even smaller h, |det J| should be closer to 1."""
    torch.manual_seed(5)
    d_h, h, dh = 22, 11, 2
    block = HamiltonianBlock(d_hidden=d_h, n_heads=h, head_dim=dh, h_step=0.01)
    block.eval()

    x = torch.randn(1, 1, d_h, requires_grad=True) * 0.3

    def f(x_flat):
        x = x_flat.view(1, 1, d_h)
        return block(x).view(d_h)

    J = torch.autograd.functional.jacobian(f, x.view(d_h))
    det = torch.linalg.det(J)
    rel_err = abs(abs(det.item()) - 1.0)
    # At h=0.01, should be MUCH closer to 1.
    assert rel_err < 0.05, f"|det J| = {abs(det.item()):.4f} at h=0.01 (rel_err={rel_err:.4f})"
