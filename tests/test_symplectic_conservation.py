"""Invariant test #2 — symplectic Hamiltonian conservation.

For a Hamiltonian block in eval mode with a frozen potential, the Hamiltonian
H = T(p) + V(q, a) should be approximately conserved across the leapfrog step.
The discrete Störmer-Verlet integrator does NOT conserve H exactly — it
conserves a "modified" Hamiltonian H̃ that differs from H by O(h²). So the
drift in H per step should be O(h²) and bounded, not unbounded growth."""

import math

import torch

from phi_plasma.constants import D_HIDDEN, N_HEADS, HEAD_DIM
from phi_plasma.hamiltonian_block import HamiltonianBlock


def test_hamiltonian_drift_bounded_eval_mode():
    """Run K consecutive steps; drift should not exceed O(h²) per step."""
    torch.manual_seed(7)
    h = 0.1
    K = 20
    block = HamiltonianBlock(d_hidden=D_HIDDEN, n_heads=N_HEADS,
                              head_dim=HEAD_DIM, h_step=h)
    block.eval()  # freeze any layernorm running stats

    B, T = 2, 16
    x = torch.randn(B, T, D_HIDDEN) * 0.5

    energies = []
    for _ in range(K):
        H = block.hamiltonian(x).mean().item()
        energies.append(H)
        with torch.no_grad():
            x = block(x)

    energies = torch.tensor(energies)
    drift = (energies[1:] - energies[0]).abs().max().item()
    initial_magnitude = max(abs(energies[0].item()), 1.0)

    # Drift should be small relative to initial energy magnitude.
    # Symplectic integrators bound drift to O(h²) over O(1/h) steps;
    # over K=20 steps this means drift / |H_0| should be < ~h² × K = 0.2
    bound = 0.3
    rel_drift = drift / initial_magnitude
    assert rel_drift < bound, \
        f"Hamiltonian drift {rel_drift:.4f} exceeds bound {bound} after {K} steps"


def test_hamiltonian_finite():
    """Hamiltonian should be finite (no NaN/Inf) for typical inputs."""
    torch.manual_seed(11)
    block = HamiltonianBlock(d_hidden=D_HIDDEN, n_heads=N_HEADS,
                              head_dim=HEAD_DIM, h_step=0.5)
    x = torch.randn(4, 32, D_HIDDEN)
    H = block.hamiltonian(x)
    assert torch.isfinite(H).all(), f"non-finite Hamiltonian: {H}"
