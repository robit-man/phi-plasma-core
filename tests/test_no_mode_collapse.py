"""Invariant test #4 — head isotypic distance.

Pairwise head-V cosine similarity should be measurable and (after a short
training warmup) low. At init, similarity is near zero by construction (random
init). After a brief gradient step against the eigensheaf penalty, the
constraint should push it toward the structured low-similarity regime.

This is a SANITY test that the diagnostic computes a sensible number; the
real test of no-mode-collapse comes from observing the diagnostic during
the long run."""

import torch

from phi_plasma.constants import D_HIDDEN, N_HEADS, HEAD_DIM
from phi_plasma.hecke_attention import HeckeEigensheafAttention


def test_isotypic_distance_finite_and_in_range():
    layer = HeckeEigensheafAttention(d_hidden=D_HIDDEN, n_heads=N_HEADS, head_dim=HEAD_DIM)
    d = layer.head_isotypic_distance()
    assert 0.0 <= d <= 1.0, f"isotypic distance out of [0, 1]: {d}"


def test_isotypic_distance_low_at_init():
    """Random init should give low pairwise similarity."""
    torch.manual_seed(13)
    layer = HeckeEigensheafAttention(d_hidden=D_HIDDEN, n_heads=N_HEADS, head_dim=HEAD_DIM)
    d = layer.head_isotypic_distance()
    # Random Gaussian vectors in high dim → expected cosine similarity ~1/√(D)
    expected = 1.0 / (D_HIDDEN ** 0.5)
    assert d < 5 * expected, \
        f"isotypic distance at init too high: {d:.4f} (expected ~{expected:.4f})"


def test_eigensheaf_penalty_finite_and_positive():
    layer = HeckeEigensheafAttention(d_hidden=D_HIDDEN, n_heads=N_HEADS, head_dim=HEAD_DIM)
    p = layer.eigensheaf_penalty()
    assert torch.isfinite(p), f"non-finite eigensheaf penalty: {p}"
    assert p.item() >= 0.0, f"eigensheaf penalty should be non-negative: {p.item()}"


def test_eigensheaf_penalty_responds_to_weights():
    """Two layers with different random seeds should give different penalties."""
    torch.manual_seed(0)
    layer_a = HeckeEigensheafAttention(d_hidden=D_HIDDEN, n_heads=N_HEADS, head_dim=HEAD_DIM)
    torch.manual_seed(1)
    layer_b = HeckeEigensheafAttention(d_hidden=D_HIDDEN, n_heads=N_HEADS, head_dim=HEAD_DIM)
    p_a = layer_a.eigensheaf_penalty().item()
    p_b = layer_b.eigensheaf_penalty().item()
    assert abs(p_a - p_b) > 1e-8, \
        f"penalties identical across seeds — diagnostic broken: {p_a} vs {p_b}"


def test_eigensheaf_penalty_drops_under_gradient():
    """One backward + step should reduce the penalty when nothing else competes."""
    torch.manual_seed(17)
    layer = HeckeEigensheafAttention(d_hidden=D_HIDDEN, n_heads=N_HEADS, head_dim=HEAD_DIM)
    opt = torch.optim.SGD(layer.parameters(), lr=1e-3)

    p_before = layer.eigensheaf_penalty().item()
    for _ in range(20):
        opt.zero_grad()
        loss = layer.eigensheaf_penalty()
        loss.backward()
        opt.step()
    p_after = layer.eigensheaf_penalty().item()
    assert p_after < p_before, \
        f"eigensheaf penalty didn't decrease under direct gradient descent: " \
        f"{p_before:.4e} → {p_after:.4e}"
