"""Invariant test #1 — Hecke 2-block relations hold.

R1 (quadratic, local on 2-block) and R2 (commutation, global) must both hold
within tight tolerance at q = φ. R3 (braid) is a documented limitation of the
2-block representation and is NOT tested."""

import torch

from phi_plasma.constants import PHI, HECKE_BASE_RANK, HECKE_N_GENERATORS
from phi_plasma.hecke_algebra import HeckeAlgebra


def test_hecke_relations_hold_at_phi():
    algebra = HeckeAlgebra(n=HECKE_BASE_RANK, q=PHI)
    results = algebra.verify_relations(atol=1e-5)
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"Hecke relations failed: {failed}"


def test_hecke_generator_count():
    algebra = HeckeAlgebra(n=HECKE_BASE_RANK)
    assert algebra.n_generators == HECKE_N_GENERATORS


def test_hecke_generator_shapes():
    algebra = HeckeAlgebra(n=HECKE_BASE_RANK)
    for i in range(1, algebra.n_generators + 1):
        T_i = algebra.generator(i)
        assert T_i.shape == (HECKE_BASE_RANK, HECKE_BASE_RANK)


def test_hecke_generator_is_2_block_plus_identity():
    """Each T_i should equal identity except on the (i-1, i) 2-block."""
    algebra = HeckeAlgebra(n=HECKE_BASE_RANK, q=PHI)
    for i in range(1, algebra.n_generators + 1):
        T_i = algebra.generator(i)
        a, b = i - 1, i
        # Make a copy with the 2-block replaced by identity 2x2.
        T_check = T_i.clone()
        T_check[a:a+1, a:a+1] = 1.0
        T_check[a:a+1, b:b+1] = 0.0
        T_check[b:b+1, a:a+1] = 0.0
        T_check[b:b+1, b:b+1] = 1.0
        # Now T_check should be the identity.
        assert torch.allclose(T_check, torch.eye(HECKE_BASE_RANK), atol=1e-7), \
            f"T_{i} has nonzero entries outside the (i-1, i) 2-block"
