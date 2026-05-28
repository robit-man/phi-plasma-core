"""Iwahori-Hecke 2-block representation. Vendored from Φ-LOGOS-LITE.

Honesty notes from upstream:
  - R1 (quadratic) holds LOCALLY on each 2-block
  - R2 (commutation) holds GLOBALLY (disjoint supports)
  - R3 (braid) does NOT hold — would require Specht modules. Documented limitation.

For Hecke-Eigensheaf Attention (S2) we only need T_i as well-defined linear
operators on the head dimension; faithful R3 is not load-bearing for the
isotypic constraint we enforce."""

from __future__ import annotations
import torch

from .constants import HECKE_BASE_RANK, HECKE_PARAM_Q


def build_hecke_generator(n: int, i: int, q: float = HECKE_PARAM_Q,
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """T_i acting on ℂ^n. i is 1-indexed (i ∈ {1, ..., n-1}).
    Returns (n, n) matrix."""
    if not 1 <= i <= n - 1:
        raise ValueError(f"i must satisfy 1 ≤ i ≤ {n-1}, got {i}")
    T = torch.eye(n, dtype=dtype)
    a, b = i - 1, i
    # In basis (e_a, e_b):  T = [[q, 1], [0, -1]]
    T[a, a] = q
    T[a, b] = 1.0
    T[b, a] = 0.0
    T[b, b] = -1.0
    return T


class HeckeAlgebra:
    """Hecke algebra of rank n with parameter q. Pre-builds generators T_1..T_{n-1}."""

    def __init__(self, n: int = HECKE_BASE_RANK, q: float = HECKE_PARAM_Q,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.n = n
        self.q = q
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device
        self._generators = [
            build_hecke_generator(n, i, q, dtype).to(self.device)
            for i in range(1, n)
        ]

    @property
    def n_generators(self) -> int:
        return len(self._generators)

    def generator(self, i: int) -> torch.Tensor:
        if not 1 <= i <= self.n_generators:
            raise IndexError(f"generator index out of range: {i}")
        return self._generators[i - 1]

    def stack(self) -> torch.Tensor:
        """All generators stacked: (n_generators, n, n)."""
        return torch.stack(self._generators)

    def _local_2block(self, i: int) -> torch.Tensor:
        a, b = i - 1, i
        return self.generator(i)[[a, b], :][:, [a, b]]

    def verify_relations(self, atol: float = 1e-5) -> dict[str, bool]:
        """Real checks on R1-local + R2-global; R3 documented as limitation."""
        results: dict[str, bool] = {}
        for i in range(1, self.n_generators + 1):
            block = self._local_2block(i)
            lhs = block @ block
            rhs = (self.q - 1) * block + self.q * torch.eye(2, dtype=self.dtype,
                                                            device=self.device)
            results[f"R1_quadratic_T_{i}"] = bool(torch.allclose(lhs, rhs, atol=atol))
        for i in range(1, self.n_generators + 1):
            for j in range(i + 2, self.n_generators + 1):
                Ti, Tj = self.generator(i), self.generator(j)
                results[f"R2_commute_T_{i}_T_{j}"] = bool(torch.allclose(
                    Ti @ Tj, Tj @ Ti, atol=atol))
        return results

    def all_local_relations_hold(self, atol: float = 1e-5) -> bool:
        return all(self.verify_relations(atol).values())
