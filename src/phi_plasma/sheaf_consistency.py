"""Sheaf edge consistency loss with Fibonacci-skip topology (ported from
world_model/sheaf_attention.py).

Background
----------
The token sequence forms a graph G = (V, E) with V = {0..T-1} and edges
E connecting (i) consecutive tokens and (ii) tokens at Fibonacci offsets
(F(3)=2, F(4)=3, F(5)=5, F(6)=8, F(7)=13). A cellular sheaf F over G
assigns:
  - stalk F(v) at each vertex (a vector in ℝ^Dh)
  - stalk F(e) at each edge
  - restriction maps F_{v→e}: F(v) → F(e)

The consistency loss measures how much restriction-map outputs disagree
across each edge:

    L_sheaf = Σ_{e=(v,w)} ‖R_v(x_v) − R_w(x_w)‖²

We share restriction maps by edge-type (chain vs. specific Fibonacci skip)
rather than per-edge, giving O(n_types · Dh²) parameters instead of
O(E · Dh²).

NOTE on framing: the original world_model code wrapped the consistency loss
in a ℤ[φ] "golden norm" |a² + ab − b²|. The audit found this is decorative
— it's a learned bilinear form on pairs, no algebraic ring structure is
enforced or used. This port keeps the geometric mechanism (restriction
maps + edge topology) and drops the ℤ[φ] framing. The mechanism is honest;
the algebraic label was overpromise."""

from __future__ import annotations
import torch
import torch.nn as nn

from .constants import PHI_INV, FIB_TABLE


def build_edges(seq_len: int, max_skip: int = 8) -> torch.Tensor:
    """Build chain + Fibonacci-skip edge list as (2, E) tensor."""
    edges: list[tuple[int, int]] = []
    # Chain edges
    for i in range(seq_len - 1):
        edges.append((i, i + 1))
    # Fibonacci-skip edges, F(3)..F(max_skip)
    for k in range(3, max_skip + 1):
        skip = FIB_TABLE[k]
        if skip <= 1:
            continue
        for i in range(seq_len - skip):
            edges.append((i, i + skip))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).T


def edge_types_from_edges(edges: torch.Tensor, max_edge_types: int = 8) -> torch.Tensor:
    """Assign each edge an integer type based on its skip distance.

    Type 0: chain (w − v == 1)
    Type k: Fibonacci skip F(k+2) for k = 1..max_edge_types-1
    """
    diffs = edges[1] - edges[0]
    types = torch.zeros_like(diffs)
    skip_to_type: dict[int, int] = {1: 0}
    next_type = 1
    for k in range(3, max_edge_types + 2):
        f = FIB_TABLE[k]
        if f > 1 and f not in skip_to_type and next_type < max_edge_types:
            skip_to_type[f] = next_type
            next_type += 1
    for i in range(diffs.shape[0]):
        d = int(diffs[i].item())
        types[i] = skip_to_type.get(d, 0)
    return types


class RestrictionMaps(nn.Module):
    """Per-edge-type restriction maps R_v, R_w: stalk → stalk.

    For each edge type t (chain, Fibonacci-skip-2, skip-3, skip-5, skip-8...)
    we have one learned (Dh, Dh) pair. Shared across all edges of that type.

    Identity-initialized + small noise so the system starts close to a
    trivial sheaf (no restriction = identity)."""

    def __init__(self, stalk_dim: int, n_edge_types: int = 8):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.n_edge_types = n_edge_types
        self.R_left = nn.Parameter(
            torch.eye(stalk_dim).unsqueeze(0).repeat(n_edge_types, 1, 1)
        )
        self.R_right = nn.Parameter(
            torch.eye(stalk_dim).unsqueeze(0).repeat(n_edge_types, 1, 1)
        )
        with torch.no_grad():
            self.R_left.add_(torch.randn_like(self.R_left) * 0.01 * (PHI_INV ** 2))
            self.R_right.add_(torch.randn_like(self.R_right) * 0.01 * (PHI_INV ** 2))

    def forward(self, edge_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """edge_types: (E,) → R_left, R_right both (E, Dh, Dh)."""
        return self.R_left[edge_types], self.R_right[edge_types]


def sheaf_consistency_loss(x: torch.Tensor, edges: torch.Tensor,
                            R_left: torch.Tensor, R_right: torch.Tensor) -> torch.Tensor:
    """L_sheaf = Σ_e ‖R_v(x_v) − R_w(x_w)‖².

    Args:
        x:        (B, T, Dh) stalk values
        edges:    (2, E) edge endpoints
        R_left:   (E, Dh, Dh) per-edge src restriction
        R_right:  (E, Dh, Dh) per-edge dst restriction
    Returns scalar mean per-edge squared norm."""
    src, dst = edges[0], edges[1]
    x_src = x[:, src]
    x_dst = x[:, dst]
    Rv_x = torch.einsum("bed,edk->bek", x_src, R_left)
    Rw_x = torch.einsum("bed,edk->bek", x_dst, R_right)
    diff = Rv_x - Rw_x
    return diff.pow(2).sum(dim=-1).mean()


class SheafConsistencyHead(nn.Module):
    """Wrapper combining edge construction + restriction maps + loss computation.

    Use as a side-loss head on the model's hidden states. The edge index
    is precomputed at construction and registered as a buffer (no recomputation
    per forward pass)."""

    def __init__(self, seq_len: int, stalk_dim: int,
                 n_edge_types: int = 8, max_skip: int = 8):
        super().__init__()
        edges = build_edges(seq_len, max_skip=max_skip)
        edge_types = edge_types_from_edges(edges, max_edge_types=n_edge_types)
        self.register_buffer("edges", edges, persistent=False)
        self.register_buffer("edge_types", edge_types, persistent=False)
        self.restriction = RestrictionMaps(stalk_dim, n_edge_types=n_edge_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, Dh) → scalar consistency loss."""
        R_left, R_right = self.restriction(self.edge_types)
        return sheaf_consistency_loss(x, self.edges, R_left, R_right)
