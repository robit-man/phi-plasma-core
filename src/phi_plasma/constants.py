"""φ-derived constants. Vendored from world_model/phi_constants.py + logos/phi_logos_constants.py
and reduced to what the MVP needs."""

from __future__ import annotations
import math

# ───────────────────────────────────────────────────────────────────
# Algebraic constants.
# ───────────────────────────────────────────────────────────────────
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0           # ≈ 1.6180339887
PHI_INV: float = 1.0 / PHI                          # ≈ 0.6180339887
SPECTRAL_GAP: float = PHI_INV * PHI_INV             # = φ⁻² ≈ 0.3819660113

# Lucas numbers L(0)..L(15).
LUCAS_TABLE: tuple[int, ...] = (
    2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364
)

# Fibonacci numbers F(0)..F(15).
FIB_TABLE: tuple[int, ...] = (
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610
)

# ───────────────────────────────────────────────────────────────────
# Architectural anchors (MVP sizing).
# ───────────────────────────────────────────────────────────────────
N_HEADS: int = LUCAS_TABLE[5]                       # = 11
HECKE_BASE_RANK: int = N_HEADS                      # heads index Hecke basis
HECKE_N_GENERATORS: int = HECKE_BASE_RANK - 1       # T_1 .. T_{n-1}
HECKE_PARAM_Q: float = PHI                          # q-deformation parameter

HEAD_DIM: int = 32                                  # 32 * 11 = 352
D_HIDDEN: int = N_HEADS * HEAD_DIM                  # = 352

N_LAYERS: int = 6
VOCAB_SIZE: int = 28657                             # WikiText-2 vocab (cache)
SEQ_LEN: int = 1024
N_IMAGINED_STEPS: int = FIB_TABLE[6]                # = 8 — variational rollout depth
