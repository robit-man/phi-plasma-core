# Tier 2 Plan — Conditional on Day 8 Validation

This is the planning doc for Day 9-28 work. It executes ONLY if the Day 8 read of the Tier 1 MVP run shows:

1. Training completed without divergence (NLL doesn't NaN, doesn't diverge)
2. Head isotypic distance stayed below 0.2 (no mode collapse, the eigensheaf constraint held)
3. NLL trajectory shows *some* signal of plateau structure (step-wise drops) OR meaningfully beats the vanilla baseline at the same step count

If only 1 passes → root cause, don't proceed.
If 1+2 pass → proceed to Tier 2 cautiously (S5 first, S7 second).
If all three pass → proceed to Tier 2 confidently (S1, S5, S7 in parallel).

---

## What Tier 2 adds

### S1 — Sheaf-Laplacian Heat-Flow Recurrence

**Replaces** the recurrence between Hamiltonian blocks (currently just stacked block-by-block) with sheaf-Laplacian diffusion. The block-to-block flow becomes:

```
x_{ℓ+1} = (I + Δt · L_F)^(-1) · (HamiltonianBlock(x_ℓ))
```

where `L_F = δ^⊤ δ` is the sheaf Laplacian over a token-graph with Fibonacci-skip edges (F(3)=2, F(4)=3, F(5)=5, F(6)=8, F(7)=13) plus chain edges (i, i+1). The implicit Euler step is one sparse linear solve.

**Implementation strategy:**
- Build edge index once at model creation (depends only on seq_len)
- Restriction maps δ are learned (small `head_dim × head_dim` matrices, one per edge type)
- Solve via conjugate gradient (sparse PSD system) — `torch.linalg.solve` for small contexts, custom CG for ≥1024
- Penalty: `L_sheaf = ‖δx‖² / ‖x‖²` as auxiliary loss

**Estimated effort:** 3 days. **Files:** `src/phi_plasma/sheaf_laplacian.py` + integration into `plasma_core.py`.

**Falsifier:** Equilibrium states under sheaf flow should carry more semantic invariance than last-layer states of Tier 1. Test: take final-layer activations, project onto the null space of `L_F`, see if the projected representations are *better* (downstream probe accuracy) than the unprojected ones.

---

### S5 — Mourre-Commutator Attention

**Replaces** the standard `softmax(QK^T/√d)` scoring with anti-Hermitian commutator scoring:

```
α(q_i, k_j) = ⟨q_i, i[H, A] k_j⟩
```

with `H, A` learned Hermitian operators per head. Score is Hermitian (real-valued) and vanishes on joint eigenstates.

**Implementation strategy:**
- Add two learned `head_dim × head_dim` Hermitian operators per head (parameterize as `H = M + M^⊤` for a learned `M`)
- Score: `α_{ij} = q_i^⊤ (HA - AH) k_j` — computed as `(q (HA - AH))_i · k_j` (one matmul per pair)
- No softmax — the score IS the score, possibly with a `tanh` or normalization for numerical stability
- Inherit from existing `HeckeEigensheafAttention`

**Estimated effort:** 2 days. **Files:** `src/phi_plasma/mourre_attention.py` + config switch.

**Falsifier:** Mech-interp probe — do MoCA heads in the trained model specialize to *transitions* (sentence boundaries, topic shifts) more than vanilla heads? Compare with linear probes on attention patterns.

---

### S7 — Microlocal-Wavefront Sparse Attention

**Adds** data-dependent attention sparsity via the φ-wavelet wavefront-set indicator from your existing `microlocal_wavefront.py` (in `phi_corpus/world_model/`).

**Implementation strategy:**
- Vendor the wavefront surrogate (φ-wavelet differences orthogonal to dynamics direction)
- Per token t, compute a directional regularity indicator `σ(t) ∈ ℝ^{n_directions}` (already in upstream code)
- Build a per-batch sparse attention mask: token i attends to token j iff `σ(i)` and `σ(j)` share at least one direction above a learned threshold
- Use `torch.nn.functional.scaled_dot_product_attention` with the resulting mask
- Cheaper attention at smooth regions (sentence interiors); fuller attention at singularities (sentence boundaries, code blocks, equations)

**Estimated effort:** 2 days. **Files:** `src/phi_plasma/microlocal_wavefront.py` (vendored) + mask integration in `hecke_attention.py`.

**Falsifier:** Compared to dense attention at equal *FLOPs budget*, MWSA should match or beat perplexity. Compared to BigBird/Longformer at equal sparsity ratio, MWSA should match or beat on long-document QA.

---

## Composition order (recommended)

1. **S5 first** — smallest change, single-file extension of existing attention, biggest mechanistic-interpretability win.
2. **S7 second** — builds on attention infrastructure, gives compute savings that pay for S1.
3. **S1 third** — biggest architectural change (replaces inter-block recurrence), highest research risk.

After all three:
- Re-run the 30K-step validation
- Compare against the pure-Tier-1 result
- Compare against vanilla baseline
- If all three contributions stack additively (loss + interpretability), the Tier 2 set is the *complete* current architecture
- If one regresses, ablate to find which

## What gets stubbed in Day 9

When Day 8 says "go," Day 9 deliverables (planning, not coding):

- `src/phi_plasma/sheaf_laplacian.py` — module skeleton with the SheafLaplacianFlow class signature
- `src/phi_plasma/mourre_attention.py` — module skeleton with MourreCommutatorAttention class signature
- `src/phi_plasma/microlocal_wavefront.py` — vendored from world_model with the MicrolocalWavefrontIndicator class
- `tests/test_sheaf_laplacian.py` — invariants (PSD, null space dim, harmonic preservation)
- `tests/test_mourre_attention.py` — invariants (anti-Hermitian score, joint-kernel vanishing)
- `tests/test_microlocal_wavefront.py` — invariants (φ-wavelet correctness, directional regularity)
- `configs/tier2_full.yaml` — config with S1+S2+S3+S5+S7 all on
- `configs/tier2_s5_only.yaml`, `configs/tier2_s7_only.yaml`, `configs/tier2_s1_only.yaml` — ablations

## What gets deferred to Tier 3

- **S4 (Vlasov Mean-Field Attention)** — needs Poisson solver, mesh management. ~1 month standalone.
- **S6 (Landau-Damped Memory)** — depends on S4's machinery. ~3 weeks after S4.

## Open math questions

1. Does the sheaf-Laplacian's null space dimension match the "right" semantic granularity? Too low → over-compression; too high → not enough constraint.
2. Do Mourre-commutator scores need a softmax wrapper for numerical stability, or can they remain raw (which is the "honest" version that vanishes on eigenstates)?
3. Should the wavefront-set threshold be learned per layer or shared globally?

These are research questions, not engineering ones. The Tier 2 build answers them empirically.
