# HEALTH

Updated: build session 2026-05-27

## Status

| Component | State |
|---|---|
| Project scaffolding | ✓ shipped |
| Hecke algebra (vendored from logos) | ✓ shipped, 16/16 tests pass |
| Symplectic AdamW (vendored from world_model) | ✓ shipped |
| S3 Hamiltonian block (NEW) | ✓ shipped, symplectic + volume tests pass |
| S2 Hecke-Eigensheaf attention (NEW) | ✓ shipped, eigensheaf penalty tested |
| PlasmaCore composed model | ✓ shipped, 15.4M params with tied embeddings |
| VanillaTransformer baseline | ✓ shipped |
| WikiText-2 loader | ✓ uses existing token cache |
| Training loop | ✓ shipped, NLL drops from 10.33 → 6.89 in 500 steps |
| 4 invariant tests | ✓ all pass |
| Smoke run | ✓ 500 steps in 52s on M2 Pro / MPS |
| Overnight launcher | ✓ `scripts/overnight.sh` ready |
| Ablation harness | ✓ `scripts/ablate.py` ready |
| Results reader | ✓ `scripts/read_results.py` ready |

## Compute reality check

Original estimate: 12–17 hours for 30K steps.
**Smoke-derived estimate: ~50–90 minutes.** Reasons:
- MPS on M2 Pro is faster than the per-step model I had
- ~10 steps/sec at seq_len=256 batch=4
- Scaling to seq_len=1024 batch=8: roughly 4× slower per step → ~2.5 sps → 30K steps ≈ 200 minutes worst case
- Could be 50 minutes if MPS scales sub-linearly with context (which it tends to)

## What "go/no-go" looks like (Day 8 in the original plan)

After `scripts/overnight.sh` finishes, `scripts/py scripts/read_results.py` answers four questions:

1. **Did training converge cleanly (no NaN/divergence)?** — required for go.
2. **Did head isotypic distance stay below 0.2?** — gate for "no mode collapse" claim.
3. **Did the NLL show plateau structure?** — *the* diagnostic for whether geometric strata are being discovered.
4. **Was the eigensheaf penalty actively constraining, or trivially satisfied?** — if final penalty is tiny (say <1e-10) the constraint isn't really biting; not necessarily a failure, but a flag to consider stronger constraints in Tier 2.

If all four pass → Tier 2 is justified (S1, S5, S7).
If 1–2 pass → root-cause before Tier 2.
If 0 pass → re-examine assumptions; possible the architectural design has a flaw.

## Open issues / future work

- Eigensheaf penalty is currently an off-diagonal Gram penalty. The stronger version — explicit projection onto trivial / alternating / standard irreducibles of the Hecke algebra — is deferred to Tier 2.
- `head_isotypic_distance` is the diagnostic; in high-dim spaces random Gaussian vectors are already near-orthogonal so the "starting" iso is tiny. The risk we're really gating against is *drift to 0.4–0.6* during training, which would manifest in long runs. 500 steps is too short to see that drift in either direction.
- Volume preservation test tolerates 20% drift at h=0.5 — this is because the action `a` depends on `q` (via attention), introducing a small non-symplectic coupling. Strictly symplectic would require freezing `a` across the step; we accept the trade-off for representational power.
- Currently no FlashAttention; using vanilla `softmax(QK^T/√d)` for portability on MPS. Tier 2 can swap in `torch.nn.functional.scaled_dot_product_attention` for ~2× speedup.

## Reproducibility

```bash
# From scratch:
cd ~/phi_plasma_core
PYTHONPATH=src /Users/uvizius/.claude/ring/.venv/bin/python -m pytest tests/
# 16 passed in ~1.5s

# Smoke training:
PYTHONPATH=src /Users/uvizius/.claude/ring/.venv/bin/python -m phi_plasma.train --config configs/smoke.yaml
# 500 steps in ~52s; NLL 10.33 → 6.89

# Overnight (Tier 1 MVP):
./scripts/overnight.sh
# tail -f logs/mvp_18m/run.log
```
