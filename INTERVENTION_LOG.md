# Active-Monitor Intervention Log

Real-time interventions made during the Day 7-8 training run, with the reasoning preserved for the user.

## v1 (vacuous orthogonality penalty)

**Config**: `h_step=0.5`, `eig_max=0.5`, eigensheaf penalty = mean off-diagonal Gram squared.

**Observation**: Penalty dropped from 7.74e-5 → 1e-17 in ~500 steps. Head iso pinned at 0.000. The penalty is trivially satisfied because random Gaussian V-projections in 22K-dim space are already near-orthogonal at init. The "constraint" applies no pressure.

**Trajectory** (killed at step 2400):
- NLL 10.31 → 5.49 (PPL 30K → 241)
- val_ppl at step 2000: 397.43
- Smooth log-decay, NO visible plateau structure

**Verdict**: Penalty design is vacuous. Loss trajectory looks like vanilla parameter-matched transformer at this scale.

**Intervention**: Replace constraint with Hecke centralizer condition `[T_i, UU^T]=0` (Schur's lemma: heads partition into isotypic blocks). Also tighten h_step to 0.2.

---

## v2 (Hecke centralizer constraint)

**Config**: `h_step=0.2`, `eig_max=100`, eigensheaf penalty = `Σ_i ‖[T_i, UU^T]‖²_F / ‖UU^T‖²_F`.

**Initial penalty at random init**: 8.21e-04 (10× v1's 7.74e-5 — better, but still small in absolute scale).

**Observation** (killed at step 175): Penalty already at 1.77e-07. The centralizer constraint *also* trivially satisfied because high-dim Gaussian UU^T → scalar I in expectation, and scalar I commutes with every T_i. **Same vacuous-saturation failure mode as v1, different mechanism.**

**Verdict**: My structural constraint designs share a common flaw — they're trivially satisfied by isotropic high-dim Gaussians. The constraint can't bite without breaking isotropy in a specific direction, and any non-isotropy target requires fixed reference directions that are hard to motivate without empirical evidence.

**Intervention**: Drop the eigensheaf penalty entirely. The architectural distinction lives in the forward pass (Hamiltonian flow + Hecke head-mixing via `word_policy`). The auxiliary penalty was adding nothing. Run plasma WITHOUT the penalty against vanilla baseline as a clean A/B.

---

## v3 (plasma vs vanilla, no auxiliary penalty)

**Config**:
- Plasma arm: `arch=plasma`, `h_step=0.2`, `eig_max=0.0`, all other features (Hamiltonian flow, Hecke word policy in forward pass) intact.
- Vanilla arm: `arch=vanilla`, parameter-matched, same SymplecticAdamW optimizer, same data, same 3000-step schedule, same seed.

**What this actually tests**: Does Hamiltonian-flow + Hecke head-mixing alone outperform a vanilla transformer at the same parameter count under the same optimizer? This is the *real* architectural comparison — no decorative constraints, just the structural moves.

**Expected outcomes**:
- **Strong PASS**: plasma val_ppl < vanilla val_ppl at step 3000 by ≥5% AND plasma's loss trajectory has visible plateau structure that vanilla's doesn't.
- **Weak PASS**: plasma matches vanilla closely (within 2%), and the symplectic machinery's *engineering claims* (volume preservation, stable integration over thousands of leapfrog steps) hold. This is a positive result for stability without a positive result for distinctive learning dynamics.
- **FAIL**: plasma materially worse than vanilla. The Hamiltonian-flow + Hecke combo has hurt rather than helped. Day 8 verdict: don't proceed to Tier 2; the design assumption is wrong.

---

## What we learned by intervening

1. **Auxiliary penalties designed to enforce algebraic structure are hard.** In high-dimensional regimes, most "structured" constraints (orthogonality, centralizer, scalar irreps) are trivially satisfied by isotropic random init. A non-vacuous algebraic constraint needs to identify a specific *direction* the algebra picks out — and we don't have one yet without empirical guidance.

2. **The architectural features can stand alone.** Hamiltonian flow, Hecke head-mixing via soft word, symplectic optimizer — these don't need a regularization term to be present. Whether they *help* is the open question.

3. **The smooth-log-decay observation is informative.** v1's loss curve was smooth log-decay, indistinguishable from vanilla transformer behavior. If v3's plasma arm also produces smooth log-decay AND matches vanilla val_ppl, the conclusion is: the geometric machinery is *neutral* — it doesn't hurt, but it doesn't help. If plasma underperforms vanilla, the Hamiltonian flow is actively costing capacity. If plasma overperforms, there's a real architectural win.

4. **30K steps was overspec.** The original plan said 30K steps over 6 hours. We're testing the central hypothesis with 3K steps in ~40 min per arm — sufficient to see the trajectory shape and final val_ppl. If 3K validates, extending to 30K is just confirmation.

5. **Aggressive intervention beat passive monitoring.** Sticking with v1 for the full overnight run would have produced one data point (smooth-log-decay plasma) without a control. The three-iteration approach gave us: (a) a vacuous-penalty negative control, (b) a different-vacuous-penalty negative control, (c) a clean A/B test in the remaining time.

## Conclusion (to be filled when v3 completes)

`<pending v3 results>`
