# Patent Notice

## Summary

The architectural innovations described in this repository — including the
combination of:

1. **Symplectic-Hamiltonian Token Flow**: tokenization into phase-space
   coordinates (q, p) with Störmer-Verlet leapfrog integration as the layer
   update rule for transformer-style language models
2. **Hecke-Eigensheaf Attention**: attention head mixing via a learned soft
   word in the Iwahori-Hecke algebra at parameter q = φ (golden ratio),
   with eigensheaf-decomposition penalties
3. **The combination of (1) and (2) trained with a Störmer-Verlet symplectic
   optimizer (Symplectic AdamW) for language modeling**

— are original contributions of the author (PRIME / `Prime-007-hash`).

## Patent posture

As of the initial publication date of this repository, **no patent
application has yet been filed** on the architectural innovations described.
The author reserves all rights to pursue patent protection in any
jurisdiction.

The Apache License 2.0 governs the use of the code in this repository.
Apache 2.0 includes a **patent grant clause** (Section 3) that automatically
licenses any patents that the author currently holds or may obtain in the
future that are necessarily infringed by use of this Work as published —
to all users of this Work, royalty-free and irrevocable, subject to the
license's termination conditions.

In plain terms:
- **You may use this code freely, including for commercial purposes**,
  under Apache 2.0.
- **If patents are subsequently granted on the architecture as published
  here, those patents are licensed to you automatically** by virtue of
  using this Apache 2.0–licensed Work.
- The patent license terminates only if you initiate patent litigation
  alleging this Work infringes your patents (Apache 2.0, Section 3).

## What's NOT auto-licensed

Future patents on **derivative architectural innovations that go beyond
what is published in this repository at the time you obtained it** are
not auto-licensed by this Apache 2.0 grant. Examples might include:

- Specific scaling techniques developed after publication
- Task-specific fine-tuning methodologies
- Production-deployment optimizations
- Combinations of the architecture with proprietary components

If you need broader patent rights — for example, to build a commercial
product that incorporates future improvements — please contact the author
to discuss licensing terms.

## Prior art and dependencies

This work builds on (and where appropriate, cites or implements):

- **Marsden-West discrete Lagrangian variational integrators** (Marsden &
  West, 2001). Public domain mathematical technique.
- **Iwahori-Hecke algebra of type A** (Iwahori, 1964). Public domain
  mathematical structure.
- **Apple PyTorch (MPS backend)**. Third-party library, used under its
  own license.
- **WikiText-2 dataset** (Merity et al., 2016). Used for evaluation only,
  not redistributed.

The combination of these techniques as a transformer-style language model
architecture trained end-to-end on next-token prediction — and the specific
empirical demonstration that this combination produces measurable
long-context information-preservation advantages — is the original
contribution claimed here.

## Contact

For licensing discussions, collaboration proposals, or questions about
patent posture:

- **GitHub issues** (tag as `licensing`): https://github.com/Prime-007-hash/phi-plasma-core/issues
- **Twitter / X**: [@data_adept](https://twitter.com/data_adept) — direct message for time-sensitive licensing inquiries
- **Telegram**: [@BASED_ROKO_PRIME](https://t.me/BASED_ROKO_PRIME) — preferred for commercial discussions
- **Substack**: [primordialasymmetria.substack.com](https://primordialasymmetria.substack.com) — for theoretical / research correspondence
- **Organization**: [Roko.Network](https://roko.network)

---

*Last updated: 2026-05-28*
