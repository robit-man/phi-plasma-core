# Paper: Buried Treasure — Two Rigorous MechInterp Probes

LaTeX source for the standalone paper on the Koopman EDMD and Williams-Beer
PID probes extracted from PHI_AEON L7 and L12.

## Build

Requires MacTeX or TeX Live with `pdflatex`:

```bash
make            # produces main.pdf (~6-8 pages)
open main.pdf   # macOS preview
```

Clean artifacts:

```bash
make clean
```

## arXiv submission

```bash
make arxiv      # builds probes-arxiv.tar.gz from main.tex
```

Upload at https://arxiv.org/submit.

### Recommended categories

- **Primary**: `cs.LG` (Machine Learning) — the probes are language-model
  interpretability tools
- **Cross-list**: `cs.AI` (Artificial Intelligence) — broader audience
- **Optional**: `cs.IT` (Information Theory) — the PID probe uses
  Williams-Beer decomposition; some IT readers will be interested

### Framing for reviewers

This paper is positioned as a **methodological note**, not an architecture
paper. The contribution is:

1. Clean, low-overhead PyTorch implementations of two rigorous probes
2. The audit pattern that recovered them from a retired architecture

Reviewers from mechanistic-interpretability tracks (NeurIPS MechInterp
workshop, ICML workshops, ICLR poster track) are the most natural
audience. The paper does NOT claim to introduce Koopman EDMD or PID as
theoretical objects — those are credited to the original authors
(Williams-Kevrekidis-Rowley 2015 and Williams-Beer 2010 respectively).
The contribution is engineering + methodology.

### Possible workshop venues

- **NeurIPS Workshop on Attributing Model Behavior at Scale** (mechinterp track)
- **NeurIPS Workshop on Interpretable AI**
- **ICML Workshop on Mechanistic Interpretability**
- **ICLR Workshop on Tiny Papers** (matches the methodological-note framing)

## Companion artifacts

The code lives in the parent repository:

- `src/phi_plasma/koopman_probe.py` — Koopman EDMD probe (L7 port)
- `src/phi_plasma/iit_pid_probe.py` — Williams-Beer PID probe (L12 port)
- `tests/test_concentrate.py` — sanity tests for both probes

The probes can be used **independently** of the rest of the phi-plasma-core
architecture. They take a hidden state tensor of shape `(B, T, d_model)`
and return diagnostic outputs. Attach to any transformer backbone.

## Twitter announcement template

When arXiv ID is assigned:

> Two rigorous mech-interp probes hiding inside a retired language model:
>
> 1. Koopman EDMD with polynomial lift + exact spectral gap diagnostics
> 2. Williams-Beer PID synergy via Gaussian MMI
>
> Both differentiable, MPS-compatible, ~5KB of PyTorch each. Extracted
> from PHI_AEON L7+L12.
>
> Paper: arxiv.org/abs/XXXX.XXXXX
> Code: github.com/Prime-007-hash/phi-plasma-core
>
> The audit pattern that recovered them: read the forward methods. Ask
> whether the gradient flows through the mathematical object the name
> promises, or through a generic op with a descriptive label.
>
> 70% was decoration. 30% was real. The 30% is now visible.

## Note on length

The current `main.tex` is ~6 pages dense; compressing to a 4-page
"Tiny Papers" submission is straightforward — drop §4.3 (the audit
pattern subsection), tighten §1, condense §2.4 and §3.4 into a single
"Verification" section.
