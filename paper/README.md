# Paper — Φ-Plasma-Core

This directory contains the arxiv-submission-ready LaTeX source for the
research paper accompanying this repository.

## Build

Requires TeX Live (Linux/Windows) or MacTeX (macOS). Install on macOS:

```bash
brew install --cask mactex-no-gui
# or, if you want the full ~5GB distribution:
# brew install --cask mactex
```

Build the PDF:

```bash
cd paper/
make                  # produces main.pdf (~10-12 pages)
open main.pdf         # macOS preview
```

Clean build artifacts:

```bash
make clean
```

## Submission targets

### arXiv

```bash
make arxiv            # builds phi-plasma-core-arxiv.tar.gz
```

Upload the tarball to https://arxiv.org/submit. Suggested category:
**cs.LG** (Machine Learning), cross-listed to **cs.CL** (Computation
and Language) and optionally **math-ph** (Mathematical Physics).

Estimated wait for endorsement (if first arxiv submission): up to 1
week. If you don't have prior cs.LG papers, you may need a sponsor.
Reach out to a researcher on Twitter or via email — most senior
researchers will endorse first-time authors with a credible paper.

### ResearchGate

ResearchGate accepts PDF uploads directly. Build `main.pdf`, then:

1. Sign in at https://www.researchgate.net
2. Add a new publication → Research item
3. Type: Preprint
4. Upload `main.pdf`
5. Add abstract (copy from the paper's abstract)
6. Add the GitHub repo link as supplementary material

### OpenReview (workshop submissions)

For workshop submissions at NeurIPS / ICML / ICLR (typically due ~3
months before main conference), the same LaTeX source compiles to
OpenReview's requirements (usually they accept arxiv-style submissions).

### Twitter / X announcement template

Once arxiv ID is assigned:

> New paper: Φ-Plasma-Core — a symplectic-Hamiltonian transformer with
> 2.31× more stable long-context perplexity than vanilla baselines.
>
> Each layer is one Störmer-Verlet leapfrog step. Heads are mixed by
> the Hecke algebra at q=φ.
>
> arxiv: https://arxiv.org/abs/XXXX.XXXXX
> code: https://github.com/Prime-007-hash/phi-plasma-core
>
> The architectural prediction was: volume-preserving flow conserves
> information across context extension. The prediction held.

## Notes on the writing

- **Tone**: technical and honest. Limitations section is unusually
  long for an ML paper, which is a deliberate signal of rigor.
- **Negative results**: the iteration history (v1/v2/v3 eigensheaf
  penalty failures) is included as a substantive contribution.
  Reviewers usually like this.
- **Length**: ~10 pages including references, fits arxiv expectations.
  Workshops often have an 8-page limit; the paper compresses to that
  by removing some background.

## Recommended next steps before submission

1. **Run `make` to verify the PDF compiles cleanly** on your machine.
   Fix any reference / citation warnings.
2. **Add at least one figure**: a long-context perplexity vs. seq_len
   plot would be very impactful as Figure 1. The data is in
   `logs/v3_plasma/metrics.jsonl` and `logs/v3_vanilla/metrics.jsonl`.
   I can write the matplotlib script if you want.
3. **Get one external read** before submitting. Send to someone you
   trust on Twitter or by email; they'll catch obvious issues.
4. **Decide affiliation**: the paper currently lists "Independent /
   Roko.Network". Adjust if you want a different affiliation listed.
5. **Pick the arxiv categories** carefully: cs.LG primary, cs.CL
   cross-list, math-ph optional.

When you're ready to submit, run:

```bash
make arxiv
```

…and upload the resulting tarball. That establishes the arxiv timestamp
in addition to the GitHub git history timestamp.
