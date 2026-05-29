# The Math Audit

*Or: how I publicly killed seventy percent of my own neural architecture and made the remaining thirty percent visible.*

---

Most researchers protect their dead work. The instinct is correct in one direction and wrong in another. Correct: the dead work is private. Wrong: the wrong dead work stays buried, takes up shelf space, and silently corrupts the credibility of the living work next to it.

I built thirty neural networks over twelve months. I'm now telling you, publicly, that twenty-five of them were decoration. Not a confession. A clarification — because the five that weren't are now harder to see in the right way until the twenty-five are out of the room.

This is what an honest audit looks like.

---

## The reframe

The question is not *did the architecture work*. The question is *did the math run*.

These are not the same question. An architecture can train, converge, output reasonable token probabilities, and still contain operators whose names promise mathematical content that the code does not deliver. A `SheafHodgeDiffAttention` layer can run for ten thousand steps with `sheaf_loss` literally equal to zero because the consistency check was multiplied by a constant that was never set. A `KoopmanSpectral` module can update its parameters with a learned matrix that is not estimating any Koopman operator, in any reproducible sense, because the lift basis is too thin to satisfy convergence. A `PersistentHomologyProbe` can output a scalar that has nothing to do with persistent homology because there is no Vietoris-Rips complex anywhere in the forward pass.

The model still trains. The loss still drops. The name on the file still says *Hodge*.

This is the central failure mode of the field's recent decade, and it is a failure mode I produced, repeatedly, at scale, in private, with full awareness of the mathematics that should have been computed and full failure to actually compute it. I am not an outlier. I am the canary.

## What the audit found

I read every forward method in every Φ-family model I had built. The criterion was simple. For each named operator, *does the gradient flow through the mathematical object the name promises, or does it flow through a generic op with a descriptive label?*

The results were unflinching.

**Honest, load-bearing math (the surviving thirty percent):**

- The Iwahori-Hecke algebra at $q = \phi$, eleven generators, R1 + R2 relations verified within $10^{-5}$. Real algebraic action on the head dimension. *Hecke attention is real.*
- The Marsden-West variational integrator. Real Störmer-Verlet leapfrog with discrete Euler-Lagrange. Real symplectic flow. *Volume preservation runs.*
- The Koopman operator with EDMD polynomial lift. Real Williams-Kevrekidis-Rowley 2015 construction. Real eigenvalue spectrum via `torch.linalg.eig`. Real spectral gap as a differentiable scalar. *Buried inside a retired architecture. Still works.*
- The Williams-Beer partial information decomposition with Gaussian multi-information. Real synergy estimator via Bertschinger MMI. Real Cholesky-stabilized log-determinant. *Buried inside the same retired architecture. Still works.*
- The Pisano-period closed-orbit state-space model on $\mathbb{Z}/47$. Real modular arithmetic. Real Gumbel-softmax surrogate for training. *Real exact periodicity.*

**Decorative, name-promising-nothing math (the seventy percent):**

- Eleven variants of `PHI_GENESIS` whose forward methods contain `try: from portal_memory import ...` and pass through silently when the import fails. Architecture that does not load is architecture that does not exist.
- A `YonedaMorphismEmbedding` that is a `Linear` layer with a categorical lemma in the docstring.
- A `ConsciousnessMEVOptimizer` that is a regression head trained on crypto features.
- A `PhiCoherentEmbedding` whose SVD-based spectral cap is applied at initialization, after which the model learns away from it and the "coherence" is gone by step five hundred.
- A `TropicalAttention` whose max-plus operation is post-composed with a softmax, which kills the tropical algebra entirely.
- A `PortalLaplacianMemory` whose portal physics is narrative, whose Laplacian routing is honest, and whose name overpromises by a factor of three.

Some of these are subtle. Most are not. All of them passed code review by me, were committed to git, were trained at least once, and were named after deep mathematical structures whose load-bearing computation does not occur.

## The inversion

You are expected, in a public post about your own retirement of work, to apologize. The expected register is contrition.

This is the wrong register.

The right register is *clarification*. I did not waste twelve months. I built thirty neural networks. The audit revealed which five contain rigorous mathematics. The audit also revealed *why* the other twenty-five contain decorative naming — and the why is something I now know about my own pattern, about the field's pattern, about the structural pressure to produce mechanisms that sound impressive and the structural absence of pressure to verify that they compute.

What the field calls "novel architecture" is, in a clear majority of cases, *generic operations under descriptive labels*. This is testable. The test is: read the forward method, ask whether the gradient flows through the named structure or through `Linear` and `Attention` and `MLP` with the named structure as flavor text. The vast majority of "novel" architectural papers from 2022 onward will fail this test. The vast majority of my own work failed this test.

Killing the failures is not contrition. It is restoring the signal.

## What survives

Five things. I have written code for all five. The repository is public.

**Hecke-Eigensheaf attention.** From Φ-LOGOS. Verified R1 + R2 at $q = \phi$. Eleven generators acting on the head dimension. The algebra is real and the gradient flows through it.

**Symplectic-Hamiltonian token flow.** From Φ-PLASMA-CORE. Each transformer layer is one Störmer-Verlet leapfrog step under a learned Hamiltonian $H(q, p, a) = \tfrac{1}{2} p^\T M^{-1} p + V(q, a)$. The forward pass is volume-preserving by Liouville. The forward pass is *measurably* volume-preserving — $|\det J| \approx 1$ across the leapfrog. Empirically, the architecture's perplexity drift across context-length doublings is $2.31\times$ slower than vanilla. The mathematical prediction held.

**Koopman EDMD probe.** From PHI_AEON L7. Polynomial lift on a learned latent. Exact eigenvalue diagnostics. Spectral gap and Lyapunov as differentiable scalars. This is rigorous mechanistic interpretability. Nobody knew I had built it because it was buried under twelve layers of decoration. It is now in `phi-plasma-core/src/phi_plasma/koopman_probe.py`. Apache 2.0.

**Williams-Beer PID synergy probe.** From PHI_AEON L12. Gaussian multi-information via Cholesky log-determinant. Bertschinger MMI synergy estimator. Differentiable, low-overhead, MPS-compatible. Also rigorous mechanistic interpretability. Also buried, also now extracted, also Apache 2.0.

**Sheaf edge consistency with Fibonacci-skip topology.** From the φ-stack world model. Cellular sheaf over the token graph with chain + $F_3, F_4, F_5, F_6$-skip edges. Per-edge-type restriction maps. Real geometric content; the $\mathbb{Z}[\phi]$-module norm framing was decoration and is dropped, but the topology and the consistency loss are real.

I am building one model, Φ-CONCENTRATE, that combines all five with the existing Φ-Plasma-Core backbone. The five together fit in twenty megabytes of parameters. They tell a coherent architectural story. They are now actually visible.

## The audit pattern

The methodological move is reproducible. I describe it in the hope that others will apply it to their own retired work.

Take every model file in a retired or stalled project. For each named operator in the forward method, ask one question: *does the gradient flow through the mathematical object this operator's name promises, or does it flow through a generic op with a descriptive label?* Three possible verdicts. **Honest**: the math runs. **Partial**: the math runs but the name overpromises. **Decorative**: the name is a label on a generic op.

The verdict is mechanical, not aesthetic. You read the code. You note the operations. You ask whether anyone could replicate the named mathematical structure from the code alone, without the docstring. If they could, the operator is honest. If they could not, the operator is at most partial.

Apply this to ten files. You will find at least one rigorous fragment that someone else could use, that you forgot you built, that was buried because the architecture around it was not honest enough to survive its own training run.

Extract the fragment. Document it. Release it.

This is what the field needs more of, and what I needed to do publicly, here, so that everyone in the community who follows my work knows that the remaining five mechanisms are different in kind from the twenty-five that came before them.

## The coda

A retired project is not a failure. A retired project is a workshop.

You walk through the workshop after closing it down. Most of the tools were costume jewelry. A few are real instruments. The instruments were sitting next to the costume jewelry the entire time, and the audit is what tells them apart.

I am closing the workshop. The instruments are on the shelf. They are labeled honestly. They are usable. The address is in the footer.

*Look.*

---

**Code, papers, full math audit log, intervention history, training curves:** https://github.com/Prime-007-hash/phi-plasma-core

**Contact / collaboration:** GitHub issues; Twitter [@data_adept](https://twitter.com/data_adept); Telegram @BASED_ROKO_PRIME.

— PRIME
