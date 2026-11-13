# Reserved Bernoulli extension

The current API trains continuous couplings conditioned on supplied topology
masks. `TopologyConfig(mode="provided")` is the only enabled mode. Asking for
`bernoulli_gates` or another topology mode raises `NotImplementedError` so it
cannot silently change the experiment.

When learned gates return, keep the public workflow:

```text
Context -> gate logits ell_ij -> sampled adjacency A
        -> continuous decoder conditioned on A -> strengths and phases
        -> simulation -> reward -> update
```

The key separation is between predicting a topology and predicting continuous
couplings for the realized topology. The continuous decoder must see the
actual sampled adjacency. Predicting both independently from an unmasked graph
would change that dependency.

Proposed ordinary implementation pieces:

1. A gate head returns one logit per optional directed edge.
2. A sampling function keeps the required rooted backbone and samples optional
   edges with `torch.distributions.Bernoulli(logits=...)`.
3. The current graph/edge decoder receives the realized mask.
4. The joint log probability includes optional-gate and conditional-continuous
   terms. Fixed backbone edges contribute no learnable Bernoulli term.
5. Configuration explicitly specifies the sparsity/budget objective and its
   baseline/normalization behavior. These are not inferred from the mask alone.

`DesignBatch.gate_logits` and `.gate_probabilities` are `None` in provided mode.
They reserve understandable outputs for a future gate model; binary supplied
masks are available separately as `.topologies`.

Before enabling this mode, test rooted connectivity, conditional mask use,
likelihood decomposition, gradients reaching both heads, and deterministic
inspection. Keep the provided-topology reproduction tests as a separate
supported recipe. No new gate loss, sparsity penalty, or decoder weights have
been added to the current run.
