# Learned-tree scale-aware experiment

This is a new checkpoint family. It does not load or modify a previous dense
or fixed-backbone model. Training is **not** started by importing the files.
Running the training file as a Jupyter/VS Code cell uses `NOTEBOOK_ACTION =
"train"`; set that constant to `"smoke-test"` for an inference-only check.

Run from the repository root:

```bash
python -m rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_learned_tree_scale_aware --smoke-test
python -m rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_learned_tree_scale_aware --train
```

Inspect the current checkpoint after training has begun:

```bash
python -m rl.archive.inspect_saved_coupling_model_variable_m_learned_tree_scale_aware --m 10
python -m rl.archive.inspect_saved_coupling_model_variable_m_learned_tree_scale_aware --m 1000 --inference-only
```

The inspector's editable constants include target clusters, detuning span,
manual detunings, allowable phase error, and optional manual q. The model
outputs q, a learned root, and selected directed edges with their coupling
strengths and phases. Node indices refer to detuning-sorted order; the result
also contains `sorted_to_original`.

At q=0 the policy constructs a learned, weakly connected M-1 edge tree. At
q=1 it returns all M(M-1) directed links. Between those endpoints, the edge
count is `round((M-1)/(1-q*(M-1)/M))`; the count grows as O(M) for fixed q<1.
The reward is the existing tolerance-first reward and still ranks feasible
designs by q. Because the q-to-count mapping changed, q is no longer the
linear fraction of optional links used by the older models.

The training schedule is M=3..10, 100 iterations at 0.1 GHz span, 500
iterations ramping to 4 GHz, 500 more full-span phase iterations, then the
existing mild/full/refinement sparse stages through iteration 3000. The
existing simulator is reused for training and for inspection at M<=100; no
large-M dynamic simulation is claimed. The M=1000 smoke test checks topology
inference only and says nothing about physical solution quality.

The run saves one updated two-panel progress PNG directly in
`rl/artifacts/results/`, as well as the usual current/best/final checkpoints.
It does not write a progress JSON file.
