# Sparse graph prototype

This is a separate experiment. It reuses the existing physical simulator,
scale-aware context encoding, tolerance curriculum, reward, and PCGrad utility.
Existing models and checkpoints are unchanged and are incompatible with this
new policy. Training starts from scratch.

## Graph and q

Nodes are sorted by detuning (with stable tie handling at inference). The fixed
directed backbone is `0 -> 1 -> ... -> M-1`. It is weakly connected, with a root
that reaches every node; it is not strongly connected. Connectivity alone does
not guarantee synchronization or feasibility for any requested target.

Each receiver considers up to C sources using cyclic sorted-rank offsets
`+1, -1, +2, -2, ...`, excluding self and the incoming backbone edge. Thus the
wrap includes links between detuning extremes. This simple candidate family
restricts what topologies can be learned; no arbitrary all-pairs search is done.
Defaults: K=4 optional edges per receiver, C=8 candidate sources per receiver.

Let `b_i = min(K, candidate_count_i)` and `B = sum(b_i)`. The exact edge count is:

    E = M - 1 + floor(q * B + 0.5)

Cumulative rounding allocates this budget deterministically across receivers:
`k_i = floor(q * cumulative_b_i + 0.5) - floor(q * cumulative_b_previous + 0.5)`.
The learned scores choose which `k_i` sources to use. q=0 yields exactly M-1
edges; q=1 yields the maximum sparse budget. At M=1000 the defaults give 999 to
4999 edges, rather than the old possible maximum of 999000. The K bound is on
optional incoming edges, not outgoing edges. Equal-count allocation is another
deliberate prototype restriction.

The GNN performs message passing only on the chain plus candidate graph, with
actual-degree mean aggregation. A pooled global embedding coordinates edge
outputs. The M features are bounded inverse-degree features. After selection,
the shared decoder predicts continuous coupling magnitude and phase per edge.
For the configured incoming-coupling normalization, magnitudes are bounded by
`maximum_kappa_per_ns / (b_i + backbone_indegree_i)` at receiver i. This preserves
the maximum incoming sum without shrinking rates as 1/(M-1) for a sparse graph.

At inference, q uses its latent Gaussian mean and topology uses deterministic
categorical argmax at each selection step. It produces one design without
simulation or candidate search. `sorted_to_original` maps output node indices
back to the caller's input order. T remains the allowable maximum phase error
in degrees, not a coupling budget.

## Learning and stages

Training uses the existing one-step REINFORCE method, with PCGrad across M.
Optional edges are sampled categorically without replacement (at most K rounds
parallel over receivers). The sum of their conditional log probabilities is
included alongside the latent Gaussian q and selected-edge kappa/phase log
probabilities. The fixed backbone has no topology action. Shared kappa/phase
standard deviations keep the prototype small. Latent entropy is logged as a
diagnostic; no new entropy reward or sparsity tradeoff weight is introduced.

Default run (zero-based update indices):

| Updates | Curriculum | Objective |
|---|---|---|
| 0–99 | maximum span 0.1 GHz, q=1 | phase reward |
| 100–599 | maximum span ramps to 10 GHz, q=1 | phase reward |
| 600–1099 | full span range, q=1 | phase reward |
| 1100–1399 | mild sparsity; q>=0.8, T=1–3 degrees | existing tiered reward |
| 1400–2199 | q lower bound falls to 0; T ceiling grows to 60 degrees | tiered reward |
| 2200–2999 | full sparse/tolerance range; exploration narrows | tiered reward |

As in the source run, spans are sampled within the curriculum ceiling. Training
uses M=3–10, 14 contexts/update, 72 samples/context and 8 maximum-budget phase
replay samples during sparse stages. The term "dense replay" in the inherited
config now means maximum sparse budget. All rewards/schedules come from the
existing helpers. Deterministic held-out automatic and q=1 results are both
reported. The best checkpoint resets at each stage transition: phase score
during warmup, automatic tiered reward during sparse stages. Checkpoint filenames
include a distinct prototype suffix. JSON progress is under its own directory.

## Scope and commands

Only the guarded adapter constructs dense matrices, for M<=100 training,
validation, and optional passivity testing. M>100 supports inference and topology
reporting only. No sparse simulator has been implemented. An M=1000 timing is
an inference benchmark, not evidence of a working or accurate physical design.
Generalization from M=3–10 still needs future dynamic validation.

Run from the repository root, in the environment that has this project's dependencies:

```sh
python -m rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_sparse_graph_scale_aware --smoke-test
python -m pytest tests/test_sparse_graph_prototype.py -q
python -m rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_sparse_graph_scale_aware --train
python -m rl.archive.inspect_saved_coupling_model_variable_m_sparse_graph_scale_aware
python -m rl.archive.inspect_saved_coupling_model_variable_m_sparse_graph_scale_aware --m 1000 --inference-only
```

The smoke test uses untrained weights. It reports wall time and returned edge
array bytes separately (not peak process memory). The regression test also
audits PyTorch intermediate shapes and allocation growth between M=500 and 1000.
The inspector plots every directed edge for small graphs and bins edge lists
for large graphs without creating an M-by-M matrix. It reuses dynamics/far-field
plots for small M, and exposes `passivity_test(design, reflectivity, round_trip_time_fs)`.
The same mirror calibration assumptions as the existing inspector apply.
