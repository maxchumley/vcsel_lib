# Mirrored training with random root-reachable graphs

Run the single cell in `experiments/train_mirrored_reachable_1000.py`, or run
this command from the repository root:

```sh
python -m rl.rl_api.experiments.train_mirrored_reachable_1000
```

This starts a fresh run in
`runs/reachable_rms_bins5_randbands5_phase_delay_residual_anneal1to1000_lr900_standard_lk_mirrored_1k/`.
Importing the module only previews its settings. The previous experiment and
the inspector's checkpoint selection are unchanged.

The only training configuration difference from `train_mirrored_delay_1000.py`
is `TopologyConfig(backbone="random_reachable")`. Physical parameters,
corrected native trapezoid, standard LK phase model, architecture, phase
parameterization, targets, detunings, rewards, sizes 3–9, seed, 1,000 iterations,
mirrored action sampling, pair-excluded baseline, schedules, worker counts,
and batch sizes are identical. The delay-phase reference remains the minimum
mean-centered detuning, independent of the newly selected graph root.

## How each topology is constructed

1. Choose a root uniformly from all lasers, independently for every topology.
2. Randomly order every possible directed pair, excluding self-links.
3. Starting with no links, enable one link at a time until every laser is
   reachable from that chosen root. Recompute reachability through the whole
   enabled graph, including links added before their source became reachable.
4. When a density slot requests more links, continue along the random order
   until that count is met. Every remaining directed pair is eligible,
   including feedback into the root and links between non-root lasers.

There is no inserted star or spanning tree. Cycles and feedback may appear
before the chosen root reaches all lasers. The distribution follows this
growth procedure; it is not uniform over all rooted directed graphs.

Each context retains five independently constructed topology slots: the
first graph satisfying reachability, a graph with a requested count in each
of the previous sparse/middle/dense bands, and an all-to-all graph. Each slot
gets its own root and random edge order. Reachability takes priority over a
requested count, so some sparse requests become denser graphs. PCGrad still
groups by actual physical link counts and skips empty bands: 21 tasks is an
upper bound rather than a guaranteed count. The former 40/20/40 density
exposure is consequently not guaranteed. There are still 1,400 simulations
and 35 jobs per iteration with 14 workers.

## Validation and inspection

Validation keeps the same fixed phase/detuning contexts and deterministic
mean actions, with the new reachability growth rule for its masks. Requested
counts span the original sparse-to-dense levels; connectivity can raise the
actual counts.

Checkpoints record `topology_random_reachable=True`. Loading a new checkpoint
and proposing automatic topologies uses the same growth rule. Manually
supplied masks keep their existing source/receiver convention and override
automatic generation. Older star-backed checkpoints retain their original
sampler and serialization schema. To inspect this run, select its run folder
in `inspect_current.py` after training has written a checkpoint.

Learned gates and lexicographic resource objectives remain disabled, exactly
as in the preceding phase-only run. The inspector's manual budget option does
not affect training.
