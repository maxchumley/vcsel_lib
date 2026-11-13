# Reinforcement-learning coupling design

For the current configuration, training, proposal and inspection workflow,
start with the [RL API](rl_api/README.md). The latest root-encoder experiment
is `rl_api/experiments/train_mirrored_root_encoder_1000.py`. The files below
are the legacy numerical backend and standalone research entrypoints; the
API reuses their tested simulation and learning routines.

## Legacy standalone workflow

The active scripts train and inspect the Bernoulli sparse coupling policy.
Run them from the repository root:

```bash
python -m rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse
python -m rl.inspect_saved_coupling_model_variable_m_bernoulli_sparse
```

The trainer and inspector use the shared modules beside them. Checkpoints,
plots, and saved arrays are under `rl/artifacts/` and are ignored by Git.

The prepared run starts a fresh policy and uses suffix
`_star_rms_pairs5_topo4_wide180_3k`. It trains M=3–9 for 3,000 iterations.
Each size receives five target/detuning pairs, each reused across four
topologies, with ten actions per topology: eight local and two with wider phase exploration
(1,120 local + 280 wider trajectories per iteration). Wider samples add 180°
of phase standard deviation in quadrature with each learned local width.
That extra width cosine-anneals from 180° to zero over printed iterations
1,000–1,500. After iteration 1,500 all ten actions use local exploration;
strength and total-budget sampling retain their ordinary widths. Log probabilities use each action's
actual sampling width. Twenty evenly spaced topology densities cover star to
dense in every iteration, including both endpoints. They are split into four
neighboring density bands, with one mask from each band assigned to every pair.
Masks and assignments within each band are randomized each iteration. Five
detuning-span strata cover 0 to 4 GHz (0.8 GHz bins); every topology band sees
the identical five target patterns and detuning vectors. Both curricula and
the DE teacher are disabled. The lowest-detuned laser
retains its outward star backbone in every topology.

PCGrad uses **four groups of five shared pairs per size**, giving 28 tasks.
Each task includes 50 sampled actions. Small arrays repeat discrete link counts;
the sampler's density-band identities keep five pairs in every task even when
equal link counts span bands. Detuning
does not define additional PCGrad tasks. Candidate-fraction weights preserve
the equal-M objective before clipping and projection. Each task is clipped to
norm 1 before projection; the combined gradient is also clipped to norm 1.

Fourteen workers share 28 jobs, each integrating five same-size pairs ×
ten actions for one topology band in one vectorized call. There are 35 unique
pairs and 140 pair/topology contexts per iteration, totaling 1,400 simulations.
Rewards are centered separately within each ten-action topology block;
different topologies do not share a reward baseline. Training uses
`1 - (0.5*mean(e_i) + 0.5*max(e_i))/180`, where `e_i` is the nonreference
laser's wrapped phase-error RMS in degrees over the settled tail. One global
conjugate orientation is chosen for the whole pattern. Cosine phase reward
remains a separate evaluation metric. Held-out RMS errors and the fraction
with every nonreference laser within 3 degrees RMS are logged and saved with
the history every 50 iterations. Best checkpoints are selected by held-out RMS
reward. In the progress figure, size and density curves remain cosine rewards;
the additional overall RMS-reward curve has its own label.

Ordinary exploration bounds also anneal over printed iterations 1,000–1,500.
Their final bounds are held for refinement through iteration 2,000. The
learning rate is 1e-4 through iteration 2,000, then decays exponentially to
1e-5 at iteration 3,000. Run-script stage constants use printed one-based
iterations; their config mappings account for the shared helpers' internal
index conventions. No training is started by importing either script.

The independent-context 180° run remains under suffix
`_star_rms_strat20_pcgrad5_wide180_3k`. The 15° exploration run remains under suffix
`_star_rms_strat20_pcgrad5_wide15_3k`; the earlier RMS run remains under
`_star_rms_strat20_pcgrad5_3k`. Their checkpoints and histories are preserved.
The next run starts fresh and
writes to the new suffix. To inspect it after starting training, set
`MODEL_FILE_SUFFIX = TRAINING_MODEL_SUFFIX` in the inspector; its current
manually selected checkpoint family is retained.

The completed 5,000-iteration run remains under suffix
`_star_wide_phase_topology_pcgrad_no_de`. In the inspector, use
`MODEL_FILE_SUFFIX = PREVIOUS_MODEL_FILE_SUFFIX` to inspect it. User backups
in `rl/BACKUP` are not modified. Older checkpoints retain cosine reward and
their original sampling/grouping through backward-compatible config defaults.
Set `PCGRAD_TASK_GROUPING = "array_size"` for size-only grouping, or use
`"array_size_topology_density"` for the legacy sparse/middle/dense thirds.
The console prints the active task count and fraction of conflicting gradient
pairs; these diagnostics are also available in the returned training history.

Previous experiments and optional plotting scripts are in
[`archive/`](archive/README.md). Run them with `python -m rl.archive.<module>`.
