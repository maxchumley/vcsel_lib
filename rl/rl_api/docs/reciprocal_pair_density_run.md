# Density training with one action per reciprocal pair

Start both fresh experiments on Great Lakes after pulling `updates`:

```sh
git pull --ff-only origin updates
bash scripts/submit_reciprocal_pair_density_tests.sh
```

Use `--dry-run` to preview submissions or `--smoke-test` for two iterations.
The launcher submits independent `density_pair_ref` and `density_pair_zero`
jobs, each with 32 CPUs, 31 simulation workers, 64 GB and a seven-day limit.
Five contexts and eight actions remain vectorized per worker job (40 cases).

The recipes are `train_mirrored_symmetric_density_pairs_1000` and
`train_mirrored_symmetric_density_pairs_zeroref_1000`. Both start from fresh
seed-7 weights, with no checkpoint, teacher designs or correction simulations.
They match the preceding symmetric-value density recipes except for the
reciprocal action parameterization and distinct artifact names.

## Changed symmetry parameterization

Previously, two independent directed Gaussian actions were averaged:

```text
u_pair = (u_ij + u_ji)/2
```

Now the existing shared edge decoder supplies one Gaussian per unordered pair:

```text
u_pair ~ Normal(mu_pair, sigma_pair)
u_ij = u_ji = u_pair
```

This applies to both latent strength and unwrapped coupling phase. The sampler
scores each pair once, including pairs subsequently pruned. The decoder checks
exact equality instead of averaging. The physical strength transform, budget
normalization, connected spanning-tree selection and redistribution are unchanged.

For a minimal change, use the existing edge head's canonical orientation
(receiver index smaller than source index after sorting by detuning). Its output
is copied to the opposite direction; that direction's separate head output is
unused. The network layers, four node inputs, message passing, edge/budget heads
and parameter count are unchanged. Tied detunings use the existing sorted index
order. No new pair embeddings or labels are introduced.

The pair phase mean is the learned residual plus the common reference rotation:

```text
phi_pair_mean = residual_pair - 2*pi*f_reference*tau_effective
```

Laser-1 reference uses the lowest mean-centered detuning. Zero reference sets
that offset to zero. Antisymmetric source-minus-receiver target offsets are not
added: they canceled under the previous reciprocal average. The target phases
still enter the node and edge features and the simulation reward.

Ordinary pair SD and extra-wide phase SD are divided by sqrt(2), matching the
executed width of averaging two equally wide independent directed draws. This
is an approximate match if the old two SD predictions differ. Budget exploration
is unchanged. Mirrored candidates still use opposite perturbations around one
mean, with the existing pair-excluded baseline.

## Unchanged experiment

- Sizes 3–9; random phase targets and detuning distributions with spans up to 4 GHz.
- Five contexts per size, five density requests per context, eight mirrored actions:
  1,400 training simulations per iteration.
- 1,000 iterations; exploration anneals to exact means; final mean batch skips updating.
- Native standard-LK simulation, 500 ns, original time step, delay, ramp and history.
- Wrapped RMS phase reward, PCGrad grouping, learned budget and learning-rate schedule.
- 448 held-out simulations at iteration 1, every 50 iterations and the final iteration.
- Two-pane progress plots at iteration 1, every 10 iterations and the final iteration.

The unique progress files are `density_pair_ref_progress.png` and
`density_pair_zero_progress.png`, under their separate Great Lakes run/job
folders. The existing automatic plot sync discovers both without restarting.
Old checkpoints retain the original projection and scoring behavior; the new
opt-in `ModelConfig.reciprocal_pair_actions=True` is saved with each new model.
Validation and deterministic/stochastic inspection use the same pair policy.
This experiment tests the parameterization; it does not guarantee convergence.
