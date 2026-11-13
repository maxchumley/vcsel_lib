# Mirrored training with the sampled root's delay reference

Run the single cell in `experiments/train_mirrored_root_delay_1000.py`, or run
this command from the repository root:

```sh
python -m rl.rl_api.experiments.train_mirrored_root_delay_1000
```

This starts fresh in
`runs/reachable_rms_bins5_randbands5_phase_root_delay_residual_anneal1to1000_lr900_standard_lk_mirrored_1k/`.
Importing the module only previews the settings. The existing run and the
inspector's current checkpoint selection are unchanged.

The only training configuration change from
`train_mirrored_reachable_1000.py` is
`ModelConfig(phase_parameterization="target_root_delay_residual")`.
Sizes 3–9, seed 7, random reachable topologies, random optional links,
targets, detuning spans up to 4 GHz, standard LK model, corrected native
trapezoid, 500 ns simulation time, 1,000 iterations, architecture, rewards,
mirrored actions, schedules, worker counts, and batch sizes are identical.
There are still 1,400 trajectories and 35 jobs per iteration with 14 workers.

## Coupling phases

For receiver i, source j, and the sampled construction root r:

```text
phi_p,ij = target_j - target_i - 2*pi*f_ref*tau + learned_residual_ij
f_ref = detuning_r - mean(detunings)             [GHz converted to Hz]
```

Targets use the same encoded/conjugate convention as before. The delay uses
the simulator's discrete delay `int(tau/dt)*dt`, currently 0.999 ns.
The root's input detuning is the reference; the code does not prescribe or
measure the eventual locking frequency. In feedback graphs, that frequency
need not equal the selected root's free-running frequency.

The sampler now optionally returns its actual construction root without
consuming any additional random draws. Every topology retains that root,
including all-to-all graphs or feedback graphs where several lasers can
reach the array. All eight actions for that topology share its reference.
The root is not a new neural-network input and adds no model parameters.

The existing latent Gaussian coordinates and log-probability calculations
are preserved. At physical decoding, enabled phases receive the fixed
translation `-2*pi*(detuning_root - min(detunings))*tau`. Together with the
existing minimum-reference baseline, this produces the root-reference
formula above. A fixed translation has unit Jacobian, so Gaussian scores
and mirrored perturbations are unchanged. Strengths and budgets are untouched.

## Validation and inspection

Training, held-out validation, and `CouplingModel.propose` all retain the
sampled root and apply the same physical phase translation. The checkpoint
records the new phase mode so loading it automatically restores this behavior.
Automatic inspection needs only this run's checkpoint, using the existing
inspector. Its current settings were not edited.

Proposals record zero-based `raw["root_indices"]` in the returned detuning
order and the mean-centered reference in
`raw["phase_reference_detunings_ghz"]`. Automatically generated graphs use
their actual sampled root. Manually supplied masks have no construction-root
metadata: the reference uses the lowest-detuned laser that can reach every
laser in that supplied graph. A graph with a unique root therefore uses it.

Old checkpoints and the previous minimum-reference run keep their existing
behavior. Learned sparsity gates and lexicographic resource objectives remain
disabled, as in the preceding run.
