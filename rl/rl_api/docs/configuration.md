# Changing settings

Edit the explicit `CONFIG = TrainingConfig(...)` block in
`experiments/train_current.py` for RL settings, and its preceding native VCSEL
parameter cell for physical settings. Edit the top settings block in
`experiments/inspect_current.py` for inspection. These files list all public
settings, including their current values and units. Copy a run file for a
different experiment; no JSON editing or preset selection is needed.

The same Python objects can be used interactively. Configuration
classes for RL settings are ordinary frozen dataclasses; `dataclasses.replace()` makes a
modified copy without changing another experiment.

```python
from dataclasses import replace
from rl.rl_api import Experiment, TrainingConfig

config = TrainingConfig()
config = replace(config,
    optimization=replace(config.optimization, workers=14),
    schedules=replace(config.schedules, wide_phase_actions=2,
                      wide_phase_extra_sd_deg=180.),
)
experiment = Experiment("paired_api_02", config=config)
experiment.describe()
```

The run file's objects are the runtime settings. `config.py` defines their
types, validates them, and maps them into the existing numerical engine.
`presets/current_training.json` is read only by comparison diagnostics, not by
the training or inspection workflow. Ordinary Python defaults still allow
short interactive examples such as `Experiment("example")`.

## Public controls

| Section | Fields | Mapping / interpretation |
| --- | --- | --- |
| `TrainingConfig` | `sizes`, `seed`, `device` | training M values, seed, Torch device; CPU reproduces current run |
| `model` | `node_hidden_sizes`, `node_embedding_dim` | shared node encoder |
| `model` | `message_hidden_size`, `message_passing_steps` | GNN message width/depth |
| `model` | `edge_hidden_sizes` | shared continuous edge decoder; default `(128,128)` |
| `model` | `normalize_incoming_by_degree` | normalize the decoder's incoming strengths by degree |
| `model` | `softplus_strength_scale_per_ns`, `initial_budget_per_laser` | raw edge-strength scale and initial log-budget scale |
| Native `phys` | `Tmax`, `dt`, `tau` | simulation duration, timestep, delay in seconds |
| Native `phys` | `tau_p`, `tau_n` | photon and carrier lifetimes in seconds |
| Native `phys` | `g0`, `N0`, `s`, `q`, `beta` | same physical parameters as vcsel_lib |
| Native `phys` | `I`, `alpha`, `noise_amplitude`, `save_every` | pump current in A, linewidth factor, noise, saving stride |
| `TrainingConfig` | `ramp_start_delays`, `ramp_rise_delays` | coupling ramp timing in delay units |
| `topology` | `mode="provided"`, `backbone="rooted_star"` or `"random_reachable"` | star-backed masks or uniform-root random link growth until every laser is reachable |
| `topology` | `minimum_extra_link_fraction`, `density_bias` | minimum optional-link density and sampling bias |
| `sampling` | `pairs_per_size` | targets per M and number of detuning-span bins |
| `sampling` | `topologies_per_pair` | paired masks and number of topology PCGrad bands |
| `sampling` | `actions_per_topology` | Monte Carlo actions per supplied mask |
| `sampling` | `detuning_span_ghz` | maximum full detuning span |
| `sampling` | `pairs_per_job`, `actions_per_job` | target and candidate chunk sizes for simulation workers |
| `sampling` | `structured_target_fraction`, `structured_target_jitter_rad` | optional structured target mixture |
| `sampling` | `reuse_masks_across_pairs` | False reproduces twenty per-pair mask draws; True shares one star plus three band masks across all five pairs |
| `sampling` | `topology_sampling_mode` | `paired_contexts` assigns a density pool; `per_target` uses BACKUP's per-pair levels; `band_random` draws five independent density slots per pair, using the chosen backbone rule |
| `optimization` | `iterations`, `workers` | run length and spawned process count |
| `optimization` | `learning_rate`, `final_learning_rate` | initial and final Adam LR |
| `optimization` | `per_task_clip`, `global_clip` | gradient-norm limits before PCGrad and after combining |
| `optimization` | `pcgrad_grouping` | `topology_strata` uses topology rank groups; `density_bands` uses BACKUP's physical sparse/middle/dense bands and skips empty bands |
| `optimization` | `reward_std_ema_decay`, `budget_calibration_targets` | running advantage scale and initial budget-reference sample count |
| `schedules` | `ordinary_start`, `ordinary_end` | ordinary SD annealing boundaries, printed iterations |
| `schedules` | `wide_start`, `wide_end` | cosine taper of extra raw-phase SD |
| `schedules` | `wide_phase_actions`, `wide_phase_extra_sd_deg` | wider actions per mask and their extra phase SD |
| `schedules` | `learning_rate_decay_start` | last printed iteration at initial LR |
| `schedules` | `learning_rate_decay_end` | Optional printed iteration at final LR; later iterations hold that LR. None decays through the last iteration |
| `schedules` | `precision_start`, `precision_end` | Optional final refinement after ordinary annealing; None disables it |
| `schedules` | `precision_minimum_log_std`, `precision_maximum_log_std`, `precision_budget_std` | Final ordinary SD limits and nominal log-budget SD; the existing learned budget multiplier is retained |
| `schedules` | `initial_log_std`, initial/final minimum/maximum log SD | decoder initialization and ordinary exploration bounds |
| `validation` | `every`, `pairs_per_size` | validation frequency and fixed bank size per M |
| `validation` | `plot_every`, `notebook_display` | progress figure interval and inline notebook updates |
| `validation` | `plot_dpi`, `plot_use_tex` | training figure/export resolution and LaTeX serif font; both run files select 300 DPI and True |
| `reward` | `tail_fraction`, `worst_laser_weight`, `allow_global_phase_conjugate` | scoring window, worst-laser weight and global conjugation allowance |

Quick keywords on `Experiment` are `sizes`, `pairs_per_size`,
`topologies_per_pair`, and `actions_per_topology`. They override the matching
configuration fields. `.train(iterations=..., workers=...)` provides optional
overrides for explicit short checks. Reducing iterations does not rescale the
exploration stage boundaries. The LR decay ends at the configured final
iteration; runs ending before its start never enter that stage.

## Exact count mapping

Let S be the number of M values, P pairs per M, G topologies per pair, and A
actions per mask. The backend receives:

```text
targets_per_batch       = S * P
candidates_per_target   = G * A
actions_per_topology    = A
pcgrad_topology_groups  = G
simulations             = S * P * G * A
PCGrad tasks            = S * G
```

Each topology band sees all P pairs. The number of detuning bins is P, rather
than an independent control that could silently leave bins uncovered. These
are bins of the **full span**, not bins that restrict each laser to a different
absolute frequency. The existing sampler generates the distribution inside
the chosen span. Topology density sampling and its random ordering are the
same existing functions and RNG sequence.

For the next run, `reuse_masks_across_pairs=True` uses the **same matrices**
across pairs. It draws one exact star and one mask per remaining excess-density
band once per M/iteration, then broadcasts those matrices over P rows. This
changes the sampling sequence and star exposure; all numerical settings remain
as specified in the run file. See [shared-mask run](shared_masks_run.md).

With defaults, jobs contain 5 pairs × 10 candidates = 50 simulations. There
are 28 jobs, dispatched largest M first, over 14 processes. Chunk sizes affect
dispatch; context/action counts determine the number of physical cases.

## Schedule indexing

Public schedule values are printed one-based iteration numbers. The original
helpers use zero-based indices with slightly different boundary arithmetic.
The mapping deliberately retains those offsets:

| Backend field | Public mapping |
| --- | --- |
| `minimum_log_std_anneal_start_iteration` | `ordinary_start - 2` |
| `minimum_log_std_anneal_iterations` | `ordinary_end - ordinary_start + 1` |
| `exploration_anneal_start_iteration` | `ordinary_start - 1` |
| `exploration_anneal_iterations` | `ordinary_end - ordinary_start + 1` |
| `phase_exploration_start_iteration` | `wide_start - 2` |
| `phase_exploration_end_iteration` | `wide_end - 1` |
| `phase_exploration_fraction` | `wide_phase_actions / actions_per_topology` |
| `phase_exploration_start_std` | degrees converted to radians |
| `learning_rate_decay_start_iteration` | `learning_rate_decay_start - 1` |

The unchanged helpers control the precise floating-point values at stage
boundaries. Default SD bounds progress from `[-1.5,0.75]` to `[-3,-1.5]` in log
space. The two wide actions use `sqrt(local_sd**2 + extra_sd**2)` on raw phases.
The fraction still reserves those action positions after annealing, but with
zero extra width they use the ordinary distribution.

## Rewards

For each nonreference laser, e_i is the temporal RMS of wrapped target phase
error over the settled tail. With degrees as the error unit:

```text
R_rms = 1 - (0.5 * mean(e_i) + 0.5 * max(e_i)) / 180
```

The tail is the last 35% of saved samples, restricted to times after the
coupling ramp. A single global conjugate orientation can be selected for the
whole candidate. This is not a per-laser sign choice. Success requires every
nonreference laser's RMS error to be at most 3 degrees. Nonfinite tail states
receive reward -1.

RMS is the learning objective and best-checkpoint criterion. Cosine remains a
monitoring metric and the current inspector's `best` selection criterion.
`maximum_mean_phase_error_deg` measures circular-mean phase error and can be
small even when the temporal RMS is large. For multiple noise realizations,
reported rewards are averaged, while `maximum_phase_rms_deg` takes the worst
realization and success requires all realizations to pass.

## Advanced backend settings

`ModelConfig.phase_parameterization` defaults to `"absolute"`, preserving
existing runs. The opt-in `"target_residual"` mode adds the encoded source-minus-
receiver target difference to the existing phase mean, before sampling and
log-probability calculation. It adds no weights and retains the same phase
SDs, inputs and strength/budget branches. Checkpoints record the choice for
consistent validation and inspection. See the [phase-residual experiment](phase_residual_run.md).

Pass `vcsel=VCSEL(phys)` to `Experiment`. It snapshots shared scalar parameters
from `.phys`; `config.phys` stores that native-key snapshot. The internal
compatibility boundary translates to the old worker's names without changing
units. The template's M, detunings and coupling arrays are replaced per batch.
The current worker supports coupling=1 and self_feedback=0. Select VCSEL's
phase convention with one entry in the native `phys` dictionary:
`"phase_model": "standard_lk"`. The other supported choices are `ma_2019` and
`chumley_2026`; omitting the entry retains the historical `chumley_2026` default.
Training workers, validation, evaluation and saved-model inspection preserve
the choice. VCSEL owns the equations, scaling and free-running history.
Unsupported native options raise an explicit error. `N_bar` is allowed as in the
simple example; VCSEL computes its own free-running equilibrium during scaling.

`experiment.backend_config` exposes the complete numerical dataclass for
inspection. It is constructed from the Python configuration objects, fixed
algorithm choices, and a small inactive legacy-compatibility block. This version
does not expose switches for inactive DE teaching, learned gates, alternate
curricula, or alternative objectives as supported public modes. Those require
implementation/validation changes, rather than editing the historical JSON.
