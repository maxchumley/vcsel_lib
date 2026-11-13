# Mirrored actions: focused 1,000-iteration comparison

Run the single `#%%` cell in `experiments/train_mirrored_1000.py` to start.
`START_TRAINING=True` is set. Restart the Python kernel first so it imports
the new optional sampler and baseline. The setup itself does not train.
All settings, including native VCSEL parameters, are in that run file.

Run name:
`star_rms_bins5_randbands5_phase_residual_anneal1to1000_lr900_standard_lk_mirrored_1k`

This is a fresh seed-7 experiment. It does not load or resume the baseline.
Its output folder is created when training starts. If this new name already
exists, `RESTART_EXISTING_RUN=True` archives that attempt before restarting.
The independent baseline's run file and output folder are separate.

## What this test changes

Two explicit settings select the new estimator:

```python
sampling=SamplingConfig(
    # ... settings shown in full in the run file ...
    action_sampling="mirrored",
),
optimization=OptimizationConfig(
    # ...
    reward_baseline="pair_excluded",
),
```

For each topology and detuning/target context, sample three ordinary Gaussian
noise vectors and one wide vector. Evaluate both signs of each vector around
the predicted mean: `mean + std*noise` and `mean - std*noise`. The complete
latent strength, coupling-phase and budget vectors are mirrored together.
Each member has the same marginal Gaussian distribution as the independent
baseline. Physical strengths still use the existing decoding/redistribution,
and phases still use the target-plus-residual transform and wrapping.

The two wide actions occupy slots 1–2 (one pair); slots 3–8 contain three
ordinary pairs. Partners share the same topology and Gaussian widths. Both
pair members explicitly share the positive member's predicted mean and SD;
this avoids rejecting harmless floating-point differences between repeated
rows in multithreaded matrix operations. Their Gaussian scores use those
same shared parameters.

All eight slots are drawn before reflection to preserve the baseline's RNG
consumption; positive members are kept exactly. Sampling is detached from
the network and both members use their own Gaussian log probability for
REINFORCE. The positive/negative reward difference is not used as a replacement
score for learning exploration widths.

For pair 1, the reward baseline is `(R3+R4+R5+R6+R7+R8)/6`. Each pair uses
the other six actions from its own topology/context, excluding both itself
and its correlated partner. This centers the learning signal; it does not
add a phase loss or change the RMS reward. The existing global reward-SD EMA
is updated once per batch and scales the resulting advantages.

## Full recipe

| Setting | Value |
| --- | --- |
| Iterations | 1,000: 999 stochastic updates and a mean-only final check |
| Seed | 7, fresh initialization |
| Laser sizes | 3–9 |
| Context pairs per size | 5, one in each 0.8 GHz detuning-span bin |
| Topologies per pair | Star + random sparse + random middle + random dense + all-to-all |
| Intermediate counts | Independent uniform integer counts in each physical density band |
| Wiring | Fresh optional links for each pair and iteration |
| Rooted connectivity | Outward star from lowest-detuned laser 1 enabled |
| Mask reuse across context pairs | Disabled |
| Actions per topology | 8: 3 ordinary mirrored pairs and 1 wide pair |
| Simulations per iteration | 1,400 |
| Workers and jobs | 14 workers, 35 vectorized jobs of 40 simulations |
| PCGrad | 21 size × physical density tasks; 40% / 20% / 40% within each size |
| Weight per size | 1/7 |
| Detuning PCGrad | Disabled |
| Per-task / combined norm limits | 1.0 / 1.0 |
| Learning reward | Wrapped RMS, worst-laser weight 0.5, final 35% of trajectory |
| Global target conjugacy | Enabled |
| Phase decoder | Target phase differences + learned residual |
| Model weights | 127,304 |
| Physics | Native `standard_lk`, 500 ns, zero noise |
| Topology/detuning curricula | Disabled |
| Teacher / measured-feedback loss | Disabled |
| Validation | Exact mean actions every 50 iterations, fixed 64 contexts per size |
| Best checkpoint selection | Mean held-out RMS |
| Plotting | 300 DPI, LaTeX serif font, iteration in title |

## Schedule and outputs

All ordinary strength/phase SDs, budget log SD and the extra-wide 180-degree
phase SD use the baseline's multiplier `(1000 - i)/999` for printed iteration
`i`. Learned initial log-SD bounds are -1.5 to 0.75. There is no extra precision
stage. LR is 1e-4 through iteration 900, then decays exponentially to 1e-5 at
1000. At 1000 the sampler returns exact means without drawing noise; the
optimizer and its momentum are left untouched.

Checkpoints, history and the run manifest record `action_sampling`,
`reward_baseline`, the pair counts and exploration state. The single cell
creates `runs/<RUN_NAME>/` when started. The shared live-viewer image is
`runs/current_training_progress.png`, one folder above the run directory.

Mirroring is a training-batch option. Mean validation/inference is identical
at equal weights. Sampled inference can request any count; it uses the same
marginal Gaussian widths without requiring paired inference samples.

Compare this run with `train_anneal_1000.py` at matching iterations, using
held-out RMS, sparse-phase errors and the 3-degree success rate. The frozen
checkpoint diagnostic found lower gradient variance with mirroring; it did
not demonstrate improved validation after training. Sparse-specific results
were less certain than the overall reduction.

## Implementation

- `exploration.py`: optional mirroring in the annealed sampler; independent
  sampling remains its default.
- `advantages.py`: pair-excluded reward centering with the existing reward scale.
- `config.py`: user-facing settings and pair-count checks.
- `_compat.py`: optional checkpoint schema; older backend settings omit the
  new fields and retain their numerical behavior.
- `training_batch.py`: selects the configured sampler/baseline.

No original RL, BACKUP or archive source is edited. The independent run file
does not enable either new option.
