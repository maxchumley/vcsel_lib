# Verified phase-feedback teaching test

Run `experiments/train_phase_feedback_1000.py` as one cell, or from the repository root:

```bash
python -m rl.rl_api.experiments.train_phase_feedback_1000
```

Importing the module only previews. Running it starts a fresh model in its
own run directory; it does not modify or continue an existing experiment.

## What stays the same

All settings match `train_mirrored_root_encoder_1000.py` except the correction
settings and run name: 3–9 lasers, 1,000 iterations, five detuning-span bins
over 0–4 GHz, five topology draws per pair, eight mirrored actions per topology,
1,400 ordinary simulations per iteration, RMS reward, density PCGrad, 500 ns
native `standard_lk` simulations, zero noise, and the same action/LR schedules.

Each topology chooses a uniform random construction root and adds arbitrary
directed links until every laser is reachable from that root. Optional links
and feedback into the root remain eligible. The phase reference is that
root's input detuning, and the encoder retains its binary root flag.

## Additional training loss

For each existing size/density PCGrad group:

```text
L_group = L_REINFORCE + weight * L_correction
```

`L_correction` is the mean of `1 - cos(predicted_mean_phase - teacher_phase)`
over enabled links, averaged over accepted teaching cases in the group.
It is zero if that group has no accepted examples. PCGrad combines these
group gradients before the existing global clip and Adam step. The new
run starts with `weight=1.0`, editable in its `PhaseCorrectionConfig`.

The procedure uses the ordinary sampled trajectories already computed:

1. Measure the post-ramp tail used by the RMS reward. Require finite states,
   fitted frequency spread below **1 MHz**, and maximum relative-phase jitter
   about its own circular mean below **1 degree**. These checks do not require
   agreement with the requested target and do not prove asymptotic stability.
2. For each target/topology block, select its highest-RMS-reward locked action.
   There is at most one verification rollout per block, at most 175 per iteration.
3. Set `beta_i = orientation * target_i - achieved_relative_phase_i`, retaining
   the current reward's global conjugate convention. On each active link
   `j -> i`, propose `wrap(phi_p_ij + beta_j - beta_i)`.
4. Re-simulate from a fresh free-running history with exactly the same strengths,
   topology, detunings, duration and physical model. Accept only a still-locked,
   finite correction with strictly higher RMS reward.
5. Use its physical coupling phases as detached teaching targets for the existing
   mean phase decoder. The loss accounts for the root-reference translation;
   disabled edges, strength outputs and Gaussian SDs have no direct teaching loss.
   Shared network features still receive phase-loss gradients.

The additional rollouts reuse the simulation worker pool. Their rewards are
excluded from the ordinary RL advantages and running reward scale. No new
network head or simulator equation is added. Inference retains a single decoder
prediction; validation does not correct or select improved designs using feedback.
The final exact-mean iteration performs no correction rollouts or optimizer update.

## Monitoring

Progress output and saved history include the eligible locked fraction,
verification attempts, accepted teachers, circular correction loss, and mean
accepted RMS reward improvement. Teacher eligibility and loss are training
metrics; held-out accuracy still measures unassisted decoder predictions.
The public settings are saved in `run.json`, and correction settings are also
included in checkpoint `sampling_recipe` metadata.

For a batch launch, pass this recipe to the existing launcher:

```bash
python -m rl.rl_api.experiments.train_batch train_phase_feedback_1000 --preview
sbatch scripts/train_rl_great_lakes.slurm train_phase_feedback_1000 --smoke-test
sbatch scripts/train_rl_great_lakes.slurm train_phase_feedback_1000
```

These commands require the updated code on the cluster. No job is submitted
by setting up the local experiment.
