# Random link counts and exploration taper over 1,000 iterations

Run the single `#%%` cell in `experiments/train_anneal_1000.py` when ready.
`START_TRAINING=True` is already set. This setup does not start training.
Restart the Python kernel before starting so it imports the new optional
exploration schedule. The older run files retain their settings.

Run name:
`star_rms_bins5_randbands5_phase_residual_anneal1to1000_lr900_standard_lk_1k`

All settings are in the run file, including native VCSEL `phys` parameters.
No preset JSON is loaded. A new output folder is created when training starts.
`RESTART_EXISTING_RUN=True` archives an earlier attempt at this same new name
before starting fresh from seed 7; it does not resume an optimizer.

## Sampling

| Setting | Value |
| --- | --- |
| Laser sizes | 3 through 9 |
| Detuning/target pairs per size | 5, one in each 0.8 GHz span bin |
| Detuning span | 0–4 GHz |
| Topologies per pair | Star, random sparse, random middle, random dense, all-to-all |
| Intermediate link counts | Independent uniform integer draws within their physical density bands |
| Wiring | Independently sampled optional links for every pair and iteration |
| Rooted connectivity | Outward star from lowest-detuned laser 1 enabled |
| Reuse masks across pairs | Disabled |
| Shuffled count lists | Disabled |
| Actions per topology | 8, including 2 initially extra-wide phase actions |
| Simulations per iteration | 7 × 5 × 5 × 8 = 1,400 |
| Worker jobs | 35 jobs of 5 pairs × 8 actions, across 14 workers |
| PCGrad tasks | 21: size × physical sparse/middle/dense band |
| Within-size PCGrad weights | 40% sparse, 20% middle, 40% dense |
| Weight per laser size | 1/7 |
| Task/global gradient norm limits | 1.0 / 1.0 |

For six lasers, the five counts are 5, a draw from 6–13, a draw from 14–21,
a draw from 22–29, and 30. Every legal count has positive probability; this
does not guarantee that every count is used within a specified number of
iterations. Held-out validation keeps its fixed bank and broader count coverage.

## Learning schedule

For printed iteration `i`, multiply every Gaussian SD by:

```python
scale = (1000 - i) / 999
```

The multiplier is clipped to [0, 1]. It scales ordinary strength and phase
SDs, budget log-action SD, and the extra-wide phase SD. Scaling is in SD,
not in log SD. The learned ordinary SDs still use their initial bounds
(-1.5 to 0.75 in log units); the maximum permitted actual width decreases
with the common multiplier. The two wide actions retain their slot identities
while their extra 180-degree SD tapers to zero.

| Printed iteration | Action-SD multiplier | Extra-wide phase SD | Nominal budget log SD | Learning rate |
| --- | --- | --- | --- | --- |
| 1 | 1 | 180 degrees | 0.35 | 1e-4 |
| 500 | 500/999 | 90.09 degrees | 0.17518 | 1e-4 |
| 900 | 100/999 | 18.018 degrees | 0.03504 | 1e-4 |
| 999 | 1/999 | 0.18018 degrees | 0.00035 | about 1.0233e-5 |
| 1000 | 0 | 0 | 0 | 1e-5 |

The budget's existing learned multiplier `exp(0.25*tanh(w))` also applies.
The phase SD of a wide action combines its ordinary SD and extra SD in
quadrature, just as in the current model.

Learning rate is constant through iteration 900, then decays exponentially
to 1e-5 at 1000. There is no separate precision-refinement stage.

Iterations 1–999 make stochastic REINFORCE/PCGrad updates using wrapped RMS.
At iteration 1000 all eight actions for each topology are exact action means;
the batch is simulated and recorded, but the optimizer step is skipped.
Adam momentum is also left untouched. This is 999 learning updates plus a
deterministic final check, labelled as a 1,000-iteration run.

The zero-width endpoint never constructs a zero-variance Gaussian in the API
training sampler. The final checkpoint records zero exploration. Requesting
sampled actions from that checkpoint also returns exact means; earlier
checkpoints retain their recorded sampling widths. Best checkpoints may have
been selected before the zero-width endpoint.

## Model, objective and outputs

The model uses the current target-plus-residual phase parameterization,
127,304 parameters, and native `standard_lk` physics. Wrapped RMS is the only
learning reward; cosine remains a monitoring metric. Global phase conjugacy
is enabled, with the same worst-laser weight of 0.5 and tail fraction of 0.35.
No strength conditioning, measured phase-correction loss, teacher, analytic
delay offset, or detuning/topology curriculum is enabled.

Validation uses mean actions every 50 iterations and at the final iteration,
on the same fixed 64 contexts per size. Checkpoint selection uses RMS.
Plots retain 300 DPI and the LaTeX serif font. The live-viewer image is saved
one folder above the run at `runs/current_training_progress.png`.

The optional schedule is implemented in `exploration.py`, the opt-in scaled
policy classes in `models/policy.py`, and the mean-only endpoint handling in
`training_batch.py`. The existing run files do not enable it. No original RL,
BACKUP or archive source is edited.
