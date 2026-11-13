# Mirrored 1k run with explicit delay phase

Launch in the `vcsel` environment from the repository root:

```bash
python -m rl.rl_api.experiments.train_mirrored_delay_1000
```

The file also runs as one cell, like `train_mirrored_1000.py`. Importing it
prints a preview and does not start training. This setup has not launched the
full training run.

The training configuration matches `train_mirrored_1000.py` except for
`ModelConfig.phase_parameterization="target_delay_residual"`. The run has its
own name and directory:

`rl/rl_api/runs/star_rms_bins5_randbands5_phase_delay_residual_anneal1to1000_lr900_standard_lk_mirrored_1k/`

The phase mean on each directed edge is

```text
learned residual + target_source - target_receiver
                 - 2*pi*f_reference*tau_effective
```

`f_reference` is the lowest mean-centered input detuning in Hz. The existing
four input features already determine it, including for asymmetric detuning
spacings. `tau_effective = int(tau/dt)*dt = 0.999 ns` matches the current
simulator's discrete delay. The offset is applied before Gaussian sampling
and scoring, and checkpoint configuration records the new convention for
validation and inference. There are no additional weights, features, reward
terms, feedback simulations, or frequency-estimation networks.

The run starts with fresh seed-7 weights. Its 1,000 iterations, sizes 3–9,
0–4 GHz span sampling, topology sampling, mirrored action pairs, pair-excluded
baseline, exploration annealing, learning rates, PCGrad settings, physical
parameters, validation, plots, and 14 workers match the mirrored run.

The launch-only exception is `VERIFY_REFERENCE=False`: the historical
reference hash expects the old trapezoid integrator, before the reviewed
delayed-stage fix. The old manifest and original mirrored launcher are
unchanged. This switch does not change sampling, optimization, or simulation.

Regression tests compare the complete old and new training configurations,
initial weights and RNG state, all-size first batches, mirrored Gaussian
scores, the exact-mean endpoint, checkpoint/inference round trips, and a
three-update pipeline with test physics. They also check the sign and units
against the delayed coupling-phase argument.
