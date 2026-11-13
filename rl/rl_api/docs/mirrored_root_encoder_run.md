# Mirrored training with the selected root in the encoder

Restart your Python kernel after the API update. Run the single cell in
`experiments/train_mirrored_root_encoder_1000.py`, or execute from the repo root:

```sh
python -m rl.rl_api.experiments.train_mirrored_root_encoder_1000
```

This starts fresh in
`runs/reachable_rms_bins5_randbands5_root_encoder_phase_root_delay_residual_anneal1to1000_lr900_standard_lk_mirrored_1k/`.
The existing completed root-reference run is preserved.

The only public training setting changed from
`train_mirrored_root_delay_1000.py` is `ModelConfig(encode_root=True)`.
Each node encoder now receives:

```text
[cos(target), sin(target), normalized detuning, detuning span, root flag]
root flag = 1 at the sampled construction root, 0 at all other lasers
```

The graph message layers, local edge features, decoder heads, target
convention, and phase formula are unchanged. The same root supplies both
the encoder flag and the physical delay reference:

```text
phi_p,ij = target_j - target_i - 2*pi*f_root*tau + learned_residual_ij
```

For the current 64-wide first layer, the extra column adds 64 parameters:
127,368 total instead of 127,304. It starts at zero, preserving all shared
initial weights and the initial RNG state. Its gradient can learn immediately.
No model is resumed or transferred from the completed run.

Contexts, masks and construction roots consume the same NumPy draws as the
previous run. All eight actions of each topology receive that topology's
root flag. It is supplied before prediction in training, validation and
loaded inference. The decoder consequently receives root-aware embeddings.
The saved `encode_root=True` setting restores the five-feature encoder when
loading the new checkpoints; older checkpoints retain four features.

Sizes 3–9, seed 7, targets, random reachable graphs and optional links,
detuning spans up to 4 GHz, native corrected trapezoid with one correction,
standard LK model, zero noise, 500 ns, rewards, mirrored sampling,
pair-excluded baseline, schedules, workers and batch sizes are unchanged.
The run still has 1,000 iterations, 1,400 trajectories per iteration,
35 simulation jobs and 14 workers. Learned gates and lexicographic resource
objectives remain disabled.

Automatic inspection retains the sampler's actual root, even on feedback
graphs with several possible roots. For a manually supplied mask, the
existing deterministic rule chooses the lowest-detuned laser that reaches
every laser; both the flag and phase reference use it. Inspector settings
were not changed.

Importing the runner previews its settings. Running its single cell or
module starts training because `START_TRAINING=True`. Full training was
not started while preparing this experiment.
