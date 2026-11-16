# Extend the successful star policy to arbitrary paths from laser 1

Run the editable single cell in
`experiments/train_mirrored_laser1_reachable_1000.py`, or from the repository root:

```sh
python -m rl.rl_api.experiments.train_mirrored_laser1_reachable_1000
```

Importing the module only previews settings. Executing it starts a new
1,000-iteration run. The default output is:

```text
rl/rl_api/runs/laser1_reachable_from_star_rms_bins5_randbands5_phase_delay_residual_anneal1to1000_lr900_standard_lk_mirrored_1k/
```

`INITIAL_CHECKPOINT` points to the successful delay/star run's `final.pt`.
The new run copies its learned weights **and frozen per-size budget calibration**.
On an identical target, detuning vector and mask, initial deterministic physical
strengths and phases are unchanged. Adam state, reward scaling, sampling RNG,
history and the exploration/learning-rate schedules start fresh. This is
fine-tuning in a separate run, not an exact continuation of the old optimizer.
The source checkpoint is read only. Restarting this new recipe archives only
its own previous output directory.

Laser 1 always means the **lowest-detuned laser after sorting**, not the first
laser in an unsorted input vector. `TopologyConfig(root_index=0)` fixes the
connectivity root to that laser. No root flag or additional network input is
needed. The phase convention remains the successful `target_delay_residual`:

```text
phi_p[receiver i, source j] = target_j - target_i
                            - 2*pi*f_1*tau_effective + learned_residual_ij
tau_effective = int(tau/dt)*dt
```

For each topology, randomly order all directed non-self edges and enable them
until laser 1 can reach every laser. Continue enabling random edges to meet the
requested count. No direct outward-star links are mandatory. Paths can pass
through other lasers; cycles, feedback into laser 1 and links among other lasers
are eligible. This preserves reachability, not an unforced root or a prescribed
actual locking frequency. Strongly connected graphs can also have other roots;
laser 1 is simply the fixed construction root and phase reference for this test.

Each target/detuning pair has five topology slots: first reachable, requested
sparse/middle/dense counts, and all-to-all. Reachability can require more links
than requested. PCGrad uses the actual physical density band, as in the existing
random-reachable recipes. The first reachable slot is not guaranteed to be a
minimal tree. Training, held-out validation and automatic inspection all retain
the fixed-root rule, including after loading a checkpoint.

All other public settings match `train_mirrored_delay_1000.py`: sizes 3–9,
0–4 GHz spans, random targets, mirrored sampling, 5 pairs per size, 5 topologies,
8 actions, 14 workers, wrapped-RMS reward, 500 ns, the corrected native integrator,
and the original 1,000-iteration annealing/LR schedule. There are **1,400 ordinary
training simulations per iteration and no feedback-teaching simulations**.
The final iteration evaluates exact mean actions without updating weights.
Training plots and histories use the existing output format.

Batch preview:

```sh
python -m rl.rl_api.experiments.train_batch train_mirrored_laser1_reachable_1000 --preview
```

For a server where the source results directory is absent, copy the source
checkpoint separately and pass its location; it does not need to be in Git:

```sh
python -m rl.rl_api.experiments.train_batch train_mirrored_laser1_reachable_1000 \
  --initial-checkpoint /path/to/star/final.pt
```

This test measures adaptation away from a mandatory star while holding the
frequency anchor fixed. It does not establish performance for topologies where
laser 1 cannot reach every laser, or for an arbitrary choice of reference root.
