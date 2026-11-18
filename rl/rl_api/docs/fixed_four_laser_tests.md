# Fresh models on four fixed detuning spans

Four independent jobs train from random seed-7 weights at **1, 2, 3 and 4 GHz**
peak-to-peak detuning spans. Each run uses equally spaced frequencies:

| Span | Detunings, GHz | Job name | Progress image |
|---|---|---|---|
| 1 GHz | -0.5, -1/6, +1/6, +0.5 | four_fixed_1GHz | four_fixed_1ghz_progress.png |
| 2 GHz | -1, -1/3, +1/3, +1 | four_fixed_2GHz | four_fixed_2ghz_progress.png |
| 3 GHz | -1.5, -0.5, +0.5, +1.5 | four_fixed_3GHz | four_fixed_3ghz_progress.png |
| 4 GHz | -2, -2/3, +2/3, +2 | four_fixed_4GHz | four_fixed_4ghz_progress.png |

All four use in-phase targets and the same **five-pair topology**: all reciprocal
pairs except lasers 2--3. Both physical coupling matrices remain symmetric.
The entire original GNN, strengths, phases and total budget are learned. Nothing
is initialized from an earlier checkpoint, and no corrected phase labels or
optimized designs are supplied. The initial weights are identical across spans
to make the span comparison controlled; the models train independently.

## What is unchanged

Architecture, standard-LK physics, corrected native trapezoid, 500 ns duration,
5.4 ps timestep, free-running history, original coupling ramp, and zero noise.
The setup reuses A's reciprocal-pair Gaussian score and laser-1 reference.
REINFORCE uses the existing RMS reward and pair-excluded baseline. Eight mirrored
actions comprise three ordinary pairs and one wide pair. Exploration anneals to
exact means over 1,000 iterations; the learning rate decays in the final 10%.
The final iteration evaluates means without an optimizer update.

The earlier proposed **0.5-degree phase-only continuation is not this test**.
These fresh joint models retain the existing broader exploration schedule.
Only one context and topology are supplied throughout each run. The resulting
accuracy measures fitting that case; it does not measure generalization.
With one task, the update uses its clipped gradient directly because there is
no second task for PCGrad to project against. Existing multi-task updates are
unchanged.

## Great Lakes

After these code changes have been committed, pushed and pulled on the cluster:

```bash
cd ~/vcsel_lib
bash scripts/submit_fixed_four_laser_tests.sh --smoke-test
```

Each smoke test runs two iterations with the actual 500 ns solver. After checking
their logs, submit the full suite:

```bash
bash scripts/submit_fixed_four_laser_tests.sh
watch -n 2 "squeue -u $USER -o '%.18i %.24j %.9T %.10M %.20R'"
```

Each job requests account `arrays0`, one node, four CPUs, 16 GB, and four hours.
Three simulation workers share each allocation. Jobs have no dependencies;
Slurm may run them concurrently when resources are available. Each iteration
simulates eight sampled designs and one deterministic mean design, rather than
the previous 1,520 training trajectories. Sampled and mean designs are simulated
together in vectorized groups; means and checkpoints describe the weights
before that iteration's update. The final mean-only iteration contains no update.
`--iterations N` changes the iteration count and stretches the same schedules
over that count. `--dry-run` prints submissions without launching them.

Logs: `logs/four_fixed_<span>GHz-<jobid>.out` and `.err`. Checkpoints, physical
matrices, metadata and history are saved under
`rl/rl_api/runs/great_lakes/four_fixed_fresh_<span>ghz/<jobid>/`.
The `initial.pt` file and its state hash record the random initialization.
`current.pt`, `best.pt` and `final.pt` load through the existing `CouplingModel`.
`best.pt` is chosen by mean RMS reward on this fixed context. To inspect it,
supply the same fixed detunings, zero targets and the five-pair topology.
`current_design.json` and `best_design.json` contain the actual physical matrices.

## Live image viewer

On your Mac:

```bash
python3 scripts/sync_great_lakes_plot.py --fixed-context-tests
```

This copies all four uniquely named images every ten seconds into the existing
`rl/rl_api/runs/great_lakes_live/four_laser_symmetric` viewer directory. Each
image has two panes: sampled/mean RMS rewards, and mean-design phase error with
the 3-degree target. Images update every ten iterations, plus the first and last.
The existing viewer and Cloudflare tunnel do not need to restart.
