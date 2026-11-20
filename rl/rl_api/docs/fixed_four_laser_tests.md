# Fresh fixed-span models with 64 exploration directions

Four independent jobs train from random seed-7 weights at **1, 2, 3 and 4 GHz**
peak-to-peak detuning spans. Each run uses equally spaced frequencies:

| Span | Detunings, GHz | Initial budget/M, ns^-1 | Job name | Progress image |
|---|---|---:|---|---|
| 1 GHz | -0.5, -1/6, +1/6, +0.5 | 4 | four_dirs64_1GHz | four_fixed_1ghz_progress.png |
| 2 GHz | -1, -1/3, +1/3, +1 | 6 | four_dirs64_2GHz | four_fixed_2ghz_progress.png |
| 3 GHz | -1.5, -0.5, +0.5, +1.5 | 9 | four_dirs64_3GHz | four_fixed_3ghz_progress.png |
| 4 GHz | -2, -2/3, +2/3, +2 | 12 | four_dirs64_4GHz | four_fixed_4ghz_progress.png |

All four use in-phase targets and the same **five-pair topology**: all reciprocal
pairs except lasers 2--3. Both physical coupling matrices remain symmetric.
The entire original GNN, strengths, phases and total budget are learned. Nothing
is initialized from an earlier checkpoint, and no corrected phase labels or
optimized designs are supplied. The initial weights are identical across spans
to make the span comparison controlled; the models train independently.

This test increases the sampled designs per iteration from **8 to 128**:
**64 independent mirrored directions**, compared with four previously. There
are 48 ordinary mirrored pairs and 16 wide-phase mirrored pairs. The original
75% ordinary / 25% wide mixture and all perturbation widths are retained.
One additional central prediction gives **129 simulations per iteration**.
The purpose is to test whether less noisy updates improve convergence; more
directions do not guarantee success.

The higher starting budgets in the table remain unchanged from the preceding
test. They are starting offsets, with no budget floor or penalty added.
Strength allocation, coupling phases, budget and all shared weights remain
trainable. The manifest records the nominal initial budget; the initial physical
prediction can differ slightly because the random budget head also contributes.
For an explicit low-budget control, the Python runner accepts
`--initial-budget-per-laser 0.5`.

## What is unchanged

Architecture, standard-LK physics, corrected native trapezoid, 500 ns duration,
5.4 ps timestep, free-running history, original coupling ramp, and zero noise.
The setup reuses A's reciprocal-pair Gaussian score and laser-1 reference.
REINFORCE uses the existing RMS reward and pair-excluded baseline. The baseline
for each pair averages the other 126 sampled designs, excluding both partners.
Exploration anneals to exact means over 1,000 iterations; the learning rate decays in the final 10%.
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
git pull --ff-only origin updates
bash scripts/submit_fixed_four_laser_tests.sh --smoke-test
```

Each smoke test runs two iterations with the actual 500 ns solver. After checking
their logs, submit the full suite:

```bash
bash scripts/submit_fixed_four_laser_tests.sh
watch -n 2 "squeue -u $USER -o '%.18i %.24j %.9T %.10M %.20R'"
```

Each job requests account `arrays0`, one node, **32 CPUs, 64 GB, and 48 hours**.
Smoke tests also retain the 48-hour request, but exit after two iterations.
The time is a limit: jobs finish as soon as training completes. Thirty-one
simulation workers share each allocation with the trainer. The 129 designs are
split into 31 groups of four or five trajectories so all workers can be used.
Jobs have no dependencies; Slurm may run them concurrently when resources are
available. Runtime and scaling need to be measured on the allocated node.
Sampled and mean designs are simulated together in vectorized groups;
means and checkpoints describe the weights
before that iteration's update. The final mean-only iteration contains no update.
`--iterations N` changes the iteration count and stretches the same schedules
over that count. `--actions-per-iteration N` changes the sampled batch size;
N must be a positive multiple of eight to retain complete ordinary and wide
mirrored pairs. Use `--actions-per-iteration 8` to reproduce the preceding
small-batch recipe. `--dry-run` prints submissions without launching them.

Logs: `logs/four_dirs64_<span>GHz-<jobid>.out` and `.err`. Checkpoints, physical
matrices, metadata and history are saved under
`rl/rl_api/runs/great_lakes/four_fixed_fresh_<span>ghz/<jobid>/`.
The `initial.pt` file and its state hash record the random initialization.
`current.pt`, `best.pt` and `final.pt` load through the existing `CouplingModel`.
`best.pt` is chosen by mean RMS reward on this fixed context. To inspect it,
supply the same fixed detunings, zero targets and the five-pair topology.
`current_design.json` and `best_design.json` contain the actual physical matrices.

## Asymmetric comparison

To repeat the same four fresh experiments while allowing independent coupling
strengths and phases in each direction, run:

```bash
bash scripts/submit_fixed_four_laser_tests.sh --asymmetric
```

The five-pair topology stays the same; only the requirement that reciprocal
links have equal values is removed. Both strength and phase matrices can be
asymmetric. The GNN architecture, initialization seed, starting budgets,
128 sampled designs plus one central prediction, 1,000-iteration schedule,
reward, simulator, 32 CPUs, 64 GB and 48-hour requests are retained. The score
uses the executed directed Gaussian actions instead of reciprocal averages.
Mirrored positive/negative exploration remains enabled; this is independent
of whether the physical coupling values are symmetric.

Job names end in `_asym`, for example `four_dirs64_4GHz_asym`. Runs are stored
under `rl/rl_api/runs/great_lakes/four_fixed_fresh_<span>ghz_asym/<jobid>/`.
Figures are named `four_fixed_<span>ghz_progress_asym.png`, keeping the existing
symmetric runs and figures separate. Add `--smoke-test` for a two-iteration check.

## Live image viewer

On your Mac:

```bash
python3 scripts/sync_great_lakes_plot.py
```

This automatically discovers all run-level progress images every ten seconds, including new tests, and copies them into the existing
`rl/rl_api/runs/great_lakes_live/four_laser_symmetric` viewer directory. Each
image has two panes: sampled/mean RMS rewards, and mean-design phase error with
the 3-degree target. Images update every ten iterations, plus the first and last.
The sampled curve averages the 128 exploratory designs. The mean-design curve
measures the central prediction without exploration noise and is monitored
separately from the sampled-action gradient. Plot titles show the direction count.
The existing viewer and Cloudflare tunnel do not need to restart.
Local image names include the run name to avoid collisions. New Slurm job IDs
create new run directories; all earlier checkpoints remain in theirs.

Automatic sync discovers both the four symmetric plots and the four asymmetric
plots. Leave it running when starting another test:

```bash
python3 scripts/sync_great_lakes_plot.py
```

All plots use the same viewer directory. The viewer and Cloudflare tunnel can
continue running. The old suite-selection flags remain available for a limited view.

## Symmetric 4 GHz precision continuation

This separate run continues the **best checkpoint from job 66366725**, saved at
iteration 653 with about 2.73 degrees worst-laser phase RMS. It uses the same
fixed five-pair topology, equally spaced 4 GHz detunings, in-phase target,
physical equations, simulator settings, architecture, reward and budget
calibration. All original weights, strengths, phases and budget remain trainable.
Only checkpoint weights and calibration are carried over; Adam momentum,
sampling seed, reward scaling and history start fresh. There are no teacher
labels or update-rejection rules.

The 128 sampled designs form 64 narrow mirrored pairs; extra-wide phase
exploration is disabled. Over 999 updates, the common exploration-SD multiplier
decreases geometrically from 0.03 to 0.003. For the source checkpoint this gives
about 0.27 to 0.027 degrees executed reciprocal coupling-phase SD, with strength
and budget perturbations narrowed by the same multiplier. Learned SD heads
remain active, so these physical widths can change during training. The learning
rate decreases from 1e-5 to 1e-6 over those updates. Iteration 1,000 is an exact
mean evaluation without an optimizer update. Each iteration still measures
128 samples plus one central prediction; current and best predictions are
recorded separately.

On Great Lakes:

```bash
cd ~/vcsel_lib
git pull --ff-only origin updates
bash scripts/submit_four_laser_precision.sh
```

The launcher calls `sbatch` with job name **four_4GHz_precision**, using the
existing 32-CPU, 64-GB, **48-hour** batch allocation. It defaults to
`rl/rl_api/runs/great_lakes/four_fixed_fresh_4ghz/66366725/best.pt` and checks that
the file exists before submission. `--checkpoint PATH` selects another
compatible fixed-case symmetric 4 GHz checkpoint. `--smoke-test` runs only two
iterations; `--dry-run` prints the submission without starting a job.

Outputs go to `rl/rl_api/runs/great_lakes/four_fixed_4ghz_precision/<jobid>/`.
The separate **four_fixed_4ghz_progress_precision.png** image is also copied to
that run's parent folder. The checkpoint SHA-256, source iteration, fresh
optimizer initialization, exploration scales and learning rates are recorded.
`initial.pt` reproduces the original checkpoint's physical mean design;
`best.pt` retains the best measured RMS reward from the continuation, including
that initial design. `final.pt` remains the actual last measured model.

The automatic Mac sync command discovers this precision plot alongside the
prior fixed-context plots without restarting or changing flags:

```bash
python3 scripts/sync_great_lakes_plot.py
```
