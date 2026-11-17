# Training on Great Lakes

Submit one CPU job on one node. The existing trainer uses a local process pool;
multiple Slurm tasks or nodes would not distribute this training run.
The supplied job requests 16 CPUs, 64 GB RAM and a 48-hour limit under `arrays0`.
It keeps the current recipe's 14 simulation workers, with room for the trainer.
Memory and time are starting requests; use the first timing job to size them.

## Install once on the cluster

In the VS Code SSH terminal, change into the cloned repository's root. Load the
Python distribution and check whether a suitable environment already exists:

```bash
module purge
module load python3.11-anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list
```

If there is no `vcsel` environment, create one:

```bash
conda create -n vcsel python=3.11 pip -y
```

Activate it and install the project. CPU PyTorch is sufficient: the numerical
VCSEL simulations run on CPUs, and the current experiment selects `device="cpu"`.
Installing CPU PyTorch first avoids downloading CUDA libraries for this job.

```bash
conda activate vcsel
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,rl]"
python -m pip check
```

`rl` currently runs from the source checkout. Keep the clone available and launch
from its root; installing the `vcsel-lib` wheel alone does not include the RL tree.
If the module name is unavailable, use `module spider python3.11-anaconda` to
find the installed version and use that same module in the batch script.

## Preview and submit

Preview only; this imports the recipe and prints settings without starting
simulations, creating a run directory, or executing its `START_TRAINING` block:

```bash
python -m rl.rl_api.experiments.train_batch --preview
mkdir -p logs
```

The log directory must exist **before** `sbatch`, since Slurm opens its log files
before executing the script. Start with a two-iteration launch check on a compute node:

```bash
sbatch --time=01:00:00 scripts/train_rl_great_lakes.slurm --smoke-test
```

This check uses the full physics, sizes, batches and validation bank, but stops
after iteration 2. For an anneal-to-mean recipe, its action annealing is compressed
into those two iterations because the API requires zero exploration at the final
iteration. These weights are only a launch check. The full run below retains all
original schedules. Allow for initial calibration and the check's two validation
steps when using its runtime to estimate the full job's time limit.

Once that succeeds, submit the unchanged full recipe:

```bash
sbatch scripts/train_rl_great_lakes.slurm
```

Both commands default to `train_mirrored_root_encoder_1000`. All physics,
sampling, reward, network, seed, validation frequency, and training schedules
come from that existing run file. The batch launcher only turns off notebook
display and external LaTeX rendering; it still saves progress PNGs. It ignores
the recipe's interactive start/restart/output flags and starts fresh without
archiving existing results. Each Slurm job gets its own directory:

```text
rl/rl_api/runs/great_lakes/<RUN_NAME>/<JOB_ID>/
    run.json
    current.pt
    best.pt
    final.pt                       # only after the run completes
    current_history.json
    current_training_progress.png
```

Use a different existing recipe by passing its module name:

```bash
sbatch scripts/train_rl_great_lakes.slurm train_mirrored_1000
```

Edit scientific settings in the chosen experiment's `CONFIG`. To change only the
worker count, change the CPU request and pass `--workers`, leaving one CPU for
the trainer. For example, `sbatch --cpus-per-task=8 ... --workers 7`.
Use `VCSEL_CONDA_ENV` for an environment with another name/path. Override
`--output-directory` with an absolute path when writing to project storage.
Use a fresh output directory for every submission; an existing nonempty directory
is rejected. Generated runs, checkpoints and scheduler logs stay ignored by Git.

## Monitor and inspect

```bash
squeue -u "$USER"
tail -f logs/rl-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

Replace `JOB_ID` with the number printed by `sbatch`. Errors are in the matching
`logs/rl-JOB_ID.err`. Stop a job with `scancel JOB_ID`.
The output log prints `Batch Python` and `Batch PyTorch` from the compute node
before training starts. The script launches `$CONDA_PREFIX/bin/python` by its
full path and uses `srun --export=ALL` to retain the activated environment.
If an import fails, compare those paths with
`python -s -c "import sys, torch; print(sys.executable); print(torch.__file__)"`
in the activated login-node environment before reinstalling packages.
Open `current_training_progress.png` through VS Code to view progress; the text
log records every iteration. Set `CHECKPOINT` in the existing inspect script
to the saved `best.pt` or `final.pt` for inspection. `current_history.json` is
written at validation steps; `current.pt` is updated every iteration.

The current API saves checkpoints but does **not** resume a partly completed
training run. A time limit or interruption therefore cannot be recovered by
simply requeueing. `restart=True` starts again from iteration 1 and archives the
old attempt. Choose the full job's time limit using the timing job, and keep
important results on durable storage if using temporary `/scratch` space.

## Mirror the plot into the Mac's live image viewer

The existing Cloudflare viewer serves folders on the Mac. Its temporary public
URL can change without affecting synchronization: mirror the cluster PNG into
a watched local folder, and keep the viewer/tunnel running normally.

Run a one-file transfer check in a **Mac terminal**, not in the SSH terminal:

```bash
cd ~/Documents/GitHub/vcsel_lib
python3 scripts/sync_great_lakes_plot.py --once
```

Authenticate to Great Lakes normally when prompted. For continuous updates:

```bash
python3 scripts/sync_great_lakes_plot.py
```

This opens one private persistent SSH connection, checks for a changed PNG
every 10 seconds, and copies only `current_training_progress.png` with rsync.
SSH authentication is reused across checks; connection failure stops the sync
instead of repeatedly prompting. Ctrl+C stops only this sync connection.
Training and the viewer remain running. No remote files, packages or Slurm
settings are changed. These commands do not restart the training job.

The default remote source is the four-laser run's stable parent-folder image:

```text
~/vcsel_lib/rl/rl_api/runs/great_lakes/<four-laser RUN_NAME>/current_training_progress.png
```

Use `--job-id JOBID` to select one exact job instead, or `--run RUN_NAME` and
`--repository REMOTE_PATH` for another output location. `--dry-run` shows the
paths without connecting. The normal SSH host/configuration is used; `--host`
can override it. No passwords are stored in this helper.

The default local mirror folder is separate from the local run's plot:

```text
rl/rl_api/runs/great_lakes_live/four_laser_symmetric/
```

The existing viewer can watch it through this `sources.txt` entry:

```text
Great Lakes Four-Laser RL:1:/Users/mchumley/Documents/GitHub/vcsel_lib/rl/rl_api/runs/great_lakes_live/four_laser_symmetric
```

The viewer reloads its sources while polling, so this entry does not require
a new public tunnel. Choose **Great Lakes Four-Laser RL** in the source menu.
The Mac must remain running/connected to serve and refresh this local mirror.
The training PNG itself is written every ten iterations, roughly six minutes
at the current measured iteration time; syncing more often does not create
new training frames. Generated images stay ignored by Git.

U-M documents SSH-based file transfers in its
[file-transfer guide](https://documentation.its.umich.edu/node/5092).

## Three matched four-laser symmetric tests

All three start from the same fresh seed-7 weights. Every iteration uses the
same five detuning contexts, all 38 symmetric topologies, eight mirrored actions,
1,520 simulations, in-phase targets, 0-4 GHz spans and the original 1,000-iteration
schedules. Both physical coupling matrices stay exactly symmetric. The network,
reward, integrator, ramp, validation bank and batch sizes are unchanged.

| Test / Slurm name | Gaussian action score | Phase baseline | Progress PNG |
|---|---|---|---|
| A / `four_A_pairs` | Shared reciprocal pair | Laser 1 detuning | `four_A_pairscore_progress.png` |
| B / `four_B_meanref` | Original independent directed actions | Zero mean detuning | `four_B_meanref_progress.png` |
| C / `four_C_both` | Shared reciprocal pair | Zero mean detuning | `four_C_both_progress.png` |

Shared-pair scoring marginalizes the directional difference discarded by the
physical averaging step. It keeps the original sampled actions, RNG draws,
physical matrices and budget samples unchanged at equal policy weights; only
the REINFORCE score changes. The sum of both directed variances is included in
the pair's Gaussian variance, including wide-phase exploration. The mean-reference
variants use the existing `target_residual` parameterization: no laser-1 delay
offset is added. Their detunings remain mean-centered; the LK equations are
unchanged, and the locking frequency is not constrained.

After these code changes are available in the Great Lakes checkout, submit
from **Great Lakes**:

```bash
cd ~/vcsel_lib
bash scripts/submit_four_laser_symmetric_tests.sh --dry-run
bash scripts/submit_four_laser_symmetric_tests.sh
```

The launcher submits three independent jobs, each requesting one node, 32 CPUs
(one trainer and 31 workers), 64 GB and 48 hours under `arrays0`. Three concurrent
jobs therefore request 96 CPUs and 192 GB in total; Slurm starts each when its
allocation is available. There are no dependencies between the jobs. To launch
all three as two-iteration smoke tests instead:

```bash
bash scripts/submit_four_laser_symmetric_tests.sh --smoke-test
```

Each has a descriptive Slurm name as well as its mandatory numeric job ID.
Use this format so `squeue` does not truncate the names:

```bash
squeue -u "$USER" -o "%.18i %.24j %.9T %.10M %.20R"
```

Logs use `logs/<job-name>-<job-ID>.out` and `.err`. Checkpoints/history remain
isolated under `rl/rl_api/runs/great_lakes/<RUN_NAME>/<job-ID>/`. The three run
names are `four_sym_A_pairscore_laser1_1k`, `four_sym_B_directed_meanref_1k`, and
`four_sym_C_pairscore_meanref_1k`. Their uniquely named PNGs are saved inside
each job directory and atomically copied to its run's parent folder.

On the **Mac**, replace the one-image sync command with:

```bash
python3 scripts/sync_great_lakes_plot.py --symmetric-tests
```

It copies all three uniquely named images into the existing watched folder,
`rl/rl_api/runs/great_lakes_live/four_laser_symmetric/`, through one authenticated
SSH connection, checking every ten seconds. A queued job's missing PNG does not
prevent the other two from syncing. Use the existing **Great Lakes Four-Laser
RL** viewer source; the site/tunnel does not need restarting. Images update at
iteration 1, every ten iterations and the final iteration. A single variant
can also be mirrored with `--run RUN_NAME --filename PROGRESS_FILENAME`.

The editable variant modules are
`experiments/train_four_laser_symmetric_pairscore_1000.py`,
`experiments/train_four_laser_symmetric_meanref_1000.py`, and
`experiments/train_four_laser_symmetric_pairscore_meanref_1000.py`. They share
the original recipe through `experiments/four_laser_symmetric_comparison.py`.

References: [U-M Python/PyTorch environment setup](https://documentation.its.umich.edu/arc-software/pytorch/installation),
[Slurm user guide](https://documentation.its.umich.edu/arc-hpc/slurm-user-guide),
and [Great Lakes partition and storage policies](https://documentation.its.umich.edu/arc-hpc/greatlakes/policies).
