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
Open `current_training_progress.png` through VS Code to view progress; the text
log records every iteration. Set `CHECKPOINT` in the existing inspect script
to the saved `best.pt` or `final.pt` for inspection. `current_history.json` is
written at validation steps; `current.pt` is updated every iteration.

The current API saves checkpoints but does **not** resume a partly completed
training run. A time limit or interruption therefore cannot be recovered by
simply requeueing. `restart=True` starts again from iteration 1 and archives the
old attempt. Choose the full job's time limit using the timing job, and keep
important results on durable storage if using temporary `/scratch` space.

References: [U-M Python/PyTorch environment setup](https://documentation.its.umich.edu/arc-software/pytorch/installation),
[Slurm user guide](https://documentation.its.umich.edu/arc-hpc/slurm-user-guide),
and [Great Lakes partition and storage policies](https://documentation.its.umich.edu/arc-hpc/greatlakes/policies).
