# Next test: BACKUP topology sampling, five detuning bins

Start in `experiments/train_backup_sampling.py`. The goal is to test whether
the successful BACKUP approach works without detuning or topology curricula,
while covering more detuning spans. All physical parameters and training
settings are visible in the run file. It supports `# %%` cells and standalone
execution. `START_TRAINING=False`; setup has not started training.

Run name: **star_rms_backup_sampling_bins5_topo4_actions8_3k**

Default output:
`rl/rl_api/runs/star_rms_backup_sampling_bins5_topo4_actions8_3k/`

The live-viewer image is `rl/rl_api/runs/current_training_progress.png`, one
folder above the run. Each plot update saves it there and keeps a copy inside
the run folder. Both images retain the configured DPI and LaTeX font.

This is a fresh experiment. It does not resume or overwrite a checkpoint.
The prior paired and shared-mask run files keep their own settings and names.

To repeat this experiment under the same name, the run file sets
`RESTART_EXISTING_RUN=True`. When you start training, the existing run folder
is moved to `runs/restarts/<run-name>_<UTC timestamp>/`, then a fresh model and
optimizer start at iteration 1 in the original folder. Previous files are
preserved. Set the flag to False to reject an occupied output folder instead.
Stop the prior training before starting another attempt. After updating API
code in an existing notebook session, restart the kernel and rerun the cells.

## Comparison with the successful BACKUP run

| Setting | BACKUP | This test |
| --- | --- | --- |
| Array sizes | 3 through 9 | 3 through 9 |
| GNN architecture | 127,304 parameters | Same architecture and initialization |
| Target/detuning pairs per size | 2 | 5 |
| Detuning spans | Stratified up to 4 GHz | Five 0.8 GHz bins covering 0–4 GHz |
| Topology sampling | Full star-to-dense range separately for each pair | Same sampling rule, four levels per pair |
| Topologies per pair | 9 | 4 |
| Actual intermediate masks shared across pairs | Disabled | Disabled |
| Actions per topology | 8, including 2 wide | 8, including 2 wide |
| Training simulations per iteration | 1,008 | 1,120 |
| Learning reward | Cosine | Wrapped RMS, as requested |
| Cosine monitoring | Enabled | Enabled |
| Detuning/topology curricula | 500 iterations | Disabled |
| PCGrad grouping | Physical sparse/middle/dense bands within each M | Same band boundaries, weighting and empty-band rule |
| Task/global gradient clipping | Both 1.0 | Both 1.0 |
| Advantage baseline | Mean across actions within each pair/mask | Same baseline and shared running reward scale |
| Rooted connectivity | Lowest-detuned laser drives every other laser | Same mandatory outward star |
| Physical model | 500 ns VCSEL simulation | Same physical values and numerical backend |
| Ordinary exploration annealing | About 1500–2500 | 1000–1500, retaining the current schedule |
| Extra phase SD | 180 degrees, taper about 500–4000 | 180 degrees, taper 1000–1500 |
| LR | 1e-4, decay about 3500–5000 to 1e-5 | 1e-4, decay 2000–3000 to 1e-5 |
| Total iterations | 5,000 | 3,000 |
| Simulation dispatch | One pair and 72 actions per job | Five pairs and 8 actions per job; 28 jobs over 14 workers |

RMS and the shorter schedule are intentional user choices. This is therefore
a comparison of several changes, not an experiment that isolates the effect
of removing the curricula. Recovering BACKUP's final performance is the goal;
setup and numerical parity checks cannot establish that learning outcome.

## What happens each iteration

For each M, sample five new phase targets and five detuning distributions.
One distribution spans each of 0–0.8, 0.8–1.6, 1.6–2.4, 2.4–3.2 and 3.2–4 GHz.
Span-to-target assignments are shuffled. Lasers are ordered by detuning.

For **each pair separately**, use BACKUP's sampler to choose four link counts:

```python
counts = np.rint(np.linspace(M - 1, M * (M - 1), 4)).astype(int)
```

Randomize their order and choose optional edges independently for that pair.
The outward star remains mandatory. Star and all-to-all masks necessarily
match across pairs; intermediate masks are independently drawn. All masks are
redrawn at the next iteration. There is no 20-mask pool assigned across bins
and no broadcasting of one four-mask set to every pair.

Each pair/mask gets eight independent sampled actions: six ordinary and two
with extra phase noise, tapered by the configured schedule. Thus there are
20 contexts and 160 simulations per M, or 1,120 total. The mean supplied
fraction is fixed at approximately **0.5949263** across M=3…9 because link
counts are fixed, even though edges and pair assignments change.

## PCGrad with four density levels

BACKUP groups candidates using integer physical excess-density boundaries,
then skips empty bands. With only four evenly spaced levels, rounding puts
both intermediate levels outside the middle band for several M values:

| M | Sorted link counts | Occupied PCGrad bands |
| --- | --- | --- |
| 3 | 2, 3, 5, 6 | sparse, dense |
| 4 | 3, 6, 9, 12 | sparse, middle, dense |
| 5 | 4, 9, 15, 20 | sparse, dense |
| 6 | 5, 13, 22, 30 | sparse, dense |
| 7 | 6, 18, 30, 42 | sparse, middle, dense |
| 8 | 7, 23, 40, 56 | sparse, dense |
| 9 | 8, 29, 51, 72 | sparse, dense |

There are **16 nonempty tasks** overall. Each band includes its candidates
from all five pairs. There is no detuning PCGrad. The code preserves BACKUP's
classification and task weights; it does not force extra middle-band samples
or change the boundaries to manufacture 21 tasks.

## Implementation and checks

`SamplingConfig.topology_sampling_mode="per_target"` selects independent
BACKUP-style sampling. `OptimizationConfig.pcgrad_grouping="density_bands"`
selects BACKUP's physical bands. The existing numerical batch handles both,
including RMS scoring, clipping, PCGrad and Adam. Older API recipes retain
their default sampling and gradient grouping.

Tests compare sampler output and subsequent RNG state directly against the
actual BACKUP numerical module for every M. A cosine-only comparison test
checks full BACKUP/current batch equivalence across exploration stages,
including model weights, Adam state, running normalization and RNG states.
The production recipe remains RMS. A separate synthetic-physics RMS update
checks 1,120 cases in 28 vectorized jobs, 16 tasks and loadable checkpoints.
These checks do not run the user's training or prove final model quality.

Training plots retain 300 DPI and the user's LaTeX serif font. Checkpoints
and history record the independent-per-pair sampling recipe. Validation uses
the fixed held-out cases and deterministic mean actions; best-checkpoint
selection uses RMS. `experiments/inspect_current.py` defaults to this run's
`current.pt` and prints the loaded iteration. Set `CHECKPOINT_KIND="best"`
to inspect the best validation checkpoint instead. The inspector has one
cell: edit its settings, then run the cell to load, predict, evaluate and plot.
