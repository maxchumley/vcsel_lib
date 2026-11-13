# Next run: four masks reused across five pairs

Open `experiments/train_shared_masks.py`. All physical and RL settings are in
that file, using native `phys` keys and `vcsel = VCSEL(phys)`. It supports cells
and standalone execution, with one `__name__` guard protecting the Train cell.
`START_TRAINING=False` previews only. No training has been started during setup.

Training plots use the LaTeX serif font from `simple_example.py` and 300 DPI
(3300 x 1200 pixels). Set `validation.plot_dpi` and `validation.plot_use_tex`
in the run file to adjust these. LaTeX rendering requires a working local TeX
installation; set `plot_use_tex=False` to use Matplotlib's regular font.
The same settings apply to saved progress PNGs, notebook figures and
`result.plot()`. Plot styling does not change learning or validation metrics.

Run name: **star_rms_shared4_pairs5_wide180_3k**

Default output: `rl/rl_api/runs/star_rms_shared4_pairs5_wide180_3k/`

After training, set `CHECKPOINT` in `experiments/inspect_current.py` to this
folder's `best.pt` or `current.pt` to inspect it.

## What is shared

For each M, at every iteration:

1. Sample five target/detuning pairs, one from each 0.8 GHz span bin.
2. Draw four topology matrices once: the exact outward star, one random sparse
   mask, one random middle mask, and one random dense mask.
3. Give those **same four binary matrices** to every pair. Matrix elements,
   rather than just link counts, are equal across the five pairs.
4. For each pair/mask combination, predict and sample ten actions independently.
5. Group PCGrad by M and mask identity: four tasks per M, each seeing five pairs.

Masks are redrawn for each M and each iteration. Optional edges always include
the lowest-detuned laser's outward star. Other masks sample a link count from
the sparse/middle/dense thirds of excess links, excluding the exact-star count,
then choose optional directed edges randomly. Thus the first task always owns
the star; the remaining tasks retain ordered density-band identities.

This replaces the old pool of twenty density positions per M. It intentionally
gives the exact star one quarter of training actions. The supplied fraction
rho now varies with the sampled band densities; it is not fixed near 0.595.

Sharing matrices does not share target phases, detunings, decoded continuous
couplings, action noise, trajectories, or rewards. Every pair/mask/action is a
separate physical simulation.

## Counts and retained settings

| Setting | Next run |
| --- | --- |
| Array sizes | M=3 through 9 |
| Pairs per M | 5 |
| Actual training masks per M | 4, shared across all five pairs |
| Contexts per M | 20 |
| Actions per context | 10 |
| Policy simulations per iteration | 7 x 5 x 4 x 10 = 1400 |
| Star simulations per M | 5 x 10 = 50 |
| Initial wide star actions per M | 5 x 2 = 10 |
| Initial wide actions overall | 280 |
| PCGrad tasks | 28; five pairs per task |
| Simulation dispatch | 28 jobs of 50 cases over 14 processes |
| Learning reward | Wrapped RMS; cosine monitoring/validation curves retained |
| Curricula | Detuning and topology curricula disabled |
| Ordinary/wide annealing | 1000–1500 printed iterations |
| LR | 1e-4 through 2000; decay to 1e-5 by 3000 |
| Physical rollout | 500 ns; existing VCSEL numerical settings |

Validation uses the existing held-out targets, detunings, topology sampler,
and deterministic mean actions, making the curves comparable to the current
paired run.

## Implementation and verification

`SamplingConfig.reuse_masks_across_pairs=True` selects the new parent-process
batch in `training_batch.py`. It uses the shared sampler in `sampling.py` and
the existing policy, decoder, VCSEL workers, rewards, baseline normalization,
clipping, PCGrad, and Adam. The default False still reproduces the original
experiment. Existing numerical source files and their globals are untouched.

Tests verify identical masks across pairs for every M, independent action
noise, rooted connectivity, exactly one star per pair, fresh masks on later
iterations, 1400 cases in 28 jobs, out-of-order result collection and 28 PCGrad
tasks with five pairs each. With the original sampler injected, the new active
batch reproduces the original weights, Adam state, normalization state and
NumPy/Torch RNG states across three updates spanning the exploration stages.
These optimizer tests use synthetic physical trajectories; they do not start
a full training run.

A physical smoke check also sent two shared-star cases from different detuning
pairs through the actual worker in serial and through two spawned processes.
All returned reward/phase arrays matched exactly; no optimizer was created.

The public flag is saved in `run.json`. Checkpoints and history also contain
`sampling_recipe`, because the original backend's config schema has no field
for actual-mask reuse. Existing checkpoint loaders can still read the weights.
Exact mid-run resume remains unsupported.
