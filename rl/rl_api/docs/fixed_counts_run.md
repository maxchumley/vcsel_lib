# Five fixed link counts

The next experiment restores the [four-topology setup](four_counts_run.md).
This document describes the preserved five-count comparison.

Run the single `#%%` cell in `experiments/train_fixed_counts.py`.
`START_TRAINING=True` starts the experiment when you run it; importing the
file does not start training. Setup and verification do not start the run.

Run name: **star_rms_bins5_fixedcounts5_precision_standard_lk_3k**

Output: `rl/rl_api/runs/star_rms_bins5_fixedcounts5_precision_standard_lk_3k/`

This copies the random-count recipe in `train_precision.py`, changing only
`sampling.topology_sampling_mode` from `band_random` to `band_fixed` and the
output name. The original run file and its outputs are preserved.

## Link counts

Each pair receives five counts computed by
`np.rint(np.linspace(M-1, M*(M-1), 5)).astype(int)`. Rounding follows the old
evenly spaced sampler. Counts stay fixed across pairs and iterations.

| Lasers | Directed link counts |
| --- | --- |
| 3 | 2, 3, 4, 5, 6 |
| 4 | 3, 5, 8, 10, 12 |
| 5 | 4, 8, 12, 16, 20 |
| 6 | 5, 11, 18, 24, 30 |
| 7 | 6, 15, 24, 33, 42 |
| 8 | 7, 19, 32, 44, 56 |
| 9 | 8, 24, 40, 56, 72 |

Optional edges are still randomly drawn independently for each pair and each
intermediate topology on every iteration. The first mask is the exact outward
star from laser 1; the last is all-to-all. Masks are not shared across pairs.
Candidate order stays star, sparse, middle, dense, all-to-all.

## Full comparison settings

The run retains native `standard_lk` physics, the same 127,304-parameter model,
wrapped RMS learning reward, cosine monitoring, seed 7, five detuning/target
pairs per size, five 0.8 GHz span bins, eight actions per topology, and both
curricula disabled. Each iteration has 1,400 training cases in 35 vectorized
jobs across 14 workers.

All 21 size/density PCGrad tasks are populated. Within each size, sparse,
middle and dense weights are 40%, 20%, 40%, with 80, 40, 80 cases respectively.
Per-task and final clipping limits are both 1.0.

The full 3,000-iteration [precision schedule](precision_run.md#stages-within-3000-iterations)
is retained: ordinary and wide phase annealing over 1000–1500; ordinary/budget
precision refinement and LR decay over 1500–2000; then 1,000 iterations at LR
1e-5. Wide exploration initially affects two of eight actions with extra
180-degree SD. The held-out bank, validation frequency, checkpoint selection,
LaTeX font, plot resolution and parent-folder live image are also retained.

`run.json` records `band_fixed`. Checkpoints and history identify this sampler
and store the same precision schedule and actual exploration bounds. The
numerical backend fields match the random-count run; the API selects the
fixed-count sampler before assembling each batch.

Tests check the configuration comparison, all fixed counts and density bands,
fresh rooted wiring, the 1,400-case/35-job batch, all 21 task weights, precision
refinement, checkpoint metadata and one-cell startup.
