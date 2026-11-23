# Fresh full-density conditional-phase test

This run learns from seed-7 random weights, with fresh budget calibration and
Adam state. It does not load a checkpoint, refine a prior model or progressively
tighten symmetry. It covers 3–9 lasers, random relative phase targets, random
detuning distributions and stratified spans up to 4 GHz. Each context requests
a tree, one density in each third of [0,1], and all-to-all connectivity.

Both connectivity and physical coupling values are symmetric from the first
iteration. As in the existing reciprocal-pair recipe, one strength and phase
are sampled per unordered pair and copied into both directions; values are
not produced by averaging independent directed actions.

The original GNN predicts pair strengths and total budget using the density
input. After selecting connected links and redistributing the budget, a second
pass through the shared graph layers receives the actual physical strengths.
A 129-parameter zero-initialized linear head adds a conditional phase residual.
Its canonical-pair output is shared in both directions. Each sampled phase is
scored once under its conditional Gaussian. Mirrored partners use their own
strength-conditioned phase means with opposite phase noises. This adds no
simulator calls or optimized design labels.

All other numerical settings match
`train_mirrored_symmetric_density_pairs_1000`: 1,000 iterations, 7 sizes × 5
target/detuning contexts × 5 density requests × 8 mirrored actions = 1,400
simulations per iteration, 500 ns trajectories, the laser-1 delay-frequency
reference, RMS reward, PCGrad by size/density band, exploration annealed to zero
over 1,000 iterations and learning-rate decay over iterations 900–1,000.
The last iteration evaluates exact mean actions without an optimizer update.

Validation uses 64 fixed held-out contexts per size (448 total) every 50
iterations. Figures update every 10 iterations and use the standard two-panel
training plot: training cosine/RMS rewards and enabled fraction on the left;
held-out rewards by size and density band on the right. This matches the
requested density-training curve rather than the refinement plot.

Submit with `bash scripts/submit_density_conditioned_fresh.sh`. The additional
job `density_cond_fresh` requests 32 CPUs, 31 workers, 64 GB and seven days. Run
files live under
`rl/rl_api/runs/great_lakes/symmetric_density_pairs_conditioned_phase_fresh_1k/<job-id>/`.
Its unique `density_conditioned_sym_fresh_progress.png` is automatically
discovered by the existing plot sync. All existing jobs remain unchanged.
