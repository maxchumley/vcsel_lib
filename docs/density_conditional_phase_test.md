# Conditional phase refinement test

This is an additional matched test of the physical-bound **both matrices** run.
It starts from the same working asymmetric density checkpoint (iteration 950,
SHA-256 `83ad841f91a2022cd869b8d9cb21a82a57ce11a60f99f85e3ef5b7766008e43b`).
Existing jobs continue unchanged.

The original policy first samples strengths and budget from the target phases,
detunings, array size and requested density. The decoder selects connected
symmetric links, redistributes the budget and applies the current strength
symmetry bounds. A second pass through the shared graph layers receives
`log(1 + kappa / strength_scale)` in the two adjacency feature slots. Zeros
represent absent links; nonzero values retain the actual strength and budget
information. A new linear head adds a phase residual to the original phase mean.
Its 129 weights start at zero, preserving the checkpoint's initial predictions.

The joint policy remains REINFORCE:

`log p(strength, budget | context, density) + log p(phase | context, density, selected strengths)`.

Physical strengths used as phase inputs are detached. Their original Gaussian
scores still train strength and topology selection. The phase score uses the
conditional mean for that particular sampled strength matrix. Mirrored strength
partners can therefore have different conditional phase means, with opposite
phase noise around their own means. No labels, simulation gradients, feedback
rollouts, or extra simulator calls are added.

All remaining settings match the physical-bound run: 3–9 lasers, random targets
and detunings up to 4 GHz, five density requests and eight actions per context,
1,400 simulations per update, 500 ns trajectories, laser-1 delay reference,
1,000 updates, validation every 50, figures every 10, narrow exploration and the
same accuracy guard before tightening symmetry. The final endpoint enforces
both symmetric strength and phase values; partial progress is reported separately.

Submit with `bash scripts/submit_density_conditional_phase.sh`. The additional
job `density_cond_phase` requests 32 CPUs, 31 workers, 64 GB and seven days.
Its plot is `density_conditional_phase_progress.png`; automatic plot sync finds
it alongside the other jobs. Run files live under
`rl/rl_api/runs/great_lakes/density_conditional_phase_bounds_both_from_working_laser1/<job-id>/`.

Compare its held-out phase accuracy and accepted symmetry level against
`density_bound_both`. This tests whether making phases conditional on executed
strengths helps; it does not establish feasibility or guarantee convergence.
