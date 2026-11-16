# Four lasers, in-phase targets, and all connected symmetric topologies

Recipe: `experiments/train_four_laser_inphase_all_symmetric_1000.py`.
Cluster submission: `scripts/train_four_laser_symmetric_great_lakes.slurm`.
This is a separate, fresh model; the density-conditioned experiment is unchanged.

## Scientific setup

- Exactly four lasers; target phases all zero relative to laser 1.
- Five new random detuning distributions per iteration, with one span from
  each of the five bins spanning 0–4 GHz. Span means peak-to-peak bandwidth.
- Every context is evaluated on **all 38 connected labeled symmetric masks**:
  16 trees (three reciprocal pairs), 15 four-pair graphs, six five-pair graphs,
  and the six-pair all-to-all graph. Disconnected masks are excluded. Labels
  follow sorted detuning order, so differently wired labeled graphs matter.
- Eight mirrored continuous actions per context and graph: **1,520 physical
  simulations per iteration**, for 1,000 iterations.
- The actual binary topology is supplied to the existing GNN. This experiment
  does not use the density-only input or a learned topology selector.
- Existing 64-wide node/message layers, three message steps, 128/128 edge MLP,
  edge/budget/SD heads and parameter count are unchanged.
- `INITIAL_CHECKPOINT=None`: seed-7 random initialization and new calibration.
- Native corrected trapezoid integrator, standard LK model, 500 ns, zero noise,
  same coupling ramp, wrapped-RMS reward, pair-excluded baseline and annealing
  schedules as the existing mirrored runs. The phase baseline retains laser 1's
  lowest mean-centered free-running detuning. There is no designated graph root.

## Exact value symmetry

Before physical decoding, average each pair of reciprocal **unwrapped latent**
strength and phase actions. Then apply the existing softplus strength transform,
budget normalization, phase wrapping and symmetric physical mask. Consequently:

```text
kappa[i,j] = kappa[j,i]
phi_p[i,j] = phi_p[j,i]
```

This is a deterministic output projection, with no new trainable weight.
REINFORCE still scores the original independent directed Gaussian samples,
before averaging; it does not incorrectly treat duplicated pair values as
independent observations. For the in-phase target, both the target phase offset
and the reference-delay offset are already symmetric. Total budget remains
the predicted sum of directed strengths, so each reciprocal pair contributes
both directions to that sum. The complete mask bank is saved in
`symmetric_topology_bank.json` and checkpoint metadata.

Training remains phase-only: no link penalty, budget penalty or feedback
teacher is added. The experiment measures feasibility under the specified
symmetry constraints. Connectivity alone does not establish physical locking.

Validation uses ten fixed new detuning contexts, each on all 38 masks:
**380 simulations** at iteration 1, every 50 iterations, and the final one.
Three PCGrad tasks use excess reciprocal-pair density: tree, four-pair, and
five/six-pair masks, weighted by actual rollout count. Plots retain two panels
and update every ten iterations; checkpoints update every iteration.

## Parallel execution on Great Lakes

The wrapper requests account `arrays0`, one standard node, one Slurm task,
32 CPUs, 64 GB and 48 hours. It reuses the existing working Conda activation
and explicit Python path. One CPU is reserved for the trainer; 31 processes
simulate independent batches, with one BLAS/Torch thread per process.

The previous process-pool cap used the number of contexts. For this single-size
experiment that would limit the pool to five workers. The trainer now honors
the configured worker count. At the default batching, five contexts × ten
candidate slots give **31 vectorized jobs** per iteration: 30 batches of 50
trajectories and a final batch of 20. These chunks can cross graph boundaries;
the learning rewards still compare eight actions within each graph. Validation
flattens all graph/context combinations before dispatch, keeping workers busy
there as well. No simulation time step, physical parameter or learning batch
is changed by this scheduling.

After transferring the updated code to the server, from the repository root:

```bash
python -m rl.rl_api.experiments.train_batch train_four_laser_inphase_all_symmetric_1000 --preview
mkdir -p logs
sbatch --time=01:00:00 scripts/train_four_laser_symmetric_great_lakes.slurm --smoke-test
```

After the launch check succeeds:

```bash
sbatch scripts/train_four_laser_symmetric_great_lakes.slurm
```

Each submission has a separate `logs/four-sym-JOBID.out/.err` and output under:

```text
rl/rl_api/runs/great_lakes/<RUN_NAME>/<JOBID>/
```

The smoke test retains the full 1,520-case batch and 380-case validation, but
compresses action annealing into two iterations. It tests launch/execution;
its final weights are not evidence of completed training. The API does not
resume optimizer/sampler state after interruptions.

## Tune CPU count and vectorization

The new log records `sim=`, `update=` and `validation=` wall seconds, also in
the JSON history. Compare actual timings rather than assuming linear scaling.
More processes reduce independent work per CPU; bigger vectorized batches
reduce repeated Python time-step overhead. Either can win on a particular node.
These two examples keep the scientific batch and exploration unchanged:

```bash
# Same vectorized batches, 15 workers instead of 31.
sbatch --cpus-per-task=16 --time=01:00:00 scripts/train_four_laser_symmetric_great_lakes.slurm --smoke-test

# 31 workers, smaller eight-trajectory batches (190 jobs total).
sbatch --time=01:00:00 scripts/train_four_laser_symmetric_great_lakes.slurm --smoke-test --pairs-per-job 1 --actions-per-job 8
```

Pass `--pairs-per-job` and `--actions-per-job` to change only vectorization.
The wrapper automatically chooses CPUs minus one workers. A CPU request of 36
would use 35 workers. Multiple nodes/Slurm tasks do not automatically distribute
this Python process pool; independent experiments can instead run as separate
Slurm jobs. The CPU NumPy integrator, not the small GNN, dominates the rollout
work, so a GPU allocation does not accelerate the existing integrator.

Great Lakes currently lists 36 physical cores per standard node:
[U-M hardware configuration](https://documentation.its.umich.edu/node/4976).
Concurrent jobs remain subject to the account's available allocation and limits:
[partition/account policies](https://documentation.its.umich.edu/arc-hpc/greatlakes/policies).

## Inspect

Load this run's checkpoint, set four lasers with in-phase targets, and request
38 candidates to cover the complete bank once. Scalar density is not an input:

```python
model = CouplingModel.load('/path/to/run/final.pt')
context = Context(np.zeros(4), [-2., -.6, .3, 2.])
designs = model.propose(context, count=38, seed=7)
results = designs.evaluate()
```

Use an explicit connected symmetric `Topology` to inspect one prescribed graph.
Mean inference on the same graph is deterministic. The existing inspector
automatically follows the saved model mode; set `N_LASERS=4`, `N_CLUSTERS=1`,
`NUMBER_OF_CANDIDATES=38`, and point `RUN_DIRECTORY` at the desired Slurm job.
