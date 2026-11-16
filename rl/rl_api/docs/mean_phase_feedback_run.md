# Mean-design teachers with a separate phase-only update

Run the single cell in `experiments/train_mean_phase_feedback_1000.py`, or:

```bash
python -m rl.rl_api.experiments.train_mean_phase_feedback_1000
```

Restart the Python kernel first so the updated API is loaded. Importing only
previews. Executing starts fresh in a separate folder; the completed feedback
run and its checkpoints are preserved.

## Settings retained

The only recipe differences from `train_phase_feedback_1000.py` are the run
name and `phase_correction.mode="mean_phase_only"`. The architecture, 3–9 laser
sizes, random-root connectivity, optional feedback links, root-detuning delay
reference, root encoder input, mirrored sampling, RMS reward, PCGrad groups,
1,000 iterations, 500 ns simulations and learning-rate schedule are retained.
The reference frequency has not been removed.

Any laser can be selected as the construction root. Ordinary RL still updates
all model weights. Only the additional correction step freezes shared features.
There is no new network head and no new checkpoint parameter layout.

## Each training iteration

1. Run the usual 1,400 sampled simulations and the existing RL/PCGrad/Adam update.
2. On the same contexts and topology masks, decode the updated model's mean
   strength, budget and phase actions. This is not an average of sampled matrices.
3. Simulate the 175 mean designs. A teaching candidate must have finite states,
   frequency spread below 1 MHz and relative-phase jitter below 1 degree.
4. Construct `phi_new_ij = wrap(phi_mean_ij + beta_j - beta_i)` from its achieved
   settled phases. Verify using the **identical mean strength matrix and budget**.
   Require continued locking, improved RMS reward and worst-laser RMS <=3 degrees.
5. Cache the existing decoder's frozen shared features. Take one dedicated
   gradient-descent step on the mean circular phase-teaching loss, updating only
   row 1 of `edge_output.weight` and `edge_output.bias`. That row predicts phase
   means. Strength, budget and standard-deviation outputs and all shared weights
   are untouched during this step.

The correction step uses the current RL learning rate and the existing gradient
clip and correction weight (1.0). It does not reuse Adam momentum or pass through
PCGrad. If it fails to decrease the fixed teaching loss, the phase row is restored
and the step is skipped. The actual decoded mean strength matrices are checked
again afterward; any change raises an error.

The same-strength guarantee applies to this dedicated correction step. The next
ordinary RL update remains free to change strengths, phases, budget and shared
features. Restricting the correction to an existing linear output row limits its
capacity; this is a controlled training test, not a guarantee of unseen accuracy.

## Cost and monitoring

Each nonfinal iteration adds 175 mean-design simulations and up to 175 correction
verifications: at most **1,750 simulations total**, versus the previous test's
maximum of 1,575. Verification uses the existing worker pool.

The progress line reports:

- `phase-teachers=accepted/attempted`: corrected mean designs passing verification.
- `mean-locked`: the fraction of the 175 mean designs passing the locking checks.
- `correction-loss=before->after`: the restricted step's fixed-label loss.
- `strength-change=0`: maximum absolute decoded-strength change in that step.

These values, the number of mean checks and whether a correction step was kept
are stored in `current_history.json`. The training figure is unchanged.
Validation and inference use unassisted predictions. The final exact-mean
iteration performs no optimizer update or extra teaching simulations;
`mean-locked=nan%` then means it was not measured, not that locking failed.

The checkpoint sampling metadata records the correction mode, acceptance rule
and update scheme. The phase-only step has no additional optimizer state.

## Batch usage

```bash
python -m rl.rl_api.experiments.train_batch train_mean_phase_feedback_1000 --preview
sbatch scripts/train_rl_great_lakes.slurm train_mean_phase_feedback_1000 --smoke-test
sbatch scripts/train_rl_great_lakes.slurm train_mean_phase_feedback_1000
```

The updated code must be present on the cluster before submission. Setting up
this recipe does not submit a job or start the full training run.
