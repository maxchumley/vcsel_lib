# Matched density PPO experiments

Both recipes start from fresh weights with seed 7 and preserve the last
`train_density_actor_critic_replay_asymphase_1000` experiment's contexts,
physics, architecture, symmetry constraints and exploration/LR schedules.
Neither constructs a critic, archives experiences, or maintains historical replay.

| Recipe | Explicit source-minus-receiver target offset | Progress image |
| --- | --- | --- |
| `train_density_ppo_1000` | Off, matching the previous mixed-phase run | `density_ppo_progress.png` |
| `train_density_ppo_target_offset_1000` | On | `density_ppo_target_offset_progress.png` |

Both retain the common laser-1 delay-frequency rotation. The second adds
`target_phase_offsets(features)` to the directed phase mean before sampling:
the encoded source target angle minus the encoded receiver target angle.
Its learned base phase and strength-conditioned phase residual remain trainable.
The switch is checkpointed and also applies to validation and local inspection.
There are no added model weights for this offset.

Each iteration simulates 7 sizes (3–9 lasers), 5 target/detuning contexts per
size, 5 density requests per context and 8 mirrored candidates per request:
**1,400 training simulations**, each using the existing 500 ns trajectory.
Detuning distributions remain random, with spans through 4 GHz. Topologies
are connected and symmetric; strengths are symmetric and phases are directed.

PPO makes at most **four total** full-batch actor updates, rather than one
REINFORCE update plus four more updates. All use measured simulator rewards
and the existing pair-excluded baseline, computed/scaled once per batch.
The existing PCGrad size/density grouping and gradient clipping are retained.
The clip ratio is 0.1. Historical batches are discarded after their updates.
There is no entropy loss or value-function loss. Initial LR is 1e-4, with the
original taper to 1e-5 at iterations 900–1000. Exploration still tapers to zero
at iteration 1000; that final mean-only batch skips optimizer updates.

Likelihoods score one canonical strength per unordered pair, all directed
phases and the budget jointly. Pruned latent coordinates are included because
strength samples determine connectivity. Conditional phases are rescored
using the original sampled strengths, not newly predicted strengths. Ordinary
and wide slots retain their collection-time widths during the updates.

After every update, the old-to-new Gaussian KL is summed over a complete
design and averaged separately for each array size. Its conditional phase
part is evaluated at the collected strength samples. If any size exceeds
0.01, remaining updates stop. This is early stopping, not a rollback or a
guarantee of reward improvement. Logs/history include update count, maximum
size-mean KL, clipped fraction and whether the KL stop was triggered.

Validation remains 448 simulations at the first, every 50th and final
iteration. Progress images update at the first, every 10th and final iteration.
The existing two-panel training plot is retained and labeled PPO.

From the repository root on Great Lakes, after syncing these code changes:

```bash
bash scripts/submit_density_ppo_tests.sh --dry-run
bash scripts/submit_density_ppo_tests.sh --smoke-test
bash scripts/submit_density_ppo_tests.sh
```

The launcher submits two independent jobs named `density_ppo` and
`density_ppo_offset`, each with 32 CPUs, 31 simulation workers, 64 GB and seven
days. It does not stop existing jobs. Models and figures use separate run
directories and image names compatible with the existing sync script.
