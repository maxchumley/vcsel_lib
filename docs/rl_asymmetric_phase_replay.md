# Symmetric strengths with independent directed phases

```bash
bash scripts/submit_density_ac_replay_asymphase.sh
```

This adds one fresh job, `density_ac_asymphase`, without cancelling existing
training. It requests 32 CPUs, 31 simulation workers, 64 GB and seven days on
`arrays0`. Add `--dry-run` to preview or `--smoke-test` for two iterations.

The recipe matches the fresh actor–critic replay run: sizes 3–9, random phase
targets and detuning distributions with spans up to 4 GHz, five density requests
per context, eight mirrored actions and five contexts per size. It uses 1,400
real 500 ns simulations per iteration, 1,000 iterations, a 100,000-design replay
capacity, the same critic quality gate, validation schedule, learning rates and
exploration annealing. Two of eight actions retain wider phase exploration.

The only physical constraint relaxed is phase reciprocity. Masks remain
symmetric and connected. Strengths share one canonical Gaussian action per
unordered pair and are exactly equal in both directions after decoding. Phases
use separate directed Gaussian actions and separate strength-conditioned
residuals. Each marginal exploration width matches the reciprocal baseline,
including the sqrt(2) width adjustment. The score counts each strength once
and each directed phase once. The network has the same layers and parameter
count; reverse-direction phase outputs are now used. The common laser-1 delay
reference remains in place, with no additional target-phase offset injected.

The opt-in `ModelConfig.asymmetric_coupling_phases=True` is recorded in saved
checkpoints, so inference and critic-generated samples use the same mixed
constraints. It defaults to false, preserving previous recipes and checkpoints.
The actor predicts phases itself; inspection does not apply feedback corrections.

The unique image is `density_ac_replay_asymphase_progress.png`. Existing plot
and checkpoint discovery automatically finds this run after its first output.
The run name is `density_conditioned_ac_replay_symstrength_asymphase_fresh_1k`.
