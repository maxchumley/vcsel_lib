# vcsel_lib

VCSEL simulation utilities for delayed-coupling arrays with optional self-feedback, noise, and optical injection. The library separates physical (SI) parameters from nondimensional parameters used by the solver and provides helpers for histories, equilibria, and stability analysis.

This README is organized as:
1. Quick start
2. Core concepts and data shapes
3. Main API
4. Equilibrium solving and stability
5. Coupling and injection utilities
6. Examples

For the repository layout, generated-data policy, and cleanup commands, see
[`docs/repository_maintenance.md`](docs/repository_maintenance.md).

## 1) Quick start

```python
import numpy as np
from vcsel_lib import VCSEL

# --- physical parameters (SI units) ---
phys = {
    "tau_p": 5.4e-12,
    "tau_n": 0.25e-9,
    "g0": 8.75e-4 * 1e9,
    "N0": 2.86e5,
    "s": 4e-6,
    "beta": 1e-3,
    "kappa_c_mat": np.zeros((1000, 2, 2)),
    "phi_p_mat": np.zeros((1, 2, 2)),
    "I": 1.0e-3,
    "q": 1.602e-19,
    "alpha": 2.0,
    "delta": np.array([0.0, 0.0]),
    "coupling": 1.0,
    "self_feedback": 0.0,
    "noise_amplitude": 0.0,
    "dt": 1e-12,
    "Tmax": 1e-9,
    "tau": 1e-9,
    "N_lasers": 2,
    "save_every": 1,
    "max_output_gb": 10.0,
}

vcsel = VCSEL(phys)
nd = vcsel.scale_params()

# Create a history and integrate
history, freq_hist, eq, results = vcsel.generate_history(nd, shape="FR", n_cases=1)
t_dim, y, freqs = vcsel.integrate(history, nd=nd, progress=False)
```

## 2) Core concepts and data shapes

**Physical vs nondimensional parameters**
- `phys`: SI-unit parameters you provide.
- `nd`: nondimensional parameters produced by `scale_params()`.

**State vector ordering**
- For each laser: `[n, S, phi]`
- For N lasers, the state ordering is:
  `n1, S1, phi1, n2, S2, phi2, ..., nN, SN, phiN`
- State arrays:
  - Single state: shape `(3*N,)`
  - Time series: shape `(n_cases, 3*N, steps)`
  - With `save_every > 1`, integration outputs are compressed to saved points only.

**History arrays**
- Delay solver needs `t in [-2*tau, 0]`.
- History shape: `(n_cases, 3*N, 2*delay_steps)`
- `freq_hist` returned by `generate_history` is in **GHz**.

**Coupling matrices**
- `kappa` typically has shape `(steps, N, N)` for time-varying coupling.
- Some functions accept a single matrix `(N, N)` for constant coupling.

**Phase offsets**
- `phi_p` can be scalar, `(N, N)`, or `(n_cases, N, N)` depending on context.

## 3) Main API

### `VCSEL(phys_params)`
Construct a simulator using physical parameters.

### `scale_params() -> nd`
Creates nondimensional parameters (`nd`) used by all simulation routines.

Key outputs in `nd`:
- `dt`, `Tmax`, `steps`, `delay_steps`
- `kappa`, `phi_p`, `delta_p`
- `nbar`, `sbar`, `Gs`, `beta_n`, `beta_const`
- `save_every`, `max_output_gb`
- injection controls (`injection`, `injection_topology`, `injected_strength`, `injected_frequency`, `kappa_inj`)

### `generate_history(nd, shape="FR", n_cases=1, counts=None, guesses=None)`
Generates history for the delay solver. Always returns four values:
`history, freq_hist, eq, results`

Supported shapes:
- `"FR"`: free-running equilibrium history
- `"ZF"`: zero-field history
- `"EQ"`: equilibrium histories based on solving for equilibria

Notes:
- For `"EQ"`, `counts` and `guesses` are passed to `solve_equilibria`.
- For `"EQ"`, output case count equals the number of equilibria found.
- For non-`"EQ"` shapes, `results` is `None`.
- `freq_hist` is returned in GHz.

### `integrate(history, nd=None, progress=False, theta=0.5, max_iter=5, smooth_freqs=True)`
Integrates the nondimensional DDE with a trapezoidal predictor-corrector plus optional Euler–Maruyama noise. Returns:
`t_dim, y, freqs`

If `save_every == 1`, output shapes are `(n_cases, 3*N, steps)` and `(n_cases, N, steps)`.
If `save_every > 1`, both are returned only on the saved time grid.

Important integration controls in `nd`:
- `save_every` (int): save every k-th sample.
- `max_output_gb` (float or `None`): auto-increase `save_every` to cap output size.
- `delay_interp` (`None`, `"linear"`, or `"cubic"`): delayed-state interpolation mode.
- `noise_substeps` (int): stochastic substeps per deterministic step.
- `smooth_window_delays` (float): smoothing window length in units of delay for frequency smoothing.

Returned `freqs` are nondimensional phase derivatives (`dphi/dt'`). Convert to GHz with:
`freq_ghz = freqs * 1e-9 / (2*np.pi*tau_p)`

### `f_nd(x, x_tau, x_2tau, j, phi_p, nd=None)`
Vectorized nondimensional VCSEL rate equations used by the integrator.

### `compute_noise_sample(y_c, noise_amplitude, dt, nd)`
Noise increments for Langevin noise.

### `invert_scaling(y, phys)`
Convert nondimensional states back to physical units (in-place).

### `order_parameter(y_segment)`
Computes Kuramoto-like order parameter over a segment.

## 4) Equilibrium solving and stability

### `solve_equilibria(nd, guesses=None, counts=None, n_jobs=-1)`
Finds equilibria using an adaptive phase/omega grid.

Adaptive grid settings via `counts`:
- `phase_count` (default 30)
- `freq_count` (default 100)
- `adaptive_grid` (default True)
- `refine_factor` (default 2)
- `max_refine` (default 3)

Returns:
`final_root, results, E_tot`

### `residuals(x, nd=None, verbose=False)`
Residuals for equilibrium solving. Set `verbose=True` to emit warnings when stacked/3D parameters are reduced.

### `compute_jacobians(x, nd, verbose=False)`
Analytic Jacobians at a rotating-frame equilibrium.

### `compute_spectrum(A_list, tau_list, N, ..., verbose=False)`
Chebyshev collocation spectrum for DDE stability.

### `compute_stability(x_eq, nd, ..., verbose=False)`
Convenience wrapper that builds Jacobians and computes eigenvalues.

## 5) Coupling and injection utilities

### `cosine_ramp(t, t_start, rise_10_90, kappa_initial=0.0, kappa_final=1.0)`
Smooth half-cosine ramp.

### `build_coupling_matrix(time_arr, kappa_initial, kappa_final, N_lasers, ramp_start, ramp_shape, tau, scheme="ATA", aMAT=None, dx=None, plot=False)`
Builds a time-varying coupling matrix using a cosine ramp.

Supported schemes:
- `"ATA"`: all-to-all
- `"NN"`: nearest neighbors
- `"CUSTOM"`: uses adjacency matrix `aMAT`
- `"RANDOM"`: random adjacency
- `"DECAYED"`: distance-based decay (uses `dx`)

Injection setup notes:
- Set `phys["injection"] = True` and `phys["injection_topology"]` (shape `(N,)`, boolean/int mask).
- Provide `injected_strength`, `kappa_injection`, and `injected_frequency`.
- Set `nd["injected_phase_diff"]` before calling `integrate()` (scalar or per-case array).

## 6) Examples

Examples live in `examples/`. A few useful entry points:
- `examples/basics/simple_example.py`
- `examples/stability/verify_stability.py`
- `examples/stability/stability_manifold.py`

### Reinforcement-learning matrix design

`rl/reinforcement_matrix_design.py` is a small, dependency-free REINFORCE example
that learns static `kappa` and `phi_p` matrices from batched VCSEL rollouts. Each
training rollout ramps `kappa` from zero using the same schedule as validation:

```bash
python -m rl.reinforcement_matrix_design
```

Edit `target_phases` and the reward weights near the top of the file to describe
the desired phase pattern and frequency-locking behavior. The phase objective
blends the average phase score with the score of the worst non-reference laser;
`worst_laser_phase_weight` controls that split. Training uses `n_restarts`
independently seeded searches, with `n_iterations` iterations in each restart.
Up to `parallel_restarts` searches run concurrently, and the best design across
all of them is kept. The inline figure updates after every parallel iteration
and overlays each restart's mean and best histories with the global-best and
worst-laser progress. The final matrices and histories are saved in
`rl/artifacts/data/reinforcement_matrix_design.npz`. Matrix entry `[i, j]` describes delayed
source `j` coupling into receiver `i`; `kappa_ns` is in ns^-1 and `phi_p` is in
radians. A final cell reloads that file, ramps the learned coupling matrix on,
and plots frequency, each laser's wrapped phase relative to laser 1, and
output-power trajectories in the same format as
`examples/basics/simple_example.py`.

### Closed-loop neural coupling control

`rl/neural_policy_matrix_design.py` uses replay-buffered TD3 for continuous
closed-loop control:

```bash
python -m pip install -e '.[rl]'
python -m rl.neural_policy_matrix_design
```

```text
requested/current phases + explicit error + recent history
                              |
                    deterministic TD3 actor
                              |
              90 delta-kappa + 90 delta-phi_p actions
                              |
               integrate one continuous DDE chunk
                              |
                n-step replay, twin critics, repeat
```

Every episode begins from a newly generated noise-free free-running delay
history with zero coupling and a random phase target. The observation contains
the target and current relative phases, their explicit wrapped errors, photon
powers, current controls, and a five-decision history of phase errors and
relative frequencies. A `512 -> 512 -> 256` actor independently controls all
90 directed off-diagonal `kappa` entries and all 90 corresponding `phi_p`
entries. The diagonal remains zero.

The actor outputs link increments rather than complete matrices. Kappa
increments are bounded by `maximum_kappa_change_per_step_ns`, and each
magnitude is clipped to `[0, maximum_kappa_ns]`. Phase increments are bounded
by `maximum_phi_p_change_per_step_rad` and wrapped to `[-pi, pi]`. The
integrator linearly ramps both controls from their previous values to their
new values during the next interval, making them continuous and slew-limited.

After each interval, the integrator returns the final full-resolution `2*tau`
delay history. That exact history becomes the next step's input, so an episode
is one continuous DDE trajectory rather than a sequence of restarted
simulations. The reward contains no explicit frequency-locking term. It combines the
time-averaged requested-phase score, improvement from the preceding control
step, and additional terminal phase reward. Frequency remains a plotted
diagnostic. Success requires both the mean phase score and the weakest laser's
phase score to remain high for several consecutive steps; sustained drifting
phases therefore cannot count as success.

Each transition is stored in a replay buffer. Five-step returns preserve the
discounted reward from a short action sequence, and twin critics bootstrap
from stored outcomes while the delayed actor is updated less frequently.
Target networks and clipped target-action noise stabilize off-policy learning.

The default training run contains 10 parallel free-running episodes, 50
control decisions spanning 100 ns per episode, and two TD3 gradient updates
per simulation step after `learning_starts` transitions are stored.
Deterministic evaluation uses the same fixed targets. The
prominent evaluation curves and matrix panels always show the result for
`requested_phases`; held-out means remain available in the saved histories.
The checkpoint is saved to `dynamic_coupling_td3.pt`. Final validation plots
frequency, relative phase, power, the time-varying kappa range, and
representative time-varying `phi_p` links.

## Notes and tips

- If you pass time-varying `kappa` or `phi_p` into functions that expect a single matrix, the code will use the last entry and (optionally) warn when `verbose=True`.
- For high coupling strengths, the adaptive grid can uncover more equilibria without manually expanding the grid size.
- For deterministic testing, set `noise_amplitude=0`.
