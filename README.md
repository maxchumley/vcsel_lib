# vcsel_lib

VCSEL simulation utilities for delayed-coupling arrays with optional self-feedback, noise, and optical injection. The library separates physical (SI) parameters from nondimensional parameters used by the solver and provides helpers for histories, equilibria, and stability analysis.

This README is organized as:
1. Quick start
2. Core concepts and data shapes
3. Main API
4. Equilibrium solving and stability
5. Coupling and injection utilities
6. Examples

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

**History arrays**
- Delay solver needs `t in [-2*tau, 0]`.
- History shape: `(n_cases, 3*N, 2*delay_steps)`

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

### `generate_history(nd, shape="FR", n_cases=1, counts=None, guesses=None)`
Generates history for the delay solver. Always returns four values:
`history, freq_hist, eq, results`

Supported shapes:
- `"FR"`: free-running equilibrium history
- `"ZF"`: zero-field history
- `"EQ"`: equilibrium histories based on solving for equilibria

Notes:
- For `"EQ"`, `counts` and `guesses` are passed to `solve_equilibria`.
- For non-`"EQ"` shapes, `results` is `None`.

### `integrate(history, nd=None, progress=False, theta=0.5, max_iter=5)`
Integrates the nondimensional DDE with a trapezoidal predictor-corrector plus optional Euler–Maruyama noise. Returns:
`t_dim, y, freqs`

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

## 6) Examples

Examples live in `examples/` and `examples/working_examples/`. A few useful entry points:
- `examples/simple_example.py`
- `examples/verify_stability.py`
- `examples/working_examples/verify_stability.py`
- `examples/working_examples/stability_manifold.py`

## Notes and tips

- If you pass time-varying `kappa` or `phi_p` into functions that expect a single matrix, the code will use the last entry and (optionally) warn when `verbose=True`.
- For high coupling strengths, the adaptive grid can uncover more equilibria without manually expanding the grid size.
- For deterministic testing, set `noise_amplitude=0`.
