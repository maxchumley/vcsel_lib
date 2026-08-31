# %%
"""Linewidths of every stable external-cavity mode (ECM).

This example keeps the fixed-kappa stochastic linewidth calculation from
``linewidth_autocorr.py``, but replaces its free-running continuation history
with an exact rotating-wave history for every stable equilibrium returned by
``VCSEL.solve_equilibria`` and ``VCSEL.compute_stability``.

The output layout is deliberately flat: one result file corresponds to one
ECM at one kappa.  This avoids allocating a dense ``(kappa, branch, frequency)``
array when the number of equilibria changes along the sweep.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue as queue_module
import threading
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

try:
    from examples._paths import LINEWIDTH_RESULTS_DIR
except ModuleNotFoundError:
    LINEWIDTH_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "linewidth_estimation"

import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

from vcsel_lib import VCSEL

try:
    # Import implementation helpers only. All parameter values used by this
    # script are defined below and overwrite the helper module's globals.
    from examples import linewidth_autocorr as linewidth_engine
except ImportError:
    # Same helper-only import for ``python examples/all_stable_linewidths.py``.
    import linewidth_autocorr as linewidth_engine


RESULT_SCHEMA_VERSION = 2


# ----------------------------- User controls -----------------------------

# This is a self-contained parameter block. Nothing here is inherited from
# linewidth_autocorr.py, so changing that example cannot alter this sweep.
use_ma_et_al_2019_parameters = False
ma_et_al_2019_delay_regime = "long_fiber"  # "short", "long", or "long_fiber"

N_lasers = 1
kappa_initial = 0.0
kappa_final = 40.0e9
n_kappa_steps = 500
detuning_ghz = 0.0

alpha = 2.0
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1.0e9
N0 = 2.86e5
saturation = 4.0e-6
q = 1.602e-19
beta = 1.0e-3
tau = 1.0e-9
eta = 0.9
threshold_multiplier = 3.0
phi_p = 0.0

# Each stable ECM receives this many independent Langevin-noise trajectories.
n_noise_iterations = 50
noise_amplitude = 1.0
random_seed = None

# ``linewidth_analysis_time`` is the retained record used by the PSD. After
# the integrator consumes its deterministic two-delay history, a noisy burn-in
# is simulated and discarded because the ECM has no stationary noise history
# at t=0.
linewidth_analysis_time = 5.0e-6
noise_burn_in_time = 1.0e-6
minimum_burn_in_decay_time_constants = 5.0
dt_multiplier = 1.0

output_name_suffix = ""

# --- Optional Ma et al. (2019) parameter preset --------------------------
# Source: Ma et al., Appl. Sci. 2019, "Linewidth Narrowing of Mutually
# Injection Locked Semiconductor Lasers with Short and Long Delay", Table 1.
# These values intentionally duplicate linewidth_autocorr.py so this file
# remains independently editable and does not import that example's globals.
if use_ma_et_al_2019_parameters:
    output_name_suffix = "_ma_et_al"

    N_lasers = 2
    n_noise_iterations = 50
    n_kappa_steps = 500

    alpha = 4.0
    tau_p = 7.15e-12
    tau_n = 0.33e-9
    g0 = 1.13e4
    N0 = 8.2e6
    saturation = 0.0
    beta = 3.54e-5
    eta = 1.0
    threshold_multiplier = 4.0
    phi_p = 0.0

    if ma_et_al_2019_delay_regime.lower() == "short":
        tau = 5.0e-12
        kappa_initial = 0.0
        kappa_final = 2.0e10
        detuning_ghz = 0.0
        dt_multiplier = min(float(dt_multiplier), 0.25)
    elif ma_et_al_2019_delay_regime.lower() == "long":
        # Fig. 4 numerical long-delay time-domain example.
        tau = 5.0e-9
        kappa_initial = 0.0
        kappa_final = 5.0e9
        detuning_ghz = 0.3
    elif ma_et_al_2019_delay_regime.lower() == "long_fiber":
        # Fig. 15 / fiber-link linewidth estimate: 10 m coupling fiber.
        tau = 50.0e-9
        kappa_initial = 0.0
        kappa_final = 1.0e9
        detuning_ghz = 0.0
    else:
        raise ValueError(
            "ma_et_al_2019_delay_regime must be 'short', 'long', or "
            "'long_fiber'."
        )

# Keep this >= 2.  VCSEL.integrate then uses its delay-sized ring buffer and
# returns only selected phase states instead of allocating a full-state array
# for every integration step.
analysis_save_every = 2
analysis_output_dtype = np.float64
max_output_gb_per_worker = 10.0

# Each worker concatenates this many ECMs, each with n_noise_iterations cases,
# into one vectorized integration. A value of 5 means 250 trajectories per
# worker when n_noise_iterations=50.
long_run_branch_chunk_size = 5
long_run_jobs = 10
queued_long_job_waves = 1  # one active wave plus one small queued wave
serialize_psd_and_saving = True
show_long_run_integration_progress = True
long_run_progress_update_steps = 2000
# Fork is compatible with this project's IPython workflow. Workers are
# eagerly started before any sparse eigensolver/BLAS work to make that choice
# safe on macOS. A standalone script may instead use "spawn".
multiprocessing_start_method = "fork"

# The common phase is the primary linewidth coordinate.  It remains defined
# for anti-phase ECMs whose summed optical field is nearly dark.  Individual
# laser linewidth statistics are also saved; their full PSD traces are not.
psd_floor_band_hz = (0.0, 2.0e6)
psd_welch_nperseg = None
psd_welch_overlap_fraction = 0.5
save_full_common_psd_for_all_ecms = False
full_psd_branch_ids: set[int] = set()
full_psd_kappa_indices: set[int] = set()
maximum_saved_psd_points = 6000
save_per_case_linewidths = True

# A weakly stable/noisy realization can leave the ECM basin. The full final
# 2*tau history is cheap to return even though the long output is phase-only.
# Only realizations that remain near the seeded root contribute to linewidth.
validate_final_ecm = True
minimum_valid_noise_cases = 3
final_photon_relative_tolerance = 0.50
final_carrier_absolute_tolerance = 0.25
final_relative_phase_tolerance_rad = 0.75
final_relative_phase_min_coherence = 0.70
final_frequency_tolerance_ghz = 0.50

# ``equilibrium_solver_jobs`` parallelizes the independent initial guesses
# inside solve_equilibria for one kappa. Kappa values themselves remain
# sequential because branch matching uses the preceding kappa's roots.
# Stability spectra are also checked root-by-root in the parent process.
equilibrium_counts = {
    "phase_count": 5,
    "freq_count": 50,
    "max_refine": 2,
    "refine_factor": 2,
}
equilibrium_solver_jobs = 10
stability_collocation_order = 30
stability_newton_max_iterations = 10000
stability_sparse = True
stability_spectral_shift = 0.01 + 0.01j
stability_real_tolerance = 0.0
require_nonempty_stability_spectrum = True
# Sparse shift-invert is fast for rejecting unstable roots. Confirm every
# provisional stable root with the complete dense collocation spectrum before
# spending hours on its stochastic linewidth.
confirm_stable_roots_with_dense_spectrum = True
save_stability_eigenvalues = False
# At kappa=0, identical uncoupled lasers have a physical neutral relative-phase
# continuum that compute_stability's near-zero filter cannot distinguish from
# a numerical zero. Do not call those arbitrary phase points locked ECMs.
skip_uncoupled_identical_laser_ecms = True

# Linear interpolation is important here.  Usually tau/dt is not an integer;
# without interpolation, an analytic rotating-wave history is not exactly
# consistent with the delay sampled by the DDE integrator.
delay_interpolation = "linear"

# Adjacent-kappa roots are matched only for plot labels.  The linewidth
# calculation itself does not depend on branch matching.
branch_match_carrier_scale = 0.10
branch_match_log_photon_scale = 0.50
branch_match_phase_scale_rad = 0.50
branch_match_frequency_scale_ghz = 0.50
branch_match_max_cost = 4.0

resume_existing_results = False
reuse_equilibrium_catalogs = False
# Per-ECM NPZ files are the crash-safe source of truth. Rebuild the small flat
# summary every few completions instead of rewriting it for every ECM.
summary_checkpoint_every = 25
write_overview_figure = True
write_kappa_ecm_psd_figures = True
# Rewrite the live overview while catalogs and linewidth results arrive. The
# minimum interval prevents a fast equilibrium sweep from spending most of its  
# time plotting; every long result still triggers an update once this interval
# has elapsed.
write_overview_progress = True
overview_progress_min_interval_seconds = 10.0
render_overview_in_disposable_process = True
overview_colormap = "jet"
overview_figure_size = (10.5, 8.2)
overview_figure_dpi = 200
overview_font_family = "serif"
overview_font_size = 16
overview_use_latex = True
overview_linewidth_ylim_mhz = (1.0e-2 , 1.0e10)
overview_linewidth_yticks_mhz = [
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1.0e3,
    1.0e4,
]
# Inset in the upper linewidth panel. It separates the ECMs at the final
# completed coupling by plotting their linewidth against their ECM frequency
# shift.
overview_linewidth_inset_bounds = (0.075, 0.53, 0.40, 0.39)
overview_linewidth_inset_ylim_hz = (1.0e4, 1.0e5)
overview_stable_catalog_marker_size = 10
overview_unstable_marker_size = 2.5
overview_completed_marker_size = 8
overview_insufficient_marker_size = overview_stable_catalog_marker_size

output_dir = Path(
    LINEWIDTH_RESULTS_DIR / f"{N_lasers}_lasers/"
    f"all_stable_linewidths{output_name_suffix}"
)
catalog_dir = output_dir / "equilibrium_catalogs"
ecm_result_dir = output_dir / "ecm_results"
summary_dir = output_dir / "summary"
kappa_psd_figure_dir = output_dir / "kappa_ecm_psds"
worker_error_dir = output_dir / "worker_errors"


# --------------------------- Engine configuration -------------------------

dt = dt_multiplier * tau_p
delay_history_time = 2 * int(tau / dt) * dt
analysis_start_time = delay_history_time + noise_burn_in_time
total_simulation_time = analysis_start_time + linewidth_analysis_time
adjacency = np.ones((N_lasers, N_lasers), dtype=float) #- np.eye(N_lasers)
detuning_distribution = np.linspace(
    -detuning_ghz * np.pi * 1.0e9,
    detuning_ghz * np.pi * 1.0e9,
    N_lasers,
)
current = eta * threshold_multiplier * q / tau_n * (N0 + 1.0 / (g0 * tau_p))


def configure_linewidth_engine():
    """Synchronize globals read by linewidth_autocorr's PSD helpers."""
    engine_values = {
        "N_lasers": N_lasers,
        "n_noise_iterations": n_noise_iterations,
        "Tmax_long": total_simulation_time,
        "dt_multiplier": dt_multiplier,
        "analysis_save_every": analysis_save_every,
        "analysis_output_dtype": np.dtype(analysis_output_dtype),
        "detuning_ghz": detuning_ghz,
        "long_run_jobs": long_run_jobs,
        "serialize_long_run_postprocessing": serialize_psd_and_saving,
        "max_output_gb": max_output_gb_per_worker,
        "psd_floor_band_hz": tuple(psd_floor_band_hz),
        "psd_welch_nperseg": psd_welch_nperseg,
        "psd_welch_overlap_fraction": psd_welch_overlap_fraction,
        "compute_phase_variance_linewidth": False,
        "long_run_history_is_steady_state": True,
        "phase_plot_tail_window_us": None,
        "noise_amplitude": noise_amplitude,
        "alpha": alpha,
        "tau_p": tau_p,
        "render_figures_in_disposable_process": (
            render_overview_in_disposable_process
        ),
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "saturation": saturation,
        "q": q,
        "beta": beta,
        "tau": tau,
        "eta": eta,
        "threshold_multiplier": threshold_multiplier,
        "phi_p": phi_p,
        "dt": dt,
        "adjacency": adjacency,
        "detuning_distribution": detuning_distribution,
        "current": current,
    }
    for name, value in engine_values.items():
        setattr(linewidth_engine, name, value)

    long_steps = int(np.floor(total_simulation_time / dt))
    linewidth_engine.long_steps = long_steps
    linewidth_engine.time_array_long = np.arange(long_steps, dtype=float) * dt
    linewidth_engine.delay_steps = int(tau / dt)


configure_linewidth_engine()


# ------------------------------- Utilities --------------------------------

SCALAR_RESULT_FIELDS = (
    "kappa_index",
    "solution_index",
    "branch_id",
    "kappa_hz",
    "frequency_ghz",
    "omega_nd",
    "leading_eigenvalue_real_nd",
    "leading_eigenvalue_real_per_second",
    "stability_eigenvalue_count",
    "stability_dense_confirmed",
    "equilibrium_order_parameter",
    "equilibrium_total_power",
    "psd_linewidth_common_hz",
    "psd_linewidth_common_mean_hz",
    "psd_linewidth_common_std_hz",
    "psd_linewidth_common_q16_hz",
    "psd_linewidth_common_q84_hz",
    "n_cases",
    "n_cases_total",
    "n_cases_valid",
    "valid_case_fraction",
    "burn_in_decay_time_constants",
    "effective_save_every",
    "saved_sample_dt_seconds",
    "retained_duration_seconds",
    "welch_nperseg",
    "psd_floor_bin_count",
    "psd_floor_loglog_slope",
    "saved_psd_points",
    "full_psd_saved",
    "resolved_run_seed",
    "job_seed",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def catalog_configuration_payload(kappa_values):
    """Configuration that changes equilibrium roots/stability/branch IDs."""
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "use_ma_et_al_2019_parameters": bool(use_ma_et_al_2019_parameters),
        "ma_et_al_2019_delay_regime": str(ma_et_al_2019_delay_regime),
        "N_lasers": N_lasers,
        "kappa_values": np.asarray(kappa_values),
        "detuning_ghz": detuning_ghz,
        "adjacency": adjacency,
        "physical": {
            "alpha": alpha,
            "tau_p": tau_p,
            "tau_n": tau_n,
            "g0": g0,
            "N0": N0,
            "saturation": saturation,
            "q": q,
            "beta": beta,
            "tau": tau,
            "eta": eta,
            "threshold_multiplier": threshold_multiplier,
            "phi_p": phi_p,
        },
        "equilibria": {
            "counts": equilibrium_counts,
            "stability_collocation_order": stability_collocation_order,
            "stability_newton_max_iterations": stability_newton_max_iterations,
            "stability_sparse": stability_sparse,
            "stability_spectral_shift": stability_spectral_shift,
            "stability_real_tolerance": stability_real_tolerance,
            "require_nonempty_spectrum": require_nonempty_stability_spectrum,
            "confirm_stable_with_dense": confirm_stable_roots_with_dense_spectrum,
            "skip_uncoupled_identical": skip_uncoupled_identical_laser_ecms,
        },
        "branch_matching": {
            "carrier_scale": branch_match_carrier_scale,
            "log_photon_scale": branch_match_log_photon_scale,
            "phase_scale_rad": branch_match_phase_scale_rad,
            "frequency_scale_ghz": branch_match_frequency_scale_ghz,
            "max_cost": branch_match_max_cost,
        },
    }


def linewidth_configuration_payload(catalog_signature, resolved_seed):
    """Configuration that changes a stochastic linewidth result."""
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "catalog_signature": catalog_signature,
        "noise": {
            "n_noise_iterations": n_noise_iterations,
            "long_run_branch_chunk_size": long_run_branch_chunk_size,
            "noise_amplitude": noise_amplitude,
            "resolved_seed": int(resolved_seed),
            "analysis_time": linewidth_analysis_time,
            "burn_in_time": noise_burn_in_time,
            "dt_multiplier": dt_multiplier,
            "analysis_save_every": analysis_save_every,
            "output_dtype": np.dtype(analysis_output_dtype).str,
            "max_output_gb": max_output_gb_per_worker,
            "delay_interpolation": delay_interpolation,
        },
        "psd": {
            "floor_band_hz": psd_floor_band_hz,
            "welch_nperseg": psd_welch_nperseg,
            "welch_overlap_fraction": psd_welch_overlap_fraction,
        },
        "final_ecm_validation": {
            "enabled": validate_final_ecm,
            "minimum_valid_noise_cases": minimum_valid_noise_cases,
            "photon_relative_tolerance": final_photon_relative_tolerance,
            "carrier_absolute_tolerance": final_carrier_absolute_tolerance,
            "relative_phase_tolerance_rad": final_relative_phase_tolerance_rad,
            "relative_phase_min_coherence": final_relative_phase_min_coherence,
            "frequency_tolerance_ghz": final_frequency_tolerance_ghz,
        },
    }


def storage_configuration_payload():
    """Non-scientific controls for the amount of diagnostic data on disk."""
    return {
        "save_full_for_all": save_full_common_psd_for_all_ecms,
        "full_psd_branch_ids": sorted(full_psd_branch_ids),
        "full_psd_kappa_indices": sorted(full_psd_kappa_indices),
        "maximum_saved_psd_points": maximum_saved_psd_points,
        "save_per_case_linewidths": save_per_case_linewidths,
        "write_overview_figure": write_overview_figure,
        "write_kappa_ecm_psd_figures": write_kappa_ecm_psd_figures,
        "write_overview_progress": write_overview_progress,
        "overview_progress_min_interval_seconds": (
            overview_progress_min_interval_seconds
        ),
        "overview_colormap": overview_colormap,
    }


def payload_signature(payload):
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_savez(path, **arrays):
    """Write an NPZ atomically so interrupted files are never resumed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def resolve_run_seed():
    """Reuse one generated seed across resumed runs when random_seed is None."""
    if random_seed is not None:
        return int(random_seed)
    seed_path = output_dir / "resolved_random_seed.txt"
    if seed_path.exists():
        try:
            return int(seed_path.read_text().strip())
        except ValueError:
            pass
    value = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(f"{value}\n")
    return value


def result_path_for(kappa_index, solution_index, branch_id):
    return ecm_result_dir / (
        f"kappa_{int(kappa_index):04d}_branch_{int(branch_id):04d}_"
        f"solution_{int(solution_index):03d}.npz"
    )


def catalog_path_for(kappa_index):
    return catalog_dir / f"kappa_{int(kappa_index):04d}.npz"


def equilibrium_guess(root):
    root = np.asarray(root, dtype=float)
    return np.concatenate(
        (
            root[1 : 2 * N_lasers : 2],
            root[2 * N_lasers : 3 * N_lasers - 1],
            root[-1:],
        )
    )


def canonicalize_and_deduplicate_roots(roots):
    """Remove periodic 0/2pi duplicates left by the equilibrium solver."""
    roots = np.asarray(roots, dtype=float)
    if roots.size == 0:
        return np.empty((0, 3 * N_lasers), dtype=float)
    if roots.ndim == 1:
        roots = roots.reshape(1, -1)
    unique = []
    phase_slice = slice(2 * N_lasers, 3 * N_lasers - 1)
    for candidate in roots:
        candidate = candidate.copy()
        candidate[phase_slice] = np.mod(candidate[phase_slice], 2.0 * np.pi)
        near_two_pi = np.isclose(
            candidate[phase_slice], 2.0 * np.pi, atol=1.0e-7, rtol=0.0
        )
        candidate[phase_slice][near_two_pi] = 0.0
        duplicate = False
        for accepted in unique:
            phase_difference = np.angle(
                np.exp(1j * (candidate[phase_slice] - accepted[phase_slice]))
            )
            nonphase_candidate = np.concatenate(
                (candidate[: 2 * N_lasers], candidate[-1:])
            )
            nonphase_accepted = np.concatenate(
                (accepted[: 2 * N_lasers], accepted[-1:])
            )
            if np.allclose(
                nonphase_candidate,
                nonphase_accepted,
                atol=1.0e-3,
                rtol=1.0e-3,
            ) and np.all(np.abs(phase_difference) <= 1.0e-3):
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return np.asarray(unique, dtype=float).reshape(-1, 3 * N_lasers)


def ecm_history(root, nd):
    """Build the exact rotating-wave history expected by VCSEL.integrate."""
    root = np.asarray(root, dtype=float)
    delay_steps = int(nd["delay_steps"])
    history_time_nd = np.arange(2 * delay_steps, dtype=float) * float(nd["dt"])
    offsets = np.concatenate(([0.0], root[2 * N_lasers : 3 * N_lasers - 1]))
    history = np.zeros((1, 3 * N_lasers, 2 * delay_steps), dtype=np.float64)
    history[0, 0::3, :] = root[0 : 2 * N_lasers : 2, None]
    history[0, 1::3, :] = root[1 : 2 * N_lasers : 2, None]
    history[0, 2::3, :] = root[-1] * history_time_nd[None, :] + offsets[:, None]
    return history


def equilibrium_observables(root):
    root = np.asarray(root, dtype=float)
    photon = np.maximum(root[1 : 2 * N_lasers : 2], 0.0)
    phase = np.concatenate(([0.0], root[2 * N_lasers : 3 * N_lasers - 1]))
    field = np.sqrt(photon) * np.exp(1j * phase)
    total_power = float(np.abs(np.sum(field)) ** 2)
    denominator = float(max(N_lasers * np.sum(photon), np.finfo(float).tiny))
    return {
        "frequency_ghz": float(root[-1] / (2.0 * np.pi * 1.0e9 * tau_p)),
        "order_parameter": total_power / denominator,
        "total_power": total_power,
    }


def branch_distance(previous_root, current_root):
    previous_root = np.asarray(previous_root, dtype=float)
    current_root = np.asarray(current_root, dtype=float)
    previous_n = previous_root[0 : 2 * N_lasers : 2]
    current_n = current_root[0 : 2 * N_lasers : 2]
    previous_s = np.maximum(previous_root[1 : 2 * N_lasers : 2], np.finfo(float).tiny)
    current_s = np.maximum(current_root[1 : 2 * N_lasers : 2], np.finfo(float).tiny)
    previous_phase = previous_root[2 * N_lasers : 3 * N_lasers - 1]
    current_phase = current_root[2 * N_lasers : 3 * N_lasers - 1]
    phase_delta = np.angle(np.exp(1j * (current_phase - previous_phase)))
    frequency_delta_ghz = (current_root[-1] - previous_root[-1]) / (
        2.0 * np.pi * 1.0e9 * tau_p
    )

    terms = [
        np.mean(((current_n - previous_n) / branch_match_carrier_scale) ** 2),
        np.mean(
            ((np.log(current_s) - np.log(previous_s)) / branch_match_log_photon_scale)
            ** 2
        ),
        (frequency_delta_ghz / branch_match_frequency_scale_ghz) ** 2,
    ]
    if phase_delta.size:
        terms.append(np.mean((phase_delta / branch_match_phase_scale_rad) ** 2))
    return float(np.sqrt(np.sum(terms)))


def assign_branch_ids(previous_roots, previous_ids, current_roots, next_branch_id):
    """Match adjacent-kappa roots for plotting; create IDs for births."""
    current_roots = np.asarray(current_roots, dtype=float)
    branch_ids = np.full(current_roots.shape[0], -1, dtype=int)
    if current_roots.size and previous_roots is not None and len(previous_roots):
        costs = np.empty((len(previous_roots), len(current_roots)), dtype=float)
        for i, previous_root in enumerate(previous_roots):
            for j, current_root in enumerate(current_roots):
                costs[i, j] = branch_distance(previous_root, current_root)
        row_indices, column_indices = linear_sum_assignment(costs)
        for row, column in zip(row_indices, column_indices):
            if costs[row, column] <= branch_match_max_cost:
                branch_ids[column] = int(previous_ids[row])

    for index in np.flatnonzero(branch_ids < 0):
        branch_ids[index] = int(next_branch_id)
        next_branch_id += 1
    return branch_ids, int(next_branch_id)


def constant_kappa_system(kappa_hz, noise_level=0.0):
    kappa_matrix = VCSEL.build_coupling_matrix(
        time_arr=np.asarray([0.0]),
        kappa_initial=float(kappa_hz),
        kappa_final=float(kappa_hz),
        N_lasers=N_lasers,
        ramp_start=0.0,
        ramp_shape=1.0,
        tau=tau,
        scheme="CUSTOM",
        aMAT=adjacency,
    )[0]
    vcsel, nd = linewidth_engine.make_vcsel_from_kappa_matrix(
        kappa_matrix,
        total_simulation_time,
        analysis_save_every,
        extra_physical_parameters={"noise_amplitude": float(noise_level)},
    )
    nd["delay_interp"] = delay_interpolation
    return vcsel, nd, kappa_matrix


def solve_and_classify_equilibria(kappa_hz, previous_roots, previous_ids, next_id):
    vcsel, nd, _ = constant_kappa_system(kappa_hz, noise_level=0.0)
    guesses = []
    if previous_roots is not None:
        guesses = [equilibrium_guess(root) for root in previous_roots]
    _, roots, _ = vcsel.solve_equilibria(
        nd,
        guesses=guesses,
        counts=dict(equilibrium_counts),
        n_jobs=int(equilibrium_solver_jobs),
    )
    roots = canonicalize_and_deduplicate_roots(roots)

    branch_ids, next_id = assign_branch_ids(
        previous_roots,
        previous_ids,
        roots,
        next_id,
    )
    stable_reported = np.zeros(len(roots), dtype=bool)
    stable_selected = np.zeros(len(roots), dtype=bool)
    dense_confirmed = np.zeros(len(roots), dtype=bool)
    leading_real = np.full(len(roots), np.nan, dtype=float)
    eigenvalue_count = np.zeros(len(roots), dtype=int)
    eigenvalue_rows = []
    requested_eigenvalues = int(stability_collocation_order) * 3 * N_lasers - 1

    for index, root in enumerate(roots):
        stable_flag, eigenvalues = vcsel.compute_stability(
            root,
            nd,
            N=int(stability_collocation_order),
            newton_maxit=int(stability_newton_max_iterations),
            threshold=1.0e-10,
            sparse=bool(stability_sparse),
            spectral_shift=stability_spectral_shift,
            n_eigenvalues=requested_eigenvalues,
        )
        eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
        provisional_leading = (
            float(np.max(eigenvalues.real)) if eigenvalues.size else np.nan
        )
        provisional_stable = bool(
            (eigenvalues.size and provisional_leading <= stability_real_tolerance)
            or (
                not eigenvalues.size
                and bool(stable_flag)
                and not require_nonempty_stability_spectrum
            )
        )
        if provisional_stable and confirm_stable_roots_with_dense_spectrum:
            stable_flag, eigenvalues = vcsel.compute_stability(
                root,
                nd,
                N=int(stability_collocation_order),
                newton_maxit=int(stability_newton_max_iterations),
                threshold=1.0e-10,
                sparse=False,
                spectral_shift=stability_spectral_shift,
                n_eigenvalues=requested_eigenvalues,
            )
            eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
            dense_confirmed[index] = True

        eigenvalue_rows.append(eigenvalues)
        eigenvalue_count[index] = eigenvalues.size
        stable_reported[index] = bool(stable_flag)
        if eigenvalues.size:
            leading_real[index] = float(np.max(eigenvalues.real))
            # Apply the user tolerance directly to the final spectrum. This
            # allows a small positive tolerance to admit numerical marginals.
            stable_selected[index] = bool(
                leading_real[index] <= stability_real_tolerance
            )
        else:
            stable_selected[index] = bool(
                stable_reported[index] and not require_nonempty_stability_spectrum
            )

    if (
        skip_uncoupled_identical_laser_ecms
        and N_lasers > 1
        and np.isclose(float(kappa_hz), 0.0, atol=1.0e-15)
        and np.ptp(detuning_distribution) <= 1.0e-12
    ):
        stable_selected[:] = False

    return {
        "vcsel": vcsel,
        "nd": nd,
        "roots": roots,
        "branch_ids": branch_ids,
        "stable_reported": stable_reported,
        "stable_selected": stable_selected,
        "dense_confirmed": dense_confirmed,
        "leading_real": leading_real,
        "eigenvalue_count": eigenvalue_count,
        "eigenvalues": eigenvalue_rows,
        "next_branch_id": next_id,
    }


def save_equilibrium_catalog(path, signature, kappa_index, kappa_hz, catalog):
    arrays = {
        "run_signature": np.asarray(signature),
        "kappa_index": np.asarray(kappa_index, dtype=int),
        "kappa_hz": np.asarray(kappa_hz, dtype=float),
        "roots": np.asarray(catalog["roots"], dtype=float),
        "branch_ids": np.asarray(catalog["branch_ids"], dtype=int),
        "stable_reported": np.asarray(catalog["stable_reported"], dtype=bool),
        "stable_selected": np.asarray(catalog["stable_selected"], dtype=bool),
        "stability_dense_confirmed": np.asarray(catalog["dense_confirmed"], dtype=bool),
        "leading_eigenvalue_real_nd": np.asarray(catalog["leading_real"], dtype=float),
        "stability_eigenvalue_count": np.asarray(
            catalog["eigenvalue_count"], dtype=int
        ),
    }
    if save_stability_eigenvalues:
        width = max((len(row) for row in catalog["eigenvalues"]), default=0)
        eigenvalue_matrix = np.full(
            (len(catalog["eigenvalues"]), width),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        for row_index, row in enumerate(catalog["eigenvalues"]):
            eigenvalue_matrix[row_index, : len(row)] = row
        arrays["stability_eigenvalues_nd"] = eigenvalue_matrix
    atomic_savez(path, **arrays)


def load_equilibrium_catalog(path, signature, kappa_hz):
    if not reuse_equilibrium_catalogs or not Path(path).exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["run_signature"].item()) != signature:
                return None
            if (
                save_stability_eigenvalues
                and "stability_eigenvalues_nd" not in saved.files
            ):
                return None
            if not np.isclose(float(saved["kappa_hz"]), float(kappa_hz)):
                return None
            return {
                "roots": np.asarray(saved["roots"], dtype=float),
                "branch_ids": np.asarray(saved["branch_ids"], dtype=int),
                "stable_reported": np.asarray(saved["stable_reported"], dtype=bool),
                "stable_selected": np.asarray(saved["stable_selected"], dtype=bool),
                "dense_confirmed": np.asarray(
                    saved["stability_dense_confirmed"], dtype=bool
                ),
                "leading_real": np.asarray(
                    saved["leading_eigenvalue_real_nd"], dtype=float
                ),
                "eigenvalue_count": np.asarray(
                    saved["stability_eigenvalue_count"], dtype=int
                ),
                "eigenvalues": [],
            }
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _linewidth_statistics(channel):
    return {
        "hz": float(channel.get("linewidth_hz", np.nan)),
        "mean_hz": float(channel.get("linewidth_mean_hz", np.nan)),
        "std_hz": float(channel.get("linewidth_std_hz", np.nan)),
        "q16_hz": float(channel.get("linewidth_q16_hz", np.nan)),
        "q84_hz": float(channel.get("linewidth_q84_hz", np.nan)),
    }


def _psd_save_indices(frequency_hz, save_full):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if save_full or frequency_hz.size <= int(maximum_saved_psd_points):
        return np.arange(frequency_hz.size, dtype=int)
    positive = np.flatnonzero(frequency_hz > 0.0)
    if not positive.size:
        return np.arange(min(frequency_hz.size, maximum_saved_psd_points), dtype=int)

    targets = np.geomspace(
        frequency_hz[positive[0]],
        frequency_hz[positive[-1]],
        int(max(2, maximum_saved_psd_points)),
    )
    indices = np.searchsorted(frequency_hz, targets)
    indices = np.clip(indices, 0, frequency_hz.size - 1)
    floor_low, floor_high = linewidth_engine.psd_floor_band_hz
    floor_indices = np.flatnonzero(
        (frequency_hz >= floor_low)
        & (frequency_hz <= floor_high)
        & (frequency_hz > 0.0)
    )
    return np.unique(np.concatenate(([0], indices, floor_indices)))


def validate_final_history_against_ecm(final_history, root, nd):
    """Identify noise realizations that remain near the seeded ECM."""
    final_history = np.asarray(final_history, dtype=np.float64)
    root = np.asarray(root, dtype=float)
    n_cases = final_history.shape[0]
    if not validate_final_ecm:
        return {
            "mask": np.ones(n_cases, dtype=bool),
            "photon_relative_error": np.full(n_cases, np.nan),
            "carrier_absolute_error": np.full(n_cases, np.nan),
            "relative_phase_error_rad": np.full(n_cases, np.nan),
            "relative_phase_coherence": np.full(n_cases, np.nan),
            "frequency_error_ghz": np.full(n_cases, np.nan),
        }

    # Average the final delay to suppress instantaneous Langevin fluctuations.
    tail_start = final_history.shape[-1] // 2
    tail = final_history[:, :, tail_start:]
    carrier_mean = np.mean(tail[:, 0::3, :], axis=-1)
    photon_mean = np.mean(tail[:, 1::3, :], axis=-1)
    root_carrier = root[0 : 2 * N_lasers : 2]
    root_photon = np.maximum(root[1 : 2 * N_lasers : 2], np.finfo(float).tiny)
    carrier_error = np.max(np.abs(carrier_mean - root_carrier[None, :]), axis=1)
    photon_error = np.max(
        np.abs(photon_mean - root_photon[None, :]) / root_photon[None, :],
        axis=1,
    )

    phase = final_history[:, 2::3, :]
    root_offsets = np.concatenate(([0.0], root[2 * N_lasers : 3 * N_lasers - 1]))
    if N_lasers > 1:
        maximum_phase_error = np.zeros(n_cases, dtype=float)
        minimum_phase_coherence = np.ones(n_cases, dtype=float)
        for laser_index in range(1, N_lasers):
            relative_error = (
                phase[:, laser_index, tail_start:]
                - phase[:, 0, tail_start:]
                - (root_offsets[laser_index] - root_offsets[0])
            )
            mean_phasor = np.mean(np.exp(1j * relative_error), axis=-1)
            maximum_phase_error = np.maximum(
                maximum_phase_error,
                np.abs(np.angle(mean_phasor)),
            )
            minimum_phase_coherence = np.minimum(
                minimum_phase_coherence,
                np.abs(mean_phasor),
            )
            del relative_error, mean_phasor
    else:
        maximum_phase_error = np.zeros(n_cases, dtype=float)
        minimum_phase_coherence = np.ones(n_cases, dtype=float)

    # A least-squares phase slope over the complete final 2*tau history catches
    # switches to a different ECM with similar photon numbers/relative phase.
    phase_unwrapped = np.unwrap(phase, axis=-1)
    time_nd = np.arange(phase.shape[-1], dtype=float) * float(nd["dt"])
    centered_time = time_nd - np.mean(time_nd)
    slope_denominator = float(np.sum(centered_time * centered_time))
    if slope_denominator > 0.0:
        phase_unwrapped -= np.mean(phase_unwrapped, axis=-1, keepdims=True)
        slopes_nd = (
            np.sum(phase_unwrapped * centered_time[None, None, :], axis=-1)
            / slope_denominator
        )
        common_slope_nd = np.mean(slopes_nd, axis=1)
        frequency_error_ghz = np.abs(common_slope_nd - root[-1]) / (
            2.0 * np.pi * 1.0e9 * tau_p
        )
    else:
        frequency_error_ghz = np.full(n_cases, np.inf)

    finite = (
        np.isfinite(photon_error)
        & np.isfinite(carrier_error)
        & np.isfinite(maximum_phase_error)
        & np.isfinite(minimum_phase_coherence)
        & np.isfinite(frequency_error_ghz)
    )
    stayed = (
        finite
        & (photon_error <= final_photon_relative_tolerance)
        & (carrier_error <= final_carrier_absolute_tolerance)
        & (maximum_phase_error <= final_relative_phase_tolerance_rad)
        & (minimum_phase_coherence >= final_relative_phase_min_coherence)
        & (frequency_error_ghz <= final_frequency_tolerance_ghz)
    )
    return {
        "mask": stayed,
        "photon_relative_error": photon_error,
        "carrier_absolute_error": carrier_error,
        "relative_phase_error_rad": maximum_phase_error,
        "relative_phase_coherence": minimum_phase_coherence,
        "frequency_error_ghz": frequency_error_ghz,
    }


def empty_phase_psd_analysis(n_cases, save_full_psd=False):
    nan_statistics = {
        "hz": np.nan,
        "mean_hz": np.nan,
        "std_hz": np.nan,
        "q16_hz": np.nan,
        "q84_hz": np.nan,
    }
    return {
        "psd_frequency_hz": np.asarray([], dtype=np.float64),
        "psd_common_hz2_per_hz": np.asarray([], dtype=np.float32),
        "common_statistics": nan_statistics,
        "common_floor_cases_hz2_per_hz": np.full(n_cases, np.nan, dtype=np.float32),
        "common_linewidth_cases_hz": np.full(n_cases, np.nan, dtype=np.float32),
        "laser_statistics": np.full((N_lasers, 5), np.nan, dtype=np.float64),
        "n_cases": int(n_cases),
        "full_psd_saved": bool(save_full_psd),
        "saved_sample_dt_seconds": np.nan,
        "effective_save_every": 0,
        "retained_duration_seconds": np.nan,
        "welch_nperseg": 0,
        "psd_floor_bin_count": 0,
        "psd_floor_loglog_slope": np.nan,
    }


def analyze_phase_only_psd(t, phase_output, save_full_psd=False):
    """Compute common/individual phase PSDs without retaining intensity."""
    t = np.asarray(t, dtype=float)
    phase_output = np.asarray(phase_output)
    analysis_start = int(np.searchsorted(t, analysis_start_time, side="left"))
    if t.size - analysis_start < 16:
        raise ValueError(
            "Noise burn-in leaves fewer than 16 saved samples. Increase "
            "linewidth_analysis_time or reduce noise_burn_in_time."
        )
    t_analysis = t[analysis_start:] - t[analysis_start]
    phase = phase_output[:, :, analysis_start:]
    if phase.shape[0] < int(max(1, minimum_valid_noise_cases)):
        return empty_phase_psd_analysis(phase.shape[0], save_full_psd=save_full_psd)

    common_phase = np.zeros((phase.shape[0], phase.shape[-1]), dtype=np.float64)
    for laser_index in range(N_lasers):
        common_phase += np.unwrap(phase[:, laser_index, :], axis=-1)
    common_phase /= float(N_lasers)
    frequency_hz, common_channel = linewidth_engine._frequency_noise_psd_channel(
        t_analysis,
        common_phase,
        already_unwrapped=True,
    )
    del common_phase

    full_frequency_count = frequency_hz.size
    floor_low, floor_high = linewidth_engine.psd_floor_band_hz
    floor_mask = (
        (frequency_hz >= floor_low)
        & (frequency_hz <= floor_high)
        & (frequency_hz > 0.0)
    )
    psd_trace = np.asarray(common_channel["psd"], dtype=float)
    flatness_mask = floor_mask & np.isfinite(psd_trace) & (psd_trace > 0.0)
    if np.count_nonzero(flatness_mask) >= 3:
        psd_floor_loglog_slope = float(
            np.polyfit(
                np.log10(frequency_hz[flatness_mask]),
                np.log10(psd_trace[flatness_mask]),
                1,
            )[0]
        )
    else:
        psd_floor_loglog_slope = np.nan
    save_indices = _psd_save_indices(frequency_hz, save_full_psd)
    saved_frequency = np.asarray(frequency_hz[save_indices], dtype=np.float64)
    saved_common_psd = np.asarray(common_channel["psd"][save_indices], dtype=np.float32)
    common_statistics = _linewidth_statistics(common_channel)
    common_floor_cases = np.asarray(common_channel["floor_cases"], dtype=np.float32)
    common_linewidth_cases = np.asarray(
        common_channel["linewidth_cases_hz"], dtype=np.float32
    )
    del common_channel, frequency_hz

    laser_statistics = np.full((N_lasers, 5), np.nan, dtype=np.float64)
    for laser_index in range(N_lasers):
        _, laser_channel = linewidth_engine._frequency_noise_psd_channel(
            t_analysis,
            phase[:, laser_index, :],
            already_unwrapped=False,
        )
        stats = _linewidth_statistics(laser_channel)
        laser_statistics[laser_index] = (
            stats["hz"],
            stats["mean_hz"],
            stats["std_hz"],
            stats["q16_hz"],
            stats["q84_hz"],
        )
        del laser_channel

    return {
        "psd_frequency_hz": saved_frequency,
        "psd_common_hz2_per_hz": saved_common_psd,
        "common_statistics": common_statistics,
        "common_floor_cases_hz2_per_hz": common_floor_cases,
        "common_linewidth_cases_hz": common_linewidth_cases,
        "laser_statistics": laser_statistics,
        "n_cases": int(phase.shape[0]),
        "full_psd_saved": bool(save_indices.size == full_frequency_count),
        "saved_sample_dt_seconds": float(np.median(np.diff(t_analysis))),
        "effective_save_every": int(round(float(np.median(np.diff(t_analysis))) / dt)),
        "retained_duration_seconds": float(t_analysis[-1] - t_analysis[0]),
        "welch_nperseg": int(
            t_analysis.size - 1
            if psd_welch_nperseg is None
            else min(int(psd_welch_nperseg), t_analysis.size - 1)
        ),
        "psd_floor_bin_count": int(np.count_nonzero(floor_mask)),
        "psd_floor_loglog_slope": psd_floor_loglog_slope,
    }


def _worker_initializer(postprocess_lock, progress_queue, worker_count):
    configure_linewidth_engine()
    linewidth_engine.LONG_RUN_POSTPROCESS_LOCK = postprocess_lock
    linewidth_engine.LONG_RUN_PROGRESS_QUEUE = progress_queue
    process = mp.current_process()
    identity = getattr(process, "_identity", ())
    raw_number = int(identity[-1]) if identity else 1
    worker_number = (raw_number - 1) % int(max(1, worker_count)) + 1
    linewidth_engine.set_activity_monitor_process_title(f"stable-ecm{worker_number}")


def _worker_ready():
    """Small eager-start task used before parent BLAS/eigensolver work."""
    return os.getpid()


def _postprocess_stable_ecm_result(job, t_long, phase_output, final_history, nd):
    """Validate, estimate, and save one ECM slice from a vectorized chunk."""
    lock = (
        linewidth_engine.LONG_RUN_POSTPROCESS_LOCK if serialize_psd_and_saving else None
    )
    if lock is not None:
        lock.acquire()
    try:
        validation = validate_final_history_against_ecm(
            final_history,
            job["root"],
            nd,
        )
        valid_case_mask = np.asarray(validation["mask"], dtype=bool)
        valid_phase_output = (
            phase_output if np.all(valid_case_mask) else phase_output[valid_case_mask]
        )
        # If filtering made a compact copy, release rejected rows before the
        # FFT/PSD workspace is allocated.
        phase_output = None
        psd = analyze_phase_only_psd(
            t_long,
            valid_phase_output,
            save_full_psd=bool(job["save_full_psd"]),
        )
        common = psd["common_statistics"]
        leading_real = float(job["leading_real"])
        burn_in_decay_constants = (
            noise_burn_in_time * abs(leading_real) / tau_p
            if np.isfinite(leading_real) and leading_real < 0.0
            else 0.0
        )
        result_arrays = {
            "run_signature": np.asarray(job["run_signature"]),
            "root": np.asarray(job["root"], dtype=np.float64),
            "kappa_index": np.asarray(job["kappa_index"], dtype=int),
            "solution_index": np.asarray(job["solution_index"], dtype=int),
            "branch_id": np.asarray(job["branch_id"], dtype=int),
            "kappa_hz": np.asarray(job["kappa_hz"], dtype=float),
            "frequency_ghz": np.asarray(job["frequency_ghz"], dtype=float),
            "omega_nd": np.asarray(job["root"][-1], dtype=float),
            "leading_eigenvalue_real_nd": np.asarray(job["leading_real"], dtype=float),
            "leading_eigenvalue_real_per_second": np.asarray(
                job["leading_real"] / tau_p, dtype=float
            ),
            "stability_eigenvalue_count": np.asarray(
                job["eigenvalue_count"], dtype=int
            ),
            "stability_dense_confirmed": np.asarray(job["dense_confirmed"], dtype=bool),
            "equilibrium_order_parameter": np.asarray(
                job["order_parameter"], dtype=float
            ),
            "equilibrium_total_power": np.asarray(job["total_power"], dtype=float),
            "psd_frequency_hz": psd["psd_frequency_hz"],
            "psd_common_hz2_per_hz": psd["psd_common_hz2_per_hz"],
            "psd_linewidth_common_hz": np.asarray(common["hz"], dtype=float),
            "psd_linewidth_common_mean_hz": np.asarray(common["mean_hz"], dtype=float),
            "psd_linewidth_common_std_hz": np.asarray(common["std_hz"], dtype=float),
            "psd_linewidth_common_q16_hz": np.asarray(common["q16_hz"], dtype=float),
            "psd_linewidth_common_q84_hz": np.asarray(common["q84_hz"], dtype=float),
            "psd_linewidth_laser_statistics_hz": np.asarray(
                psd["laser_statistics"], dtype=float
            ),
            "n_cases": np.asarray(psd["n_cases"], dtype=int),
            "n_cases_total": np.asarray(valid_case_mask.size, dtype=int),
            "n_cases_valid": np.asarray(np.count_nonzero(valid_case_mask), dtype=int),
            "valid_case_fraction": np.asarray(np.mean(valid_case_mask), dtype=float),
            "burn_in_decay_time_constants": np.asarray(
                burn_in_decay_constants, dtype=float
            ),
            "saved_sample_dt_seconds": np.asarray(
                psd["saved_sample_dt_seconds"], dtype=float
            ),
            "effective_save_every": np.asarray(psd["effective_save_every"], dtype=int),
            "retained_duration_seconds": np.asarray(
                psd["retained_duration_seconds"], dtype=float
            ),
            "welch_nperseg": np.asarray(psd["welch_nperseg"], dtype=int),
            "psd_floor_bin_count": np.asarray(psd["psd_floor_bin_count"], dtype=int),
            "psd_floor_loglog_slope": np.asarray(
                psd["psd_floor_loglog_slope"], dtype=float
            ),
            "saved_psd_points": np.asarray(psd["psd_frequency_hz"].size, dtype=int),
            "full_psd_saved": np.asarray(psd["full_psd_saved"], dtype=bool),
            "resolved_run_seed": np.asarray(job["resolved_run_seed"], dtype=np.uint32),
            "job_seed": np.asarray(job["seed"], dtype=np.uint32),
            "stayed_on_seeded_ecm": valid_case_mask,
            "final_photon_relative_error": np.asarray(
                validation["photon_relative_error"], dtype=np.float32
            ),
            "final_carrier_absolute_error": np.asarray(
                validation["carrier_absolute_error"], dtype=np.float32
            ),
            "final_relative_phase_error_rad": np.asarray(
                validation["relative_phase_error_rad"], dtype=np.float32
            ),
            "final_relative_phase_coherence": np.asarray(
                validation["relative_phase_coherence"], dtype=np.float32
            ),
            "final_frequency_error_ghz": np.asarray(
                validation["frequency_error_ghz"], dtype=np.float32
            ),
            "psd_floor_band_hz": np.asarray(
                linewidth_engine.psd_floor_band_hz, dtype=float
            ),
        }
        if save_per_case_linewidths:
            result_arrays["common_floor_cases_hz2_per_hz"] = psd[
                "common_floor_cases_hz2_per_hz"
            ]
            result_arrays["common_linewidth_cases_hz"] = psd[
                "common_linewidth_cases_hz"
            ]
        atomic_savez(job["result_path"], **result_arrays)
        summary = {
            field: _jsonable(result_arrays[field].item())
            for field in SCALAR_RESULT_FIELDS
        }
        summary["root"] = np.asarray(job["root"], dtype=float)
        summary["laser_statistics"] = np.asarray(psd["laser_statistics"], dtype=float)
        summary["result_path"] = str(Path(job["result_path"]).resolve())
    finally:
        if lock is not None:
            lock.release()

    del valid_phase_output, validation, valid_case_mask, psd
    gc.collect()
    return summary


def _run_stable_ecm_linewidth_chunk(jobs):
    """Integrate several ECMs as one vectorized batch, then split the result."""
    jobs = list(jobs)
    if not jobs:
        return []

    # Each ECM retains its independent deterministic seed even though all
    # trajectories advance through one vectorized VCSEL.integrate call.
    seed_sequence = np.random.SeedSequence([int(job["seed"]) for job in jobs])
    np.random.seed(int(seed_sequence.generate_state(1, dtype=np.uint32)[0]))
    histories = []
    kappa_cases = []
    case_slices = []
    case_start = 0
    for job in jobs:
        n_cases = int(job["n_noise_cases"])
        histories.append(
            linewidth_engine.replicate_physical_history(job["history"], n_cases)
        )
        kappa_cases.append(
            np.repeat(
                np.asarray(job["kappa_matrix"], dtype=np.float64)[None, :, :],
                n_cases,
                axis=0,
            )
        )
        case_slices.append(slice(case_start, case_start + n_cases))
        case_start += n_cases

    history = np.concatenate(histories, axis=0)
    kappa_matrix = np.concatenate(kappa_cases, axis=0)
    vcsel, nd = linewidth_engine.make_vcsel_from_kappa_matrix(
        kappa_matrix,
        total_simulation_time,
        analysis_save_every,
    )
    nd["kappa_case_dependent"] = True
    nd["delay_interp"] = delay_interpolation
    nd["store_freqs"] = False
    nd["output_state_indices"] = np.arange(N_lasers, dtype=int) * 3 + 2

    progress_callback, flush_progress = linewidth_engine.make_worker_progress_callback(
        jobs[0]
    )
    try:
        t_long, phase_output, _, final_history = (
            linewidth_engine.integrate_with_optional_progress_callback(
                vcsel,
                history,
                nd=nd,
                progress=False,
                max_iter=5,
                smooth_freqs=False,
                return_final_history=True,
                message=f"stable ECM chunk ({len(jobs)} branches)",
                progress_callback=progress_callback,
            )
        )
    finally:
        flush_progress()

    summaries = []
    for job, case_slice in zip(jobs, case_slices):
        summaries.append(
            _postprocess_stable_ecm_result(
                job,
                t_long,
                phase_output[case_slice],
                final_history[case_slice],
                nd,
            )
        )

    del history, histories, kappa_matrix, kappa_cases
    del phase_output, final_history, t_long, vcsel, nd
    gc.collect()
    return summaries


def run_stable_ecm_linewidth_chunk(jobs):
    try:
        return _run_stable_ecm_linewidth_chunk(jobs)
    except BaseException:
        worker_error_dir.mkdir(parents=True, exist_ok=True)
        first = jobs[0] if jobs else {"kappa_index": -1, "branch_id": -1}
        label = (
            f"chunk_kappa_{int(first['kappa_index']):04d}_"
            f"branch_{int(first['branch_id']):04d}.txt"
        )
        (worker_error_dir / label).write_text(traceback.format_exc())
        raise


def load_result_summary(path, signature, root, expected_full_psd=False):
    if not resume_existing_results or not Path(path).exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["run_signature"].item()) != signature:
                return None
            if expected_full_psd and not bool(saved["full_psd_saved"]):
                return None
            if save_per_case_linewidths and (
                "common_floor_cases_hz2_per_hz" not in saved.files
                or "common_linewidth_cases_hz" not in saved.files
            ):
                return None
            if not np.allclose(
                np.asarray(saved["root"], dtype=float),
                np.asarray(root, dtype=float),
                rtol=1.0e-8,
                atol=1.0e-10,
            ):
                return None
            summary = {
                field: _jsonable(saved[field].item()) for field in SCALAR_RESULT_FIELDS
            }
            summary["root"] = np.asarray(saved["root"], dtype=float)
            summary["laser_statistics"] = np.asarray(
                saved["psd_linewidth_laser_statistics_hz"], dtype=float
            )
            summary["result_path"] = str(Path(path).resolve())
            return summary
    except (OSError, ValueError, KeyError, EOFError):
        return None


def should_save_full_psd(kappa_index, branch_id):
    return bool(
        save_full_common_psd_for_all_ecms
        or int(branch_id) in full_psd_branch_ids
        or int(kappa_index) in full_psd_kappa_indices
    )


def make_stable_job(
    signature,
    resolved_seed,
    kappa_index,
    solution_index,
    branch_id,
    kappa_hz,
    root,
    leading_real,
    eigenvalue_count,
    dense_confirmed,
    nd,
    kappa_matrix,
    result_path,
):
    observables = equilibrium_observables(root)
    seed = int(
        np.random.SeedSequence(
            [resolved_seed, int(kappa_index), int(solution_index), int(branch_id)]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    save_full = should_save_full_psd(kappa_index, branch_id)
    return {
        "run_signature": signature,
        "seed": seed,
        "resolved_run_seed": int(resolved_seed),
        "kappa_index": int(kappa_index),
        "solution_index": int(solution_index),
        "branch_id": int(branch_id),
        "kappa_hz": float(kappa_hz),
        "root": np.asarray(root, dtype=float),
        "leading_real": float(leading_real),
        "eigenvalue_count": int(eigenvalue_count),
        "dense_confirmed": bool(dense_confirmed),
        "frequency_ghz": observables["frequency_ghz"],
        "order_parameter": observables["order_parameter"],
        "total_power": observables["total_power"],
        "history": ecm_history(root, nd),
        # This matrix is tiny. Keep the exact float64 values used by the
        # equilibrium and stability calculations.
        "kappa_matrix": np.asarray(kappa_matrix, dtype=np.float64),
        "n_noise_cases": int(max(1, n_noise_iterations)),
        "save_full_psd": save_full,
        "result_path": str(result_path),
    }


# ------------------------------ Persistence -------------------------------


def sorted_summaries(summaries):
    return sorted(
        summaries,
        key=lambda row: (
            int(row["kappa_index"]),
            int(row["branch_id"]),
            int(row["solution_index"]),
        ),
    )


def save_flat_summary(summaries, signature, base_output_dir=None):
    summaries = sorted_summaries(summaries)
    destination_summary_dir = (
        summary_dir
        if base_output_dir is None
        else Path(base_output_dir) / "summary"
    )
    destination_summary_dir.mkdir(parents=True, exist_ok=True)
    npz_path = destination_summary_dir / "all_stable_linewidths_summary.npz"
    csv_path = destination_summary_dir / "all_stable_linewidths_summary.csv"
    if not summaries:
        atomic_savez(
            npz_path,
            run_signature=np.asarray(signature),
            roots=np.empty((0, 3 * N_lasers), dtype=float),
        )
        csv_path.write_text("")
        return

    arrays = {"run_signature": np.asarray(signature)}
    for field in SCALAR_RESULT_FIELDS:
        values = [row[field] for row in summaries]
        arrays[field] = np.asarray(values)
    arrays["roots"] = np.stack([row["root"] for row in summaries])
    arrays["laser_statistics_hz"] = np.stack(
        [row["laser_statistics"] for row in summaries]
    )
    arrays["result_paths"] = np.asarray(
        [row["result_path"] for row in summaries], dtype="U"
    )
    atomic_savez(npz_path, **arrays)

    field_names = [
        "kappa_index",
        "solution_index",
        "branch_id",
        "kappa_hz",
        "kappa_ns_inv",
        "frequency_ghz",
        "leading_eigenvalue_real_nd",
        "leading_eigenvalue_real_per_second",
        "stability_eigenvalue_count",
        "stability_dense_confirmed",
        "equilibrium_order_parameter",
        "equilibrium_total_power",
        "psd_linewidth_common_hz",
        "psd_linewidth_common_mean_hz",
        "psd_linewidth_common_std_hz",
        "psd_linewidth_common_q16_hz",
        "psd_linewidth_common_q84_hz",
        "n_cases",
        "n_cases_total",
        "n_cases_valid",
        "valid_case_fraction",
        "burn_in_decay_time_constants",
        "effective_save_every",
        "saved_sample_dt_seconds",
        "retained_duration_seconds",
        "welch_nperseg",
        "psd_floor_bin_count",
        "psd_floor_loglog_slope",
        "resolved_run_seed",
        "job_seed",
        "result_path",
    ]
    field_names += [f"n_{i + 1}_nd" for i in range(N_lasers)]
    field_names += [f"S_{i + 1}_nd" for i in range(N_lasers)]
    field_names += [f"phase_{i + 2}_minus_phase_1_rad" for i in range(N_lasers - 1)]
    field_names += ["omega_nd"]
    for laser_index in range(N_lasers):
        for statistic in ("median", "mean", "std", "q16", "q84"):
            field_names.append(f"laser_{laser_index + 1}_linewidth_{statistic}_hz")

    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for summary in summaries:
            root = np.asarray(summary["root"], dtype=float)
            row = {name: summary.get(name, np.nan) for name in field_names}
            row["kappa_ns_inv"] = float(summary["kappa_hz"]) * 1.0e-9
            row["result_path"] = summary["result_path"]
            for index, value in enumerate(root[0 : 2 * N_lasers : 2]):
                row[f"n_{index + 1}_nd"] = value
            for index, value in enumerate(root[1 : 2 * N_lasers : 2]):
                row[f"S_{index + 1}_nd"] = value
            for index, value in enumerate(root[2 * N_lasers : 3 * N_lasers - 1]):
                row[f"phase_{index + 2}_minus_phase_1_rad"] = value
            row["omega_nd"] = root[-1]
            laser_statistics = np.asarray(summary["laser_statistics"])
            for laser_index in range(N_lasers):
                for statistic_index, statistic in enumerate(
                    ("median", "mean", "std", "q16", "q84")
                ):
                    row[f"laser_{laser_index + 1}_linewidth_{statistic}_hz"] = (
                        laser_statistics[laser_index, statistic_index]
                    )
            writer.writerow(row)
    os.replace(temporary_csv, csv_path)


# ------------------------------- Plotting ---------------------------------


def manifold_rows_from_catalog(kappa_index, kappa_hz, catalog):
    """Return lightweight plotting rows for every catalogued equilibrium."""
    roots = np.asarray(catalog["roots"], dtype=float).reshape(-1, 3 * N_lasers)
    branch_ids = np.asarray(catalog["branch_ids"], dtype=int)
    stable_selected = np.asarray(catalog["stable_selected"], dtype=bool)
    stable_reported = np.asarray(catalog["stable_reported"], dtype=bool)
    leading_real = np.asarray(catalog["leading_real"], dtype=float)
    eigenvalue_count = np.asarray(catalog["eigenvalue_count"], dtype=int)
    dense_confirmed = np.asarray(catalog["dense_confirmed"], dtype=bool)
    rows = []
    for solution_index, root in enumerate(roots):
        rows.append(
            {
                "kappa_index": int(kappa_index),
                "kappa_hz": float(kappa_hz),
                "solution_index": int(solution_index),
                "branch_id": int(branch_ids[solution_index]),
                "frequency_ghz": equilibrium_observables(root)["frequency_ghz"],
                "stable_reported": bool(stable_reported[solution_index]),
                "stable_selected": bool(stable_selected[solution_index]),
                "dense_confirmed": bool(dense_confirmed[solution_index]),
                "leading_real": float(leading_real[solution_index]),
                "eigenvalue_count": int(eigenvalue_count[solution_index]),
            }
        )
    return rows


def _plot_contiguous_branch(ax, sequence_index, x, y, **kwargs):
    """Connect adjacent kappa samples without bridging missing branch points."""
    sequence_index = np.asarray(sequence_index, dtype=int)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(sequence_index)
    sequence_index = sequence_index[order]
    x = x[order]
    y = y[order]
    if not x.size:
        return
    split_points = np.flatnonzero(np.diff(sequence_index) > 1) + 1
    for indices in np.split(np.arange(x.size), split_points):
        if indices.size >= 2:
            ax.plot(x[indices], y[indices], **kwargs)


def _finite_linear_norm(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.02, 1.0e-6)
        vmin -= padding
        vmax += padding
    return Normalize(vmin=vmin, vmax=vmax)


def _finite_log_norm(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if not values.size:
        return None
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        factor = np.sqrt(10.0)
        vmin /= factor
        vmax *= factor
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_overview(summaries, manifold_rows=None, figure_path=None):
    """Plot stable linewidths and every catalogued stable/unstable ECM."""
    summaries = sorted_summaries(summaries)
    manifold_rows = [] if manifold_rows is None else list(manifold_rows)
    if not summaries and not manifold_rows:
        return None

    kappa_index = np.asarray(
        [row["kappa_index"] for row in summaries], dtype=int
    )
    kappa_ns = np.asarray(
        [row["kappa_hz"] for row in summaries], dtype=float
    ) * 1.0e-9
    branch_id = np.asarray([row["branch_id"] for row in summaries], dtype=int)
    linewidth = np.asarray(
        [row["psd_linewidth_common_hz"] for row in summaries], dtype=float
    )
    linewidth_mhz = linewidth * 1.0e-6
    frequency_ghz = np.asarray(
        [row["frequency_ghz"] for row in summaries], dtype=float
    )
    valid = np.isfinite(linewidth) & (linewidth > 0.0)
    finite_frequency = np.isfinite(frequency_ghz)

    manifold_kappa_ns = np.asarray(
        [row["kappa_hz"] for row in manifold_rows], dtype=float
    ) * 1.0e-9
    manifold_frequency = np.asarray(
        [row["frequency_ghz"] for row in manifold_rows], dtype=float
    )
    manifold_stable = np.asarray(
        [row["stable_selected"] for row in manifold_rows], dtype=bool
    )
    manifold_leading_real = np.asarray(
        [row["leading_real"] for row in manifold_rows], dtype=float
    )
    manifold_eigenvalue_count = np.asarray(
        [row["eigenvalue_count"] for row in manifold_rows], dtype=int
    )
    manifold_finite_frequency = np.isfinite(manifold_frequency)
    manifold_unstable = (
        (~manifold_stable)
        & (manifold_eigenvalue_count > 0)
        & np.isfinite(manifold_leading_real)
        & (manifold_leading_real > stability_real_tolerance)
    )
    manifold_unclassified = (~manifold_stable) & (~manifold_unstable)

    style = {
        "text.usetex": bool(overview_use_latex),
        "font.family": overview_font_family,
        "mathtext.fontset": "stix",
        "font.size": overview_font_size,
        "axes.labelsize": overview_font_size + 1,
        "axes.titlesize": overview_font_size + 2,
        "legend.fontsize": overview_font_size - 2,
        "xtick.labelsize": overview_font_size - 1,
        "ytick.labelsize": overview_font_size - 1,
    }
    with plt.rc_context(style):
        fig, (ax_linewidth, ax_frequency) = plt.subplots(
            2,
            1,
            figsize=overview_figure_size,
            dpi=overview_figure_dpi,
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": (1.0, 1.05)},
        )
        try:
            for current_branch in np.unique(branch_id):
                mask = (branch_id == current_branch) & valid
                _plot_contiguous_branch(
                    ax_linewidth,
                    kappa_index[mask],
                    kappa_ns[mask],
                    linewidth_mhz[mask],
                    color="0.62",
                    linewidth=0.65,
                    alpha=0.55,
                    zorder=1,
                )

            top_color_mask = valid & finite_frequency
            frequency_norm = _finite_linear_norm(frequency_ghz[top_color_mask])
            completed_frequency_mappable = None
            if frequency_norm is not None:
                completed_frequency_mappable = ax_linewidth.scatter(
                    kappa_ns[top_color_mask],
                    linewidth_mhz[top_color_mask],
                    c=frequency_ghz[top_color_mask],
                    cmap=overview_colormap,
                    norm=frequency_norm,
                    s=overview_completed_marker_size,
                    linewidths=0,
                    zorder=3,
                )

            if np.any(valid):
                unique_kappa = np.unique(kappa_index[valid])
                envelope_kappa_ns = np.full(unique_kappa.size, np.nan)
                minimum = np.full(unique_kappa.size, np.nan)
                median = np.full(unique_kappa.size, np.nan)
                maximum = np.full(unique_kappa.size, np.nan)
                for position, index in enumerate(unique_kappa):
                    mask = (kappa_index == index) & valid
                    envelope_kappa_ns[position] = float(np.nanmedian(kappa_ns[mask]))
                    minimum[position] = np.nanmin(linewidth_mhz[mask])
                    median[position] = np.nanmedian(linewidth_mhz[mask])
                    maximum[position] = np.nanmax(linewidth_mhz[mask])
                ax_linewidth.fill_between(
                    envelope_kappa_ns,
                    minimum,
                    maximum,
                    color="k",
                    alpha=0.07,
                    label="min–max of completed stable ECMs",
                    zorder=0,
                )
                ax_linewidth.plot(
                    envelope_kappa_ns,
                    median,
                    color="k",
                    linewidth=1.5,
                    label="median of completed stable ECMs",
                    zorder=2,
                )
                ax_linewidth.legend(loc="upper right", frameon=False)
            else:
                ax_linewidth.text(
                    0.5,
                    0.5,
                    "Waiting for completed stable-ECM linewidths",
                    transform=ax_linewidth.transAxes,
                    ha="center",
                    va="center",
                    color="0.4",
                )

            ax_linewidth.set_yscale("log")
            ax_linewidth.set_ylim(*overview_linewidth_ylim_mhz)
            ax_linewidth.set_yticks(overview_linewidth_yticks_mhz)
            ax_linewidth.minorticks_on()
            ax_linewidth.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
            )
            ax_linewidth.yaxis.set_minor_formatter(NullFormatter())
            ax_linewidth.set_ylabel(
                r"$\Delta\nu_{\mathrm{common}}$ (MHz)"
            )
            ax_linewidth.set_title("Common-phase linewidth of stable ECMs", pad=8)
            ax_linewidth.grid(
                True,
                axis="y",
                which="major",
                linestyle="-",
                linewidth=0.8,
                alpha=0.5,
            )
            ax_linewidth.grid(
                True,
                axis="y",
                which="minor",
                linestyle="--",
                linewidth=0.65,
                alpha=0.38,
            )
            ax_linewidth.grid(
                True,
                axis="x",
                which="major",
                linestyle="--",
                linewidth=0.6,
                alpha=0.25,
            )
            ax_linewidth.tick_params(axis="y", which="minor", length=3.5)

            # All ECMs at one kappa share the same horizontal coordinate in
            # the overview. Separate the endpoint values by ECM frequency in
            # an inset so their linewidth spread can be inspected directly.
            if np.any(top_color_mask):
                completed_kappa_values = np.unique(kappa_ns[top_color_mask])
                inset_target_kappa = float(np.max(completed_kappa_values))
                inset_mask = (
                    top_color_mask
                    & np.isclose(
                        kappa_ns,
                        inset_target_kappa,
                        rtol=0.0,
                        atol=max(1.0e-12, abs(inset_target_kappa) * 1.0e-12),
                    )
                )
                if np.any(inset_mask):
                    inset = ax_linewidth.inset_axes(
                        overview_linewidth_inset_bounds
                    )
                    inset.scatter(
                        frequency_ghz[inset_mask],
                        linewidth[inset_mask],
                        c=frequency_ghz[inset_mask],
                        cmap=overview_colormap,
                        norm=frequency_norm,
                        s=max(8.0, 0.80 * overview_completed_marker_size),
                        linewidths=0,
                        zorder=3,
                    )
                    inset.set_yscale("log")
                    inset_positive_linewidths = linewidth[inset_mask]
                    inset_positive_linewidths = inset_positive_linewidths[
                        np.isfinite(inset_positive_linewidths)
                        & (inset_positive_linewidths > 0.0)
                    ]
                    if overview_linewidth_inset_ylim_hz is None:
                        inset_minimum = float(np.min(inset_positive_linewidths))
                        inset_maximum = float(np.max(inset_positive_linewidths))
                        if np.isclose(inset_minimum, inset_maximum):
                            inset_minimum /= np.sqrt(10.0)
                            inset_maximum *= np.sqrt(10.0)
                        else:
                            log_padding = max(
                                0.08
                                * np.log10(inset_maximum / inset_minimum),
                                0.08,
                            )
                            padding_factor = 10.0**log_padding
                            inset_minimum /= padding_factor
                            inset_maximum *= padding_factor
                        inset_ylim_hz = (inset_minimum, inset_maximum)
                    else:
                        inset_ylim_hz = tuple(
                            float(value)
                            for value in overview_linewidth_inset_ylim_hz
                        )
                    inset.set_ylim(*inset_ylim_hz)
                    inset.set_yticks([1.0e4, 1.0e5])
                    inset.yaxis.set_minor_locator(NullLocator())
                    inset.set_title(
                        rf"$\kappa_c={inset_target_kappa:.3g}\,"
                        r"\mathrm{ns}^{-1}$",
                        pad=2,
                        fontsize=overview_font_size - 2,
                    )
                    inset.set_xlabel(
                        r"$\omega$ (GHz)",
                        labelpad=1,
                        fontsize=overview_font_size - 4,
                    )
                    inset.set_ylabel(
                        r"$\Delta\nu$ (Hz)",
                        labelpad=1,
                        fontsize=overview_font_size - 4,
                    )
                    inset.grid(
                        True,
                        which="both",
                        linestyle="--",
                        linewidth=0.45,
                        alpha=0.35,
                    )
                    inset.tick_params(
                        which="both",
                        direction="in",
                        top=True,
                        right=True,
                        labelsize=overview_font_size - 5,
                    )

            stable_catalog = manifold_stable & manifold_finite_frequency
            if np.any(stable_catalog):
                ax_frequency.scatter(
                    manifold_kappa_ns[stable_catalog],
                    manifold_frequency[stable_catalog],
                    marker="o",
                    facecolors="none",
                    edgecolors="0.48",
                    s=overview_stable_catalog_marker_size,
                    linewidths=0.55,
                    label="stable ECM",
                    zorder=2,
                )
            unstable_catalog = manifold_unstable & manifold_finite_frequency
            if np.any(unstable_catalog):
                ax_frequency.scatter(
                    manifold_kappa_ns[unstable_catalog],
                    manifold_frequency[unstable_catalog],
                    marker=".",
                    color="0.22",
                    alpha=0.8,
                    s=overview_unstable_marker_size,
                    linewidths=0,
                    label="unstable ECM",
                    zorder=1,
                )
            unclassified_catalog = manifold_unclassified & manifold_finite_frequency
            if np.any(unclassified_catalog):
                ax_frequency.scatter(
                    manifold_kappa_ns[unclassified_catalog],
                    manifold_frequency[unclassified_catalog],
                    marker="+",
                    color="0.65",
                    s=22,
                    linewidths=0.75,
                    label="neutral or unclassified ECM",
                    zorder=1,
                )

            bottom_color_mask = valid & finite_frequency
            if frequency_norm is not None:
                ax_frequency.scatter(
                    kappa_ns[bottom_color_mask],
                    frequency_ghz[bottom_color_mask],
                    c=frequency_ghz[bottom_color_mask],
                    cmap=overview_colormap,
                    norm=frequency_norm,
                    marker="o",
                    s=overview_completed_marker_size,
                    linewidths=0,
                    label="stable ECM with linewidth",
                    zorder=4,
                )
            if completed_frequency_mappable is not None:
                colorbar = fig.colorbar(
                    completed_frequency_mappable,
                    ax=(ax_linewidth, ax_frequency),
                    pad=0.015,
                    fraction=0.038,
                    aspect=28,
                )
                colorbar.set_label(
                    r"$\omega$ (GHz)"
                )

            invalid_frequency = (~valid) & finite_frequency
            if np.any(invalid_frequency):
                ax_frequency.scatter(
                    kappa_ns[invalid_frequency],
                    frequency_ghz[invalid_frequency],
                    marker="o",
                    color="0.05",
                    s=overview_insufficient_marker_size,
                    linewidths=0,
                    label="stable ECM, insufficient valid cases",
                    zorder=5,
                )

            if manifold_rows or summaries:
                handles, labels = ax_frequency.get_legend_handles_labels()
                unique = {}
                for handle, label in zip(handles, labels):
                    unique.setdefault(label, handle)
                if unique:
                    ax_frequency.legend(
                        unique.values(),
                        unique.keys(),
                        # Keep the dense equilibrium manifold unobstructed.
                        # bbox_inches="tight" below retains this external
                        # legend in both progress and final figures.
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.20),
                        frameon=False,
                        ncol=3,
                        columnspacing=1.1,
                        handletextpad=0.5,
                        borderaxespad=0.0,
                    )

            ax_frequency.set_ylabel(
                r"$\omega$ (GHz)"
            )
            ax_frequency.set_xlabel(r"Coupling rate, $\kappa_c$ (ns$^{-1}$)")
            ax_frequency.set_title("Equilibrium manifold and stability", pad=8)
            ax_frequency.grid(True, which="major", alpha=0.22, linewidth=0.7)

            available_kappa = []
            if kappa_ns.size:
                available_kappa.append(float(np.nanmax(kappa_ns)))
            if manifold_kappa_ns.size:
                available_kappa.append(float(np.nanmax(manifold_kappa_ns)))
            configured_max = max(0.0, float(kappa_final) * 1.0e-9)
            x_max = max([configured_max, *available_kappa], default=1.0)
            if x_max <= 0.0:
                x_max = 1.0
            ax_frequency.set_xticks(np.linspace(0.0, x_max, 6))
            ax_frequency.set_xlim(0.0, x_max * 1.015)

            for axis in (ax_linewidth, ax_frequency):
                axis.tick_params(
                    which="both",
                    direction="in",
                    top=True,
                    right=True,
                    length=4,
                )
                axis.tick_params(which="minor", length=2.5)

            if figure_path is None:
                figure_path = summary_dir / "all_stable_linewidths_overview.png"
            figure_path = Path(figure_path)
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = figure_path.with_name(
                f".{figure_path.stem}.tmp{figure_path.suffix}"
            )
            fig.savefig(temporary_path, bbox_inches="tight")
            os.replace(temporary_path, figure_path)
        finally:
            plt.close(fig)
            if "temporary_path" in locals() and temporary_path.exists():
                temporary_path.unlink()
    return figure_path


def plot_saved_common_psd(result_path, figure_path=None):
    """Create a lightweight diagnostic plot from one saved ECM result."""
    result_path = Path(result_path)
    with np.load(result_path, allow_pickle=False) as saved:
        frequency = np.asarray(saved["psd_frequency_hz"], dtype=float)
        psd = np.asarray(saved["psd_common_hz2_per_hz"], dtype=float)
        floor_band = np.asarray(saved["psd_floor_band_hz"], dtype=float)
        linewidth_hz = float(saved["psd_linewidth_common_hz"])
        kappa_ns = float(saved["kappa_hz"]) * 1.0e-9
        branch_id = int(saved["branch_id"])
    mask = (frequency > 0.0) & np.isfinite(psd) & (psd > 0.0)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.loglog(frequency[mask], psd[mask], color="k", linewidth=1.0)
    ax.axvspan(floor_band[0], floor_band[1], color="0.5", alpha=0.16)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"common FM-noise PSD (Hz$^2$/Hz)")
    ax.set_title(
        rf"$\kappa_c={kappa_ns:.4g}$ ns$^{{-1}}$, branch {branch_id}; "
        rf"$\Delta\nu={linewidth_hz:.4g}$ Hz"
    )
    ax.grid(True, which="both", alpha=0.22)
    if figure_path is None:
        figure_path = result_path.with_suffix(".png")
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return Path(figure_path)


def save_kappa_ecm_psd_figures(summaries, figure_dir=None, common_psd_ylim=None):
    """Save one common-FM-noise PSD overlay for each completed kappa."""
    summaries = sorted_summaries(summaries)
    if not summaries:
        return []
    if figure_dir is None:
        figure_dir = kappa_psd_figure_dir
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    linewidth_hz = np.asarray(
        [row["psd_linewidth_common_hz"] for row in summaries], dtype=float
    )
    frequency_ghz = np.asarray(
        [row["frequency_ghz"] for row in summaries], dtype=float
    )
    color_mask = (
        np.isfinite(linewidth_hz)
        & (linewidth_hz > 0.0)
        & np.isfinite(frequency_ghz)
    )
    frequency_norm = _finite_linear_norm(frequency_ghz[color_mask])
    if frequency_norm is None:
        return []
    colormap = matplotlib.colormaps[overview_colormap]

    if common_psd_ylim is None:
        global_minimum = np.inf
        global_maximum = -np.inf
        for row in summaries:
            result_path = Path(row["result_path"])
            if not result_path.exists():
                continue
            try:
                with np.load(result_path, allow_pickle=False) as saved:
                    psd_common = np.asarray(
                        saved["psd_common_hz2_per_hz"], dtype=float
                    )
            except (OSError, ValueError, KeyError, EOFError):
                continue
            finite_positive = psd_common[
                np.isfinite(psd_common) & (psd_common > 0.0)
            ]
            if finite_positive.size:
                global_minimum = min(global_minimum, float(np.min(finite_positive)))
                global_maximum = max(global_maximum, float(np.max(finite_positive)))
        if np.isfinite(global_minimum) and np.isfinite(global_maximum):
            lower = 10.0 ** np.floor(np.log10(global_minimum))
            upper = 10.0 ** np.ceil(np.log10(global_maximum))
            if upper <= lower:
                upper = lower * 10.0
            common_psd_ylim = (lower, upper)
    elif common_psd_ylim is not None:
        common_psd_ylim = tuple(float(value) for value in common_psd_ylim)

    grouped = {}
    for row in summaries:
        grouped.setdefault(int(row["kappa_index"]), []).append(row)

    saved_paths = []
    style = {
        "text.usetex": bool(overview_use_latex),
        "font.family": overview_font_family,
        "mathtext.fontset": "stix",
        "font.size": overview_font_size,
        "axes.labelsize": overview_font_size + 1,
        "axes.titlesize": overview_font_size + 2,
        "xtick.labelsize": overview_font_size - 1,
        "ytick.labelsize": overview_font_size - 1,
    }
    with plt.rc_context(style):
        for kappa_index, rows in sorted(grouped.items()):
            kappa_ns = float(rows[0]["kappa_hz"]) * 1.0e-9
            temporary_path = None
            fig, ax = plt.subplots(
                figsize=(9.5, 5.8),
                dpi=overview_figure_dpi,
                constrained_layout=True,
            )
            plotted = 0
            floor_bands = []
            try:
                for row in sorted(rows, key=lambda item: float(item["frequency_ghz"])):
                    result_path = Path(row["result_path"])
                    if not result_path.exists():
                        continue
                    try:
                        with np.load(result_path, allow_pickle=False) as saved:
                            psd_frequency = np.asarray(
                                saved["psd_frequency_hz"], dtype=float
                            )
                            psd_common = np.asarray(
                                saved["psd_common_hz2_per_hz"], dtype=float
                            )
                            if "psd_floor_band_hz" in saved.files:
                                floor_bands.append(
                                    np.asarray(saved["psd_floor_band_hz"], dtype=float)
                                )
                    except (OSError, ValueError, KeyError, EOFError):
                        continue
                    mask = (
                        (psd_frequency > 0.0)
                        & np.isfinite(psd_frequency)
                        & np.isfinite(psd_common)
                        & (psd_common > 0.0)
                    )
                    row_frequency = float(row["frequency_ghz"])
                    if np.count_nonzero(mask) < 2 or not np.isfinite(row_frequency):
                        continue
                    ax.loglog(
                        psd_frequency[mask],
                        psd_common[mask],
                        color=colormap(frequency_norm(row_frequency)),
                        linewidth=0.9,
                        alpha=0.82,
                        zorder=2,
                    )
                    plotted += 1

                if floor_bands:
                    floor_band = np.nanmedian(np.vstack(floor_bands), axis=0)
                    if np.all(np.isfinite(floor_band)):
                        ax.axvspan(
                            floor_band[0],
                            floor_band[1],
                            color="0.5",
                            alpha=0.12,
                            zorder=0,
                        )
                if plotted == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No ECM had enough valid cases for a PSD",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        color="0.4",
                    )
                ax.set_xlabel(r"Frequency (Hz)")
                ax.set_ylabel(r"Common FM-noise PSD (Hz$^2$/Hz)")
                if common_psd_ylim is not None:
                    ax.set_ylim(*common_psd_ylim)
                ax.set_title(
                    rf"$\kappa_c={kappa_ns:.4g}\,\mathrm{{ns}}^{{-1}}$; "
                    rf"{plotted} stable ECM PSDs"
                )
                ax.grid(True, which="major", linewidth=0.7, alpha=0.30)
                ax.grid(
                    True,
                    which="minor",
                    linestyle="--",
                    linewidth=0.45,
                    alpha=0.16,
                )
                mappable = matplotlib.cm.ScalarMappable(
                    norm=frequency_norm,
                    cmap=colormap,
                )
                colorbar = fig.colorbar(
                    mappable,
                    ax=ax,
                    pad=0.015,
                    fraction=0.045,
                    aspect=28,
                )
                colorbar.set_label(
                    r"ECM frequency shift, $\Omega/(2\pi)$ (GHz)"
                )
                figure_path = figure_dir / (
                    f"kappa_{kappa_index:04d}_{kappa_ns:.6f}ns_inv_all_ecm_psds.png"
                )
                temporary_path = figure_path.with_name(
                    f".{figure_path.stem}.tmp{figure_path.suffix}"
                )
                fig.savefig(temporary_path, bbox_inches="tight")
                os.replace(temporary_path, figure_path)
                saved_paths.append(str(figure_path.resolve()))
            finally:
                plt.close(fig)
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
    return saved_paths


def load_saved_ecm_summaries_for_replot(base_output_dir=None):
    """Load lightweight summaries from saved NPZs without running simulations."""
    base_output_dir = output_dir if base_output_dir is None else Path(base_output_dir)
    result_directory = base_output_dir / "ecm_results"
    metadata_path = base_output_dir / "run_configuration.json"
    expected_signature = None
    if metadata_path.exists():
        try:
            expected_signature = str(
                json.loads(metadata_path.read_text())["linewidth_signature"]
            )
        except (OSError, ValueError, KeyError, TypeError):
            expected_signature = None

    summaries = []
    for result_path in sorted(result_directory.glob("*.npz")):
        try:
            with np.load(result_path, allow_pickle=False) as saved:
                if expected_signature is not None:
                    if "run_signature" not in saved.files:
                        continue
                    if str(saved["run_signature"].item()) != expected_signature:
                        continue
                summaries.append(
                    {
                        "kappa_index": int(saved["kappa_index"]),
                        "solution_index": int(saved["solution_index"]),
                        "branch_id": int(saved["branch_id"]),
                        "kappa_hz": float(saved["kappa_hz"]),
                        "frequency_ghz": float(saved["frequency_ghz"]),
                        "psd_linewidth_common_hz": float(
                            saved["psd_linewidth_common_hz"]
                        ),
                        "result_path": str(result_path.resolve()),
                    }
                )
        except (OSError, ValueError, KeyError, EOFError):
            continue
    return sorted_summaries(summaries)


def load_saved_full_ecm_summaries(base_output_dir=None, expected_signature=None):
    """Load complete summary rows from the crash-safe per-ECM result files.

    Unlike :func:`load_saved_ecm_summaries_for_replot`, this retains every
    field needed to reconstruct the flat NPZ/CSV summary.  It is also used
    when a run exits so that an interrupted resume cannot replace a complete
    summary with only the subset rediscovered during that invocation.
    """
    base_output_dir = output_dir if base_output_dir is None else Path(base_output_dir)
    result_directory = base_output_dir / "ecm_results"
    if expected_signature is None:
        metadata_path = base_output_dir / "run_configuration.json"
        if metadata_path.exists():
            try:
                expected_signature = str(
                    json.loads(metadata_path.read_text())["linewidth_signature"]
                )
            except (OSError, ValueError, KeyError, TypeError):
                expected_signature = None

    summaries = []
    for result_path in sorted(result_directory.glob("*.npz")):
        try:
            with np.load(result_path, allow_pickle=False) as saved:
                if expected_signature is not None:
                    if "run_signature" not in saved.files:
                        continue
                    if str(saved["run_signature"].item()) != expected_signature:
                        continue
                summary = {
                    field: _jsonable(saved[field].item())
                    for field in SCALAR_RESULT_FIELDS
                }
                summary["root"] = np.asarray(saved["root"], dtype=float)
                summary["laser_statistics"] = np.asarray(
                    saved["psd_linewidth_laser_statistics_hz"], dtype=float
                )
                summary["result_path"] = str(result_path.resolve())
                summaries.append(summary)
        except (OSError, ValueError, KeyError, EOFError):
            continue
    return sorted_summaries(summaries)


def load_saved_manifold_rows_for_replot(base_output_dir=None):
    """Load every saved equilibrium for overview plotting only."""
    base_output_dir = output_dir if base_output_dir is None else Path(base_output_dir)
    catalog_directory = base_output_dir / "equilibrium_catalogs"
    metadata_path = base_output_dir / "run_configuration.json"
    expected_signature = None
    if metadata_path.exists():
        try:
            expected_signature = str(
                json.loads(metadata_path.read_text())["catalog_signature"]
            )
        except (OSError, ValueError, KeyError, TypeError):
            expected_signature = None

    rows = []
    for saved_path in sorted(catalog_directory.glob("*.npz")):
        try:
            with np.load(saved_path, allow_pickle=False) as saved:
                if expected_signature is not None:
                    if "run_signature" not in saved.files:
                        continue
                    if str(saved["run_signature"].item()) != expected_signature:
                        continue
                catalog = {
                    "roots": np.asarray(saved["roots"], dtype=float),
                    "branch_ids": np.asarray(saved["branch_ids"], dtype=int),
                    "stable_reported": np.asarray(
                        saved["stable_reported"], dtype=bool
                    ),
                    "stable_selected": np.asarray(
                        saved["stable_selected"], dtype=bool
                    ),
                    "dense_confirmed": np.asarray(
                        saved["stability_dense_confirmed"], dtype=bool
                    ),
                    "leading_real": np.asarray(
                        saved["leading_eigenvalue_real_nd"], dtype=float
                    ),
                    "eigenvalue_count": np.asarray(
                        saved["stability_eigenvalue_count"], dtype=int
                    ),
                }
                rows.extend(
                    manifold_rows_from_catalog(
                        int(saved["kappa_index"]),
                        float(saved["kappa_hz"]),
                        catalog,
                    )
                )
        except (OSError, ValueError, KeyError, EOFError):
            continue
    return rows


def replot_saved_overview(base_output_dir=None):
    """Regenerate the overview PNG from saved NPZ files, without simulation."""
    base_output_dir = output_dir if base_output_dir is None else Path(base_output_dir)
    summaries = load_saved_ecm_summaries_for_replot(base_output_dir)
    manifold_rows = load_saved_manifold_rows_for_replot(base_output_dir)
    if not summaries and not manifold_rows:
        raise FileNotFoundError(
            "No matching saved ECM results or equilibrium catalogs were found "
            f"under {base_output_dir.resolve()}."
        )
    figure_path = (
        base_output_dir / "summary" / "all_stable_linewidths_overview.png"
    )
    linewidth_engine.render_figure(
        plot_overview,
        summaries,
        manifold_rows,
        figure_path,
    )
    print(
        f"Replotted overview from {len(summaries)} linewidth results and "
        f"{len(manifold_rows)} equilibrium points: {figure_path.resolve()}"
    )
    return figure_path


def replot_saved_kappa_ecm_psds(base_output_dir=None, common_psd_ylim=None):
    """Regenerate all per-kappa PSD PNGs from existing result files only."""
    base_output_dir = output_dir if base_output_dir is None else Path(base_output_dir)
    summaries = load_saved_ecm_summaries_for_replot(base_output_dir)
    if not summaries:
        raise FileNotFoundError(
            "No saved ECM results matching the current run signature were found "
            f"under {(base_output_dir / 'ecm_results').resolve()}."
        )
    figure_directory = base_output_dir / "kappa_ecm_psds"
    linewidth_engine.render_figure(
        save_kappa_ecm_psd_figures,
        summaries,
        figure_directory,
        common_psd_ylim,
    )
    print(
        f"Replotted {len(set(int(row['kappa_index']) for row in summaries))} "
        f"kappa PSD figures with a common y-axis under "
        f"{figure_directory.resolve()}"
    )
    return figure_directory


# ----------------------------- Diagnostics --------------------------------


def sampling_and_memory_estimate():
    steps = int(np.floor(total_simulation_time / dt))
    output_states = N_lasers
    cases_per_worker = (
        int(max(1, long_run_branch_chunk_size))
        * int(max(1, n_noise_iterations))
    )
    bytes_per_step = cases_per_worker * output_states * 8
    uncapped_bytes = bytes_per_step * steps
    cap_save_every = 1
    if max_output_gb_per_worker is not None and max_output_gb_per_worker > 0.0:
        cap_save_every = max(
            1,
            int(np.ceil(uncapped_bytes / (max_output_gb_per_worker * 1.0e9))),
        )
    effective_save_every = max(analysis_save_every, cap_save_every)
    saved_dt = effective_save_every * dt
    saved_samples = int(np.ceil(steps / effective_save_every))
    burn_samples = int(np.ceil(analysis_start_time / saved_dt))
    retained_samples = max(0, saved_samples - burn_samples)
    frequency_samples = max(0, retained_samples - 1)
    nperseg = (
        frequency_samples
        if psd_welch_nperseg is None
        else min(int(psd_welch_nperseg), frequency_samples)
    )
    first_positive = 1.0 / (nperseg * saved_dt) if nperseg >= 2 else np.nan
    nyquist = 1.0 / (2.0 * saved_dt)
    trajectory_gb = cases_per_worker * output_states * saved_samples * 8 / 1.0e9
    workspace_gb = n_noise_iterations * retained_samples * 8 * 8 / 1.0e9
    active_gb = trajectory_gb * max(1, long_run_jobs) + workspace_gb * (
        1 if serialize_psd_and_saving else max(1, long_run_jobs)
    )
    return {
        "effective_save_every": effective_save_every,
        "saved_dt": saved_dt,
        "retained_samples": retained_samples,
        "first_positive_hz": first_positive,
        "nyquist_hz": nyquist,
        "trajectory_gb_per_worker": trajectory_gb,
        "cases_per_worker": cases_per_worker,
        "serialized_workspace_gb": workspace_gb,
        "active_gb": active_gb,
    }


def print_run_estimate():
    estimate = sampling_and_memory_estimate()
    print(
        "Stable-ECM PSD range estimate: "
        f"{estimate['first_positive_hz'] / 1.0e3:.4g} kHz to "
        f"{estimate['nyquist_hz'] / 1.0e9:.4g} GHz "
        f"(save_every={estimate['effective_save_every']}, "
        f"retained analysis time={linewidth_analysis_time * 1.0e6:.4g} us)."
    )
    print(
        "RAM estimate for phase-only output: "
        f"{estimate['cases_per_worker']} trajectories per worker, "
        f"{estimate['trajectory_gb_per_worker']:.2f} GB trajectory per active "
        f"worker, {estimate['serialized_workspace_gb']:.2f} GB for the one "
        f"serialized PSD workspace, about {estimate['active_gb']:.2f} GB total "
        "before solver/Python overhead."
    )


# --------------------------------- Main -----------------------------------


def main():
    configure_linewidth_engine()
    linewidth_engine.set_activity_monitor_process_title(
        "stable-ecm-catalog", warn_if_unavailable=True
    )
    if analysis_save_every < 2:
        warnings.warn(
            "analysis_save_every < 2 disables the integrator's streaming "
            "output path and can greatly increase RAM.",
            RuntimeWarning,
            stacklevel=2,
        )
    if noise_burn_in_time <= 0.0:
        warnings.warn(
            "noise_burn_in_time is nonpositive. The PSD will retain the "
            "stochastic turn-on transient.",
            RuntimeWarning,
            stacklevel=2,
        )

    for directory in (
        output_dir,
        catalog_dir,
        ecm_result_dir,
        summary_dir,
        kappa_psd_figure_dir,
        worker_error_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    kappa_values = np.linspace(kappa_initial, kappa_final, int(n_kappa_steps))
    resolved_seed = resolve_run_seed()
    catalog_payload = catalog_configuration_payload(kappa_values)
    catalog_signature = payload_signature(catalog_payload)
    linewidth_payload = linewidth_configuration_payload(
        catalog_signature,
        resolved_seed,
    )
    signature = payload_signature(linewidth_payload)
    metadata_path = output_dir / "run_configuration.json"
    if resume_existing_results and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text())
            existing_catalog_signature = str(
                existing_metadata["catalog_signature"]
            )
            existing_linewidth_signature = str(
                existing_metadata["linewidth_signature"]
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "resume_existing_results=True, but the existing "
                f"{metadata_path.resolve()} cannot be validated. Move the "
                "existing output directory aside or disable resume explicitly."
            ) from exc
        if (
            existing_catalog_signature != catalog_signature
            or existing_linewidth_signature != signature
        ):
            raise RuntimeError(
                "Refusing to resume into an output directory created by a "
                "different configuration. This protects existing per-ECM "
                "results from filename collisions. Choose a new output "
                "directory, restore the previous settings, or set "
                "resume_existing_results=False only if replacement is "
                "intentional.\n"
                f"Existing catalog signature: {existing_catalog_signature}\n"
                f"Current catalog signature:  {catalog_signature}\n"
                f"Existing linewidth signature: {existing_linewidth_signature}\n"
                f"Current linewidth signature:  {signature}"
            )

    metadata_payload = {
        "catalog_signature": catalog_signature,
        "linewidth_signature": signature,
        "resolved_run_seed": resolved_seed,
        "catalog_configuration": _jsonable(catalog_payload),
        "linewidth_configuration": _jsonable(linewidth_payload),
        "storage_configuration": _jsonable(storage_configuration_payload()),
    }
    temporary_metadata_path = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary_metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_metadata_path, metadata_path)
    print_run_estimate()
    print(f"Outputs: {output_dir.resolve()}")
    if write_overview_figure:
        print(
            "Live overview: "
            f"{(summary_dir / 'all_stable_linewidths_overview.png').resolve()}"
        )
    if write_kappa_ecm_psd_figures:
        print(f"Per-kappa ECM PSD figures: {kappa_psd_figure_dir.resolve()}")

    use_pool = int(max(1, long_run_jobs)) > 1
    executor = None
    postprocess_lock = None
    integration_progress_queue = None
    pending = set()
    current_long_chunk = []
    future_progress_id = {}
    integration_progress_bars = {}
    available_progress_positions = []
    integration_progress_lock = threading.Lock()
    integration_progress_stop = threading.Event()
    integration_progress_thread = None
    summaries = []
    completed_keys = set()
    manifold_rows_by_key = {}
    weak_burn_in_ecm_count = 0
    max_pending = int(max(1, long_run_jobs)) * (1 + int(max(0, queued_long_job_waves)))
    run_completed = False
    last_overview_render_time = -np.inf
    overview_progress_enabled = True
    latest_overview_path = None

    if use_pool:
        context = mp.get_context(multiprocessing_start_method)
        postprocess_lock = context.Lock() if serialize_psd_and_saving else None
        integration_progress_queue = (
            context.Queue() if show_long_run_integration_progress else None
        )
        executor = ProcessPoolExecutor(
            max_workers=int(max(1, long_run_jobs)),
            mp_context=context,
            initializer=_worker_initializer,
            initargs=(
                postprocess_lock,
                integration_progress_queue,
                int(max(1, long_run_jobs)),
            ),
        )
        # ProcessPoolExecutor starts its complete worker set on first submit.
        # Do that now, before the parent calls sparse eigensolvers/BLAS; forking
        # a process after those libraries have created threads is unsafe.
        executor.submit(_worker_ready).result()
    else:
        integration_progress_queue = (
            queue_module.Queue() if show_long_run_integration_progress else None
        )
        _worker_initializer(None, integration_progress_queue, 1)

    long_bar = tqdm(
        total=0,
        desc="stable ECM linewidths complete",
        unit="ECM",
        position=0,
        dynamic_ncols=True,
    )
    available_progress_positions.extend(
        range(2, 2 + max(1, max_pending))
    )

    def drain_integration_progress(block_timeout=None):
        """Move child integration-step updates onto parent-owned tqdm bars."""
        if integration_progress_queue is None:
            return False
        received = False
        first = True
        while True:
            try:
                if first and block_timeout is not None:
                    progress_id, n_steps = integration_progress_queue.get(
                        timeout=float(block_timeout)
                    )
                else:
                    progress_id, n_steps = integration_progress_queue.get_nowait()
            except queue_module.Empty:
                break
            first = False
            received = True
            with integration_progress_lock:
                bar = integration_progress_bars.get(progress_id)
                if bar is not None:
                    remaining = max(0, int(bar.total) - int(bar.n))
                    bar.update(min(int(n_steps), remaining))
        return received

    def open_integration_progress(jobs):
        if integration_progress_queue is None:
            return None
        first_job = jobs[0]
        last_job = jobs[-1]
        progress_id = (
            f"k{int(first_job['kappa_index'])}_b{int(first_job['branch_id'])}_"
            f"to_k{int(last_job['kappa_index'])}_b{int(last_job['branch_id'])}"
        )
        position = available_progress_positions.pop(0)
        total_steps = max(
            1,
            int(linewidth_engine.long_steps)
            - 1
            - 2 * int(linewidth_engine.delay_steps),
        )
        with integration_progress_lock:
            integration_progress_bars[progress_id] = tqdm(
                total=total_steps,
                desc=(
                    f"integrate {len(jobs)} ECMs, "
                    f"k={int(first_job['kappa_index'])}–"
                    f"{int(last_job['kappa_index'])}"
                ),
                unit="step",
                position=position,
                leave=False,
                dynamic_ncols=True,
            )
        first_job["progress_id"] = progress_id
        first_job["progress_update_steps"] = int(
            max(1, long_run_progress_update_steps)
        )
        first_job["progress_position"] = position
        return progress_id

    def close_integration_progress(progress_id):
        if progress_id is None:
            return
        drain_integration_progress()
        with integration_progress_lock:
            bar = integration_progress_bars.pop(progress_id, None)
            if bar is not None:
                if bar.n < bar.total:
                    bar.update(bar.total - bar.n)
                position = int(getattr(bar, "pos", -1))
                bar.close()
                # tqdm stores non-primary positions as negative internally.
                position = abs(position)
                if position >= 2 and position not in available_progress_positions:
                    available_progress_positions.append(position)
                    available_progress_positions.sort()

    def integration_progress_reporter():
        """Keep bars live while the parent is busy solving/stabilizing ECMs."""
        while not integration_progress_stop.is_set():
            drain_integration_progress(block_timeout=0.25)
        drain_integration_progress()

    if integration_progress_queue is not None:
        integration_progress_thread = threading.Thread(
            target=integration_progress_reporter,
            name="stable-ecm-progress",
            daemon=True,
        )
        integration_progress_thread.start()

    def maybe_render_overview(force=False):
        """Atomically refresh the live overview from parent-owned snapshots."""
        nonlocal last_overview_render_time, overview_progress_enabled
        nonlocal latest_overview_path
        if not write_overview_figure:
            return None
        if not force and (not write_overview_progress or not overview_progress_enabled):
            return None
        now = time.monotonic()
        if (
            not force
            and now - last_overview_render_time
            < max(0.0, float(overview_progress_min_interval_seconds))
        ):
            return None
        if not summaries and not manifold_rows_by_key:
            return None

        figure_path = summary_dir / "all_stable_linewidths_overview.png"
        try:
            linewidth_engine.render_figure(
                plot_overview,
                list(summaries),
                list(manifold_rows_by_key.values()),
                figure_path,
            )
        except Exception as exc:
            if force:
                warnings.warn(
                    f"Final overview rendering failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                overview_progress_enabled = False
                warnings.warn(
                    "Live overview rendering failed; linewidth calculations "
                    f"will continue and final rendering will be retried. {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None

        last_overview_render_time = now
        latest_overview_path = figure_path
        return figure_path

    def store_manifold_catalog(kappa_index, kappa_hz, catalog):
        for row in manifold_rows_from_catalog(kappa_index, kappa_hz, catalog):
            key = (
                int(row["kappa_index"]),
                int(row["solution_index"]),
                int(row["branch_id"]),
            )
            manifold_rows_by_key[key] = row

    def store_summary(summary):
        key = (
            int(summary["kappa_index"]),
            int(summary["solution_index"]),
            int(summary["branch_id"]),
        )
        if key in completed_keys:
            return
        completed_keys.add(key)
        summaries.append(summary)
        long_bar.update(1)
        if (
            int(summary_checkpoint_every) > 0
            and len(summaries) % int(summary_checkpoint_every) == 0
        ):
            save_flat_summary(summaries, signature)
        maybe_render_overview(force=False)

    def process_future(future):
        progress_id = future_progress_id.pop(future, None)
        try:
            chunk_summaries = future.result()
        except Exception as exc:
            close_integration_progress(progress_id)
            raise RuntimeError(
                "A stable-ECM linewidth worker failed. See worker_errors for "
                "the child traceback."
            ) from exc
        close_integration_progress(progress_id)
        for summary in chunk_summaries:
            store_summary(summary)

    def collect_one(block):
        if not pending:
            return False
        if block:
            future = None
            while future is None:
                done, _ = wait(
                    pending,
                    timeout=0.25,
                    return_when=FIRST_COMPLETED,
                )
                drain_integration_progress()
                if done:
                    future = next(iter(done))
        else:
            drain_integration_progress()
            future = next(
                (candidate for candidate in pending if candidate.done()), None
            )
            if future is None:
                return False
        pending.remove(future)
        process_future(future)
        return True

    def submit_long_chunk(force=False):
        """Submit one vectorized multi-ECM integration chunk."""
        nonlocal current_long_chunk
        chunk_size = int(max(1, long_run_branch_chunk_size))
        if not current_long_chunk or (not force and len(current_long_chunk) < chunk_size):
            return False
        jobs = current_long_chunk[:chunk_size]
        current_long_chunk = current_long_chunk[chunk_size:]
        if use_pool:
            while len(pending) >= max_pending:
                collect_one(block=True)
            progress_id = open_integration_progress(jobs)
            future = executor.submit(run_stable_ecm_linewidth_chunk, jobs)
            pending.add(future)
            future_progress_id[future] = progress_id
            collect_one(block=False)
        else:
            progress_id = open_integration_progress(jobs)
            try:
                for summary in run_stable_ecm_linewidth_chunk(jobs):
                    store_summary(summary)
            finally:
                close_integration_progress(progress_id)
        return True

    previous_roots = None
    previous_branch_ids = None
    next_branch_id = 0
    kappa_bar = tqdm(
        enumerate(kappa_values),
        total=len(kappa_values),
        desc="equilibrium and stability sweep",
        unit="kappa",
        position=1,
        dynamic_ncols=True,
    )

    try:
        for kappa_index, kappa_hz in kappa_bar:
            kappa_bar.set_postfix(kappa_ns=f"{kappa_hz * 1.0e-9:.4g}")
            catalog_path = catalog_path_for(kappa_index)
            catalog = load_equilibrium_catalog(
                catalog_path, catalog_signature, kappa_hz
            )
            if catalog is None:
                solved = solve_and_classify_equilibria(
                    kappa_hz,
                    previous_roots,
                    previous_branch_ids,
                    next_branch_id,
                )
                catalog = {
                    key: solved[key]
                    for key in (
                        "roots",
                        "branch_ids",
                        "stable_reported",
                        "stable_selected",
                        "dense_confirmed",
                        "leading_real",
                        "eigenvalue_count",
                        "eigenvalues",
                    )
                }
                next_branch_id = int(solved["next_branch_id"])
                save_equilibrium_catalog(
                    catalog_path,
                    catalog_signature,
                    kappa_index,
                    kappa_hz,
                    catalog,
                )
                vcsel = solved["vcsel"]
                nd = solved["nd"]
                _, _, kappa_matrix = constant_kappa_system(kappa_hz, noise_level=0.0)
            else:
                if catalog["branch_ids"].size:
                    next_branch_id = max(
                        next_branch_id,
                        int(np.max(catalog["branch_ids"])) + 1,
                    )
                vcsel, nd, kappa_matrix = constant_kappa_system(
                    kappa_hz, noise_level=0.0
                )

            store_manifold_catalog(kappa_index, kappa_hz, catalog)
            stable_indices = np.flatnonzero(catalog["stable_selected"])
            long_bar.total += int(stable_indices.size)
            long_bar.refresh()
            for solution_index in stable_indices:
                leading_real = float(catalog["leading_real"][solution_index])
                burn_in_constants = (
                    noise_burn_in_time * abs(leading_real) / tau_p
                    if np.isfinite(leading_real) and leading_real < 0.0
                    else 0.0
                )
                if burn_in_constants < minimum_burn_in_decay_time_constants:
                    weak_burn_in_ecm_count += 1
                branch_id = int(catalog["branch_ids"][solution_index])
                root = np.asarray(catalog["roots"][solution_index], dtype=float)
                result_path = result_path_for(kappa_index, solution_index, branch_id)
                resumed = load_result_summary(
                    result_path,
                    signature,
                    root,
                    expected_full_psd=should_save_full_psd(kappa_index, branch_id),
                )
                if resumed is not None:
                    store_summary(resumed)
                    continue

                job = make_stable_job(
                    signature,
                    resolved_seed,
                    kappa_index,
                    solution_index,
                    branch_id,
                    kappa_hz,
                    root,
                    catalog["leading_real"][solution_index],
                    catalog["eigenvalue_count"][solution_index],
                    catalog["dense_confirmed"][solution_index],
                    nd,
                    kappa_matrix,
                    result_path,
                )
                current_long_chunk.append(job)
                submit_long_chunk(force=False)

            # Process one completed result even when this catalog has no
            # stable roots, then refresh catalog-only progress if the cadence
            # allows it.
            if use_pool:
                collect_one(block=False)
            maybe_render_overview(force=False)

            previous_roots = np.asarray(catalog["roots"], dtype=float)
            previous_branch_ids = np.asarray(catalog["branch_ids"], dtype=int)
            del catalog, vcsel, nd, kappa_matrix
            gc.collect()

        while current_long_chunk:
            submit_long_chunk(force=True)
        while pending:
            collect_one(block=True)
        run_completed = True
    finally:
        integration_progress_stop.set()
        if integration_progress_thread is not None:
            integration_progress_thread.join(timeout=2.0)
        drain_integration_progress()
        with integration_progress_lock:
            for bar in list(integration_progress_bars.values()):
                bar.close()
            integration_progress_bars.clear()
        kappa_bar.close()
        long_bar.close()
        if executor is not None:
            linewidth_engine.shutdown_process_pool(executor, completed=run_completed)

        # The per-ECM files are the source of truth. In particular, do not let
        # Ctrl-C during a resume replace a complete summary with only the
        # subset that this invocation happened to rediscover before stopping.
        persisted_summaries = load_saved_full_ecm_summaries(
            output_dir,
            expected_signature=signature,
        )
        summaries_by_key = {
            (
                int(row["kappa_index"]),
                int(row["solution_index"]),
                int(row["branch_id"]),
            ): row
            for row in summaries
        }
        for row in persisted_summaries:
            key = (
                int(row["kappa_index"]),
                int(row["solution_index"]),
                int(row["branch_id"]),
            )
            summaries_by_key[key] = row
        summaries = sorted_summaries(list(summaries_by_key.values()))

        # Recover the complete previously saved manifold as well when resume
        # is interrupted before the catalog loop reaches its former endpoint.
        for row in load_saved_manifold_rows_for_replot(output_dir):
            key = (
                int(row["kappa_index"]),
                int(row["solution_index"]),
                int(row["branch_id"]),
            )
            manifold_rows_by_key[key] = row
        save_flat_summary(summaries, signature)
        maybe_render_overview(force=True)
        if write_kappa_ecm_psd_figures and summaries:
            try:
                linewidth_engine.render_figure(
                    save_kappa_ecm_psd_figures,
                    list(summaries),
                    kappa_psd_figure_dir,
                )
            except Exception as exc:
                warnings.warn(
                    "Per-kappa ECM PSD figure generation failed after the "
                    f"scientific result files were saved: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if latest_overview_path is not None:
        print(f"Saved overview: {latest_overview_path.resolve()}")
    if weak_burn_in_ecm_count:
        warnings.warn(
            f"{weak_burn_in_ecm_count} stable ECM(s) had fewer than "
            f"{minimum_burn_in_decay_time_constants:g} leading-mode decay "
            "time constants in noise_burn_in_time. Inspect "
            "burn_in_decay_time_constants in the summary and increase the "
            "burn-in for near-marginal branches.",
            RuntimeWarning,
            stacklevel=2,
        )
    print(
        f"Saved {len(summaries)} stable-ECM linewidth results and summary files "
        f"under {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
