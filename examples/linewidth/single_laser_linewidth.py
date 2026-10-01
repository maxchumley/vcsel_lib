#%%
"""Single-laser delayed-feedback linewidth from phase diffusion.

This mirrors the continuation structure in ``examples/linewidth_autocorr.py``:

1. Run short continuation simulations in increasing feedback strength.
2. Use the final averaged history from each short run as the initial history
   for a longer fixed-feedback run.
3. Run those longer fixed-feedback cases in parallel chunks.
4. Estimate linewidth from the phase-increment variance

       var[phi(t + Delta t) - phi(t)] = 2*pi*Delta nu*Delta t.

For comparison, the progress plot overlays the deterministic weak-feedback
Henry external-cavity-mode branches for the same feedback phase.

Implementation note:
``vcsel_lib`` has a separate ``self_feedback`` flag for a 2*tau term.  This
script intentionally leaves that off and uses the diagonal entry of the
coupling matrix, ``kappa_c_mat[..., 0, 0]``, as the one-delay feedback path.
"""

from __future__ import annotations

import gc
import inspect
import multiprocessing as mp
import os
import queue as queue_module
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
# Make the repositorys ``examples`` package available to scripts and notebooks.
import sys

_path_search_starts = [Path.cwd().resolve()]
if "__file__" in globals():
    _path_search_starts.append(Path(__file__).resolve().parent)
_repo_root = next(
    (candidate for start in _path_search_starts for candidate in (start, *start.parents)
     if (candidate / "examples" / "_paths.py").is_file()),
    None,
)
if _repo_root is None:
    raise ModuleNotFoundError(
        "Could not locate the vcsel_lib repository root containing examples/_paths.py."
    )
sys.path.insert(0, str(_repo_root))
for _module_name in tuple(sys.modules):
    if _module_name == "examples" or _module_name.startswith("examples."):
        del sys.modules[_module_name]


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator
from tqdm.auto import tqdm


def suppress_macos_malloc_stack_logging_warnings():
    """Prevent macOS malloc-debug env vars from leaking into worker processes."""
    for key in (
        "MallocStackLogging",
        "MallocStackLoggingNoCompact",
        "MallocStackLoggingDirectory",
        "MallocStackLoggingCompact",
    ):
        os.environ.pop(key, None)


suppress_macos_malloc_stack_logging_warnings()


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._paths import LINEWIDTH_RESULTS_DIR
from vcsel_lib import VCSEL


# ----------------------------- User controls -----------------------------

n_noise_iterations = 50
random_seed = None

kappa_initial = 0.0
kappa_final = 40.0e9
n_kappa_steps = 100

Tmax_continuation = 5.0e-7
Tmax_long = 5.0e-6
dt_multiplier = 1.0
continuation_save_every = 8
analysis_save_every = 4

two_stage_kappa_chunk_size = 5
long_run_jobs = 2
show_long_run_worker_progress = True
show_output_size_messages = False

plot_lag_max_ns = 2000.0
fit_lag_ns = (50.0, 500.0)
n_lag_points = 2000

write_progress_arrays = True
write_progress_frame = True
write_fit_frames = True
write_time_series_frames = True


# ------------------------ Plot style / fixed choices ----------------------

long_run_progress_update_steps = 200
max_plot_points = 6000
time_series_max_points = 6000
figure_dpi = 160
continuation_frame_dpi = 300
continuation_frame_font_size = 22
continuation_frame_title_pad = 16
continuation_linewidth_legend_font_size = 13
linewidth_plot_ylim_mhz = (1e-3, 1e7)
linewidth_plot_yticks_mhz = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 1e5, 1e6, 1e7]
phase_variance_colorbar_log_scale = True
phase_variance_log_vmin_rad2 = None
phase_variance_log_vmax_rad2 = None
phase_variance_log_vmin_percentile = 1.0
phase_variance_log_vmax_percentile = 99.5
phase_diffusion_colorbar_ticks = None
instantaneous_frequency_colorbar_percentiles = (1.0, 99.0)
optical_spectrum_colorbar_floor_db = -90.0
optical_spectrum_colorbar_percentiles = (1.0, 99.8)
intensity_psd_colorbar_percentiles = (5.0, 99.5)
intensity_psd_floor = 1e-30
kappa_ramp_start_tau = 5.0
kappa_ramp_shape_tau = 200.0
phi_p = 0.0
noise_amplitude = 1.0
average_continuation_history_across_noise = True

analysis_fraction = 1.0
post_ramp_settling_ns = 250.0
phase_variance_lag_axis_ns = (0.0, 2000.0)
phase_diffusion_fit_ylim = None
phase_diffusion_fit_yticks = None
maximum_fit_lag_fraction = 0.1
use_available_data_if_tmax_too_short = True
minimum_analysis_samples = 16
long_run_history_is_steady_state = True
phase_plot_tail_window_us = 2.0

output_dir = LINEWIDTH_RESULTS_DIR / "1_laser/phase_diffusion_linewidth_continuation"
frames_dir = output_dir / "kappa_frames"
fit_frames_dir = output_dir / "phase_variance_fit_frames"
time_series_frames_dir = output_dir / "time_series_frames"
arrays_dir = output_dir / "numpy_arrays"
worker_arrays_dir = output_dir / "worker_arrays"


PHASE_VARIANCE_LABEL = r"$\mathrm{var}[\phi(t+\Delta t)-\phi(t)]$ (rad$^2$)"
PHASE_VARIANCE_SHORT_LABEL = r"phase-increment variance (rad$^2$)"


# --------------------------- Model parameters ----------------------------

alpha = 2.0
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
saturation = 4e-6
q = 1.602e-19
beta = 1.0e-3
tau = 1.0e-9
eta = 0.9
threshold_multiplier = 3.0

N_lasers = 1
dt = dt_multiplier * tau_p
continuation_steps = int(np.floor(Tmax_continuation / dt))
time_array_continuation = np.arange(continuation_steps, dtype=float) * dt
Tmax_continuation = time_array_continuation[-1]
long_steps = int(np.floor(Tmax_long / dt))
time_array_long = np.arange(long_steps, dtype=float) * dt
Tmax_long = time_array_long[-1]
delay_steps = int(round(tau / dt))


def threshold_current():
    return threshold_multiplier * q / tau_n * (N0 + 1.0 / (g0 * tau_p))


def drive_current():
    return eta * threshold_current()


def average_history_across_cases(history):
    """Average a final history over noise cases and copy it to every case."""
    history = np.asarray(history)
    if history.ndim != 3 or history.shape[0] <= 1:
        return history.copy()
    mean_history = np.mean(history, axis=0, keepdims=True)
    return np.repeat(mean_history, history.shape[0], axis=0).astype(
        history.dtype,
        copy=False,
    )


def linear_fit_with_r_squared(x, y, min_points):
    """Fit y = slope*x + intercept and return slope/intercept/R²."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < int(max(2, min_points)):
        return np.nan, np.nan, np.nan
    x_fit = x[finite]
    y_fit = y[finite]
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    y_line = slope * x_fit + intercept
    residual_sum = np.sum((y_fit - y_line) ** 2)
    total_sum = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else np.nan
    return float(slope), float(intercept), float(r_squared)


def phase_diffusion_linewidth(
    t,
    phase,
    lag_range_ns,
    max_lag_ns,
    n_lags,
    maximum_fit_fraction=0.25,
):
    """Estimate linewidth from unwrapped phase-increment variance growth."""
    t = np.asarray(t, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if phase.ndim == 1:
        phase = phase[None, :]
    if phase.ndim != 2 or phase.shape[1] != t.size:
        raise ValueError("phase must have shape (n_cases, Nt).")

    dt_values = np.diff(t)
    dt_local = float(np.median(dt_values))
    if np.any(dt_values <= 0.0) or not np.allclose(dt_values, dt_local, rtol=1e-6):
        raise ValueError("t must be uniformly sampled and strictly increasing.")

    lag_min = float(lag_range_ns[0]) * 1e-9
    requested_lag_max = float(lag_range_ns[1]) * 1e-9
    analysis_duration = t[-1] - t[0]
    reliable_fit_lag_max = maximum_fit_fraction * analysis_duration
    lag_limit_tolerance = max(dt_local, 1e-12 * analysis_duration)
    largest_valid_lag = (t.size - 2) * dt_local
    if requested_lag_max > reliable_fit_lag_max + lag_limit_tolerance:
        if use_available_data_if_tmax_too_short:
            fit_lag_max = min(requested_lag_max, reliable_fit_lag_max, largest_valid_lag)
        else:
            required_duration = requested_lag_max / maximum_fit_fraction
            raise ValueError(
                f"fit_lag_ns ends at {requested_lag_max * 1e9:.3g} ns, but "
                f"the retained analysis duration is only {analysis_duration * 1e9:.3g} ns. "
                f"Retain at least {required_duration * 1e6:.3g} us or shorten fit_lag_ns."
            )
    else:
        fit_lag_max = min(requested_lag_max, largest_valid_lag)

    plot_lag_max = min(float(max_lag_ns) * 1e-9, largest_valid_lag)
    plot_lag_max = max(plot_lag_max, fit_lag_max)
    if fit_lag_max <= lag_min:
        if use_available_data_if_tmax_too_short:
            lag_min = dt_local
            fit_lag_max = min(largest_valid_lag, max(fit_lag_max, 3.0 * dt_local))
            plot_lag_max = max(plot_lag_max, fit_lag_max)
        if fit_lag_max <= lag_min:
            raise ValueError("The analysis trace is too short for the selected lag range.")

    max_lag_samples = int(np.floor(plot_lag_max / dt_local))
    lag_samples = np.unique(
        np.round(np.linspace(1, max_lag_samples, int(max(10, n_lags)))).astype(int)
    )
    lag_times = lag_samples * dt_local

    # Remove constant offsets and the deterministic lasing-frequency slope.
    t_centered = t - np.mean(t)
    trend_denominator = np.dot(t_centered, t_centered)
    n_samples = t.size
    fft_length = 1 << int(np.ceil(np.log2(2 * n_samples - 1)))

    variance_cases = np.empty((phase.shape[0], lag_samples.size), dtype=float)
    for case_index in range(phase.shape[0]):
        trace = np.unwrap(np.asarray(phase[case_index], dtype=float))
        trace -= np.mean(trace)
        trend_slope = np.dot(trace, t_centered) / trend_denominator
        trace -= trend_slope * t_centered

        spectrum = np.fft.rfft(trace, n=fft_length)
        autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_length)[:n_samples]

        cumulative = np.concatenate(([0.0], np.cumsum(trace)))
        cumulative_square = np.concatenate(([0.0], np.cumsum(trace * trace)))
        pair_counts = n_samples - lag_samples
        left_sum = cumulative[n_samples - lag_samples]
        right_sum = cumulative[n_samples] - cumulative[lag_samples]
        left_square_sum = cumulative_square[n_samples - lag_samples]
        right_square_sum = cumulative_square[n_samples] - cumulative_square[lag_samples]
        mean_increment = (right_sum - left_sum) / pair_counts
        mean_square_increment = (
            left_square_sum + right_square_sum - 2.0 * autocorrelation[lag_samples]
        ) / pair_counts
        variance_cases[case_index] = np.maximum(
            mean_square_increment - mean_increment * mean_increment,
            0.0,
        )

    # Finite-record trend removal makes an ideal Brownian phase look like a
    # Brownian bridge. Correct that bias in the small-lag regime used for fits.
    increment_record_duration = analysis_duration - lag_times
    bridge_factor = (
        1.0
        - lag_times / increment_record_duration
        + lag_times * lag_times / (3.0 * increment_record_duration**2)
    )
    correctable = (
        (lag_times <= 0.5 * analysis_duration)
        & np.isfinite(bridge_factor)
        & (bridge_factor > 0.0)
    )
    variance_cases[:, correctable] /= bridge_factor[correctable]

    fit_mask = (lag_times >= lag_min) & (lag_times <= fit_lag_max)
    min_fit_points = 3
    if np.count_nonzero(fit_mask) < min_fit_points:
        if use_available_data_if_tmax_too_short and lag_times.size >= min_fit_points:
            fit_mask = np.zeros_like(lag_times, dtype=bool)
            fit_mask[:min_fit_points] = True
        else:
            raise ValueError("The selected fit range contains too few lag points.")

    phase_variance = np.nanmean(variance_cases, axis=0)
    finite_counts = np.sum(np.isfinite(variance_cases), axis=0)
    if variance_cases.shape[0] > 1:
        phase_variance_sem = np.nanstd(variance_cases, axis=0, ddof=1) / np.sqrt(
            np.maximum(finite_counts, 1)
        )
        phase_variance_sem[finite_counts < 2] = 0.0
    else:
        phase_variance_sem = np.zeros_like(phase_variance)

    linewidth_cases_hz = np.full(phase.shape[0], np.nan, dtype=float)
    case_r_squared = np.full(phase.shape[0], np.nan, dtype=float)
    for case_index in range(phase.shape[0]):
        case_slope, _, case_r_squared[case_index] = linear_fit_with_r_squared(
            lag_times[fit_mask],
            variance_cases[case_index, fit_mask],
            min_points=min_fit_points,
        )
        if np.isfinite(case_slope) and case_slope > 0.0:
            linewidth_cases_hz[case_index] = case_slope / (2.0 * np.pi)

    fit_time = lag_times[fit_mask]
    fit_variance = phase_variance[fit_mask]
    finite_fit = np.isfinite(fit_variance)
    if np.count_nonzero(finite_fit) < min_fit_points:
        raise ValueError("Too few finite phase-variance points in the fit range.")
    fit_time = fit_time[finite_fit]
    fit_variance = fit_variance[finite_fit]
    slope, intercept, r_squared = linear_fit_with_r_squared(
        fit_time,
        fit_variance,
        min_points=min_fit_points,
    )
    fit_line = slope * fit_time + intercept
    linewidth_hz = slope / (2.0 * np.pi) if slope > 0.0 else np.nan

    valid_linewidths = linewidth_cases_hz[np.isfinite(linewidth_cases_hz)]
    linewidth_median_hz = float(np.median(valid_linewidths)) if valid_linewidths.size else np.nan
    linewidth_std_hz = (
        float(np.std(valid_linewidths, ddof=1)) if valid_linewidths.size > 1 else np.nan
    )

    return {
        "linewidth_hz": linewidth_hz,
        "linewidth_cases_hz": linewidth_cases_hz,
        "linewidth_median_hz": linewidth_median_hz,
        "linewidth_std_hz": linewidth_std_hz,
        "r_squared": r_squared,
        "case_r_squared": case_r_squared,
        "lag_times": lag_times,
        "phase_variance": phase_variance,
        "phase_variance_sem": phase_variance_sem,
        "fit_time": fit_time,
        "fit_line": fit_line,
    }


# -------------------------- VCSEL construction ---------------------------


def make_kappa_matrix(time_array, kappa_start, kappa_stop):
    """One-laser one-delay feedback matrix with a smooth continuation ramp."""
    kappa_trace = VCSEL.cosine_ramp(
        time_array,
        t_start=kappa_ramp_start_tau * tau,
        rise_10_90=kappa_ramp_shape_tau * tau,
        kappa_initial=float(kappa_start),
        kappa_final=float(kappa_stop),
    )
    kappa_matrix = np.zeros((time_array.size, 1, 1), dtype=float)
    kappa_matrix[:, 0, 0] = kappa_trace
    return kappa_matrix


def make_vcsel_from_kappa_matrix(kappa_matrix, tmax, save_every):
    physical_parameters = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "s": saturation,
        "q": q,
        "alpha": alpha,
        "beta": beta,
        "kappa_c_mat": kappa_matrix,
        "phi_p_mat": np.asarray([[phi_p]], dtype=float),
        "tau": tau,
        "I": drive_current(),
        "noise_amplitude": noise_amplitude,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "delta": np.asarray([0.0], dtype=float),
        "Tmax": tmax,
        "dt": dt,
        "N_lasers": 1,
        "save_every": int(max(1, save_every)),
        "output_dtype": np.float32,
    }
    vcsel = VCSEL(physical_parameters)
    nd = vcsel.scale_params()
    nd["store_freqs"] = False
    nd["show_output_size_message"] = bool(show_output_size_messages)
    nd["output_dtype"] = np.float32
    nd["output_state_indices"] = np.asarray([1, 2], dtype=int)
    return vcsel, nd


def make_vcsel_ramp(kappa_start, kappa_stop, tmax, time_array, save_every):
    return make_vcsel_from_kappa_matrix(
        make_kappa_matrix(time_array, kappa_start, kappa_stop),
        tmax,
        save_every,
    )


def analysis_start_time_for_record(t_stop):
    full_ramp_duration = kappa_ramp_shape_tau * tau / 0.8
    ramp_end_time = kappa_ramp_start_tau * tau + full_ramp_duration
    stationary_start_time = ramp_end_time + post_ramp_settling_ns * 1e-9
    fraction_start_time = (1.0 - analysis_fraction) * t_stop
    return max(stationary_start_time, fraction_start_time)


def analysis_start_index_for_record(t, already_steady=False):
    t = np.asarray(t, dtype=float)
    if t.size < 4:
        raise ValueError("The saved analysis record has fewer than four samples.")
    if already_steady:
        return 0

    requested_start_time = analysis_start_time_for_record(t[-1])
    requested_start = int(np.searchsorted(t, requested_start_time, side="left"))
    if not use_available_data_if_tmax_too_short:
        return requested_start

    min_samples = int(max(4, minimum_analysis_samples))
    min_samples = min(min_samples, t.size)
    latest_start = max(0, t.size - min_samples)
    if requested_start >= t.size - 3:
        warnings.warn(
            "Configured ramp/settling cutoff leaves no stationary samples; "
            "using the full available record for phase-diffusion analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0
    if requested_start > latest_start:
        warnings.warn(
            "Configured analysis cutoff leaves only a very short tail; "
            f"backing up to keep {t.size - latest_start} saved samples.",
            RuntimeWarning,
            stacklevel=2,
        )
        return latest_start
    return requested_start


def validate_analysis_window():
    analysis_start_time = 0.0 if long_run_history_is_steady_state else analysis_start_time_for_record(Tmax_long)
    retained_duration = Tmax_long - analysis_start_time
    requested_fit_lag_max = float(fit_lag_ns[1]) * 1e-9
    allowed_fit_lag_max = maximum_fit_lag_fraction * retained_duration
    lag_limit_tolerance = max(dt, 1e-12 * max(retained_duration, 0.0))

    if retained_duration <= 0.0:
        if use_available_data_if_tmax_too_short:
            warnings.warn(
                "No stationary analysis record remains; using the full long-run record.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        raise ValueError("No stationary analysis record remains.")

    if requested_fit_lag_max > allowed_fit_lag_max + lag_limit_tolerance:
        if use_available_data_if_tmax_too_short:
            warnings.warn(
                "Requested fit_lag_ns extends beyond the reliable lag range "
                "for this Tmax_long; clipping the fit window to the available record.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        required_retained_duration = requested_fit_lag_max / maximum_fit_lag_fraction
        raise ValueError(
            "The linewidth fit window is too long for the retained analysis record.\n"
            f"  retained record = {retained_duration * 1e9:.1f} ns\n"
            f"  fit_lag_ns ends at {fit_lag_ns[1]:.1f} ns\n"
            f"  use Tmax_long >= {required_retained_duration * 1e6:.3f} us."
        )


def estimate_long_output_gb(kappa_chunk_size=None):
    if kappa_chunk_size is None:
        kappa_chunk_size = two_stage_kappa_chunk_size
    saved_steps = int(np.ceil(long_steps / int(max(1, analysis_save_every))))
    n_cases_worker = int(max(1, kappa_chunk_size)) * int(max(1, n_noise_iterations))
    bytes_est = n_cases_worker * 2 * saved_steps * np.dtype(np.float32).itemsize
    return bytes_est / 1e9


# -------------------------- Henry comparison -----------------------------


def free_running_henry_linewidth_hz(nd):
    """Free-running Schawlow-Townes-Henry linewidth from dimensional model variables.

    The usual dimensional rate-equation form is

        Δν0 = (1 + alpha**2) R_sp / (4π S_bar),

    where S_bar is the steady-state intracavity photon number and R_sp is the
    spontaneous-emission rate into the lasing mode.  In the vcsel_lib rate
    equations,

        R_sp = beta * N_bar / tau_n,
        N_bar = (nbar + n0 + 1) / (g0 * tau_p),
        S_bar = sbar / (g0 * tau_n).

    Substituting those dimensional quantities gives the same value as the
    compact scaled expression, but this form keeps the physical origin visible.
    """
    n0 = float(nd["n0"])
    nbar = float(nd["nbar"])
    sbar = float(nd["sbar"])
    carrier_number = (nbar + n0 + 1.0) / (g0 * tau_p)
    photon_number = sbar / (g0 * tau_n)
    spontaneous_emission_rate = beta * carrier_number / tau_n
    return (1.0 + alpha**2) * spontaneous_emission_rate / (
        4.0 * np.pi * max(photon_number, 1e-30)
    )


def henry_feedback_linewidth_hz(delta_nu_0_hz, kappa_value, omega_rad_s):
    """Weak-feedback Henry correction for one-delay feedback."""
    phase = omega_rad_s * tau + phi_p + np.arctan(alpha)
    denominator = 1.0 + kappa_value * tau * np.sqrt(1.0 + alpha**2) * np.cos(phase)
    return delta_nu_0_hz / np.maximum(denominator**2, 1e-30)


def external_cavity_mode_roots(kappa_value):
    """Return deterministic ECM roots Ω for a feedback strength.

    Roots solve x + C sin(x + phi_p + atan(alpha)) = 0, with x = Ωτ.
    """
    feedback_c = float(kappa_value) * tau * np.sqrt(1.0 + alpha**2)
    if not np.isfinite(feedback_c) or feedback_c < 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if feedback_c == 0.0:
        root = np.array([0.0], dtype=float)
        denominator = np.array([1.0], dtype=float)
        return root / tau, denominator

    theta = phi_p + np.arctan(alpha)
    x_grid = np.linspace(-feedback_c, feedback_c, int(max(1000, np.ceil(128.0 * max(1.0, feedback_c) / np.pi))))
    f_grid = x_grid + feedback_c * np.sin(x_grid + theta)
    roots = []

    exact = np.flatnonzero(np.isclose(f_grid, 0.0, atol=1e-12, rtol=0.0))
    roots.extend(float(x_grid[idx]) for idx in exact)

    sign_change = np.flatnonzero(f_grid[:-1] * f_grid[1:] < 0.0)
    for idx in sign_change:
        lo = float(x_grid[idx])
        hi = float(x_grid[idx + 1])
        flo = float(f_grid[idx])
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = mid + feedback_c * np.sin(mid + theta)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo = mid
                flo = fmid
            if abs(hi - lo) < 1e-13:
                break
        roots.append(0.5 * (lo + hi))

    if not roots:
        return np.array([], dtype=float), np.array([], dtype=float)

    roots = np.asarray(sorted(roots), dtype=float)
    deduped = [float(roots[0])]
    for root in roots[1:]:
        if abs(root - deduped[-1]) > 1e-8:
            deduped.append(float(root))
    roots = np.asarray(deduped, dtype=float)
    denominator = 1.0 + feedback_c * np.cos(roots + theta)
    return roots / tau, denominator


def henry_ecm_branch_points(kappa_values, delta_nu_0_hz):
    """Flatten all deterministic Henry ECM branches into plottable point arrays."""
    kappa_points = []
    linewidth_points = []
    positive_slope_points = []
    omega_points = []
    denominator_points = []

    for kappa_value in np.asarray(kappa_values, dtype=float):
        omega_roots, denominators = external_cavity_mode_roots(kappa_value)
        if omega_roots.size == 0:
            continue
        branch_linewidth = henry_feedback_linewidth_hz(
            delta_nu_0_hz,
            kappa_value,
            omega_roots,
        )
        finite = np.isfinite(branch_linewidth) & (branch_linewidth > 0.0)
        if not np.any(finite):
            continue
        kappa_points.append(np.full(np.count_nonzero(finite), kappa_value, dtype=float))
        linewidth_points.append(branch_linewidth[finite])
        positive_slope_points.append(denominators[finite] > 0.0)
        omega_points.append(omega_roots[finite])
        denominator_points.append(denominators[finite])

    if not kappa_points:
        return {
            "kappa_values": np.array([], dtype=float),
            "linewidth_hz": np.array([], dtype=float),
            "omega_rad_s": np.array([], dtype=float),
            "denominator": np.array([], dtype=float),
            "positive_slope": np.array([], dtype=bool),
        }

    return {
        "kappa_values": np.concatenate(kappa_points),
        "linewidth_hz": np.concatenate(linewidth_points),
        "omega_rad_s": np.concatenate(omega_points),
        "denominator": np.concatenate(denominator_points),
        "positive_slope": np.concatenate(positive_slope_points),
    }


# ----------------------------- Analysis ----------------------------------


def feedback_phase_trace(t, phase):
    """Wrapped one-delay feedback phase phi(t-tau)-phi(t)-phi_p."""
    phase = np.unwrap(np.asarray(phase), axis=-1)
    lag_samples = int(round(tau / np.median(np.diff(t))))
    lag_samples = max(1, min(lag_samples, phase.shape[-1] - 1))
    phase_lag = phase[:, :-lag_samples] - phase[:, lag_samples:] - phi_p
    return t[lag_samples:], np.angle(np.exp(1j * phase_lag))


def instantaneous_frequency_trace(t, phase):
    """Mean and noise-realization variance of instantaneous frequency in GHz."""
    t = np.asarray(t, dtype=float)
    phase = np.unwrap(np.asarray(phase), axis=-1)
    if t.size < 2 or phase.shape[-1] < 2:
        nan_trace = np.full(t.shape, np.nan, dtype=float)
        return t, nan_trace, nan_trace
    omega_cases = np.gradient(phase, t, axis=-1, edge_order=1)
    frequency_cases_ghz = omega_cases / (2.0 * np.pi * 1e9)
    frequency_mean_ghz = np.nanmean(frequency_cases_ghz, axis=0)
    if frequency_cases_ghz.shape[0] > 1:
        frequency_variance_ghz2 = np.nanvar(frequency_cases_ghz, axis=0, ddof=1)
    else:
        frequency_variance_ghz2 = np.zeros_like(frequency_mean_ghz)
    return t, frequency_mean_ghz, frequency_variance_ghz2


def single_laser_time_series_summary(t, intensity, phase):
    """Mean/std time-series diagnostics for one fixed-kappa single-laser run."""
    t = np.asarray(t, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    phase = np.unwrap(np.asarray(phase), axis=-1)
    if intensity.ndim == 1:
        intensity = intensity[None, :]
    if phase.ndim == 1:
        phase = phase[None, :]
    if t.size < 4 or intensity.shape[-1] < 4 or phase.shape[-1] < 4:
        empty = np.array([], dtype=float)
        return {
            "time": empty,
            "intensity_mean": empty,
            "intensity_std": empty,
            "frequency_mean_ghz": empty,
            "frequency_std_ghz": empty,
            "feedback_phase_time": empty,
            "feedback_phase_abs_mean": empty,
            "feedback_phase_abs_std": empty,
        }

    n_samples = min(t.size, intensity.shape[-1], phase.shape[-1])
    t = t[:n_samples]
    intensity = intensity[:, :n_samples]
    phase = phase[:, :n_samples]
    intensity_mean = np.nanmean(intensity, axis=0)
    intensity_std = (
        np.nanstd(intensity, axis=0, ddof=1)
        if intensity.shape[0] > 1
        else np.zeros_like(intensity_mean)
    )

    omega_cases = np.gradient(phase, t, axis=-1, edge_order=1)
    frequency_cases_ghz = omega_cases / (2.0 * np.pi * 1e9)
    frequency_mean_ghz = np.nanmean(frequency_cases_ghz, axis=0)
    frequency_std_ghz = (
        np.nanstd(frequency_cases_ghz, axis=0, ddof=1)
        if frequency_cases_ghz.shape[0] > 1
        else np.zeros_like(frequency_mean_ghz)
    )

    dt_local = float(np.median(np.diff(t)))
    if np.isfinite(dt_local) and dt_local > 0.0 and n_samples > 2:
        lag_samples = int(round(tau / dt_local))
        lag_samples = max(1, min(lag_samples, n_samples - 1))
        feedback_phase_cases = np.abs(
            np.angle(
                np.exp(
                    1j
                    * (
                        phase[:, :-lag_samples]
                        - phase[:, lag_samples:]
                        - phi_p
                    )
                )
            )
        )
        feedback_phase_time = t[lag_samples:]
        feedback_phase_abs_mean = np.nanmean(feedback_phase_cases, axis=0)
        feedback_phase_abs_std = (
            np.nanstd(feedback_phase_cases, axis=0, ddof=1)
            if feedback_phase_cases.shape[0] > 1
            else np.zeros_like(feedback_phase_abs_mean)
        )
    else:
        feedback_phase_time = np.array([], dtype=float)
        feedback_phase_abs_mean = np.array([], dtype=float)
        feedback_phase_abs_std = np.array([], dtype=float)

    t_ds, values_ds = downsample_time_series(
        t - t[0],
        np.vstack(
            [
                intensity_mean,
                intensity_std,
                frequency_mean_ghz,
                frequency_std_ghz,
            ]
        ),
        max_points=time_series_max_points,
    )
    if feedback_phase_time.size:
        feedback_t_ds, feedback_values_ds = downsample_time_series(
            feedback_phase_time - t[0],
            np.vstack([feedback_phase_abs_mean, feedback_phase_abs_std]),
            max_points=time_series_max_points,
        )
    else:
        feedback_t_ds = feedback_phase_time
        feedback_values_ds = np.empty((2, 0), dtype=float)

    return {
        "time": t_ds,
        "intensity_mean": values_ds[0],
        "intensity_std": values_ds[1],
        "frequency_mean_ghz": values_ds[2],
        "frequency_std_ghz": values_ds[3],
        "feedback_phase_time": feedback_t_ds,
        "feedback_phase_abs_mean": feedback_values_ds[0],
        "feedback_phase_abs_std": feedback_values_ds[1],
    }


def optical_field_spectrum(t, intensity, phase):
    """Mean normalized optical spectrum of E=sqrt(S) exp(i phi), in dB."""
    t = np.asarray(t, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    phase = np.unwrap(np.asarray(phase), axis=-1)
    if intensity.ndim == 1:
        intensity = intensity[None, :]
    if phase.ndim == 1:
        phase = phase[None, :]
    if t.size < 4 or intensity.shape[-1] < 4 or phase.shape[-1] < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    dt_local = float(np.median(np.diff(t)))
    if not np.isfinite(dt_local) or dt_local <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_samples = min(intensity.shape[-1], phase.shape[-1])
    window = np.hanning(n_samples)
    window_power = float(np.sum(window**2))
    if window_power <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)

    frequency_ghz = np.fft.fftshift(np.fft.fftfreq(n_samples, d=dt_local)) * 1e-9
    power_sum = np.zeros(n_samples, dtype=float)
    n_valid = 0

    for intensity_case, phase_case in zip(intensity, phase):
        intensity_case = np.asarray(intensity_case[..., :n_samples], dtype=float)
        phase_case = np.asarray(phase_case[..., :n_samples], dtype=float)
        finite_case = np.isfinite(intensity_case) & np.isfinite(phase_case)
        if not np.any(finite_case):
            continue
        field = np.sqrt(np.maximum(np.nan_to_num(intensity_case), 0.0)) * np.exp(
            1j * np.nan_to_num(phase_case)
        )
        spectrum = np.fft.fft(field * window)
        power_sum += np.abs(np.fft.fftshift(spectrum)) ** 2 / window_power
        n_valid += 1

    if n_valid == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    mean_power = power_sum / n_valid
    peak_power = float(np.nanmax(mean_power))
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)
    normalized_power = mean_power / peak_power
    spectrum_db = 10.0 * np.log10(
        np.maximum(normalized_power, 10.0 ** (optical_spectrum_colorbar_floor_db / 10.0))
    )
    return frequency_ghz, spectrum_db


def relative_intensity_psd(t, intensity):
    """Mean one-sided PSD of relative intensity fluctuations in dB/Hz."""
    t = np.asarray(t, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim == 1:
        intensity = intensity[None, :]
    if t.size < 4 or intensity.shape[-1] < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    dt_local = float(np.median(np.diff(t)))
    if not np.isfinite(dt_local) or dt_local <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_samples = intensity.shape[-1]
    window = np.hanning(n_samples)
    window_power = float(np.sum(window**2))
    if window_power <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)

    frequency_hz = np.fft.rfftfreq(n_samples, d=dt_local)
    psd_sum = np.zeros(frequency_hz.size, dtype=float)
    n_valid = 0

    for intensity_case in intensity:
        intensity_case = np.asarray(intensity_case, dtype=float)
        mean_intensity = float(np.nanmean(intensity_case))
        if not np.isfinite(mean_intensity) or abs(mean_intensity) <= 0.0:
            continue
        relative_fluctuation = (intensity_case - mean_intensity) / mean_intensity
        relative_fluctuation = np.nan_to_num(relative_fluctuation, copy=False)
        spectrum = np.fft.rfft(relative_fluctuation * window)
        psd = (dt_local / window_power) * np.abs(spectrum) ** 2
        if psd.size > 2:
            psd[1:-1] *= 2.0
        psd_sum += psd
        n_valid += 1

    if n_valid == 0 or frequency_hz.size <= 1:
        return np.array([], dtype=float), np.array([], dtype=float)

    mean_psd = psd_sum / n_valid
    frequency_ghz = frequency_hz[1:] * 1e-9
    psd_db_hz = 10.0 * np.log10(np.maximum(mean_psd[1:], intensity_psd_floor))
    return frequency_ghz, psd_db_hz


def downsample_time_series(t, values, max_points=max_plot_points):
    t = np.asarray(t)
    values = np.asarray(values)
    if t.size <= max_points:
        return t, values
    idx = np.linspace(0, t.size - 1, int(max_points)).astype(int)
    return t[idx], values[..., idx]


def axis_centers_to_edges(axis):
    """Convert monotone center coordinates to pcolormesh edge coordinates."""
    axis = np.asarray(axis, dtype=float)
    if axis.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if axis.size == 1:
        half_width = 0.5 * abs(axis[0]) if axis[0] != 0.0 else 0.5
        return np.array([axis[0] - half_width, axis[0] + half_width], dtype=float)
    edges = np.empty(axis.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return edges


def analyze_phase_variance_summary(t, y):
    """Return single-laser phase-diffusion results for one fixed-kappa run."""
    n_cases = y.shape[0]
    S = np.maximum(y[:, 0:1, :], 0.0)
    phi = y[:, 1:2, :]

    analysis_start = analysis_start_index_for_record(
        t,
        already_steady=long_run_history_is_steady_state,
    )
    if analysis_start >= t.size - 3:
        raise ValueError("No stationary analysis window remains.")

    t_analysis = t[analysis_start:]
    S_analysis = S[:, 0, analysis_start:]
    phi_analysis = phi[:, 0, analysis_start:]

    phase_time, feedback_phase = feedback_phase_trace(t_analysis, phi_analysis)
    feedback_phase_abs = np.abs(np.angle(np.nanmean(np.exp(1j * feedback_phase), axis=0)))
    frequency_time, frequency_mean_ghz, frequency_variance_ghz2 = instantaneous_frequency_trace(
        t_analysis,
        phi_analysis,
    )
    psd_frequency_ghz, rin_psd_db_hz = relative_intensity_psd(
        t_analysis,
        S_analysis,
    )
    optical_frequency_ghz, optical_spectrum_db = optical_field_spectrum(
        t_analysis,
        S_analysis,
        phi_analysis,
    )
    time_series = single_laser_time_series_summary(
        t_analysis,
        S_analysis,
        phi_analysis,
    )

    linewidth_result = phase_diffusion_linewidth(
        t_analysis,
        phi_analysis,
        lag_range_ns=fit_lag_ns,
        max_lag_ns=plot_lag_max_ns,
        n_lags=n_lag_points,
        maximum_fit_fraction=maximum_fit_lag_fraction,
    )

    summary = {
        "lag_times": linewidth_result["lag_times"].astype(np.float32, copy=False),
        "phase_variance": linewidth_result["phase_variance"].astype(np.float32, copy=False),
        "phase_variance_sem": linewidth_result["phase_variance_sem"].astype(np.float32, copy=False),
        "fit_time": linewidth_result["fit_time"].astype(np.float32, copy=False),
        "fit_line": linewidth_result["fit_line"].astype(np.float32, copy=False),
        "feedback_phase_time": (phase_time - phase_time[0]).astype(np.float32, copy=False),
        "feedback_phase_abs": feedback_phase_abs.astype(np.float32, copy=False),
        "frequency_time": (frequency_time - frequency_time[0]).astype(np.float32, copy=False),
        "frequency_mean_ghz": frequency_mean_ghz.astype(np.float32, copy=False),
        "frequency_variance_ghz2": frequency_variance_ghz2.astype(np.float32, copy=False),
        "optical_frequency_ghz": optical_frequency_ghz.astype(np.float32, copy=False),
        "optical_spectrum_db": optical_spectrum_db.astype(np.float32, copy=False),
        "psd_frequency_ghz": psd_frequency_ghz.astype(np.float32, copy=False),
        "rin_psd_db_hz": rin_psd_db_hz.astype(np.float32, copy=False),
        "time_series_time": time_series["time"].astype(np.float32, copy=False),
        "time_series_intensity_mean": time_series["intensity_mean"].astype(np.float32, copy=False),
        "time_series_intensity_std": time_series["intensity_std"].astype(np.float32, copy=False),
        "time_series_frequency_mean_ghz": time_series["frequency_mean_ghz"].astype(np.float32, copy=False),
        "time_series_frequency_std_ghz": time_series["frequency_std_ghz"].astype(np.float32, copy=False),
        "time_series_feedback_phase_time": time_series["feedback_phase_time"].astype(np.float32, copy=False),
        "time_series_feedback_phase_abs_mean": time_series["feedback_phase_abs_mean"].astype(np.float32, copy=False),
        "time_series_feedback_phase_abs_std": time_series["feedback_phase_abs_std"].astype(np.float32, copy=False),
        "linewidth_hz": float(linewidth_result["linewidth_hz"]),
        "linewidth_median_hz": float(linewidth_result["linewidth_median_hz"]),
        "linewidth_std_hz": float(linewidth_result["linewidth_std_hz"]),
        "r_squared": float(linewidth_result["r_squared"]),
        "n_cases": int(n_cases),
    }
    del S, phi, S_analysis, phi_analysis, linewidth_result
    return summary


def plot_phase_variance_fit_result(result, kappa_value, figure_path):
    lag_ns = np.asarray(result["lag_times"]) * 1e9
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=figure_dpi)

    phase_variance = np.asarray(result["phase_variance"])
    phase_variance_sem = np.asarray(result["phase_variance_sem"])
    fit_lag_ns_used = np.asarray(result["fit_time"]) * 1e9
    fit_line = np.asarray(result["fit_line"])

    ax.plot(
        lag_ns,
        phase_variance,
        "o-",
        markersize=3,
        linewidth=1,
        label=PHASE_VARIANCE_SHORT_LABEL,
    )
    ax.fill_between(
        lag_ns,
        phase_variance - phase_variance_sem,
        phase_variance + phase_variance_sem,
        color="tab:blue",
        alpha=0.18,
        linewidth=0,
        label=r"Mean $\pm$ SEM",
    )
    if fit_lag_ns_used.size:
        ax.axvspan(
            fit_lag_ns_used[0],
            fit_lag_ns_used[-1],
            color="gray",
            alpha=0.18,
            label="Fit window",
        )
        ax.plot(
            fit_lag_ns_used,
            fit_line,
            color="red",
            linewidth=2.4,
            label="Linear fit",
        )

    linewidth_mean_mhz = float(result["linewidth_hz"]) * 1e-6
    linewidth_median_mhz = float(result["linewidth_median_hz"]) * 1e-6
    ax.set_title(
        rf"$\kappa$: {kappa_value * 1e-9:.3f} ns$^{{-1}}$; "
        rf"$\Delta\nu_\mathrm{{mean}}={linewidth_mean_mhz:.3f}$ MHz, "
        rf"$\Delta\nu_\mathrm{{median}}={linewidth_median_mhz:.3f}$ MHz, "
        rf"$R^2={float(result['r_squared']):.4f}$"
    )
    ax.set_xlabel(r"Phase-diffusion lag $\Delta t$ (ns)")
    ax.set_ylabel(PHASE_VARIANCE_LABEL)
    ax.set_xlim(*phase_variance_lag_axis_ns)
    if phase_diffusion_fit_ylim is not None:
        ax.set_ylim(*phase_diffusion_fit_ylim)
    if phase_diffusion_fit_yticks is not None:
        ax.set_yticks(phase_diffusion_fit_yticks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_time_series_frame(result, kappa_value, figure_path):
    """Save a simple-example-style time-series diagnostic frame."""
    time_us = np.asarray(result["time_series_time"], dtype=float) * 1e6
    intensity_mean = np.asarray(result["time_series_intensity_mean"], dtype=float)
    intensity_std = np.asarray(result["time_series_intensity_std"], dtype=float)
    frequency_mean = np.asarray(result["time_series_frequency_mean_ghz"], dtype=float)
    frequency_std = np.asarray(result["time_series_frequency_std_ghz"], dtype=float)
    feedback_time_us = (
        np.asarray(result["time_series_feedback_phase_time"], dtype=float) * 1e6
    )
    feedback_mean = np.asarray(
        result["time_series_feedback_phase_abs_mean"],
        dtype=float,
    )
    feedback_std = np.asarray(
        result["time_series_feedback_phase_abs_std"],
        dtype=float,
    )

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(14, 14),
        dpi=200,
        sharex=True,
    )

    if time_us.size:
        axs[0].plot(
            time_us,
            frequency_mean,
            color="tab:blue",
            linewidth=2.0,
            label=r"$\dot{\phi}/2\pi$",
        )
        axs[0].fill_between(
            time_us,
            frequency_mean - frequency_std,
            frequency_mean + frequency_std,
            color="tab:blue",
            alpha=0.3,
            linewidth=0,
            label=r"mean $\pm$ std",
        )
    axs[0].set_xlabel(r"Time ($\mu$s)", fontsize=22)
    axs[0].set_ylabel(r"$\dot{\phi}/2\pi$ (GHz)", fontsize=22)
    axs[0].set_ylim(-15.0, 15.0)
    axs[0].legend(loc="upper left", fontsize=14)
    axs[0].grid(True, alpha=0.2)
    axs[0].tick_params(axis="both", which="major", labelsize=18)
    axs[0].set_title(
        rf"Single laser delayed feedback, $\kappa={kappa_value * 1e-9:.3f}$ ns$^{{-1}}$",
        fontsize=24,
        pad=20,
    )

    if feedback_time_us.size:
        phase_lower = np.clip(feedback_mean - feedback_std, 0.0, np.pi)
        phase_upper = np.clip(feedback_mean + feedback_std, 0.0, np.pi)
        axs[1].plot(
            feedback_time_us,
            feedback_mean,
            color="tab:orange",
            linewidth=2.0,
            label=r"$|\phi(t-\tau)-\phi(t)-\phi_p|$",
        )
        axs[1].fill_between(
            feedback_time_us,
            phase_lower,
            phase_upper,
            color="tab:orange",
            alpha=0.3,
            linewidth=0,
            label=r"mean $\pm$ std",
        )
    axs[1].set_xlabel(r"Time ($\mu$s)", fontsize=22)
    axs[1].set_ylim(0.0, np.pi)
    axs[1].set_yticks([0.0, 0.5 * np.pi, np.pi])
    axs[1].set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
    axs[1].set_ylabel("Feedback phase (rad)", fontsize=22)
    axs[1].legend(loc="upper left", fontsize=14)
    axs[1].grid(True, alpha=0.2)
    axs[1].tick_params(axis="both", which="major", labelsize=18)

    if time_us.size:
        intensity_lower = np.maximum(intensity_mean - intensity_std, 0.0)
        intensity_upper = np.maximum(intensity_mean + intensity_std, 0.0)
        axs[2].plot(
            time_us,
            intensity_mean,
            color="tab:green",
            linewidth=2.0,
            label=r"$S=|E|^2$",
        )
        axs[2].fill_between(
            time_us,
            intensity_lower,
            intensity_upper,
            color="tab:green",
            alpha=0.3,
            linewidth=0,
            label=r"mean $\pm$ std",
        )
    axs[2].set_xlabel(r"Time ($\mu$s)", fontsize=22)
    axs[2].set_ylabel("Photon state $S$", fontsize=22)
    axs[2].set_ylim(0.0, 10.0)
    axs[2].legend(loc="upper left", fontsize=14)
    axs[2].grid(True, alpha=0.2)
    axs[2].tick_params(axis="both", which="major", labelsize=18)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def plot_phase_variance_continuation_frame(
    figure_path,
    kappa_values,
    feedback_phase_map,
    feedback_phase_time,
    phase_variance_map,
    lag_times,
    linewidth_mean_hz,
    linewidth_median_hz,
    linewidth_r_squared,
    henry_branches,
    completed_kappa_indices=None,
):
    if phase_variance_map is None or lag_times is None:
        return

    kappa_axis = np.asarray(kappa_values, dtype=float) * 1e-9
    lag_ns = np.asarray(lag_times, dtype=float) * 1e9
    lag_axis_min_ns, lag_axis_max_ns = phase_variance_lag_axis_ns
    font_size = int(continuation_frame_font_size)
    title_pad = float(continuation_frame_title_pad)

    completed_mask = np.zeros(kappa_axis.size, dtype=bool)
    if completed_kappa_indices is None:
        completed_mask[:] = True
    else:
        completed_indices = np.asarray(sorted(completed_kappa_indices), dtype=int)
        completed_indices = completed_indices[
            (completed_indices >= 0) & (completed_indices < completed_mask.size)
        ]
        completed_mask[completed_indices] = True

    fig = plt.figure(figsize=(18, 10), dpi=continuation_frame_dpi)
    width_ratios = [1] * 30
    width_ratios[8] = 0.25
    gs = fig.add_gridspec(
        20,
        30,
        height_ratios=[1] * 20,
        width_ratios=width_ratios,
        hspace=0.3,
    )

    ax0 = fig.add_subplot(gs[1:8, 0:-3])
    if feedback_phase_map is not None and feedback_phase_time is not None:
        phase_time_seconds = np.asarray(feedback_phase_time, dtype=float)
        phase_plot_map = np.asarray(feedback_phase_map)
        tail_window_applied = False
        if (
            phase_plot_tail_window_us is not None
            and phase_time_seconds.size > 1
            and (phase_time_seconds[-1] - phase_time_seconds[0])
            > float(phase_plot_tail_window_us) * 1e-6
        ):
            phase_plot_start = phase_time_seconds[-1] - float(phase_plot_tail_window_us) * 1e-6
            phase_plot_mask = phase_time_seconds >= phase_plot_start
            phase_time_seconds = phase_time_seconds[phase_plot_mask]
            phase_plot_map = phase_plot_map[:, phase_plot_mask]
            tail_window_applied = True

        # The feedback-phase trace starts one delay after the simulation starts,
        # and the integrator grid stops one sample before the requested Tmax.
        # For the progress figure, display the selected tail as the nominal
        # analysis window (e.g. 3.000--5.000 us) instead of exposing those tiny
        # implementation offsets (e.g. 2.999--4.999 us).
        phase_axis_seconds = phase_time_seconds
        if tail_window_applied:
            tail_window_seconds = float(phase_plot_tail_window_us) * 1e-6
            nominal_end_seconds = long_steps * dt
            nominal_start_seconds = max(0.0, nominal_end_seconds - tail_window_seconds)
            phase_axis_seconds = np.linspace(
                nominal_start_seconds,
                nominal_end_seconds,
                phase_time_seconds.size,
            )

        if phase_axis_seconds[-1] >= 1e-6:
            phase_time = phase_axis_seconds * 1e6
            phase_time_label = r"Analysis time ($\mu$s)"
        else:
            phase_time = phase_axis_seconds * 1e9
            phase_time_label = r"Analysis time (ns)"

        im0 = ax0.imshow(
            np.ma.masked_invalid(phase_plot_map),
            aspect="auto",
            extent=[phase_time[0], phase_time[-1], kappa_axis[0], kappa_axis[-1]],
            origin="lower",
            cmap="jet_r",
            vmin=0.0,
            vmax=np.pi,
            rasterized=True,
        )
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.02)
        cbar0.set_label(
            r"$|\phi(t-\tau)-\phi(t)-\phi_p|_\mathrm{circ}$ (rad)",
            fontsize=font_size,
            labelpad=0,
        )
        cbar0.set_ticks([0.0, 0.5 * np.pi, np.pi])
        cbar0.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
        cbar0.ax.tick_params(labelsize=font_size)
        ax0.set_xlim(phase_time[0], phase_time[-1])
        ax0.set_xticks(np.linspace(phase_time[0], phase_time[-1], 6))
        ax0.set_xlabel(phase_time_label, fontsize=font_size)
    ax0.set_ylabel(r"$\kappa~(\mathrm{ns}^{-1})$", fontsize=font_size)
    ax0.set_ylim(kappa_axis[0], kappa_axis[-1])
    ax0.set_yticks(np.linspace(kappa_axis[0], kappa_axis[-1], 6))
    ax0.set_title(
        rf"Single laser delayed feedback, $\phi_p={phi_p / np.pi:+.2f}\pi$, "
        rf"$\tau={tau * 1e9:.2f}$ ns",
        fontsize=font_size,
        pad=title_pad,
    )
    ax0.tick_params(axis="both", labelsize=font_size)

    ax1 = fig.add_subplot(gs[11:, 1:8])
    variance_cmap = plt.get_cmap("jet").copy()
    variance_cmap.set_bad(color="white")
    variance_values = np.asarray(phase_variance_map, dtype=float)
    finite_positive_variance = variance_values[
        np.isfinite(variance_values) & (variance_values > 0.0)
    ]
    variance_norm = None
    colorbar_tick_values = None
    if phase_diffusion_colorbar_ticks is not None:
        colorbar_tick_values = np.unique(np.asarray(phase_diffusion_colorbar_ticks, dtype=float))
        colorbar_tick_values = colorbar_tick_values[
            np.isfinite(colorbar_tick_values) & (colorbar_tick_values > 0.0)
        ]
        if colorbar_tick_values.size == 0:
            colorbar_tick_values = None
    if phase_variance_colorbar_log_scale and finite_positive_variance.size:
        if colorbar_tick_values is not None and colorbar_tick_values.size >= 2:
            variance_vmin = float(colorbar_tick_values[0])
            variance_vmax = float(colorbar_tick_values[-1])
        elif phase_variance_log_vmin_rad2 is None:
            variance_vmin = np.nanpercentile(
                finite_positive_variance,
                phase_variance_log_vmin_percentile,
            )
            variance_vmax = (
                np.nanpercentile(finite_positive_variance, phase_variance_log_vmax_percentile)
                if phase_variance_log_vmax_rad2 is None
                else float(phase_variance_log_vmax_rad2)
            )
        else:
            variance_vmin = float(phase_variance_log_vmin_rad2)
            variance_vmax = (
                np.nanpercentile(finite_positive_variance, phase_variance_log_vmax_percentile)
                if phase_variance_log_vmax_rad2 is None
                else float(phase_variance_log_vmax_rad2)
            )
        variance_vmin = max(variance_vmin, np.nanmin(finite_positive_variance), np.finfo(float).tiny)
        variance_vmax = max(variance_vmax, variance_vmin * 10.0)
        variance_norm = LogNorm(vmin=variance_vmin, vmax=variance_vmax)
        variance_plot = np.ma.masked_where(
            (~np.isfinite(variance_values)) | (variance_values <= 0.0),
            variance_values,
        )
    else:
        variance_plot = np.ma.masked_invalid(variance_values)

    im1 = ax1.imshow(
        variance_plot,
        aspect="auto",
        extent=[lag_ns[0], lag_ns[-1], kappa_axis[0], kappa_axis[-1]],
        origin="lower",
        cmap=variance_cmap,
        norm=variance_norm,
        vmin=None if variance_norm is not None else 0.0,
        rasterized=True,
    )
    cbar1 = fig.colorbar(im1, ax=ax1, pad=0.045)
    if variance_norm is not None and colorbar_tick_values is not None:
        colorbar_ticks = [
            float(tick)
            for tick in colorbar_tick_values
            if variance_norm.vmin <= float(tick) <= variance_norm.vmax
        ]
        if colorbar_ticks:
            cbar1.set_ticks(colorbar_ticks)
            cbar1.set_ticklabels([f"{tick:g}" for tick in colorbar_ticks])
            cbar1.ax.yaxis.set_minor_locator(NullLocator())
            cbar1.ax.yaxis.set_minor_formatter(NullFormatter())
            cbar1.minorticks_off()
    cbar1.set_label(PHASE_VARIANCE_SHORT_LABEL, fontsize=font_size, labelpad=4)
    cbar1.ax.tick_params(labelsize=font_size)
    ax1.set_xlabel(r"Phase-diffusion lag $\Delta t$ (ns)", fontsize=font_size, labelpad=10)
    ax1.set_ylabel(r"$\kappa~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
    ax1.set_title(PHASE_VARIANCE_LABEL, fontsize=font_size, pad=title_pad)
    ax1.set_yticks(np.linspace(kappa_axis[0], kappa_axis[-1], 6))
    ax1.set_xticks(np.linspace(lag_axis_min_ns, lag_axis_max_ns, 3))
    ax1.set_xlim(lag_axis_min_ns, lag_axis_max_ns)
    ax1.set_ylim(kappa_axis[0], kappa_axis[-1])
    ax1.tick_params(axis="both", labelsize=font_size)

    fit_min, fit_max = fit_lag_ns
    if lag_ns[0] <= fit_min <= lag_ns[-1]:
        ax1.axvline(fit_min, color="black", linestyle="--", linewidth=1.6, alpha=0.55)
    if lag_ns[0] <= fit_max <= lag_ns[-1]:
        ax1.axvline(fit_max, color="black", linestyle="--", linewidth=1.6, alpha=0.55)

    ax2 = fig.add_subplot(gs[11:, 15:29])
    if henry_branches is not None:
        branch_kappa_ns = np.asarray(henry_branches.get("kappa_values", []), dtype=float) * 1e-9
        branch_mhz = np.asarray(henry_branches.get("linewidth_hz", []), dtype=float) * 1e-6
        branch_positive = np.asarray(henry_branches.get("positive_slope", []), dtype=bool)
        branch_mask = np.isfinite(branch_kappa_ns) & np.isfinite(branch_mhz) & (branch_mhz > 0.0)
        positive_branch = branch_mask & branch_positive
        other_branch = branch_mask & ~branch_positive
        if np.any(other_branch):
            ax2.scatter(
                branch_kappa_ns[other_branch],
                branch_mhz[other_branch],
                s=5,
                color="0.65",
                alpha=0.16,
                linewidths=0,
                rasterized=True,
                label="other ECM roots",
            )
        if np.any(positive_branch):
            ax2.scatter(
                branch_kappa_ns[positive_branch],
                branch_mhz[positive_branch],
                s=10,
                color="tab:red",
                alpha=0.5,
                linewidths=0,
                rasterized=True,
                label="Henry ECM branches",
            )

    mean_mhz = np.asarray(linewidth_mean_hz, dtype=float) * 1e-6
    median_mhz = np.asarray(linewidth_median_hz, dtype=float) * 1e-6
    valid_mean = completed_mask & np.isfinite(mean_mhz) & (mean_mhz > 0.0)
    valid_median = completed_mask & np.isfinite(median_mhz) & (median_mhz > 0.0)
    if np.any(valid_mean):
        ax2.plot(
            kappa_axis[valid_mean],
            mean_mhz[valid_mean],
            color="black",
            linewidth=2.6,
            marker="o",
            markersize=3.5,
            label=r"mean phase-variance fit",
        )
    if np.any(valid_median):
        ax2.plot(
            kappa_axis[valid_median],
            median_mhz[valid_median],
            color="tab:blue",
            linewidth=1.8,
            marker="s",
            markersize=3.0,
            alpha=0.85,
            label=r"median realization",
        )
    r_values = np.asarray(linewidth_r_squared, dtype=float)[completed_mask]
    r_values = r_values[np.isfinite(r_values)]
    if r_values.size:
        ax2.plot([], [], linestyle="none", marker="", label=rf"median $R^2={np.median(r_values):.3f}$")

    ax2.set_xlabel(r"$\kappa~(\mathrm{ns}^{-1})$", fontsize=font_size + 4, labelpad=10)
    ax2.set_ylabel(r"linewidth estimate (MHz)", fontsize=font_size + 4, labelpad=4)
    ax2.set_title(r"Single-laser phase-diffusion linewidth", fontsize=font_size + 4, pad=title_pad)
    ax2.set_xlim(kappa_axis[0], kappa_axis[-1])
    ax2.set_yscale("log")
    ax2.set_ylim(*linewidth_plot_ylim_mhz)
    ax2.set_yticks(linewidth_plot_yticks_mhz)
    ax2.minorticks_on()
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.grid(True, axis="y", which="major", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.grid(True, axis="y", which="minor", linestyle="--", linewidth=0.65, alpha=0.38)
    ax2.grid(True, axis="x", which="major", linestyle="--", linewidth=0.6, alpha=0.25)
    ax2.tick_params(axis="y", which="minor", length=3.5)
    ax2.set_xticks(np.linspace(kappa_axis[0], kappa_axis[-1], 6))
    ax2.tick_params(axis="both", labelsize=font_size + 4)
    ax2.legend(fontsize=continuation_linewidth_legend_font_size, loc="upper right")

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_intensity_psd_continuation_frame(
    figure_path,
    kappa_values,
    optical_spectrum_map_db,
    optical_frequency_ghz,
    rin_psd_map_db,
    psd_frequency_ghz,
    frequency_map,
    frequency_time,
    frequency_variance_map=None,
    completed_kappa_indices=None,
):
    """Progress image of optical spectrum, RIN, and frequency fluctuation."""
    if (
        optical_spectrum_map_db is None
        or optical_frequency_ghz is None
        or rin_psd_map_db is None
        or psd_frequency_ghz is None
    ):
        return

    kappa_axis = np.asarray(kappa_values, dtype=float) * 1e-9
    optical_frequency_ghz = np.asarray(optical_frequency_ghz, dtype=float)
    optical_plot_map = np.asarray(optical_spectrum_map_db, dtype=float)
    psd_frequency_ghz = np.asarray(psd_frequency_ghz, dtype=float)
    psd_plot_map = np.asarray(rin_psd_map_db, dtype=float)
    if (
        optical_frequency_ghz.size < 2
        or optical_plot_map.size == 0
        or psd_frequency_ghz.size < 2
        or psd_plot_map.size == 0
    ):
        return

    frequency_time_seconds = np.asarray(frequency_time, dtype=float)
    frequency_plot_map = (
        None
        if frequency_map is None
        else np.asarray(frequency_map, dtype=float)
    )
    frequency_variance_plot_map = (
        None
        if frequency_variance_map is None
        else np.asarray(frequency_variance_map, dtype=float)
    )
    if (
        frequency_plot_map is not None
        and frequency_variance_plot_map is not None
        and frequency_variance_plot_map.shape != frequency_plot_map.shape
    ):
        frequency_variance_plot_map = None

    completed_mask = np.zeros(kappa_axis.size, dtype=bool)
    if completed_kappa_indices is None:
        completed_mask[:] = True
    else:
        completed_indices = np.asarray(sorted(completed_kappa_indices), dtype=int)
        completed_indices = completed_indices[
            (completed_indices >= 0) & (completed_indices < completed_mask.size)
        ]
        completed_mask[completed_indices] = True

    optical_completed = optical_plot_map[
        completed_mask,
        :,
    ]
    optical_values = optical_completed[np.isfinite(optical_completed)]
    optical_vmin = optical_vmax = None
    if optical_values.size:
        lower, upper = optical_spectrum_colorbar_percentiles
        optical_vmin = float(np.nanpercentile(optical_values, lower))
        optical_vmax = float(np.nanpercentile(optical_values, upper))
        optical_vmin = max(optical_vmin, float(optical_spectrum_colorbar_floor_db))
        optical_vmax = min(optical_vmax, 0.0)
        if (
            not np.isfinite(optical_vmin)
            or not np.isfinite(optical_vmax)
            or optical_vmin >= optical_vmax
        ):
            optical_vmin = float(optical_spectrum_colorbar_floor_db)
            optical_vmax = 0.0

    psd_completed = psd_plot_map[
        completed_mask,
        :,
    ]
    psd_values = psd_completed[np.isfinite(psd_completed)]
    psd_vmin = psd_vmax = None
    if psd_values.size:
        lower, upper = intensity_psd_colorbar_percentiles
        psd_vmin = float(np.nanpercentile(psd_values, lower))
        psd_vmax = float(np.nanpercentile(psd_values, upper))
        if not np.isfinite(psd_vmin) or not np.isfinite(psd_vmax) or psd_vmin == psd_vmax:
            psd_vmin = psd_vmax = None

    font_size = int(continuation_frame_font_size)
    title_pad = float(continuation_frame_title_pad)
    fig = plt.figure(
        figsize=(20, 5.1),
        dpi=continuation_frame_dpi,
    )
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        left=0.055,
        right=0.985,
        bottom=0.20,
        top=0.84,
        wspace=0.28,
    )
    optical_grid = outer[0].subgridspec(
        1,
        2,
        width_ratios=[1.0, 0.035],
        wspace=0.035,
    )
    rin_grid = outer[1].subgridspec(
        1,
        2,
        width_ratios=[1.0, 0.035],
        wspace=0.035,
    )
    fluctuation_grid = outer[2].subgridspec(1, 1)
    ax_opt = fig.add_subplot(optical_grid[0, 0])
    cax_opt = fig.add_subplot(optical_grid[0, 1])
    ax_rin = fig.add_subplot(rin_grid[0, 0], sharey=ax_opt)
    cax_rin = fig.add_subplot(rin_grid[0, 1])
    ax_var = fig.add_subplot(fluctuation_grid[0, 0], sharey=ax_opt)

    optical_edges = axis_centers_to_edges(optical_frequency_ghz)
    kappa_edges = axis_centers_to_edges(kappa_axis)
    im_opt = ax_opt.pcolormesh(
        optical_edges,
        kappa_edges,
        np.ma.masked_invalid(optical_plot_map),
        cmap="jet",
        shading="auto",
        vmin=optical_vmin,
        vmax=optical_vmax,
        rasterized=True,
    )
    cbar_opt = fig.colorbar(im_opt, cax=cax_opt)
    cbar_opt.ax.yaxis.set_ticks_position("right")
    cbar_opt.ax.yaxis.set_label_position("right")
    cbar_opt.ax.tick_params(
        labelsize=font_size - 2,
        pad=2,
        labelright=True,
        labelleft=False,
    )
    cbar_opt.ax.set_ylabel(
        r"spectrum (dB)",
        fontsize=font_size - 3,
        labelpad=8,
    )

    ax_opt.axvline(0.0, color="white", linewidth=0.8, alpha=0.65)
    ax_opt.set_xlim(optical_edges[0], optical_edges[-1])
    ax_opt.set_ylim(kappa_axis[0], kappa_axis[-1])
    ax_opt.set_yticks(np.linspace(kappa_axis[0], kappa_axis[-1], 6))
    ax_opt.set_xlabel(r"Frequency offset (GHz)", fontsize=font_size)
    ax_opt.set_ylabel(r"$\kappa~(\mathrm{ns}^{-1})$", fontsize=font_size)
    ax_opt.set_title("Optical spectrum", fontsize=font_size, pad=title_pad)
    ax_opt.tick_params(axis="both", labelsize=font_size)
    ax_opt.grid(True, which="major", alpha=0.18)

    frequency_edges = axis_centers_to_edges(psd_frequency_ghz)
    frequency_edges[0] = max(
        frequency_edges[0],
        0.5 * psd_frequency_ghz[0],
        np.finfo(float).tiny,
    )
    im_rin = ax_rin.pcolormesh(
        frequency_edges,
        kappa_edges,
        np.ma.masked_invalid(psd_plot_map),
        cmap="jet",
        shading="auto",
        vmin=psd_vmin,
        vmax=psd_vmax,
        rasterized=True,
    )
    cbar_rin = fig.colorbar(im_rin, cax=cax_rin)
    cbar_rin.ax.yaxis.set_ticks_position("right")
    cbar_rin.ax.yaxis.set_label_position("right")
    cbar_rin.ax.tick_params(
        labelsize=font_size - 2,
        pad=2,
        labelright=True,
        labelleft=False,
    )
    cbar_rin.ax.set_ylabel(
        r"RIN PSD (dB/Hz)",
        fontsize=font_size - 2,
        labelpad=8,
    )

    ax_rin.set_xscale("log")
    ax_rin.set_xlim(
        max(frequency_edges[0], np.finfo(float).tiny),
        frequency_edges[-1],
    )
    ax_rin.set_ylim(kappa_axis[0], kappa_axis[-1])
    ax_rin.set_xlabel(r"Offset frequency (GHz)", fontsize=font_size)
    ax_rin.set_title(
        rf"RIN, $\phi_p={phi_p / np.pi:+.2f}\pi$, "
        rf"$\tau={tau * 1e9:.2f}$ ns",
        fontsize=font_size,
        pad=title_pad,
    )
    ax_rin.tick_params(axis="both", labelsize=font_size)
    ax_rin.tick_params(axis="y", left=False, labelleft=False)
    ax_rin.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax_rin.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
    )
    ax_rin.xaxis.set_minor_formatter(NullFormatter())
    ax_rin.grid(True, which="major", alpha=0.25)
    ax_rin.grid(True, which="minor", linestyle="--", alpha=0.12)

    mean_frequency_variance = np.full(kappa_axis.size, np.nan, dtype=float)
    mean_frequency_variance_std = np.full_like(mean_frequency_variance, np.nan)
    noise_frequency_variance = np.full_like(mean_frequency_variance, np.nan)
    noise_frequency_variance_std = np.full_like(mean_frequency_variance, np.nan)
    if (
        frequency_plot_map is not None
        and frequency_time_seconds.size > 1
        and frequency_plot_map.size
    ):
        if (
            phase_plot_tail_window_us is not None
            and (frequency_time_seconds[-1] - frequency_time_seconds[0])
            > float(phase_plot_tail_window_us) * 1e-6
        ):
            frequency_plot_start = (
                frequency_time_seconds[-1] - float(phase_plot_tail_window_us) * 1e-6
            )
            frequency_plot_mask = frequency_time_seconds >= frequency_plot_start
            frequency_plot_map = frequency_plot_map[:, frequency_plot_mask]
            if frequency_variance_plot_map is not None:
                frequency_variance_plot_map = frequency_variance_plot_map[
                    :,
                    frequency_plot_mask,
                ]

        for row_index, mean_row in enumerate(frequency_plot_map):
            mean_row = np.asarray(mean_row, dtype=float)
            if frequency_variance_plot_map is None:
                variance_row = np.zeros_like(mean_row)
            else:
                variance_row = np.asarray(
                    frequency_variance_plot_map[row_index],
                    dtype=float,
                )
            finite_row = (
                np.isfinite(mean_row)
                & np.isfinite(variance_row)
                & (variance_row >= 0.0)
            )
            if not np.any(finite_row):
                continue
            mean_finite = mean_row[finite_row]
            variance_finite = variance_row[finite_row]
            mean_frequency = float(np.mean(mean_finite))
            mean_frequency_variance_trace = (mean_finite - mean_frequency) ** 2
            noise_frequency_variance_trace = variance_finite
            mean_frequency_variance[row_index] = float(
                np.mean(mean_frequency_variance_trace)
            )
            noise_frequency_variance[row_index] = float(
                np.mean(noise_frequency_variance_trace)
            )
            if mean_frequency_variance_trace.size > 1:
                mean_frequency_variance_std[row_index] = float(
                    np.std(mean_frequency_variance_trace, ddof=1)
                )
            else:
                mean_frequency_variance_std[row_index] = 0.0
            if noise_frequency_variance_trace.size > 1:
                noise_frequency_variance_std[row_index] = float(
                    np.std(noise_frequency_variance_trace, ddof=1)
                )
            else:
                noise_frequency_variance_std[row_index] = 0.0

    mean_lower = mean_frequency_variance - mean_frequency_variance_std
    mean_upper = mean_frequency_variance + mean_frequency_variance_std
    noise_lower = noise_frequency_variance - noise_frequency_variance_std
    noise_upper = noise_frequency_variance + noise_frequency_variance_std
    valid_mean_variance = (
        completed_mask
        & np.isfinite(mean_frequency_variance)
        & np.isfinite(mean_frequency_variance_std)
        & np.isfinite(mean_upper)
        & (mean_frequency_variance > 0.0)
        & (mean_upper > 0.0)
    )
    valid_noise_variance = (
        completed_mask
        & np.isfinite(noise_frequency_variance)
        & np.isfinite(noise_frequency_variance_std)
        & np.isfinite(noise_upper)
        & (noise_frequency_variance > 0.0)
        & (noise_upper > 0.0)
    )
    if np.any(valid_mean_variance) or np.any(valid_noise_variance):
        positive_bounds = np.concatenate(
            [
                mean_frequency_variance[valid_mean_variance],
                mean_upper[valid_mean_variance],
                mean_lower[
                    valid_mean_variance
                    & np.isfinite(mean_lower)
                    & (mean_lower > 0.0)
                ],
                noise_frequency_variance[valid_noise_variance],
                noise_upper[valid_noise_variance],
                noise_lower[
                    valid_noise_variance
                    & np.isfinite(noise_lower)
                    & (noise_lower > 0.0)
                ],
            ]
        )
        positive_bounds = positive_bounds[
            np.isfinite(positive_bounds) & (positive_bounds > 0.0)
        ]
        if positive_bounds.size:
            variance_xmin = 0.5 * float(np.min(positive_bounds))
            variance_xmax = 2.0 * float(np.max(positive_bounds))
            variance_xmin = max(variance_xmin, np.finfo(float).tiny)
            variance_xmax = max(variance_xmax, 10.0 * variance_xmin)

            if np.any(valid_mean_variance):
                mean_lower_plot = np.maximum(mean_lower, variance_xmin)
                ax_var.fill_betweenx(
                    kappa_axis[valid_mean_variance],
                    mean_lower_plot[valid_mean_variance],
                    mean_upper[valid_mean_variance],
                    color="tab:blue",
                    alpha=0.16,
                    linewidth=0,
                )
                ax_var.plot(
                    mean_frequency_variance[valid_mean_variance],
                    kappa_axis[valid_mean_variance],
                    color="tab:blue",
                    linewidth=2,
                    label=r"$\mathrm{var}_t[\langle f\rangle]$",
                )
            if np.any(valid_noise_variance):
                noise_lower_plot = np.maximum(noise_lower, variance_xmin)
                ax_var.fill_betweenx(
                    kappa_axis[valid_noise_variance],
                    noise_lower_plot[valid_noise_variance],
                    noise_upper[valid_noise_variance],
                    color="tab:orange",
                    alpha=0.18,
                    linewidth=0,
                )
                ax_var.plot(
                    noise_frequency_variance[valid_noise_variance],
                    kappa_axis[valid_noise_variance],
                    color="tab:orange",
                    linewidth=2,
                    label=r"$\langle\mathrm{var}_{\eta}[f]\rangle_t$",
                )
            ax_var.set_xscale("log")
            ax_var.set_xlim(variance_xmin, variance_xmax)
            ax_var.xaxis.set_minor_locator(
                LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
            )
            ax_var.xaxis.set_minor_formatter(NullFormatter())
            ax_var.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax_var.legend(
                fontsize=max(font_size - 8, 8),
                loc="best",
                framealpha=0.85,
                handlelength=1.2,
            )

    ax_var.set_title(
        r"Frequency" "\n" r"variance",
        fontsize=font_size - 2,
        pad=title_pad - 4,
    )
    ax_var.set_xlabel(
        r"variance (GHz$^2$)",
        fontsize=font_size - 1,
        labelpad=4,
    )
    ax_var.set_ylim(kappa_axis[0], kappa_axis[-1])
    ax_var.tick_params(axis="x", labelsize=font_size - 2, pad=2)
    ax_var.tick_params(axis="y", left=False, labelleft=False)
    ax_var.grid(True, which="major", alpha=0.35)
    ax_var.grid(True, which="minor", linestyle="--", alpha=0.18)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


# ---------------------------- Parallel workers ---------------------------


LONG_RUN_PROGRESS_QUEUE = None
VCSEL_INTEGRATE_SUPPORTS_PROGRESS_CALLBACK = (
    "progress_callback" in inspect.signature(VCSEL.integrate).parameters
)


def init_long_run_progress_worker(progress_queue):
    global LONG_RUN_PROGRESS_QUEUE
    LONG_RUN_PROGRESS_QUEUE = progress_queue


def make_worker_progress_callback(job):
    progress_queue = LONG_RUN_PROGRESS_QUEUE
    progress_id = job.get("progress_id", None)
    if progress_queue is None or progress_id is None:
        return None, lambda: None

    update_steps = int(max(1, job.get("progress_update_steps", 200)))
    pending = [0]

    def progress_callback(n_steps):
        pending[0] += int(n_steps)
        if pending[0] >= update_steps:
            progress_queue.put((progress_id, pending[0]))
            pending[0] = 0

    def flush_progress():
        if pending[0] > 0:
            progress_queue.put((progress_id, pending[0]))
            pending[0] = 0

    return progress_callback, flush_progress


def integrate_with_optional_progress_callback(vcsel, *args, progress_callback=None, **kwargs):
    if VCSEL_INTEGRATE_SUPPORTS_PROGRESS_CALLBACK:
        kwargs["progress_callback"] = progress_callback
    return vcsel.integrate(*args, **kwargs)


def _run_fixed_kappa_linewidth_chunk_worker(jobs):
    jobs = list(jobs)
    if not jobs:
        return []

    histories = []
    kappa_blocks = []
    case_counts = []
    for job in jobs:
        history = np.asarray(job["history"])
        n_cases_job = int(history.shape[0])
        histories.append(history)
        case_counts.append(n_cases_job)
        kappa_matrix = np.asarray([[[float(job["kappa"])]]], dtype=np.float32)[0]
        kappa_blocks.append(np.repeat(kappa_matrix[None, :, :], n_cases_job, axis=0))

    history_all = np.concatenate(histories, axis=0)
    kappa_all = np.concatenate(kappa_blocks, axis=0).astype(np.float32, copy=False)
    vcsel, nd = make_vcsel_from_kappa_matrix(kappa_all, Tmax_long, analysis_save_every)
    nd["kappa_case_dependent"] = True
    progress_callback, flush_progress = make_worker_progress_callback(jobs[0])

    try:
        t_long, y_long, _ = integrate_with_optional_progress_callback(
            vcsel,
            history_all,
            nd=nd,
            progress=False,
            max_iter=5,
            smooth_freqs=False,
            return_final_history=False,
            message=f"fixed-kappa single-laser phase-diffusion chunk {int(jobs[0]['k'])}-{int(jobs[-1]['k'])}",
            progress_callback=progress_callback,
        )
        flush_progress()
    except Exception:
        flush_progress()
        raise

    results = []
    offset = 0
    for job, n_cases_job in zip(jobs, case_counts):
        y_job = y_long[offset:offset + n_cases_job]
        offset += n_cases_job
        result = analyze_phase_variance_summary(t_long, y_job)
        result["k"] = int(job["k"])
        result["kappa"] = float(job["kappa"])
        results.append(result)
        del y_job, result

    del histories, kappa_blocks, history_all, kappa_all, y_long, vcsel, nd
    gc.collect()
    return results


def run_fixed_kappa_linewidth_chunk_worker(jobs):
    jobs = list(jobs)
    try:
        return _run_fixed_kappa_linewidth_chunk_worker(jobs)
    except BaseException:
        worker_arrays_dir.mkdir(parents=True, exist_ok=True)
        label = f"{int(jobs[0]['k']):03d}_{int(jobs[-1]['k']):03d}" if jobs else "empty"
        error_path = worker_arrays_dir / f"worker_error_chunk_{label}.txt"
        error_path.write_text(traceback.format_exc())
        raise


def make_linewidth_job(kappa_index, kappa, history):
    return {
        "k": int(kappa_index),
        "kappa": float(kappa),
        "history": history.copy(),
    }


def main():
    rc("text", usetex=True)
    rc("font", family="serif")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    fit_frames_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    worker_arrays_dir.mkdir(parents=True, exist_ok=True)
    if write_time_series_frames:
        time_series_frames_dir.mkdir(parents=True, exist_ok=True)
    validate_analysis_window()

    kappa_values = np.linspace(kappa_initial, kappa_final, int(max(1, n_kappa_steps)))
    dtype = np.float32

    lag_times = None
    phase_variance_map = None
    phase_variance_sem_map = None
    feedback_phase_time = None
    feedback_phase_map = None
    frequency_time = None
    frequency_map = None
    frequency_variance_map = None
    optical_frequency_ghz = None
    optical_spectrum_map_db = None
    psd_frequency_ghz = None
    rin_psd_map_db = None
    fit_time_map = None
    fit_line_map = None
    linewidth_mean_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    linewidth_median_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    linewidth_std_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    linewidth_r_squared = np.full(kappa_values.size, np.nan, dtype=dtype)
    completed_kappa_indices = set()

    np.random.seed(random_seed)
    vcsel0, nd0 = make_vcsel_ramp(
        kappa_values[0],
        kappa_values[0],
        Tmax_continuation,
        time_array_continuation,
        continuation_save_every,
    )
    delta_nu_0_hz = free_running_henry_linewidth_hz(nd0)
    henry_branches = henry_ecm_branch_points(kappa_values, delta_nu_0_hz)
    history, _, _, _ = vcsel0.generate_history(nd0, shape="FR", n_cases=n_noise_iterations)
    if average_continuation_history_across_noise:
        history = average_history_across_cases(history)

    use_process_pool = int(max(1, long_run_jobs)) > 1
    if use_process_pool:
        mp_context = mp.get_context("fork")
        progress_queue = mp_context.Queue() if show_long_run_worker_progress else None
        executor = ProcessPoolExecutor(
            max_workers=int(max(1, long_run_jobs)),
            mp_context=mp_context,
            initializer=init_long_run_progress_worker if show_long_run_worker_progress else None,
            initargs=(progress_queue,) if show_long_run_worker_progress else (),
        )
    else:
        progress_queue = queue_module.Queue() if show_long_run_worker_progress else None
        executor = None

    pending_futures = []
    future_progress_id = {}
    progress_bars = {}
    current_chunk = []
    progress_counter = 0
    # Keep one wave of long chunks executing and one equally sized wave
    # queued behind it. For example, long_run_jobs == 3 permits three active
    # chunks plus three waiting chunks, rather than allowing the sequential
    # continuation to enqueue the entire kappa sweep.
    active_long_chunk_slots = int(max(1, long_run_jobs))
    queued_long_chunk_slots = int(max(1, long_run_jobs))
    max_pending_long_chunks = active_long_chunk_slots + queued_long_chunk_slots

    def process_chunk_results(results, progress_id=None):
        nonlocal lag_times, phase_variance_map, phase_variance_sem_map
        nonlocal feedback_phase_time, feedback_phase_map
        nonlocal frequency_time, frequency_map, frequency_variance_map
        nonlocal optical_frequency_ghz, optical_spectrum_map_db
        nonlocal psd_frequency_ghz, rin_psd_map_db
        nonlocal fit_time_map, fit_line_map

        if progress_id in progress_bars:
            progress_bars[progress_id].n = progress_bars[progress_id].total
            progress_bars[progress_id].refresh()
            progress_bars[progress_id].close()
            del progress_bars[progress_id]

        for result in sorted(results, key=lambda item: item["k"]):
            kappa_index = int(result["k"])
            kappa_value = float(result["kappa"])
            completed_kappa_indices.add(kappa_index)

            if lag_times is None:
                lag_times = result["lag_times"].astype(np.float32, copy=False)
                phase_variance_map = np.full(
                    (kappa_values.size, lag_times.size),
                    np.nan,
                    dtype=dtype,
                )
                phase_variance_sem_map = np.full_like(phase_variance_map, np.nan)
            if result["lag_times"].shape != lag_times.shape or not np.allclose(result["lag_times"], lag_times):
                raise ValueError("Phase-diffusion lag grid changed between kappa steps.")
            phase_variance_map[kappa_index] = result["phase_variance"].astype(dtype, copy=False)
            phase_variance_sem_map[kappa_index] = result["phase_variance_sem"].astype(dtype, copy=False)

            if feedback_phase_time is None:
                phase_time_ds, phase_abs_ds = downsample_time_series(
                    result["feedback_phase_time"],
                    result["feedback_phase_abs"],
                    max_points=max_plot_points,
                )
                feedback_phase_time = phase_time_ds.astype(np.float32, copy=False)
                feedback_phase_map = np.full(
                    (kappa_values.size, feedback_phase_time.size),
                    np.nan,
                    dtype=dtype,
                )
            else:
                phase_time_ds, phase_abs_ds = downsample_time_series(
                    result["feedback_phase_time"],
                    result["feedback_phase_abs"],
                    max_points=max_plot_points,
                )
                if phase_abs_ds.shape[0] != feedback_phase_time.shape[0]:
                    phase_abs_ds = np.interp(feedback_phase_time, phase_time_ds, phase_abs_ds)
            feedback_phase_map[kappa_index] = phase_abs_ds.astype(dtype, copy=False)

            if frequency_time is None:
                frequency_time_ds, frequency_mean_ds = downsample_time_series(
                    result["frequency_time"],
                    result["frequency_mean_ghz"],
                    max_points=max_plot_points,
                )
                _, frequency_variance_ds = downsample_time_series(
                    result["frequency_time"],
                    result["frequency_variance_ghz2"],
                    max_points=max_plot_points,
                )
                frequency_time = frequency_time_ds.astype(np.float32, copy=False)
                frequency_map = np.full(
                    (kappa_values.size, frequency_time.size),
                    np.nan,
                    dtype=dtype,
                )
                frequency_variance_map = np.full_like(frequency_map, np.nan)
            else:
                frequency_time_ds, frequency_mean_ds = downsample_time_series(
                    result["frequency_time"],
                    result["frequency_mean_ghz"],
                    max_points=max_plot_points,
                )
                frequency_variance_time_ds, frequency_variance_ds = downsample_time_series(
                    result["frequency_time"],
                    result["frequency_variance_ghz2"],
                    max_points=max_plot_points,
                )
                if frequency_mean_ds.shape[0] != frequency_time.shape[0]:
                    frequency_mean_ds = np.interp(
                        frequency_time,
                        frequency_time_ds,
                        frequency_mean_ds,
                    )
                if frequency_variance_ds.shape[0] != frequency_time.shape[0]:
                    frequency_variance_ds = np.interp(
                        frequency_time,
                        frequency_variance_time_ds,
                        frequency_variance_ds,
                    )
            frequency_map[kappa_index] = frequency_mean_ds.astype(dtype, copy=False)
            frequency_variance_map[kappa_index] = frequency_variance_ds.astype(
                dtype,
                copy=False,
            )

            if optical_frequency_ghz is None:
                optical_frequency_ds, optical_spectrum_ds = downsample_time_series(
                    result["optical_frequency_ghz"],
                    result["optical_spectrum_db"],
                    max_points=max_plot_points,
                )
                optical_frequency_ghz = optical_frequency_ds.astype(np.float32, copy=False)
                optical_spectrum_map_db = np.full(
                    (kappa_values.size, optical_frequency_ghz.size),
                    np.nan,
                    dtype=dtype,
                )
            else:
                optical_frequency_ds, optical_spectrum_ds = downsample_time_series(
                    result["optical_frequency_ghz"],
                    result["optical_spectrum_db"],
                    max_points=max_plot_points,
                )
                if optical_spectrum_ds.shape[0] != optical_frequency_ghz.shape[0]:
                    optical_spectrum_ds = np.interp(
                        optical_frequency_ghz,
                        optical_frequency_ds,
                        optical_spectrum_ds,
                    )
            optical_spectrum_map_db[kappa_index] = optical_spectrum_ds.astype(
                dtype,
                copy=False,
            )

            if psd_frequency_ghz is None:
                psd_frequency_ds, rin_psd_ds = downsample_time_series(
                    result["psd_frequency_ghz"],
                    result["rin_psd_db_hz"],
                    max_points=max_plot_points,
                )
                psd_frequency_ghz = psd_frequency_ds.astype(np.float32, copy=False)
                rin_psd_map_db = np.full(
                    (kappa_values.size, psd_frequency_ghz.size),
                    np.nan,
                    dtype=dtype,
                )
            else:
                psd_frequency_ds, rin_psd_ds = downsample_time_series(
                    result["psd_frequency_ghz"],
                    result["rin_psd_db_hz"],
                    max_points=max_plot_points,
                )
                if rin_psd_ds.shape[0] != psd_frequency_ghz.shape[0]:
                    rin_psd_ds = np.interp(
                        psd_frequency_ghz,
                        psd_frequency_ds,
                        rin_psd_ds,
                    )
            rin_psd_map_db[kappa_index] = rin_psd_ds.astype(dtype, copy=False)

            if write_time_series_frames:
                plot_time_series_frame(
                    result,
                    kappa_value,
                    time_series_frames_dir
                    / f"kappa_{kappa_index:04d}_{kappa_value * 1e-9:.3f}ns_inv_time_series.png",
                )

            linewidth_mean_hz[kappa_index] = result["linewidth_hz"]
            linewidth_median_hz[kappa_index] = result["linewidth_median_hz"]
            linewidth_std_hz[kappa_index] = result["linewidth_std_hz"]
            linewidth_r_squared[kappa_index] = result["r_squared"]

            if fit_time_map is None:
                fit_time_map = np.full((kappa_values.size, result["fit_time"].size), np.nan, dtype=dtype)
                fit_line_map = np.full_like(fit_time_map, np.nan)
            if result["fit_time"].size == fit_time_map.shape[1]:
                fit_time_map[kappa_index] = result["fit_time"].astype(dtype, copy=False)
                fit_line_map[kappa_index] = result["fit_line"].astype(dtype, copy=False)

            if write_fit_frames:
                figure_path = (
                    fit_frames_dir
                    / f"kappa_{kappa_index:04d}_{kappa_value * 1e-9:.3f}ns_inv_fit.png"
                )
                plot_phase_variance_fit_result(result, kappa_value, figure_path)

        if write_progress_frame and phase_variance_map is not None:
            plot_phase_variance_continuation_frame(
                frames_dir / "linewidth_progress.png",
                kappa_values,
                feedback_phase_map,
                feedback_phase_time,
                phase_variance_map,
                lag_times,
                linewidth_mean_hz,
                linewidth_median_hz,
                linewidth_r_squared,
                henry_branches,
                completed_kappa_indices=completed_kappa_indices,
            )
            plot_intensity_psd_continuation_frame(
                frames_dir / "intensity_psd_progress.png",
                kappa_values,
                optical_spectrum_map_db,
                optical_frequency_ghz,
                rin_psd_map_db,
                psd_frequency_ghz,
                frequency_map,
                frequency_time,
                frequency_variance_map,
                completed_kappa_indices=completed_kappa_indices,
            )

    def process_future(future):
        progress_id = future_progress_id.pop(future, None)
        try:
            results = future.result()
        except BrokenProcessPool as exc:
            if progress_id in progress_bars:
                progress_bars[progress_id].close()
                del progress_bars[progress_id]
            raise RuntimeError(
                "A long fixed-kappa worker process was killed abruptly. "
                "Try long_run_jobs = 1 to expose the original error path."
            ) from exc
        except Exception as exc:
            if progress_id in progress_bars:
                progress_bars[progress_id].close()
                del progress_bars[progress_id]
            raise RuntimeError(
                "A long fixed-kappa single-laser phase-diffusion chunk failed. "
                "The original exception is chained below."
            ) from exc
        process_chunk_results(results, progress_id)

    def collect_finished_chunks(block=False, stop_after_one=False):
        while pending_futures:
            made_progress = False
            for future in list(pending_futures):
                if future.done():
                    pending_futures.remove(future)
                    process_future(future)
                    made_progress = True
                    if stop_after_one:
                        return
            if not block or not pending_futures:
                return
            if show_long_run_worker_progress and progress_queue is not None:
                try:
                    progress_id, n_steps = progress_queue.get(timeout=0.1)
                    if progress_id in progress_bars:
                        progress_bars[progress_id].update(int(n_steps))
                except queue_module.Empty:
                    pass
            elif not made_progress:
                # Avoid a tight loop when running without worker progress queues.
                time_wait = 0.1
                try:
                    import time

                    time.sleep(time_wait)
                except Exception:
                    pass

    def drain_progress_queue():
        if not show_long_run_worker_progress or progress_queue is None:
            return
        while True:
            try:
                progress_id, n_steps = progress_queue.get_nowait()
            except queue_module.Empty:
                break
            if progress_id in progress_bars:
                progress_bars[progress_id].update(int(n_steps))

    def submit_chunk():
        nonlocal current_chunk, progress_counter
        if not current_chunk:
            return
        jobs = current_chunk
        current_chunk = []
        progress_id = progress_counter
        progress_counter += 1
        for job in jobs:
            job["progress_id"] = progress_id
            job["progress_update_steps"] = long_run_progress_update_steps

        if show_long_run_worker_progress:
            total_steps = long_steps
            progress_bars[progress_id] = tqdm(
                total=total_steps,
                desc=f"long single-laser k={jobs[0]['k']}-{jobs[-1]['k']}",
                unit="step",
                leave=False,
            )

        if executor is None:
            result = run_fixed_kappa_linewidth_chunk_worker(jobs)
            process_chunk_results(result, progress_id)
        else:
            future = executor.submit(run_fixed_kappa_linewidth_chunk_worker, jobs)
            pending_futures.append(future)
            future_progress_id[future] = progress_id

    try:
        previous_kappa = float(kappa_values[0])
        short_bar = tqdm(
            enumerate(kappa_values),
            total=kappa_values.size,
            desc="short kappa continuation",
            unit="step",
        )
        for kappa_index, kappa in short_bar:
            kappa = float(kappa)
            short_bar.set_postfix(kappa_ns=f"{kappa * 1e-9:.3f}")
            vcsel, nd = make_vcsel_ramp(
                previous_kappa,
                kappa,
                Tmax_continuation,
                time_array_continuation,
                continuation_save_every,
            )
            _, _, _, final_history = vcsel.integrate(
                history,
                nd=nd,
                progress=False,
                max_iter=5,
                smooth_freqs=False,
                return_final_history=True,
                message=f"short continuation k={kappa_index}",
            )
            history = (
                average_history_across_cases(final_history)
                if average_continuation_history_across_noise
                else final_history
            )
            previous_kappa = kappa
            current_chunk.append(make_linewidth_job(kappa_index, kappa, history))
            if (
                len(current_chunk) >= int(max(1, two_stage_kappa_chunk_size))
                or kappa_index == len(kappa_values) - 1
            ):
                submit_chunk()
            drain_progress_queue()
            collect_finished_chunks(block=False)

            # Once the executing and queued waves are both full, wait for one
            # long chunk only. Continuation then prepares one replacement
            # chunk while all other workers continue running.
            if (
                use_process_pool
                and len(pending_futures) >= max_pending_long_chunks
            ):
                short_bar.set_postfix(
                    kappa_ns=f"{kappa * 1e-9:.3f}",
                    state="long-worker queue full",
                )
                collect_finished_chunks(block=True, stop_after_one=True)
                gc.collect()

        short_bar.close()
        submit_chunk()
        collect_finished_chunks(block=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        for progress_bar in list(progress_bars.values()):
            progress_bar.close()
        progress_bars.clear()

    if write_progress_frame and phase_variance_map is not None:
        plot_phase_variance_continuation_frame(
            frames_dir / "linewidth_progress.png",
            kappa_values,
            feedback_phase_map,
            feedback_phase_time,
            phase_variance_map,
            lag_times,
            linewidth_mean_hz,
            linewidth_median_hz,
            linewidth_r_squared,
            henry_branches,
            completed_kappa_indices=completed_kappa_indices,
        )
        plot_intensity_psd_continuation_frame(
            frames_dir / "intensity_psd_progress.png",
            kappa_values,
            optical_spectrum_map_db,
            optical_frequency_ghz,
            rin_psd_map_db,
            psd_frequency_ghz,
            frequency_map,
            frequency_time,
            frequency_variance_map,
            completed_kappa_indices=completed_kappa_indices,
        )

    if write_progress_arrays:
        np.savez(
            arrays_dir / "single_laser_linewidth.npz",
            linewidth_estimator="single_laser_phase_diffusion",
            linewidth_fit_quantity="phase_increment_variance",
            kappa_values=kappa_values,
            linewidth_hz=linewidth_mean_hz,
            linewidth_median_hz=linewidth_median_hz,
            linewidth_std_hz=linewidth_std_hz,
            r_squared=linewidth_r_squared,
            lag_times=lag_times,
            phase_variance_map=phase_variance_map,
            phase_variance_sem_map=phase_variance_sem_map,
            fit_time_map=fit_time_map,
            fit_line_map=fit_line_map,
            feedback_phase_time=feedback_phase_time,
            feedback_phase_map=feedback_phase_map,
            frequency_time=frequency_time,
            frequency_map=frequency_map,
            frequency_variance_map=frequency_variance_map,
            optical_frequency_ghz=optical_frequency_ghz,
            optical_spectrum_map_db=optical_spectrum_map_db,
            psd_frequency_ghz=psd_frequency_ghz,
            rin_psd_map_db=rin_psd_map_db,
            henry_ecm_branch_kappa_values=henry_branches["kappa_values"],
            henry_ecm_branch_linewidth_hz=henry_branches["linewidth_hz"],
            henry_ecm_branch_omega_rad_s=henry_branches["omega_rad_s"],
            henry_ecm_branch_denominator=henry_branches["denominator"],
            henry_ecm_branch_positive_slope=henry_branches["positive_slope"],
            free_running_henry_hz=delta_nu_0_hz,
        )

    print(f"Free-running Henry linewidth: {delta_nu_0_hz * 1e-6:.3f} MHz")
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
