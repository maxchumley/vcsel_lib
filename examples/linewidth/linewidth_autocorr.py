#%%
"""Estimate collective phase-diffusion linewidth over a continuation in coupling."""

import multiprocessing as mp
import gc
import ctypes
import inspect
import os
import queue as queue_module
import sys
import time
import traceback
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import matplotlib
import numpy as np
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator
from matplotlib import rc
from tqdm.auto import tqdm 

from vcsel_lib import VCSEL
try:
    from examples._paths import LINEWIDTH_RESULTS_DIR
except ModuleNotFoundError:
    LINEWIDTH_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "linewidth_estimation"

try:
    from setproctitle import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None


def set_activity_monitor_process_title(role, warn_if_unavailable=False):
    """Give this Python process a readable macOS Activity Monitor title."""
    # Put the unique role first because Activity Monitor often truncates the
    # right side of process titles in its default narrow name column.
    title = f"{role} - python{sys.version_info.major}.{sys.version_info.minor}"
    if _setproctitle is None:
        if warn_if_unavailable:
            warnings.warn(
                "Process titles were requested but setproctitle is not "
                "installed. Reinstall vcsel-lib (or install setproctitle) to "
                "show continuation/long-worker names in Activity Monitor.",
                RuntimeWarning,
                stacklevel=2,
            )
        return False
    _setproctitle(title)
    if sys.platform == "darwin":
        # Activity Monitor can obtain names through macOS APIs that do not
        # always use the argv/comm title updated by setproctitle. Set the
        # Darwin program name and main-thread name as well. Keep these short
        # and role-first so long1/long2/long3 cannot be visually truncated to
        # the same value.
        libc = ctypes.CDLL(None)
        role_bytes = str(role).encode("utf-8")[:63]
        setprogname = getattr(libc, "setprogname", None)
        if setprogname is not None:
            setprogname.argtypes = [ctypes.c_char_p]
            setprogname.restype = None
            setprogname(role_bytes)
        pthread_setname_np = getattr(libc, "pthread_setname_np", None)
        if pthread_setname_np is not None:
            pthread_setname_np.argtypes = [ctypes.c_char_p]
            pthread_setname_np.restype = ctypes.c_int
            pthread_setname_np(role_bytes)
    return True


def _disposable_plot_worker(plot_function, args, kwargs, error_connection):
    """Render one figure in a child whose exit releases all plotting RAM."""
    # Do not call setproctitle here.  This process is created with ``fork``
    # while the IPython kernel and long-worker pool have active threads.
    # On macOS, setproctitle enters CoreFoundation on the child side of that
    # fork and can crash with EXC_BAD_ACCESS before plotting begins.  The
    # disposable process is intentionally short-lived, so a custom Activity
    # Monitor name is not worth that native crash risk.
    try:
        plot_function(*args, **kwargs)
    except BaseException:
        try:
            error_connection.send(traceback.format_exc())
        finally:
            error_connection.close()
        raise
    error_connection.close()


def render_figure(plot_function, *args, **kwargs):
    """Render directly or in a verified, short-lived forked process."""
    if not render_figures_in_disposable_process:
        return plot_function(*args, **kwargs)

    # Fork is intentional here: the plotter reads the parent's large NumPy
    # maps through copy-on-write memory instead of serializing/copying them.
    # All Matplotlib/NumPy temporaries allocated while rendering disappear
    # when this process exits.
    plot_context = mp.get_context("fork")
    error_reader, error_writer = plot_context.Pipe(duplex=False)
    plot_process = plot_context.Process(
        target=_disposable_plot_worker,
        args=(plot_function, args, kwargs, error_writer),
        name="linewidth-plotter",
    )
    try:
        plot_process.start()
        error_writer.close()
        plot_process.join()
        try:
            error_text = error_reader.recv() if error_reader.poll() else None
        except EOFError:
            error_text = None
        exit_code = plot_process.exitcode
    finally:
        error_reader.close()
        error_writer.close()
        if plot_process.pid is not None:
            plot_process.close()

    if exit_code != 0:
        details = error_text or "Plotter exited without returning a traceback."
        raise RuntimeError(
            f"Disposable figure process failed with exit code {exit_code}.\n{details}"
        )


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


# ----------------------------- User controls -----------------------------

N_lasers = 2
n_noise_iterations = 50
random_seed = None

# Follow one physical, noise-free continuation trajectory.  Its complete
# delayed history is replicated only when a long fixed-kappa noise ensemble
# starts, so averaging wrapped phases or unrelated stochastic histories never
# creates the branch history.
continuation_n_cases = 1
continuation_noise_amplitude = 0.0

# These are the simulation-duration controls. Parameter presets below should
# not override them. The long run includes a noisy burn-in that is discarded
# before PSD and linewidth analysis. With the fixed-kappa history setting
# below, its retained duration is ``Tmax_long - long_run_burn_in_time``.
Tmax_continuation = 1.0e-6
Tmax_long = 12.0e-6
long_run_burn_in_time = 2.0e-6
dt_multiplier = 1.0
continuation_save_every = 8 
analysis_save_every = 2
analysis_output_dtype = np.float64

detuning_ghz = 0.0
kappa_initial = 0.0
kappa_final = 100.0e9
n_kappa_steps = 100

two_stage_kappa_chunk_size = 5
long_run_jobs = 10
serialize_long_run_postprocessing = True
show_long_run_worker_progress = True
show_output_size_messages = False
print_psd_frequency_range_on_start = True
max_output_gb = 10.0

plot_lag_max_ns = 2000.0
fit_lag_ns = (50.0, 500.0)
n_lag_points = 2000

write_progress_arrays = True
write_progress_frame = True
write_fit_frames = False
write_psd_frames = True
render_figures_in_disposable_process = True
plot_kappa0_henry_reference = True
compute_phase_variance_linewidth = False

# --- Temporary Ma et al. (2019) preset -----------------------------------
# Flip this to True for a one-off Ma et al. parameter run.  When you are done,
# remove this whole boxed block plus the small "Apply Ma et al." block below.
use_ma_et_al_2019_parameters = False
ma_et_al_2019_delay_regime = "long_fiber"  # "short", "long", or "long_fiber".

# ------------------------ Plot style / fixed choices ----------------------

long_run_progress_update_steps = 200
max_plot_points = 6000
figure_dpi = 160
continuation_frame_dpi = 300
continuation_frame_font_size = 22
continuation_frame_title_pad = 16
continuation_linewidth_legend_font_size = 13
linewidth_plot_ylim_mhz = (1e-5, 1e6)
linewidth_plot_yticks_mhz = [1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]

phase_variance_colorbar_log_scale = True
phase_variance_log_vmin_rad2 = None
phase_variance_log_vmax_rad2 = None
phase_variance_log_vmin_percentile = 1.0
phase_variance_log_vmax_percentile = 99.5
phase_diffusion_colorbar_ticks = None
kappa_ramp_start_tau = 5.0
kappa_ramp_shape_tau = 200.0
phi_p = 0.0
noise_amplitude = 1.0
analysis_fraction = 1.0
post_ramp_settling_ns = 250.0
phase_variance_lag_axis_ns = (0.0, 2000.0)
phase_diffusion_fit_ylim = None
phase_diffusion_fit_yticks = None
maximum_fit_lag_fraction = 0.1
phase_slip_step_threshold_rad = np.pi
use_available_data_if_tmax_too_short = True
minimum_analysis_samples = 16
# This skips the legacy ramp/settling cutoff for fixed-kappa histories. The
# explicit ``long_run_burn_in_time`` above is still always discarded.
long_run_history_is_steady_state = True
phase_plot_tail_window_us = 2.0

# Frequency-noise PSD linewidth estimator.  The progress plot uses the
# combined bright-field channel by default because that is closest to a
# far-field measurement, while the individual-laser and common-phase channels
# are still computed for comparison.
psd_heatmap_channel = "common"  # Paper-comparison channel; alternatives: combined/common.
paper_target_kappa_hz = 5.0e8
paper_target_linewidth_hz = 103.5
minimum_phase_diffusion_r_squared_for_plot = 0.9
# Ma et al. use Delta_nu = pi*S_0 (S_0 = 20 Hz^2/Hz gives 62.8 Hz).
# A nonpositive lower edge is an intentional sentinel meaning "start at the
# lowest positive frequency bin available from this retained record." It is
# resolved once before workers start, so changing Tmax_long automatically
# changes the lower edge while the upper edge remains fixed at 2 MHz.
psd_floor_band_hz = (0.0, 2.0e6)
# Fallback display bounds only.  When a PSD grid exists, the plots use the
# full positive frequency range available from the simulation.  The floor band
# below only controls the linewidth-floor estimate and the shaded guide band.
psd_plot_frequency_axis_ghz = (1e-3, 50.0)
# None uses the entire retained record, so the lowest plotted frequency is
# determined only by record duration (~1/T), never by psd_floor_band_hz.
# Set an integer only when intentionally trading low-frequency reach for more
# Welch segments within each noise realization.
psd_welch_nperseg = None
psd_welch_overlap_fraction = 0.5
psd_colorbar_vmin_hz2_per_hz = None
psd_colorbar_vmax_hz2_per_hz = None
psd_colorbar_vmin_percentile = 2.0
psd_colorbar_vmax_percentile = 98.0
psd_frame_ylim_hz2_per_hz = (1e0, 1e12) 
psd_frame_yticks_hz2_per_hz = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e11, 1e12]
plot_phase_variance_linewidth_comparison = False


PHASE_VARIANCE_LABEL = r"$\mathrm{var}[\Phi(t+\Delta t)-\Phi(t)]$ (rad$^2$)"
PHASE_VARIANCE_SHORT_LABEL = r"phase-increment variance (rad$^2$)"
FREQUENCY_PSD_LABEL = r"frequency-noise PSD, $S_\nu$ (Hz$^2$/Hz)"


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


output_name_suffix = ""

# --- Apply temporary Ma et al. (2019) preset -------------------------------
# Source: Ma et al., Appl. Sci. 2019, "Linewidth Narrowing of Mutually
# Injection Locked Semiconductor Lasers with Short and Long Delay", Table 1.
# This intentionally lives in one compact block so it is easy to delete after
# the one-off comparison run. Keep output_name_suffix above if you keep the
# generic named_output() helper below.
if use_ma_et_al_2019_parameters:
    output_name_suffix = "_ma_et_al"

    N_lasers = 2
    n_noise_iterations = 50
    n_kappa_steps = 101
    two_stage_kappa_chunk_size = 5
    kappa_ramp_shape_tau = 40.0

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
        kappa_ramp_start_tau = 1.0
        kappa_ramp_shape_tau = 8.0
    else:
        raise ValueError(
            "ma_et_al_2019_delay_regime must be 'short', 'long', or "
            "'long_fiber'."
        )


def named_output(stem, extension):
    """Return an output file name with the temporary preset suffix, if enabled."""
    return f"{stem}{output_name_suffix}{extension}"


output_dir = Path(
    LINEWIDTH_RESULTS_DIR / f"{N_lasers}_lasers/"
    f"phase_diffusion_linewidth_continuation{output_name_suffix}"
)
frames_dir = output_dir / "kappa_frames"
fit_frames_dir = output_dir / "phase_variance_fit_frames"
psd_frames_dir = output_dir / "frequency_noise_psd_frames"
arrays_dir = output_dir / "numpy_arrays"
worker_arrays_dir = output_dir / "worker_arrays"


def replicate_physical_history(history, n_cases):
    """Replicate one physical delayed trajectory for a noise ensemble."""
    history = np.asarray(history)
    if history.ndim != 3 or history.shape[0] != 1:
        raise ValueError(
            "Continuation history must contain exactly one physical trajectory; "
            f"got {history.shape}."
        )
    return np.repeat(history, int(max(1, n_cases)), axis=0)


def average_wrapped_phase_trace_from_phi(phi):
    """Absolute circular average of wrapped phi_i - phi_0 over cases/lasers."""
    phi = np.asarray(phi)
    if phi.ndim != 3:
        raise ValueError(f"Expected phi with shape (n_cases,N_lasers,time), got {phi.shape}")
    if phi.shape[1] <= 1:
        return np.zeros(phi.shape[-1], dtype=np.float32)
    wrapped_diff = np.angle(np.exp(1j * (phi[:, 1:, :] - phi[:, 0:1, :])))
    mean_phasor = np.mean(np.exp(1j * wrapped_diff), axis=(0, 1))
    return np.abs(np.angle(mean_phasor)).astype(np.float32)


def order_parameter_from_s_phi(S, phi):
    """Kuramoto-like order parameter, evaluated one case at a time for RAM."""
    S = np.asarray(S)
    phi = np.asarray(phi)
    if S.shape != phi.shape or S.ndim != 3:
        raise ValueError(
            "Expected S and phi with matching shape "
            "(n_cases,N_lasers,time)."
        )
    eps = 1e-12
    order_parameter = np.full(S.shape[0], np.nan, dtype=np.float64)
    for case_index in range(S.shape[0]):
        E_sum = np.zeros(S.shape[-1], dtype=np.complex64)
        denom_sum = np.zeros(S.shape[-1], dtype=np.float32)
        for laser_idx in range(S.shape[1]):
            S_i = np.maximum(
                S[case_index, laser_idx, :],
                eps,
            ).astype(np.float32, copy=False)
            E_i = (
                np.sqrt(S_i)
                * np.exp(
                    (1j * phi[case_index, laser_idx, :]).astype(np.complex64)
                )
            ).astype(np.complex64, copy=False)
            E_sum += E_i
            denom_sum += S_i
            del E_i
        numerator = np.abs(E_sum) ** 2
        denom = S.shape[1] * denom_sum
        order_parameter[case_index] = np.mean(
            numerator / np.maximum(denom, eps)
        )
        del E_sum, denom_sum, numerator, denom
    return order_parameter


def empty_phase_diffusion_result(t, n_cases):
    """Return a shape-compatible NaN linewidth result for too-short tests."""
    t = np.asarray(t, dtype=float)
    if t.size > 1:
        dt_values = np.diff(t)
        positive_dt = dt_values[dt_values > 0.0]
        sample_dt = (
            float(np.median(positive_dt))
            if positive_dt.size
            else float(analysis_save_every * dt)
        )
    else:
        sample_dt = float(analysis_save_every * dt)

    if t.size > 2:
        largest_valid_lag = max(sample_dt, (t.size - 2) * sample_dt)
    else:
        largest_valid_lag = sample_dt
    plot_lag_max = min(float(plot_lag_max_ns) * 1e-9, largest_valid_lag)
    max_lag_samples = max(1, int(np.floor(plot_lag_max / sample_dt)))
    lag_samples = np.unique(
        np.round(
            np.linspace(1, max_lag_samples, int(max(10, n_lag_points)))
        ).astype(int)
    )
    lag_times = lag_samples * sample_dt
    phase_variance = np.full(lag_times.size, np.nan, dtype=float)
    phase_variance_sem = np.full_like(phase_variance, np.nan)
    return {
        "linewidth_hz": np.nan,
        "linewidth_cases_hz": np.full(int(n_cases), np.nan, dtype=float),
        "linewidth_median_hz": np.nan,
        "linewidth_std_hz": np.nan,
        "r_squared": np.nan,
        "case_r_squared": np.full(int(n_cases), np.nan, dtype=float),
        "phase_slip_counts": np.zeros(int(n_cases), dtype=int),
        "phase_slip_cases": np.zeros(int(n_cases), dtype=bool),
        "lag_times": lag_times,
        "phase_variance": phase_variance,
        "phase_variance_sem": phase_variance_sem,
        "fit_time": np.asarray([], dtype=float),
        "fit_line": np.asarray([], dtype=float),
        "detrended_phase": np.empty((int(n_cases), 0), dtype=float),
    }


def plot_order_parameter_panel(
    ax,
    order_param,
    kappa_values,
    font_size,
    pad=12,
    order_param_min=None,
    order_param_max=None,
):
    """Plot mean order parameter with optional min/max noise-realization envelope."""
    kappa_axis = np.asarray(kappa_values) * 1e-9
    order_param = np.asarray(order_param)
    if order_param_min is not None and order_param_max is not None:
        order_param_min = np.asarray(order_param_min)
        order_param_max = np.asarray(order_param_max)
        valid_env = (
            np.isfinite(order_param_min)
            & np.isfinite(order_param_max)
            & (order_param_min <= order_param_max)
        )
        if np.any(valid_env):
            ax.fill_betweenx(
                kappa_axis[valid_env],
                order_param_min[valid_env],
                order_param_max[valid_env],
                color="black",
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                order_param_min[valid_env],
                kappa_axis[valid_env],
                color="black",
                linewidth=0.7,
                alpha=0.35,
            )
            ax.plot(
                order_param_max[valid_env],
                kappa_axis[valid_env],
                color="black",
                linewidth=0.7,
                alpha=0.35,
            )

    valid_mean = np.isfinite(order_param)
    if np.any(valid_mean):
        ax.plot(
            order_param[valid_mean],
            kappa_axis[valid_mean],
            color="black",
            linewidth=2,
        )
    ax.set_title("Order Parameter", fontsize=font_size, pad=pad)
    ax.set_ylim(kappa_axis[0], kappa_axis[-1])
    ax.tick_params(axis="both", labelsize=font_size)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 1, 2))
    ax.set_xlim(0, 1)


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
    """Estimate linewidth from the variance growth of an unwrapped phase.

    For white phase diffusion,

        var[Phi(t + lag) - Phi(t)] = 2*pi*Delta_nu*lag,

    so Delta_nu is the fitted variance slope divided by 2*pi.  The phase
    supplied here should represent the collective lasing phase.  A linear
    frequency trend is removed independently from every noise realization.

    The phase-increment variance is evaluated with FFT autocorrelations,
    avoiding an O(number_of_lags * number_of_samples) difference loop.
    """
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
            fit_lag_max = min(
                requested_lag_max,
                reliable_fit_lag_max,
                largest_valid_lag,
            )
        else:
            required_duration = requested_lag_max / maximum_fit_fraction
            raise ValueError(
                f"fit_lag_ns ends at {requested_lag_max * 1e9:.3g} ns, but "
                f"the retained analysis duration is only "
                f"{analysis_duration * 1e9:.3g} ns. Keep the fit below "
                f"{100.0 * maximum_fit_fraction:.0f}% of the analysis record "
                f"(currently {reliable_fit_lag_max * 1e9:.3g} ns), or retain "
                f"at least {required_duration * 1e6:.3g} us of stationary data."
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
        np.round(
            np.linspace(1, max_lag_samples, int(max(10, n_lags)))
        ).astype(int)
    )
    lag_times = lag_samples * dt_local

    # Constant offsets and linear frequency trends do not contribute to phase
    # diffusion. Process one realization at a time to keep worker RAM bounded.
    t_centered = t - np.mean(t)
    trend_denominator = np.dot(t_centered, t_centered)

    n_samples = t.size
    fft_length = 1 << int(np.ceil(np.log2(2 * n_samples - 1)))
    variance_cases = np.empty((phase.shape[0], lag_samples.size), dtype=float)
    for case_index in range(phase.shape[0]):
        trace = np.unwrap(
            np.asarray(phase[case_index], dtype=float),
        )
        trace -= np.mean(trace)
        trend_slope = np.dot(trace, t_centered) / trend_denominator
        trace -= trend_slope * t_centered
        spectrum = np.fft.rfft(trace, n=fft_length)
        autocorrelation = np.fft.irfft(
            spectrum * np.conj(spectrum),
            n=fft_length,
        )[:n_samples]

        cumulative = np.concatenate(([0.0], np.cumsum(trace)))
        cumulative_square = np.concatenate(([0.0], np.cumsum(trace * trace)))
        pair_counts = n_samples - lag_samples
        left_sum = cumulative[n_samples - lag_samples]
        right_sum = cumulative[n_samples] - cumulative[lag_samples]
        left_square_sum = cumulative_square[n_samples - lag_samples]
        right_square_sum = (
            cumulative_square[n_samples] - cumulative_square[lag_samples]
        )
        mean_increment = (right_sum - left_sum) / pair_counts
        mean_square_increment = (
            left_square_sum
            + right_square_sum
            - 2.0 * autocorrelation[lag_samples]
        ) / pair_counts
        variance_cases[case_index] = np.maximum(
            mean_square_increment - mean_increment * mean_increment,
            0.0,
        )

    # Removing the finite-record mean increment turns ideal Brownian phase
    # noise into a Brownian bridge and biases the sample variance downward.
    # For lag <= half the record, divide by the exact continuous-time bridge
    # factor. The configured fit is deliberately limited to 10% of the record,
    # where this correction is modest and the number of independent increments
    # remains useful.
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
        phase_variance_sem = np.nanstd(
            variance_cases,
            axis=0,
            ddof=1,
        ) / np.sqrt(np.maximum(finite_counts, 1))
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
    linewidth_median_hz = (
        float(np.median(valid_linewidths))
        if valid_linewidths.size
        else np.nan
    )
    linewidth_std_hz = (
        float(np.std(valid_linewidths, ddof=1))
        if valid_linewidths.size > 1
        else np.nan
    )

    return {
        "linewidth_hz": linewidth_hz,
        "linewidth_cases_hz": linewidth_cases_hz,
        "linewidth_median_hz": linewidth_median_hz,
        "linewidth_std_hz": linewidth_std_hz,
        "r_squared": r_squared,
        "case_r_squared": case_r_squared,
        "phase_slip_counts": np.zeros(phase.shape[0], dtype=int),
        "phase_slip_cases": np.zeros(phase.shape[0], dtype=bool),
        "lag_times": lag_times,
        "phase_variance": phase_variance,
        "phase_variance_sem": phase_variance_sem,
        "fit_time": fit_time,
        "fit_line": fit_line,
        "detrended_phase": np.empty((phase.shape[0], 0), dtype=float),
    }


def _largest_power_of_two_at_most(value):
    """Return the largest power of two <= value, with a minimum of one."""
    value = int(max(1, value))
    return 1 << int(np.floor(np.log2(value)))


def one_sided_welch_psd(
    traces,
    sample_dt,
    nperseg=None,
    overlap_fraction=0.5,
    minimum_frequency_hz=None,
):
    """Estimate one-sided PSDs for a batch of uniformly sampled real traces.

    Parameters
    ----------
    traces:
        Array with shape (n_cases, n_time).
    sample_dt:
        Time spacing in seconds.

    Returns
    -------
    frequencies_hz, psd_cases
        `psd_cases` has shape (n_cases, n_frequencies).  Each row is the
        Welch average for one realization.
    """
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim == 1:
        traces = traces[None, :]
    if traces.ndim != 2:
        raise ValueError("PSD traces must have shape (n_cases, n_time).")

    n_time = traces.shape[-1]
    if n_time < 8:
        return np.asarray([], dtype=float), np.empty((traces.shape[0], 0), dtype=float)

    if nperseg is None:
        # The full record exposes the lowest frequency supported by the
        # simulation duration. Ensemble averaging across independent noise
        # cases supplies statistical averaging without shortening the segment.
        # minimum_frequency_hz is retained in the public signature for
        # compatibility, but deliberately does not control the PSD grid.
        nperseg_use = n_time
    else:
        nperseg_use = min(int(max(8, nperseg)), n_time)
    nperseg_use = max(8, nperseg_use)

    overlap_fraction = float(np.clip(overlap_fraction, 0.0, 0.95))
    step = int(max(1, round(nperseg_use * (1.0 - overlap_fraction))))
    starts = np.arange(0, n_time - nperseg_use + 1, step, dtype=int)
    if starts.size == 0:
        starts = np.asarray([0], dtype=int)

    fs = 1.0 / float(sample_dt)
    window = np.hanning(nperseg_use)
    window_power = float(np.sum(window * window))
    if window_power <= 0.0:
        window = np.ones(nperseg_use, dtype=float)
        window_power = float(nperseg_use)

    frequencies = np.fft.rfftfreq(nperseg_use, d=sample_dt)
    psd_cases = np.full((traces.shape[0], frequencies.size), np.nan, dtype=float)
    for case_index, trace in enumerate(traces):
        psd_accum = np.zeros(frequencies.size, dtype=float)
        valid_segments = 0
        for start in starts:
            segment = np.asarray(trace[start:start + nperseg_use], dtype=float)
            if segment.size != nperseg_use or not np.all(np.isfinite(segment)):
                continue
            segment = segment - np.mean(segment)
            spectrum = np.fft.rfft(segment * window)
            psd = (np.abs(spectrum) ** 2) / (fs * window_power)
            if psd.size > 2:
                if nperseg_use % 2 == 0:
                    psd[1:-1] *= 2.0
                else:
                    psd[1:] *= 2.0
            psd_accum += psd
            valid_segments += 1
        if valid_segments:
            psd_cases[case_index] = psd_accum / valid_segments
    return frequencies, psd_cases


def _unwrap_phase_batch(phase):
    """Unwrap phase along time for every realization."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim == 1:
        phase = phase[None, :]
    return np.unwrap(phase, axis=-1)


def _frequency_traces_from_phase(t, phase):
    """Convert phase traces to mean-removed instantaneous frequency traces."""
    t = np.asarray(t, dtype=float)
    phase = _unwrap_phase_batch(phase)
    dt_values = np.diff(t)
    dt_local = float(np.median(dt_values))
    if np.any(dt_values <= 0.0) or not np.allclose(dt_values, dt_local, rtol=1e-6):
        raise ValueError("t must be uniformly sampled for frequency-noise PSD.")
    frequency = np.diff(phase, axis=-1) / (2.0 * np.pi * dt_local)
    frequency -= np.mean(frequency, axis=-1, keepdims=True)
    return frequency, dt_local


def ensure_resolvable_psd_floor_band(frequencies_hz, minimum_bins=3):
    """Expand the configured floor band to the first resolvable bin window.

    The adjustment affects only the bins averaged for the scalar white-floor
    estimate. PSD plots continue to use every positive frequency bin.
    """
    global psd_floor_band_hz

    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    positive = np.unique(
        frequencies_hz[np.isfinite(frequencies_hz) & (frequencies_hz > 0.0)]
    )
    minimum_bins = int(max(1, minimum_bins))
    if positive.size < minimum_bins:
        warnings.warn(
            "PSD grid has fewer than three positive frequency bins; a white-"
            "floor estimate is unavailable for this record.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros_like(frequencies_hz, dtype=bool)

    requested_min_hz, requested_max_hz = [
        float(value) for value in psd_floor_band_hz
    ]
    if requested_min_hz <= 0.0:
        requested_min_hz = float(positive[0])
        psd_floor_band_hz = (requested_min_hz, requested_max_hz)
    if positive.size > 1:
        edge_tolerance = 0.5 * float(np.nanmedian(np.diff(positive)))
    else:
        edge_tolerance = 0.0
    requested_mask = (
        (frequencies_hz >= requested_min_hz - edge_tolerance)
        & (frequencies_hz <= requested_max_hz + edge_tolerance)
        & (frequencies_hz > 0.0)
    )
    if np.count_nonzero(requested_mask) >= minimum_bins:
        return requested_mask

    candidates = positive[positive >= requested_min_hz - edge_tolerance]
    if candidates.size < minimum_bins:
        candidates = positive
    first_window = candidates[:minimum_bins]
    adjusted_band = (float(first_window[0]), float(first_window[-1]))
    old_band = tuple(psd_floor_band_hz)
    psd_floor_band_hz = adjusted_band
    warnings.warn(
        "Configured psd_floor_band_hz does not contain at least "
        f"{minimum_bins} available positive PSD bins. Automatically changed "
        f"it from {old_band} Hz to {adjusted_band} Hz. This changes only the "
        "noise-floor averaging window; the full positive PSD range is still "
        "plotted.",
        RuntimeWarning,
        stacklevel=2,
    )
    return (
        (frequencies_hz >= adjusted_band[0] - edge_tolerance)
        & (frequencies_hz <= adjusted_band[1] + edge_tolerance)
        & (frequencies_hz > 0.0)
    )


def _estimate_psd_linewidth_from_band(frequencies_hz, psd_cases):
    """Convert a white frequency-noise floor estimate into linewidth."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    psd_cases = np.asarray(psd_cases, dtype=float)
    if frequencies_hz.size == 0 or psd_cases.size == 0:
        n_cases = psd_cases.shape[0] if psd_cases.ndim == 2 else 0
        return {
            "floor_cases": np.full(n_cases, np.nan, dtype=float),
            "linewidth_cases_hz": np.full(n_cases, np.nan, dtype=float),
            "linewidth_hz": np.nan,
            "linewidth_mean_hz": np.nan,
            "linewidth_std_hz": np.nan,
            "linewidth_q16_hz": np.nan,
            "linewidth_q84_hz": np.nan,
            "band_mask": np.zeros_like(frequencies_hz, dtype=bool),
        }

    band_mask = ensure_resolvable_psd_floor_band(frequencies_hz, minimum_bins=3)

    floor_cases = np.full(psd_cases.shape[0], np.nan, dtype=float)
    if np.count_nonzero(band_mask):
        floor_cases = np.nanmedian(psd_cases[:, band_mask], axis=1)

    # One-sided frequency-noise convention: Lorentzian FWHM Δν = π Sν,white.
    linewidth_cases_hz = np.pi * floor_cases
    finite = np.isfinite(linewidth_cases_hz) & (linewidth_cases_hz > 0.0)
    finite_linewidths = linewidth_cases_hz[finite]
    return {
        "floor_cases": floor_cases,
        "linewidth_cases_hz": linewidth_cases_hz,
        "linewidth_hz": (
            float(np.nanmedian(linewidth_cases_hz[finite]))
            if np.any(finite)
            else np.nan
        ),
        "linewidth_mean_hz": (
            float(np.nanmean(linewidth_cases_hz[finite]))
            if np.any(finite)
            else np.nan
        ),
        "linewidth_std_hz": (
            float(np.nanstd(finite_linewidths, ddof=1))
            if np.count_nonzero(finite) > 1
            else np.nan
        ),
        "linewidth_q16_hz": (
            float(np.nanpercentile(finite_linewidths, 16.0))
            if finite_linewidths.size
            else np.nan
        ),
        "linewidth_q84_hz": (
            float(np.nanpercentile(finite_linewidths, 84.0))
            if finite_linewidths.size
            else np.nan
        ),
        "band_mask": band_mask,
    }


def _frequency_noise_psd_channel(t, phase_trace, already_unwrapped=False):
    """Analyze one phase channel, releasing its large work arrays on return."""
    phase_trace = np.asarray(phase_trace, dtype=np.float64)
    if already_unwrapped:
        dt_values = np.diff(np.asarray(t, dtype=float))
        sample_dt = float(np.median(dt_values))
        if np.any(dt_values <= 0.0) or not np.allclose(
            dt_values,
            sample_dt,
            rtol=1e-6,
        ):
            raise ValueError("t must be uniformly sampled for frequency-noise PSD.")
        frequency_trace = np.diff(phase_trace, axis=-1) / (
            2.0 * np.pi * sample_dt
        )
        frequency_trace -= np.mean(frequency_trace, axis=-1, keepdims=True)
    else:
        frequency_trace, sample_dt = _frequency_traces_from_phase(t, phase_trace)

    frequencies_hz, psd_cases = one_sided_welch_psd(
        frequency_trace,
        sample_dt,
        nperseg=psd_welch_nperseg,
        overlap_fraction=psd_welch_overlap_fraction,
        minimum_frequency_hz=float(psd_floor_band_hz[0]),
    )
    floor_result = _estimate_psd_linewidth_from_band(frequencies_hz, psd_cases)
    result = {
        "psd": (
            np.nanmedian(psd_cases, axis=0)
            if psd_cases.size
            else np.asarray([], dtype=float)
        ),
        "linewidth_hz": floor_result["linewidth_hz"],
        "linewidth_mean_hz": floor_result["linewidth_mean_hz"],
        "linewidth_std_hz": floor_result["linewidth_std_hz"],
        "linewidth_q16_hz": floor_result["linewidth_q16_hz"],
        "linewidth_q84_hz": floor_result["linewidth_q84_hz"],
        "floor_cases": floor_result["floor_cases"],
        "linewidth_cases_hz": floor_result["linewidth_cases_hz"],
    }
    del frequency_trace, psd_cases, floor_result
    return frequencies_hz, result


def frequency_noise_psd_linewidth(t, S, phi):
    """Estimate white-floor linewidths from frequency-noise PSDs.

    Channels returned:
      - combined: phase of sum_i sqrt(S_i) exp(i phi_i)
      - common: mean of individually unwrapped laser phases
      - laser_i: each individual laser phase
    """
    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if S.shape != phi.shape or S.ndim != 3:
        raise ValueError("Expected S and phi with shape (n_cases,N_lasers,time).")

    channel_results = {}
    frequencies_hz = None

    # Accumulate the bright field one laser at a time.  This is algebraically
    # identical to sum_i sqrt(S_i) exp(1j*phi_i), but avoids retaining a full
    # complex (cases, lasers, time) field array.
    combined_real = np.zeros((S.shape[0], S.shape[-1]), dtype=np.float64)
    combined_imag = np.zeros_like(combined_real)
    component = np.empty_like(combined_real)
    for laser_index in range(S.shape[1]):
        amplitude = np.sqrt(
            np.maximum(S[:, laser_index, :], 0.0) + 1e-30
        )
        np.cos(phi[:, laser_index, :], out=component)
        component *= amplitude
        combined_real += component
        np.sin(phi[:, laser_index, :], out=component)
        component *= amplitude
        combined_imag += component
        del amplitude
    del component

    combined_phase = np.arctan2(combined_imag, combined_real)
    combined_phase = np.unwrap(combined_phase, axis=-1)
    np.square(combined_real, out=combined_real)
    np.square(combined_imag, out=combined_imag)
    combined_real += combined_imag
    combined_power_mean = float(np.nanmean(combined_real))
    combined_power_min = float(np.nanmin(np.mean(combined_real, axis=-1)))
    del combined_real, combined_imag

    frequencies_hz, channel_results["combined"] = _frequency_noise_psd_channel(
        t,
        combined_phase,
        already_unwrapped=True,
    )
    del combined_phase

    # Form the common phase without keeping every laser's unwrapped phase.
    common_phase = np.zeros((phi.shape[0], phi.shape[-1]), dtype=np.float64)
    for laser_index in range(phi.shape[1]):
        laser_phase = np.unwrap(phi[:, laser_index, :], axis=-1)
        common_phase += laser_phase
        del laser_phase
    common_phase /= float(phi.shape[1])
    common_frequencies_hz, channel_results["common"] = (
        _frequency_noise_psd_channel(t, common_phase, already_unwrapped=True)
    )
    if frequencies_hz is None:
        frequencies_hz = common_frequencies_hz
    del common_phase, common_frequencies_hz

    # Analyze individual lasers sequentially so only one unwrapped phase,
    # frequency trace, and Welch workspace is live at a time.
    for laser_index in range(phi.shape[1]):
        laser_frequencies_hz, channel_results[f"laser{laser_index + 1}"] = (
            _frequency_noise_psd_channel(
                t,
                phi[:, laser_index, :],
                already_unwrapped=False,
            )
        )
        if frequencies_hz is None:
            frequencies_hz = laser_frequencies_hz
        del laser_frequencies_hz

    if frequencies_hz is None:
        frequencies_hz = np.asarray([], dtype=float)

    heatmap_name = str(psd_heatmap_channel).lower()
    if heatmap_name.startswith("laser"):
        try:
            laser_number = int(heatmap_name.replace("laser", ""))
        except ValueError:
            laser_number = 1
        heatmap_name = f"laser{int(np.clip(laser_number, 1, phi.shape[1]))}"
    if heatmap_name not in channel_results:
        heatmap_name = "combined"

    return {
        "frequencies_hz": frequencies_hz,
        "heatmap_channel": heatmap_name,
        "heatmap_psd": channel_results[heatmap_name]["psd"],
        "combined_power_mean": combined_power_mean,
        "combined_power_min": combined_power_min,
        "channels": channel_results,
    }


def center_edges_from_centers(values, log=False):
    """Build pcolormesh edges from 1-D cell centers."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if values.size == 1:
        delta = values[0] * 0.1 if log and values[0] > 0.0 else 0.5
        return np.asarray([values[0] - delta, values[0] + delta], dtype=float)
    if log:
        safe_values = np.maximum(values, np.finfo(float).tiny)
        log_values = np.log10(safe_values)
        log_edges = np.empty(values.size + 1, dtype=float)
        log_edges[1:-1] = 0.5 * (log_values[:-1] + log_values[1:])
        log_edges[0] = log_values[0] - (log_edges[1] - log_values[0])
        log_edges[-1] = log_values[-1] + (log_values[-1] - log_edges[-2])
        return 10.0 ** log_edges
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - (edges[1] - values[0])
    edges[-1] = values[-1] + (values[-1] - edges[-2])
    return edges


# -------------------------- Continuation helpers -------------------------

dt = dt_multiplier * tau_p
continuation_steps = int(np.floor(Tmax_continuation / dt))
time_array_continuation = np.arange(continuation_steps, dtype=float) * dt
long_steps = int(np.floor(Tmax_long / dt))
time_array_long = np.arange(long_steps, dtype=float) * dt
delay_steps = int(round(tau / dt))

adjacency = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)
detuning = detuning_ghz * 2.0 * np.pi * 1e9
detuning_distribution = np.linspace(
    -detuning / 2.0,
    detuning / 2.0,
    N_lasers,
)
current = (
    eta
    * threshold_multiplier
    * q
    / tau_n
    * (N0 + 1.0 / (g0 * tau_p))
)


def make_kappa_matrix(time_array, kappa_start, kappa_stop):
    return VCSEL.build_coupling_matrix(
        time_arr=time_array,
        kappa_initial=float(kappa_start),
        kappa_final=float(kappa_stop),
        N_lasers=N_lasers,
        ramp_start=kappa_ramp_start_tau,
        ramp_shape=kappa_ramp_shape_tau,
        tau=tau,
        scheme="CUSTOM",
        aMAT=adjacency,
    )


def make_vcsel_from_kappa_matrix(
    kappa_matrix,
    tmax,
    save_every,
    extra_physical_parameters=None,
):
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
        "phi_p_mat": np.full((N_lasers, N_lasers), phi_p),
        "tau": tau,
        "I": current,
        "noise_amplitude": noise_amplitude,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "delta": detuning_distribution,
        "Tmax": tmax,
        "dt": dt,
        "N_lasers": N_lasers,
        "save_every": int(max(1, save_every)),
        "output_dtype": analysis_output_dtype,
        "max_output_gb": max_output_gb,
    }
    if extra_physical_parameters:
        physical_parameters.update(extra_physical_parameters)
    vcsel = VCSEL(physical_parameters)
    nd = vcsel.scale_params()
    nd["store_freqs"] = False
    nd["show_output_size_message"] = bool(show_output_size_messages)
    nd["output_dtype"] = np.dtype(analysis_output_dtype)
    nd["output_state_indices"] = np.ravel(
        np.column_stack((
            np.arange(N_lasers) * 3 + 1,
            np.arange(N_lasers) * 3 + 2,
        ))
    )
    return vcsel, nd


def make_vcsel_ramp(
    kappa_start,
    kappa_stop,
    tmax,
    time_array,
    save_every,
    extra_physical_parameters=None,
):
    return make_vcsel_from_kappa_matrix(
        make_kappa_matrix(time_array, kappa_start, kappa_stop),
        tmax,
        save_every,
        extra_physical_parameters=extra_physical_parameters,
    )


def free_running_henry_linewidth_hz(nd):
    """Free-running Henry linewidth implied by the rate-equation scaling."""
    carrier_number = (float(nd["nbar"]) + float(nd["n0"]) + 1.0) / (g0 * tau_p)
    photon_number = float(nd["sbar"]) / (g0 * tau_n)
    spontaneous_emission_rate = beta * carrier_number / tau_n
    return (1.0 + alpha**2) * spontaneous_emission_rate / (
        4.0 * np.pi * max(photon_number, 1e-30)
    )


def collective_kappa0_henry_reference_hz(nd):
    """Henry reference comparable to the mean unwrapped collective phase."""
    return free_running_henry_linewidth_hz(nd) / float(max(1, N_lasers))


def analysis_start_time_for_record(t_stop):
    """Mirror analyze_trajectory's retained-window start estimate."""
    full_ramp_duration = kappa_ramp_shape_tau * tau / 0.8
    ramp_end_time = kappa_ramp_start_tau * tau + full_ramp_duration
    stationary_start_time = ramp_end_time + post_ramp_settling_ns * 1e-9
    fraction_start_time = (1.0 - analysis_fraction) * t_stop
    return max(stationary_start_time, fraction_start_time)


def long_run_analysis_start_time(t_stop):
    """Return the absolute time at which long-run analysis may begin."""
    burn_in_time = float(long_run_burn_in_time)
    if not np.isfinite(burn_in_time) or burn_in_time < 0.0:
        raise ValueError("long_run_burn_in_time must be finite and nonnegative.")
    if long_run_history_is_steady_state:
        return burn_in_time
    return max(burn_in_time, analysis_start_time_for_record(t_stop))


def analysis_start_index_for_record(t, already_steady=False):
    """Return a robust analysis start index, falling back when Tmax is short."""
    t = np.asarray(t, dtype=float)
    if t.size < 4:
        raise ValueError("The saved analysis record has fewer than four samples.")
    requested_start_time = max(0.0, float(long_run_burn_in_time))
    if not already_steady:
        requested_start_time = max(
            requested_start_time,
            analysis_start_time_for_record(t[-1]),
        )
    requested_start = int(np.searchsorted(t, requested_start_time, side="left"))
    if not use_available_data_if_tmax_too_short:
        return requested_start

    min_samples = int(max(4, minimum_analysis_samples))
    min_samples = min(min_samples, t.size)
    latest_start = max(0, t.size - min_samples)

    if requested_start >= t.size - 3:
        warnings.warn(
            "Configured burn-in/ramp/settling cutoff leaves no stationary samples; "
            "using the full available record for linewidth analysis.",
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
    """Warn or fail early if the configured linewidth fit window cannot be reached."""
    analysis_start_time = long_run_analysis_start_time(Tmax_long)
    retained_duration = Tmax_long - analysis_start_time
    if float(long_run_burn_in_time) >= float(Tmax_long):
        raise ValueError(
            "long_run_burn_in_time must be shorter than Tmax_long; got "
            f"{long_run_burn_in_time * 1e6:.6g} us versus "
            f"{Tmax_long * 1e6:.6g} us."
        )
    requested_fit_lag_max = float(fit_lag_ns[1]) * 1e-9
    allowed_fit_lag_max = maximum_fit_lag_fraction * retained_duration
    lag_limit_tolerance = max(dt, 1e-12 * max(retained_duration, 0.0))

    if retained_duration <= 0.0:
        if use_available_data_if_tmax_too_short:
            warnings.warn(
                "No stationary analysis record remains after the configured "
                "burn-in/ramp/settling cutoff; using the full available long-run "
                "record instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        raise ValueError(
            "No stationary analysis record remains after the configured burn-in/"
            "ramp/settling cutoff. Increase Tmax_long, decrease "
            "long_run_burn_in_time or post_ramp_settling_ns, or reduce the "
            "ramp timing controls."
        )

    if requested_fit_lag_max > allowed_fit_lag_max + lag_limit_tolerance:
        if use_available_data_if_tmax_too_short:
            warnings.warn(
                "Requested fit_lag_ns extends beyond the reliable lag range "
                "for this Tmax_long; clipping the fit window to the available "
                "record.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        required_retained_duration = requested_fit_lag_max / maximum_fit_lag_fraction
        required_tmax = analysis_start_time + required_retained_duration
        raise ValueError(
            "The linewidth fit window is too long for the retained analysis "
            "record.\n"
            f"  Tmax_long = {Tmax_long * 1e6:.3f} us\n"
            f"  analysis starts at about {analysis_start_time * 1e9:.1f} ns\n"
            f"  retained record = {retained_duration * 1e9:.1f} ns\n"
            f"  fit_lag_ns ends at {fit_lag_ns[1]:.1f} ns\n"
            f"  maximum_fit_lag_fraction = {maximum_fit_lag_fraction:.3f}, "
            f"so the allowed fit lag is only "
            f"{allowed_fit_lag_max * 1e9:.1f} ns.\n"
            "Use a shorter fit_lag_ns upper bound, increase "
            "maximum_fit_lag_fraction, or set Tmax_long to at least "
            f"{required_tmax * 1e6:.3f} us for the current fit window."
        )


def estimate_long_output_gb(kappa_chunk_size=None):
    """Estimate returned y_long size for one vectorized long worker chunk."""
    if kappa_chunk_size is None:
        kappa_chunk_size = two_stage_kappa_chunk_size
    saved_steps = int(np.ceil(long_steps / int(max(1, analysis_save_every))))
    output_states = 2 * N_lasers
    n_cases_worker = int(max(1, kappa_chunk_size)) * int(max(1, n_noise_iterations))
    bytes_est = (
        n_cases_worker
        * output_states
        * saved_steps
        * np.dtype(analysis_output_dtype).itemsize
    )
    return bytes_est / 1e9


def estimate_effective_long_psd_sampling(kappa_chunk_size=None):
    """Estimate the PSD grid after the vcsel_lib output-size save_every cap."""
    if kappa_chunk_size is None:
        kappa_chunk_size = two_stage_kappa_chunk_size

    requested_save_every = int(max(1, analysis_save_every))
    kappa_chunk_size = int(max(1, kappa_chunk_size))
    n_cases_worker = kappa_chunk_size * int(max(1, n_noise_iterations))
    output_states = 2 * N_lasers

    # Match vcsel_lib.integrate's conservative cap calculation. It estimates
    # eight bytes per output value, which is exact for the float64 phase output
    # used here to keep a ~20 Hz^2/Hz floor above quantization noise.
    cap_bytes_per_saved_step = n_cases_worker * output_states * 8
    uncapped_cap_estimate_gb = cap_bytes_per_saved_step * long_steps / 1e9

    cap_save_every = 1
    if max_output_gb is not None:
        max_bytes = float(max_output_gb) * 1e9
        if max_bytes > 0.0:
            uncapped_bytes = cap_bytes_per_saved_step * long_steps
            if uncapped_bytes > max_bytes:
                cap_save_every = int(np.ceil(uncapped_bytes / max_bytes))

    effective_save_every = int(max(requested_save_every, cap_save_every))
    saved_dt = effective_save_every * dt
    saved_samples = int(np.ceil(long_steps / effective_save_every))
    analysis_start_time = long_run_analysis_start_time(Tmax_long)
    analysis_start_saved_index = min(
        saved_samples,
        int(np.ceil(analysis_start_time / saved_dt)),
    )
    retained_saved_samples = max(0, saved_samples - analysis_start_saved_index)
    frequency_samples = max(0, retained_saved_samples - 1)

    if frequency_samples >= 8:
        if psd_welch_nperseg is None:
            nperseg_use = frequency_samples
        else:
            nperseg_use = min(int(max(8, psd_welch_nperseg)), frequency_samples)
        nperseg_use = int(max(8, nperseg_use))
        first_positive_hz = 1.0 / (nperseg_use * saved_dt)
        nyquist_hz = 1.0 / (2.0 * saved_dt)
    else:
        nperseg_use = 0
        first_positive_hz = np.nan
        nyquist_hz = np.nan

    actual_output_gb = (
        n_cases_worker
        * output_states
        * saved_samples
        * np.dtype(analysis_output_dtype).itemsize
        / 1e9
    )

    return {
        "requested_save_every": requested_save_every,
        "cap_save_every": cap_save_every,
        "effective_save_every": effective_save_every,
        "saved_dt": saved_dt,
        "saved_samples": saved_samples,
        "analysis_start_time": analysis_start_time,
        "analysis_start_saved_index": analysis_start_saved_index,
        "retained_saved_samples": retained_saved_samples,
        "retained_duration": max(0.0, Tmax_long - analysis_start_time),
        "frequency_samples": frequency_samples,
        "welch_nperseg": nperseg_use,
        "first_positive_hz": first_positive_hz,
        "nyquist_hz": nyquist_hz,
        "uncapped_cap_estimate_gb": uncapped_cap_estimate_gb,
        "actual_output_gb": actual_output_gb,
        "n_cases_worker": n_cases_worker,
    }


def estimate_long_run_memory_gb(kappa_chunk_size=None):
    """Conservatively estimate solver, analysis, and parent-map RAM usage."""
    summary = estimate_effective_long_psd_sampling(kappa_chunk_size)
    saved_samples = int(summary["saved_samples"])
    retained_saved_samples = int(summary["retained_saved_samples"])
    single_channel_bytes = (
        int(max(1, n_noise_iterations))
        * retained_saved_samples
        * np.dtype(np.float64).itemsize
    )

    # The sequential implementation normally needs fewer arrays than this,
    # but the allowance also covers unwrap, frequency, Welch, FFT, and
    # percentile temporaries that briefly overlap.
    analysis_workspace_bytes = 12 * single_channel_bytes
    worker_peak_gb = (
        float(summary["actual_output_gb"])
        + analysis_workspace_bytes / 1e9
    )
    active_worker_count = int(max(1, long_run_jobs))
    all_active_workers_peak_gb = (
        float(summary["actual_output_gb"]) * active_worker_count
        + analysis_workspace_bytes
        / 1e9
        * (1 if serialize_long_run_postprocessing else active_worker_count)
    )

    psd_bins = (
        int(summary["welch_nperseg"]) // 2 + 1
        if int(summary["welch_nperseg"]) >= 2
        else 0
    )
    if phase_plot_tail_window_us is None:
        stored_phase_samples = retained_saved_samples
    else:
        stored_phase_samples = min(
            retained_saved_samples,
            int(
                np.ceil(
                    float(phase_plot_tail_window_us)
                    * 1e-6
                    / float(summary["saved_dt"])
                )
            )
            + 1,
        )
    parent_map_bytes = (
        int(max(1, n_kappa_steps))
        * (psd_bins + stored_phase_samples)
        * np.dtype(np.float32).itemsize
    )
    return {
        "worker_peak_gb": worker_peak_gb,
        "all_active_workers_peak_gb": all_active_workers_peak_gb,
        "parent_map_gb": parent_map_bytes / 1e9,
        "analysis_workspace_gb": analysis_workspace_bytes / 1e9,
        "postprocessing_serialized": bool(serialize_long_run_postprocessing),
    }


def physical_memory_gb():
    """Return installed physical RAM when the platform exposes it."""
    try:
        return (
            float(os.sysconf("SC_PHYS_PAGES"))
            * float(os.sysconf("SC_PAGE_SIZE"))
            / 1e9
        )
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return np.nan


def _format_hz_for_message(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return "unavailable"
    if value >= 1e9:
        return f"{value / 1e9:.4g} GHz"
    if value >= 1e6:
        return f"{value / 1e6:.4g} MHz"
    if value >= 1e3:
        return f"{value / 1e3:.4g} kHz"
    return f"{value:.4g} Hz"


def print_expected_psd_frequency_range():
    """Print the effective PSD frequency range for the current long-run setup."""
    summary = estimate_effective_long_psd_sampling()
    memory = estimate_long_run_memory_gb()
    save_every_note = (
        "unchanged"
        if summary["effective_save_every"] == summary["requested_save_every"]
        else (
            f"raised from {summary['requested_save_every']} to "
            f"{summary['effective_save_every']} by max_output_gb={max_output_gb:g}"
        )
    )
    print(
        "Expected full plotted long-run PSD frequency range: "
        f"{_format_hz_for_message(summary['first_positive_hz'])} to "
        f"{_format_hz_for_message(summary['nyquist_hz'])} "
        f"(save_every={summary['effective_save_every']}, {save_every_note}; "
        f"Welch nperseg={summary['welch_nperseg']}, "
        f"saved dt={summary['saved_dt'] * 1e12:.3g} ps; "
        f"analysis cutoff={summary['analysis_start_time'] * 1e6:.3g} us; "
        f"retained={summary['retained_duration'] * 1e6:.3g} us)."
    )
    print(
        "Long worker output estimate: "
        f"~{summary['actual_output_gb']:.2f} GB returned per "
        f"{two_stage_kappa_chunk_size}-kappa chunk "
        f"({summary['uncapped_cap_estimate_gb']:.2f} GB uncapped estimate "
        "used by vcsel_lib)."
    )
    print(
        "Conservative RAM estimate after sequential PSD processing: "
        f"~{memory['worker_peak_gb']:.2f} GB peak per active long worker "
        f"(~{memory['all_active_workers_peak_gb']:.2f} GB for "
        f"long_run_jobs={int(max(1, long_run_jobs))}), plus "
        f"~{memory['parent_map_gb']:.2f} GB for parent PSD/phase maps. "
        f"Chunk post-processing is "
        f"{'serialized' if serialize_long_run_postprocessing else 'concurrent'}. "
        "This excludes solver-internal overhead, Python, Matplotlib, and macOS."
    )
    installed_memory_gb = physical_memory_gb()
    estimated_active_gb = (
        memory["all_active_workers_peak_gb"] + memory["parent_map_gb"]
    )
    if (
        np.isfinite(installed_memory_gb)
        and installed_memory_gb > 0.0
        and estimated_active_gb > 0.8 * installed_memory_gb
    ):
        warnings.warn(
            "The conservative active-worker RAM estimate "
            f"({estimated_active_gb:.2f} GB) exceeds 80% of detected physical "
            f"RAM ({installed_memory_gb:.2f} GB). Vectorized chunk size remains "
            f"{int(max(1, two_stage_kappa_chunk_size))} as configured. If the "
            "machine still experiences memory pressure, reduce long_run_jobs; "
            "that lowers concurrent RAM without changing PSD resolution or "
            "the number of kappa values vectorized in each chunk.",
            RuntimeWarning,
            stacklevel=2,
        )


def adjust_psd_floor_band_before_workers():
    """Resolve the floor window once so every forked worker uses one band."""
    summary = estimate_effective_long_psd_sampling()
    nperseg = int(summary["welch_nperseg"])
    if nperseg < 2:
        return
    frequency_hz = np.fft.rfftfreq(nperseg, d=float(summary["saved_dt"]))
    ensure_resolvable_psd_floor_band(frequency_hz, minimum_bins=3)


def analyze_phase_variance_summary(t, y):
    """Return collective phase-diffusion results for one fixed-kappa run."""
    n_cases = y.shape[0]
    # Keep views of the solver output.  Consumers clip photon numbers as
    # needed, avoiding a full-size duplicate of every photon trajectory.
    S = y[:, 0::2, :]
    phi = y[:, 1::2, :]

    analysis_start = analysis_start_index_for_record(
        t,
        already_steady=long_run_history_is_steady_state,
    )
    if analysis_start >= t.size - 3:
        raise ValueError(
            "No stationary analysis window remains after the configured burn-in/"
            "ramp/settling interval. Increase Tmax_long or reduce "
            "long_run_burn_in_time or post_ramp_settling_ns."
        )

    t_analysis = t[analysis_start:]
    S_analysis = S[:, :, analysis_start:]
    phi_analysis = phi[:, :, analysis_start:]
    wrapped_phase_time_full = t_analysis - t_analysis[0]
    phase_plot_start_index = 0
    if phase_plot_tail_window_us is not None and wrapped_phase_time_full.size > 1:
        phase_plot_start_time = (
            wrapped_phase_time_full[-1]
            - float(phase_plot_tail_window_us) * 1e-6
        )
        phase_plot_start_index = int(
            np.searchsorted(
                wrapped_phase_time_full,
                phase_plot_start_time,
                side="left",
            )
        )
    wrapped_phase_time = wrapped_phase_time_full[phase_plot_start_index:]
    wrapped_phase_abs = average_wrapped_phase_trace_from_phi(
        phi_analysis[:, :, phase_plot_start_index:]
    )
    del wrapped_phase_time_full
    order_param_cases = order_parameter_from_s_phi(S_analysis, phi_analysis)

    collective_phase = None
    if compute_phase_variance_linewidth:
        # A synchronized array has one common phase even when its locked
        # spatial pattern is out of phase and the summed field nearly
        # cancels. Averaging individually unwrapped phases extracts that
        # common coordinate without total-field phase singularities.
        collective_phase = np.zeros(
            (phi_analysis.shape[0], phi_analysis.shape[-1]),
            dtype=np.float64,
        )
        for case_index in range(phi_analysis.shape[0]):
            for laser_index in range(phi_analysis.shape[1]):
                collective_phase[case_index] += np.unwrap(
                    np.asarray(
                        phi_analysis[case_index, laser_index, :],
                        dtype=np.float64,
                    )
                )
        collective_phase /= float(phi_analysis.shape[1])

        linewidth_result = phase_diffusion_linewidth(
            t_analysis,
            collective_phase,
            lag_range_ns=fit_lag_ns,
            max_lag_ns=plot_lag_max_ns,
            n_lags=n_lag_points,
            maximum_fit_fraction=maximum_fit_lag_fraction,
        )
    else:
        linewidth_result = empty_phase_diffusion_result(t_analysis, n_cases)
    psd_result = frequency_noise_psd_linewidth(t_analysis, S_analysis, phi_analysis)
    psd_channels = psd_result["channels"]
    psd_frequency_hz = np.asarray(psd_result["frequencies_hz"], dtype=float)

    def median_psd_trace(channel_name):
        trace = np.asarray(
            psd_channels.get(channel_name, {}).get("psd", []),
            dtype=float,
        )
        if trace.shape == psd_frequency_hz.shape:
            return trace
        padded = np.full(psd_frequency_hz.shape, np.nan, dtype=float)
        n_copy = min(padded.size, trace.size)
        if n_copy:
            padded[:n_copy] = trace[:n_copy]
        return padded

    psd_laser_linewidth_hz = np.full(N_lasers, np.nan, dtype=np.float32)
    psd_laser_linewidth_mean_hz = np.full(N_lasers, np.nan, dtype=np.float32)
    psd_laser_linewidth_std_hz = np.full(N_lasers, np.nan, dtype=np.float32)
    psd_laser_linewidth_q16_hz = np.full(N_lasers, np.nan, dtype=np.float32)
    psd_laser_linewidth_q84_hz = np.full(N_lasers, np.nan, dtype=np.float32)
    psd_laser_traces = np.full(
        (N_lasers, psd_frequency_hz.size),
        np.nan,
        dtype=np.float32,
    )
    for laser_index in range(N_lasers):
        channel = psd_channels.get(f"laser{laser_index + 1}", {})
        psd_laser_linewidth_hz[laser_index] = float(channel.get("linewidth_hz", np.nan))
        psd_laser_linewidth_mean_hz[laser_index] = float(
            channel.get("linewidth_mean_hz", np.nan)
        )
        psd_laser_linewidth_std_hz[laser_index] = float(
            channel.get("linewidth_std_hz", np.nan)
        )
        psd_laser_linewidth_q16_hz[laser_index] = float(
            channel.get("linewidth_q16_hz", np.nan)
        )
        psd_laser_linewidth_q84_hz[laser_index] = float(
            channel.get("linewidth_q84_hz", np.nan)
        )
        psd_laser_traces[laser_index] = median_psd_trace(
            f"laser{laser_index + 1}"
        ).astype(np.float32, copy=False)

    if N_lasers > 1:
        relative_phase_slip_counts = np.zeros(n_cases, dtype=int)
        for laser_index in range(1, N_lasers):
            wrapped_relative_phase = (
                phi_analysis[:, laser_index, :]
                - phi_analysis[:, 0, :]
                + np.pi
            )
            np.remainder(
                wrapped_relative_phase,
                2.0 * np.pi,
                out=wrapped_relative_phase,
            )
            wrapped_relative_phase -= np.pi
            unwrapped_relative_phase = np.unwrap(
                wrapped_relative_phase,
                axis=-1,
            )
            unwrapped_relative_phase -= unwrapped_relative_phase[:, 0:1]
            relative_phase_steps = np.diff(unwrapped_relative_phase, axis=-1)
            np.abs(relative_phase_steps, out=relative_phase_steps)
            relative_phase_slip_counts += np.sum(
                relative_phase_steps > phase_slip_step_threshold_rad,
                axis=1,
            )
            del (
                wrapped_relative_phase,
                unwrapped_relative_phase,
                relative_phase_steps,
            )
        relative_phase_slip_cases = relative_phase_slip_counts > 0
    else:
        relative_phase_slip_counts = np.zeros(n_cases, dtype=int)
        relative_phase_slip_cases = np.zeros(n_cases, dtype=bool)

    summary = {
        "lag_times": linewidth_result["lag_times"].astype(np.float32, copy=False),
        "phase_variance": linewidth_result["phase_variance"].astype(np.float32, copy=False),
        "phase_variance_sem": linewidth_result["phase_variance_sem"].astype(np.float32, copy=False),
        "fit_time": linewidth_result["fit_time"].astype(np.float32, copy=False),
        "fit_line": linewidth_result["fit_line"].astype(np.float32, copy=False),
        "wrapped_phase_time": wrapped_phase_time.astype(np.float32, copy=False),
        "wrapped_phase_abs": wrapped_phase_abs.astype(np.float32, copy=False),
        "order_parameter": float(np.mean(order_param_cases)),
        "order_parameter_min": float(np.min(order_param_cases)),
        "order_parameter_max": float(np.max(order_param_cases)),
        "linewidth_hz": float(linewidth_result["linewidth_hz"]),
        "linewidth_median_hz": float(linewidth_result["linewidth_median_hz"]),
        "linewidth_std_hz": float(linewidth_result["linewidth_std_hz"]),
        "r_squared": float(linewidth_result["r_squared"]),
        "psd_frequency_hz": psd_frequency_hz.astype(np.float32, copy=False),
        "psd_heatmap_channel": str(psd_result["heatmap_channel"]),
        "psd_heatmap": psd_result["heatmap_psd"].astype(np.float32, copy=False),
        "psd_combined": median_psd_trace("combined").astype(np.float32, copy=False),
        "psd_common": median_psd_trace("common").astype(np.float32, copy=False),
        "psd_laser": psd_laser_traces,
        "psd_linewidth_combined_hz": float(
            psd_channels["combined"]["linewidth_hz"]
        ),
        "psd_linewidth_combined_mean_hz": float(
            psd_channels["combined"]["linewidth_mean_hz"]
        ),
        "psd_linewidth_combined_std_hz": float(
            psd_channels["combined"]["linewidth_std_hz"]
        ),
        "psd_linewidth_combined_q16_hz": float(
            psd_channels["combined"]["linewidth_q16_hz"]
        ),
        "psd_linewidth_combined_q84_hz": float(
            psd_channels["combined"]["linewidth_q84_hz"]
        ),
        "psd_linewidth_common_hz": float(
            psd_channels["common"]["linewidth_hz"]
        ),
        "psd_linewidth_common_mean_hz": float(
            psd_channels["common"]["linewidth_mean_hz"]
        ),
        "psd_linewidth_common_std_hz": float(
            psd_channels["common"]["linewidth_std_hz"]
        ),
        "psd_linewidth_common_q16_hz": float(
            psd_channels["common"]["linewidth_q16_hz"]
        ),
        "psd_linewidth_common_q84_hz": float(
            psd_channels["common"]["linewidth_q84_hz"]
        ),
        "psd_linewidth_laser_hz": psd_laser_linewidth_hz,
        "psd_linewidth_laser_mean_hz": psd_laser_linewidth_mean_hz,
        "psd_linewidth_laser_std_hz": psd_laser_linewidth_std_hz,
        "psd_linewidth_laser_q16_hz": psd_laser_linewidth_q16_hz,
        "psd_linewidth_laser_q84_hz": psd_laser_linewidth_q84_hz,
        "combined_field_power_mean": float(psd_result["combined_power_mean"]),
        "combined_field_power_min": float(psd_result["combined_power_min"]),
        "phase_slip_case_count": int(np.count_nonzero(linewidth_result["phase_slip_cases"])),
        "relative_phase_slip_case_count": int(np.count_nonzero(relative_phase_slip_cases)),
        "n_cases": int(n_cases),
    }
    del (
        S,
        phi,
        S_analysis,
        phi_analysis,
        collective_phase,
        wrapped_phase_abs,
        wrapped_phase_time,
        order_param_cases,
        linewidth_result,
        psd_result,
        psd_channels,
        psd_frequency_hz,
        psd_laser_linewidth_hz,
        psd_laser_linewidth_mean_hz,
        psd_laser_linewidth_std_hz,
        psd_laser_linewidth_q16_hz,
        psd_laser_linewidth_q84_hz,
        psd_laser_traces,
        relative_phase_slip_counts,
        relative_phase_slip_cases,
    )
    return summary


def plot_phase_variance_fit_result(result, kappa_start, kappa_stop, figure_path):
    """Save one compact collective phase-diffusion linewidth fit figure."""
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
        rf"$\kappa_c$: {kappa_start * 1e-9:.3f}"
        rf"$\rightarrow${kappa_stop * 1e-9:.3f} ns$^{{-1}}$; "
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


def _linewidth_mhz_text(linewidth_hz):
    if not np.isfinite(linewidth_hz) or linewidth_hz <= 0.0:
        return "nan"
    return f"{float(linewidth_hz) * 1e-6:.3g}"


def _format_frequency_hz_latex(frequency_hz):
    frequency_hz = float(frequency_hz)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return r"\mathrm{nan}"
    exponent = int(np.floor(np.log10(frequency_hz)))
    mantissa = frequency_hz / (10.0**exponent)
    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:g}\times10^{{{exponent}}}"


def _configure_psd_frequency_ticks(ax):
    """Show every frequency decade on PSD axes, even in narrow panels."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0,), numticks=100)
    )
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())


def _psd_plot_bounds_hz():
    """Return fallback PSD plot bounds in Hz.

    These bounds are only used if a PSD frequency grid is unavailable.  The
    PSD-floor band intentionally does not enter here: it should choose the
    averaging window for the linewidth estimate, not crop or expand the plot.
    """
    plot_min_ghz, plot_max_ghz = psd_plot_frequency_axis_ghz
    plot_min_hz = float(plot_min_ghz) * 1e9
    plot_max_hz = float(plot_max_ghz) * 1e9
    plot_min_hz = max(plot_min_hz, np.finfo(float).tiny)
    if not np.isfinite(plot_max_hz) or plot_max_hz <= plot_min_hz:
        plot_max_hz = plot_min_hz * 10.0
    return plot_min_hz, plot_max_hz


def _psd_plot_mask_and_bounds(frequency_hz):
    """Return PSD plot mask/bounds from the actually available grid.

    `psd_floor_band_hz` is deliberately not used here.  It only controls the
    PSD-floor averaging window and the shaded guide drawn on top of the plot.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    positive = np.isfinite(frequency_hz) & (frequency_hz > 0.0)
    selected = np.flatnonzero(positive)
    if selected.size:
        plot_min_hz = float(np.nanmin(frequency_hz[selected]))
        plot_max_hz = float(np.nanmax(frequency_hz[selected]))
        if not np.isfinite(plot_max_hz) or plot_max_hz <= plot_min_hz:
            plot_max_hz = plot_min_hz * 10.0
        return positive, plot_min_hz, plot_max_hz

    plot_min_hz, plot_max_hz = _psd_plot_bounds_hz()
    return positive, plot_min_hz, plot_max_hz


def _log_spaced_psd_plot_indices(frequency_hz, plot_mask, max_points=None):
    """Select display-only PSD samples while retaining the complete range."""
    frequency_hz = np.asarray(frequency_hz)
    selected = np.flatnonzero(np.asarray(plot_mask, dtype=bool))
    if max_points is None:
        max_points = max_plot_points
    max_points = int(max(2, max_points))
    if selected.size <= max_points:
        return selected

    positive_frequency = np.asarray(frequency_hz[selected], dtype=np.float64)
    targets = np.geomspace(
        positive_frequency[0],
        positive_frequency[-1],
        max_points,
    )
    positions = np.searchsorted(positive_frequency, targets, side="left")
    positions = np.clip(positions, 0, selected.size - 1)
    # Include both endpoints explicitly so downsampling never crops the range.
    positions = np.unique(
        np.concatenate((np.asarray([0]), positions, np.asarray([selected.size - 1])))
    )
    return selected[positions]


def psd_available_frequency_range_hz(frequency_hz):
    """Return the full positive PSD range represented by a frequency grid."""
    plot_mask, plot_min_hz, plot_max_hz = _psd_plot_mask_and_bounds(frequency_hz)
    if np.count_nonzero(plot_mask) < 2:
        return np.asarray([np.nan, np.nan], dtype=float)
    return np.asarray([plot_min_hz, plot_max_hz], dtype=float)


def plot_frequency_noise_psd_result(result, kappa_start, kappa_stop, figure_path):
    """Plot the full positive PSD range for one fixed-kappa run.

    The floor band is drawn only as a guide and is never used as a plot mask.
    Display traces are log-downsampled to control plotting RAM; the linewidth
    calculation and stored PSD continue to use every available Welch bin.
    """
    frequency_hz = np.asarray(result.get("psd_frequency_hz", []), dtype=float)
    floor_min_hz, floor_max_hz = [float(value) for value in psd_floor_band_hz]
    plot_mask, plot_min_hz, plot_max_hz = _psd_plot_mask_and_bounds(frequency_hz)
    plot_indices = _log_spaced_psd_plot_indices(frequency_hz, plot_mask)
    frequency_plot_hz = frequency_hz[plot_indices]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=figure_dpi)
    finite_psd_values = []

    def plot_psd_trace(key, label, linewidth_key, color, linestyle="-", linewidth=1.8):
        trace = np.asarray(result.get(key, []))
        if trace.shape != frequency_hz.shape or plot_indices.size < 2:
            return
        trace_plot = np.maximum(
            np.asarray(trace[plot_indices], dtype=float),
            np.finfo(float).tiny,
        )
        finite = np.isfinite(trace_plot) & (trace_plot > 0.0)
        if not np.any(finite):
            return
        finite_psd_values.append(trace_plot[finite])
        linewidth_text = _linewidth_mhz_text(float(result.get(linewidth_key, np.nan)))
        ax.plot(
            frequency_plot_hz,
            trace_plot,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=rf"{label}, $\Delta\nu={linewidth_text}$ MHz",
        )

    plot_psd_trace(
        "psd_combined",
        "combined field",
        "psd_linewidth_combined_hz",
        "black",
        linewidth=2.4,
    )
    plot_psd_trace(
        "psd_common",
        "mean laser phase",
        "psd_linewidth_common_hz",
        "0.35",
        linestyle="--",
        linewidth=1.8,
    )

    psd_laser = np.asarray(result.get("psd_laser", []), dtype=float)
    laser_linewidths = np.asarray(
        result.get("psd_linewidth_laser_hz", []),
        dtype=float,
    )
    if psd_laser.ndim == 2 and plot_indices.size >= 2:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for laser_index in range(min(N_lasers, psd_laser.shape[0])):
            trace = psd_laser[laser_index]
            if trace.shape != frequency_hz.shape:
                continue
            trace_plot = np.maximum(
                np.asarray(trace[plot_indices], dtype=float),
                np.finfo(float).tiny,
            )
            finite = np.isfinite(trace_plot) & (trace_plot > 0.0)
            if not np.any(finite):
                continue
            finite_psd_values.append(trace_plot[finite])
            linewidth_hz = (
                laser_linewidths[laser_index]
                if laser_index < laser_linewidths.size
                else np.nan
            )
            linewidth_text = _linewidth_mhz_text(float(linewidth_hz))
            ax.plot(
                frequency_plot_hz,
                trace_plot,
                color=colors[laser_index % len(colors)],
                linewidth=1.3,
                alpha=0.9,
                label=rf"laser {laser_index + 1}, $\Delta\nu={linewidth_text}$ MHz",
            )

    span_min = max(floor_min_hz, plot_min_hz)
    span_max = min(floor_max_hz, plot_max_hz)
    if span_max > span_min:
        ax.axvspan(
            span_min,
            span_max,
            color="gray",
            alpha=0.14,
            label="PSD-floor band",
        )

    if plot_indices.size >= 2:
        _configure_psd_frequency_ticks(ax)
        ax.set_yscale("log")
        ax.set_xlim(plot_min_hz, plot_max_hz)
    else:
        ax.text(
            0.5,
            0.5,
            "PSD record too short for this kappa",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fixed_ylim = psd_frame_ylim_hz2_per_hz
    if fixed_ylim is not None:
        ymin, ymax = [float(value) for value in fixed_ylim]
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0.0 and ymax > ymin:
            ax.set_ylim(ymin, ymax)
    elif finite_psd_values:
        finite_psd = np.concatenate(finite_psd_values)
        ymin = np.nanpercentile(finite_psd, 1.0)
        ymax = np.nanpercentile(finite_psd, 99.0)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            ax.set_ylim(ymin / 1.5, ymax * 1.5)

    if psd_frame_yticks_hz2_per_hz is not None:
        ytick_values = [
            float(value)
            for value in psd_frame_yticks_hz2_per_hz
            if np.isfinite(float(value)) and float(value) > 0.0
        ]
        if ytick_values:
            ax.set_yticks(ytick_values)
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_title(
        rf"$\kappa_c$: {kappa_start * 1e-9:.3f}"
        rf"$\rightarrow${kappa_stop * 1e-9:.3f} ns$^{{-1}}$; "
        rf"PSD floor band ${_format_frequency_hz_latex(floor_min_hz)}$--"
        rf"${_format_frequency_hz_latex(floor_max_hz)}$ Hz"
    )
    ax.set_xlabel(r"Frequency (Hz)")
    ax.set_ylabel(r"PSD of FM noise (Hz$^2$/Hz)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.35, alpha=0.15)
    ax.legend(fontsize=7, loc="lower right")

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_variance_continuation_frame(
    figure_path,
    kappa_values,
    wrapped_phase_map,
    wrapped_phase_time,
    phase_variance_map,
    lag_times,
    linewidth_mean_hz,
    linewidth_median_hz,
    psd_frequency_hz=None,
    psd_map=None,
    psd_linewidth_combined_hz=None,
    psd_linewidth_common_hz=None,
    psd_linewidth_laser_hz=None,
    psd_linewidth_combined_std_hz=None,
    psd_linewidth_common_std_hz=None,
    psd_linewidth_laser_std_hz=None,
    psd_linewidth_combined_q16_hz=None,
    psd_linewidth_combined_q84_hz=None,
    psd_linewidth_common_q16_hz=None,
    psd_linewidth_common_q84_hz=None,
    psd_linewidth_laser_q16_hz=None,
    psd_linewidth_laser_q84_hz=None,
    psd_heatmap_channel_label=None,
    order_param=None,
    order_param_min=None,
    order_param_max=None,
    linewidth_r_squared=None,
    henry_kappa0_reference_hz=None,
    henry_single_laser_reference_hz=None,
    completed_kappa_indices=None,
):
    """Save an optical-spectrum-style collective phase-diffusion summary."""
    if phase_variance_map is None or lag_times is None:
        return

    kappa_axis = np.asarray(kappa_values, dtype=float) * 1e-9
    lag_ns = np.asarray(lag_times, dtype=float) * 1e9
    lag_axis_min_ns, lag_axis_max_ns = phase_variance_lag_axis_ns
    font_size = int(continuation_frame_font_size)
    title_pad = float(continuation_frame_title_pad)

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

    phase_overlay_by_kappa = None
    if wrapped_phase_map is not None and wrapped_phase_time is not None:
        phase_time_seconds = np.asarray(wrapped_phase_time, dtype=float)
        phase_plot_map = np.asarray(wrapped_phase_map)
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
        if phase_plot_map.shape[0] == kappa_axis.size:
            finite_phase = np.isfinite(phase_plot_map)
            finite_phase_counts = np.sum(finite_phase, axis=1)
            phase_overlay_by_kappa = np.full(
                phase_plot_map.shape[0],
                np.nan,
                dtype=float,
            )
            np.divide(
                np.nansum(phase_plot_map, axis=1),
                finite_phase_counts,
                out=phase_overlay_by_kappa,
                where=finite_phase_counts > 0,
            )
        if phase_time_seconds[-1] >= 1e-6:
            phase_time = phase_time_seconds * 1e6
            phase_time_label = r"Analysis time ($\mu$s)"
        else:
            phase_time = phase_time_seconds * 1e9
            phase_time_label = r"Analysis time (ns)"

        ax0 = fig.add_subplot(gs[1:8, 0:-3])
        im0 = ax0.imshow(
            np.ma.masked_invalid(phase_plot_map),
            aspect="auto",
            extent=[
                phase_time[0],
                phase_time[-1],
                kappa_axis[0],
                kappa_axis[-1],
            ],
            origin="lower",
            cmap="jet_r",
            vmin=0.0,
            vmax=np.pi,
            rasterized=True,
        )
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.02)
        cbar0.set_label(
            r"$|\langle\Delta\phi\rangle_{\rm circ}|$ (rad)",
            fontsize=font_size,
            labelpad=0,
        )
        cbar0.set_ticks([0.0, 0.5 * np.pi, np.pi])
        cbar0.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
        cbar0.ax.tick_params(labelsize=font_size)
        ax0.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size)
        ax0.set_xlabel(phase_time_label, fontsize=font_size)
        ax0.set_xlim(phase_time[0], phase_time[-1])
        ax0.set_xticks(np.linspace(phase_time[0], phase_time[-1], 6))
        detuning_distribution_ghz = np.asarray(detuning_distribution, dtype=float) / (
            2.0 * np.pi * 1e9
        )
        ax0.set_title(
            r"$\phi_p={:+.2f}\pi,\ \delta_i=[{}]\,\mathrm{{GHz}},\ \Delta t={:.1f}\tau_p$".format(
                float(phi_p) / np.pi,
                ", ".join(f"{d:.1f}" for d in detuning_distribution_ghz),
                dt / tau_p,
            ),
            fontsize=font_size,
            pad=title_pad,
        )
        ax0.set_yticks(np.linspace(kappa_axis[0], kappa_axis[-1], 6))
        ax0.tick_params(axis="both", labelsize=font_size)

        if order_param is not None:
            ax_order = fig.add_subplot(gs[1:8, -2:])
            plot_order_parameter_panel(
                ax_order,
                order_param,
                kappa_values,
                font_size,
                pad=title_pad,
                order_param_min=order_param_min,
                order_param_max=order_param_max,
            )

    ax1 = fig.add_subplot(gs[11:, 1:8])
    psd_available = (
        psd_frequency_hz is not None
        and psd_map is not None
        and np.asarray(psd_map).ndim == 2
        and np.count_nonzero(
            np.isfinite(np.asarray(psd_frequency_hz, dtype=float))
            & (np.asarray(psd_frequency_hz, dtype=float) > 0.0)
        )
        >= 2
    )
    if psd_available:
        frequency_hz = np.asarray(psd_frequency_hz, dtype=float)
        plot_mask, plot_min_hz, plot_max_hz = _psd_plot_mask_and_bounds(frequency_hz)
        if np.count_nonzero(plot_mask) < 2:
            psd_available = False
    if psd_available:
        plot_indices = _log_spaced_psd_plot_indices(frequency_hz, plot_mask)
        frequency_plot_hz = frequency_hz[plot_indices]
        # Slice the float32 map before converting it for Matplotlib.  This
        # keeps plot RAM proportional to max_plot_points while the stored map
        # and linewidth estimator retain every Welch bin.
        psd_plot = np.asarray(
            np.asarray(psd_map)[:, plot_indices],
            dtype=float,
        )
        psd_plot[psd_plot <= 0.0] = np.nan
        finite_psd = psd_plot[np.isfinite(psd_plot) & (psd_plot > 0.0)]
        if finite_psd.size:
            if psd_colorbar_vmin_hz2_per_hz is None:
                psd_vmin = np.nanpercentile(
                    finite_psd,
                    psd_colorbar_vmin_percentile,
                )
            else:
                psd_vmin = float(psd_colorbar_vmin_hz2_per_hz)
            if psd_colorbar_vmax_hz2_per_hz is None:
                psd_vmax = np.nanpercentile(
                    finite_psd,
                    psd_colorbar_vmax_percentile,
                )
            else:
                psd_vmax = float(psd_colorbar_vmax_hz2_per_hz)
            if not np.isfinite(psd_vmin) or psd_vmin <= 0.0:
                psd_vmin = np.nanmin(finite_psd)
            if not np.isfinite(psd_vmax) or psd_vmax <= psd_vmin:
                psd_vmax = max(psd_vmin * 10.0, np.nanmax(finite_psd))
            if psd_vmax <= psd_vmin:
                psd_vmax = psd_vmin * 10.0
        else:
            psd_vmin, psd_vmax = 1.0e-6, 1.0
        kappa_edges = center_edges_from_centers(kappa_axis, log=False)
        frequency_edges = center_edges_from_centers(frequency_plot_hz, log=True)
        psd_cmap = plt.get_cmap("jet").copy()
        psd_cmap.set_bad(color="white")
        im1 = ax1.pcolormesh(
            frequency_edges,
            kappa_edges,
            np.ma.masked_invalid(psd_plot),
            shading="auto",
            cmap=psd_cmap,
            norm=LogNorm(vmin=psd_vmin, vmax=psd_vmax),
            rasterized=True,
        )
        cbar1 = fig.colorbar(im1, ax=ax1, pad=0.045)
        cbar1.set_label(FREQUENCY_PSD_LABEL, fontsize=font_size, labelpad=4)
        cbar1.ax.yaxis.set_major_locator(LogLocator(base=10))
        cbar1.ax.tick_params(labelsize=font_size)
        _configure_psd_frequency_ticks(ax1)
        ax1.set_xlabel(r"Frequency (Hz)", fontsize=font_size, labelpad=10)
        ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
        channel_label = psd_heatmap_channel_label or psd_heatmap_channel
        ax1.set_title(
            rf"PSD of ${channel_label}$ frequency noise",
            fontsize=font_size,
            pad=title_pad,
        )
        ax1.set_xlim(plot_min_hz, plot_max_hz)
        ax1.set_ylim(kappa_axis[0], kappa_axis[-1])
        ax1.set_yticks(np.linspace(kappa_axis[0], kappa_axis[-1], 6))
        floor_min_hz, floor_max_hz = [float(value) for value in psd_floor_band_hz]
        for floor_edge_hz in (floor_min_hz, floor_max_hz):
            if ax1.get_xlim()[0] <= floor_edge_hz <= ax1.get_xlim()[1]:
                ax1.axvline(
                    floor_edge_hz,
                    color="black",
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.55,
                )
        ax1.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.2)
        ax1.grid(True, which="minor", linestyle="--", linewidth=0.3, alpha=0.12)
        ax1.tick_params(axis="both", labelsize=font_size)
    else:
        variance_cmap = plt.get_cmap("jet").copy()
        variance_cmap.set_bad(color="white")
        variance_values = np.asarray(phase_variance_map, dtype=float)
        finite_positive_variance = variance_values[
            np.isfinite(variance_values) & (variance_values > 0.0)
        ]
        variance_norm = None
        colorbar_tick_values = None
        if phase_diffusion_colorbar_ticks is not None:
            colorbar_tick_values = np.unique(
                np.asarray(phase_diffusion_colorbar_ticks, dtype=float)
            )
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
            else:
                variance_vmin = float(phase_variance_log_vmin_rad2)
            if colorbar_tick_values is None or colorbar_tick_values.size < 2:
                if phase_variance_log_vmax_rad2 is None:
                    variance_vmax = np.nanpercentile(
                        finite_positive_variance,
                        phase_variance_log_vmax_percentile,
                    )
                else:
                    variance_vmax = float(phase_variance_log_vmax_rad2)
                variance_vmin = max(
                    variance_vmin,
                    np.nanmin(finite_positive_variance),
                    np.finfo(float).tiny,
                )
                variance_vmax = max(variance_vmax, variance_vmin * 10.0)
            else:
                variance_vmin = max(variance_vmin, np.finfo(float).tiny)
                variance_vmax = max(variance_vmax, variance_vmin * 1.001)
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
        ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
        ax1.set_title(
            PHASE_VARIANCE_LABEL,
            fontsize=font_size,
            pad=title_pad,
        )
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
    completed_mask = np.zeros(kappa_axis.size, dtype=bool)
    if completed_kappa_indices is None:
        completed_mask[:] = True
    else:
        completed_indices = np.asarray(sorted(completed_kappa_indices), dtype=int)
        completed_indices = completed_indices[
            (completed_indices >= 0) & (completed_indices < completed_mask.size)
        ]
        completed_mask[completed_indices] = True

    if (
        phase_overlay_by_kappa is not None
        and np.asarray(phase_overlay_by_kappa).shape[0] == kappa_axis.size
    ):
        phase_overlay = np.asarray(phase_overlay_by_kappa, dtype=float)
        if kappa_axis.size > 1:
            kappa_edges = np.empty(kappa_axis.size + 1, dtype=float)
            kappa_edges[1:-1] = 0.5 * (kappa_axis[:-1] + kappa_axis[1:])
            kappa_edges[0] = kappa_axis[0] - (kappa_edges[1] - kappa_axis[0])
            kappa_edges[-1] = kappa_axis[-1] + (kappa_axis[-1] - kappa_edges[-2])
        else:
            kappa_edges = np.asarray([kappa_axis[0] - 0.5, kappa_axis[0] + 0.5])
        phase_cmap = plt.get_cmap("jet_r")
        phase_norm = Normalize(vmin=0.0, vmax=np.pi)
        for kappa_index, phase_value in enumerate(phase_overlay):
            if not completed_mask[kappa_index] or not np.isfinite(phase_value):
                continue
            ax2.axvspan(
                kappa_edges[kappa_index],
                kappa_edges[kappa_index + 1],
                color=phase_cmap(phase_norm(np.clip(phase_value, 0.0, np.pi))),
                alpha=0.13,
                linewidth=0,
                zorder=0,
            )

    psd_linewidth_available = psd_linewidth_combined_hz is not None
    if psd_linewidth_available:
        combined_mhz = np.asarray(psd_linewidth_combined_hz, dtype=float) * 1e-6
        combined_q16_mhz = (
            np.asarray(psd_linewidth_combined_q16_hz, dtype=float) * 1e-6
            if psd_linewidth_combined_q16_hz is not None else None
        )
        combined_q84_mhz = (
            np.asarray(psd_linewidth_combined_q84_hz, dtype=float) * 1e-6
            if psd_linewidth_combined_q84_hz is not None else None
        )
        valid_combined = completed_mask & np.isfinite(combined_mhz) & (combined_mhz > 0.0)
        if np.any(valid_combined):
            if combined_q16_mhz is not None and combined_q84_mhz is not None:
                lower = np.maximum(combined_q16_mhz, np.finfo(float).tiny)
                upper = combined_q84_mhz
                fill_valid = valid_combined & np.isfinite(lower) & np.isfinite(upper)
                ax2.fill_between(
                    kappa_axis[fill_valid],
                    lower[fill_valid],
                    upper[fill_valid],
                    color="black",
                    alpha=0.10,
                    linewidth=0,
                )
            ax2.plot(
                kappa_axis[valid_combined],
                combined_mhz[valid_combined],
                color="black",
                linewidth=2.6,
                marker="o",
                markersize=3.5,
                label=r"combined field PSD floor",
            )

        if psd_linewidth_common_hz is not None:
            common_mhz = np.asarray(psd_linewidth_common_hz, dtype=float) * 1e-6
            common_q16_mhz = (
                np.asarray(psd_linewidth_common_q16_hz, dtype=float) * 1e-6
                if psd_linewidth_common_q16_hz is not None else None
            )
            common_q84_mhz = (
                np.asarray(psd_linewidth_common_q84_hz, dtype=float) * 1e-6
                if psd_linewidth_common_q84_hz is not None else None
            )
            valid_common = completed_mask & np.isfinite(common_mhz) & (common_mhz > 0.0)
            if np.any(valid_common):
                if common_q16_mhz is not None and common_q84_mhz is not None:
                    lower = np.maximum(common_q16_mhz, np.finfo(float).tiny)
                    upper = common_q84_mhz
                    fill_valid = valid_common & np.isfinite(lower) & np.isfinite(upper)
                    ax2.fill_between(
                        kappa_axis[fill_valid],
                        lower[fill_valid],
                        upper[fill_valid],
                        color="0.35",
                        alpha=0.08,
                        linewidth=0,
                    )
                ax2.plot(
                    kappa_axis[valid_common],
                    common_mhz[valid_common],
                    color="0.35",
                    linestyle="--",
                    linewidth=1.8,
                    label=r"mean laser phase PSD floor",
                )

        if psd_linewidth_laser_hz is not None:
            laser_linewidths = np.asarray(psd_linewidth_laser_hz, dtype=float)
            laser_q16 = (
                np.asarray(psd_linewidth_laser_q16_hz, dtype=float)
                if psd_linewidth_laser_q16_hz is not None else None
            )
            laser_q84 = (
                np.asarray(psd_linewidth_laser_q84_hz, dtype=float)
                if psd_linewidth_laser_q84_hz is not None else None
            )
            if laser_linewidths.ndim == 2:
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                for laser_index in range(min(N_lasers, laser_linewidths.shape[0])):
                    laser_mhz = laser_linewidths[laser_index] * 1e-6
                    valid_laser = (
                        completed_mask & np.isfinite(laser_mhz) & (laser_mhz > 0.0)
                    )
                    if not np.any(valid_laser):
                        continue
                    color = colors[laser_index % len(colors)]
                    if (
                        laser_q16 is not None and laser_q84 is not None
                        and laser_q16.shape == laser_linewidths.shape
                        and laser_q84.shape == laser_linewidths.shape
                    ):
                        lower = np.maximum(
                            laser_q16[laser_index] * 1e-6,
                            np.finfo(float).tiny,
                        )
                        upper = laser_q84[laser_index] * 1e-6
                        fill_valid = valid_laser & np.isfinite(lower) & np.isfinite(upper)
                        ax2.fill_between(
                            kappa_axis[fill_valid],
                            lower[fill_valid],
                            upper[fill_valid],
                            color=color,
                            alpha=0.08,
                            linewidth=0,
                        )
                    ax2.plot(
                        kappa_axis[valid_laser],
                        laser_mhz[valid_laser],
                        color=color,
                        linewidth=1.8,
                        marker="s",
                        markersize=3.0,
                        alpha=0.85,
                        label=rf"laser {laser_index + 1} PSD floor",
                    )

        if plot_phase_variance_linewidth_comparison:
            median_mhz = np.asarray(linewidth_median_hz, dtype=float) * 1e-6
            valid_median = completed_mask & np.isfinite(median_mhz) & (median_mhz > 0.0)
            if linewidth_r_squared is not None:
                valid_median &= (
                    np.asarray(linewidth_r_squared, dtype=float)
                    >= float(minimum_phase_diffusion_r_squared_for_plot)
                )
            if np.any(valid_median):
                ax2.plot(
                    kappa_axis[valid_median],
                    median_mhz[valid_median],
                    color="0.65",
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.75,
                    label=r"phase-diffusion fit",
                )
    else:
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
    if (
        plot_kappa0_henry_reference
        and henry_single_laser_reference_hz is not None
        and np.isfinite(henry_single_laser_reference_hz)
        and henry_single_laser_reference_hz > 0.0
        and kappa_axis[0] <= 0.0 <= kappa_axis[-1]
    ):
        ax2.scatter(
            [0.0],
            [float(henry_single_laser_reference_hz) * 1e-6],
            s=190,
            color="red",
            edgecolor="black",
            linewidth=1.1,
            zorder=8,
            label=r"single-laser Henry $\kappa_c=0$",
        )
    if (
        use_ma_et_al_2019_parameters
        and kappa_axis[0] <= paper_target_kappa_hz * 1e-9 <= kappa_axis[-1]
    ):
        ax2.axvline(
            paper_target_kappa_hz * 1e-9,
            color="tab:green",
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
        )
        ax2.axhline(
            paper_target_linewidth_hz * 1e-6,
            color="tab:green",
            linestyle=":",
            linewidth=1.2,
            alpha=0.65,
        )
    if linewidth_r_squared is not None and np.any(completed_mask):
        r_values = np.asarray(linewidth_r_squared, dtype=float)[completed_mask]
        r_values = r_values[np.isfinite(r_values)]
        if r_values.size:
            ax2.plot(
                [],
                [],
                linestyle="none",
                marker="",
                label=(
                    rf"median phase-fit $R^2={np.median(r_values):.3f}$"
                ),
            )
    ax2.set_xlabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size + 4, labelpad=10)
    ax2.set_ylabel(r"linewidth estimate (MHz)", fontsize=font_size + 4, labelpad=4)
    ax2.set_title(r"Frequency-noise PSD linewidth", fontsize=font_size + 4, pad=title_pad)
    ax2.set_xlim(kappa_axis[0], kappa_axis[-1])
    ax2.set_yscale("log")
    ax2.set_ylim(*linewidth_plot_ylim_mhz)
    ax2.set_yticks(linewidth_plot_yticks_mhz)
    ax2.minorticks_on()
    ax2.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
    )
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


LONG_RUN_PROGRESS_QUEUE = None
LONG_RUN_POSTPROCESS_LOCK = None
VCSEL_INTEGRATE_SUPPORTS_PROGRESS_CALLBACK = (
    "progress_callback" in inspect.signature(VCSEL.integrate).parameters
)


def init_long_run_progress_worker(
    progress_queue,
    worker_count=None,
    postprocess_lock=None,
):
    """Install progress routing, post-processing lock, and worker title."""
    global LONG_RUN_PROGRESS_QUEUE, LONG_RUN_POSTPROCESS_LOCK
    LONG_RUN_PROGRESS_QUEUE = progress_queue
    LONG_RUN_POSTPROCESS_LOCK = postprocess_lock
    process = mp.current_process()
    if process.name != "MainProcess":
        identity = getattr(process, "_identity", ())
        raw_number = int(identity[-1]) if identity else 1
        if worker_count is not None:
            worker_number = (raw_number - 1) % int(max(1, worker_count)) + 1
        else:
            worker_number = raw_number
        set_activity_monitor_process_title(f"long{worker_number}")


def make_worker_progress_callback(job):
    """Create a throttled callback that reports worker progress to the parent."""
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
    """Call VCSEL.integrate with progress callback only when supported."""
    if VCSEL_INTEGRATE_SUPPORTS_PROGRESS_CALLBACK:
        kwargs["progress_callback"] = progress_callback
    return vcsel.integrate(*args, **kwargs)


def shutdown_process_pool(executor, completed):
    """Close a ProcessPoolExecutor and avoid orphaned workers after interrupts."""
    if executor is None:
        return
    processes = list((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(
        wait=bool(completed),
        cancel_futures=not bool(completed),
    )
    if completed:
        return

    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
    for process in processes:
        process.join(timeout=1.0)


def close_progress_queue(progress_queue):
    """Release multiprocessing queue resources when worker progress is finished."""
    if progress_queue is None:
        return
    close = getattr(progress_queue, "close", None)
    if close is not None:
        close()
    join_thread = getattr(progress_queue, "join_thread", None)
    if join_thread is not None:
        join_thread()


def _run_fixed_kappa_linewidth_chunk_worker(jobs):
    """Run one submitted kappa chunk as one vectorized long integration."""
    jobs = list(jobs)
    if not jobs:
        return []

    chunk_seed = int(jobs[0]["chunk_seed"])
    np.random.seed(chunk_seed)

    histories = []
    kappa_blocks = []
    case_counts = []
    for job in jobs:
        n_cases_job = int(job["n_noise_cases"])
        history = replicate_physical_history(job["history"], n_cases_job)
        histories.append(history)
        case_counts.append(n_cases_job)
        kappa_matrix = VCSEL.build_coupling_matrix(
            time_arr=time_array_long[:1],
            kappa_initial=float(job["kappa"]),
            kappa_final=float(job["kappa"]),
            N_lasers=N_lasers,
            ramp_start=0.0,
            ramp_shape=1.0,
            tau=tau,
            scheme="CUSTOM",
            aMAT=adjacency,
        )[0].astype(np.float32, copy=False)
        kappa_blocks.append(
            np.repeat(kappa_matrix[None, :, :], n_cases_job, axis=0)
        )

    history_all = np.concatenate(histories, axis=0)
    kappa_all = np.concatenate(kappa_blocks, axis=0).astype(np.float32, copy=False)
    vcsel, nd = make_vcsel_from_kappa_matrix(
        kappa_all,
        Tmax_long,
        analysis_save_every,
    )
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
            message=(
                f"fixed-kappa phase-diffusion chunk "
                f"{int(jobs[0]['k'])}-{int(jobs[-1]['k'])}"
            ),
            progress_callback=progress_callback,
        )
        flush_progress()
    except Exception:
        flush_progress()
        raise

    results = []
    postprocess_lock = (
        LONG_RUN_POSTPROCESS_LOCK
        if serialize_long_run_postprocessing
        else None
    )
    if postprocess_lock is not None:
        postprocess_lock.acquire()
    try:
        # Hold the shared slot for the complete vectorized chunk. Other long
        # integrations may continue, but a worker that has finished its
        # integration waits with only y_long resident instead of allocating a
        # second simultaneous FFT/PSD workspace.
        offset = 0
        for job, n_cases_job in zip(jobs, case_counts):
            y_job = y_long[offset:offset + n_cases_job]
            offset += n_cases_job
            result = analyze_phase_variance_summary(t_long, y_job)
            result["k"] = int(job["k"])
            result["kappa"] = float(job["kappa"])
            results.append(result)
            del y_job, result
    finally:
        if postprocess_lock is not None:
            postprocess_lock.release()

    del histories, kappa_blocks, history_all, kappa_all, y_long, vcsel, nd
    gc.collect()

    return results


def run_fixed_kappa_linewidth_chunk_worker(jobs):
    """Worker wrapper that records normal Python exceptions to disk."""
    jobs = list(jobs)
    try:
        return _run_fixed_kappa_linewidth_chunk_worker(jobs)
    except BaseException:
        worker_arrays_dir.mkdir(parents=True, exist_ok=True)
        if jobs:
            label = f"{int(jobs[0]['k']):03d}_{int(jobs[-1]['k']):03d}"
        else:
            label = "empty"
        error_path = worker_arrays_dir / named_output(f"worker_error_chunk_{label}", ".txt")
        error_path.write_text(traceback.format_exc())
        raise


def make_linewidth_job(kappa_index, kappa, history):
    return {
        "k": int(kappa_index),
        "kappa": float(kappa),
        "history": history.copy(),
        "n_noise_cases": int(max(1, n_noise_iterations)),
    }


def main():
    set_activity_monitor_process_title("continuation", warn_if_unavailable=True)
    rc("text", usetex=True)
    rc("font", family="serif")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    fit_frames_dir.mkdir(parents=True, exist_ok=True)
    psd_frames_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    worker_arrays_dir.mkdir(parents=True, exist_ok=True)
    validate_analysis_window()
    adjust_psd_floor_band_before_workers()
    if print_psd_frequency_range_on_start:
        print_expected_psd_frequency_range()
    if show_output_size_messages:
        psd_sampling = estimate_effective_long_psd_sampling()
        print(
            "Long linewidth sampling: "
            f"requested_save_every={analysis_save_every}, "
            f"effective_save_every={psd_sampling['effective_save_every']}, "
            f"saved_dt={psd_sampling['saved_dt'] * 1e9:.3f} ns, "
            f"~{psd_sampling['actual_output_gb']:.2f} GB y-output per long worker chunk."
        )

    kappa_values = np.linspace(kappa_initial, kappa_final, int(max(1, n_kappa_steps)))
    dtype = np.float32
    lag_times = None
    phase_variance_map = None
    phase_variance_sem_map = None
    psd_frequency_hz = None
    psd_map = None
    wrapped_phase_time = None
    wrapped_phase_map = None
    fit_time_map = None
    fit_line_map = None
    linewidth_mean_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    linewidth_median_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    linewidth_std_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    linewidth_r_squared = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_combined_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_combined_mean_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_combined_std_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_combined_q16_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_combined_q84_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_common_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_common_mean_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_common_std_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_common_q16_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_common_q84_hz = np.full(kappa_values.size, np.nan, dtype=dtype)
    psd_linewidth_laser_hz = np.full((N_lasers, kappa_values.size), np.nan, dtype=dtype)
    psd_linewidth_laser_mean_hz = np.full_like(psd_linewidth_laser_hz, np.nan)
    psd_linewidth_laser_std_hz = np.full_like(psd_linewidth_laser_hz, np.nan)
    psd_linewidth_laser_q16_hz = np.full_like(psd_linewidth_laser_hz, np.nan)
    psd_linewidth_laser_q84_hz = np.full_like(psd_linewidth_laser_hz, np.nan)
    combined_field_power_mean = np.full(kappa_values.size, np.nan, dtype=dtype)
    combined_field_power_min = np.full(kappa_values.size, np.nan, dtype=dtype)
    order_param = np.full(kappa_values.size, np.nan, dtype=dtype)
    order_param_min = np.full(kappa_values.size, np.nan, dtype=dtype)
    order_param_max = np.full(kappa_values.size, np.nan, dtype=dtype)
    phase_slip_case_counts = np.zeros(kappa_values.size, dtype=int)
    relative_phase_slip_case_counts = np.zeros(kappa_values.size, dtype=int)
    completed_kappa_indices = set()
    progress_frame_paths = []
    psd_frame_paths = []

    master_seed_sequence = np.random.SeedSequence(random_seed)
    resolved_random_seed = int(
        master_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )
    np.random.seed(resolved_random_seed)
    long_seed_sequence = np.random.SeedSequence(resolved_random_seed)
    vcsel0, nd0 = make_vcsel_ramp(
        kappa_values[0],
        kappa_values[0],
        Tmax_continuation,
        time_array_continuation,
        continuation_save_every,
        extra_physical_parameters={
            "noise_amplitude": float(continuation_noise_amplitude),
        },
    )
    history, _, _, _ = vcsel0.generate_history(
        nd0,
        shape="FR",
        n_cases=int(max(1, continuation_n_cases)),
    )
    henry_single_laser_reference_hz = free_running_henry_linewidth_hz(nd0)
    henry_kappa0_reference_hz = collective_kappa0_henry_reference_hz(nd0)
    if history.shape[0] != 1:
        raise ValueError(
            "continuation_n_cases must be 1 so the branch is represented by "
            "one physical delayed trajectory."
        )

    use_process_pool = int(max(1, long_run_jobs)) > 1
    if use_process_pool:
        mp_context = mp.get_context("fork")
        spectrum_progress_queue = (
            mp_context.Queue() if show_long_run_worker_progress else None
        )
        long_run_postprocess_lock = (
            mp_context.Lock()
            if serialize_long_run_postprocessing
            else None
        )
        spectrum_executor = ProcessPoolExecutor(
            max_workers=int(max(1, long_run_jobs)),
            mp_context=mp_context,
            initializer=init_long_run_progress_worker,
            initargs=(
                spectrum_progress_queue,
                int(max(1, long_run_jobs)),
                long_run_postprocess_lock,
            ),
        )
    else:
        long_run_postprocess_lock = None
        spectrum_progress_queue = (
            queue_module.Queue() if show_long_run_worker_progress else None
        )
        spectrum_executor = None
        if show_long_run_worker_progress:
            init_long_run_progress_worker(
                spectrum_progress_queue,
                1,
                long_run_postprocess_lock,
            )

    pending_futures = []
    future_progress_id = {}
    progress_bars = {}
    next_progress_position = 2 if show_long_run_worker_progress else 1
    submitted_chunks = 0
    completed_chunks = 0
    total_chunks = int(np.ceil(len(kappa_values) / int(max(1, two_stage_kappa_chunk_size))))
    # Keep one complete wave of vectorized long chunks queued behind the wave
    # currently executing. With long_run_jobs == 3 this permits three active
    # chunks plus three fully prepared chunks waiting in ProcessPoolExecutor,
    # while preventing continuation histories for the rest of the sweep from
    # accumulating in the parent process.
    active_long_chunk_slots = int(max(1, long_run_jobs))
    queued_long_chunk_slots = int(max(1, long_run_jobs))
    max_pending_long_chunks = active_long_chunk_slots + queued_long_chunk_slots
    current_chunk = []
    run_completed = False

    def drain_long_run_progress_queue():
        if spectrum_progress_queue is None:
            return
        while True:
            try:
                progress_id, n_steps = spectrum_progress_queue.get_nowait()
            except queue_module.Empty:
                break
            bar = progress_bars.get(progress_id)
            if bar is not None:
                remaining = bar.total - bar.n if bar.total is not None else int(n_steps)
                bar.update(min(int(n_steps), max(0, remaining)))

    def make_long_run_progress_bar(progress_id, chunk_indices):
        nonlocal next_progress_position
        if spectrum_progress_queue is None:
            return
        # VCSEL.integrate advances over range(start_idx, steps - 1), so the
        # number of progress-callback updates is one less than the number of
        # sample points from start_idx to the end.
        total_steps = max(1, long_steps - 1 - 2 * delay_steps)
        desc = f"long linewidth k={chunk_indices[0]}-{chunk_indices[-1]}"
        progress_bars[progress_id] = tqdm(
            total=total_steps,
            desc=desc,
            unit="step",
            position=next_progress_position,
            leave=False,
            dynamic_ncols=True,
        )
        next_progress_position += 1

    def attach_long_run_progress(chunk, progress_id):
        if spectrum_progress_queue is None:
            return chunk
        for job in chunk:
            job["progress_id"] = progress_id
            job["progress_update_steps"] = int(max(1, long_run_progress_update_steps))
        return chunk

    def submit_chunk():
        nonlocal current_chunk, submitted_chunks
        if not current_chunk:
            return
        chunk = current_chunk
        current_chunk = []
        submitted_chunks += 1
        chunk_seed = int(
            long_seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint32)[0]
        )
        for job in chunk:
            job["chunk_seed"] = chunk_seed
        chunk_indices = [int(job["k"]) for job in chunk]
        progress_id = f"linewidth_chunk_{submitted_chunks}"
        make_long_run_progress_bar(progress_id, chunk_indices)
        chunk = attach_long_run_progress(chunk, progress_id)
        if use_process_pool:
            future = spectrum_executor.submit(run_fixed_kappa_linewidth_chunk_worker, chunk)
            pending_futures.append(future)
            future_progress_id[future] = progress_id
        else:
            result_paths = run_fixed_kappa_linewidth_chunk_worker(chunk)
            drain_long_run_progress_queue()
            process_chunk_results(result_paths, progress_id)

    def store_phase_variance_results(results):
        nonlocal lag_times, fit_time_map, phase_variance_map, phase_variance_sem_map
        nonlocal psd_frequency_hz, psd_map
        nonlocal fit_line_map, wrapped_phase_time, wrapped_phase_map
        if not results:
            return []
        if isinstance(results, dict):
            results = [results]
        results = sorted(results, key=lambda item: item["k"])

        if lag_times is None:
            lag_times = np.asarray(results[0]["lag_times"], dtype=dtype)
            n_lags = lag_times.size
            phase_variance_map = np.full((kappa_values.size, n_lags), np.nan, dtype=dtype)
            phase_variance_sem_map = np.full_like(phase_variance_map, np.nan)

        if psd_frequency_hz is None and "psd_frequency_hz" in results[0]:
            psd_frequency_hz = np.asarray(results[0]["psd_frequency_hz"], dtype=dtype)
            psd_map = np.full(
                (kappa_values.size, psd_frequency_hz.size),
                np.nan,
                dtype=dtype,
            )

        if wrapped_phase_time is None and "wrapped_phase_time" in results[0]:
            wrapped_phase_time = np.asarray(results[0]["wrapped_phase_time"], dtype=dtype)
            wrapped_phase_map = np.full(
                (kappa_values.size, wrapped_phase_time.size),
                np.nan,
                dtype=dtype,
            )

        if fit_time_map is None:
            fit_time_map = np.asarray(results[0]["fit_time"], dtype=dtype)
            fit_line_map = np.full(
                (kappa_values.size, fit_time_map.size),
                np.nan,
                dtype=dtype,
            )

        result_indices = []
        for result in results:
            kappa_index = int(result["k"])
            kappa_stop = kappa_values[kappa_index]
            kappa_start = (
                kappa_values[kappa_index - 1]
                if kappa_index > 0
                else kappa_stop
            )
            result_lag_times = np.asarray(result["lag_times"], dtype=dtype)
            if result_lag_times.shape != lag_times.shape or not np.allclose(
                result_lag_times,
                lag_times,
            ):
                raise ValueError("Long fixed-kappa phase-variance results do not share the same lag grid.")

            phase_variance_map[kappa_index] = np.asarray(
                result["phase_variance"],
                dtype=dtype,
            )
            phase_variance_sem_map[kappa_index] = np.asarray(
                result["phase_variance_sem"],
                dtype=dtype,
            )
            if psd_map is not None and "psd_heatmap" in result:
                result_psd_frequency = np.asarray(
                    result["psd_frequency_hz"],
                    dtype=dtype,
                )
                if (
                    result_psd_frequency.shape != psd_frequency_hz.shape
                    or not np.allclose(result_psd_frequency, psd_frequency_hz)
                ):
                    raise ValueError("Long fixed-kappa PSD results do not share the same frequency grid.")
                psd_map[kappa_index] = np.asarray(
                    result["psd_heatmap"],
                    dtype=dtype,
                )
            if wrapped_phase_map is not None and "wrapped_phase_abs" in result:
                result_phase_time = np.asarray(
                    result["wrapped_phase_time"],
                    dtype=dtype,
                )
                if (
                    result_phase_time.shape != wrapped_phase_time.shape
                    or not np.allclose(result_phase_time, wrapped_phase_time)
                ):
                    raise ValueError("Long fixed-kappa wrapped phase results do not share the same time grid.")
                wrapped_phase_map[kappa_index] = np.asarray(
                    result["wrapped_phase_abs"],
                    dtype=dtype,
                )
            linewidth_mean_hz[kappa_index] = result["linewidth_hz"]
            linewidth_median_hz[kappa_index] = result["linewidth_median_hz"]
            linewidth_std_hz[kappa_index] = result["linewidth_std_hz"]
            linewidth_r_squared[kappa_index] = result["r_squared"]
            psd_linewidth_combined_hz[kappa_index] = result[
                "psd_linewidth_combined_hz"
            ]
            psd_linewidth_combined_mean_hz[kappa_index] = result[
                "psd_linewidth_combined_mean_hz"
            ]
            psd_linewidth_combined_std_hz[kappa_index] = result[
                "psd_linewidth_combined_std_hz"
            ]
            psd_linewidth_combined_q16_hz[kappa_index] = result[
                "psd_linewidth_combined_q16_hz"
            ]
            psd_linewidth_combined_q84_hz[kappa_index] = result[
                "psd_linewidth_combined_q84_hz"
            ]
            psd_linewidth_common_hz[kappa_index] = result[
                "psd_linewidth_common_hz"
            ]
            psd_linewidth_common_mean_hz[kappa_index] = result[
                "psd_linewidth_common_mean_hz"
            ]
            psd_linewidth_common_std_hz[kappa_index] = result[
                "psd_linewidth_common_std_hz"
            ]
            psd_linewidth_common_q16_hz[kappa_index] = result[
                "psd_linewidth_common_q16_hz"
            ]
            psd_linewidth_common_q84_hz[kappa_index] = result[
                "psd_linewidth_common_q84_hz"
            ]
            result_laser_linewidths = np.asarray(
                result["psd_linewidth_laser_hz"],
                dtype=dtype,
            )
            result_laser_linewidth_means = np.asarray(
                result["psd_linewidth_laser_mean_hz"],
                dtype=dtype,
            )
            result_laser_linewidth_stds = np.asarray(
                result["psd_linewidth_laser_std_hz"],
                dtype=dtype,
            )
            result_laser_linewidth_q16 = np.asarray(
                result["psd_linewidth_laser_q16_hz"], dtype=dtype
            )
            result_laser_linewidth_q84 = np.asarray(
                result["psd_linewidth_laser_q84_hz"], dtype=dtype
            )
            n_result_lasers = min(N_lasers, result_laser_linewidths.size)
            psd_linewidth_laser_hz[:n_result_lasers, kappa_index] = (
                result_laser_linewidths[:n_result_lasers]
            )
            psd_linewidth_laser_mean_hz[:n_result_lasers, kappa_index] = (
                result_laser_linewidth_means[:n_result_lasers]
            )
            psd_linewidth_laser_std_hz[:n_result_lasers, kappa_index] = (
                result_laser_linewidth_stds[:n_result_lasers]
            )
            psd_linewidth_laser_q16_hz[:n_result_lasers, kappa_index] = (
                result_laser_linewidth_q16[:n_result_lasers]
            )
            psd_linewidth_laser_q84_hz[:n_result_lasers, kappa_index] = (
                result_laser_linewidth_q84[:n_result_lasers]
            )
            combined_field_power_mean[kappa_index] = result["combined_field_power_mean"]
            combined_field_power_min[kappa_index] = result["combined_field_power_min"]
            order_param[kappa_index] = result["order_parameter"]
            order_param_min[kappa_index] = result["order_parameter_min"]
            order_param_max[kappa_index] = result["order_parameter_max"]
            phase_slip_case_counts[kappa_index] = result["phase_slip_case_count"]
            relative_phase_slip_case_counts[kappa_index] = result[
                "relative_phase_slip_case_count"
            ]
            if fit_line_map is not None:
                fit_line = np.asarray(result["fit_line"], dtype=dtype)
                if fit_line.shape == fit_line_map[kappa_index].shape:
                    fit_line_map[kappa_index] = fit_line
            completed_kappa_indices.add(kappa_index)
            result_indices.append(kappa_index)

            if write_fit_frames and compute_phase_variance_linewidth:
                figure_path = fit_frames_dir / (
                    named_output(
                        f"kappa_{kappa_index:03d}_{kappa_stop * 1e-9:.3f}ns_inv_fit",
                        ".png",
                    )
                )
                render_figure(
                    plot_phase_variance_fit_result,
                    result,
                    kappa_start,
                    kappa_stop,
                    figure_path,
                )
                progress_frame_paths.append(str(figure_path.resolve()))

            if write_psd_frames:
                figure_path = psd_frames_dir / (
                    named_output(
                        f"kappa_{kappa_index:03d}_{kappa_stop * 1e-9:.3f}ns_inv_psd",
                        ".png",
                    )
                )
                render_figure(
                    plot_frequency_noise_psd_result,
                    result,
                    kappa_start,
                    kappa_stop,
                    figure_path,
                )
                psd_frame_paths.append(str(figure_path.resolve()))

        gc.collect()
        return result_indices

    def save_progress_arrays():
        if not write_progress_arrays or phase_variance_map is None:
            return
        np.save(arrays_dir / named_output("phase_variance_map", ".npy"), phase_variance_map)
        np.save(
            arrays_dir / named_output("phase_variance_sem_map", ".npy"),
            phase_variance_sem_map,
        )
        if wrapped_phase_map is not None and wrapped_phase_time is not None:
            np.save(
                arrays_dir / named_output("wrapped_phase_abs_map", ".npy"),
                wrapped_phase_map,
            )
            np.save(
                arrays_dir / named_output("wrapped_phase_time_seconds", ".npy"),
                wrapped_phase_time,
            )
        np.save(arrays_dir / named_output("lag_times_seconds", ".npy"), lag_times)
        np.save(arrays_dir / named_output("fit_lag_times_seconds", ".npy"), fit_time_map)
        np.save(arrays_dir / named_output("fit_line_map", ".npy"), fit_line_map)
        np.save(arrays_dir / named_output("kappa_rad_s", ".npy"), kappa_values)
        np.save(arrays_dir / named_output("linewidth_mean_hz", ".npy"), linewidth_mean_hz)
        np.save(arrays_dir / named_output("linewidth_median_hz", ".npy"), linewidth_median_hz)
        np.save(arrays_dir / named_output("linewidth_std_hz", ".npy"), linewidth_std_hz)
        np.save(arrays_dir / named_output("linewidth_r_squared", ".npy"), linewidth_r_squared)
        if psd_frequency_hz is not None and psd_map is not None:
            np.save(
                arrays_dir / named_output("frequency_noise_psd_frequency_hz", ".npy"),
                psd_frequency_hz,
            )
            np.save(
                arrays_dir / named_output("frequency_noise_psd_map", ".npy"),
                psd_map,
            )
            np.save(
                arrays_dir / named_output("psd_linewidth_combined_hz", ".npy"),
                psd_linewidth_combined_hz,
            )
            np.save(
                arrays_dir / named_output("psd_linewidth_combined_mean_hz", ".npy"),
                psd_linewidth_combined_mean_hz,
            )
            np.save(
                arrays_dir / named_output("psd_linewidth_combined_std_hz", ".npy"),
                psd_linewidth_combined_std_hz,
            )
            np.save(arrays_dir / named_output("psd_linewidth_combined_q16_hz", ".npy"), psd_linewidth_combined_q16_hz)
            np.save(arrays_dir / named_output("psd_linewidth_combined_q84_hz", ".npy"), psd_linewidth_combined_q84_hz)
            np.save(
                arrays_dir / named_output("psd_linewidth_common_hz", ".npy"),
                psd_linewidth_common_hz,
            )
            np.save(
                arrays_dir / named_output("psd_linewidth_common_mean_hz", ".npy"),
                psd_linewidth_common_mean_hz,
            )
            np.save(
                arrays_dir / named_output("psd_linewidth_common_std_hz", ".npy"),
                psd_linewidth_common_std_hz,
            )
            np.save(arrays_dir / named_output("psd_linewidth_common_q16_hz", ".npy"), psd_linewidth_common_q16_hz)
            np.save(arrays_dir / named_output("psd_linewidth_common_q84_hz", ".npy"), psd_linewidth_common_q84_hz)
            np.save(
                arrays_dir / named_output("psd_linewidth_laser_hz", ".npy"),
                psd_linewidth_laser_hz,
            )
            np.save(
                arrays_dir / named_output("psd_linewidth_laser_mean_hz", ".npy"),
                psd_linewidth_laser_mean_hz,
            )
            np.save(
                arrays_dir / named_output("psd_linewidth_laser_std_hz", ".npy"),
                psd_linewidth_laser_std_hz,
            )
            np.save(arrays_dir / named_output("psd_linewidth_laser_q16_hz", ".npy"), psd_linewidth_laser_q16_hz)
            np.save(arrays_dir / named_output("psd_linewidth_laser_q84_hz", ".npy"), psd_linewidth_laser_q84_hz)
            np.save(
                arrays_dir / named_output("combined_field_power_mean", ".npy"),
                combined_field_power_mean,
            )
            np.save(
                arrays_dir / named_output("combined_field_power_min", ".npy"),
                combined_field_power_min,
            )
        np.save(arrays_dir / named_output("order_parameter", ".npy"), order_param)
        np.save(arrays_dir / named_output("order_parameter_min", ".npy"), order_param_min)
        np.save(arrays_dir / named_output("order_parameter_max", ".npy"), order_param_max)
        np.save(arrays_dir / named_output("phase_slip_case_counts", ".npy"), phase_slip_case_counts)
        np.save(
            arrays_dir / named_output("relative_phase_slip_case_counts", ".npy"),
            relative_phase_slip_case_counts,
        )
        progress_psd_sampling = estimate_effective_long_psd_sampling()
        np.savez(
            arrays_dir / named_output("phase_variance_progress", ".npz"),
            completed_kappa_indices=np.array(
                sorted(completed_kappa_indices),
                dtype=int,
            ),
            submitted_chunks=int(submitted_chunks),
            total_kappa_values=int(kappa_values.size),
            linewidth_estimator="frequency_noise_psd_white_floor",
            psd_heatmap_channel=str(psd_heatmap_channel),
            psd_floor_band_hz=np.asarray(psd_floor_band_hz, dtype=float),
            Tmax_long=float(Tmax_long),
            long_run_burn_in_time=float(long_run_burn_in_time),
            long_run_analysis_start_time=float(
                progress_psd_sampling["analysis_start_time"]
            ),
            long_run_analysis_duration=float(
                progress_psd_sampling["retained_duration"]
            ),
            effective_analysis_save_every=int(
                progress_psd_sampling["effective_save_every"]
            ),
            effective_saved_dt_seconds=float(progress_psd_sampling["saved_dt"]),
            long_run_retained_saved_samples=int(
                progress_psd_sampling["retained_saved_samples"]
            ),
        )

    def save_progress_frame(k_completed=None):
        if not write_progress_frame or phase_variance_map is None:
            return

        live_path = frames_dir / named_output("phase_variance_progress", ".png")
        render_figure(
            plot_phase_variance_continuation_frame,
            live_path,
            kappa_values,
            wrapped_phase_map,
            wrapped_phase_time,
            phase_variance_map,
            lag_times,
            linewidth_mean_hz,
            linewidth_median_hz,
            psd_frequency_hz=psd_frequency_hz,
            psd_map=psd_map,
            psd_linewidth_combined_hz=psd_linewidth_combined_hz,
            psd_linewidth_common_hz=psd_linewidth_common_hz,
            psd_linewidth_laser_hz=psd_linewidth_laser_hz,
            psd_linewidth_combined_std_hz=psd_linewidth_combined_std_hz,
            psd_linewidth_common_std_hz=psd_linewidth_common_std_hz,
            psd_linewidth_laser_std_hz=psd_linewidth_laser_std_hz,
            psd_linewidth_combined_q16_hz=psd_linewidth_combined_q16_hz,
            psd_linewidth_combined_q84_hz=psd_linewidth_combined_q84_hz,
            psd_linewidth_common_q16_hz=psd_linewidth_common_q16_hz,
            psd_linewidth_common_q84_hz=psd_linewidth_common_q84_hz,
            psd_linewidth_laser_q16_hz=psd_linewidth_laser_q16_hz,
            psd_linewidth_laser_q84_hz=psd_linewidth_laser_q84_hz,
            psd_heatmap_channel_label=psd_heatmap_channel,
            order_param=order_param,
            order_param_min=order_param_min,
            order_param_max=order_param_max,
            linewidth_r_squared=linewidth_r_squared,
            henry_kappa0_reference_hz=henry_kappa0_reference_hz,
            henry_single_laser_reference_hz=henry_single_laser_reference_hz,
            completed_kappa_indices=completed_kappa_indices,
        )
    def close_long_run_progress_bar(progress_id):
        if progress_id in progress_bars:
            drain_long_run_progress_queue()
            bar = progress_bars.pop(progress_id)
            if bar.total is not None and bar.n < bar.total:
                bar.update(bar.total - bar.n)
            bar.close()

    def process_chunk_results(results, progress_id):
        nonlocal completed_chunks
        result_indices = store_phase_variance_results(results)
        if result_indices:
            save_progress_arrays()
            save_progress_frame(k_completed=max(result_indices))
        close_long_run_progress_bar(progress_id)
        completed_chunks += 1
        collect_bar.update(1)

    def process_future(future):
        progress_id = future_progress_id.pop(future, None)
        try:
            result_paths = future.result()
        except BrokenProcessPool as exc:
            if progress_id in progress_bars:
                progress_bars[progress_id].close()
                del progress_bars[progress_id]
            raise RuntimeError(
                "A long fixed-kappa worker process was killed abruptly. This "
                "usually means a subprocess/fork problem or memory pressure, "
                "not a linewidth-fit-window issue. Try long_run_jobs = 1 to "
                "run the long chunks in the main process and expose the real "
                "error path."
            ) from exc
        except Exception as exc:
            if progress_id in progress_bars:
                progress_bars[progress_id].close()
                del progress_bars[progress_id]
            raise RuntimeError(
                "A long fixed-kappa linewidth chunk failed. The original "
                "exception is chained below."
            ) from exc
        process_chunk_results(result_paths, progress_id)

    def collect_finished_chunks(block=False, stop_after_one=False):
        drain_long_run_progress_queue()
        if block:
            while pending_futures:
                drain_long_run_progress_queue()
                for future in list(pending_futures):
                    if future.done():
                        pending_futures.remove(future)
                        process_future(future)
                        if stop_after_one:
                            drain_long_run_progress_queue()
                            return
                if pending_futures:
                    try:
                        progress_id, n_steps = spectrum_progress_queue.get(timeout=0.25)
                        bar = progress_bars.get(progress_id)
                        if bar is not None:
                            remaining = bar.total - bar.n if bar.total is not None else int(n_steps)
                            bar.update(min(int(n_steps), max(0, remaining)))
                    except queue_module.Empty:
                        pass
                    except AttributeError:
                        time.sleep(0.25)
            drain_long_run_progress_queue()
            return
        for future in list(pending_futures):
            if future.done():
                pending_futures.remove(future)
                process_future(future)

    try:
        collect_bar = tqdm(
            total=total_chunks,
            desc="long linewidth chunks complete",
            unit="chunk",
            position=0,
            dynamic_ncols=True,
        )
        short_bar = tqdm(
            enumerate(kappa_values),
            total=len(kappa_values),
            desc="short kappa continuation",
            unit="step",
            position=1 if show_long_run_worker_progress else 0,
            dynamic_ncols=True,
        )
        for kappa_index, kappa_stop in short_bar:
            kappa_start = (
                kappa_values[kappa_index - 1]
                if kappa_index > 0
                else kappa_stop
            )
            short_bar.set_postfix(kappa_ns=f"{kappa_stop * 1e-9:.2f}")
            vcsel, nd = make_vcsel_ramp(
                kappa_start,
                kappa_stop,
                Tmax_continuation,
                time_array_continuation,
                continuation_save_every,
                extra_physical_parameters={
                    "noise_amplitude": float(continuation_noise_amplitude),
                },
            )
            t_short, y_short, _, final_history = vcsel.integrate(
                history,
                nd=nd,
                progress=False,
                max_iter=5,
                smooth_freqs=False,
                return_final_history=True,
                message=f"short kappa {kappa_stop * 1e-9:.3g} ns^-1",
            )
            if final_history.shape[0] != 1:
                raise RuntimeError(
                    "Continuation unexpectedly returned more than one history."
                )
            current_chunk.append(
                make_linewidth_job(
                    kappa_index,
                    kappa_stop,
                    final_history,
                )
            )
            history = final_history
            # Keep only `history`, the single physical delayed state needed by
            # the next continuation step.  The job owns its separate copy;
            # the saved continuation trajectory and temporary model objects
            # are no longer needed once this step has produced final_history.
            del t_short, y_short, final_history, vcsel, nd

            if (
                len(current_chunk) >= int(max(1, two_stage_kappa_chunk_size))
                or kappa_index == len(kappa_values) - 1
            ):
                submit_chunk()
            collect_finished_chunks(block=False)

            # Once both the executing wave and the queued wave are full, wait
            # for only one chunk to finish. Continuation then prepares exactly
            # one replacement chunk and refills that rolling queue slot.
            if (
                use_process_pool
                and len(pending_futures) >= max_pending_long_chunks
            ):
                short_bar.set_postfix(
                    kappa_ns=f"{kappa_stop * 1e-9:.2f}",
                    state="long-worker queue full",
                )
                collect_finished_chunks(block=True, stop_after_one=True)
                gc.collect()

        short_bar.close()
        submit_chunk()
        collect_finished_chunks(block=True)
        run_completed = True
    finally:
        drain_long_run_progress_queue()
        for bar in progress_bars.values():
            bar.close()
        if not run_completed:
            for future in pending_futures:
                future.cancel()
        if spectrum_executor is not None:
            shutdown_process_pool(spectrum_executor, run_completed)
        close_progress_queue(spectrum_progress_queue)
        if "collect_bar" in locals():
            collect_bar.close()

    save_progress_arrays()
    save_progress_frame()

    paper_target_actual_kappa_hz = np.nan
    paper_target_laser_linewidths_hz = np.full(N_lasers, np.nan, dtype=float)
    paper_target_laser_floor_hz2_per_hz = np.full(N_lasers, np.nan, dtype=float)
    paper_target_laser_q16_hz = np.full(N_lasers, np.nan, dtype=float)
    paper_target_laser_q84_hz = np.full(N_lasers, np.nan, dtype=float)
    if use_ma_et_al_2019_parameters:
        paper_target_index = int(
            np.argmin(np.abs(kappa_values - float(paper_target_kappa_hz)))
        )
        paper_target_actual_kappa_hz = float(kappa_values[paper_target_index])
        paper_target_laser_linewidths_hz = np.asarray(
            psd_linewidth_laser_hz[:, paper_target_index], dtype=float
        )
        paper_target_laser_floor_hz2_per_hz = (
            paper_target_laser_linewidths_hz / np.pi
        )
        paper_target_laser_q16_hz = np.asarray(
            psd_linewidth_laser_q16_hz[:, paper_target_index], dtype=float
        )
        paper_target_laser_q84_hz = np.asarray(
            psd_linewidth_laser_q84_hz[:, paper_target_index], dtype=float
        )
        print(
            "Ma et al. target comparison at "
            f"kappa={paper_target_actual_kappa_hz * 1e-9:.6g} ns^-1: "
            f"laser linewidths={paper_target_laser_linewidths_hz} Hz, "
            f"white floors={paper_target_laser_floor_hz2_per_hz} Hz^2/Hz, "
            f"16th-84th percentiles=[{paper_target_laser_q16_hz}, "
            f"{paper_target_laser_q84_hz}] Hz; paper target="
            f"{paper_target_linewidth_hz:g} Hz."
        )

    final_psd_sampling = estimate_effective_long_psd_sampling()
    np.savez(
        output_dir / named_output("linewidth_autocorr_kappa_continuation", ".npz"),
        kappa_rad_s=kappa_values,
        paper_target_kappa_requested_hz=(
            float(paper_target_kappa_hz)
            if use_ma_et_al_2019_parameters
            else np.nan
        ),
        paper_target_kappa_actual_hz=paper_target_actual_kappa_hz,
        paper_target_linewidth_reference_hz=(
            float(paper_target_linewidth_hz)
            if use_ma_et_al_2019_parameters
            else np.nan
        ),
        paper_target_laser_linewidths_hz=paper_target_laser_linewidths_hz,
        paper_target_laser_floor_hz2_per_hz=paper_target_laser_floor_hz2_per_hz,
        paper_target_laser_q16_hz=paper_target_laser_q16_hz,
        paper_target_laser_q84_hz=paper_target_laser_q84_hz,
        lag_times_seconds=lag_times,
        phase_variance_map=phase_variance_map,
        phase_variance_sem_map=phase_variance_sem_map,
        wrapped_phase_abs_map=wrapped_phase_map,
        wrapped_phase_time_seconds=wrapped_phase_time,
        fit_lag_times_seconds=fit_time_map,
        fit_line_map=fit_line_map,
        linewidth_mean_hz=linewidth_mean_hz,
        linewidth_median_hz=linewidth_median_hz,
        linewidth_std_hz=linewidth_std_hz,
        linewidth_r_squared=linewidth_r_squared,
        psd_frequency_hz=psd_frequency_hz,
        psd_available_frequency_range_hz=psd_available_frequency_range_hz(
            psd_frequency_hz
        ),
        psd_map=psd_map,
        psd_heatmap_channel=str(psd_heatmap_channel),
        psd_floor_band_hz=np.asarray(psd_floor_band_hz, dtype=float),
        psd_plot_frequency_axis_ghz=np.asarray(psd_plot_frequency_axis_ghz, dtype=float),
        psd_linewidth_combined_hz=psd_linewidth_combined_hz,
        psd_linewidth_combined_mean_hz=psd_linewidth_combined_mean_hz,
        psd_linewidth_combined_std_hz=psd_linewidth_combined_std_hz,
        psd_linewidth_combined_q16_hz=psd_linewidth_combined_q16_hz,
        psd_linewidth_combined_q84_hz=psd_linewidth_combined_q84_hz,
        psd_linewidth_common_hz=psd_linewidth_common_hz,
        psd_linewidth_common_mean_hz=psd_linewidth_common_mean_hz,
        psd_linewidth_common_std_hz=psd_linewidth_common_std_hz,
        psd_linewidth_common_q16_hz=psd_linewidth_common_q16_hz,
        psd_linewidth_common_q84_hz=psd_linewidth_common_q84_hz,
        psd_linewidth_laser_hz=psd_linewidth_laser_hz,
        psd_linewidth_laser_mean_hz=psd_linewidth_laser_mean_hz,
        psd_linewidth_laser_std_hz=psd_linewidth_laser_std_hz,
        psd_linewidth_laser_q16_hz=psd_linewidth_laser_q16_hz,
        psd_linewidth_laser_q84_hz=psd_linewidth_laser_q84_hz,
        combined_field_power_mean=combined_field_power_mean,
        combined_field_power_min=combined_field_power_min,
        henry_kappa0_reference_hz=float(henry_kappa0_reference_hz),
        henry_single_laser_reference_hz=float(henry_single_laser_reference_hz),
        plot_kappa0_henry_reference=bool(plot_kappa0_henry_reference),
        order_parameter=order_param,
        order_parameter_min=order_param_min,
        order_parameter_max=order_param_max,
        phase_slip_case_counts=phase_slip_case_counts,
        relative_phase_slip_case_counts=relative_phase_slip_case_counts,
        progress_frame_paths=np.asarray(progress_frame_paths),
        psd_frame_paths=np.asarray(psd_frame_paths),
        progress_frames_dir=str(frames_dir.resolve()),
        phase_variance_fit_frames_dir=str(fit_frames_dir.resolve()),
        frequency_noise_psd_frames_dir=str(psd_frames_dir.resolve()),
        linewidth_estimator="frequency_noise_psd_white_floor",
        linewidth_fit_quantity="white_frequency_noise_floor",
        phase_variance_linewidth_estimator=(
            "collective_phase_diffusion"
            if compute_phase_variance_linewidth
            else "disabled"
        ),
        compute_phase_variance_linewidth=bool(compute_phase_variance_linewidth),
        collective_phase_definition="mean_of_individually_unwrapped_laser_phases",
        long_run_jobs=int(max(1, long_run_jobs)),
        serialize_long_run_postprocessing=bool(
            serialize_long_run_postprocessing
        ),
        two_stage_kappa_chunk_size=int(max(1, two_stage_kappa_chunk_size)),
        continuation_n_cases=int(max(1, continuation_n_cases)),
        continuation_noise_amplitude=float(continuation_noise_amplitude),
        continuation_history_strategy="single_physical_trajectory_replicated_for_long_noise_ensemble",
        resolved_random_seed=np.uint32(resolved_random_seed),
        use_available_data_if_tmax_too_short=bool(use_available_data_if_tmax_too_short),
        minimum_analysis_samples=int(max(4, minimum_analysis_samples)),
        long_run_history_is_steady_state=bool(long_run_history_is_steady_state),
        phase_plot_tail_window_us=(
            np.nan
            if phase_plot_tail_window_us is None
            else float(phase_plot_tail_window_us)
        ),
        write_progress_arrays=bool(write_progress_arrays),
        write_progress_frame=bool(write_progress_frame),
        write_fit_frames=bool(write_fit_frames),
        write_psd_frames=bool(write_psd_frames),
        render_figures_in_disposable_process=bool(
            render_figures_in_disposable_process
        ),
        max_plot_points=int(max(1, max_plot_points)),
        figure_dpi=int(figure_dpi),
        continuation_frame_dpi=int(continuation_frame_dpi),
        continuation_frame_font_size=int(continuation_frame_font_size),
        continuation_frame_title_pad=float(continuation_frame_title_pad),
        continuation_linewidth_legend_font_size=int(continuation_linewidth_legend_font_size),
        analysis_save_every=int(max(1, analysis_save_every)),
        effective_analysis_save_every=int(
            final_psd_sampling["effective_save_every"]
        ),
        analysis_output_dtype=str(np.dtype(analysis_output_dtype)),
        linewidth_plot_ylim_mhz=np.asarray(linewidth_plot_ylim_mhz, dtype=float),
        linewidth_plot_yticks_mhz=np.asarray(linewidth_plot_yticks_mhz, dtype=float),
        phase_variance_colorbar_log_scale=bool(phase_variance_colorbar_log_scale),
        phase_variance_log_vmin_rad2=(
            np.nan
            if phase_variance_log_vmin_rad2 is None
            else float(phase_variance_log_vmin_rad2)
        ),
        phase_variance_log_vmax_rad2=(
            np.nan
            if phase_variance_log_vmax_rad2 is None
            else float(phase_variance_log_vmax_rad2)
        ),
        phase_variance_lag_axis_ns=np.asarray(phase_variance_lag_axis_ns, dtype=float),
        fit_lag_ns=np.asarray(fit_lag_ns, dtype=float),
        phase_diffusion_fit_ylim=(
            np.asarray([np.nan, np.nan], dtype=float)
            if phase_diffusion_fit_ylim is None
            else np.asarray(phase_diffusion_fit_ylim, dtype=float)
        ),
        phase_diffusion_fit_yticks=(
            np.asarray([], dtype=float)
            if phase_diffusion_fit_yticks is None
            else np.asarray(phase_diffusion_fit_yticks, dtype=float)
        ),
        phase_variance_log_vmin_percentile=float(phase_variance_log_vmin_percentile),
        phase_variance_log_vmax_percentile=float(phase_variance_log_vmax_percentile),
        phase_variance_saved_dt_seconds=float(final_psd_sampling["saved_dt"]),
        estimated_long_output_gb_per_worker=float(estimate_long_output_gb()),
        use_ma_et_al_2019_parameters=bool(use_ma_et_al_2019_parameters),
        ma_et_al_2019_delay_regime=str(ma_et_al_2019_delay_regime),
        output_name_suffix=str(output_name_suffix),
        Tmax_continuation=Tmax_continuation,
        Tmax_long=Tmax_long,
        long_run_burn_in_time=long_run_burn_in_time,
        long_run_analysis_start_time=float(
            final_psd_sampling["analysis_start_time"]
        ),
        long_run_analysis_duration=float(final_psd_sampling["retained_duration"]),
        long_run_retained_saved_samples=int(
            final_psd_sampling["retained_saved_samples"]
        ),
        psd_welch_nperseg_effective=int(final_psd_sampling["welch_nperseg"]),
    )
    print(f"\nSaved phase-variance outputs to {output_dir.resolve()}")
    print(
        "Saved summary arrays to "
        f"{(output_dir / named_output('linewidth_autocorr_kappa_continuation', '.npz')).resolve()}"
    )


if __name__ == "__main__":
    main()
