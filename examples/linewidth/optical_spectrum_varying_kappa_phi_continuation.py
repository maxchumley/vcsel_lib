#%%
# Example usage
import numpy as np
import multiprocessing as mp
from vcsel_lib import VCSEL
from sympy import symbols, Eq, solve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from matplotlib import texmanager
from IPython.display import clear_output
import gc
import os
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

import queue as queue_module
import time
import inspect
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
from IPython.display import clear_output
from joblib import Parallel, delayed, parallel
try:
    from examples._paths import LINEWIDTH_RESULTS_DIR
except ModuleNotFoundError:
    LINEWIDTH_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "linewidth_estimation"
from tqdm.auto import tqdm



@contextmanager
def tqdm_joblib(tqdm_object):
    """Route joblib batch completions into one clean tqdm bar."""
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def average_history_across_cases(history):
    """Average a history over noise cases, then copy it back to every case."""
    history = np.asarray(history)
    if history.ndim != 3 or history.shape[0] <= 1:
        return history.copy()
    mean_history = np.mean(history, axis=0, keepdims=True)
    return np.repeat(mean_history, history.shape[0], axis=0).astype(
        history.dtype,
        copy=False,
    )


def continuation_history_from_saved(
    y,
    nd,
    final_history=None,
    average_cases=False,
):
    """Return the full 2*tau history required by the current integrator API."""
    if final_history is not None:
        delay_steps_nd = int(nd['delay_steps'])
        history_len = 2 * delay_steps_nd
        if final_history.shape[2] < history_len:
            raise ValueError(
                f"Returned final history has only {final_history.shape[2]} samples, "
                f"but continuation requires {history_len} samples."
            )
        history = final_history[:, :, -history_len:].copy()
        if average_cases:
            history = average_history_across_cases(history)
        return history

    save_every = int(max(1, nd.get('save_every', 1)))
    if save_every != 1:
        raise ValueError(
            "This continuation script needs full-resolution saved output to "
            "reuse the final 2*tau as the next history. Keep save_every=1."
        )

    delay_steps_nd = int(nd['delay_steps'])
    history_len = 2 * delay_steps_nd
    if y.shape[2] < history_len:
        raise ValueError(
            f"Saved trajectory has only {y.shape[2]} samples, but continuation "
            f"requires {history_len} samples of 2*tau history."
        )
    history = y[:, :, -history_len:].copy()
    if average_cases:
        history = average_history_across_cases(history)
    return history



def set_latex_plot_style(tex_cache_dir=None):
    """Apply TeX/Computer Modern font settings in this process."""
    if tex_cache_dir is not None:
        tex_cache_dir = os.path.abspath(tex_cache_dir)
        os.makedirs(tex_cache_dir, exist_ok=True)
        texmanager.TexManager._texcache = tex_cache_dir
    rc('text', usetex=True)
    rc('font', family='serif')
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


set_latex_plot_style()

n_phi_p = 10
n_phi_jobs = 1
run_phi_sweep = True
save_spectrum_frames = True
spectrum_frame_stride = 10
trajectory_save_every = 2
spectrum_trajectory_save_every = 8
trajectory_output_dtype = np.float32
linewidth_level_db = 3.0
linewidth_smooth_sigma_bins = 0.0
linewidth_trace_smooth_window = 3
plot_raw_linewidth_traces = True
linewidth_plot_ylim_mhz = (0.01, 300.0)
linewidth_plot_yticks_mhz = [0.01, 0.1, 1.0, 10.0, 100.0]
cos_plot_all_saved_points = True
cos_window_ns = 200.0  # Used only when cos_plot_all_saved_points is False.
cos_two_stage_long_tail_only = True
cos_long_tail_window_us = 1.0
# The historical "cos_*" names below now carry the circular average of the
# wrapped phase difference, kept to avoid rewriting the continuation plumbing.
spectrum_after_ramp_only = True
spectrum_post_ramp_buffer_tau = 0.0
spectrum_zero_padding_factor = 4
spectrum_max_saved_freq_points = 60001
use_two_stage_spectrum = True
Tmax_continuation_requested = 5.0e-7
Tmax_spectrum_requested = 5e-6
noise_amplitude = 1.0
two_stage_kappa_chunk_size = 10
long_spectrum_jobs = 10
vectorize_two_stage_kappa_chunks = True
show_two_stage_progress = True
show_long_run_worker_progress = True
long_run_progress_update_steps = 200
print_two_stage_messages = False
save_two_stage_progress_arrays = True
save_two_stage_progress_frames = True
save_two_stage_chunk_frames = False
average_continuation_history_across_noise = True

# These controls mirror linewidth_convergence_tmax_continuation.py.  The
# continuation handoff uses return_final_history, so saved trajectories can be
# decimated without losing the full-resolution 2*tau history needed for the
# next kappa segment.
dt_multiplier = 1.0
integration_scheme = "trapezoid"
integrator_theta = 0.5
integrator_max_iter = 5
delay_interpolation = "linear"  # None, "linear", or "cubic"
max_noise_substep_dt_tau_p = 1.0


def filename_float(value, precision=4):
    """Stable compact float label for filenames."""
    return (
        f"{float(value):.{precision}g}"
        .replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
    )


def apply_linewidth_integrator_controls(nd, noise_substeps):
    """Attach optional coarse-dt controls consumed by VCSEL.integrate()."""
    nd["integration_scheme"] = integration_scheme
    if delay_interpolation is not None:
        nd["delay_interp"] = delay_interpolation
    nd["noise_substeps"] = int(max(1, noise_substeps))
    return nd


def linewidth_resolution_floor_mhz(Tmax_seconds):
    """Approximate Hann-window 3 dB linewidth floor, returned in MHz."""
    return 1.44 / float(Tmax_seconds) * 1e-6


def moving_average_nan_safe(x, window):
    """Centered moving average that ignores NaNs and preserves missing regions."""
    x = np.asarray(x, dtype=float)
    window = int(max(1, window))
    if window <= 1 or x.size == 0:
        return x.copy()
    window = min(window, x.size)
    mask = np.isfinite(x).astype(float)
    x_filled = np.where(np.isfinite(x), x, 0.0)
    kernel = np.ones(window, dtype=float)
    num = np.convolve(x_filled, kernel, mode="same")
    den = np.convolve(mask, kernel, mode="same")
    return num / np.where(den == 0, np.nan, den)


def average_cos_phase_trace_from_phi(phi):
    """Average cos(phi_i - phi_0) over cases and non-reference lasers."""
    phi = np.asarray(phi)
    if phi.ndim != 3:
        raise ValueError(f"Expected phi with shape (n_cases,N_lasers,time), got {phi.shape}")
    if phi.shape[1] <= 1:
        return np.ones(phi.shape[-1], dtype=np.float32)
    phase_diff = phi - phi[:, 0:1, :]
    return np.mean(np.mean(np.cos(phase_diff), axis=0)[1:], axis=0).astype(np.float32)


def average_wrapped_phase_trace_from_phi(phi):
    """Absolute circular average of wrapped phi_i - phi_0 over cases/non-reference lasers."""
    phi = np.asarray(phi)
    if phi.ndim != 3:
        raise ValueError(f"Expected phi with shape (n_cases,N_lasers,time), got {phi.shape}")
    if phi.shape[1] <= 1:
        return np.zeros(phi.shape[-1], dtype=np.float32)
    wrapped_diff = np.angle(np.exp(1j * (phi[:, 1:, :] - phi[:, 0:1, :])))
    mean_phasor = np.mean(np.exp(1j * wrapped_diff), axis=(0, 1))
    return np.abs(np.angle(mean_phasor)).astype(np.float32)


def plot_order_parameter_panel(
    ax,
    order_param,
    kappa_c,
    font_size,
    pad=12,
    order_param_min=None,
    order_param_max=None,
):
    """Plot mean order parameter with optional min/max noise-realization envelope."""
    kappa_axis = np.asarray(kappa_c) * 1e-9
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


def linewidth_output_dirs(n_lasers):
    """Return output directories rooted under linewidth_estimation/{N}_lasers/phi_linewidth_continuation."""
    base_dir = LINEWIDTH_RESULTS_DIR / f"{int(n_lasers)}_lasers/phi_linewidth_continuation"
    return {
        "base": base_dir,
        "arrays": f"{base_dir}/numpy_arrays",
        "frames": f"{base_dir}/spectrum_frames_detuning",
        "tex_cache": f"{base_dir}/matplotlib_tex_cache",
    }


def linewidth_from_db_spectrum_reference_style(
    spectrum_db,
    freqs_hz,
    level_db=3.0,
    smooth_sigma_bins=0.0,
):
    """Peak-centered threshold-crossing linewidth in Hz."""
    spectrum_db = np.asarray(spectrum_db, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    finite = np.isfinite(spectrum_db) & np.isfinite(freqs_hz)
    if np.count_nonzero(finite) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.full_like(freqs_hz, np.nan)

    spectrum_db = spectrum_db[finite]
    freqs_hz = freqs_hz[finite]
    sort_idx = np.argsort(freqs_hz)
    spectrum_db = spectrum_db[sort_idx]
    freqs_hz = freqs_hz[sort_idx]
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        from scipy.ndimage import gaussian_filter1d

        spectrum_db = gaussian_filter1d(
            spectrum_db,
            sigma=float(smooth_sigma_bins),
            mode="nearest",
        )

    peak_idx = int(np.nanargmax(spectrum_db))
    peak_level = spectrum_db[peak_idx]
    target_level = peak_level - level_db
    shifted_freq = freqs_hz - freqs_hz[peak_idx]

    left = np.where(spectrum_db[:peak_idx + 1] <= target_level)[0]
    right = np.where(spectrum_db[peak_idx:] <= target_level)[0] + peak_idx
    if len(left) == 0 or len(right) == 0:
        return np.nan, np.nan, np.nan, target_level, 0.0, shifted_freq

    li = left[-1]
    ri = right[0]
    if li == peak_idx or ri == peak_idx:
        return np.nan, np.nan, np.nan, target_level, 0.0, shifted_freq

    freq_left = np.interp(
        target_level,
        [spectrum_db[li], spectrum_db[li + 1]],
        [shifted_freq[li], shifted_freq[li + 1]],
    )
    freq_right = np.interp(
        target_level,
        [spectrum_db[ri - 1], spectrum_db[ri]],
        [shifted_freq[ri - 1], shifted_freq[ri]],
    )
    linewidth = freq_right - freq_left
    return linewidth, freq_left, freq_right, target_level, 0.0, shifted_freq


def compute_optical_spectra_and_linewidths(
    E_all,
    E_tot,
    t,
    tau_p,
    nu_0,
    N_lasers,
    spectrum_start_time_seconds=0.0,
):
    """Compute normalized total/per-laser optical spectra and reported linewidths."""
    if len(t) < 2:
        raise ValueError("Need at least two saved samples for spectral analysis.")

    saved_dt_nd = float(np.median(np.diff(t)) / tau_p)
    fs = 1.0 / saved_dt_nd
    desired_df = 1e6 * tau_p
    spectrum_start_idx = int(np.searchsorted(t, spectrum_start_time_seconds, side="left"))
    spectrum_start_idx = min(max(spectrum_start_idx, 0), E_all.shape[-1] - 2)

    E_all_spec = E_all[:, :, spectrum_start_idx:]
    E_tot_spec = E_tot[:, spectrum_start_idx:]
    nperseg = np.shape(E_all_spec)[-1]
    if nperseg < 8:
        raise ValueError(
            f"Only {nperseg} saved samples are available for spectra. "
            "Decrease dt_multiplier/trajectory_save_every, reduce ramp time, "
            "or increase Tmax."
        )

    noverlap = nperseg // 2
    N_fft = max(
        int(np.ceil(fs / desired_df)),
        int(spectrum_zero_padding_factor * nperseg),
        nperseg,
    )

    h = 6.626e-34
    conversion = h * nu_0 / tau_p

    f, psd_tot = welch(
        E_tot_spec,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=N_fft,
        return_onesided=False,
        scaling="density",
    )
    psd_tot_watts = np.maximum(np.real(psd_tot), 0.0) * conversion
    spectrum_db = 10*np.log10(np.mean(psd_tot_watts, axis=0)/1e-3 + 1e-20)

    spectra_db = np.zeros((N_lasers, len(f)), dtype=np.float32)
    for i in range(N_lasers):
        _, psd_i = welch(
            E_all_spec[:, i, :],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=N_fft,
            return_onesided=False,
            scaling="density",
        )
        psd_i_watts = np.maximum(np.real(psd_i), 0.0) * conversion
        spectra_db[i] = 10*np.log10(np.mean(psd_i_watts, axis=0)/1e-3 + 1e-20)

    idx_sort = np.argsort(f)
    f_sorted = f[idx_sort]
    f_plot_min = -15.0
    f_plot_max = 15.0
    mask = (f_sorted >= f_plot_min*1e9*tau_p) & (f_sorted <= f_plot_max*1e9*tau_p)

    f_window = f_sorted[mask]
    spectrum_db = spectrum_db[idx_sort][mask]
    spectra_db = spectra_db[:, idx_sort][:, mask]

    spectrum_db_norm = (spectrum_db - np.max(spectrum_db)).astype(np.float32, copy=False)
    spectrum_laser_db_norm = (
        spectra_db - np.max(spectra_db, axis=1, keepdims=True)
    ).astype(np.float32, copy=False)

    laser_linewidths_mhz = np.full(N_lasers, np.nan, dtype=np.float32)
    for laser_idx in range(N_lasers):
        laser_linewidths_mhz[laser_idx], *_ = linewidth_from_db_spectrum_reference_style(
            spectrum_laser_db_norm[laser_idx],
            f_window/tau_p,
            level_db=linewidth_level_db,
            smooth_sigma_bins=linewidth_smooth_sigma_bins,
        )
        laser_linewidths_mhz[laser_idx] *= 1e-6

    linewidth_mhz, *_ = linewidth_from_db_spectrum_reference_style(
        spectrum_db_norm,
        f_window/tau_p,
        level_db=linewidth_level_db,
        smooth_sigma_bins=linewidth_smooth_sigma_bins,
    )
    linewidth_mhz *= 1e-6

    save_stride = max(
        1,
        int(np.ceil(len(f_window) / int(max(1, spectrum_max_saved_freq_points)))),
    )
    f_window_saved = f_window[::save_stride]
    spectrum_db_norm_saved = spectrum_db_norm[::save_stride]
    spectrum_laser_db_norm_saved = spectrum_laser_db_norm[:, ::save_stride]

    return {
        "f_sorted": f_window_saved,
        "f_window": f_window_saved,
        "spectrum_db_norm": spectrum_db_norm_saved,
        "spectrum_laser_db_norm": spectrum_laser_db_norm_saved,
        "linewidth_mhz": np.float32(linewidth_mhz),
        "laser_linewidths_mhz": laser_linewidths_mhz,
        "nperseg": nperseg,
        "spectrum_start_idx": spectrum_start_idx,
        "spectrum_save_stride": save_stride,
    }


LONG_RUN_PROGRESS_QUEUE = None
VCSEL_INTEGRATE_SUPPORTS_PROGRESS_CALLBACK = (
    "progress_callback" in inspect.signature(VCSEL.integrate).parameters
)


def init_long_run_progress_worker(progress_queue):
    """Install the parent-owned progress queue inside forked workers."""
    global LONG_RUN_PROGRESS_QUEUE
    LONG_RUN_PROGRESS_QUEUE = progress_queue


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
    """Call VCSEL.integrate with progress callback only when this import supports it."""
    if VCSEL_INTEGRATE_SUPPORTS_PROGRESS_CALLBACK:
        kwargs["progress_callback"] = progress_callback
    return vcsel.integrate(*args, **kwargs)


def run_fixed_kappa_spectrum_job_worker(job):
    """Worker-side long fixed-kappa spectral measurement."""
    k_job = int(job["k"])
    kappa_job = float(job["kappa"])
    phys_spec = dict(job["phys"])
    phys_spec["Tmax"] = float(job["Tmax_spectrum"])
    phys_spec["save_every"] = int(max(1, job.get("spectrum_trajectory_save_every", phys_spec.get("save_every", 1))))
    phys_spec["kappa_c_mat"] = VCSEL.build_coupling_matrix(
        time_arr=job["time_arr_spectrum"],
        kappa_initial=kappa_job,
        kappa_final=kappa_job,
        N_lasers=int(job["N_lasers"]),
        ramp_start=0.0,
        ramp_shape=1.0,
        tau=float(job["tau"]),
        scheme=job["coupling_scheme"],
        plot=False,
        dx=job["dx"],
        aMAT=job["aMAT"],
    ).astype(np.float32, copy=False)

    vcsel_spec = VCSEL(phys_spec)
    nd_spec = vcsel_spec.scale_params()
    nd_spec["integration_scheme"] = job["integration_scheme"]
    nd_spec["store_freqs"] = False
    if job["delay_interpolation"] is not None:
        nd_spec["delay_interp"] = job["delay_interpolation"]
    nd_spec["noise_substeps"] = int(max(1, job["noise_substeps"]))
    progress_callback, flush_progress = make_worker_progress_callback(job)

    try:
        t_spec, y_spec, freqs_spec = integrate_with_optional_progress_callback(
            vcsel_spec,
            job["history"],
            nd=nd_spec,
            progress=False,
            theta=float(job["integrator_theta"]),
            max_iter=int(job["integrator_max_iter"]),
            smooth_freqs=False,
            integration_scheme=job["integration_scheme"],
            return_final_history=False,
            message=f"fixed-kappa spectrum {k_job}",
            progress_callback=progress_callback,
        )
        flush_progress()
    except Exception as exc:
        flush_progress()
        raise RuntimeError(
            f"Long fixed-kappa spectrum failed for phi index {job['p']}, "
            f"kappa={kappa_job*1e-9:.3f} ns^-1."
        ) from exc
    if not np.all(np.isfinite(y_spec)):
        raise FloatingPointError(
            f"Non-finite state in long fixed-kappa spectrum for phi index {job['p']}, "
            f"kappa={kappa_job*1e-9:.3f} ns^-1."
        )

    N_lasers = int(job["N_lasers"])
    S_spec = np.maximum(y_spec[:, 1::3, :], 0.0)
    phi_spec = y_spec[:, 2::3, :]
    E_all_spec_full = (
        np.sqrt(S_spec.astype(np.float32, copy=False))
        * np.exp((1j * phi_spec).astype(np.complex64))
    ).astype(np.complex64, copy=False)
    E_tot_spec_full = np.sum(E_all_spec_full, axis=1)
    result = compute_optical_spectra_and_linewidths(
        E_all_spec_full,
        E_tot_spec_full,
        t_spec,
        float(job["tau_p"]),
        float(job["nu_0"]),
        N_lasers,
        spectrum_start_time_seconds=float(job["spectrum_post_ramp_buffer_tau"]) * float(job["tau"]),
    )
    cos_trace = average_wrapped_phase_trace_from_phi(phi_spec)
    cos_return_len = int(job.get("cos_spectrum_trace_len", len(cos_trace)))
    result["cos_trace"] = cos_trace[-cos_return_len:]
    order_param_cases = vcsel_spec.order_parameter(
        y_spec[:, :, -int(y_spec.shape[-1] / 2):]
    )
    result["order_param"] = np.float32(np.mean(order_param_cases))
    result["order_param_min"] = np.float32(np.min(order_param_cases))
    result["order_param_max"] = np.float32(np.max(order_param_cases))
    result["k"] = k_job
    result["kappa"] = kappa_job
    del order_param_cases
    del y_spec, freqs_spec
    del S_spec, phi_spec, E_all_spec_full, E_tot_spec_full
    return result


def run_fixed_kappa_spectrum_chunk_worker(jobs):
    """Run one submitted kappa chunk as one vectorized integration."""
    jobs = list(jobs)
    if not jobs:
        return []
    if len(jobs) == 1:
        return [run_fixed_kappa_spectrum_job_worker(jobs[0])]

    first_job = jobs[0]
    N_lasers = int(first_job["N_lasers"])
    tau = float(first_job["tau"])
    tau_p = float(first_job["tau_p"])
    Tmax_spectrum = float(first_job["Tmax_spectrum"])
    time_arr_spectrum = first_job["time_arr_spectrum"]

    histories = []
    phi_blocks = []
    kappa_blocks = []
    case_counts = []

    for job in jobs:
        if int(job["N_lasers"]) != N_lasers:
            raise ValueError("All jobs in a vectorized spectrum chunk must share N_lasers.")
        if not np.isclose(float(job["Tmax_spectrum"]), Tmax_spectrum):
            raise ValueError("All jobs in a vectorized spectrum chunk must share Tmax_spectrum.")
        if not np.allclose(job["time_arr_spectrum"], time_arr_spectrum):
            raise ValueError("All jobs in a vectorized spectrum chunk must share time_arr_spectrum.")

        history_job = np.asarray(job["history"])
        n_cases_job = history_job.shape[0]
        histories.append(history_job)
        case_counts.append(n_cases_job)

        phi_job = np.asarray(job["phys"]["phi_p_mat"])
        if phi_job.ndim == 0:
            phi_job = np.full((n_cases_job, N_lasers, N_lasers), float(phi_job))
        elif phi_job.ndim == 2:
            phi_job = np.repeat(phi_job[None, :, :], n_cases_job, axis=0)
        elif phi_job.ndim == 3 and phi_job.shape[0] == 1:
            phi_job = np.repeat(phi_job, n_cases_job, axis=0)
        elif phi_job.ndim == 3 and phi_job.shape[0] == n_cases_job:
            phi_job = phi_job
        else:
            raise ValueError(
                "phi_p_mat in a vectorized spectrum chunk must be scalar, "
                f"(N,N), (1,N,N), or ({n_cases_job},N,N); got {phi_job.shape}"
            )
        phi_blocks.append(phi_job)

        kappa_job = VCSEL.build_coupling_matrix(
            time_arr=time_arr_spectrum[:1],
            kappa_initial=float(job["kappa"]),
            kappa_final=float(job["kappa"]),
            N_lasers=N_lasers,
            ramp_start=0.0,
            ramp_shape=1.0,
            tau=tau,
            scheme=job["coupling_scheme"],
            plot=False,
            dx=job["dx"],
            aMAT=job["aMAT"],
        ).astype(np.float32, copy=False)
        kappa_blocks.append(np.repeat(kappa_job[0][None, :, :], n_cases_job, axis=0))

    history_all = np.concatenate(histories, axis=0)
    phi_all = np.concatenate(phi_blocks, axis=0)
    kappa_all = np.concatenate(kappa_blocks, axis=0).astype(np.float32, copy=False)

    phys_spec = dict(first_job["phys"])
    phys_spec["Tmax"] = Tmax_spectrum
    phys_spec["save_every"] = int(max(1, first_job.get("spectrum_trajectory_save_every", phys_spec.get("save_every", 1))))
    phys_spec["phi_p_mat"] = phi_all
    phys_spec["kappa_c_mat"] = kappa_all

    vcsel_spec = VCSEL(phys_spec)
    nd_spec = vcsel_spec.scale_params()
    nd_spec["integration_scheme"] = first_job["integration_scheme"]
    nd_spec["kappa_case_dependent"] = True
    nd_spec["store_freqs"] = False
    if first_job["delay_interpolation"] is not None:
        nd_spec["delay_interp"] = first_job["delay_interpolation"]
    nd_spec["noise_substeps"] = int(max(1, first_job["noise_substeps"]))
    progress_callback, flush_progress = make_worker_progress_callback(first_job)

    try:
        t_spec, y_spec, freqs_spec = integrate_with_optional_progress_callback(
            vcsel_spec,
            history_all,
            nd=nd_spec,
            progress=False,
            theta=float(first_job["integrator_theta"]),
            max_iter=int(first_job["integrator_max_iter"]),
            smooth_freqs=False,
            integration_scheme=first_job["integration_scheme"],
            return_final_history=False,
            message=(
                f"fixed-kappa spectrum chunk "
                f"{int(jobs[0]['k'])}-{int(jobs[-1]['k'])}"
            ),
            progress_callback=progress_callback,
        )
        flush_progress()
    except Exception as exc:
        flush_progress()
        raise RuntimeError(
            f"Vectorized long fixed-kappa spectrum failed for phi index {first_job['p']}, "
            f"k={int(jobs[0]['k'])}-{int(jobs[-1]['k'])}."
        ) from exc
    if not np.all(np.isfinite(y_spec)):
        raise FloatingPointError(
            f"Non-finite state in vectorized long fixed-kappa spectrum for phi index "
            f"{first_job['p']}, k={int(jobs[0]['k'])}-{int(jobs[-1]['k'])}."
        )

    results = []
    offset = 0
    for job, n_cases_job in zip(jobs, case_counts):
        y_job = y_spec[offset:offset + n_cases_job]
        offset += n_cases_job
        S_spec = np.maximum(y_job[:, 1::3, :], 0.0)
        phi_spec = y_job[:, 2::3, :]
        E_all_spec_full = (
            np.sqrt(S_spec.astype(np.float32, copy=False))
            * np.exp((1j * phi_spec).astype(np.complex64))
        ).astype(np.complex64, copy=False)
        E_tot_spec_full = np.sum(E_all_spec_full, axis=1)
        result = compute_optical_spectra_and_linewidths(
            E_all_spec_full,
            E_tot_spec_full,
            t_spec,
            tau_p,
            float(job["nu_0"]),
            N_lasers,
            spectrum_start_time_seconds=float(job["spectrum_post_ramp_buffer_tau"]) * tau,
        )
        cos_trace = average_wrapped_phase_trace_from_phi(phi_spec)
        cos_return_len = int(job.get("cos_spectrum_trace_len", len(cos_trace)))
        result["cos_trace"] = cos_trace[-cos_return_len:]
        order_param_cases = vcsel_spec.order_parameter(
            y_job[:, :, -int(y_job.shape[-1] / 2):]
        )
        result["order_param"] = np.float32(np.mean(order_param_cases))
        result["order_param_min"] = np.float32(np.min(order_param_cases))
        result["order_param_max"] = np.float32(np.max(order_param_cases))
        result["k"] = int(job["k"])
        result["kappa"] = float(job["kappa"])
        results.append(result)
        del y_job, S_spec, phi_spec, E_all_spec_full, E_tot_spec_full, order_param_cases

    del history_all, phi_all, kappa_all, y_spec, freqs_spec
    return results


def run_phi_p_continuation(p):
    # --- Parameters ---
    alpha = 2
    tau_p = 5.4e-12
    tau_n = 0.25e-9
    g0 = 8.75e-4 * 1e9
    N0 = 2.86e5
    s = 4e-6 
    q = 1.602e-19
    beta = 1.e-3
    # kappa_c = 12e9
    tau = 1e-9  # delay (s)
    eta = 0.9
    current_threshold = 3

    I = eta*current_threshold * q/ tau_n * (N0 + 1/(g0*tau_p))

    # print(f"p={g0*tau_p * (I*tau_n/(q) - N0) - 1:.3f}")


    self_feedback = 0.0
    coupling = 1.0
    N_lasers = 2

    detuning_ghz = 4.0
    detuning = detuning_ghz  # detuning (GHz)
    delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
    if N_lasers == 1:
        delta_distribution = np.array([0.0])
    else:
        delta_distribution = np.linspace(-delta/2, delta/2, N_lasers)


    dt = float(dt_multiplier) * tau_p
    Tmax_requested = float(Tmax_continuation_requested if use_two_stage_spectrum else Tmax_spectrum_requested)


    steps = int(Tmax_requested / dt)
    delay_steps = int(tau / dt)
    if delay_steps < 2:
        raise ValueError(
            f"dt={dt/tau_p:g} tau_p leaves only {delay_steps} delay samples. "
            "Decrease dt_multiplier so tau/dt is at least 2."
        )
    if steps <= 2 * delay_steps + 4:
        raise ValueError(
            f"Tmax={Tmax_requested*1e6:g} us and dt={dt/tau_p:g} tau_p give "
            f"only {steps} samples, but continuation needs more than "
            f"{2 * delay_steps} samples for the 2*tau history."
        )

    # Use the same integer-step duration that scale_params/integrate will use.
    Tmax = steps * dt
    time_arr = np.arange(steps, dtype=float) * dt
    segment_start = steps // 2
    segment_len = steps - segment_start
    saved_dt = dt * int(max(1, trajectory_save_every))
    spectrum_saved_dt = dt * int(max(1, spectrum_trajectory_save_every))
    saved_steps_est = max(1, int(np.ceil(steps / int(max(1, trajectory_save_every)))))
    spectrum_steps_est = int(float(Tmax_spectrum_requested) / dt) if use_two_stage_spectrum else 0
    spectrum_saved_steps_est = (
        max(1, int(np.ceil(spectrum_steps_est / int(max(1, spectrum_trajectory_save_every)))))
        if use_two_stage_spectrum
        else 0
    )
    if use_two_stage_spectrum and cos_two_stage_long_tail_only:
        cos_cont_trace_len = 0
        cos_spectrum_trace_len = min(
            spectrum_saved_steps_est,
            max(1, int(np.ceil(float(cos_long_tail_window_us)*1e-6 / spectrum_saved_dt))),
        )
    elif cos_plot_all_saved_points:
        cos_cont_trace_len = saved_steps_est
        cos_spectrum_trace_len = spectrum_saved_steps_est
    else:
        cos_cont_trace_len = min(
            saved_steps_est,
            max(1, int(np.ceil(cos_window_ns*1e-9 / saved_dt))),
        )
        cos_spectrum_trace_len = 0
    cos_trace_len = cos_cont_trace_len + cos_spectrum_trace_len
    cos_cont_time_axis = np.arange(cos_cont_trace_len, dtype=float) * saved_dt
    if use_two_stage_spectrum and cos_two_stage_long_tail_only:
        cos_spectrum_time_offset = max(
            0,
            spectrum_saved_steps_est - cos_spectrum_trace_len,
        ) * spectrum_saved_dt
    else:
        cos_spectrum_time_offset = (
            cos_cont_time_axis[-1] + spectrum_saved_dt
            if cos_cont_trace_len > 0
            else 0.0
        )
    cos_spectrum_time_axis = (
        cos_spectrum_time_offset
        + np.arange(cos_spectrum_trace_len, dtype=float) * spectrum_saved_dt
    )
    cos_time_axis_seconds = np.concatenate([cos_cont_time_axis, cos_spectrum_time_axis])

    resolution = 200
    output_dirs = linewidth_output_dirs(N_lasers)
    save_dir = output_dirs["arrays"]
    frame_dir = output_dirs["frames"]
    tex_cache_dir = f"{output_dirs['tex_cache']}/phi_{p:03d}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    set_latex_plot_style(tex_cache_dir=tex_cache_dir)
    coupling_scheme = 'CUSTOM'  # 'ATA', 'NN' or 'RANDOM'
    ramp_start = 2

    dx=1.0
    aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)

    kappa_c = np.linspace(0e9,40e9,resolution)
    detuning_label = f"{detuning_ghz:.2f}"



    phi_p_vals = np.array([np.linspace(0,2*np.pi,n_phi_p)[p]])
    phi_p_value = float(phi_p_vals[0])
    noise_substeps = 1
    if max_noise_substep_dt_tau_p is not None:
        noise_substeps = max(
            1,
            int(np.ceil((dt / tau_p) / float(max_noise_substep_dt_tau_p))),
        )

    scheme_label = f"scheme{integration_scheme.lower()}"
    dt_label = f"dtmult{filename_float(dt / tau_p)}"
    interp_label = f"interp{delay_interpolation or 'none'}"
    substep_label = f"nsub{noise_substeps}"
    save_label = f"saveevery{int(max(1, trajectory_save_every))}"
    stage_label = (
        f"twostage_contus{filename_float(Tmax*1e6)}_specus{filename_float(float(Tmax_spectrum_requested)*1e6)}"
        if use_two_stage_spectrum
        else f"singlestage_tmaxus{filename_float(Tmax*1e6)}"
    )
    phi_p_label = (
        f"phiidx{p:03d}_phipi{phi_p_value/np.pi:.4f}_"
        f"{stage_label}_{scheme_label}_{dt_label}_{save_label}_{interp_label}_{substep_label}"
    )
    run_label = f"self{self_feedback:.2f}_detuning{detuning_label}_{phi_p_label}"

    n_iterations = 50

    phys = {
        'tau_p': tau_p,
        'tau_n': tau_n,
        'g0': g0,
        'N0': N0,
        'N_bar': N0 + 1/(g0*tau_p),
        's': s,
        'beta': beta,
        'kappa_c_mat': None,
        'phi_p_mat': np.ones(shape=(n_iterations,N_lasers,N_lasers))*phi_p_vals[:,None,None],
        'I': I,
        'q': q,
        'alpha': alpha,
        'delta': delta_distribution,
        'coupling': coupling,     
        'self_feedback': self_feedback, 
        'noise_amplitude': noise_amplitude,
        'dt': dt,
        'Tmax': Tmax,
        'tau': tau,
        'N_lasers': N_lasers,
        'noise_substeps': noise_substeps,
        'save_every': int(max(1, trajectory_save_every)),
        'output_dtype': trajectory_output_dtype,
        'max_output_gb': None,
    }




    # t, y, freqs = vcsel.integrate(history, nd=nd, progress=True)



    ramp_start = 10
    ramp_shape = 10


    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=0,
        kappa_final=kappa_c[1],
        N_lasers=N_lasers,
        ramp_start=ramp_start,
        ramp_shape=ramp_shape,
        tau=tau,
        scheme=coupling_scheme,
        plot=False,
        dx=dx,
        aMAT=aMAT,
    )
    kappa_arr = kappa_arr.astype(np.float32, copy=False)

    phys['kappa_c_mat'] = kappa_arr
    delta_distribution_ghz = np.asarray(phys['delta'], dtype=float) / (2*np.pi*1e9)
    detuning_vals_ghz = np.tile(delta_distribution_ghz[:, None], (1, len(kappa_c)))

    vcsel = VCSEL(phys)
    nd = apply_linewidth_integrator_controls(vcsel.scale_params(), noise_substeps)



    n_cases = n_iterations


    history, freq_history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)
    if average_continuation_history_across_noise:
        history = average_history_across_cases(history)



    from scipy.signal import butter, filtfilt, decimate, welch
    import warnings
    import psutil
    import matplotlib 
    # matplotlib.use('Agg')  

    # Calculate phi_p based on free running wavelength lambda_0 and delay tau
    lambda_0 = 910e-9  # meters (example value, update as needed)
    c = 3e8  # speed of light (m/s)
    nu_0 = c / lambda_0  # optical frequency (Hz)
    phi_p = phys['phi_p_mat'][0]#np.pi#(2 * np.pi * nu_0 * tau) % (2 * np.pi)  # phase shift due to delay, wrapped to [0, 2π]

    # print(f"Calculated phi_p: {phi_p:.4f} radians ({phi_p/np.pi:.4f} π)")



    y = None
    kappa_prev = 0.0e9 
    dtype = np.float32  # cut memory in half
    spectrum_db_norm_list = None
    spectrum_laser_db_norm_list = None
    cos_phase_diff_time = np.full((len(kappa_c), cos_trace_len), np.nan, dtype=dtype)
    order_param = np.full(len(kappa_c), np.nan, dtype=dtype)
    order_param_min = np.full(len(kappa_c), np.nan, dtype=dtype)
    order_param_max = np.full(len(kappa_c), np.nan, dtype=dtype)
    linewidth_trace_mhz = np.full(len(kappa_c), np.nan, dtype=dtype)
    laser_linewidth_traces_mhz = np.full((N_lasers, len(kappa_c)), np.nan, dtype=dtype)
    spectrum_jobs = []
    pending_spectrum_futures = []
    spectrum_future_progress_id = {}
    spectrum_progress_bars = {}
    spectrum_progress_queue = None
    next_progress_position = 1
    spectrum_executor = None
    completed_spectrum_indices = set()
    submitted_spectrum_chunks = 0
    f_sorted = None
    f_window = None

    import sys



    show_kappa_progress = n_phi_jobs == 1
    pbar = tqdm(
        kappa_c,
        desc=f"phi {p + 1}/{n_phi_p}",
        unit="step",
        leave=True,
        disable=not show_kappa_progress,
    )

    Tmax_for_linewidth = Tmax
    Tmax_spectrum = None
    time_arr_spectrum = None

    if use_two_stage_spectrum:
        spectrum_steps = int(float(Tmax_spectrum_requested) / dt)
        if spectrum_steps <= 2 * delay_steps + 4:
            raise ValueError(
                f"Tmax_spectrum_requested={float(Tmax_spectrum_requested)*1e6:g} us gives "
                f"only {spectrum_steps} samples, but spectral measurement needs more than "
                f"{2 * delay_steps} samples for the 2*tau history."
            )
        Tmax_spectrum = spectrum_steps * dt
        Tmax_for_linewidth = Tmax_spectrum
        time_arr_spectrum = np.arange(spectrum_steps, dtype=float) * dt
        spectrum_mp_context = mp.get_context("fork")
        if show_long_run_worker_progress and show_two_stage_progress:
            spectrum_progress_queue = spectrum_mp_context.Queue()
            if not VCSEL_INTEGRATE_SUPPORTS_PROGRESS_CALLBACK:
                print(
                    "Long-run step progress requires reloading the updated vcsel_lib. "
                    "This run will still work, but long-run bars will update only when "
                    "batches finish."
                )
            spectrum_executor = ProcessPoolExecutor(
                max_workers=int(max(1, long_spectrum_jobs)),
                mp_context=spectrum_mp_context,
                initializer=init_long_run_progress_worker,
                initargs=(spectrum_progress_queue,),
            )
        else:
            spectrum_executor = ProcessPoolExecutor(
                max_workers=int(max(1, long_spectrum_jobs)),
                mp_context=spectrum_mp_context,
            )
        if print_two_stage_messages:
            tqdm.write(
                f"phi {p + 1}/{n_phi_p}: short continuation over {len(kappa_c)} kappa values; "
                f"submitting long spectra in chunks of {int(max(1, two_stage_kappa_chunk_size))} "
                f"on {int(max(1, long_spectrum_jobs))} worker(s)."
            )

    def make_spectrum_job(k_job, kappa_job, history_job):
        phys_for_job = dict(phys)
        phys_for_job["kappa_c_mat"] = None
        return {
            "k": int(k_job),
            "kappa": float(kappa_job),
            "history": history_job.copy(),
            "phys": phys_for_job,
            "Tmax_spectrum": Tmax_spectrum,
            "time_arr_spectrum": time_arr_spectrum,
            "N_lasers": N_lasers,
            "tau": tau,
            "tau_p": tau_p,
            "nu_0": nu_0,
            "coupling_scheme": coupling_scheme,
            "dx": dx,
            "aMAT": aMAT,
            "noise_substeps": noise_substeps,
            "integrator_theta": integrator_theta,
            "integrator_max_iter": integrator_max_iter,
            "integration_scheme": integration_scheme,
            "delay_interpolation": delay_interpolation,
            "spectrum_post_ramp_buffer_tau": spectrum_post_ramp_buffer_tau,
            "cos_spectrum_trace_len": cos_spectrum_trace_len,
            "spectrum_trajectory_save_every": int(max(1, spectrum_trajectory_save_every)),
            "p": p,
        }

    def make_long_run_progress_bar(progress_id, chunk_indices):
        nonlocal next_progress_position
        if spectrum_progress_queue is None:
            return
        total_steps = max(1, spectrum_steps - 2 * delay_steps)
        desc = (
            f"phi {p + 1}/{n_phi_p} long "
            f"k={chunk_indices[0]}-{chunk_indices[-1]}"
        )
        spectrum_progress_bars[progress_id] = tqdm(
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

    def submit_spectrum_job_chunk():
        nonlocal spectrum_jobs, submitted_spectrum_chunks
        if not spectrum_jobs:
            return
        chunk = spectrum_jobs
        spectrum_jobs = []
        submitted_spectrum_chunks += 1
        chunk_indices = [int(job["k"]) for job in chunk]
        progress_id = f"phi{p}_chunk{submitted_spectrum_chunks}"
        if print_two_stage_messages:
            tqdm.write(
                f"phi {p + 1}/{n_phi_p}: submitted long-spectrum batch "
                f"{submitted_spectrum_chunks} for k={chunk_indices[0]}-{chunk_indices[-1]}."
            )
        if vectorize_two_stage_kappa_chunks:
            make_long_run_progress_bar(progress_id, chunk_indices)
            chunk = attach_long_run_progress(chunk, progress_id)
            future = spectrum_executor.submit(run_fixed_kappa_spectrum_chunk_worker, chunk)
            pending_spectrum_futures.append(future)
            spectrum_future_progress_id[future] = progress_id
        else:
            for job in chunk:
                single_progress_id = f"{progress_id}_k{int(job['k'])}"
                make_long_run_progress_bar(single_progress_id, [int(job["k"])])
                attach_long_run_progress([job], single_progress_id)
                future = spectrum_executor.submit(run_fixed_kappa_spectrum_job_worker, job)
                pending_spectrum_futures.append(future)
                spectrum_future_progress_id[future] = single_progress_id

    def store_spectrum_results(spectrum_results):
        nonlocal f_sorted, f_window, spectrum_db_norm_list, spectrum_laser_db_norm_list
        if not spectrum_results:
            return
        spectrum_results = sorted(spectrum_results, key=lambda item: item["k"])
        if f_window is None:
            f_sorted = spectrum_results[0]["f_sorted"]
            f_window = spectrum_results[0]["f_window"]
            n_freqs = len(f_window)
            spectrum_db_norm_list = np.full((resolution, n_freqs), np.nan, dtype=dtype)
            spectrum_laser_db_norm_list = np.full((resolution, N_lasers, n_freqs), np.nan, dtype=dtype)

        for result in spectrum_results:
            k_result = int(result["k"])
            if len(result["f_window"]) != len(f_window) or not np.allclose(result["f_window"], f_window):
                raise ValueError("Long fixed-kappa spectra do not share the same frequency grid.")
            spectrum_db_norm_list[k_result] = result["spectrum_db_norm"]
            spectrum_laser_db_norm_list[k_result] = result["spectrum_laser_db_norm"]
            linewidth_trace_mhz[k_result] = result["linewidth_mhz"]
            laser_linewidth_traces_mhz[:, k_result] = result["laser_linewidths_mhz"]
            if "order_param" in result:
                order_param[k_result] = result["order_param"]
            if "order_param_min" in result:
                order_param_min[k_result] = result["order_param_min"]
            if "order_param_max" in result:
                order_param_max[k_result] = result["order_param_max"]
            cos_trace = result.get("cos_trace", None)
            if cos_trace is not None and cos_spectrum_trace_len > 0:
                cos_trace = np.asarray(cos_trace, dtype=dtype)
                cos_tail = cos_trace[-cos_spectrum_trace_len:]
                start = cos_cont_trace_len + cos_spectrum_trace_len - len(cos_tail)
                stop = cos_cont_trace_len + cos_spectrum_trace_len
                cos_phase_diff_time[k_result, start:stop] = cos_tail

        del spectrum_results
        gc.collect()

    def save_two_stage_progress_arrays():
        if not (use_two_stage_spectrum and save_two_stage_progress_arrays):
            return
        if f_window is None or spectrum_db_norm_list is None:
            return
        np.save(f"{save_dir}/spectrum_db_norm_list_{run_label}.npy", spectrum_db_norm_list)
        np.save(f"{save_dir}/f_sorted_{run_label}.npy", f_sorted)
        np.save(f"{save_dir}/f_window_{run_label}.npy", f_window)
        np.save(f"{save_dir}/kappa_c_{run_label}.npy", kappa_c)
        np.save(f"{save_dir}/detuning_vals_ghz_{run_label}.npy", detuning_vals_ghz)
        np.save(f"{save_dir}/phi_p_{run_label}.npy", np.array([phi_p_value]))
        np.save(f"{save_dir}/linewidth_E_fields_mhz_{run_label}.npy", laser_linewidth_traces_mhz)
        np.save(f"{save_dir}/linewidth_Etot_mhz_{run_label}.npy", linewidth_trace_mhz)
        np.save(f"{save_dir}/order_parameter_{run_label}.npy", order_param)
        np.save(f"{save_dir}/order_parameter_min_{run_label}.npy", order_param_min)
        np.save(f"{save_dir}/order_parameter_max_{run_label}.npy", order_param_max)
        np.save(f"{save_dir}/wrapped_phase_diff_time_{run_label}.npy", cos_phase_diff_time)
        np.save(f"{save_dir}/wrapped_phase_time_axis_seconds_{run_label}.npy", cos_time_axis_seconds)
        np.savez(
            f"{save_dir}/two_stage_progress_{run_label}.npz",
            completed_kappa_indices=np.array(sorted(completed_spectrum_indices), dtype=int),
            submitted_chunks=int(submitted_spectrum_chunks),
            total_kappa_values=int(len(kappa_c)),
        )

    def save_two_stage_progress_frame(k_completed=None):
        if not (use_two_stage_spectrum and save_spectrum_frames and save_two_stage_progress_frames):
            return
        if f_window is None or spectrum_db_norm_list is None:
            return

        font_size = 18
        f_plot = f_window/tau_p
        f_plot_axis_min = f_plot[0] * 1e-9
        f_plot_axis_max = f_plot[-1] * 1e-9
        f_display_bound = np.floor(min(abs(f_plot_axis_min), abs(f_plot_axis_max)))
        if f_display_bound >= 1:
            f_plot_axis_min = -f_display_bound
            f_plot_axis_max = f_display_bound

        local_cos_len = cos_phase_diff_time.shape[1]
        local_time_seconds = cos_time_axis_seconds[:local_cos_len]
        if local_time_seconds[-1] >= 1e-6:
            local_time = local_time_seconds * 1e6
            local_time_label = r'Time ($\mu$s)'
        else:
            local_time = local_time_seconds * 1e9
            local_time_label = r'Time (ns)'

        fig = plt.figure(figsize=(18, 10), dpi=220)
        width_ratios = [1]*30
        width_ratios[8] = 0.25
        gs = fig.add_gridspec(20, 30, height_ratios=[1]*20, width_ratios=width_ratios, hspace=0.3)

        ax0 = fig.add_subplot(gs[1:8, 0:-3])
        im0 = ax0.imshow(
            cos_phase_diff_time[:, :local_cos_len],
            aspect='auto',
            extent=[local_time[0], local_time[-1], kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
            origin='lower',
            cmap='jet_r',
            vmin=0.0,
            vmax=np.pi,
            rasterized=True,
        )
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.02)
        cbar0.set_label(r'$|\langle\Delta\phi\rangle_{\rm circ}|$ (rad)', fontsize=font_size, labelpad=0)
        cbar0.set_ticks([0.0, 0.5*np.pi, np.pi])
        cbar0.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
        cbar0.ax.tick_params(labelsize=font_size)
        ax0.set_ylabel(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=font_size)
        ax0.set_xlabel(local_time_label, fontsize=font_size)
        ax0.set_xlim(local_time[0], local_time[-1])
        ax0.set_xticks(np.linspace(local_time[0], local_time[-1], 6))
        ax0.set_title(
            r'$\phi_p={:+.2f}\pi,\ \delta_i=[{}]\,\mathrm{{GHz}},\ \Delta t={:.1f}\tau_p$'.format(
                phi_p[0,0]/np.pi,
                ", ".join(f"{d:.1f}" for d in delta_distribution_ghz),
                dt/tau_p,
            ),
            fontsize=font_size,
            pad=12,
        )
        ax0.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
        ax0.tick_params(axis='both', labelsize=font_size)

        ax_order = fig.add_subplot(gs[1:8, -2:])
        plot_order_parameter_panel(
            ax_order,
            order_param,
            kappa_c,
            font_size,
            pad=12,
            order_param_min=order_param_min,
            order_param_max=order_param_max,
        )

        ax1 = fig.add_subplot(gs[11:, 1:8])
        spectrum_cmap = plt.get_cmap('jet').copy()
        spectrum_cmap.set_bad(color='white')
        im = ax1.imshow(
            spectrum_db_norm_list,
            aspect='auto',
            extent=[f_plot_axis_min, f_plot_axis_max, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
            origin='lower',
            cmap=spectrum_cmap,
            rasterized=True,
        )
        im.set_clim(-100, 0)
        cbar1 = fig.colorbar(im, ax=ax1, pad=0.05)
        cbar1.set_label(r'Power (dBm)', fontsize=font_size, labelpad=12)
        cbar1.ax.tick_params(labelsize=font_size)
        ax1.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
        ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
        ax1.set_title(r"$|\mathcal{F}\left(E_{tot}\right)|^2$", fontsize=font_size, pad=12)
        ax1.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
        ax1.set_xticks(np.linspace(f_plot_axis_min, f_plot_axis_max, 3))
        ax1.set_xlim(f_plot_axis_min, f_plot_axis_max)
        ax1.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
        ax1.tick_params(axis='both', labelsize=font_size)
        for delta_ghz in delta_distribution_ghz:
            ax1.plot(
                np.full_like(kappa_c, delta_ghz, dtype=float),
                kappa_c*1e-9,
                color='black',
                linestyle='--',
                linewidth=2,
                alpha=0.5,
            )

        ax2 = fig.add_subplot(gs[11:, 13:29])
        field_colors = plt.get_cmap("tab10")
        for laser_idx in range(N_lasers):
            laser_linewidth = laser_linewidth_traces_mhz[laser_idx]
            valid_laser_linewidth = np.isfinite(laser_linewidth) & (laser_linewidth > 0)
            if np.any(valid_laser_linewidth):
                laser_color = field_colors(laser_idx % field_colors.N)
                if plot_raw_linewidth_traces:
                    ax2.plot(
                        kappa_c[valid_laser_linewidth] * 1e-9,
                        laser_linewidth[valid_laser_linewidth],
                        color=laser_color,
                        linewidth=0.8,
                        alpha=0.18,
                        zorder=1,
                    )
                laser_linewidth_smooth = moving_average_nan_safe(
                    laser_linewidth,
                    linewidth_trace_smooth_window,
                )
                valid_laser_smooth = np.isfinite(laser_linewidth_smooth) & (laser_linewidth_smooth > 0)
                ax2.plot(
                    kappa_c[valid_laser_smooth] * 1e-9,
                    laser_linewidth_smooth[valid_laser_smooth],
                    color=laser_color,
                    linewidth=1.4,
                    alpha=0.55,
                    zorder=2,
                    label=rf"$E_{laser_idx + 1}$",
                )
        valid_linewidth = np.isfinite(linewidth_trace_mhz) & (linewidth_trace_mhz > 0)
        if plot_raw_linewidth_traces and np.any(valid_linewidth):
            ax2.plot(
                kappa_c[valid_linewidth] * 1e-9,
                linewidth_trace_mhz[valid_linewidth],
                color="black",
                linewidth=1.0,
                alpha=0.2,
                zorder=3,
            )
        linewidth_trace_smooth = moving_average_nan_safe(
            linewidth_trace_mhz,
            linewidth_trace_smooth_window,
        )
        valid_linewidth_smooth = np.isfinite(linewidth_trace_smooth) & (linewidth_trace_smooth > 0)
        ax2.plot(
            kappa_c[valid_linewidth_smooth] * 1e-9,
            linewidth_trace_smooth[valid_linewidth_smooth],
            color="black",
            linewidth=3.0,
            zorder=4,
            label=r"$E_{tot}$",
        )
        resolution_mhz = linewidth_resolution_floor_mhz(Tmax_for_linewidth)
        if linewidth_plot_ylim_mhz[0] <= resolution_mhz <= linewidth_plot_ylim_mhz[1]:
            ax2.axhline(
                resolution_mhz,
                color="red",
                linestyle="--",
                linewidth=2.0,
                alpha=0.8,
                zorder=2,
                label=r"$1.44/T_\mathrm{max}$",
            )
        linewidth_font_size = font_size + 4
        ax2.set_xlabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=linewidth_font_size, labelpad=10)
        ax2.set_ylabel("")
        ax2.set_title(rf"{linewidth_level_db:.0f} dB linewidth (MHz)", fontsize=linewidth_font_size, pad=12)
        ax2.set_xlim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
        ax2.set_yscale("log")
        ax2.set_ylim(*linewidth_plot_ylim_mhz)
        ax2.set_yticks(linewidth_plot_yticks_mhz)
        ax2.grid(True, linestyle="--", alpha=0.4, which="both")
        ax2.set_xticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
        ax2.tick_params(axis='both', labelsize=linewidth_font_size)
        ax2.legend(fontsize=linewidth_font_size - 10, loc="upper right")

        os.makedirs(frame_dir, exist_ok=True)
        live_filename = f"{frame_dir}/phi_{p:03d}.png"
        plt.savefig(live_filename, bbox_inches='tight')
        if save_two_stage_chunk_frames and k_completed is not None:
            progress_filename = f"{frame_dir}/phi_{p:03d}_progress_k{k_completed:03d}.png"
            plt.savefig(progress_filename, bbox_inches='tight')
        plt.close(fig)
        plt.cla(); plt.clf()
        plt.close('all')
        del fig, gs, ax0, ax_order, ax1, ax2
        del im0, im, cbar0, cbar1

    def process_spectrum_future(future):
        spectrum_results = future.result()
        if isinstance(spectrum_results, dict):
            spectrum_results = [spectrum_results]
        result_indices = [int(result["k"]) for result in spectrum_results]
        store_spectrum_results(spectrum_results)
        completed_spectrum_indices.update(result_indices)
        max_completed = max(result_indices)
        if print_two_stage_messages:
            tqdm.write(
                f"phi {p + 1}/{n_phi_p}: collected long spectra for "
                f"k={min(result_indices)}-{max_completed}; "
                f"{len(completed_spectrum_indices)}/{len(kappa_c)} kappa values complete."
            )
        save_two_stage_progress_arrays()
        save_two_stage_progress_frame(k_completed=max_completed)
        close_long_run_progress_bar(future)

    def drain_long_run_progress_queue():
        if spectrum_progress_queue is None:
            return
        while True:
            try:
                progress_id, n_steps = spectrum_progress_queue.get_nowait()
            except queue_module.Empty:
                break
            bar = spectrum_progress_bars.get(progress_id)
            if bar is not None:
                remaining = bar.total - bar.n if bar.total is not None else int(n_steps)
                bar.update(min(int(n_steps), max(0, remaining)))

    def close_long_run_progress_bar(future):
        progress_id = spectrum_future_progress_id.pop(future, None)
        if progress_id is None:
            return
        drain_long_run_progress_queue()
        bar = spectrum_progress_bars.pop(progress_id, None)
        if bar is not None:
            if bar.total is not None and bar.n < bar.total:
                bar.update(bar.total - bar.n)
            bar.close()

    def collect_finished_spectrum_chunks(block=False):
        if not pending_spectrum_futures:
            return
        drain_long_run_progress_queue()
        if block:
            futures_to_collect = list(pending_spectrum_futures)
            pending_spectrum_futures.clear()
            collect_bar = tqdm(
                total=len(futures_to_collect),
                desc=f"phi {p + 1}/{n_phi_p} long spectra complete",
                unit="batch" if vectorize_two_stage_kappa_chunks else "spectrum",
                position=0,
                leave=True,
                disable=not show_two_stage_progress,
                dynamic_ncols=True,
            )
            pending_block = set(futures_to_collect)
            try:
                while pending_block:
                    drain_long_run_progress_queue()
                    done_now = [future for future in pending_block if future.done()]
                    for future in done_now:
                        pending_block.remove(future)
                        process_spectrum_future(future)
                        collect_bar.update(1)
                    if pending_block:
                        time.sleep(0.2)
            finally:
                drain_long_run_progress_queue()
                collect_bar.close()
            return

        for future in list(pending_spectrum_futures):
            if future.done():
                pending_spectrum_futures.remove(future)
                process_spectrum_future(future)

    for k, kappa in enumerate(pbar):
        delta_distribution_ghz = np.asarray(phys['delta'], dtype=float) / (2*np.pi*1e9)
        pbar.set_postfix(
            kappa_ns=f"{kappa*1e-9:.2f}",
            delta_ghz="[" + ", ".join(f"{d:.2f}" for d in delta_distribution_ghz) + "]",
        )

        # Slice the ramp for this segment
        if k > 0:
            kappa_arr = VCSEL.build_coupling_matrix(
                time_arr=time_arr,
                kappa_initial=kappa_c[k-1],
                kappa_final=kappa_c[k],
                N_lasers=N_lasers,
                ramp_start=ramp_start,
                ramp_shape=ramp_shape,
                tau=tau,
                scheme=coupling_scheme,
                plot=False,
                dx=dx,
                aMAT=aMAT,
            )
            kappa_arr = kappa_arr.astype(np.float32, copy=False)
        phys['kappa_c_mat'] = kappa_arr
        vcsel = VCSEL(phys)
        nd = apply_linewidth_integrator_controls(vcsel.scale_params(), noise_substeps)
        try:
            t, y_scaled, freqs, final_history = vcsel.integrate(
                history,
                nd=nd,
                progress=False,
                theta=integrator_theta,
                max_iter=integrator_max_iter,
                smooth_freqs=False,
                integration_scheme=integration_scheme,
                return_final_history=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Integration failed for phi index {p}, "
                f"kappa={kappa*1e-9:.3f} ns^-1, dt={dt/tau_p:g} tau_p. "
                "Try decreasing dt_multiplier, increasing integrator_max_iter, "
                "or using delay_interpolation='linear'."
            ) from exc
        if not np.all(np.isfinite(y_scaled)):
            raise FloatingPointError(
                f"Non-finite state for phi index {p}, "
                f"kappa={kappa*1e-9:.3f} ns^-1, dt={dt/tau_p:g} tau_p. "
                "Try decreasing dt_multiplier or max_noise_substep_dt_tau_p."
            )
        next_history = continuation_history_from_saved(
            y_scaled,
            nd,
            final_history=final_history,
            average_cases=average_continuation_history_across_noise,
        )

        # Current VCSEL.integrate API returns:
        #   y     shape (n_cases, 3*N_lasers, n_saved_steps)
        #   freqs shape (n_cases, N_lasers, n_saved_steps)
        y = y_scaled
        S = np.maximum(y[:, 1::3, :], 0.0)
        phi = y[:, 2::3, :]
        del freqs

        # Pairwise phase differences, circularly averaged after wrapping to
        # [-pi, pi].
        avg_cos_pd = average_wrapped_phase_trace_from_phi(phi)

        # In two-stage mode the plotted/saved order parameter comes from the
        # long fixed-kappa spectrum run, so leave this entry NaN until the
        # corresponding worker result arrives.
        if not use_two_stage_spectrum:
            order_param_cases = vcsel.order_parameter(y[:, :, -int(len(t) / 2):])
            order_param[k] = np.mean(order_param_cases)
            order_param_min[k] = np.min(order_param_cases)
            order_param_max[k] = np.max(order_param_cases)
            del order_param_cases

        if cos_cont_trace_len > 0:
            cos_tail = avg_cos_pd[-cos_cont_trace_len:]
            cos_phase_diff_time[k, cos_cont_trace_len - len(cos_tail):cos_cont_trace_len] = cos_tail

        if use_two_stage_spectrum:
            spectrum_jobs.append(make_spectrum_job(k, kappa, next_history))
            if (
                len(spectrum_jobs) >= int(max(1, two_stage_kappa_chunk_size))
                or k == len(kappa_c) - 1
            ):
                submit_spectrum_job_chunk()
            collect_finished_spectrum_chunks(block=False)
            history = next_history
            del y, y_scaled, final_history, S, phi, avg_cos_pd
            if k % 5 == 0 or k == len(kappa_c) - 1:
                gc.collect()
            continue

        # Build fields E_i. complex64 is enough for spectra here and cuts the
        # largest temporary arrays in half.
        E_all = (
            np.sqrt(S.astype(np.float32, copy=False))
            * np.exp((1j * phi).astype(np.complex64))
        ).astype(np.complex64, copy=False)

        # Total field across lasers
        E_tot = np.sum(E_all, axis=1)
        del S, phi

        


        # ###############################################################################
        # ###############        SPECTRUM COMPUTATION (N LASERS)        #################
        # ###############################################################################

        h = 6.626e-34
        conversion = h * nu_0 / tau_p

        if len(t) < 2:
            raise ValueError("Need at least two saved samples for spectral analysis.")
        saved_dt_nd = float(np.median(np.diff(t)) / tau_p)
        fs = 1.0 / saved_dt_nd
        desired_df = 1e6 * tau_p
        if spectrum_after_ramp_only:
            ramp_end_time = (float(ramp_start) + float(ramp_shape)) * tau
            ramp_end_time += float(spectrum_post_ramp_buffer_tau) * tau
            spectrum_start_idx = int(np.searchsorted(t, ramp_end_time, side="left"))
            spectrum_start_idx = min(max(spectrum_start_idx, 0), E_all.shape[-1] - 2)
        else:
            spectrum_start_idx = 0
        E_all_spec = E_all[:, :, spectrum_start_idx:]
        E_tot_spec = E_tot[:, spectrum_start_idx:]
        nperseg = np.shape(E_all_spec)[-1]
        if nperseg < 8:
            raise ValueError(
                f"Only {nperseg} saved post-ramp samples are available for spectra. "
                "Decrease dt_multiplier/trajectory_save_every, reduce ramp time, "
                "or increase Tmax."
            )
        noverlap = nperseg // 2
        N_fft = max(
            int(np.ceil(fs / desired_df)),
            int(spectrum_zero_padding_factor * nperseg),
            nperseg,
        )

        


        # ---------- TOTAL FIELD SPECTRUM ----------
        f, psd_tot = welch(
            E_tot_spec,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=N_fft,
            return_onesided=False,
            scaling="density"
        )
        # welch(E) already returns a power spectral density estimate for the
        # complex field. Do not square it again here.
        psd_tot_watts = np.maximum(np.real(psd_tot), 0.0) * conversion
        spectrum_db = 10*np.log10(np.mean(psd_tot_watts, axis=0)/1e-3 + 1e-20)

        # ---------- PER-LASER SPECTRA ----------
        spectra_db = np.zeros((N_lasers, len(f)), dtype=np.float32)


        for i in range(N_lasers):
            _, psd_i = welch(
                E_all_spec[:, i, :],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=N_fft,
                return_onesided=False,
                scaling="density"
            )
            psd_i_watts = np.maximum(np.real(psd_i), 0.0) * conversion
            spectra_db[i] = 10*np.log10(np.mean(psd_i_watts, axis=0)/1e-3 + 1e-20)

        # ---------- Frequency window ----------
        idx_sort = np.argsort(f)
        f_sorted = f[idx_sort]

        # Match the standalone linewidth.py workflow: measure spectra on a
        # positive-frequency window, then center each linewidth trace on the
        # selected peak during post-processing.
        f_plot_min = -15.0
        f_plot_max = 15.0
        mask = (f_sorted >= f_plot_min*1e9*tau_p) & (f_sorted <= f_plot_max*1e9*tau_p)

        f_window = f_sorted[mask]
        spectrum_db = spectrum_db[idx_sort][mask]
        spectra_db = spectra_db[:, idx_sort][:, mask]

        if k == 0:
            n_freqs = len(f_window)
            spectrum_db_norm_list = np.full((resolution, n_freqs), np.nan, dtype=dtype)
            spectrum_laser_db_norm_list = np.full((resolution, N_lasers, n_freqs), np.nan, dtype=dtype)

        spectrum_db_norm_list[k] = spectrum_db - np.max(spectrum_db)
        spectrum_laser_db_norm_list[k] = spectra_db - np.max(spectra_db, axis=1, keepdims=True)

        for laser_idx in range(N_lasers):
            laser_linewidth_traces_mhz[laser_idx, k], *_ = linewidth_from_db_spectrum_reference_style(
                spectrum_laser_db_norm_list[k, laser_idx],
                f_window/tau_p,
                level_db=linewidth_level_db,
                smooth_sigma_bins=linewidth_smooth_sigma_bins,
            )
            laser_linewidth_traces_mhz[laser_idx, k] *= 1e-6

        linewidth_trace_mhz[k], *_ = linewidth_from_db_spectrum_reference_style(
            spectrum_db_norm_list[k],
            f_window/tau_p,
            level_db=linewidth_level_db,
            smooth_sigma_bins=linewidth_smooth_sigma_bins,
        )
        linewidth_trace_mhz[k] *= 1e-6


        # ###############################################################################
        # ###############################   PLOTTING   #################################
        # ###############################################################################

            # ###############################################################################
        # ###############################   PLOTTING   #################################
        # ###############################################################################

        font_size = 22
        f_plot = f_window/tau_p
        f_plot_axis_min = f_plot[0] * 1e-9
        f_plot_axis_max = f_plot[-1] * 1e-9
        f_display_bound = np.floor(min(abs(f_plot_axis_min), abs(f_plot_axis_max)))
        if f_display_bound >= 1:
            f_plot_axis_min = -f_display_bound
            f_plot_axis_max = f_display_bound

        if save_spectrum_frames and (k % spectrum_frame_stride == 0 or k == len(kappa_c)-1):
            # clear_output(wait=True)
            fig = plt.figure(figsize=(18,10), dpi=300)
            width_ratios = [1]*30
            width_ratios[8] = 0.25
            gs = fig.add_gridspec(20, 30, height_ratios=[1]*20, width_ratios=width_ratios, hspace=0.3)

            # --- Average wrapped phase difference (top row) ---
            ax0 = fig.add_subplot(gs[1:8, 0:-3])

            local_cos_len = cos_phase_diff_time.shape[1]
            local_time_seconds = cos_time_axis_seconds[:local_cos_len]
            if local_time_seconds[-1] >= 1e-6:
                local_time = local_time_seconds * 1e6
                local_time_label = r'Time ($\mu$s)'
            else:
                local_time = local_time_seconds * 1e9
                local_time_label = r'Time (ns)'
            im0 = ax0.imshow(
                cos_phase_diff_time[:, :local_cos_len],
                aspect='auto',
                extent=[
                    local_time[0],
                    local_time[-1],
                    kappa_c[0]*1e-9,
                    kappa_c[-1]*1e-9,
                ],
                origin='lower',
                cmap='jet_r',
                vmin=0.0, vmax=np.pi, rasterized=True
            )
            cbar0 = fig.colorbar(im0, ax=ax0, pad=0.02)
            cbar0.set_label(r'$|\langle\Delta\phi\rangle_{\rm circ}|$ (rad)', fontsize=font_size, labelpad=0 )
            cbar0.set_ticks([0.0, 0.5*np.pi, np.pi])
            cbar0.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
            cbar0.ax.tick_params(labelsize=font_size)
            ax0.set_ylabel(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=font_size)
            ax0.set_xlabel(local_time_label, fontsize=font_size)
            ax0.set_xlim(local_time[0], local_time[-1])
            ax0.set_xticks(np.linspace(local_time[0], local_time[-1], 6))
            ax0.set_title(
                r'$\phi_p={:+.2f}\pi,\ \delta_i=[{}]\,\mathrm{{GHz}},\ \Delta t={:.1f}\tau_p$'.format(
                    phi_p[0,0]/np.pi,
                    ", ".join(f"{d:.1f}" for d in delta_distribution_ghz),
                    dt/tau_p,
                ),
                fontsize=font_size,
                pad=16,
            )
            # ax0.set_title(r'$\phi_p\in[0,2\pi]$', fontsize=font_size, pad=16)
            ax0.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
            ax0.tick_params(axis='both', labelsize=font_size)

            # --- Order Parameter Plot (to the right of ax0) ---
            ax_order = fig.add_subplot(gs[1:8, -2:])
            plot_order_parameter_panel(
                ax_order,
                order_param,
                kappa_c,
                font_size,
                pad=16,
                order_param_min=order_param_min,
                order_param_max=order_param_max,
            )
            
            # --- Optical Spectrum (Total) ---
            ax1 = fig.add_subplot(gs[11:, 1:8])
            spectrum_cmap = plt.get_cmap('jet').copy()
            spectrum_cmap.set_bad(color='white')
            im = ax1.imshow(
                spectrum_db_norm_list,
                aspect='auto',
                extent=[f_plot_axis_min, f_plot_axis_max, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
                origin='lower',
                cmap=spectrum_cmap, rasterized=True
            )
            cbar1 = fig.colorbar(im, ax=ax1, pad=0.05)
            cbar1.set_label(r'Power (dBm)', fontsize=font_size, labelpad=12)
            cbar1.ax.tick_params(labelsize=font_size)
            im.set_clim(-100,0) 
            ax1.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
            ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
            ax1.set_title(r"$|\mathcal{F}\left(E_{tot}\right)|^2$", fontsize=font_size, pad=16)
            ax1.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
            ax1.set_xticks(np.linspace(f_plot_axis_min, f_plot_axis_max, 3))
            ax1.set_xlim(f_plot_axis_min, f_plot_axis_max)
            ax1.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
            ax1.tick_params(axis='both', labelsize=font_size)
            for delta_i, delta_ghz in enumerate(delta_distribution_ghz):
                ax1.plot(
                    np.full_like(kappa_c, delta_ghz, dtype=float),
                    kappa_c*1e-9,
                    color='black',
                    linestyle='--',
                    linewidth=2,
                    label=rf"$\delta_i$" if delta_i == 0 else None,
                    alpha=0.5,
                )

            # --- Linewidth traces for this phi_p run ---
            ax2 = fig.add_subplot(gs[11:, 13:29])
            field_colors = plt.get_cmap("tab10")
            for laser_idx in range(N_lasers):
                laser_linewidth = laser_linewidth_traces_mhz[laser_idx, :k + 1]
                valid_laser_linewidth = np.isfinite(laser_linewidth) & (laser_linewidth > 0)
                if np.any(valid_laser_linewidth):
                    laser_color = field_colors(laser_idx % field_colors.N)
                    if plot_raw_linewidth_traces:
                        ax2.plot(
                            kappa_c[:k + 1][valid_laser_linewidth] * 1e-9,
                            laser_linewidth[valid_laser_linewidth],
                            color=laser_color,
                            linewidth=0.8,
                            alpha=0.18,
                            zorder=1,
                        )
                    laser_linewidth_smooth = moving_average_nan_safe(
                        laser_linewidth,
                        linewidth_trace_smooth_window,
                    )
                    valid_laser_smooth = np.isfinite(laser_linewidth_smooth) & (laser_linewidth_smooth > 0)
                    ax2.plot(
                        kappa_c[:k + 1][valid_laser_smooth] * 1e-9,
                        laser_linewidth_smooth[valid_laser_smooth],
                        color=laser_color,
                        linewidth=1.4,
                        alpha=0.55,
                        zorder=2,
                        label=rf"$E_{laser_idx + 1}$",
                    )
            valid_linewidth = np.isfinite(linewidth_trace_mhz[:k + 1]) & (linewidth_trace_mhz[:k + 1] > 0)
            if plot_raw_linewidth_traces and np.any(valid_linewidth):
                ax2.plot(
                    kappa_c[:k + 1][valid_linewidth] * 1e-9,
                    linewidth_trace_mhz[:k + 1][valid_linewidth],
                    color="black",
                    linewidth=1.0,
                    alpha=0.2,
                    zorder=3,
                )
            linewidth_trace_smooth = moving_average_nan_safe(
                linewidth_trace_mhz[:k + 1],
                linewidth_trace_smooth_window,
            )
            valid_linewidth_smooth = np.isfinite(linewidth_trace_smooth) & (linewidth_trace_smooth > 0)
            ax2.plot(
                kappa_c[:k + 1][valid_linewidth_smooth] * 1e-9,
                linewidth_trace_smooth[valid_linewidth_smooth],
                color="black",
                linewidth=3.0,
                zorder=4,
                label=r"$E_{tot}$",
            )
            resolution_mhz = linewidth_resolution_floor_mhz(Tmax)
            if linewidth_plot_ylim_mhz[0] <= resolution_mhz <= linewidth_plot_ylim_mhz[1]:
                ax2.axhline(
                    resolution_mhz,
                    color="red",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.8,
                    zorder=2,
                    label=r"$1.44/T_\mathrm{max}$",
                )
            if np.any(valid_linewidth):
                current_width_valid = np.isfinite(linewidth_trace_mhz[k]) and linewidth_trace_mhz[k] > 0
                if current_width_valid:
                    ax2.scatter(
                        kappa_c[k] * 1e-9,
                        linewidth_trace_mhz[k],
                        color="red",
                        s=45,
                        zorder=5,
                    )
            linewidth_font_size = font_size + 4
            ax2.set_xlabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=linewidth_font_size, labelpad=10)
            ax2.set_ylabel("")
            ax2.set_title(
                rf"{linewidth_level_db:.0f} dB linewidth (MHz)",
                fontsize=linewidth_font_size,
                pad=16,
            )
            ax2.set_xlim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
            ax2.set_yscale("log")
            ax2.set_ylim(*linewidth_plot_ylim_mhz)
            ax2.set_yticks(linewidth_plot_yticks_mhz)
            ax2.grid(True, linestyle="--", alpha=0.4, which="both")
            ax2.set_xticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
            ax2.tick_params(axis='both', labelsize=linewidth_font_size)
            ax2.legend(fontsize=linewidth_font_size - 10, loc="upper right")

            os.makedirs(frame_dir, exist_ok=True)
            filename = f"{frame_dir}/phi_{p:03d}.png"
            
            # phi_p{phi_p[0,0]/np.pi:.2f}pi_continuation_noise_detuning{detuning}_alpha{alpha}_noise_{n_cases}_{n_iterations}avg_self{self_feedback:.2f}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            plt.cla(); plt.clf()
            plt.close('all')
            del fig, gs, ax0, ax_order, ax1, ax2
            del im0, im, cbar0, cbar1
            if not show_two_stage_progress:
                clear_output(wait=True)

        history = next_history
        del y, y_scaled, final_history, E_all, E_tot, E_all_spec, E_tot_spec
        del avg_cos_pd, psd_tot, psd_tot_watts, spectrum_db, spectra_db, psd_i, psd_i_watts
        if k % 5 == 0 or k == len(kappa_c) - 1:
            gc.collect()

    if use_two_stage_spectrum:
        submit_spectrum_job_chunk()
        try:
            collect_finished_spectrum_chunks(block=False)
            collect_finished_spectrum_chunks(block=True)
        finally:
            if spectrum_executor is not None:
                spectrum_executor.shutdown(wait=True, cancel_futures=False)
            drain_long_run_progress_queue()
            for bar in list(spectrum_progress_bars.values()):
                bar.close()
            spectrum_progress_bars.clear()
            spectrum_future_progress_id.clear()
        if f_window is None:
            raise RuntimeError("Two-stage spectral run completed without returning any spectra.")
    else:
        Tmax_for_linewidth = Tmax

    if use_two_stage_spectrum and save_spectrum_frames:
        font_size = 22
        f_plot = f_window/tau_p
        f_plot_axis_min = f_plot[0] * 1e-9
        f_plot_axis_max = f_plot[-1] * 1e-9
        f_display_bound = np.floor(min(abs(f_plot_axis_min), abs(f_plot_axis_max)))
        if f_display_bound >= 1:
            f_plot_axis_min = -f_display_bound
            f_plot_axis_max = f_display_bound

        fig = plt.figure(figsize=(18,10), dpi=300)
        width_ratios = [1]*30
        width_ratios[8] = 0.25
        gs = fig.add_gridspec(20, 30, height_ratios=[1]*20, width_ratios=width_ratios, hspace=0.3)

        ax0 = fig.add_subplot(gs[1:8, 0:-3])
        local_cos_len = cos_phase_diff_time.shape[1]
        local_time_seconds = cos_time_axis_seconds[:local_cos_len]
        if local_time_seconds[-1] >= 1e-6:
            local_time = local_time_seconds * 1e6
            local_time_label = r'Time ($\mu$s)'
        else:
            local_time = local_time_seconds * 1e9
            local_time_label = r'Time (ns)'
        im0 = ax0.imshow(
            cos_phase_diff_time[:, :local_cos_len],
            aspect='auto',
            extent=[local_time[0], local_time[-1], kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
            origin='lower',
            cmap='jet_r',
            vmin=0.0,
            vmax=np.pi,
            rasterized=True,
        )
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.02)
        cbar0.set_label(r'$|\langle\Delta\phi\rangle_{\rm circ}|$ (rad)', fontsize=font_size, labelpad=0)
        cbar0.set_ticks([0.0, 0.5*np.pi, np.pi])
        cbar0.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
        cbar0.ax.tick_params(labelsize=font_size)
        ax0.set_ylabel(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=font_size)
        ax0.set_xlabel(local_time_label, fontsize=font_size)
        ax0.set_xlim(local_time[0], local_time[-1])
        ax0.set_xticks(np.linspace(local_time[0], local_time[-1], 6))
        ax0.set_title(
            r'$\phi_p={:+.2f}\pi,\ \delta_i=[{}]\,\mathrm{{GHz}},\ \Delta t={:.1f}\tau_p$'.format(
                phi_p[0,0]/np.pi,
                ", ".join(f"{d:.1f}" for d in delta_distribution_ghz),
                dt/tau_p,
            ),
            fontsize=font_size,
            pad=16,
        )
        ax0.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
        ax0.tick_params(axis='both', labelsize=font_size)

        ax_order = fig.add_subplot(gs[1:8, -2:])
        plot_order_parameter_panel(
            ax_order,
            order_param,
            kappa_c,
            font_size,
            pad=16,
            order_param_min=order_param_min,
            order_param_max=order_param_max,
        )

        ax1 = fig.add_subplot(gs[11:, 1:8])
        spectrum_cmap = plt.get_cmap('jet').copy()
        spectrum_cmap.set_bad(color='white')
        im = ax1.imshow(
            spectrum_db_norm_list,
            aspect='auto',
            extent=[f_plot_axis_min, f_plot_axis_max, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
            origin='lower',
            cmap=spectrum_cmap,
            rasterized=True,
        )
        cbar1 = fig.colorbar(im, ax=ax1, pad=0.05)
        cbar1.set_label(r'Power (dBm)', fontsize=font_size, labelpad=12)
        cbar1.ax.tick_params(labelsize=font_size)
        im.set_clim(-100,0)
        ax1.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
        ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
        ax1.set_title(r"$|\mathcal{F}\left(E_{tot}\right)|^2$", fontsize=font_size, pad=16)
        ax1.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
        ax1.set_xticks(np.linspace(f_plot_axis_min, f_plot_axis_max, 3))
        ax1.set_xlim(f_plot_axis_min, f_plot_axis_max)
        ax1.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
        ax1.tick_params(axis='both', labelsize=font_size)
        for delta_i, delta_ghz in enumerate(delta_distribution_ghz):
            ax1.plot(
                np.full_like(kappa_c, delta_ghz, dtype=float),
                kappa_c*1e-9,
                color='black',
                linestyle='--',
                linewidth=2,
                label=rf"$\delta_i$" if delta_i == 0 else None,
                alpha=0.5,
            )

        ax2 = fig.add_subplot(gs[11:, 13:29])
        field_colors = plt.get_cmap("tab10")
        for laser_idx in range(N_lasers):
            laser_linewidth = laser_linewidth_traces_mhz[laser_idx]
            valid_laser_linewidth = np.isfinite(laser_linewidth) & (laser_linewidth > 0)
            if np.any(valid_laser_linewidth):
                laser_color = field_colors(laser_idx % field_colors.N)
                if plot_raw_linewidth_traces:
                    ax2.plot(
                        kappa_c[valid_laser_linewidth] * 1e-9,
                        laser_linewidth[valid_laser_linewidth],
                        color=laser_color,
                        linewidth=0.8,
                        alpha=0.18,
                        zorder=1,
                    )
                laser_linewidth_smooth = moving_average_nan_safe(
                    laser_linewidth,
                    linewidth_trace_smooth_window,
                )
                valid_laser_smooth = np.isfinite(laser_linewidth_smooth) & (laser_linewidth_smooth > 0)
                ax2.plot(
                    kappa_c[valid_laser_smooth] * 1e-9,
                    laser_linewidth_smooth[valid_laser_smooth],
                    color=laser_color,
                    linewidth=1.4,
                    alpha=0.55,
                    zorder=2,
                    label=rf"$E_{laser_idx + 1}$",
                )
        valid_linewidth = np.isfinite(linewidth_trace_mhz) & (linewidth_trace_mhz > 0)
        if plot_raw_linewidth_traces and np.any(valid_linewidth):
            ax2.plot(
                kappa_c[valid_linewidth] * 1e-9,
                linewidth_trace_mhz[valid_linewidth],
                color="black",
                linewidth=1.0,
                alpha=0.2,
                zorder=3,
            )
        linewidth_trace_smooth = moving_average_nan_safe(
            linewidth_trace_mhz,
            linewidth_trace_smooth_window,
        )
        valid_linewidth_smooth = np.isfinite(linewidth_trace_smooth) & (linewidth_trace_smooth > 0)
        ax2.plot(
            kappa_c[valid_linewidth_smooth] * 1e-9,
            linewidth_trace_smooth[valid_linewidth_smooth],
            color="black",
            linewidth=3.0,
            zorder=4,
            label=r"$E_{tot}$",
        )
        resolution_mhz = linewidth_resolution_floor_mhz(Tmax_for_linewidth)
        if linewidth_plot_ylim_mhz[0] <= resolution_mhz <= linewidth_plot_ylim_mhz[1]:
            ax2.axhline(
                resolution_mhz,
                color="red",
                linestyle="--",
                linewidth=2.0,
                alpha=0.8,
                zorder=2,
                label=r"$1.44/T_\mathrm{max}$",
            )
        linewidth_font_size = font_size + 4
        ax2.set_xlabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=linewidth_font_size, labelpad=10)
        ax2.set_ylabel("")
        ax2.set_title(rf"{linewidth_level_db:.0f} dB linewidth (MHz)", fontsize=linewidth_font_size, pad=16)
        ax2.set_xlim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
        ax2.set_yscale("log")
        ax2.set_ylim(*linewidth_plot_ylim_mhz)
        ax2.set_yticks(linewidth_plot_yticks_mhz)
        ax2.grid(True, linestyle="--", alpha=0.4, which="both")
        ax2.set_xticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
        ax2.tick_params(axis='both', labelsize=linewidth_font_size)
        ax2.legend(fontsize=linewidth_font_size - 10, loc="upper right")

        filename = f"{frame_dir}/phi_{p:03d}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        plt.cla(); plt.clf()
        plt.close('all')
        del fig, gs, ax0, ax_order, ax1, ax2
        del im0, im, cbar0, cbar1
        if not show_two_stage_progress:
            clear_output(wait=True)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    run_label = f"self{self_feedback:.2f}_detuning{detuning_label}_{phi_p_label}"
    np.save(f"{save_dir}/spectrum_db_norm_list_{run_label}.npy", spectrum_db_norm_list)
    np.save(f"{save_dir}/f_sorted_{run_label}.npy", f_sorted)
    np.save(f"{save_dir}/f_window_{run_label}.npy", f_window)
    np.save(f"{save_dir}/kappa_c_{run_label}.npy", kappa_c)
    np.save(f"{save_dir}/detuning_vals_ghz_{run_label}.npy", detuning_vals_ghz)
    np.save(f"{save_dir}/phi_p_{run_label}.npy", np.array([phi_p_value]))
    np.save(f"{save_dir}/linewidth_E_fields_mhz_{run_label}.npy", laser_linewidth_traces_mhz)
    np.save(f"{save_dir}/linewidth_Etot_mhz_{run_label}.npy", linewidth_trace_mhz)
    np.save(f"{save_dir}/order_parameter_{run_label}.npy", order_param)
    np.save(f"{save_dir}/order_parameter_min_{run_label}.npy", order_param_min)
    np.save(f"{save_dir}/order_parameter_max_{run_label}.npy", order_param_max)
    np.save(f"{save_dir}/wrapped_phase_diff_time_{run_label}.npy", cos_phase_diff_time)
    np.save(f"{save_dir}/wrapped_phase_time_axis_seconds_{run_label}.npy", cos_time_axis_seconds)
    np.savez(
        f"{save_dir}/integrator_settings_{run_label}.npz",
        dt_multiplier=dt / tau_p,
        dt_seconds=dt,
        Tmax_seconds=Tmax_for_linewidth,
        Tmax_continuation_seconds=Tmax,
        Tmax_spectrum_seconds=Tmax_for_linewidth,
        use_two_stage_spectrum=bool(use_two_stage_spectrum),
        two_stage_kappa_chunk_size=int(max(1, two_stage_kappa_chunk_size)),
        long_spectrum_jobs=int(max(1, long_spectrum_jobs)),
        vectorize_two_stage_kappa_chunks=bool(vectorize_two_stage_kappa_chunks),
        average_continuation_history_across_noise=bool(average_continuation_history_across_noise),
        show_two_stage_progress=bool(show_two_stage_progress),
        print_two_stage_messages=bool(print_two_stage_messages),
        save_two_stage_progress_arrays=bool(save_two_stage_progress_arrays),
        save_two_stage_progress_frames=bool(save_two_stage_progress_frames),
        save_two_stage_chunk_frames=bool(save_two_stage_chunk_frames),
        integration_scheme=integration_scheme,
        integrator_theta=integrator_theta,
        integrator_max_iter=integrator_max_iter,
        delay_interpolation=delay_interpolation or "none",
        noise_substeps=noise_substeps,
        max_noise_substep_dt_tau_p=max_noise_substep_dt_tau_p,
        trajectory_save_every=int(max(1, trajectory_save_every)),
        spectrum_trajectory_save_every=int(max(1, spectrum_trajectory_save_every)),
        trajectory_output_dtype=np.dtype(trajectory_output_dtype).name,
        linewidth_level_db=float(linewidth_level_db),
        linewidth_smooth_sigma_bins=float(linewidth_smooth_sigma_bins),
        linewidth_trace_smooth_window=int(linewidth_trace_smooth_window),
        plot_raw_linewidth_traces=bool(plot_raw_linewidth_traces),
        spectrum_after_ramp_only=bool(spectrum_after_ramp_only),
        spectrum_post_ramp_buffer_tau=float(spectrum_post_ramp_buffer_tau),
        spectrum_start_time_seconds=(
            float(spectrum_post_ramp_buffer_tau) * tau
            if use_two_stage_spectrum
            else float((float(ramp_start) + float(ramp_shape) + float(spectrum_post_ramp_buffer_tau)) * tau)
        ),
        spectrum_zero_padding_factor=int(spectrum_zero_padding_factor),
        spectrum_max_saved_freq_points=int(spectrum_max_saved_freq_points),
        wrapped_phase_plot_all_saved_points=bool(cos_plot_all_saved_points),
        wrapped_phase_window_ns=float(cos_window_ns),
        wrapped_phase_two_stage_long_tail_only=bool(cos_two_stage_long_tail_only),
        wrapped_phase_long_tail_window_us=float(cos_long_tail_window_us),
        wrapped_phase_trace_len=int(cos_trace_len),
        wrapped_phase_cont_trace_len=int(cos_cont_trace_len),
        wrapped_phase_spectrum_trace_len=int(cos_spectrum_trace_len),
    )

    return {
        "p": p,
        "run_label": run_label,
        "phi_p": phi_p_value,
        "detuning_label": detuning_label,
        "self_feedback": self_feedback,
        "N_lasers": N_lasers,
    }




if run_phi_sweep:
    if n_phi_jobs == 1:
        phi_sweep_results = []
        for p in range(n_phi_p):
            if print_two_stage_messages:
                tqdm.write(f"starting phi {p + 1}/{n_phi_p}")
            phi_sweep_results.append(run_phi_p_continuation(p))
    else:
        # Recycle each worker after one phi_p run so large arrays are released
        # instead of accumulating inside long-lived worker processes.
        mp_context = mp.get_context("fork")
        with mp_context.Pool(processes=n_phi_jobs, maxtasksperchild=1) as pool:
            phi_sweep_results = list(tqdm(
                pool.imap_unordered(run_phi_p_continuation, range(n_phi_p), chunksize=1),
                total=n_phi_p,
                desc="phi_p sweep",
                unit="phi",
            ))

    # Populate the legacy single-phi filenames from the first completed run so
    # the exploratory cells below still have a default dataset to load. This is
    # done serially after the parallel workers finish to avoid file races.
    first_result = sorted(phi_sweep_results, key=lambda item: item["p"])[0]
    N_lasers = first_result["N_lasers"]
    save_dir = linewidth_output_dirs(N_lasers)["arrays"]
    run_label = first_result["run_label"]
    detuning_label = first_result["detuning_label"]
    self_feedback = first_result["self_feedback"]
    spectrum_db_norm_list = np.load(f"{save_dir}/spectrum_db_norm_list_{run_label}.npy")
    f_sorted = np.load(f"{save_dir}/f_sorted_{run_label}.npy")
    kappa_c = np.load(f"{save_dir}/kappa_c_{run_label}.npy")
    detuning_vals_ghz = np.load(f"{save_dir}/detuning_vals_ghz_{run_label}.npy")
    np.save(f"{save_dir}/spectrum_db_norm_list_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning_label}.npy", spectrum_db_norm_list)
    np.save(f"{save_dir}/f_sorted_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning_label}.npy", f_sorted)
    np.save(f"{save_dir}/detuning_vals_ghz_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning_label}.npy", detuning_vals_ghz)




#%%

# Per-phi spectra are saved by each parallel worker above. After the workers
# finish, the first phi_p run is copied into legacy E_tot filenames for the
# single-phi analysis cells below.
# %%

# --- Load saved data at desired detuning frequency---

detuning = float(detuning_ghz if "detuning_ghz" in globals() else 4.0)
detuning_label = globals().get("detuning_label", f"{detuning:.2f}")
self_feedback = globals().get("self_feedback", 0.0)
tau_p = globals().get("tau_p", 5.4e-12)
N_lasers = 3#int(globals().get("N_lasers", 2))
output_dirs = linewidth_output_dirs(N_lasers)
save_dir = output_dirs["arrays"]

import numpy as np
spectrum_db_norm_list = np.load(f"{save_dir}/spectrum_db_norm_list_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning_label}.npy")
f_sorted = np.load(f"{save_dir}/f_sorted_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning_label}.npy")
detuning_vals_ghz = np.load(f"{save_dir}/detuning_vals_ghz_self{self_feedback:.2f}_200_avg_spectra_detuning{detuning_label}.npy")

def frequency_axis_for_spectrum(f_sorted, n_freqs, tau_p, f_plot_min_ghz=None, f_plot_max_ghz=None):
    """Return a physical frequency axis whose length matches a saved spectrum."""
    f_sorted = np.asarray(f_sorted)
    if len(f_sorted) == n_freqs:
        return f_sorted / tau_p

    if f_plot_min_ghz is not None and f_plot_max_ghz is not None:
        mask = (
            (f_sorted >= f_plot_min_ghz * 1e9 * tau_p)
            & (f_sorted <= f_plot_max_ghz * 1e9 * tau_p)
        )
        if np.count_nonzero(mask) == n_freqs:
            return f_sorted[mask] / tau_p

    center_idx = int(np.argmin(np.abs(f_sorted)))
    start = max(0, center_idx - n_freqs // 2)
    stop = start + n_freqs
    if stop > len(f_sorted):
        stop = len(f_sorted)
        start = stop - n_freqs
    if start < 0 or stop > len(f_sorted):
        raise ValueError(
            f"Cannot build a frequency axis with {n_freqs} points from "
            f"f_sorted with {len(f_sorted)} points."
        )
    return f_sorted[start:stop] / tau_p

n_cases = spectrum_db_norm_list.shape[0]
kappa_max = 40e9
kappa_arr = np.linspace(0, kappa_max, n_cases)
kappa_c = np.linspace(0e9, kappa_max, n_cases)
f_plot = frequency_axis_for_spectrum(
    f_sorted,
    spectrum_db_norm_list.shape[1],
    tau_p,
    f_plot_min_ghz=0.0,
    f_plot_max_ghz=15.0,
)
f_plot_min = f_plot[0] * 1e-9
f_plot_max = f_plot[-1] * 1e-9
f_display_bound = np.floor(min(abs(f_plot_min), abs(f_plot_max)))
if f_display_bound >= 1:
    f_plot_min = -f_display_bound
    f_plot_max = f_display_bound


#%%
# %matplotlib inline  # IPython only; keep commented so this file is valid Python.
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from IPython.display import clear_output
from scipy.signal import find_peaks
from scipy.ndimage import median_filter


from scipy.signal import savgol_filter

# Example parameters (tune for your data)
window_length = 5   # must be odd
polyorder = 3


plot_spectrum = spectrum_db_norm_list
num_curves = min(200, plot_spectrum.shape[0], len(kappa_arr))
color_den = max(num_curves - 1, 1)
colors = [cm.Blues(0.3 + 0.7 * i / color_den) for i in range(num_curves)]

spec_smooth_reflected = np.zeros((plot_spectrum.shape[0], plot_spectrum.shape[1]), dtype=plot_spectrum.dtype)

fwhm = []

peak_pts = []
fig = plt.figure(figsize=(8, 5), dpi=300)

for i in range(num_curves):
    # Find the index of the maximum for this spectrum
    # i = 0
    if kappa_arr[i]*1e-9 > -1:# and kappa_arr[i]*1e-9 < 11:
        

        clear_output(wait=True)

        # Find FWHM of the peak centered at freq_shift
        # peak_power = spectrum_db_norm_list[i, max_idx]
        half_max =  - 3  # 3 dB down for FWHM


        spec_smooth = plot_spectrum[i, :]#savgol_filter(plot_spectrum[i, :], window_length, polyorder)
        # Reflect spec_smooth about the peak (index max_idx)
        spec_smooth_reflected[i,:] = spec_smooth#np.concatenate([spec_smooth[::-1], spec_smooth])
        f_plot_reflected = f_plot[:spec_smooth_reflected.shape[1]]#np.concatenate([-f_plot[::-1], f_plot])



        # max_idx = np.argmax(spec_smooth)
        # spec_smooth = spec_smooth - np.max(spec_smooth)
        # spec_smooth[f_plot<=0] = -1000
        # spec_smooth = spec_smooth - np.max(spec_smooth)
        max_idx = np.argmax(spec_smooth_reflected[i,:])
        freq_shift = np.abs(f_plot_reflected[max_idx])
        shifted_freq = (f_plot_reflected - freq_shift) * 1e-6

        
        

        # plt.plot(-f_plot[::-1]*1e-6,  spectrum_db_norm_list[i, ::-1], color=colors[i], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$', alpha=0.5)
        # plt.plot(f_plot*1e-6,  spectrum_db_norm_list[i, :], color=colors[i], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$', alpha=0.5)



        # plt.plot(f_plot_reflected*1e-6 - np.abs(f_plot_reflected[max_idx]*1e-6),  spec_smooth_reflected[i,:], color=colors[i], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$')

        plt.plot(f_plot_reflected*1e-6, spec_smooth_reflected[i, :], color=colors[i], label=rf'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$')

        peak_pts.append((f_plot_reflected[max_idx]*1e-6, spec_smooth_reflected[i, max_idx]))

        

        # plt.plot(shifted_freq,  spectrum_db_norm_list[i, :], color=colors[0], label=f'$\kappa_c$={kappa_arr[i]*1e-9:.2f} ns$^{{-1}}$')


plt.axhline(half_max, color='red', linestyle='--', linewidth=2, label='Half max (FWHM)')
peak_pts = np.array(peak_pts)
if len(peak_pts):
    plt.plot(peak_pts[:,0], peak_pts[:,1], 'gx', markersize=10, label='Peak Points')

plt.xlabel("Frequency offset from peak (MHz)", fontsize=18)
plt.ylabel("Power (dBm)", fontsize=18)
# plt.title(rf"Normalized Optical Spectrum $\kappa={kappa_c[i]*1e-9:.4f}~\mathrm{{ns}}^{{-1}}$", fontsize=18) 
plt.xlim(-5000, 5000)
plt.ylim(-80, 5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Add colorbar for kappa_c
norm = mpl.colors.Normalize(vmin=kappa_arr[0]*1e-9, vmax=kappa_arr[num_curves-1]*1e-9)
sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.02, ax=plt.gca())
cbar.set_label(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=18)
cbar.ax.tick_params(labelsize=24)
plt.title(r"$E_{tot}$",fontsize =24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

plt.savefig(f"{output_dirs['base']}/3db_spectra_Etot.png", bbox_inches="tight")
plt.show()
plt.close(fig)
        # break
        


#%%

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import glob
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

linewidth_level_db = 3.0

def linewidth_from_db_spectrum_reference_style(spectrum_db, freqs_hz, level_db=3.0):
    """Use the same peak-centered threshold-crossing linewidth as linewidth.py."""
    spectrum = np.asarray(spectrum_db, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float) * 1e-6

    peaks, props = find_peaks(spectrum, height=0)
    if len(peaks) > 0:
        peak_pos = peaks[np.argmax(props["peak_heights"])]
    else:
        peak_pos = int(np.nanargmax(spectrum))

    shifted_freq = freqs - freqs[peak_pos]
    target_level = spectrum[peak_pos] - level_db

    left_idx = np.where(spectrum[:peak_pos] <= target_level)[0]
    if len(left_idx) > 0:
        i = left_idx[-1]
        freq_left = np.interp(
            target_level,
            [spectrum[i], spectrum[i + 1]],
            [shifted_freq[i], shifted_freq[i + 1]],
        )
    else:
        freq_left = shifted_freq[0]

    right_idx = np.where(spectrum[peak_pos:] <= target_level)[0]
    if len(right_idx) > 0:
        i = right_idx[0] + peak_pos
        freq_right = np.interp(
            target_level,
            [spectrum[i - 1], spectrum[i]],
            [shifted_freq[i - 1], shifted_freq[i]],
        )
    else:
        freq_right = shifted_freq[-1]

    linewidth = freq_right - freq_left
    return linewidth, freq_left, freq_right, target_level, peak_pos, shifted_freq

# Array to store E_tot linewidth
fwhm_Etot_list = np.zeros(len(kappa_arr))

# Initial guess for FWHM in Hz
fwhm0_Etot = 0e6

plot_freq = 300e6

window_length = 1001    # must be odd
polyorder = 3



detuning = float(detuning_ghz if "detuning_ghz" in globals() else detuning)

for detuning in [detuning]:
    n_cases = spectrum_db_norm_list.shape[0]
    kappa_arr = np.linspace(0, kappa_max, n_cases)
    f_plot = frequency_axis_for_spectrum(
        f_sorted,
        spectrum_db_norm_list.shape[1],
        tau_p,
        f_plot_min_ghz=0.0,
        f_plot_max_ghz=15.0,
    )
    f_plot_min = f_plot[0] * 1e-9
    f_plot_max = f_plot[-1] * 1e-9
    f_display_bound = np.floor(min(abs(f_plot_min), abs(f_plot_max)))
    if f_display_bound >= 1:
        f_plot_min = -f_display_bound
        f_plot_max = f_display_bound

    fwhm_Etot_list = np.full(n_cases, np.nan)



    def compute_linewidth(
        spectrum_db_norm_list, field_name, fwhm_list_db3, fwhm0, folder, plot=False
    ):
        """
        Fits Lorentzian to spectra and extracts linewidths.
        
        Parameters
        ----------
        spectrum_db_norm_list : 2D array [cases, freqs]
            Normalized spectra in dB (peak = 0 dB).
        field_name : str
            Name used for saving plots.
        fwhm_list_db3 : array
            Output array for fitted -3 dB linewidths.
        fwhm0 : float
            Initial guess for FWHM (Hz). If <= 0, estimated from data.
        folder : str
            Output folder name.
        """
        if plot:
            os.makedirs(f'{output_dirs["base"]}/{folder}', exist_ok=True)

        for k in range(len(spectrum_db_norm_list)):
            if kappa_arr[k]*1e-9 > -.01 and k >= 0:
                clear_output(wait=True)
                spectrum = spectrum_db_norm_list[k, :]
                (
                    fwhm_list_db3[k],
                    freq_left,
                    freq_right,
                    target_level,
                    peak_pos,
                    shifted_freq,
                ) = linewidth_from_db_spectrum_reference_style(
                    spectrum,
                    f_plot,
                    level_db=linewidth_level_db,
                )

                if plot:
                    fig = plt.figure(figsize=(8,5), dpi=150)
                    plt.plot(shifted_freq, spectrum, label="Spectrum")
                    plt.plot(
                        [freq_left, freq_right],
                        [target_level, target_level],
                        color='red',
                        linewidth=3,
                        label=f'-{linewidth_level_db:.0f} dB width',
                    )
                    peak_freq = shifted_freq[peak_pos]
                    plt.plot(peak_freq, spectrum[peak_pos], 'ko')
                    plt.xlabel('Frequency offset (MHz)', fontsize=24)
                    plt.ylabel('Power (dBm)', fontsize=24)
                    plt.xlim(-300, 300)
                    plt.ylim(-50, 1)
                    plt.xticks(fontsize=22)
                    plt.yticks(fontsize=22)
                    plt.title(f"$\\kappa_c$={kappa_arr[k]*1e-9:.2f} ns$^{{-1}}$", fontsize=28)
                    plt.legend(fontsize=16, loc='upper right')
                    plt.tight_layout()
                    plt.savefig(f'{output_dirs["base"]}/{folder}/kappa_{kappa_arr[k]*1e-9:.2f}.png', dpi=150)
                    plt.show()
                    plt.close(fig)

        return fwhm_list_db3


    plot = False

    phi_spectrum_pattern = (
        f"{save_dir}/spectrum_db_norm_list_"
        f"self{self_feedback:.2f}_detuning{detuning_label}_phiidx*_phipi*.npy"
    )
    phi_spectrum_files = sorted(glob.glob(phi_spectrum_pattern))

    linewidth_phi_traces = []
    phi_values_for_linewidth = []
    kappa_ref = None
    linewidth_floor_tmax_seconds = float(globals().get("Tmax", 5e-6))

    if phi_spectrum_files:
        for spectrum_path in phi_spectrum_files:
            run_label = os.path.basename(spectrum_path).removeprefix(
                "spectrum_db_norm_list_"
            ).removesuffix(".npy")
            spectrum_phi = np.load(spectrum_path)
            f_sorted_phi = np.load(f"{save_dir}/f_sorted_{run_label}.npy")
            kappa_phi = np.load(f"{save_dir}/kappa_c_{run_label}.npy")
            settings_path = f"{save_dir}/integrator_settings_{run_label}.npz"
            if os.path.exists(settings_path):
                with np.load(settings_path, allow_pickle=True) as settings:
                    linewidth_floor_tmax_seconds = float(settings["Tmax_seconds"])
            phi_path = f"{save_dir}/phi_p_{run_label}.npy"
            if os.path.exists(phi_path):
                phi_values_for_linewidth.append(float(np.load(phi_path)[0]))

            f_plot_phi = frequency_axis_for_spectrum(
                f_sorted_phi,
                spectrum_phi.shape[1],
                tau_p,
                f_plot_min_ghz=0.0,
                f_plot_max_ghz=15.0,
            )
            linewidth_trace = np.full(spectrum_phi.shape[0], np.nan)
            for k in range(spectrum_phi.shape[0]):
                linewidth_trace[k], *_ = linewidth_from_db_spectrum_reference_style(
                    spectrum_phi[k],
                    f_plot_phi,
                    level_db=linewidth_level_db,
                )

            linewidth_phi_traces.append(linewidth_trace)
            if kappa_ref is None:
                kappa_ref = kappa_phi
            elif len(kappa_ref) != len(kappa_phi) or not np.allclose(kappa_ref, kappa_phi):
                raise ValueError("Saved phi_p runs do not share the same kappa grid.")
    else:
        # Fallback for old runs that only saved the legacy single-phi spectrum.
        linewidth_trace = compute_linewidth(
            spectrum_db_norm_list,
            'E_tot',
            fwhm_Etot_list,
            fwhm0_Etot,
            'E_tot',
            plot=plot,
        )
        linewidth_phi_traces.append(linewidth_trace)
        kappa_ref = kappa_arr

    linewidth_phi_traces = np.asarray(linewidth_phi_traces, dtype=float)
    linewidth_mean_mhz = np.nanmean(linewidth_phi_traces, axis=0)
    linewidth_std_mhz = np.nanstd(linewidth_phi_traces, axis=0)
    linewidth_lower_mhz = linewidth_mean_mhz - linewidth_std_mhz
    linewidth_upper_mhz = linewidth_mean_mhz + linewidth_std_mhz
    linewidth_lower_mhz = np.where(linewidth_lower_mhz > 0, linewidth_lower_mhz, np.nan)

    np.save(f"{save_dir}/linewidth_Etot_phi_mean_detuning{detuning_label}.npy", linewidth_mean_mhz)
    np.save(f"{save_dir}/linewidth_Etot_phi_std_detuning{detuning_label}.npy", linewidth_std_mhz)

    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=200, constrained_layout=True)
    # Moving-average plots (NaN-safe)
    def moving_average_nan_safe(x, window):
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(x).astype(float)
        x_filled = np.where(np.isfinite(x), x, 0.0)
        kernel = np.ones(int(window), dtype=float)
        num = np.convolve(x_filled, kernel, mode='same')
        den = np.convolve(mask, kernel, mode='same')
        return num / np.where(den == 0, np.nan, den)

    ma_window = 1  # adjust as needed
    linewidth_mean_ma = moving_average_nan_safe(linewidth_mean_mhz, ma_window)
    linewidth_lower_ma = moving_average_nan_safe(linewidth_lower_mhz, ma_window)
    linewidth_upper_ma = moving_average_nan_safe(linewidth_upper_mhz, ma_window)

    n = min(len(kappa_ref), len(linewidth_mean_ma))
    kappa_plot = kappa_ref[:n] * 1e-9

    plt.plot(
        kappa_plot,
        linewidth_mean_ma[:n],
        linewidth=2,
        color='blue',
        label=rf'$E_{{tot}}$ mean over $\phi_p$',
        linestyle='-'
    )
    plt.fill_between(
        kappa_plot,
        linewidth_lower_ma[:n],
        linewidth_upper_ma[:n],
        color='blue',
        alpha=0.2,
        linewidth=0,
        label=rf'$\pm 1\sigma$ over $\phi_p$',
    )
    resolution_mhz = linewidth_resolution_floor_mhz(linewidth_floor_tmax_seconds)
    plot_ylim = (linewidth_plot_ylim_mhz[0], 5e1)
    if plot_ylim[0] <= resolution_mhz <= plot_ylim[1]:
        plt.axhline(
            resolution_mhz,
            color="red",
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            zorder=2,
            label=r"$1.44/T_\mathrm{max}$",
        )
    plt.xlabel(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=28)
    plt.ylabel('3 dB Linewidth (MHz)', fontsize=28)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5, which='both')


    plt.legend(loc='upper right', fontsize=20)

    # plt.xlim(7.25,11)
    plt.ylim(*plot_ylim)
    plt.xticks(np.linspace(kappa_plot[0], kappa_plot[-1], 6), fontsize=24)
    plt.yticks(fontsize=24)
    detuning_distribution_plot_ghz = np.unique(np.round(np.asarray(detuning_vals_ghz, dtype=float).ravel(), 6))
    detuning_distribution_label = ", ".join(f"{d:.2f}" for d in detuning_distribution_plot_ghz)
    plt.title(
        rf"{N_lasers} Laser Linewidth, $\delta_i=[{detuning_distribution_label}]$ GHz",
        fontsize=28,
    )
    plt.savefig(
        f"{output_dirs['base']}/fwhm_linewidth_Etot_phi_avg_detuning{detuning:.2f}.png",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.show()





#%%

font_size = 22
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

# --- Precompute peak trajectories ---
n_cases = len(kappa_c)
peak_freq_tot = np.full(n_cases, np.nan)

for k in range(n_cases):
    # Total field
    row = spectrum_db_norm_list[k]
    peaks, _ = find_peaks(row)
    if len(peaks):
        p = peaks[np.argmax(row[peaks])]
        peak_freq_tot[k] = f_plot[p]*1e-9  # GHz

if "detuning_vals_ghz" in globals():
    detuning_plot_vals_ghz = np.asarray(detuning_vals_ghz, dtype=float)
    if detuning_plot_vals_ghz.ndim == 1:
        detuning_plot_vals_ghz = detuning_plot_vals_ghz.reshape(1, -1)
    if detuning_plot_vals_ghz.shape[-1] != len(kappa_c):
        detuning_plot_vals_ghz = np.full((1, len(kappa_c)), detuning, dtype=float)
else:
    detuning_plot_vals_ghz = np.full((1, len(kappa_c)), detuning, dtype=float)

# --- Total spectrum ---
im = ax1.imshow(
    spectrum_db_norm_list,
    aspect='auto',
    extent=[f_plot[0]*1e-9, f_plot[-1]*1e-9, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
    origin='lower',
    cmap='viridis',
    rasterized=True
)
cbar = fig.colorbar(im, ax=ax1, pad=0.04)
cbar.set_label(r'Power (dBm)', fontsize=font_size, labelpad=12)
cbar.ax.tick_params(labelsize=font_size)
im.set_clim(-200, 0)
ax1.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
ax1.set_title(r"$|\mathcal{F}\left(E_{tot}\right)|^2$", fontsize=font_size, pad=16)
ax1.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
ax1.set_xticks(np.linspace(f_plot_min, f_plot_max, 3))
ax1.set_xlim(f_plot_min, f_plot_max)
ax1.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
ax1.tick_params(axis='both', labelsize=font_size)
for delta_i, delta_curve_ghz in enumerate(detuning_plot_vals_ghz):
    ax1.plot(
        delta_curve_ghz,
        kappa_c*1e-9,
        color='black',
        linestyle='--',
        linewidth=2,
        label=rf"$\delta_i$" if delta_i == 0 else None,
        alpha=0.5,
    )
# Peak path
valid_tot = ~np.isnan(peak_freq_tot)
ax1.plot(peak_freq_tot[valid_tot], kappa_c[valid_tot]*1e-9, color='red', lw=2, alpha=0.5)
# ax1.scatter(peak_freq_tot[valid_tot], kappa_c[valid_tot]*1e-9, color='red', s=10)
ax1.legend(fontsize=font_size, loc='upper left')

plt.tight_layout()
plt.show()
plt.close(fig)


#%%
# --- Build linewidth map over phi_p and kappa_c for E_tot ---
import glob
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import zoom

linewidth_level_db = 3.0
N_lasers = int(globals().get("N_lasers", 3))
output_dirs = linewidth_output_dirs(N_lasers)
map_save_dir = output_dirs["base"]
array_dir = f"{map_save_dir}/numpy_arrays"
os.makedirs(map_save_dir, exist_ok=True)

detuning = float(detuning_ghz if "detuning_ghz" in globals() else detuning)
detuning_label = globals().get("detuning_label", f"{detuning:.2f}")
self_feedback = globals().get("self_feedback", 0.0)
tau_p = globals().get("tau_p", 5.4e-12)

if "frequency_axis_for_spectrum" not in globals():
    def frequency_axis_for_spectrum(f_sorted, n_freqs, tau_p, f_plot_min_ghz=None, f_plot_max_ghz=None):
        """Return a physical frequency axis whose length matches a saved spectrum."""
        f_sorted = np.asarray(f_sorted)
        if len(f_sorted) == n_freqs:
            return f_sorted / tau_p

        if f_plot_min_ghz is not None and f_plot_max_ghz is not None:
            mask = (
                (f_sorted >= f_plot_min_ghz * 1e9 * tau_p)
                & (f_sorted <= f_plot_max_ghz * 1e9 * tau_p)
            )
            if np.count_nonzero(mask) == n_freqs:
                return f_sorted[mask] / tau_p

        center_idx = int(np.argmin(np.abs(f_sorted)))
        start = max(0, center_idx - n_freqs // 2)
        stop = start + n_freqs
        if stop > len(f_sorted):
            stop = len(f_sorted)
            start = stop - n_freqs
        if start < 0 or stop > len(f_sorted):
            raise ValueError(
                f"Cannot build a frequency axis with {n_freqs} points from "
                f"f_sorted with {len(f_sorted)} points."
            )
        return f_sorted[start:stop] / tau_p

pattern = (
    f"{array_dir}/spectrum_db_norm_list_"
    f"self{self_feedback:.2f}_detuning{detuning_label}_phiidx*_phipi*.npy"
)
spectrum_files = sorted(glob.glob(pattern))
if not spectrum_files:
    raise FileNotFoundError(
        f"No phi sweep spectra found for pattern:\n{pattern}\n"
        "Run the first simulation cell over phi_p first."
    )

if "linewidth_from_db_spectrum_reference_style" not in globals():
    def linewidth_from_db_spectrum_reference_style(spectrum_db, freqs_hz, level_db=3.0):
        """Use the same peak-centered threshold-crossing linewidth as linewidth.py."""
        spectrum = np.asarray(spectrum_db, dtype=float)
        freqs = np.asarray(freqs_hz, dtype=float) * 1e-6

        peaks, props = find_peaks(spectrum, height=0)
        if len(peaks) > 0:
            peak_pos = peaks[np.argmax(props["peak_heights"])]
        else:
            peak_pos = int(np.nanargmax(spectrum))

        shifted_freq = freqs - freqs[peak_pos]
        target_level = spectrum[peak_pos] - level_db

        left_idx = np.where(spectrum[:peak_pos] <= target_level)[0]
        if len(left_idx) > 0:
            i = left_idx[-1]
            freq_left = np.interp(
                target_level,
                [spectrum[i], spectrum[i + 1]],
                [shifted_freq[i], shifted_freq[i + 1]],
            )
        else:
            freq_left = shifted_freq[0]

        right_idx = np.where(spectrum[peak_pos:] <= target_level)[0]
        if len(right_idx) > 0:
            i = right_idx[0] + peak_pos
            freq_right = np.interp(
                target_level,
                [spectrum[i - 1], spectrum[i]],
                [shifted_freq[i - 1], shifted_freq[i]],
            )
        else:
            freq_right = shifted_freq[-1]

        linewidth = freq_right - freq_left
        return linewidth, freq_left, freq_right, target_level, peak_pos, shifted_freq

def linewidth_from_db_spectrum(spectrum_db, freqs_hz, level_db=3.0):
    """Return peak-centered width in MHz at peak - level_db."""
    linewidth, *_ = linewidth_from_db_spectrum_reference_style(
        spectrum_db,
        freqs_hz,
        level_db=level_db,
    )
    return linewidth

phi_vals = []
linewidth_columns = []
kappa_c_ref = None

for spectrum_path in spectrum_files:
    basename = os.path.basename(spectrum_path)
    run_label = basename.removeprefix("spectrum_db_norm_list_").removesuffix(".npy")
    match = re.search(r"_phipi([-+0-9.]+)$", run_label)
    if not match:
        raise ValueError(f"Could not parse phi_p/pi from {basename}")

    spectrum = np.load(spectrum_path)
    f_sorted_run = np.load(f"{array_dir}/f_sorted_{run_label}.npy")
    kappa_c_run = np.load(f"{array_dir}/kappa_c_{run_label}.npy")
    phi_path = f"{array_dir}/phi_p_{run_label}.npy"
    phi_value = float(np.load(phi_path)[0]) if os.path.exists(phi_path) else float(match.group(1)) * np.pi
    freqs_hz = frequency_axis_for_spectrum(
        f_sorted_run,
        spectrum.shape[1],
        tau_p,
        f_plot_min_ghz=0.0,
        f_plot_max_ghz=15.0,
    )

    linewidth_mhz = np.array([
        linewidth_from_db_spectrum(row, freqs_hz, level_db=linewidth_level_db)
        for row in spectrum
    ])

    phi_vals.append(phi_value)
    linewidth_columns.append(linewidth_mhz)
    if kappa_c_ref is None:
        kappa_c_ref = kappa_c_run
    elif len(kappa_c_ref) != len(kappa_c_run) or not np.allclose(kappa_c_ref, kappa_c_run):
        raise ValueError("Saved phi_p runs do not share the same kappa_c grid.")

sort_idx = np.argsort(phi_vals)
phi_vals = np.asarray(phi_vals)[sort_idx]
linewidth_map_mhz = np.column_stack(linewidth_columns)[:, sort_idx]

map_label = f"self{self_feedback:.2f}_detuning{detuning_label}_{linewidth_level_db:.0f}db"
np.save(f"{array_dir}/linewidth_map_Etot_{map_label}.npy", linewidth_map_mhz)
np.save(f"{array_dir}/linewidth_map_phi_vals_{map_label}.npy", phi_vals)
np.save(f"{array_dir}/linewidth_map_kappa_c_{map_label}.npy", kappa_c_ref)



fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
linewidth_colorbar_min_mhz = 1
linewidth_colorbar_max_mhz = 50.0
linewidth_map_plot_mhz = np.clip(
    linewidth_map_mhz,
    linewidth_colorbar_min_mhz,
    linewidth_colorbar_max_mhz,
)
linewidth_map_interp_factor = 4
linewidth_map_plot_mhz = np.exp(
    zoom(
        np.log(linewidth_map_plot_mhz),
        linewidth_map_interp_factor,
        order=3,
        mode="nearest",
    )
)
im = ax.imshow(
    linewidth_map_plot_mhz,
    aspect="auto",
    origin="lower",
    extent=[
        phi_vals[0] / np.pi,
        phi_vals[-1] / np.pi,
        kappa_c_ref[0] * 1e-9,
        kappa_c_ref[-1] * 1e-9,
    ],
    cmap="viridis",
    norm=matplotlib.colors.LogNorm(
        vmin=linewidth_colorbar_min_mhz,
        vmax=linewidth_colorbar_max_mhz,
    ),
    interpolation="bicubic",
    rasterized=True,
)
cbar = fig.colorbar(im, ax=ax, pad=0.04)
cbar.set_label(f"{linewidth_level_db:.0f} dB linewidth (MHz)", fontsize=24)
cbar.set_ticks([linewidth_colorbar_min_mhz, 10, linewidth_colorbar_max_mhz])
cbar.set_ticklabels(["1", "10", "50"])
cbar.ax.tick_params(labelsize=20)
ax.set_xlabel(r"$\phi_p/\pi$", fontsize=26)
ax.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=26)
ax.set_title(r"$E_{tot}$ linewidth map", fontsize=28, pad=12)
ax.tick_params(axis="both", labelsize=20)
ax.set_xticks(np.linspace(0, 2, 5))
ax.set_yticks(np.linspace(kappa_c_ref[0] * 1e-9, kappa_c_ref[-1] * 1e-9, 6))
plt.tight_layout()
plt.savefig(f"{map_save_dir}/linewidth_map_Etot_{map_label}.png", bbox_inches="tight")
plt.show()
plt.close(fig)
