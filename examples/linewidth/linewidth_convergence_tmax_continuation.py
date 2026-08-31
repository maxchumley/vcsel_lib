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
import queue as queue_module
import shutil
import time
import inspect
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import uniform_filter1d
from IPython.display import clear_output
from joblib import Parallel, delayed, parallel
try:
    from examples._paths import LINEWIDTH_RESULTS_DIR
except ModuleNotFoundError:
    LINEWIDTH_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "linewidth_estimation"
try:
    from tqdm.notebook import tqdm
except Exception:
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


def continuation_history_from_saved(y, nd, final_history=None):
    """Return the full 2*tau history required by the current integrator API."""
    if final_history is not None:
        delay_steps_nd = int(nd['delay_steps'])
        history_len = 2 * delay_steps_nd
        if final_history.shape[2] < history_len:
            raise ValueError(
                f"Returned final history has only {final_history.shape[2]} samples, "
                f"but continuation requires {history_len} samples."
            )
        return final_history[:, :, -history_len:].copy()

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
    return y[:, :, -history_len:].copy()



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

phi_p_value = 0.0  # fixed phase for the convergence sweep
Tmax_values = np.linspace(10e-6, 50.0e-6, 5)
run_tmax_sweep = True
save_spectrum_frames = True
spectrum_frame_stride = 10
trajectory_save_every = 5
spectrum_trajectory_save_every = 5
trajectory_output_dtype = np.float32
linewidth_level_db = 3.0
linewidth_smooth_sigma_bins = 1.0
linewidth_estimate_mode = "relative_db"  # optical spectrum: peak-minus-linewidth_level_db
linewidth_plot_ylim_mhz = (0.1, 300.0)
linewidth_plot_yticks_mhz = [0.01, 0.1, 1.0, 10.0, 100.0]
use_two_stage_tmax_spectrum = True
Tmax_continuation_requested = 1.0e-6
two_stage_kappa_chunk_size = 10
long_spectrum_jobs = 3
vectorize_two_stage_kappa_chunks = True
save_after_each_long_integration = False
spectrum_max_saved_freq_points = 60001
spectrum_frequency_window_ghz = 6.0
spectrum_output_field_only = True
spectrum_max_output_gb = 5.0
plot_psd_in_spectrum_panel = True
psd_plot_dynamic_range_db = 100.0
estimate_white_noise_floor = True
white_noise_floor_edge_fraction = 0.20
white_noise_floor_percentile = 50.0
white_noise_floor_min_bins = 16
show_two_stage_progress = True
show_long_run_worker_progress = True
long_run_progress_update_steps = 200
n_tmax_jobs = 1 if use_two_stage_tmax_spectrum else min(len(Tmax_values), os.cpu_count() or 1)

# Coarser timesteps make long Tmax sweeps much more practical.  The library
# integrator still advances on this coarse dt, while the optional controls below
# make delay lookup/noise updates less brittle when dt is several tau_p.
dt_multiplier = 2.0
# "trapezoid" is the iterative Heun / predictor-corrector path in VCSEL.integrate.
integration_scheme = "trapezoid"
integrator_theta = 0.5
integrator_max_iter = 5
delay_interpolation = "linear"  # None, "linear", or "cubic"
max_noise_substep_dt_tau_p = 1.0
load_only_current_dt = True


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


def linewidth_quantity_label():
    """Human-readable label for the active linewidth estimator."""
    mode = str(linewidth_estimate_mode).lower()
    if mode in {"white_noise_floor", "floor", "noise_floor"}:
        return "floor-corrected FWHM"
    return rf"{linewidth_level_db:.0f} dB linewidth"


def linewidth_output_dirs(n_lasers):
    """Return output directories rooted under linewidth_estimation/{N}_lasers/tmax_linewidth_convergence."""
    base_dir = LINEWIDTH_RESULTS_DIR / f"{int(n_lasers)}_lasers/tmax_linewidth_convergence"
    return {
        "base": base_dir,
        "arrays": f"{base_dir}/numpy_arrays",
        "frames": f"{base_dir}/spectrum_frames_detuning",
        "tex_cache": f"{base_dir}/matplotlib_tex_cache",
    }


def all_paths_exist(paths):
    """Return True when every expected output exists."""
    return all(os.path.exists(path) for path in paths)


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


def linewidth_from_psd_white_floor(
    spectrum_psd_dbmhz,
    freqs_hz,
    white_noise_floor_dbmhz,
    smooth_sigma_bins=0.0,
):
    """Floor-aware FWHM in Hz using half-height above the white PSD floor."""
    spectrum_psd_dbmhz = np.asarray(spectrum_psd_dbmhz, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    finite = np.isfinite(spectrum_psd_dbmhz) & np.isfinite(freqs_hz)
    if np.count_nonzero(finite) < 3 or not np.isfinite(white_noise_floor_dbmhz):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.full_like(freqs_hz, np.nan)

    spectrum_psd_dbmhz = spectrum_psd_dbmhz[finite]
    freqs_hz = freqs_hz[finite]
    sort_idx = np.argsort(freqs_hz)
    spectrum_psd_dbmhz = spectrum_psd_dbmhz[sort_idx]
    freqs_hz = freqs_hz[sort_idx]
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        from scipy.ndimage import gaussian_filter1d

        spectrum_psd_dbmhz = gaussian_filter1d(
            spectrum_psd_dbmhz,
            sigma=float(smooth_sigma_bins),
            mode="nearest",
        )

    peak_idx = int(np.nanargmax(spectrum_psd_dbmhz))
    peak_level_dbmhz = float(spectrum_psd_dbmhz[peak_idx])
    floor_linear = 10.0 ** (float(white_noise_floor_dbmhz) / 10.0)
    peak_linear = 10.0 ** (peak_level_dbmhz / 10.0)
    if not np.isfinite(floor_linear) or not np.isfinite(peak_linear) or peak_linear <= floor_linear:
        return np.nan, np.nan, np.nan, np.nan, np.nan, freqs_hz - freqs_hz[peak_idx]

    target_linear = floor_linear + 0.5 * (peak_linear - floor_linear)
    target_level_dbmhz = 10.0 * np.log10(target_linear)
    shifted_freq = freqs_hz - freqs_hz[peak_idx]

    left = np.where(spectrum_psd_dbmhz[:peak_idx + 1] <= target_level_dbmhz)[0]
    right = np.where(spectrum_psd_dbmhz[peak_idx:] <= target_level_dbmhz)[0] + peak_idx
    if len(left) == 0 or len(right) == 0:
        return np.nan, np.nan, np.nan, target_level_dbmhz, white_noise_floor_dbmhz, shifted_freq

    li = left[-1]
    ri = right[0]
    if li == peak_idx or ri == peak_idx:
        return np.nan, np.nan, np.nan, target_level_dbmhz, white_noise_floor_dbmhz, shifted_freq

    freq_left = np.interp(
        target_level_dbmhz,
        [spectrum_psd_dbmhz[li], spectrum_psd_dbmhz[li + 1]],
        [shifted_freq[li], shifted_freq[li + 1]],
    )
    freq_right = np.interp(
        target_level_dbmhz,
        [spectrum_psd_dbmhz[ri - 1], spectrum_psd_dbmhz[ri]],
        [shifted_freq[ri - 1], shifted_freq[ri]],
    )
    linewidth = freq_right - freq_left
    return linewidth, freq_left, freq_right, target_level_dbmhz, white_noise_floor_dbmhz, shifted_freq


def linewidth_from_current_mode(
    spectrum_db_norm,
    freqs_hz,
    spectrum_psd_dbmhz=None,
    white_noise_floor_dbmhz=np.nan,
):
    """Dispatch linewidth estimation based on linewidth_estimate_mode."""
    mode = str(linewidth_estimate_mode).lower()
    if mode in {"white_noise_floor", "floor", "noise_floor"} and spectrum_psd_dbmhz is not None:
        linewidth = linewidth_from_psd_white_floor(
            spectrum_psd_dbmhz,
            freqs_hz,
            white_noise_floor_dbmhz,
            smooth_sigma_bins=linewidth_smooth_sigma_bins,
        )
        if np.isfinite(linewidth[0]):
            return linewidth

    return linewidth_from_db_spectrum_reference_style(
        spectrum_db_norm,
        freqs_hz,
        level_db=linewidth_level_db,
        smooth_sigma_bins=linewidth_smooth_sigma_bins,
    )


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


def average_cos_phase_trace_from_phi(phi):
    """Average cos(phi_i - phi_0) over cases and non-reference lasers."""
    if phi.shape[1] <= 1:
        return np.ones(phi.shape[-1], dtype=np.float32)
    avg_trace = np.zeros(phi.shape[-1], dtype=np.float32)
    phi_ref = phi[:, 0, :]
    for laser_idx in range(1, phi.shape[1]):
        avg_trace += np.mean(np.cos(phi[:, laser_idx, :] - phi_ref), axis=0).astype(np.float32)
    avg_trace /= max(1, phi.shape[1] - 1)
    return avg_trace


def downsample_frequency_axis(f_window, *arrays):
    """Limit saved spectral grids while preserving full-resolution linewidth fits."""
    max_points = int(max(3, spectrum_max_saved_freq_points))
    if len(f_window) <= max_points:
        return (f_window,) + arrays
    keep = np.unique(np.linspace(0, len(f_window) - 1, max_points, dtype=int))
    return (f_window[keep],) + tuple(arr[..., keep] for arr in arrays)


def psd_color_limits(psd_dbmhz_map):
    """Return stable color limits for PSD maps with incomplete rows."""
    finite = np.asarray(psd_dbmhz_map)[np.isfinite(psd_dbmhz_map)]
    if finite.size == 0:
        return -100.0, 0.0
    vmax = float(np.nanmax(finite))
    vmin = vmax - float(psd_plot_dynamic_range_db)
    return vmin, vmax


def estimate_white_noise_floor_dbmhz(psd_dbmhz, freqs_hz):
    """Estimate the flat PSD background from the outer frequency-window bins."""
    if not estimate_white_noise_floor:
        return np.nan

    psd_dbmhz = np.asarray(psd_dbmhz, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    finite = np.isfinite(psd_dbmhz) & np.isfinite(freqs_hz)
    if np.count_nonzero(finite) < int(white_noise_floor_min_bins):
        return np.nan

    abs_freq = np.abs(freqs_hz)
    max_abs_freq = np.nanmax(abs_freq[finite])
    if not np.isfinite(max_abs_freq) or max_abs_freq <= 0:
        return np.nan

    edge_fraction = float(np.clip(white_noise_floor_edge_fraction, 0.0, 1.0))
    edge_cutoff = (1.0 - edge_fraction) * max_abs_freq
    edge_mask = finite & (abs_freq >= edge_cutoff)
    if np.count_nonzero(edge_mask) < int(white_noise_floor_min_bins):
        edge_mask = finite

    return float(np.nanpercentile(psd_dbmhz[edge_mask], white_noise_floor_percentile))


def extract_s_phi_from_saved_state(y, N_lasers):
    """Return S and phi from either full [N,S,phi] or compact [S,phi] output."""
    if y.shape[1] == 3 * N_lasers:
        return np.maximum(y[:, 1::3, :], 0.0), y[:, 2::3, :]
    if y.shape[1] == 2 * N_lasers:
        return np.maximum(y[:, 0::2, :], 0.0), y[:, 1::2, :]
    raise ValueError(
        f"Expected {3 * N_lasers} full states or {2 * N_lasers} compact field states, "
        f"got {y.shape[1]} states."
    )


def order_parameter_from_s_phi(S, phi):
    """Kuramoto-like order parameter from saved intensity and phase arrays."""
    eps = 1e-12
    E_sum = np.zeros((S.shape[0], S.shape[-1]), dtype=np.complex64)
    denom_sum = np.zeros((S.shape[0], S.shape[-1]), dtype=np.float32)
    for laser_idx in range(S.shape[1]):
        S_i = np.maximum(S[:, laser_idx, :], eps).astype(np.float32, copy=False)
        E_i = (
            np.sqrt(S_i)
            * np.exp((1j * phi[:, laser_idx, :]).astype(np.complex64))
        ).astype(np.complex64, copy=False)
        E_sum += E_i
        denom_sum += S_i
        del E_i
    numerator = np.abs(E_sum) ** 2
    denom = S.shape[1] * denom_sum
    return np.mean(numerator / np.maximum(denom, eps), axis=1)


def compute_tmax_spectra_and_linewidths(y, t, tau_p, nu_0, N_lasers):
    """Compute total spectrum, per-field linewidths, Etot linewidth, order, and cosine trace."""
    from scipy.signal import welch

    h = 6.626e-34

    if len(t) < 2:
        raise ValueError("Need at least two saved samples for spectral analysis.")
    saved_dt_nd = float(np.median(np.diff(t)) / tau_p)
    fs = 1.0 / saved_dt_nd
    desired_df = 0.5e6 * tau_p

    S, phi = extract_s_phi_from_saved_state(y, N_lasers)
    order_param = np.float32(
        np.mean(order_parameter_from_s_phi(S[:, :, -int(S.shape[-1] / 2):], phi[:, :, -int(phi.shape[-1] / 2):]))
    )
    nperseg = S.shape[-1]
    if nperseg < 8:
        raise ValueError(
            f"Only {nperseg} saved samples are available for spectra. "
            "Decrease dt_multiplier, decrease spectrum_trajectory_save_every, or increase Tmax."
        )
    noverlap = nperseg // 2
    N_fft = max(int(np.ceil(fs / desired_df)), nperseg)

    E_tot = np.zeros((S.shape[0], S.shape[-1]), dtype=np.complex64)
    for laser_idx in range(N_lasers):
        E_i = (
            np.sqrt(S[:, laser_idx, :].astype(np.float32, copy=False))
            * np.exp((1j * phi[:, laser_idx, :]).astype(np.complex64))
        ).astype(np.complex64, copy=False)
        E_tot += E_i
        del E_i

    f, psd_tot = welch(
        E_tot,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=N_fft,
        return_onesided=False,
        scaling="density",
        axis=-1,
    )
    # Welch was computed with a dimensionless sampling frequency. Multiplying
    # by h*nu_0 converts the field-density estimate to W/Hz for plotting.
    psd_tot_watts_per_hz = np.maximum(np.real(psd_tot), 0.0) * h * float(nu_0)
    spectrum_psd_dbmhz = 10 * np.log10(np.mean(psd_tot_watts_per_hz, axis=0) / 1e-3 + 1e-20)

    idx_sort = np.argsort(f)
    f_sorted = f[idx_sort]
    f_plot_min = -float(spectrum_frequency_window_ghz)
    f_plot_max = float(spectrum_frequency_window_ghz)
    mask = (f_sorted >= f_plot_min * 1e9 * tau_p) & (f_sorted <= f_plot_max * 1e9 * tau_p)
    f_window_full = f_sorted[mask]
    if f_window_full.size < 3:
        raise ValueError(
            f"Frequency window [{f_plot_min}, {f_plot_max}] GHz has only "
            f"{f_window_full.size} bins."
        )

    spectrum_psd_dbmhz = spectrum_psd_dbmhz[idx_sort][mask]
    white_noise_floor_dbmhz = estimate_white_noise_floor_dbmhz(
        spectrum_psd_dbmhz,
        f_window_full / tau_p,
    )
    spectrum_db_norm_full = (spectrum_psd_dbmhz - np.max(spectrum_psd_dbmhz)).astype(np.float32)
    linewidth_mhz, *_ = linewidth_from_current_mode(
        spectrum_db_norm_full,
        f_window_full / tau_p,
        spectrum_psd_dbmhz=spectrum_psd_dbmhz,
        white_noise_floor_dbmhz=white_noise_floor_dbmhz,
    )
    linewidth_mhz *= 1e-6

    laser_linewidths_mhz = np.full(N_lasers, np.nan, dtype=np.float32)
    laser_white_noise_floor_dbmhz = np.full(N_lasers, np.nan, dtype=np.float32)
    for laser_idx in range(N_lasers):
        E_i = (
            np.sqrt(S[:, laser_idx, :].astype(np.float32, copy=False))
            * np.exp((1j * phi[:, laser_idx, :]).astype(np.complex64))
        ).astype(np.complex64, copy=False)
        _, psd_laser = welch(
            E_i,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=N_fft,
            return_onesided=False,
            scaling="density",
            axis=-1,
        )
        del E_i
        psd_laser_watts_per_hz = np.maximum(np.real(psd_laser), 0.0) * h * float(nu_0)
        spectrum_laser_db = 10 * np.log10(np.mean(psd_laser_watts_per_hz, axis=0) / 1e-3 + 1e-20)
        spectrum_laser_db = spectrum_laser_db[idx_sort][mask]
        laser_white_noise_floor_dbmhz[laser_idx] = estimate_white_noise_floor_dbmhz(
            spectrum_laser_db,
            f_window_full / tau_p,
        )
        spectrum_laser_db_norm = spectrum_laser_db - np.max(spectrum_laser_db)
        laser_linewidths_mhz[laser_idx], *_ = linewidth_from_current_mode(
            spectrum_laser_db_norm,
            f_window_full / tau_p,
            spectrum_psd_dbmhz=spectrum_laser_db,
            white_noise_floor_dbmhz=laser_white_noise_floor_dbmhz[laser_idx],
        )
        laser_linewidths_mhz[laser_idx] *= 1e-6

    f_window, spectrum_db_norm, spectrum_psd_dbmhz = downsample_frequency_axis(
        f_window_full.astype(np.float32),
        spectrum_db_norm_full,
        spectrum_psd_dbmhz.astype(np.float32),
    )

    cos_trace = average_cos_phase_trace_from_phi(phi)
    del S, phi, E_tot
    return {
        "f_window": f_window,
        "spectrum_db_norm": spectrum_db_norm,
        "spectrum_psd_dbmhz": spectrum_psd_dbmhz,
        "linewidth_mhz": np.float32(linewidth_mhz),
        "laser_linewidths_mhz": laser_linewidths_mhz,
        "white_noise_floor_dbmhz": np.float32(white_noise_floor_dbmhz),
        "laser_white_noise_floor_dbmhz": laser_white_noise_floor_dbmhz,
        "cos_trace": cos_trace,
        "order_param": order_param,
    }


def run_tmax_fixed_kappa_spectrum_worker(job):
    """Run a long fixed-kappa spectral measurement for one kappa value."""
    k_job = int(job["k"])
    kappa_job = float(job["kappa"])
    phys_spec = dict(job["phys"])
    phys_spec["Tmax"] = float(job["Tmax_spectrum"])
    phys_spec["save_every"] = int(max(1, job["spectrum_trajectory_save_every"]))
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
    nd_spec = apply_linewidth_integrator_controls(vcsel_spec.scale_params(), job["noise_substeps"])
    nd_spec["store_freqs"] = False
    if spectrum_max_output_gb is not None:
        nd_spec["max_output_gb"] = float(spectrum_max_output_gb)
    if spectrum_output_field_only:
        field_output_indices = np.ravel(
            np.column_stack((
                np.arange(int(job["N_lasers"])) * 3 + 1,
                np.arange(int(job["N_lasers"])) * 3 + 2,
            ))
        )
        nd_spec["output_state_indices"] = field_output_indices
    progress_callback, flush_progress = make_worker_progress_callback(job)

    try:
        t_spec, y_spec, _ = integrate_with_optional_progress_callback(
            vcsel_spec,
            job["history"],
            nd=nd_spec,
            progress=False,
            theta=float(job["integrator_theta"]),
            max_iter=int(job["integrator_max_iter"]),
            smooth_freqs=False,
            integration_scheme=job["integration_scheme"],
            return_final_history=False,
            progress_callback=progress_callback,
        )
        flush_progress()
    except Exception as exc:
        flush_progress()
        raise RuntimeError(
            f"Long Tmax spectrum failed for Tmax index {job['tmax_idx']}, "
            f"kappa={kappa_job*1e-9:.3f} ns^-1."
        ) from exc
    if not np.all(np.isfinite(y_spec)):
        raise FloatingPointError(
            f"Non-finite state in long Tmax spectrum for Tmax index {job['tmax_idx']}, "
            f"kappa={kappa_job*1e-9:.3f} ns^-1."
        )

    result = compute_tmax_spectra_and_linewidths(
        y_spec,
        t_spec,
        float(job["tau_p"]),
        float(job["nu_0"]),
        int(job["N_lasers"]),
    )
    result["k"] = k_job
    result["kappa"] = kappa_job
    del y_spec
    return result


def run_tmax_fixed_kappa_spectrum_chunk_worker(jobs):
    """Run one kappa chunk as one vectorized long spectral integration."""
    jobs = list(jobs)
    if not jobs:
        return []
    if len(jobs) == 1:
        return [run_tmax_fixed_kappa_spectrum_worker(jobs[0])]

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
        history_job = np.asarray(job["history"])
        n_cases_job = history_job.shape[0]
        histories.append(history_job)
        case_counts.append(n_cases_job)
        phi_job = np.asarray(job["phys"]["phi_p_mat"])
        if phi_job.ndim == 2:
            phi_job = np.repeat(phi_job[None, :, :], n_cases_job, axis=0)
        elif phi_job.ndim == 3 and phi_job.shape[0] == 1:
            phi_job = np.repeat(phi_job, n_cases_job, axis=0)
        elif phi_job.ndim == 3 and phi_job.shape[0] == n_cases_job:
            pass
        else:
            raise ValueError(f"Unsupported phi_p_mat shape in chunk: {phi_job.shape}")
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
    phys_spec["save_every"] = int(max(1, first_job["spectrum_trajectory_save_every"]))
    phys_spec["phi_p_mat"] = phi_all
    phys_spec["kappa_c_mat"] = kappa_all

    vcsel_spec = VCSEL(phys_spec)
    nd_spec = apply_linewidth_integrator_controls(vcsel_spec.scale_params(), first_job["noise_substeps"])
    nd_spec["kappa_case_dependent"] = True
    nd_spec["store_freqs"] = False
    if spectrum_max_output_gb is not None:
        nd_spec["max_output_gb"] = float(spectrum_max_output_gb)
    if spectrum_output_field_only:
        field_output_indices = np.ravel(
            np.column_stack((
                np.arange(N_lasers) * 3 + 1,
                np.arange(N_lasers) * 3 + 2,
            ))
        )
        nd_spec["output_state_indices"] = field_output_indices
    progress_callback, flush_progress = make_worker_progress_callback(first_job)

    try:
        t_spec, y_spec, _ = integrate_with_optional_progress_callback(
            vcsel_spec,
            history_all,
            nd=nd_spec,
            progress=False,
            theta=float(first_job["integrator_theta"]),
            max_iter=int(first_job["integrator_max_iter"]),
            smooth_freqs=False,
            integration_scheme=first_job["integration_scheme"],
            return_final_history=False,
            progress_callback=progress_callback,
        )
        flush_progress()
    except Exception as exc:
        flush_progress()
        raise RuntimeError(
            f"Vectorized long Tmax spectrum failed for Tmax index {first_job['tmax_idx']}, "
            f"k={int(jobs[0]['k'])}-{int(jobs[-1]['k'])}."
        ) from exc
    if not np.all(np.isfinite(y_spec)):
        raise FloatingPointError(
            f"Non-finite state in vectorized long Tmax spectrum for Tmax index "
            f"{first_job['tmax_idx']}, k={int(jobs[0]['k'])}-{int(jobs[-1]['k'])}."
        )

    results = []
    offset = 0
    for job, n_cases_job in zip(jobs, case_counts):
        y_job = y_spec[offset:offset + n_cases_job]
        offset += n_cases_job
        result = compute_tmax_spectra_and_linewidths(
            y_job,
            t_spec,
            tau_p,
            float(job["nu_0"]),
            N_lasers,
        )
        result["k"] = int(job["k"])
        result["kappa"] = float(job["kappa"])
        results.append(result)
        del y_job

    del history_all, phi_all, kappa_all, y_spec
    return results


def run_tmax_continuation(tmax_idx):
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
    noise_amplitude = 1.0

    N_lasers = 3
    detuning_ghz = 4.0
    detuning = detuning_ghz  # detuning (GHz)
    delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
    delta_distribution = np.sort(np.linspace(-delta/2, delta/2, N_lasers))


    dt = float(dt_multiplier) * tau_p
    Tmax_spectrum_requested = float(Tmax_values[tmax_idx])
    Tmax_requested = (
        float(Tmax_continuation_requested)
        if use_two_stage_tmax_spectrum
        else Tmax_spectrum_requested
    )
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
    if use_two_stage_tmax_spectrum:
        spectrum_steps = int(Tmax_spectrum_requested / dt)
        if spectrum_steps <= 2 * delay_steps + 4:
            raise ValueError(
                f"Tmax={Tmax_spectrum_requested*1e6:g} us and dt={dt/tau_p:g} tau_p give "
                f"only {spectrum_steps} samples, but spectral measurement needs more than "
                f"{2 * delay_steps} samples for the 2*tau history."
            )
        Tmax_for_linewidth = spectrum_steps * dt
        time_arr_spectrum = np.arange(spectrum_steps, dtype=float) * dt
    else:
        spectrum_steps = steps
        Tmax_for_linewidth = Tmax
        time_arr_spectrum = time_arr
    segment_start = steps // 2
    segment_len = steps - segment_start
    cos_window_ns = 200.0
    saved_dt = dt * int(max(1, trajectory_save_every))
    saved_steps_est = max(1, int(np.ceil(steps / int(max(1, trajectory_save_every)))))
    cos_saved_dt = (
        dt * int(max(1, spectrum_trajectory_save_every))
        if use_two_stage_tmax_spectrum
        else saved_dt
    )
    cos_saved_steps_est = (
        max(1, int(np.ceil(spectrum_steps / int(max(1, spectrum_trajectory_save_every)))))
        if use_two_stage_tmax_spectrum
        else saved_steps_est
    )
    cos_trace_len = min(cos_saved_steps_est, int(np.ceil(cos_window_ns*1e-9 / cos_saved_dt)))
 
    resolution = 200
    output_dirs = linewidth_output_dirs(N_lasers)
    save_dir = output_dirs["arrays"]
    frame_dir = output_dirs["frames"]
    tex_cache_dir = f"{output_dirs['tex_cache']}/tmax_{tmax_idx:03d}_pid{os.getpid()}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    shutil.rmtree(tex_cache_dir, ignore_errors=True)
    set_latex_plot_style(tex_cache_dir=tex_cache_dir)
    coupling_scheme = 'CUSTOM'  # 'ATA', 'NN' or 'RANDOM'
    ramp_start = 2

    dx=1.0
    aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)

    kappa_c = np.linspace(0e9,120e9,resolution)
    detuning_label = f"{detuning_ghz:.2f}"



    fixed_phi_p = float(phi_p_value)
    phi_p_vals = np.array([fixed_phi_p])
    noise_substeps = 1
    if max_noise_substep_dt_tau_p is not None:
        noise_substeps = max(
            1,
            int(np.ceil((dt / tau_p) / float(max_noise_substep_dt_tau_p))),
        )

    tmax_label = f"tmaxidx{tmax_idx:03d}_tmaxus{Tmax_for_linewidth*1e6:.4f}"
    stage_label = (
        f"twostage_contus{filename_float(Tmax*1e6)}_specus{filename_float(Tmax_for_linewidth*1e6)}"
        if use_two_stage_tmax_spectrum
        else "singlestage"
    )
    phi_p_label = f"phipi{fixed_phi_p/np.pi:.4f}"
    scheme_label = f"scheme{integration_scheme.lower()}"
    dt_label = f"dtmult{filename_float(dt / tau_p)}"
    interp_label = f"interp{delay_interpolation or 'none'}"
    substep_label = f"nsub{noise_substeps}"
    save_label = f"saveevery{int(max(1, trajectory_save_every))}"
    run_label = (
        f"self{self_feedback:.2f}_detuning{detuning_label}_{phi_p_label}_"
        f"{tmax_label}_{stage_label}_{scheme_label}_{dt_label}_{save_label}_{interp_label}_{substep_label}"
    )
    settings_path = f"{save_dir}/integrator_settings_{run_label}.npz"
    final_output_paths = [
        f"{save_dir}/spectrum_db_norm_list_{run_label}.npy",
        f"{save_dir}/spectrum_psd_dbmhz_list_{run_label}.npy",
        f"{save_dir}/f_window_{run_label}.npy",
        f"{save_dir}/kappa_c_{run_label}.npy",
        f"{save_dir}/detuning_vals_ghz_{run_label}.npy",
        f"{save_dir}/phi_p_{run_label}.npy",
        f"{save_dir}/Tmax_{run_label}.npy",
        f"{save_dir}/linewidth_E_fields_mhz_{run_label}.npy",
        f"{save_dir}/linewidth_Etot_mhz_{run_label}.npy",
        f"{save_dir}/white_noise_floor_E_fields_dbmhz_{run_label}.npy",
        f"{save_dir}/white_noise_floor_Etot_dbmhz_{run_label}.npy",
        settings_path,
    ]
    if all_paths_exist(final_output_paths):
        print(
            f"Skipping Tmax index {tmax_idx:03d} "
            f"({Tmax*1e6:.4f} us): complete output files already exist.",
            flush=True,
        )
        return {
            "tmax_idx": tmax_idx,
            "run_label": run_label,
            "phi_p": fixed_phi_p,
            "Tmax": Tmax_for_linewidth,
            "detuning_label": detuning_label,
            "self_feedback": self_feedback,
            "N_lasers": N_lasers,
            "skipped_existing": True,
        }
    checkpoint_path = f"{save_dir}/checkpoint_{run_label}.npz"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

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
        'phi_p_mat': np.ones(shape=(n_iterations,N_lasers,N_lasers))*fixed_phi_p,
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
        # Keep only every nth trajectory sample for spectra/plots. The
        # integrator returns the final full-resolution 2*tau history separately
        # so continuation still has the exact delay history it needs.
        'save_every': int(max(1, trajectory_save_every)),
        'output_dtype': trajectory_output_dtype,
        'max_output_gb': None,
    }




    # t, y, freqs = vcsel.integrate(history, nd=nd, progress=True)



    ramp_start = 10
    ramp_shape = 100


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
    spectrum_psd_dbmhz_list = None
    cos_phase_diff_time = np.full((len(kappa_c), cos_trace_len), np.nan, dtype=dtype)
    order_param = np.zeros(len(kappa_c), dtype=dtype)
    linewidth_trace_mhz = np.full(len(kappa_c), np.nan, dtype=dtype)
    laser_linewidth_traces_mhz = np.full((N_lasers, len(kappa_c)), np.nan, dtype=dtype)
    white_noise_floor_trace_dbmhz = np.full(len(kappa_c), np.nan, dtype=dtype)
    laser_white_noise_floor_traces_dbmhz = np.full((N_lasers, len(kappa_c)), np.nan, dtype=dtype)
    f_window = None
    spectrum_jobs = []
    pending_spectrum_futures = []
    spectrum_future_progress_id = {}
    spectrum_progress_bars = {}
    spectrum_progress_queue = None
    next_progress_position = 1
    spectrum_executor = None

    import sys

    if use_two_stage_tmax_spectrum:
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

    def make_spectrum_job(k_job, kappa_job, history_job):
        phys_for_job = dict(phys)
        phys_for_job["kappa_c_mat"] = None
        return {
            "k": int(k_job),
            "kappa": float(kappa_job),
            "history": history_job.copy(),
            "phys": phys_for_job,
            "Tmax_spectrum": Tmax_for_linewidth,
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
            "spectrum_trajectory_save_every": int(max(1, spectrum_trajectory_save_every)),
            "tmax_idx": tmax_idx,
        }

    def make_long_run_progress_bar(progress_id, chunk_indices):
        nonlocal next_progress_position
        if spectrum_progress_queue is None:
            return
        total_steps = max(1, spectrum_steps - 2 * delay_steps)
        desc = (
            f"Tmax {tmax_idx + 1}/{len(Tmax_values)} long "
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
        nonlocal spectrum_jobs
        if not spectrum_jobs:
            return
        chunk = spectrum_jobs
        spectrum_jobs = []
        chunk_indices = [int(job["k"]) for job in chunk]
        progress_id = f"tmax{tmax_idx}_chunk{chunk_indices[0]}_{chunk_indices[-1]}"
        use_vectorized_chunk = (
            vectorize_two_stage_kappa_chunks
            and not save_after_each_long_integration
        )
        if use_vectorized_chunk:
            make_long_run_progress_bar(progress_id, chunk_indices)
            chunk = attach_long_run_progress(chunk, progress_id)
            future = spectrum_executor.submit(run_tmax_fixed_kappa_spectrum_chunk_worker, chunk)
            pending_spectrum_futures.append(future)
            spectrum_future_progress_id[future] = progress_id
        else:
            for job in chunk:
                single_progress_id = f"{progress_id}_k{int(job['k'])}"
                make_long_run_progress_bar(single_progress_id, [int(job["k"])])
                attach_long_run_progress([job], single_progress_id)
                future = spectrum_executor.submit(run_tmax_fixed_kappa_spectrum_worker, job)
                pending_spectrum_futures.append(future)
                spectrum_future_progress_id[future] = single_progress_id

    def store_spectrum_results(spectrum_results):
        nonlocal f_window, spectrum_db_norm_list, spectrum_psd_dbmhz_list
        if isinstance(spectrum_results, dict):
            spectrum_results = [spectrum_results]
        spectrum_results = sorted(spectrum_results, key=lambda item: item["k"])
        if not spectrum_results:
            return
        if f_window is None:
            f_window = spectrum_results[0]["f_window"]
            spectrum_db_norm_list = np.full((resolution, len(f_window)), np.nan, dtype=dtype)
            spectrum_psd_dbmhz_list = np.full((resolution, len(f_window)), np.nan, dtype=dtype)

        for result in spectrum_results:
            k_result = int(result["k"])
            if len(result["f_window"]) != len(f_window) or not np.allclose(result["f_window"], f_window):
                raise ValueError("Long Tmax spectra do not share the same frequency grid.")
            spectrum_db_norm_list[k_result] = result["spectrum_db_norm"]
            if "spectrum_psd_dbmhz" in result:
                spectrum_psd_dbmhz_list[k_result] = result["spectrum_psd_dbmhz"]
            linewidth_trace_mhz[k_result] = result["linewidth_mhz"]
            laser_linewidth_traces_mhz[:, k_result] = result["laser_linewidths_mhz"]
            if "white_noise_floor_dbmhz" in result:
                white_noise_floor_trace_dbmhz[k_result] = result["white_noise_floor_dbmhz"]
            if "laser_white_noise_floor_dbmhz" in result:
                laser_white_noise_floor_traces_dbmhz[:, k_result] = result["laser_white_noise_floor_dbmhz"]
            order_param[k_result] = result["order_param"]
            cos_trace = np.asarray(result["cos_trace"], dtype=dtype)
            cos_tail = cos_trace[-cos_trace_len:]
            cos_phase_diff_time[k_result, -len(cos_tail):] = cos_tail

    def save_two_stage_progress_arrays():
        if not use_two_stage_tmax_spectrum:
            return
        if f_window is None or spectrum_db_norm_list is None:
            return
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/spectrum_db_norm_list_{run_label}.npy", spectrum_db_norm_list)
        if spectrum_psd_dbmhz_list is not None:
            np.save(f"{save_dir}/spectrum_psd_dbmhz_list_{run_label}.npy", spectrum_psd_dbmhz_list)
        np.save(f"{save_dir}/f_window_{run_label}.npy", f_window)
        np.save(f"{save_dir}/kappa_c_{run_label}.npy", kappa_c)
        np.save(f"{save_dir}/detuning_vals_ghz_{run_label}.npy", detuning_vals_ghz)
        np.save(f"{save_dir}/phi_p_{run_label}.npy", np.array([fixed_phi_p]))
        np.save(f"{save_dir}/Tmax_{run_label}.npy", np.array([Tmax_for_linewidth]))
        np.save(f"{save_dir}/linewidth_E_fields_mhz_{run_label}.npy", laser_linewidth_traces_mhz)
        np.save(f"{save_dir}/linewidth_Etot_mhz_{run_label}.npy", linewidth_trace_mhz)
        np.save(f"{save_dir}/white_noise_floor_E_fields_dbmhz_{run_label}.npy", laser_white_noise_floor_traces_dbmhz)
        np.save(f"{save_dir}/white_noise_floor_Etot_dbmhz_{run_label}.npy", white_noise_floor_trace_dbmhz)
        np.save(f"{save_dir}/order_parameter_{run_label}.npy", order_param)
        np.save(f"{save_dir}/cos_phase_diff_time_{run_label}.npy", cos_phase_diff_time)

    def save_two_stage_progress_frame(k_completed=None):
        if not (use_two_stage_tmax_spectrum and save_spectrum_frames):
            return
        if f_window is None or spectrum_db_norm_list is None:
            return

        font_size = 22
        f_plot = f_window / tau_p
        f_plot_axis_min = f_plot[0] * 1e-9
        f_plot_axis_max = f_plot[-1] * 1e-9
        cos_time_ns = np.arange(cos_trace_len, dtype=float) * cos_saved_dt * 1e9

        fig = plt.figure(figsize=(18, 10), dpi=300)
        width_ratios = [1] * 30
        width_ratios[8] = 0.25
        gs = fig.add_gridspec(20, 30, height_ratios=[1] * 20, width_ratios=width_ratios, hspace=0.3)

        ax0 = fig.add_subplot(gs[1:8, 0:-3])
        im0 = ax0.imshow(
            cos_phase_diff_time,
            aspect="auto",
            extent=[cos_time_ns[0], cos_time_ns[-1], kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9],
            origin="lower",
            cmap="jet",
            vmin=-1,
            vmax=1,
            rasterized=True,
        )
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.02)
        cbar0.set_label(r"$\cos(\Delta\phi)$", fontsize=font_size, labelpad=0)
        cbar0.ax.tick_params(labelsize=font_size)
        ax0.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size)
        ax0.set_xlabel(r"Time (ns)", fontsize=font_size)
        ax0.set_xlim(cos_time_ns[0], cos_time_ns[-1])
        ax0.set_xticks(np.linspace(cos_time_ns[0], cos_time_ns[-1], 6))
        ax0.set_title(
            r"$T_\mathrm{{max}}={:.2f}\,\mu\mathrm{{s}},\ \phi_p={:+.2f}\pi,\ \delta_i=[{}]\,\mathrm{{GHz}}$".format(
                Tmax_for_linewidth * 1e6,
                fixed_phi_p / np.pi,
                ", ".join(f"{d:.1f}" for d in delta_distribution_ghz),
            ),
            fontsize=font_size,
            pad=16,
        )
        ax0.set_yticks(np.linspace(kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9, 6))
        ax0.tick_params(axis="both", labelsize=font_size)

        ax_order = fig.add_subplot(gs[1:8, -2:])
        ax_order.plot(order_param, kappa_c * 1e-9, color="black", linewidth=2)
        ax_order.set_title("Order Parameter", fontsize=font_size, pad=16)
        ax_order.set_ylim(kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9)
        ax_order.tick_params(axis="both", labelsize=font_size)
        ax_order.set_yticks([])
        ax_order.set_xticks(np.linspace(0, 1, 2))
        ax_order.set_xlim(0, 1)

        ax1 = fig.add_subplot(gs[11:, 1:8])
        spectrum_cmap = plt.get_cmap("jet").copy()
        spectrum_cmap.set_bad(color="white")
        if plot_psd_in_spectrum_panel and spectrum_psd_dbmhz_list is not None:
            spectrum_panel = spectrum_psd_dbmhz_list
            spectrum_vmin, spectrum_vmax = psd_color_limits(spectrum_panel)
            spectrum_colorbar_label = r"PSD (dBm/Hz)"
            spectrum_title = r"PSD$\left(E_{tot}\right)$"
        else:
            spectrum_panel = spectrum_db_norm_list
            spectrum_vmin, spectrum_vmax = -100.0, 0.0
            spectrum_colorbar_label = r"Relative PSD (dB)"
            spectrum_title = r"PSD$\left(E_{tot}\right)$, peak-normalized"
        im = ax1.imshow(
            spectrum_panel,
            aspect="auto",
            extent=[f_plot_axis_min, f_plot_axis_max, kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9],
            origin="lower",
            cmap=spectrum_cmap,
            rasterized=True,
        )
        im.set_clim(spectrum_vmin, spectrum_vmax)
        cbar1 = fig.colorbar(im, ax=ax1, pad=0.05)
        cbar1.set_label(spectrum_colorbar_label, fontsize=font_size, labelpad=12)
        cbar1.ax.tick_params(labelsize=font_size)
        ax1.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
        ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
        ax1.set_title(spectrum_title, fontsize=font_size, pad=16)
        ax1.set_yticks(np.linspace(kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9, 6))
        ax1.set_xticks(np.linspace(f_plot_axis_min, f_plot_axis_max, 3))
        ax1.set_xlim(f_plot_axis_min, f_plot_axis_max)
        ax1.set_ylim(kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9)
        ax1.tick_params(axis="both", labelsize=font_size)
        for delta_i, delta_ghz in enumerate(delta_distribution_ghz):
            ax1.plot(
                np.full_like(kappa_c, delta_ghz, dtype=float),
                kappa_c * 1e-9,
                color="black",
                linestyle="--",
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
                ax2.plot(
                    kappa_c[valid_laser_linewidth] * 1e-9,
                    laser_linewidth[valid_laser_linewidth],
                    color=field_colors(laser_idx % field_colors.N),
                    linewidth=1.4,
                    alpha=0.55,
                    zorder=1,
                    label=rf"$E_{laser_idx + 1}$",
                )
        valid_linewidth = np.isfinite(linewidth_trace_mhz) & (linewidth_trace_mhz > 0)
        ax2.plot(
            kappa_c[valid_linewidth] * 1e-9,
            linewidth_trace_mhz[valid_linewidth],
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
        ax2.set_title(f"{linewidth_quantity_label()} (MHz)", fontsize=linewidth_font_size, pad=16)
        ax2.set_xlim(kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9)
        ax2.set_yscale("log")
        ax2.set_ylim(*linewidth_plot_ylim_mhz)
        ax2.set_yticks(linewidth_plot_yticks_mhz)
        ax2.grid(True, linestyle="--", alpha=0.4, which="both")
        ax2.set_xticks(np.linspace(kappa_c[0] * 1e-9, kappa_c[-1] * 1e-9, 6))
        ax2.tick_params(axis="both", labelsize=linewidth_font_size)
        ax2.legend(fontsize=linewidth_font_size - 10, loc="upper right")

        os.makedirs(frame_dir, exist_ok=True)
        filename = f"{frame_dir}/tmax_{tmax_idx:03d}_{scheme_label}_{dt_label}_{interp_label}_{substep_label}.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        plt.cla()
        plt.clf()
        plt.close("all")
        del fig, gs, ax0, ax_order, ax1, ax2
        del im0, im, cbar0, cbar1

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

    def process_spectrum_future(future):
        spectrum_results = future.result()
        if isinstance(spectrum_results, dict):
            spectrum_results = [spectrum_results]
        result_indices = [int(result["k"]) for result in spectrum_results]
        store_spectrum_results(spectrum_results)
        close_long_run_progress_bar(future)
        save_two_stage_progress_arrays()
        if result_indices:
            save_two_stage_progress_frame(k_completed=max(result_indices))

    def collect_finished_spectrum_chunks(block=False):
        if not pending_spectrum_futures:
            return
        drain_long_run_progress_queue()
        if block:
            futures_to_collect = list(pending_spectrum_futures)
            pending_spectrum_futures.clear()
            collect_bar = tqdm(
                total=len(futures_to_collect),
                desc=f"Tmax {tmax_idx + 1}/{len(Tmax_values)} long spectra complete",
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



    show_kappa_progress = n_tmax_jobs == 1
    pbar = tqdm(
        range(len(kappa_c)),
        desc=f"Tmax {tmax_idx + 1}/{len(Tmax_values)}",
        unit="step",
        leave=False,
        position=0,
        dynamic_ncols=True,
        disable=not show_kappa_progress,
    )
    for k in pbar:
        kappa = kappa_c[k]
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
                f"Integration failed for Tmax index {tmax_idx}, "
                f"kappa={kappa*1e-9:.3f} ns^-1, dt={dt/tau_p:g} tau_p. "
                "Try decreasing dt_multiplier, increasing integrator_max_iter, "
                "or using delay_interpolation='linear'."
            ) from exc
        if not np.all(np.isfinite(y_scaled)):
            raise FloatingPointError(
                f"Non-finite state for Tmax index {tmax_idx}, "
                f"kappa={kappa*1e-9:.3f} ns^-1, dt={dt/tau_p:g} tau_p. "
                "Try decreasing dt_multiplier or max_noise_substep_dt_tau_p."
            )
        next_history = continuation_history_from_saved(y_scaled, nd, final_history=final_history)

        if use_two_stage_tmax_spectrum:
            spectrum_jobs.append(make_spectrum_job(k, kappa, next_history))
            if (
                len(spectrum_jobs) >= int(max(1, two_stage_kappa_chunk_size))
                or k == len(kappa_c) - 1
            ):
                submit_spectrum_job_chunk()
            collect_finished_spectrum_chunks(block=False)
            history = next_history
            del y_scaled, freqs, final_history
            if k % 5 == 0 or k == len(kappa_c) - 1:
                gc.collect()
            continue

        # Current VCSEL.integrate API returns:
        #   y     shape (n_cases, 3*N_lasers, n_saved_steps)
        #   freqs shape (n_cases, N_lasers, n_saved_steps)
        y = y_scaled
        S = np.maximum(y[:, 1::3, :], 0.0)
        phi = y[:, 2::3, :]
        del freqs

        # Pairwise phase differences (using laser 0 as reference). The cosine
        # does not need phase unwrapping, avoiding an extra full-size copy.
        phase_diff = phi - phi[:, 0:1, :]
        avg_cos_pd = np.mean(np.mean(np.cos(phase_diff), axis=0)[1:], axis=0)
        del phase_diff

        # Order parameter now works for N lasers
        order_param[k] = np.mean(vcsel.order_parameter(y[:,:,-int(len(t)/2):]))

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

        if len(t) < 2:
            raise ValueError("Need at least two saved samples for spectral analysis.")
        saved_dt_nd = float(np.median(np.diff(t)) / tau_p)
        fs = 1.0 / saved_dt_nd
        desired_df = 0.5e6 * tau_p
        nperseg = np.shape(E_all)[-1]
        if nperseg < 8:
            raise ValueError(
                f"Only {nperseg} saved samples are available for spectra. "
                "Decrease dt_multiplier or increase Tmax."
            )
        noverlap = nperseg // 2
        N_fft = max(int(np.ceil(fs / desired_df)), nperseg)

        


        # ---------- TOTAL FIELD SPECTRUM ----------
        f, psd_tot = welch(
            E_tot,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=N_fft,
            return_onesided=False,
            scaling="density"
        )
        # welch(E) already returns a power spectral density estimate for the
        # complex field. Do not square it again here.
        psd_tot_watts_per_hz = np.maximum(np.real(psd_tot), 0.0) * h * nu_0
        spectrum_db = 10*np.log10(np.mean(psd_tot_watts_per_hz, axis=0)/1e-3 + 1e-20)

        cos_tail = avg_cos_pd[-cos_trace_len:]
        cos_phase_diff_time[k, -len(cos_tail):] = cos_tail

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
        if f_window.size < 3:
            nyquist_ghz = 0.5 / dt * 1e-9
            raise ValueError(
                f"Frequency window [{f_plot_min}, {f_plot_max}] GHz has only "
                f"{f_window.size} bins at dt={dt/tau_p:g} tau_p "
                f"(Nyquist={nyquist_ghz:.3g} GHz)."
            )

        if k == 0:
            n_freqs = len(f_window)
            spectrum_db_norm_list = np.zeros((resolution, n_freqs), dtype=dtype)
            spectrum_psd_dbmhz_list = np.full((resolution, n_freqs), np.nan, dtype=dtype)

        spectrum_psd_dbmhz_list[k] = spectrum_db.astype(dtype, copy=False)
        white_noise_floor_trace_dbmhz[k] = estimate_white_noise_floor_dbmhz(
            spectrum_db,
            f_window / tau_p,
        )
        spectrum_db_norm_list[k] = spectrum_db - np.max(spectrum_db)

        for laser_idx in range(N_lasers):
            _, psd_laser = welch(
                E_all[:, laser_idx, :],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=N_fft,
                return_onesided=False,
                scaling="density",
            )
            psd_laser_watts_per_hz = np.maximum(np.real(psd_laser), 0.0) * h * nu_0
            spectrum_laser_db = 10*np.log10(np.mean(psd_laser_watts_per_hz, axis=0)/1e-3 + 1e-20)
            spectrum_laser_db = spectrum_laser_db[idx_sort][mask]
            laser_white_noise_floor_traces_dbmhz[laser_idx, k] = estimate_white_noise_floor_dbmhz(
                spectrum_laser_db,
                f_window / tau_p,
            )
            spectrum_laser_db_norm = spectrum_laser_db - np.max(spectrum_laser_db)
            laser_linewidth_traces_mhz[laser_idx, k], *_ = linewidth_from_current_mode(
                spectrum_laser_db_norm,
                f_window/tau_p,
                spectrum_psd_dbmhz=spectrum_laser_db,
                white_noise_floor_dbmhz=laser_white_noise_floor_traces_dbmhz[laser_idx, k],
            )
            laser_linewidth_traces_mhz[laser_idx, k] *= 1e-6

        linewidth_trace_mhz[k], *_ = linewidth_from_current_mode(
            spectrum_db_norm_list[k],
            f_window/tau_p,
            spectrum_psd_dbmhz=spectrum_db,
            white_noise_floor_dbmhz=white_noise_floor_trace_dbmhz[k],
        )
        linewidth_trace_mhz[k] *= 1e-6
        next_history = continuation_history_from_saved(y, nd, final_history=final_history)


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

        if save_spectrum_frames and (k % spectrum_frame_stride == 0 or k == len(kappa_c)-1):
            # clear_output(wait=True)
            fig = plt.figure(figsize=(18,10), dpi=300)
            width_ratios = [1]*30
            width_ratios[8] = 0.25
            gs = fig.add_gridspec(20, 30, height_ratios=[1]*20, width_ratios=width_ratios, hspace=0.3)

            # --- Cosine of Phase Difference (top row, spans all columns) ---
            ax0 = fig.add_subplot(gs[1:8, 0:-3])

            cos_time_arr = t[-cos_trace_len:]
            time_window = np.arange(cos_trace_len)
            local_time_ns = (cos_time_arr[time_window] - cos_time_arr[time_window[0]]) * 1e9
            im0 = ax0.imshow(
                cos_phase_diff_time[:, time_window],
                aspect='auto',
                extent=[
                    local_time_ns[0],
                    local_time_ns[-1],
                    kappa_c[0]*1e-9,
                    kappa_c[-1]*1e-9,
                ],
                origin='lower',
                cmap='jet',
                vmin=-1, vmax=1, rasterized=True
            )
            cbar0 = fig.colorbar(im0, ax=ax0, pad=0.02)
            cbar0.set_label(r'$\cos(\Delta\phi)$', fontsize=font_size, labelpad=0 )
            cbar0.ax.tick_params(labelsize=font_size)
            ax0.set_ylabel(r'$\kappa_c~(\mathrm{ns}^{-1})$', fontsize=font_size)
            ax0.set_xlabel(r'Time (ns)', fontsize=font_size)
            ax0.set_xlim(local_time_ns[0], local_time_ns[-1])
            ax0.set_xticks(np.linspace(local_time_ns[0], local_time_ns[-1], 6))
            ax0.set_title(
                r'$T_\mathrm{{max}}={:.2f}\,\mu\mathrm{{s}},\ \phi_p={:+.2f}\pi,\ \delta_i=[{}]\,\mathrm{{GHz}}$'.format(
                    Tmax*1e6,
                    fixed_phi_p/np.pi,
                    ", ".join(f"{d:.1f}" for d in delta_distribution_ghz),
                ),
                fontsize=font_size,
                pad=16,
            )
            # ax0.set_title(r'$\phi_p\in[0,2\pi]$', fontsize=font_size, pad=16)
            ax0.set_yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6))
            ax0.tick_params(axis='both', labelsize=font_size)

            # --- Order Parameter Plot (to the right of ax0) ---
            ax_order = fig.add_subplot(gs[1:8, -2:])
            ax_order.plot(order_param, kappa_c * 1e-9, color='black', linewidth=2)
            ax_order.set_title('Order Parameter', fontsize=font_size, pad=16)
            ax_order.set_ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
            ax_order.tick_params(axis='both', labelsize=font_size)
            ax_order.set_yticks([])  # Remove y axis ticks
            ax_order.set_xticks(np.linspace(0, 1, 2))
            ax_order.set_xlim(0, 1)
            
            # --- Optical Spectrum (Total) ---
            ax1 = fig.add_subplot(gs[11:, 1:8])
            if plot_psd_in_spectrum_panel and spectrum_psd_dbmhz_list is not None:
                spectrum_panel = spectrum_psd_dbmhz_list
                spectrum_vmin, spectrum_vmax = psd_color_limits(spectrum_panel)
                spectrum_colorbar_label = r"PSD (dBm/Hz)"
                spectrum_title = r"PSD$\left(E_{tot}\right)$"
            else:
                spectrum_panel = spectrum_db_norm_list
                spectrum_vmin, spectrum_vmax = -100.0, 0.0
                spectrum_colorbar_label = r"Relative PSD (dB)"
                spectrum_title = r"PSD$\left(E_{tot}\right)$, peak-normalized"
            im = ax1.imshow(
                spectrum_panel,
                aspect='auto',
                extent=[f_plot_axis_min, f_plot_axis_max, kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
                origin='lower',
                cmap='jet', rasterized=True
            )
            cbar1 = fig.colorbar(im, ax=ax1, pad=0.05)
            cbar1.set_label(spectrum_colorbar_label, fontsize=font_size, labelpad=12)
            cbar1.ax.tick_params(labelsize=font_size)
            im.set_clim(spectrum_vmin, spectrum_vmax)
            ax1.set_xlabel("Frequency (GHz)", fontsize=font_size, labelpad=10)
            ax1.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=10)
            ax1.set_title(spectrum_title, fontsize=font_size, pad=16)
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

            # --- Linewidth trace for this Tmax run ---
            ax2 = fig.add_subplot(gs[11:, 13:29])
            field_colors = plt.get_cmap("tab10")
            for laser_idx in range(N_lasers):
                laser_linewidth = laser_linewidth_traces_mhz[laser_idx, :k + 1]
                valid_laser_linewidth = np.isfinite(laser_linewidth) & (laser_linewidth > 0)
                if np.any(valid_laser_linewidth):
                    ax2.plot(
                        kappa_c[:k + 1][valid_laser_linewidth] * 1e-9,
                        laser_linewidth[valid_laser_linewidth],
                        color=field_colors(laser_idx % field_colors.N),
                        linewidth=1.4,
                        alpha=0.55,
                        zorder=1,
                        label=rf"$E_{laser_idx + 1}$",
                    )
            valid_linewidth = np.isfinite(linewidth_trace_mhz[:k + 1]) & (linewidth_trace_mhz[:k + 1] > 0)
            ax2.plot(
                kappa_c[:k + 1][valid_linewidth] * 1e-9,
                linewidth_trace_mhz[:k + 1][valid_linewidth],
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
                f"{linewidth_quantity_label()} (MHz)",
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
            filename = f"{frame_dir}/tmax_{tmax_idx:03d}_{scheme_label}_{dt_label}_{interp_label}_{substep_label}.png"
            
            # phi_p{phi_p[0,0]/np.pi:.2f}pi_continuation_noise_detuning{detuning}_alpha{alpha}_noise_{n_cases}_{n_iterations}avg_self{self_feedback:.2f}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            plt.cla(); plt.clf()
            plt.close('all')
            del fig, gs, ax0, ax_order, ax1, ax2
            del im0, im, cbar0, cbar1
            clear_output(wait=True)

        history = next_history
        del y, y_scaled, final_history, E_all, E_tot
        del avg_cos_pd, psd_tot, psd_tot_watts_per_hz, spectrum_db
        del psd_laser, psd_laser_watts_per_hz, spectrum_laser_db, spectrum_laser_db_norm
        if k % 5 == 0 or k == len(kappa_c) - 1:
            gc.collect()

    if use_two_stage_tmax_spectrum:
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
        if f_window is None or spectrum_db_norm_list is None:
            raise RuntimeError("Two-stage Tmax spectral run completed without returning any spectra.")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    np.save(f"{save_dir}/spectrum_db_norm_list_{run_label}.npy", spectrum_db_norm_list)
    if spectrum_psd_dbmhz_list is not None:
        np.save(f"{save_dir}/spectrum_psd_dbmhz_list_{run_label}.npy", spectrum_psd_dbmhz_list)
    np.save(f"{save_dir}/f_window_{run_label}.npy", f_window)
    np.save(f"{save_dir}/kappa_c_{run_label}.npy", kappa_c)
    np.save(f"{save_dir}/detuning_vals_ghz_{run_label}.npy", detuning_vals_ghz)
    np.save(f"{save_dir}/phi_p_{run_label}.npy", np.array([fixed_phi_p]))
    np.save(f"{save_dir}/Tmax_{run_label}.npy", np.array([Tmax_for_linewidth]))
    np.save(f"{save_dir}/linewidth_E_fields_mhz_{run_label}.npy", laser_linewidth_traces_mhz)
    np.save(f"{save_dir}/linewidth_Etot_mhz_{run_label}.npy", linewidth_trace_mhz)
    np.save(f"{save_dir}/white_noise_floor_E_fields_dbmhz_{run_label}.npy", laser_white_noise_floor_traces_dbmhz)
    np.save(f"{save_dir}/white_noise_floor_Etot_dbmhz_{run_label}.npy", white_noise_floor_trace_dbmhz)
    np.savez(
        settings_path,
        dt=dt,
        dt_multiplier=dt / tau_p,
        requested_Tmax=Tmax_spectrum_requested,
        effective_Tmax=Tmax_for_linewidth,
        continuation_Tmax=Tmax,
        use_two_stage_tmax_spectrum=bool(use_two_stage_tmax_spectrum),
        two_stage_kappa_chunk_size=int(max(1, two_stage_kappa_chunk_size)),
        long_spectrum_jobs=int(max(1, long_spectrum_jobs)),
        vectorize_two_stage_kappa_chunks=bool(vectorize_two_stage_kappa_chunks),
        save_after_each_long_integration=bool(save_after_each_long_integration),
        steps=steps,
        spectrum_steps=spectrum_steps,
        delay_steps=delay_steps,
        integration_scheme=integration_scheme,
        trajectory_save_every=int(max(1, trajectory_save_every)),
        spectrum_trajectory_save_every=int(max(1, spectrum_trajectory_save_every)),
        trajectory_output_dtype=str(np.dtype(trajectory_output_dtype)),
        linewidth_estimate_mode=str(linewidth_estimate_mode),
        linewidth_level_db=float(linewidth_level_db),
        linewidth_smooth_sigma_bins=float(linewidth_smooth_sigma_bins),
        spectrum_frequency_window_ghz=float(spectrum_frequency_window_ghz),
        spectrum_output_field_only=bool(spectrum_output_field_only),
        spectrum_max_output_gb=np.nan if spectrum_max_output_gb is None else float(spectrum_max_output_gb),
        spectrum_max_saved_freq_points=int(max(3, spectrum_max_saved_freq_points)),
        plot_psd_in_spectrum_panel=bool(plot_psd_in_spectrum_panel),
        psd_plot_dynamic_range_db=float(psd_plot_dynamic_range_db),
        estimate_white_noise_floor=bool(estimate_white_noise_floor),
        white_noise_floor_edge_fraction=float(white_noise_floor_edge_fraction),
        white_noise_floor_percentile=float(white_noise_floor_percentile),
        white_noise_floor_min_bins=int(white_noise_floor_min_bins),
        delay_interpolation="" if delay_interpolation is None else delay_interpolation,
        integrator_theta=integrator_theta,
        integrator_max_iter=integrator_max_iter,
        noise_substeps=noise_substeps,
        max_noise_substep_dt_tau_p=max_noise_substep_dt_tau_p,
    )
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return {
        "tmax_idx": tmax_idx,
        "run_label": run_label,
        "phi_p": fixed_phi_p,
        "Tmax": Tmax_for_linewidth,
        "detuning_label": detuning_label,
        "self_feedback": self_feedback,
        "N_lasers": N_lasers,
    }








if run_tmax_sweep:
    if n_tmax_jobs == 1:
        if use_two_stage_tmax_spectrum:
            tmax_sweep_results = [
                run_tmax_continuation(tmax_idx)
                for tmax_idx in range(len(Tmax_values))
            ]
        else:
            tmax_sweep_results = [
                run_tmax_continuation(tmax_idx)
                for tmax_idx in tqdm(range(len(Tmax_values)), desc="Tmax sweep", unit="Tmax")
            ]
    else:
        # Recycle each worker after one Tmax run so large arrays are released
        # instead of accumulating inside long-lived worker processes.
        mp_context = mp.get_context("fork")
        with mp_context.Pool(processes=n_tmax_jobs, maxtasksperchild=1) as pool:
            tmax_sweep_results = list(tqdm(
                pool.imap_unordered(run_tmax_continuation, range(len(Tmax_values)), chunksize=1),
                total=len(Tmax_values),
                desc="Tmax sweep",
                unit="Tmax",
            ))

    first_result = sorted(tmax_sweep_results, key=lambda item: item["tmax_idx"])[0]
    N_lasers = first_result["N_lasers"]
    save_dir = linewidth_output_dirs(N_lasers)["arrays"]
    run_label = first_result["run_label"]
    detuning_label = first_result["detuning_label"]
    self_feedback = first_result["self_feedback"]
    spectrum_db_norm_list = np.load(f"{save_dir}/spectrum_db_norm_list_{run_label}.npy")
    spectrum_psd_dbmhz_list = None
    spectrum_psd_path = f"{save_dir}/spectrum_psd_dbmhz_list_{run_label}.npy"
    if os.path.exists(spectrum_psd_path):
        spectrum_psd_dbmhz_list = np.load(spectrum_psd_path)
    f_window = np.load(f"{save_dir}/f_window_{run_label}.npy")
    kappa_c = np.load(f"{save_dir}/kappa_c_{run_label}.npy")
    detuning_vals_ghz = np.load(f"{save_dir}/detuning_vals_ghz_{run_label}.npy")


#%%


#%%
# --- Linewidth convergence vs Tmax for fixed phi_p ---
linewidth_level_db = 3.0
linewidth_smooth_sigma_bins = 1.0


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

    freq_left = np.interp(target_level, [spectrum_db[li], spectrum_db[li + 1]], [shifted_freq[li], shifted_freq[li + 1]])
    freq_right = np.interp(target_level, [spectrum_db[ri - 1], spectrum_db[ri]], [shifted_freq[ri - 1], shifted_freq[ri]])
    linewidth = freq_right - freq_left
    return linewidth, freq_left, freq_right, target_level, 0.0, shifted_freq


import glob
import os
import re


N_lasers = int(globals().get("N_lasers", 2))
output_dirs = linewidth_output_dirs(N_lasers)
map_save_dir = output_dirs["base"]
array_dir = output_dirs["arrays"]
os.makedirs(map_save_dir, exist_ok=True)

detuning_label = globals().get("detuning_label", "4.00")
self_feedback = globals().get("self_feedback", 0.0)
tau_p = globals().get("tau_p", 5.4e-12)
linewidth_plot_ylim_mhz = globals().get("linewidth_plot_ylim_mhz", (0.1, 300.0))
linewidth_plot_yticks_mhz = globals().get("linewidth_plot_yticks_mhz", [0.1, 1.0, 10.0, 100.0])

run_records = sorted(tmax_sweep_results, key=lambda item: item["Tmax"]) if "tmax_sweep_results" in globals() else []
if len(run_records) == 0:
    scheme_pattern = ""
    if globals().get("load_only_current_dt", True) and "integration_scheme" in globals():
        scheme_pattern = f"_scheme{str(globals()['integration_scheme']).lower()}"
    dt_pattern = ""
    if globals().get("load_only_current_dt", True) and "dt_multiplier" in globals():
        dt_pattern = f"_dtmult{filename_float(globals()['dt_multiplier'])}"
    save_pattern = ""
    if globals().get("load_only_current_dt", True) and "trajectory_save_every" in globals():
        save_pattern = f"_saveevery{int(max(1, globals()['trajectory_save_every']))}"
    pattern = (
        f"{array_dir}/spectrum_db_norm_list_"
        f"self{self_feedback:.2f}_detuning{detuning_label}_phipi*_"
        f"tmaxidx*_tmaxus*{scheme_pattern}{dt_pattern}{save_pattern}_*.npy"
    )
    spectrum_files = sorted(glob.glob(pattern))
    if not spectrum_files:
        raise FileNotFoundError(
            f"No Tmax sweep spectra found for pattern:\n{pattern}\n"
            "Run the Tmax sweep cell first so linewidth map arrays exist."
        )

    run_records = []
    for spectrum_path in spectrum_files:
        run_label = os.path.basename(spectrum_path).removeprefix(
            "spectrum_db_norm_list_"
        ).removesuffix(".npy")
        tmax_path = f"{array_dir}/Tmax_{run_label}.npy"
        if os.path.exists(tmax_path):
            Tmax = float(np.load(tmax_path)[0])
        else:
            match = re.search(r"_tmaxus([-+0-9.]+)", run_label)
            if not match:
                raise ValueError(f"Could not parse Tmax from {os.path.basename(spectrum_path)}")
            Tmax = float(match.group(1)) * 1e-6

        run_records.append({
            "run_label": run_label,
            "Tmax": Tmax,
        })

    run_records = sorted(run_records, key=lambda item: item["Tmax"])

linewidth_traces_mhz = []
field_linewidth_traces_mhz = []
field_linewidths_available = True
Tmax_loaded_us = []
kappa_ref = None
for record in run_records:
    run_label = record["run_label"]
    spectrum = np.load(f"{array_dir}/spectrum_db_norm_list_{run_label}.npy")
    spectrum_psd_path = f"{array_dir}/spectrum_psd_dbmhz_list_{run_label}.npy"
    spectrum_psd = np.load(spectrum_psd_path) if os.path.exists(spectrum_psd_path) else None
    white_floor_path = f"{array_dir}/white_noise_floor_Etot_dbmhz_{run_label}.npy"
    white_floor = np.load(white_floor_path) if os.path.exists(white_floor_path) else None
    freqs = np.load(f"{array_dir}/f_window_{run_label}.npy") / tau_p
    kappa = np.load(f"{array_dir}/kappa_c_{run_label}.npy")
    if kappa_ref is None:
        kappa_ref = kappa
    elif not np.allclose(kappa_ref, kappa):
        raise ValueError("Saved Tmax runs do not share the same kappa grid.")

    linewidth_etot_path = f"{array_dir}/linewidth_Etot_mhz_{run_label}.npy"
    if os.path.exists(linewidth_etot_path):
        linewidth_trace_mhz = np.load(linewidth_etot_path)
    else:
        linewidth_trace = np.full(spectrum.shape[0], np.nan)
        for k in range(spectrum.shape[0]):
            spectrum_psd_k = spectrum_psd[k] if spectrum_psd is not None else None
            white_floor_k = white_floor[k] if white_floor is not None else np.nan
            linewidth_trace[k], *_ = linewidth_from_current_mode(
                spectrum[k],
                freqs,
                spectrum_psd_dbmhz=spectrum_psd_k,
                white_noise_floor_dbmhz=white_floor_k,
            )
        linewidth_trace_mhz = linewidth_trace * 1e-6

    linewidth_traces_mhz.append(linewidth_trace_mhz)

    linewidth_fields_path = f"{array_dir}/linewidth_E_fields_mhz_{run_label}.npy"
    if os.path.exists(linewidth_fields_path):
        field_linewidth_traces_mhz.append(np.load(linewidth_fields_path))
    else:
        field_linewidths_available = False
    Tmax_loaded_us.append(record["Tmax"] * 1e6)

linewidth_traces_mhz = np.asarray(linewidth_traces_mhz)
if field_linewidths_available and len(field_linewidth_traces_mhz) == len(linewidth_traces_mhz):
    field_linewidth_traces_mhz = np.asarray(field_linewidth_traces_mhz)
else:
    field_linewidth_traces_mhz = None
Tmax_loaded_us = np.asarray(Tmax_loaded_us)
kappa_ns = kappa_ref * 1e-9
linewidth_map_mhz = linewidth_traces_mhz.T

np.save(f"{array_dir}/Tmax_values_us_linewidth_convergence.npy", Tmax_loaded_us)
np.save(f"{array_dir}/linewidth_traces_mhz_vs_Tmax.npy", linewidth_traces_mhz)
np.save(f"{array_dir}/linewidth_map_Etot_Tmax_kappa_mhz.npy", linewidth_map_mhz)
np.save(f"{array_dir}/linewidth_map_Tmax_values_us.npy", Tmax_loaded_us)
np.save(f"{array_dir}/linewidth_map_kappa_c.npy", kappa_ref)
np.save(f"{array_dir}/linewidth_smooth_sigma_bins.npy", np.array([linewidth_smooth_sigma_bins]))
if field_linewidth_traces_mhz is not None:
    np.save(f"{array_dir}/linewidth_traces_E_fields_mhz_vs_Tmax.npy", field_linewidth_traces_mhz)

def plot_linewidth_curves_vs_tmax(trace_matrix_mhz, field_label, filename_label):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    norm = matplotlib.colors.Normalize(vmin=Tmax_loaded_us[0], vmax=Tmax_loaded_us[-1])
    cmap = plt.get_cmap("viridis")
    resolution_label_added = False
    for idx, Tmax_us in enumerate(Tmax_loaded_us):
        ax.plot(
            kappa_ns,
            trace_matrix_mhz[idx],
            color=cmap(norm(Tmax_us)),
            linewidth=1.8,
            alpha=0.9,
        )
        resolution_mhz = 1.44 / Tmax_us
        if linewidth_plot_ylim_mhz[0] <= resolution_mhz <= linewidth_plot_ylim_mhz[1]:
            ax.axhline(
                resolution_mhz,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.25,
                label=r"$1.44/T_\mathrm{max}$" if not resolution_label_added else None,
            )
            resolution_label_added = True

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.04)
    cbar.set_label(r"$T_\mathrm{max}$ ($\mu$s)", fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    ax.set_xlabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=24)
    ax.set_ylabel(f"{linewidth_quantity_label()} (MHz)", fontsize=24)
    ax.set_title(
        rf"${field_label}$ linewidth vs coupling, $\sigma_f={linewidth_smooth_sigma_bins:.1f}$ bins",
        fontsize=26,
        pad=12,
    )
    ax.set_yscale("log")
    ax.set_ylim(*linewidth_plot_ylim_mhz)
    ax.set_yticks(linewidth_plot_yticks_mhz)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xticks(np.linspace(kappa_ns[0], kappa_ns[-1], 6))
    plt.tight_layout()
    plt.savefig(
        f"{map_save_dir}/linewidth_curves_{filename_label}_vs_kappa_Tmax_smooth{linewidth_smooth_sigma_bins:.1f}bins.png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


plot_linewidth_curves_vs_tmax(linewidth_traces_mhz, "E_{tot}", "Etot")
if field_linewidth_traces_mhz is not None:
    for laser_idx in range(field_linewidth_traces_mhz.shape[1]):
        plot_linewidth_curves_vs_tmax(
            field_linewidth_traces_mhz[:, laser_idx, :],
            f"E_{laser_idx + 1}",
            f"E{laser_idx + 1}",
        )


#%%
# --- Interpolated 3D linewidth surface over kappa_c and Tmax ---
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata

linewidth_floor_mhz, linewidth_ceiling_mhz = linewidth_plot_ylim_mhz
linewidth_ticks_log = np.log10(linewidth_plot_yticks_mhz)
surface_kappa_points = 300
surface_tmax_points = 220
surface_interp_method = "linear"  # "linear" is stable; "cubic" is smoother but can overshoot.
surface_kappa_ns = np.linspace(kappa_ns[0], kappa_ns[-1], surface_kappa_points)
surface_tmax_us = np.linspace(Tmax_loaded_us[0], Tmax_loaded_us[-1], surface_tmax_points)
KAPPA_SURF, TMAX_SURF = np.meshgrid(surface_kappa_ns, surface_tmax_us)
KAPPA_DATA, TMAX_DATA = np.meshgrid(kappa_ns, Tmax_loaded_us)


def interpolated_linewidth_surface_log(trace_matrix_mhz):
    linewidth_surface_input = np.clip(
        trace_matrix_mhz,
        linewidth_floor_mhz,
        linewidth_ceiling_mhz,
    )
    valid_surface = np.isfinite(linewidth_surface_input) & (linewidth_surface_input > 0)
    linewidth_surface_log = griddata(
        (KAPPA_DATA[valid_surface], TMAX_DATA[valid_surface]),
        np.log10(linewidth_surface_input[valid_surface]),
        (KAPPA_SURF, TMAX_SURF),
        method=surface_interp_method,
    )
    if np.any(~np.isfinite(linewidth_surface_log)):
        linewidth_surface_nearest = griddata(
            (KAPPA_DATA[valid_surface], TMAX_DATA[valid_surface]),
            np.log10(linewidth_surface_input[valid_surface]),
            (KAPPA_SURF, TMAX_SURF),
            method="nearest",
        )
        linewidth_surface_log = np.where(
            np.isfinite(linewidth_surface_log),
            linewidth_surface_log,
            linewidth_surface_nearest,
        )
    return linewidth_surface_log


def plot_linewidth_surface(trace_matrix_mhz, field_label, filename_label):
    linewidth_surface_log = interpolated_linewidth_surface_log(trace_matrix_mhz)
    fig = plt.figure(figsize=(11, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")
    norm = matplotlib.colors.Normalize(vmin=Tmax_loaded_us[0], vmax=Tmax_loaded_us[-1])
    cmap = plt.get_cmap("viridis")
    ax.plot_surface(
        KAPPA_SURF,
        TMAX_SURF,
        linewidth_surface_log,
        facecolors=cmap(norm(TMAX_SURF)),
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        shade=False,
    )
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(r"$T_\mathrm{max}$ ($\mu$s)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=16, labelpad=10)
    ax.set_ylabel(r"$T_\mathrm{max}$ ($\mu$s)", fontsize=16, labelpad=10)
    ax.set_zlabel(f"{linewidth_quantity_label()} (MHz)", fontsize=16, labelpad=10)
    ax.set_title(
        rf"Interpolated ${field_label}$ linewidth surface, $\sigma_f={linewidth_smooth_sigma_bins:.1f}$ bins",
        fontsize=20,
        pad=16,
    )
    kappa_buffer = 0.03 * (kappa_ns[-1] - kappa_ns[0])
    tmax_buffer = 0.03 * (Tmax_loaded_us[-1] - Tmax_loaded_us[0])
    linewidth_log_min = np.log10(linewidth_floor_mhz)
    linewidth_log_max = np.log10(linewidth_ceiling_mhz)
    linewidth_log_buffer = 0.06 * (linewidth_log_max - linewidth_log_min)
    ax.set_xlim(kappa_ns[0] - kappa_buffer, kappa_ns[-1] + kappa_buffer)
    ax.set_ylim(Tmax_loaded_us[0] - tmax_buffer, Tmax_loaded_us[-1] + tmax_buffer)
    ax.set_zlim(linewidth_log_min - linewidth_log_buffer, linewidth_log_max + linewidth_log_buffer)
    ax.set_xticks(np.linspace(kappa_ns[0], kappa_ns[-1], 6))
    ax.set_yticks(np.linspace(Tmax_loaded_us[0], Tmax_loaded_us[-1], 5))
    ax.set_zticks(linewidth_ticks_log)
    ax.set_zticklabels([f"{tick:g}" for tick in linewidth_plot_yticks_mhz])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=6, azim=45)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{map_save_dir}/linewidth_surface_3d_{filename_label}_vs_kappa_Tmax_smooth{linewidth_smooth_sigma_bins:.1f}bins.png",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.show()
    plt.close(fig)


plot_linewidth_surface(linewidth_traces_mhz, "E_{tot}", "Etot")
if field_linewidth_traces_mhz is not None:
    for laser_idx in range(field_linewidth_traces_mhz.shape[1]):
        plot_linewidth_surface(
            field_linewidth_traces_mhz[:, laser_idx, :],
            f"E_{laser_idx + 1}",
            f"E{laser_idx + 1}",
        )

#%%
# --- Fit linewidth convergence averaged over a kappa window ---
# Model:
#   Delta nu(Tmax) = Delta nu_infinity + a / Tmax^p
# with Delta nu and Delta nu_infinity in MHz and Tmax in microseconds. Since
# Tmax is measured in microseconds here, a has units MHz * us^p.
from scipy.optimize import curve_fit

fit_kappa_index = -1  # Used only when fit_average_last_n_kappa is None.
fit_average_last_n_kappa = 20  # Set to None to fit one kappa index.
fit_min_Tmax_us = 1  # e.g. 3.0 to ignore shorter runs; None uses all.
fit_include_hann_floor = True


def linewidth_power_law_model(Tmax_us, linewidth_inf_mhz, fit_a, fit_p):
    return linewidth_inf_mhz + fit_a / np.power(Tmax_us, fit_p)


def fit_linewidth_vs_inverse_tmax(Tmax_us, linewidth_mhz):
    Tmax_us = np.asarray(Tmax_us, dtype=float)
    linewidth_mhz = np.asarray(linewidth_mhz, dtype=float)
    valid = np.isfinite(Tmax_us) & np.isfinite(linewidth_mhz) & (Tmax_us > 0) & (linewidth_mhz > 0)
    if fit_min_Tmax_us is not None:
        valid &= Tmax_us >= float(fit_min_Tmax_us)
    if np.count_nonzero(valid) < 3:
        return np.nan, np.nan, np.nan, np.full_like(Tmax_us, np.nan), valid

    x = Tmax_us[valid]
    y = linewidth_mhz[valid]
    linear_a, linear_inf = np.polyfit(1.0 / x, y, deg=1)
    p0 = [
        max(0.0, float(linear_inf)),
        max(1e-12, float(linear_a)),
        1.0,
    ]
    bounds = (
        [0.0, 0.0, 0.05],
        [np.inf, np.inf, 6.0],
    )
    try:
        popt, _ = curve_fit(
            linewidth_power_law_model,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        linewidth_inf_mhz, fit_a, fit_p = popt
    except Exception:
        linewidth_inf_mhz, fit_a, fit_p = np.nan, np.nan, np.nan
    fit_all = linewidth_power_law_model(Tmax_us, linewidth_inf_mhz, fit_a, fit_p)
    return linewidth_inf_mhz, fit_a, fit_p, fit_all, valid


def plot_linewidth_inverse_tmax_fit(trace_matrix_mhz, field_label, filename_label):
    if fit_average_last_n_kappa is None:
        kappa_idx = int(fit_kappa_index)
        if kappa_idx < 0:
            kappa_idx = len(kappa_ns) + kappa_idx
        kappa_slice = slice(kappa_idx, kappa_idx + 1)
        kappa_label = rf"$\kappa_c={kappa_ns[kappa_idx]:.2f}\,\mathrm{{ns}}^{{-1}}$"
        filename_kappa_label = f"kappaidx{kappa_idx:03d}"
    else:
        n_fit_kappa = int(max(1, fit_average_last_n_kappa))
        kappa_start_idx = max(0, len(kappa_ns) - n_fit_kappa)
        kappa_slice = slice(kappa_start_idx, len(kappa_ns))
        kappa_label = (
            rf"mean over $\kappa_c=[{kappa_ns[kappa_start_idx]:.2f},"
            rf"{kappa_ns[-1]:.2f}]\,\mathrm{{ns}}^{{-1}}$"
        )
        filename_kappa_label = f"last{len(kappa_ns) - kappa_start_idx:03d}kappa"

    linewidth_at_kappa = np.nanmean(trace_matrix_mhz[:, kappa_slice], axis=1)
    linewidth_inf_mhz, fit_a, fit_p, linewidth_fit_mhz, valid_fit = fit_linewidth_vs_inverse_tmax(
        Tmax_loaded_us,
        linewidth_at_kappa,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    x_all = Tmax_loaded_us
    ax.scatter(
        x_all,
        linewidth_at_kappa,
        color="black",
        s=45,
        label="Measured",
        zorder=3,
    )
    if np.any(valid_fit):
        x_fit_dense = np.linspace(
            np.nanmin(x_all[valid_fit]),
            np.nanmax(x_all[valid_fit]),
            500,
        )
        y_fit_dense = linewidth_power_law_model(
            x_fit_dense,
            linewidth_inf_mhz,
            fit_a,
            fit_p,
        )
        ax.plot(
            x_fit_dense,
            y_fit_dense,
            color="tab:blue",
            linewidth=2.5,
            label=(
                rf"Fit: $\Delta\nu_\infty={linewidth_inf_mhz:.3g}$ MHz, "
                rf"$a={fit_a:.3g}$, $p={fit_p:.3g}$"
            ),
            zorder=4,
        )
    if fit_include_hann_floor:
        hann_floor_mhz = 1.44 / Tmax_loaded_us
        order = np.argsort(x_all)
        ax.plot(
            x_all[order],
            hann_floor_mhz[order],
            color="red",
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            label=r"$1.44/T_\mathrm{max}$",
            zorder=2,
        )

    ax.set_xlabel(r"$T_\mathrm{max}$ ($\mu$s)", fontsize=22)
    ax.set_ylabel(f"{linewidth_quantity_label()} (MHz)", fontsize=22)
    ax.set_title(
        rf"${field_label}$ linewidth convergence, {kappa_label}",
        fontsize=22,
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=14)
    plt.tight_layout()
    fit_path = (
        f"{map_save_dir}/linewidth_inverse_Tmax_fit_{filename_label}_"
        f"{filename_kappa_label}_smooth{linewidth_smooth_sigma_bins:.1f}bins.png"
    )
    plt.savefig(fit_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {
        "field": filename_label,
        "kappa_index": None if fit_average_last_n_kappa is not None else int(kappa_slice.start),
        "kappa_start_index": int(kappa_slice.start),
        "kappa_stop_index": int(kappa_slice.stop),
        "kappa_start_ns": float(kappa_ns[kappa_slice.start]),
        "kappa_stop_ns": float(kappa_ns[kappa_slice.stop - 1]),
        "fit_average_last_n_kappa": fit_average_last_n_kappa,
        "linewidth_infinity_mhz": linewidth_inf_mhz,
        "fit_a": fit_a,
        "fit_p": fit_p,
        "valid_fit_mask": valid_fit,
        "figure_path": fit_path,
    }


linewidth_fit_results = [
    plot_linewidth_inverse_tmax_fit(linewidth_traces_mhz, "E_{tot}", "Etot")
]
if field_linewidth_traces_mhz is not None:
    for laser_idx in range(field_linewidth_traces_mhz.shape[1]):
        linewidth_fit_results.append(
            plot_linewidth_inverse_tmax_fit(
                field_linewidth_traces_mhz[:, laser_idx, :],
                f"E_{laser_idx + 1}",
                f"E{laser_idx + 1}",
            )
        )

np.save(
    f"{array_dir}/linewidth_inverse_Tmax_fit_results.npy",
    np.array(linewidth_fit_results, dtype=object),
    allow_pickle=True,
)
