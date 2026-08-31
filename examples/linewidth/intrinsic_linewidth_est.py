#%%
"""Estimate intrinsic linewidth from the white FM-noise floor.

This script computes the single-sided frequency-noise PSD S_nu(f) from the
phase trajectory, estimates the white plateau of S_nu in a user-selected offset
frequency band, and reports

    Delta nu = pi * S_nu,white

in MHz. This is separate from the optical-spectrum FWHM estimator used in
linewidth_convergence_tmax_continuation.py.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import texmanager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.colors import Normalize
from scipy.signal import welch
from tqdm.auto import tqdm
from vcsel_lib import VCSEL
try:
    from examples._paths import LINEWIDTH_RESULTS_DIR
except ModuleNotFoundError:
    LINEWIDTH_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "linewidth_estimation"
try:
    from IPython.display import display
except Exception:
    display = None


# ----------------------------- User controls -----------------------------

run_simulation = True

N_lasers = 2
n_iterations = 50
Tmax = 1.0e-6
dt_multiplier = 0.5
trajectory_save_every = 1

kappa_c = np.linspace(0.0e9, 40.0e9, 100)
selected_kappa_index = -1

phi_p_value = 0.0
detuning_ghz = 4.0
coupling_scheme = "CUSTOM"

analysis_fraction = 0.5
welch_nperseg = 2**15
welch_overlap_fraction = 0.5
white_floor_band_hz = (100e9, 120e9)
white_floor_percentile = 50.0
white_floor_min_bins = 16

integration_scheme = "trapezoid"
integrator_theta = 0.5
integrator_max_iter = 5
delay_interpolation = "linear"  # None, "linear", or "cubic"
max_noise_substep_dt_tau_p = 1.0

save_results = True
output_dir = LINEWIDTH_RESULTS_DIR / f"{N_lasers}_lasers/intrinsic_linewidth"
show_progress_plot = True
save_progress_plot = True
progress_plot_stride = 1
cos_max_plot_points = 4000
psd_plot_dynamic_range_db = 80.0
psd_plot_xlim_hz = None  # None plots the full available spectrum up to Nyquist.
psd_curve_stride = 1
font_size = 22
linewidth_font_size = font_size + 4



# ------------------------------- Helpers ---------------------------------

def set_latex_plot_style(tex_cache_dir=None):
    """Apply TeX/Computer Modern font settings to match the other scripts."""
    if tex_cache_dir is not None:
        tex_cache_dir = os.path.abspath(tex_cache_dir)
        os.makedirs(tex_cache_dir, exist_ok=True)
        texmanager.TexManager._texcache = tex_cache_dir
    rc("text", usetex=True)
    rc("font", family="serif")
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


set_latex_plot_style()

def continuation_history_from_saved(y, nd, final_history=None):
    """Return the full 2*tau history required by the current integrator API."""
    if final_history is not None:
        delay_steps_nd = int(nd["delay_steps"])
        history_len = 2 * delay_steps_nd
        if final_history.shape[2] < history_len:
            raise ValueError(
                f"Returned final history has only {final_history.shape[2]} samples, "
                f"but continuation requires {history_len} samples."
            )
        return final_history[:, :, -history_len:].copy()

    save_every = int(max(1, nd.get("save_every", 1)))
    if save_every != 1:
        raise ValueError(
            "Continuation from saved output requires save_every=1 unless "
            "return_final_history=True is used."
        )

    delay_steps_nd = int(nd["delay_steps"])
    history_len = 2 * delay_steps_nd
    if y.shape[2] < history_len:
        raise ValueError(
            f"Saved trajectory has only {y.shape[2]} samples, but continuation "
            f"requires {history_len} samples of 2*tau history."
        )
    return y[:, :, -history_len:].copy()


def apply_integrator_controls(nd, noise_substeps):
    nd["integration_scheme"] = integration_scheme
    if delay_interpolation is not None:
        nd["delay_interp"] = delay_interpolation
    nd["noise_substeps"] = int(max(1, noise_substeps))
    return nd


def extract_s_phi_from_saved_state(y, n_lasers):
    """Return S and phi from either full [N,S,phi] or compact [S,phi] output."""
    if y.shape[1] == 3 * n_lasers:
        return np.maximum(y[:, 1::3, :], 0.0), y[:, 2::3, :]
    if y.shape[1] == 2 * n_lasers:
        return np.maximum(y[:, 0::2, :], 0.0), y[:, 1::2, :]
    raise ValueError(
        f"Expected {3 * n_lasers} full states or {2 * n_lasers} compact field states, "
        f"got {y.shape[1]} states."
    )


def average_cos_phase_trace(phi):
    """Average cos(phi_i - phi_0) over cases and non-reference lasers."""
    if phi.shape[1] <= 1:
        return np.ones(phi.shape[-1], dtype=np.float32)
    avg_trace = np.zeros(phi.shape[-1], dtype=np.float32)
    phi_ref = phi[:, 0, :]
    for laser_idx in range(1, phi.shape[1]):
        avg_trace += np.mean(np.cos(phi[:, laser_idx, :] - phi_ref), axis=0).astype(np.float32)
    avg_trace /= max(1, phi.shape[1] - 1)
    return avg_trace


def order_parameter_from_s_phi(S, phi):
    """Kuramoto-like field coherence order parameter from S and phi."""
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
    numerator = np.abs(E_sum) ** 2
    denom = S.shape[1] * denom_sum
    return np.mean(numerator / np.maximum(denom, eps), axis=1)


def downsample_time_trace(time, trace, max_points):
    """Downsample a time trace for plotting/storage."""
    max_points = int(max(2, max_points))
    if len(time) <= max_points:
        return time.astype(np.float32), trace.astype(np.float32)
    keep = np.unique(np.linspace(0, len(time) - 1, max_points, dtype=int))
    return time[keep].astype(np.float32), trace[keep].astype(np.float32)


def format_mhz_tick(value, _pos=None):
    """Plain-number formatter for log-scaled MHz linewidth ticks."""
    if not np.isfinite(value) or value <= 0:
        return ""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def make_physical_params(kappa_c_mat):
    alpha = 2
    tau_p = 5.4e-12
    tau_n = 0.25e-9
    g0 = 8.75e-4 * 1e9
    N0 = 2.86e5
    s = 4e-6
    q = 1.602e-19
    beta = 1.0e-3
    tau = 1e-9
    eta = 0.9
    current_threshold = 3
    I = eta * current_threshold * q / tau_n * (N0 + 1.0 / (g0 * tau_p))

    delta = detuning_ghz * 2.0 * np.pi * 1e9
    delta_distribution = np.sort(np.linspace(-delta / 2.0, delta / 2.0, N_lasers))
    phi_p_mat = np.full((N_lasers, N_lasers), phi_p_value, dtype=float)

    return {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "s": s,
        "q": q,
        "alpha": alpha,
        "beta": beta,
        "kappa_c": 0.0,
        "kappa_c_mat": kappa_c_mat,
        "tau": tau,
        "I": I,
        "noise_amplitude": 1.0,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "delta": delta_distribution,
        "phi_p": phi_p_value,
        "phi_p_mat": phi_p_mat,
        "Tmax": Tmax,
        "dt": dt_multiplier * tau_p,
        "save_every": int(max(1, trajectory_save_every)),
    }


def build_kappa_matrix(time_arr, kappa_initial, kappa_final):
    a_mat = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)
    return VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=float(kappa_initial),
        kappa_final=float(kappa_final),
        N_lasers=N_lasers,
        ramp_start=2.0,
        ramp_shape=1.0,
        tau=1e-9,
        scheme=coupling_scheme,
        plot=False,
        dx=1.0,
        aMAT=a_mat,
    ).astype(np.float32, copy=False)


def frequency_noise_from_phase(phase, t):
    """Return instantaneous frequency noise in Hz from phase in radians."""
    phase = np.unwrap(np.asarray(phase, dtype=float), axis=-1)
    dt = float(np.median(np.diff(t)))
    nu = np.diff(phase, axis=-1) / (2.0 * np.pi * dt)
    nu -= np.mean(nu, axis=-1, keepdims=True)
    return nu, 1.0 / dt


def estimate_white_floor(freq_offset_hz, psd_hz2_per_hz):
    """Estimate the white S_nu floor inside white_floor_band_hz."""
    f = np.asarray(freq_offset_hz, dtype=float)
    psd = np.asarray(psd_hz2_per_hz, dtype=float)
    finite = np.isfinite(f) & np.isfinite(psd) & (psd > 0)
    if np.count_nonzero(finite) < white_floor_min_bins:
        return np.nan, np.zeros_like(f, dtype=bool)

    lo, hi = white_floor_band_hz
    hi = min(float(hi), float(np.nanmax(f[finite])))
    band = finite & (f >= float(lo)) & (f <= hi)
    if np.count_nonzero(band) < white_floor_min_bins:
        cutoff = 0.8 * float(np.nanmax(f[finite]))
        band = finite & (f >= cutoff)
    if np.count_nonzero(band) < white_floor_min_bins:
        return np.nan, band

    return float(np.nanpercentile(psd[band], white_floor_percentile)), band


def fm_psd_and_intrinsic_linewidth(phase, t):
    """Compute S_nu(f), white floor, and pi*S_nu linewidth."""
    nu_noise, fs = frequency_noise_from_phase(phase, t)
    n_time = nu_noise.shape[-1]
    nperseg = min(int(welch_nperseg), n_time)
    if nperseg < 8:
        raise ValueError("Need at least 8 frequency samples for Welch PSD.")
    noverlap = int(np.floor(nperseg * float(welch_overlap_fraction)))
    noverlap = min(max(0, noverlap), nperseg - 1)
    f, psd = welch(
        nu_noise,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    psd_mean = np.mean(psd, axis=0)
    white_floor, floor_mask = estimate_white_floor(f, psd_mean)
    linewidth_hz = np.pi * white_floor if np.isfinite(white_floor) else np.nan
    return f, psd_mean, white_floor, linewidth_hz, floor_mask


def compute_intrinsic_metrics(y, t):
    """Return FM-noise spectra and intrinsic linewidths for E_tot and E_i."""
    S_full, phi_full = extract_s_phi_from_saved_state(y, N_lasers)
    cos_time_us, cos_trace = downsample_time_trace(
        t * 1e6,
        average_cos_phase_trace(phi_full),
        cos_max_plot_points,
    )
    start = int(np.floor((1.0 - analysis_fraction) * len(t)))
    start = min(max(0, start), len(t) - 8)
    t_analysis = t[start:]
    S = S_full[:, :, start:]
    phi = phi_full[:, :, start:]
    order_param = float(np.mean(order_parameter_from_s_phi(S, phi)))

    E_tot = np.zeros((S.shape[0], S.shape[-1]), dtype=np.complex64)
    for laser_idx in range(N_lasers):
        E_tot += (
            np.sqrt(S[:, laser_idx, :].astype(np.float32, copy=False))
            * np.exp((1j * phi[:, laser_idx, :]).astype(np.complex64))
        ).astype(np.complex64, copy=False)

    total_phase = np.angle(E_tot)
    f, psd_total, floor_total, linewidth_total_hz, floor_mask = fm_psd_and_intrinsic_linewidth(
        total_phase,
        t_analysis,
    )

    psd_fields = []
    floor_fields = np.full(N_lasers, np.nan, dtype=float)
    linewidth_fields_hz = np.full(N_lasers, np.nan, dtype=float)
    for laser_idx in range(N_lasers):
        f_i, psd_i, floor_i, linewidth_i_hz, _ = fm_psd_and_intrinsic_linewidth(
            phi[:, laser_idx, :],
            t_analysis,
        )
        if not np.allclose(f_i, f):
            raise ValueError("Field frequency-noise PSD grids do not match.")
        psd_fields.append(psd_i)
        floor_fields[laser_idx] = floor_i
        linewidth_fields_hz[laser_idx] = linewidth_i_hz

    return {
        "f": f,
        "psd_total": psd_total,
        "white_floor_total": floor_total,
        "linewidth_total_mhz": linewidth_total_hz * 1e-6,
        "psd_fields": np.asarray(psd_fields),
        "white_floor_fields": floor_fields,
        "linewidth_fields_mhz": linewidth_fields_hz * 1e-6,
        "floor_mask": floor_mask,
        "analysis_time_us": (t_analysis[0] * 1e6, t_analysis[-1] * 1e6),
        "cos_time_us": cos_time_us,
        "cos_trace": cos_trace,
        "order_param": np.float32(order_param),
    }


def make_intrinsic_linewidth_figure(
    result,
    linewidth_total,
    linewidth_fields,
    kappa_values,
    psd_map_db,
    f_psd_hz,
    cos_phase_diff_time,
    cos_time_us,
    order_param,
    k_completed=None,
):
    """Four-panel intrinsic linewidth progress figure."""
    set_latex_plot_style(tex_cache_dir=f"{output_dir}/matplotlib_tex_cache/pid{os.getpid()}")
    field_colors = plt.get_cmap("tab10")
    kappa_ns = kappa_values * 1e-9
    if k_completed is None:
        valid_count = len(kappa_values)
        current_idx = None
    else:
        valid_count = int(k_completed) + 1
        current_idx = int(k_completed)

    fig = plt.figure(figsize=(18, 10), dpi=250)
    width_ratios = [1] * 30
    width_ratios[8] = 0.25
    gs = fig.add_gridspec(20, 30, height_ratios=[1] * 20, width_ratios=width_ratios, hspace=0.35)

    # --- Cosine of phase difference ---
    ax_cos = fig.add_subplot(gs[1:8, 0:-3])
    im_cos = ax_cos.imshow(
        cos_phase_diff_time,
        aspect="auto",
        extent=[cos_time_us[0], cos_time_us[-1], kappa_ns[0], kappa_ns[-1]],
        origin="lower",
        cmap="jet",
        vmin=-1,
        vmax=1,
        rasterized=True,
    )
    cbar_cos = fig.colorbar(im_cos, ax=ax_cos, pad=0.02)
    cbar_cos.set_label(r"$\cos(\Delta\phi)$", fontsize=font_size, labelpad=0)
    cbar_cos.ax.tick_params(labelsize=font_size)
    ax_cos.set_ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size)
    ax_cos.set_xlabel(r"Time ($\mu$s)", fontsize=font_size)
    ax_cos.set_title(
        rf"$\phi_p={phi_p_value/np.pi:+.2f}\pi,\ "
        rf"\delta_i=[{-detuning_ghz/2:.1f},{detuning_ghz/2:.1f}]\,\mathrm{{GHz}},\ "
        rf"T_\mathrm{{max}}={Tmax*1e6:.2f}\,\mu\mathrm{{s}}$",
        fontsize=font_size,
        pad=16,
    )
    ax_cos.set_ylim(kappa_ns[0], kappa_ns[-1])
    ax_cos.set_yticks(np.linspace(kappa_ns[0], kappa_ns[-1], 6))
    ax_cos.tick_params(axis="both", labelsize=font_size)

    # --- Order parameter ---
    ax_order = fig.add_subplot(gs[1:8, -2:])
    ax_order.plot(order_param, kappa_ns, color="black", linewidth=2)
    ax_order.set_title("Order Parameter", fontsize=font_size, pad=16)
    ax_order.set_xlim(0, 1)
    ax_order.set_ylim(kappa_ns[0], kappa_ns[-1])
    ax_order.set_yticks([])
    ax_order.set_xticks([0, 1])
    ax_order.tick_params(axis="both", labelsize=font_size)

    # --- Frequency-noise PSD curves ---
    ax_psd = fig.add_subplot(gs[11:, 1:8])
    f_offset_ghz = f_psd_hz * 1e-9
    cmap_psd = plt.get_cmap("viridis")
    norm_psd = Normalize(vmin=kappa_ns[0], vmax=kappa_ns[-1])
    row_indices = np.arange(valid_count)
    row_indices = row_indices[::int(max(1, psd_curve_stride))]
    if len(row_indices) == 0 or row_indices[-1] != valid_count - 1:
        row_indices = np.append(row_indices, valid_count - 1)
    for row_idx in row_indices:
        psd_curve = 10.0 ** (psd_map_db[row_idx] / 10.0)
        valid_psd = np.isfinite(psd_curve) & (psd_curve > 0)
        if np.any(valid_psd):
            curve_zorder = 1 + int(row_idx)
            ax_psd.plot(
                f_offset_ghz[valid_psd],
                psd_curve[valid_psd],
                color=cmap_psd(norm_psd(kappa_ns[row_idx])),
                linewidth=1.2,
                alpha=0.9,
                zorder=curve_zorder,
            )
    sm_psd = plt.cm.ScalarMappable(norm=norm_psd, cmap=cmap_psd)
    sm_psd.set_array([])
    cbar_psd = fig.colorbar(sm_psd, ax=ax_psd, pad=0.05)
    cbar_psd.set_label(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=font_size, labelpad=12)
    cbar_psd.ax.tick_params(labelsize=font_size)
    lo_ghz, hi_ghz = np.asarray(white_floor_band_hz) * 1e-9
    ax_psd.axvspan(lo_ghz, hi_ghz, color="gray", alpha=0.12, linewidth=0)
    ax_psd.set_xlabel("Offset frequency (GHz)", fontsize=font_size, labelpad=10)
    ax_psd.set_ylabel(r"$S_\nu(f)~(\mathrm{Hz}^2/\mathrm{Hz})$", fontsize=font_size, labelpad=10)
    ax_psd.set_title(r"$S_\nu(f)$ for $E_{tot}$", fontsize=font_size, pad=16)
    if psd_plot_xlim_hz is None:
        ax_psd.set_xlim(f_offset_ghz[0], f_offset_ghz[-1])
    else:
        xlim_low_ghz, xlim_high_ghz = np.asarray(psd_plot_xlim_hz, dtype=float) * 1e-9
        ax_psd.set_xlim(max(f_offset_ghz[0], xlim_low_ghz), min(f_offset_ghz[-1], xlim_high_ghz))
    ax_psd.set_yscale("log")
    positive_psd = 10.0 ** (psd_map_db[:valid_count] / 10.0)
    positive_psd = positive_psd[np.isfinite(positive_psd) & (positive_psd > 0)]
    if positive_psd.size:
        ax_psd.set_ylim(0.5 * np.nanmin(positive_psd), 2.0 * np.nanmax(positive_psd))
    ax_psd.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_psd.tick_params(axis="both", labelsize=font_size)

    # --- Intrinsic linewidth trace ---
    ax_lw = fig.add_subplot(gs[11:, 13:29])
    valid_slice = slice(0, valid_count)

    for laser_idx in range(N_lasers):
        ax_lw.plot(
            kappa_ns[valid_slice],
            linewidth_fields[laser_idx, valid_slice],
            color=field_colors(laser_idx % field_colors.N),
            linewidth=1.4,
            alpha=0.65,
            label=rf"$E_{laser_idx + 1}$",
        )
    ax_lw.plot(
        kappa_ns[valid_slice],
        linewidth_total[valid_slice],
        color="black",
        linewidth=2.5,
        label=r"$E_{tot}$",
    )
    if current_idx is not None and np.isfinite(linewidth_total[current_idx]):
        ax_lw.scatter(
            kappa_ns[current_idx],
            linewidth_total[current_idx],
            color="red",
            s=40,
            zorder=5,
        )
    ax_lw.set_xlabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=linewidth_font_size, labelpad=10)
    ax_lw.set_ylabel(r"$\Delta\nu_\mathrm{intrinsic}$ (MHz)", fontsize=linewidth_font_size, labelpad=10)
    ax_lw.set_title(r"Intrinsic linewidth from $\pi S_{\nu,\mathrm{white}}$", fontsize=linewidth_font_size, pad=16)
    ax_lw.set_xlim(kappa_ns[0], kappa_ns[-1])
    ax_lw.set_yscale("log")
    finite_lw = linewidth_total[np.isfinite(linewidth_total) & (linewidth_total > 0)]
    if finite_lw.size:
        ax_lw.set_ylim(0.5 * np.nanmin(finite_lw), 2.0 * np.nanmax(finite_lw))
    ax_lw.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
    ax_lw.yaxis.set_major_formatter(FuncFormatter(format_mhz_tick))
    ax_lw.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_lw.yaxis.set_minor_formatter(NullFormatter())
    ax_lw.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_lw.tick_params(axis="both", labelsize=linewidth_font_size)
    ax_lw.legend(fontsize=linewidth_font_size - 10, loc="upper right")

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.93)
    return fig


# ------------------------------- Run sweep --------------------------------

if run_simulation:
    os.makedirs(output_dir, exist_ok=True)

    tau_p = 5.4e-12
    dt = dt_multiplier * tau_p
    saved_dt = dt * int(max(1, trajectory_save_every))
    saved_nyquist_hz = 0.5 / saved_dt
    if float(white_floor_band_hz[1]) >= saved_nyquist_hz:
        raise ValueError(
            "white_floor_band_hz extends beyond the saved phase Nyquist frequency. "
            f"Band upper edge = {white_floor_band_hz[1] * 1e-9:.2f} GHz, "
            f"saved Nyquist = {saved_nyquist_hz * 1e-9:.2f} GHz. "
            "Decrease trajectory_save_every or dt_multiplier."
        )
    steps = int(Tmax / dt)
    time_arr = np.arange(steps, dtype=float) * dt
    delay_steps = int(1e-9 / dt)
    if steps <= 2 * delay_steps + 4:
        raise ValueError("Tmax is too short for the 2*tau history at this dt.")

    kappa_initial_matrix = build_kappa_matrix(time_arr, 0.0, 0.0)
    phys = make_physical_params(kappa_initial_matrix)
    vcsel = VCSEL(phys)
    nd = apply_integrator_controls(
        vcsel.scale_params(),
        int(max(1, np.ceil(dt_multiplier / max_noise_substep_dt_tau_p))),
    )
    history, _, _, _ = vcsel.generate_history(nd, shape="FR", n_cases=n_iterations)

    linewidth_total_mhz = np.full(len(kappa_c), np.nan, dtype=float)
    linewidth_fields_mhz = np.full((N_lasers, len(kappa_c)), np.nan, dtype=float)
    white_floor_total = np.full(len(kappa_c), np.nan, dtype=float)
    white_floor_fields = np.full((N_lasers, len(kappa_c)), np.nan, dtype=float)
    order_param = np.full(len(kappa_c), np.nan, dtype=float)
    psd_map_db = None
    f_psd_hz = None
    cos_phase_diff_time = None
    cos_time_us = None
    selected_result = None
    progress_display_handle = None

    for k, kappa in enumerate(tqdm(kappa_c, desc="intrinsic linewidth", unit="kappa")):
        kappa_initial = kappa_c[k - 1] if k > 0 else 0.0
        kappa_matrix = build_kappa_matrix(time_arr, kappa_initial, kappa)
        phys = make_physical_params(kappa_matrix)
        vcsel = VCSEL(phys)
        nd = apply_integrator_controls(
            vcsel.scale_params(),
            int(max(1, np.ceil(dt_multiplier / max_noise_substep_dt_tau_p))),
        )
        nd["store_freqs"] = False
        nd["output_state_indices"] = np.ravel(
            np.column_stack((
                np.arange(N_lasers) * 3 + 1,
                np.arange(N_lasers) * 3 + 2,
            ))
        )

        t, y, _, final_history = vcsel.integrate(
            history,
            nd=nd,
            progress=False,
            theta=integrator_theta,
            max_iter=integrator_max_iter,
            smooth_freqs=False,
            integration_scheme=integration_scheme,
            return_final_history=True,
        )
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"Non-finite state at kappa={kappa*1e-9:.3f} ns^-1.")

        metrics = compute_intrinsic_metrics(y, t)
        linewidth_total_mhz[k] = metrics["linewidth_total_mhz"]
        linewidth_fields_mhz[:, k] = metrics["linewidth_fields_mhz"]
        white_floor_total[k] = metrics["white_floor_total"]
        white_floor_fields[:, k] = metrics["white_floor_fields"]
        order_param[k] = metrics["order_param"]
        if f_psd_hz is None:
            f_psd_hz = metrics["f"].astype(np.float32)
            psd_map_db = np.full((len(kappa_c), len(f_psd_hz)), np.nan, dtype=np.float32)
        if cos_time_us is None:
            cos_time_us = metrics["cos_time_us"].astype(np.float32)
            cos_phase_diff_time = np.full((len(kappa_c), len(cos_time_us)), np.nan, dtype=np.float32)
        if len(metrics["f"]) != len(f_psd_hz) or not np.allclose(metrics["f"], f_psd_hz):
            raise ValueError("Frequency-noise PSD grids do not match across kappa values.")
        if len(metrics["cos_time_us"]) != len(cos_time_us) or not np.allclose(metrics["cos_time_us"], cos_time_us):
            raise ValueError("Cosine phase-difference time grids do not match across kappa values.")
        psd_map_db[k] = (10.0 * np.log10(np.maximum(metrics["psd_total"], 1e-300))).astype(np.float32)
        cos_phase_diff_time[k] = metrics["cos_trace"].astype(np.float32)
        if k == selected_kappa_index % len(kappa_c):
            selected_result = dict(metrics)
            selected_result["kappa"] = kappa

        if (
            show_progress_plot
            and (k % int(max(1, progress_plot_stride)) == 0 or k == len(kappa_c) - 1)
        ):
            progress_result = dict(metrics)
            progress_result["kappa"] = kappa
            fig_progress = make_intrinsic_linewidth_figure(
                progress_result,
                linewidth_total_mhz,
                linewidth_fields_mhz,
                kappa_c,
                psd_map_db,
                f_psd_hz,
                cos_phase_diff_time,
                cos_time_us,
                order_param,
                k_completed=k,
            )
            if save_progress_plot or save_results:
                fig_progress.savefig(
                    f"{output_dir}/intrinsic_linewidth_progress.png",
                    bbox_inches="tight",
                )
            if display is not None:
                if progress_display_handle is None:
                    progress_display_handle = display(fig_progress, display_id=True)
                else:
                    progress_display_handle.update(fig_progress)
            plt.close(fig_progress)

        history = continuation_history_from_saved(y, nd, final_history=final_history)
        del y, final_history

    if selected_result is None:
        selected_result = dict(metrics)
        selected_result["kappa"] = kappa_c[-1]

    if save_results:
        np.save(f"{output_dir}/kappa_c.npy", kappa_c)
        np.save(f"{output_dir}/linewidth_intrinsic_Etot_mhz.npy", linewidth_total_mhz)
        np.save(f"{output_dir}/linewidth_intrinsic_E_fields_mhz.npy", linewidth_fields_mhz)
        np.save(f"{output_dir}/white_fm_floor_Etot_hz2_per_hz.npy", white_floor_total)
        np.save(f"{output_dir}/white_fm_floor_E_fields_hz2_per_hz.npy", white_floor_fields)
        np.save(f"{output_dir}/order_parameter.npy", order_param)
        np.save(f"{output_dir}/fm_psd_Etot_db_map.npy", psd_map_db)
        np.save(f"{output_dir}/fm_psd_frequency_hz.npy", f_psd_hz)
        np.save(f"{output_dir}/cos_phase_diff_time.npy", cos_phase_diff_time)
        np.save(f"{output_dir}/cos_time_us.npy", cos_time_us)
        np.savez(
            f"{output_dir}/intrinsic_linewidth_settings.npz",
            Tmax=Tmax,
            dt_multiplier=dt_multiplier,
            trajectory_save_every=trajectory_save_every,
            n_iterations=n_iterations,
            analysis_fraction=analysis_fraction,
            white_floor_band_hz=np.asarray(white_floor_band_hz, dtype=float),
            white_floor_percentile=white_floor_percentile,
            welch_nperseg=welch_nperseg,
        )


#%%
# --------------------------------- Plot -----------------------------------

if "selected_result" not in globals() or selected_result is None:
    raise RuntimeError("Run the simulation cell first.")

fig = make_intrinsic_linewidth_figure(
    selected_result,
    linewidth_total_mhz,
    linewidth_fields_mhz,
    kappa_c,
    psd_map_db,
    f_psd_hz,
    cos_phase_diff_time,
    cos_time_us,
    order_param,
)
if save_results:
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/intrinsic_linewidth_fm_noise.png", bbox_inches="tight")
plt.show()
