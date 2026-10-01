#%%
"""Compute the relative intensity noise spectrum for a two-laser VCSEL system."""

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


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import FuncFormatter, LogLocator
from scipy.constants import c, hbar
from scipy.signal import welch
from tqdm.auto import tqdm

from vcsel_lib import VCSEL
try:
    from examples._paths import LINEWIDTH_RESULTS_DIR
except ModuleNotFoundError:
    LINEWIDTH_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "linewidth_estimation"


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', titlesize=28, labelsize=28)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('legend', fontsize=18)
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


def continuation_history_from_saved(y, nd):
    """Return the full 2*tau history required by the continuation integrator."""
    save_every = int(nd.get("save_every", 1))
    if save_every != 1:
        raise ValueError("Kappa continuation requires save_every=1.")

    history_len = 2 * int(nd["delay_steps"])
    if y.shape[2] < history_len:
        raise ValueError(
            f"Saved trajectory has only {y.shape[2]} samples, but continuation "
            f"requires {history_len} samples of 2*tau history."
        )
    return y[:, :, -history_len:].copy()


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


def build_two_laser_system():
    """Return physical parameters for a representative two-laser simulation."""
    alpha = 2
    tau_p = 5.4e-12
    tau_n = 0.25e-9
    g0 = 8.75e-4 * 1e9
    N0 = 2.86e5
    s = 4e-6
    q = 1.602e-19
    beta = 1.e-3
    tau = 1e-9
    eta = 0.9
    current_threshold = 3
    lam = 910e-9
    omega0 = 2 * np.pi * c / lam
    I = eta * current_threshold * q / tau_n * (N0 + 1 / (g0 * tau_p))

    N_lasers = 2
    n_cases = 20
    average_continuation_history_across_noise = True
    dt = 1*tau_p
    Tmax = 1e-6
    steps = int(Tmax / dt)
    time_arr = np.arange(steps) * dt
 
    detuning_ghz = 4.0
    delta = detuning_ghz * 2 * np.pi * 1e9
    delta_distribution = np.linspace(-delta / 2, delta / 2, N_lasers)

    kappa_c = 40e9
    n_kappa = 200
    kappa_values = np.linspace(0.0, kappa_c, n_kappa)
    phi_p = 0.0
    ramp_start = 10
    ramp_shape = 100
    coupling_scheme = "CUSTOM"
    dx = 1.0
    aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)
    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=0.0,
        kappa_final=kappa_values[0],
        N_lasers=N_lasers,
        ramp_start=ramp_start,
        ramp_shape=ramp_shape,
        tau=tau,
        scheme=coupling_scheme,
        plot=False,
        dx=dx,
        aMAT=aMAT,
    )

    phys = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "N_bar": N0 + 1 / (g0 * tau_p),
        "s": s,
        "beta": beta,
        "kappa_c_mat": kappa_arr,
        "phi_p_mat": np.ones((n_cases, N_lasers, N_lasers)) * phi_p,
        "I": I,
        "q": q,
        "alpha": alpha,
        "delta": delta_distribution,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "noise_amplitude": 1.0,
        "dt": dt,
        "Tmax": Tmax,
        "tau": tau,
        "N_lasers": N_lasers,
        "save_every": 1,
        "max_output_gb": None,
    }
    return phys, {
        "n_cases": n_cases,
        "average_continuation_history_across_noise": average_continuation_history_across_noise,
        "Tmax": Tmax,
        "dt": dt,
        "kappa_c": kappa_c,
        "kappa_values": kappa_values,
        "time_arr": time_arr,
        "detuning_ghz": detuning_ghz,
        "phi_p": phi_p,
        "lam": lam,
        "omega0": omega0,
        "ramp_start": ramp_start,
        "ramp_shape": ramp_shape,
        "coupling_scheme": coupling_scheme,
        "dx": dx,
        "aMAT": aMAT,
    }


def build_kappa_segment(phys, meta, kappa_initial, kappa_final):
    """Build the coupling matrix for one continuation segment."""
    return VCSEL.build_coupling_matrix(
        time_arr=meta["time_arr"],
        kappa_initial=kappa_initial,
        kappa_final=kappa_final,
        N_lasers=phys["N_lasers"],
        ramp_start=meta["ramp_start"],
        ramp_shape=meta["ramp_shape"],
        tau=phys["tau"],
        scheme=meta["coupling_scheme"],
        plot=False,
        dx=meta["dx"],
        aMAT=meta["aMAT"],
    )


def complex_fields(y):
    """Return complex laser fields for each case, laser, and time sample."""
    S = np.maximum(y[:, 1::3, :], 0.0)
    phi = y[:, 2::3, :]
    return np.sqrt(S) * np.exp(1j * phi)


def total_field_intensity(y):
    """Return coherent total-field intensity |sum_i E_i|^2 for each case."""
    return np.abs(np.sum(complex_fields(y), axis=1)) ** 2


def incoherent_total_intensity(y):
    """Return summed laser powers sum_i |E_i|^2 without interference terms."""
    return np.sum(individual_field_intensity(y), axis=1)


def phase_aligned_total_field_intensity(y, discard_fraction=0.5):
    """Return coherent intensity after removing each case's mean phase offsets.

    This keeps the residual phase noise, but removes a static combining phase
    offset. It is a useful diagnostic for separating amplitude noise from
    phase-to-intensity conversion in the coherent combiner.
    """
    E_all = complex_fields(y)
    phi = y[:, 2::3, :]
    if phi.shape[1] <= 1:
        return np.abs(np.sum(E_all, axis=1)) ** 2

    start_idx = int(np.clip(discard_fraction, 0.0, 0.99) * phi.shape[-1])
    relative_phase = np.angle(
        np.exp(1j * (phi[:, :, start_idx:] - phi[:, 0:1, start_idx:]))
    )
    mean_relative_phase = np.angle(
        np.mean(np.exp(1j * relative_phase), axis=-1)
    )
    phase_aligned_fields = E_all * np.exp(-1j * mean_relative_phase[:, :, None])
    return np.abs(np.sum(phase_aligned_fields, axis=1)) ** 2


def individual_field_intensity(y):
    """Return individual field intensities |E_i|^2 for each case."""
    return np.maximum(y[:, 1::3, :], 0.0)


def intensity_to_mw_scale(phys, meta):
    """Match the simulated-intensity to mW conversion used in steering plots."""
    return 1e3 * hbar * meta["omega0"] / (phys["g0"] * phys["tau_n"] * phys["tau_p"])


def khz_mhz_ghz_tick_label(freq_hz, _pos=None):
    """Format RIN frequency ticks using kHz, MHz, and GHz."""
    if freq_hz <= 0:
        return ""
    if freq_hz < 1e6:
        value = freq_hz / 1e3
        unit = "kHz"
    elif freq_hz > 100e6:
        value = freq_hz / 1e9
        unit = "GHz"
    else:
        value = freq_hz / 1e6
        unit = "MHz"

    if np.isclose(value, round(value)):
        value_text = f"{value:.0f}"
    elif value < 10:
        value_text = f"{value:.1f}"
    else:
        value_text = f"{value:.0f}"
    return rf"${value_text}\,\mathrm{{{unit}}}$"


def compute_rin_from_intensity(intensity, nd, discard_fraction=0.5, desired_df_hz=1.0e6):
    """Return RIN spectra for intensity arrays with time on the last axis."""
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must be in the range [0, 1).")

    tau_p = float(nd["tau_p"])
    dt_physical = float(nd["dt"]) * tau_p
    fs = 1.0 / dt_physical

    start_idx = int(discard_fraction * intensity.shape[-1])
    intensity_segment = intensity[..., start_idx:]
    if intensity_segment.shape[-1] < 2:
        raise ValueError("Not enough samples remain after discarding the transient.")

    intensity_mean = np.mean(intensity_segment, axis=-1, keepdims=True)
    relative_fluctuation = intensity_segment / intensity_mean - 1.0
    relative_fluctuation -= np.mean(relative_fluctuation, axis=-1, keepdims=True)

    nperseg = min(relative_fluctuation.shape[-1], 2**15)
    noverlap = nperseg // 2
    nfft = max(int(np.ceil(fs / desired_df_hz)), nperseg)

    freq_hz, rin_cases = welch(
        relative_fluctuation,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
    )
    return freq_hz, np.mean(rin_cases, axis=0)


def compute_field_rin_spectra(y, nd, discard_fraction=0.5, desired_df_hz=1.0e6):
    """Return coherent, incoherent, phase-aligned, and individual RIN spectra.

    RIN is computed as the PSD of relative intensity fluctuations:
        RIN(f) = S_{delta I}(f) / <I>^2
    By default, the first half of the signal is discarded before computing the
    PSD. The three total-power diagnostics are:
        coherent:      |sum_i E_i|^2
        incoherent:    sum_i |E_i|^2
        phase-aligned: |sum_i E_i exp(-i <phi_i-phi_1>)|^2
    """
    freq_hz, rin_total_per_hz = compute_rin_from_intensity(
        total_field_intensity(y),
        nd,
        discard_fraction=discard_fraction,
        desired_df_hz=desired_df_hz,
    )
    freq_incoherent_hz, rin_incoherent_per_hz = compute_rin_from_intensity(
        incoherent_total_intensity(y),
        nd,
        discard_fraction=discard_fraction,
        desired_df_hz=desired_df_hz,
    )
    freq_phase_aligned_hz, rin_phase_aligned_per_hz = compute_rin_from_intensity(
        phase_aligned_total_field_intensity(y, discard_fraction=discard_fraction),
        nd,
        discard_fraction=discard_fraction,
        desired_df_hz=desired_df_hz,
    )
    freq_ind_hz, rin_individual_per_hz = compute_rin_from_intensity(
        individual_field_intensity(y),
        nd,
        discard_fraction=discard_fraction,
        desired_df_hz=desired_df_hz,
    )
    for label, other_freq in [
        ("incoherent", freq_incoherent_hz),
        ("phase-aligned", freq_phase_aligned_hz),
        ("individual", freq_ind_hz),
    ]:
        if len(freq_hz) != len(other_freq) or not np.allclose(freq_hz, other_freq):
            raise ValueError(f"Coherent and {label} RIN frequency grids do not match.")
    return (
        freq_hz,
        rin_total_per_hz,
        rin_incoherent_per_hz,
        rin_phase_aligned_per_hz,
        rin_individual_per_hz,
    )


def prepare_total_intensity_timeseries(
    t,
    y,
    phys,
    meta,
    alignment_discard_fraction=0.5,
):
    """Return downsampled total-field intensity and coupling ramp traces."""
    power_scale_mw = intensity_to_mw_scale(phys, meta)
    total_power_mw = total_field_intensity(y) * power_scale_mw
    incoherent_total_power_mw = incoherent_total_intensity(y) * power_scale_mw
    phase_aligned_total_power_mw = (
        phase_aligned_total_field_intensity(
            y,
            discard_fraction=alignment_discard_fraction,
        )
        * power_scale_mw
    )
    individual_power_mw = individual_field_intensity(y) * power_scale_mw
    n_time = min(len(t), total_power_mw.shape[-1])
    t_plot = t[:n_time]
    total_power_mw = total_power_mw[:, :n_time]
    incoherent_total_power_mw = incoherent_total_power_mw[:, :n_time]
    phase_aligned_total_power_mw = phase_aligned_total_power_mw[:, :n_time]
    individual_power_mw = individual_power_mw[:, :, :n_time]

    target_points = 6000
    plot_stride = max(1, n_time // target_points)
    t_us = t_plot[::plot_stride] * 1e6
    power_mean_mw = np.mean(total_power_mw[:, ::plot_stride], axis=0)
    power_std_mw = np.std(total_power_mw[:, ::plot_stride], axis=0)
    incoherent_power_mean_mw = np.mean(
        incoherent_total_power_mw[:, ::plot_stride],
        axis=0,
    )
    incoherent_power_std_mw = np.std(
        incoherent_total_power_mw[:, ::plot_stride],
        axis=0,
    )
    phase_aligned_power_mean_mw = np.mean(
        phase_aligned_total_power_mw[:, ::plot_stride],
        axis=0,
    )
    phase_aligned_power_std_mw = np.std(
        phase_aligned_total_power_mw[:, ::plot_stride],
        axis=0,
    )
    individual_power_mean_mw = np.mean(individual_power_mw[:, :, ::plot_stride], axis=0)
    individual_power_std_mw = np.std(individual_power_mw[:, :, ::plot_stride], axis=0)

    kappa_mat = np.asarray(phys["kappa_c_mat"])
    kappa_trace_ns = np.sum(kappa_mat, axis=(1, 2)) * 1e-9
    kappa_time = np.asarray(meta["time_arr"])
    kappa_on_t_ns = np.interp(t_plot, kappa_time, kappa_trace_ns)
    kappa_on_t_ns = kappa_on_t_ns[::plot_stride]

    return (
        t_us,
        power_mean_mw,
        power_std_mw,
        incoherent_power_mean_mw,
        incoherent_power_std_mw,
        phase_aligned_power_mean_mw,
        phase_aligned_power_std_mw,
        individual_power_mean_mw,
        individual_power_std_mw,
        kappa_on_t_ns,
        total_power_mw.shape[0],
    )


def plot_rin_summary(
    t,
    y,
    phys,
    meta,
    freq_hz,
    rin_db_per_hz,
    rin_incoherent_db_per_hz,
    rin_phase_aligned_db_per_hz,
    rin_individual_db_per_hz,
    output_path,
    rin_discard_fraction,
    power_ylim=(0.0, 5.0),
    rin_ylim=(-150.0, 0.0), 
    show=True,
):
    """Plot total-field time series and RIN spectrum on one figure."""
    (
        t_us,
        power_mean_mw,
        power_std_mw,
        incoherent_power_mean_mw,
        incoherent_power_std_mw,
        phase_aligned_power_mean_mw,
        phase_aligned_power_std_mw,
        individual_power_mean_mw,
        individual_power_std_mw,
        kappa_on_t_ns,
        n_cases,
    ) = prepare_total_intensity_timeseries(
        t,
        y,
        phys,
        meta,
        alignment_discard_fraction=rin_discard_fraction,
    )
    positive = freq_hz > 0
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, individual_power_mean_mw.shape[0])))

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(13, 11),
        dpi=300,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.5},
    )

    ax = axs[0]
    for laser_idx in range(individual_power_mean_mw.shape[0]):
        color = colors[laser_idx]
        ax.plot(
            t_us,
            individual_power_mean_mw[laser_idx],
            color=color,
            linewidth=1.8,
            alpha=0.85,
            label=rf"$P_{{{laser_idx + 1}}}$",
            zorder=2,
        )
        if n_cases > 1:
            ax.fill_between(
                t_us,
                individual_power_mean_mw[laser_idx] - individual_power_std_mw[laser_idx],
                individual_power_mean_mw[laser_idx] + individual_power_std_mw[laser_idx],
                color=color,
                alpha=0.08,
                linewidth=0,
                zorder=1,
            )
    ax.plot(
        t_us,
        power_mean_mw,
        color="black",
        linewidth=2.6,
        label=r"$|\sum_i E_i|^2$",
        zorder=3,
    )
    ax.plot(
        t_us,
        incoherent_power_mean_mw,
        color="tab:green",
        linewidth=2.0,
        linestyle="--",
        label=r"$\sum_i |E_i|^2$",
        zorder=3,
    )
    ax.plot(
        t_us,
        phase_aligned_power_mean_mw,
        color="tab:purple",
        linewidth=2.0,
        linestyle="-.",
        label=r"$|\sum_i E_i e^{-i\langle\Delta\phi_i\rangle}|^2$",
        zorder=3,
    )
    if n_cases > 1:
        ax.fill_between(
            t_us,
            power_mean_mw - power_std_mw,
            power_mean_mw + power_std_mw,
            color="black",
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.fill_between(
            t_us,
            incoherent_power_mean_mw - incoherent_power_std_mw,
            incoherent_power_mean_mw + incoherent_power_std_mw,
            color="tab:green",
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )
        ax.fill_between(
            t_us,
            phase_aligned_power_mean_mw - phase_aligned_power_std_mw,
            phase_aligned_power_mean_mw + phase_aligned_power_std_mw,
            color="tab:purple",
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel("Power (mW)")
    if power_ylim is not None:
        ax.set_ylim(*power_ylim)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", which="major")

    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax2.patch.set_visible(False)
    ax2.plot(
        t_us,
        kappa_on_t_ns,
        color="#1F77B4",
        linestyle=(0, (1, 1)),
        linewidth=2.8,
        alpha=0.95,
        label=r"$\kappa_c$ ramp",
        zorder=20,
    )
    ax2.set_ylabel(r"Coupling $\kappa_c$ (ns$^{-1}$)")
    ax2.set_ylim(0.0, meta["kappa_values"][-1] * 1e-9)
    ax2.tick_params(axis="y", which="major")

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper right",
        frameon=True,
    )
    ax.set_title(
        rf"Total-field power, $\Delta={meta['detuning_ghz']:.1f}\,\mathrm{{GHz}}$",
        pad=12,
    )

    ax = axs[1]
    for laser_idx in range(rin_individual_db_per_hz.shape[0]):
        ax.semilogx(
            freq_hz[positive],
            rin_individual_db_per_hz[laser_idx, positive],
            color=colors[laser_idx],
            linewidth=1.8,
            alpha=0.85,
            label=rf"$E_{{{laser_idx + 1}}}$",
        )
    ax.semilogx(
        freq_hz[positive],
        rin_db_per_hz[positive],
        color="black",
        linewidth=2.6,
        label=r"$|\sum_i E_i|^2$",
    )
    ax.semilogx(
        freq_hz[positive],
        rin_incoherent_db_per_hz[positive],
        color="tab:green",
        linewidth=2.1,
        linestyle="--",
        label=r"$\sum_i |E_i|^2$",
    )
    ax.semilogx(
        freq_hz[positive],
        rin_phase_aligned_db_per_hz[positive],
        color="tab:purple",
        linewidth=2.1,
        linestyle="-.",
        label=r"$|\sum_i E_i e^{-i\langle\Delta\phi_i\rangle}|^2$",
    )
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(khz_mhz_ghz_tick_label))
    ax.set_xlabel("Offset frequency")
    ax.set_ylabel("RIN (dB/Hz)")
    if rin_ylim is not None:
        ax.set_ylim(*rin_ylim)
    ax.set_title(
        rf"Total-field RIN, last {(1.0-rin_discard_fraction)*100:.0f}\% of signal, "
        rf"$\kappa_c={meta['kappa_c']*1e-9:.1f}\,\mathrm{{ns}}^{{-1}}$",
        pad=12,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", which="major")
    ax.legend(loc="upper right", frameon=True)

    plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_rin_kappa_map(
    freq_hz,
    kappa_values,
    rin_db_map,
    output_dir,
    rin_discard_fraction,
    title_label=r"Total-field",
    filename_stem="rin_total_field",
    save_legacy_total_names=False,
):
    """Plot the RIN spectrum as a map over the kappa continuation."""
    positive = freq_hz > 0
    freq_plot = freq_hz[positive]
    rin_plot = rin_db_map[:, positive]
    kappa_ns = kappa_values * 1e-9

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    mesh = ax.pcolormesh(
        freq_plot,
        kappa_ns,
        rin_plot,
        shading="auto",
        cmap="viridis",
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(khz_mhz_ghz_tick_label))
    ax.set_xlabel("Offset frequency")
    ax.set_ylabel(r"$\kappa_c$ (ns$^{-1}$)")
    ax.set_title(
        rf"{title_label} RIN map, last {(1.0-rin_discard_fraction)*100:.0f}\% of each segment",
        pad=12,
    )
    ax.tick_params(axis="both", which="major")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("RIN (dB/Hz)")
    cbar.ax.tick_params(which="major")

    fig.tight_layout()
    plt.savefig(f"{output_dir}/{filename_stem}_vs_kappa_map.png")
    if save_legacy_total_names:
        plt.savefig(f"{output_dir}/rin_total_field_vs_kappa_map.png")
        plt.savefig(f"{output_dir}/rin_Etot_vs_kappa_map.png")
    plt.show()
    plt.close(fig)


def main():
    phys, meta = build_two_laser_system()
    output_dir = LINEWIDTH_RESULTS_DIR / "2_lasers/rin_spectrum"
    frame_dir = f"{output_dir}/frames"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)

    kappa_values = np.asarray(meta["kappa_values"], dtype=float)
    rin_discard_fraction = 0.5
    freq_hz_ref = None
    rin_per_hz_list = []
    rin_db_list = []
    rin_incoherent_per_hz_list = []
    rin_incoherent_db_list = []
    rin_phase_aligned_per_hz_list = []
    rin_phase_aligned_db_list = []
    rin_individual_per_hz_list = []
    rin_individual_db_list = []
    t_last = None
    y_last = None
    phys_last = None
    meta_last = None

    phys["kappa_c_mat"] = build_kappa_segment(
        phys,
        meta,
        kappa_initial=kappa_values[0],
        kappa_final=kappa_values[0],
    )
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    history, _, _, _ = vcsel.generate_history(nd, shape="FR", n_cases=meta["n_cases"])
    if meta.get("average_continuation_history_across_noise", False):
        history = average_history_across_cases(history)

    pbar = tqdm(kappa_values, desc="Kappa continuation", unit="step")
    for k, kappa_c in enumerate(pbar):
        kappa_initial = kappa_values[k - 1] if k > 0 else kappa_values[0]
        phys["kappa_c_mat"] = build_kappa_segment(
            phys,
            meta,
            kappa_initial=kappa_initial,
            kappa_final=kappa_c,
        )
        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()
        t, y, _ = vcsel.integrate(history, nd=nd, progress=False, max_iter=1)

        (
            freq_hz,
            rin_per_hz,
            rin_incoherent_per_hz,
            rin_phase_aligned_per_hz,
            rin_individual_per_hz,
        ) = compute_field_rin_spectra(
            y,
            nd,
            discard_fraction=rin_discard_fraction,
        )
        rin_db_per_hz = 10 * np.log10(np.maximum(rin_per_hz, 1e-300))
        rin_incoherent_db_per_hz = 10 * np.log10(np.maximum(rin_incoherent_per_hz, 1e-300))
        rin_phase_aligned_db_per_hz = 10 * np.log10(np.maximum(rin_phase_aligned_per_hz, 1e-300))
        rin_individual_db_per_hz = 10 * np.log10(np.maximum(rin_individual_per_hz, 1e-300))

        if freq_hz_ref is None:
            freq_hz_ref = freq_hz
        elif len(freq_hz_ref) != len(freq_hz) or not np.allclose(freq_hz_ref, freq_hz):
            raise ValueError("RIN frequency grid changed during kappa continuation.")

        rin_per_hz_list.append(rin_per_hz)
        rin_db_list.append(rin_db_per_hz)
        rin_incoherent_per_hz_list.append(rin_incoherent_per_hz)
        rin_incoherent_db_list.append(rin_incoherent_db_per_hz)
        rin_phase_aligned_per_hz_list.append(rin_phase_aligned_per_hz)
        rin_phase_aligned_db_list.append(rin_phase_aligned_db_per_hz)
        rin_individual_per_hz_list.append(rin_individual_per_hz)
        rin_individual_db_list.append(rin_individual_db_per_hz)
        history = continuation_history_from_saved(y, nd)
        if meta.get("average_continuation_history_across_noise", False):
            history = average_history_across_cases(history)

        meta_current = dict(meta)
        meta_current["kappa_c"] = float(kappa_c)
        frame_path = f"{frame_dir}/rin_frame_kappa_{k:03d}.png"
        plot_rin_summary(
            t,
            y,
            phys,
            meta_current,
            freq_hz,
            rin_db_per_hz,
            rin_incoherent_db_per_hz,
            rin_phase_aligned_db_per_hz,
            rin_individual_db_per_hz,
            frame_path,
            rin_discard_fraction,
            show=False,
        )
        t_last = t
        y_last = y
        phys_last = dict(phys)
        meta_last = meta_current

        pbar.set_postfix(kappa_ns=f"{kappa_c*1e-9:.2f}")

    rin_per_hz_map = np.asarray(rin_per_hz_list)
    rin_db_map = np.asarray(rin_db_list)
    rin_incoherent_per_hz_map = np.asarray(rin_incoherent_per_hz_list)
    rin_incoherent_db_map = np.asarray(rin_incoherent_db_list)
    rin_phase_aligned_per_hz_map = np.asarray(rin_phase_aligned_per_hz_list)
    rin_phase_aligned_db_map = np.asarray(rin_phase_aligned_db_list)
    rin_individual_per_hz_map = np.asarray(rin_individual_per_hz_list)
    rin_individual_db_map = np.asarray(rin_individual_db_list)

    np.save(f"{output_dir}/kappa_c_values.npy", kappa_values)
    np.save(f"{output_dir}/rin_frequency_hz.npy", freq_hz_ref)
    np.save(f"{output_dir}/rin_total_field_per_hz_vs_kappa.npy", rin_per_hz_map)
    np.save(f"{output_dir}/rin_total_field_db_per_hz_vs_kappa.npy", rin_db_map)
    np.save(f"{output_dir}/rin_incoherent_total_per_hz_vs_kappa.npy", rin_incoherent_per_hz_map)
    np.save(f"{output_dir}/rin_incoherent_total_db_per_hz_vs_kappa.npy", rin_incoherent_db_map)
    np.save(f"{output_dir}/rin_phase_aligned_total_per_hz_vs_kappa.npy", rin_phase_aligned_per_hz_map)
    np.save(f"{output_dir}/rin_phase_aligned_total_db_per_hz_vs_kappa.npy", rin_phase_aligned_db_map)
    np.save(f"{output_dir}/rin_individual_fields_per_hz_vs_kappa.npy", rin_individual_per_hz_map)
    np.save(f"{output_dir}/rin_individual_fields_db_per_hz_vs_kappa.npy", rin_individual_db_map)
    np.save(f"{output_dir}/rin_total_field_per_hz.npy", rin_per_hz_map[-1])
    np.save(f"{output_dir}/rin_total_field_db_per_hz.npy", rin_db_map[-1])
    np.save(f"{output_dir}/rin_incoherent_total_per_hz.npy", rin_incoherent_per_hz_map[-1])
    np.save(f"{output_dir}/rin_incoherent_total_db_per_hz.npy", rin_incoherent_db_map[-1])
    np.save(f"{output_dir}/rin_phase_aligned_total_per_hz.npy", rin_phase_aligned_per_hz_map[-1])
    np.save(f"{output_dir}/rin_phase_aligned_total_db_per_hz.npy", rin_phase_aligned_db_map[-1])
    np.save(f"{output_dir}/rin_individual_fields_per_hz.npy", rin_individual_per_hz_map[-1])
    np.save(f"{output_dir}/rin_individual_fields_db_per_hz.npy", rin_individual_db_map[-1])

    plot_rin_kappa_map(
        freq_hz_ref,
        kappa_values,
        rin_db_map,
        output_dir,
        rin_discard_fraction,
        title_label=r"Coherent $|\sum_i E_i|^2$",
        filename_stem="rin_coherent_total_field",
        save_legacy_total_names=True,
    )
    plot_rin_kappa_map(
        freq_hz_ref,
        kappa_values,
        rin_incoherent_db_map,
        output_dir,
        rin_discard_fraction,
        title_label=r"Incoherent $\sum_i |E_i|^2$",
        filename_stem="rin_incoherent_total",
    )
    plot_rin_kappa_map(
        freq_hz_ref,
        kappa_values,
        rin_phase_aligned_db_map,
        output_dir,
        rin_discard_fraction,
        title_label=r"Phase-aligned coherent total",
        filename_stem="rin_phase_aligned_total",
    )
    plot_rin_summary(
        t_last,
        y_last,
        phys_last,
        meta_last,
        freq_hz_ref,
        rin_db_map[-1],
        rin_incoherent_db_map[-1],
        rin_phase_aligned_db_map[-1],
        rin_individual_db_map[-1],
        f"{output_dir}/rin_total_field_with_timeseries.png",
        rin_discard_fraction,
    )


if __name__ == "__main__":
    main()

#%%
# Plot saved RIN maps for each individual laser field.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import FuncFormatter, LogLocator


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', titlesize=28, labelsize=28)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('legend', fontsize=18)
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


def saved_rin_frequency_tick_label(freq_hz, _pos=None):
    """Format frequency ticks using kHz, MHz, and GHz."""
    if freq_hz <= 0:
        return ""
    if freq_hz < 1e6:
        value = freq_hz / 1e3
        unit = "kHz"
    elif freq_hz > 100e6:
        value = freq_hz / 1e9
        unit = "GHz"
    else:
        value = freq_hz / 1e6
        unit = "MHz"

    if np.isclose(value, round(value)):
        value_text = f"{value:.0f}"
    elif value < 10:
        value_text = f"{value:.1f}"
    else:
        value_text = f"{value:.0f}"
    return rf"${value_text}\,\mathrm{{{unit}}}$"


rin_map_dir = LINEWIDTH_RESULTS_DIR / "2_lasers/rin_spectrum"
rin_map_freq_hz = np.load(f"{rin_map_dir}/rin_frequency_hz.npy")
rin_map_kappa_c = np.load(f"{rin_map_dir}/kappa_c_values.npy")
rin_total_db_map = np.load(f"{rin_map_dir}/rin_total_field_db_per_hz_vs_kappa.npy")
rin_individual_db_map = np.load(f"{rin_map_dir}/rin_individual_fields_db_per_hz_vs_kappa.npy")

use_log_offset_frequency = True
freq_min_ghz = 0.0 if not use_log_offset_frequency else 20.0e-2
freq_max_ghz = 20.0
freq_window = (
    (rin_map_freq_hz >= freq_min_ghz * 1e9)
    & (rin_map_freq_hz <= freq_max_ghz * 1e9)
)
freq_plot_ghz = rin_map_freq_hz[freq_window] * 1e-9
kappa_plot_ns = rin_map_kappa_c * 1e-9
n_fields = rin_individual_db_map.shape[1]
freq_axis_label = "Offset frequency (GHz)"
map_suffix = "log" if use_log_offset_frequency else "020"


def configure_saved_rin_frequency_axis(ax):
    if use_log_offset_frequency:
        ax.set_xscale("log")
        ax.set_xlim(freq_min_ghz, freq_max_ghz)
    else:
        ax.set_xlim(freq_min_ghz, freq_max_ghz)
    ax.set_xlabel(freq_axis_label)

fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
mesh = ax.pcolormesh(
    freq_plot_ghz,
    kappa_plot_ns,
    rin_total_db_map[:, freq_window],
    shading="auto",
    cmap="jet",
)
configure_saved_rin_frequency_axis(ax)
ax.set_ylabel(r"$\kappa_c$ (ns$^{-1}$)")
ax.set_title(r"$E_{\mathrm{tot}}$ RIN map")
ax.tick_params(axis="both", which="major")
cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
cbar.set_label("RIN (dB/Hz)")
cbar.ax.tick_params(which="major")
fig.tight_layout()
plt.savefig(f"{rin_map_dir}/rin_Etot_vs_kappa_map_{map_suffix}.png", bbox_inches="tight")
plt.show()
plt.close(fig)

for field_idx in range(n_fields):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    mesh = ax.pcolormesh(
        freq_plot_ghz,
        kappa_plot_ns,
        rin_individual_db_map[:, field_idx, freq_window],
        shading="auto",
        cmap="jet",
        vmin=-160,
        vmax=-95,
    )
    configure_saved_rin_frequency_axis(ax)
    ax.set_ylabel(r"$\kappa_c$ (ns$^{-1}$)")
    ax.set_title(rf"$E_{{{field_idx + 1}}}$ RIN map")
    ax.tick_params(axis="both", which="major")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("RIN (dB/Hz)")
    cbar.ax.tick_params(which="major")
    fig.tight_layout()
    plt.savefig(f"{rin_map_dir}/rin_E{field_idx + 1}_vs_kappa_map_{map_suffix}.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

#%%
# Plot saved total-field and individual-field RIN spectra as linear-scale frames.
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter, LogLocator

rin_frame_dir_020 = LINEWIDTH_RESULTS_DIR / "2_lasers/rin_spectrum/frames_020"
os.makedirs(rin_frame_dir_020, exist_ok=True)

rin_total_db_map = np.load(f"{rin_map_dir}/rin_total_field_db_per_hz_vs_kappa.npy")

freq_window_020 = (rin_map_freq_hz >= 0.0) & (rin_map_freq_hz <= 20.0e9)
freq_frame_ghz = rin_map_freq_hz[freq_window_020] * 1e-9
rin_total_frame_map = rin_total_db_map[:, freq_window_020]
rin_individual_frame_map = rin_individual_db_map[:, :, freq_window_020]
frame_colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_fields)))

for kappa_idx, kappa_ns in enumerate(kappa_plot_ns):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    for field_idx in range(n_fields):
        ax.plot(
            freq_frame_ghz,
            rin_individual_frame_map[kappa_idx, field_idx],
            color=frame_colors[field_idx],
            linewidth=1.8,
            alpha=0.85,
            label=rf"$E_{{{field_idx + 1}}}$",
        )
    ax.plot(
        freq_frame_ghz,
        rin_total_frame_map[kappa_idx],
        color="black",
        linewidth=2.6,
        label=r"$E_{\mathrm{tot}}$",
    )
    ax.set_xlim(0.0, 20.0) 
    ax.set_ylim(-150.0, -60.0)
    ax.set_xlabel("Offset frequency (GHz)")
    ax.set_ylabel("RIN (dB/Hz)")
    ax.set_title(rf"RIN spectra, $\kappa_c={kappa_ns:.2f}\,\mathrm{{ns}}^{{-1}}$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", which="major")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(f"{rin_frame_dir_020}/rin_Etot_020_kappa_{kappa_idx:03d}.png", bbox_inches="tight")
    plt.close(fig)
