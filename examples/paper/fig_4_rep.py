#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from vcsel_lib import VCSEL

try:
    from examples._paths import PAPER_DATA_DIR, PAPER_RESULTS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples._paths import PAPER_DATA_DIR, PAPER_RESULTS_DIR


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')


# Controls
run_simulations = True
use_saved_data = False
save_data = True

data_dir = PAPER_DATA_DIR / "fig_4_rep"
plot_dir = PAPER_RESULTS_DIR
data_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# Match the direct paper-equation test script, but use vcsel_lib integration.
tau_p = 7.15e-12
Tmax = 10.0e-6
dt = 1.0*tau_p
save_every = 100

n_cases = 100
seed = 3
use_noise = True

plot_ensemble_mean = True
plot_case_index = 0
smooth_frequency_data = True
frequency_smooth_window_ns = 50.0
frequency_smooth_window = max(
    1,
    int(round(frequency_smooth_window_ns * 1e-9 / (save_every * dt))),
)
xlim_us = (0.0, Tmax * 1e6)


# Ma et al. Figure 4 long-delay parameters.
alpha = 4.0

tau_n = 0.33e-9
g0 = 1.13e4
N0 = 8.2e6
s = 0.0
q = 1.602e-19
beta = 3.54e-5

tau = 5.0e-9
kappa_link_fig4 = 5.0e8
phi_p = 0.0
detuning_GHz = 0.3

N_lasers = 2
coupling_scheme = 'CUSTOM'
coupling = 1.0
self_feedback = 0.0
dx = 0.7

N_th = N0 + 1.0 / (g0 * tau_p)
I = 4.0 * q * N_th / tau_n
delta_dist = np.array([0.0, 2.0 * np.pi * detuning_GHz * 1e9])

# CUSTOM normalizes the whole adjacency matrix to sum(aMAT)=1. For two
# off-diagonal links, a 1 ns^-1 budget gives 5e8 s^-1 on each directed link.
kappa_budget_fig4 = 2.0 * kappa_link_fig4
final_kappa_ns_inv = kappa_budget_fig4 * 1e-9

# Original vcsel_lib coupling ramp controls.
ramp_start = 1e-10
ramp_shape = 1e-10

# Original cosine noise ramp controls.
noise_amplitude_max = 1.0
ramp_start_noise_us = 3.0
ramp_end_noise_us = 4.0

steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)


def ramped_custom_kappa(kappa_budget):
    return VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=0.0,
        kappa_final=kappa_budget,
        N_lasers=N_lasers,
        ramp_start=ramp_start,
        ramp_shape=ramp_shape,
        tau=tau,
        scheme=coupling_scheme,
        plot=False,
        dx=dx,
        aMAT=aMAT,
    )


def build_noise_amplitude():
    noise_amplitude_arr = np.zeros(steps)
    if use_noise:
        ramp_start_idx = int(ramp_start_noise_us * 1e-6 / dt)
        ramp_end_idx = int(ramp_end_noise_us * 1e-6 / dt)
        if ramp_start_idx < steps:
            ramp_end_idx = min(ramp_end_idx, steps)
            ramp_length = ramp_end_idx - ramp_start_idx
            if ramp_length > 0:
                t_normalized = np.arange(ramp_length) / ramp_length
                noise_amplitude_arr[ramp_start_idx:ramp_end_idx] = (
                    0.5 * noise_amplitude_max * (1.0 - np.cos(np.pi * t_normalized))
                )
            if ramp_end_idx < steps:
                noise_amplitude_arr[ramp_end_idx:] = noise_amplitude_max
    return noise_amplitude_arr


def moving_average(x, window):
    if window <= 1:
        return x
    window = int(window)
    left_pad = (window - 1) // 2
    right_pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(x, (left_pad, right_pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_frequency_array(freq_MHz, window):
    """Smooth frequency data with shape (n_cases, n_lasers, n_times)."""
    if not smooth_frequency_data or window <= 1:
        return np.asarray(freq_MHz, dtype=float).copy()
    freq_MHz = np.asarray(freq_MHz, dtype=float)
    smoothed = np.empty_like(freq_MHz)
    for case_idx in range(freq_MHz.shape[0]):
        for laser_idx in range(freq_MHz.shape[1]):
            smoothed[case_idx, laser_idx, :] = moving_average(freq_MHz[case_idx, laser_idx, :], window)
    return smoothed


def build_phys(kappa_budget):
    return {
        'tau_p': tau_p,
        'tau_n': tau_n,
        'g0': g0,
        'N0': N0,
        'N_bar': N_th,
        's': s,
        'beta': beta,
        'kappa_c_mat': ramped_custom_kappa(kappa_budget),
        'phi_p_mat': np.full((N_lasers, N_lasers), phi_p),
        'I': I,
        'q': q,
        'alpha': alpha,
        'delta': delta_dist,
        'coupling': coupling,
        'self_feedback': self_feedback,
        'noise_amplitude': build_noise_amplitude(),
        'dt': dt,
        'Tmax': Tmax,
        'tau': tau,
        'N_lasers': N_lasers,
        'sparse': False,
        'save_every': save_every,
        'max_output_gb': 50.0,
    }


def simulate_vcsel_case(kappa_budget, label):
    np.random.seed(seed)

    vcsel = VCSEL(build_phys(kappa_budget))
    nd = vcsel.scale_params()
    history, freq_history_GHz, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)

    # Keep phi_p as a true 2x2 matrix during integration; this avoids accidental
    # case-count expansion if history generation changes nd in future edits.
    nd['phi_p'] = np.full((N_lasers, N_lasers), phi_p)

    t, _, freqs = vcsel.integrate(
        history,
        nd=nd,
        progress=True,
        theta=0.5,
        max_iter=1,
        smooth_freqs=False,
        message=f"Ma Fig. 4 {label}",
    )

    freq_MHz = freqs * 1e-6 / (2.0 * np.pi * tau_p)
    t_idx_raw = np.rint(t / dt).astype(int)
    hist_mask = (t_idx_raw >= 0) & (t_idx_raw < freq_history_GHz.shape[2])
    if np.any(hist_mask):
        freq_MHz[:, :, hist_mask] = 1e3 * freq_history_GHz[:, :, t_idx_raw[hist_mask]]

    return {
        "label": label,
        "t": t,
        "freq_MHz": freq_MHz,
        "sigma_MHz": np.std(freq_MHz, axis=2),
        "kappa_budget": kappa_budget,
        "kappa_link": kappa_budget / 2.0,
        "dt": dt,
        "save_every": save_every,
        "use_noise": use_noise,
        "ramp_start_noise_us": ramp_start_noise_us,
        "ramp_end_noise_us": ramp_end_noise_us,
        "ramp_start": ramp_start,
        "ramp_shape": ramp_shape,
    }


noise_label = "noise_off"
if use_noise:
    noise_label = (
        f"noise_ramp_{ramp_start_noise_us:g}_to_{ramp_end_noise_us:g}us"
    ).replace(".", "p")
coupling_label = f"kappa_ramp_{ramp_start:g}_{ramp_shape:g}".replace(".", "p")
freq_label = "freq_history_mask_fix"

data_path = data_dir / (
    f"fig4_vcsel_lib_Tmax_{Tmax:.1e}_dt_{dt:.1e}_saveevery_{save_every}_"
    f"cases_{n_cases}_{noise_label}_{coupling_label}_{freq_label}.npz"
)

if use_saved_data and data_path.exists():
    loaded = np.load(data_path, allow_pickle=True)
    free_data = loaded["free"].item()
    mil_data = loaded["mil"].item()
elif run_simulations:
    free_data = simulate_vcsel_case(0.0, "free_running")
    mil_data = simulate_vcsel_case(kappa_budget_fig4, "mutually_injection_locked")
    if save_data:
        np.savez(data_path, free=free_data, mil=mil_data)
        print(f"Saved {data_path}")
else:
    raise RuntimeError("No data available. Set run_simulations=True or use_saved_data=True.")


#%%
# Figure 4-style three-panel plot.
t_free_us = free_data["t"] * 1e6
t_mil_us = mil_data["t"] * 1e6

# Smooth each noise realization first; only then average across cases.
free_freq_smoothed = smooth_frequency_array(free_data["freq_MHz"], frequency_smooth_window)
mil_freq_smoothed = smooth_frequency_array(mil_data["freq_MHz"], frequency_smooth_window)
free_sigma_data = np.std(free_freq_smoothed, axis=2)
mil_sigma_data = np.std(mil_freq_smoothed, axis=2)

if plot_ensemble_mean:
    free_freq = np.mean(free_freq_smoothed, axis=0)
    mil_freq = np.mean(mil_freq_smoothed, axis=0)
    plotted_sigma_free = np.std(free_freq, axis=1)
    plotted_sigma_mil = np.std(mil_freq, axis=1)
    plotted_label = f"ensemble mean over {free_data['freq_MHz'].shape[0]} cases"
else:
    free_freq = free_freq_smoothed[plot_case_index].copy()
    mil_freq = mil_freq_smoothed[plot_case_index].copy()
    plotted_sigma_free = free_sigma_data[plot_case_index]
    plotted_sigma_mil = mil_sigma_data[plot_case_index]
    plotted_label = f"case {plot_case_index}"

if smooth_frequency_data:
    effective_smooth_ns = frequency_smooth_window * save_every * dt * 1e9
    plotted_label += (
        f", smoothed over {effective_smooth_ns:g} ns "
        f"({frequency_smooth_window} saved samples)"
    )

fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.0), dpi=300, sharex=True)

axs[0].plot(t_free_us, free_freq[0], color="blue", linestyle="-", linewidth=1.0, label="Free-running DFB\\#1")
axs[1].plot(t_free_us, free_freq[1], color="green", linestyle="-", linewidth=1.0, label="Free-running DFB\\#2")
axs[2].plot(t_mil_us, mil_freq[0], color="black", linestyle="-", linewidth=1.0, label="MIL DFB\\#1")
axs[2].plot(t_mil_us, mil_freq[1], color="red", linestyle="--", linewidth=1.0, label="MIL DFB\\#2")

axs[0].set_ylabel("Frequency\n(MHz)", fontsize=15)
axs[1].set_ylabel("Frequency\n(MHz)", fontsize=15)
axs[2].set_ylabel("Frequency\n(MHz)", fontsize=15)
axs[2].set_xlabel(r"Time ($\mu$s)", fontsize=18)

# axs[0].set_ylim(-10, 10)
# axs[1].set_ylim(290, 310)
axs[2].set_ylim(135, 180)

for ax in axs:
    ax.set_xlim(*xlim_us)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

caption = (
    rf"$\tau_d = 5$ ns, $\kappa_c = 5\times10^8$ s$^{{-1}}$, "
    rf"$\phi_p = 0$, $\delta = 0.3$ GHz"
)
axs[0].set_title(caption, fontsize=15, pad=10)

print(f"Temporal sigma of plotted vcsel_lib frequency ({plotted_label}):")
print(f"  Free-running DFB#1: {plotted_sigma_free[0]:.3g} MHz")
print(f"  Free-running DFB#2: {plotted_sigma_free[1]:.3g} MHz")
print(f"  MIL DFB#1:          {plotted_sigma_mil[0]:.3g} MHz")
print(f"  MIL DFB#2:          {plotted_sigma_mil[1]:.3g} MHz")

print("Mean temporal sigma of individual noise realizations:")
print(f"  Free-running DFB#1: {np.mean(free_sigma_data[:, 0]):.3g} MHz")
print(f"  Free-running DFB#2: {np.mean(free_sigma_data[:, 1]):.3g} MHz")
print(f"  MIL DFB#1:          {np.mean(mil_sigma_data[:, 0]):.3g} MHz")
print(f"  MIL DFB#2:          {np.mean(mil_sigma_data[:, 1]):.3g} MHz")

plt.tight_layout()
fig_path = plot_dir / "fig_4_rep_vcsel_lib.png"
fig.savefig(fig_path, bbox_inches="tight", facecolor="white")
plt.show()
