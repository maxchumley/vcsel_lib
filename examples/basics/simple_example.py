#%%
# Time series plotting for a VCSEL model

import numpy as np
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
from itertools import combinations
from IPython.display import clear_output
from scipy.constants import hbar, c
from pathlib import Path

try:
    from examples._paths import BASICS_RESULTS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples._paths import BASICS_RESULTS_DIR


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')

# --- Parameters ---
alpha = 2
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4e-6
q = 1.602e-19
beta = 1.e-3
tau = 1e-9  # delay (s)
eta = 0.9
current_threshold = 3
I = eta * current_threshold * q / tau_n * (N0 + 1 / (g0 * tau_p))
lam = 910e-9

self_feedback = 0.0
coupling = 1.0

N_lasers = 100
coupling_scheme = 'CUSTOM'
detuning = 1.0 # detuning (GHz)
delta = detuning * 2 * np.pi * 1e9
delta_dist = delta / 2 * np.linspace(-1, 1, N_lasers)

dt = 1 * tau_p
Tmax = 2.0e-7
kappa_c_final = 1000e9
noise_amplitude = 0.0
n_iterations = 1
aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)
phi_p_vals = np.array([np.pi])
ramp_shape_tau = 10
noise_ramp_10_90_tau = 2.0
smooth_freqs = True

omega0 = 2 * np.pi * c / lam

steps = int(np.round(Tmax / dt)) + 1
time_arr = np.arange(steps, dtype=float) * dt
Tmax = time_arr[-1]
delay_steps = int(tau / dt)
kappa_c = kappa_c_final

noise_start = 1.0

kappa_arr = VCSEL.build_coupling_matrix(
    time_arr=time_arr,
    kappa_initial=0.0,
    kappa_final=kappa_c_final,
    N_lasers=N_lasers,
    ramp_start=5.0,
    ramp_shape=ramp_shape_tau,
    tau=tau,
    scheme=coupling_scheme,
    aMAT=aMAT,
)

phys = {
    'tau_p': tau_p,
    'tau_n': tau_n,
    'g0': g0,
    'N0': N0,
    'N_bar': N0 + 1/(g0*tau_p),
    's': s,
    'beta': beta,
    'kappa_c_mat': kappa_arr,
    'phi_p_mat': np.ones(shape=(n_iterations,N_lasers,N_lasers))*phi_p_vals[:,None,None],
    'I': I,
    'q': q,
    'alpha': alpha,
    'delta': delta_dist,
    'coupling': coupling,
    'self_feedback': self_feedback,
    'noise_amplitude': noise_amplitude,
    'dt': dt,
    'Tmax': Tmax,
    'tau': tau,
    'N_lasers': N_lasers,
    'save_every': 1,
}

kappa_max = 20e9

vcsel = VCSEL(phys)
nd = vcsel.scale_params()

nd['N_lasers'] = N_lasers


def noise_ramp(t):
    return VCSEL.cosine_ramp(
        np.array([t]),
        t_start=noise_start * tau,
        rise_10_90=noise_ramp_10_90_tau * tau,
        kappa_initial=0.0,
        kappa_final=noise_amplitude,
    )[0]
nd['noise_amplitude'] = noise_ramp

# nd['kappa'] = nd['kappa'][-1]
history, freq_hist, eq, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_iterations)

t, y, freqs = vcsel.integrate(history, nd=nd, progress=True, max_iter=1, smooth_freqs=smooth_freqs)

# ----------------- INITIALIZATION -----------------
N_lasers = y.shape[1] // 3
S_all = y[:, 1::3, :]     # (n_cases, N_lasers, steps)
phi_all = y[:, 2::3, :]   # (n_cases, N_lasers, steps)

# Phase derivatives per case (GHz), with initial delay steps filled from history
dphi_all = freqs * 1e-9 / (2*np.pi*tau_p)
save_every = int(max(1, nd.get('save_every', 1)))
if save_every > 1:
    freq_hist_plot = freq_hist[:, :, ::save_every]
else:
    freq_hist_plot = freq_hist
hist_len = min(freq_hist_plot.shape[2], dphi_all.shape[2])
dphi_all[:, :, :hist_len] = freq_hist_plot[:, :, :hist_len]
delay_steps_saved = max(1, int(round(delay_steps / save_every)))

# Calculate means and stds along axis 0
dphi_mean = np.mean(dphi_all, axis=0)
dphi_std = np.std(dphi_all, axis=0)
S_mean = np.mean(S_all, axis=0)
S_std = np.std(S_all, axis=0)
pairs = list(combinations(range(N_lasers), 2))
wrapped_delta_phi_pairs = []
# for i, j in pairs:
#     wrapped_delta_phi = np.angle(np.exp(1j * (phi_all[:, j, :] - phi_all[:, i, :])))
for i in range(1,N_lasers):
    wrapped_delta_phi = np.angle(np.exp(1j * (phi_all[:, i, :] - phi_all[:, 0, :])))
    wrapped_delta_phi_pairs.append(wrapped_delta_phi)
wrapped_delta_phi_pairs = np.asarray(wrapped_delta_phi_pairs)


# Output power only (total field), using existing simulation arrays `t` and `y`.

if "t" not in globals() or "y" not in globals():
    raise RuntimeError("Run the simulation/integration cell first so `t` and `y` exist.")

S_all = y[:, 1::3, :]      # (n_cases, N_lasers, steps)
phi_all = y[:, 2::3, :]    # (n_cases, N_lasers, steps)

intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)

# Total coherent field power per case
E_all = np.sqrt(S_all) * np.exp(1j * phi_all)
P_tot_cases_mW = np.abs(np.sum(E_all, axis=1))**2 * intensity_to_mW

P_tot_mean_mW = np.mean(P_tot_cases_mW, axis=0)
P_tot_std_mW = np.std(P_tot_cases_mW, axis=0)

time_plot = t * 1e6
show_uncertainty = noise_amplitude > 0
plot_dir = BASICS_RESULTS_DIR
plot_dir.mkdir(parents=True, exist_ok=True)

# fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
# ax.plot(time_plot, P_tot_mean_mW, color="g", linewidth=2.5, label=r"$P_{\rm tot}$")
# if show_uncertainty:
#     ax.fill_between(
#         time_plot,
#         P_tot_mean_mW - P_tot_std_mW,
#         P_tot_mean_mW + P_tot_std_mW,
#         color="g",
#         alpha=0.3,
#     )

# ax.set_xlabel(r"Time ($\mu$s)", fontsize=24)
# ax.set_ylabel("Output Power (mW)", fontsize=24)
# ax.tick_params(axis='both', which='major', labelsize=20)
# ax.grid(True, alpha=0.25)
# ax.legend(fontsize=16)
# plt.tight_layout()
# total_power_plot_path = plot_dir / "simple_example_total_power.png"
# fig.savefig(total_power_plot_path, bbox_inches="tight")
# plt.show()



# ----------------- PLOTTING -----------------
# clear_output(wait=True)
fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)
time_plot = t * 1e6
subsample = 1
show_uncertainty = noise_amplitude > 0

# -------- 1) Phase derivatives (mean ± std) --------
for i in range(N_lasers):
    axs[0].plot(time_plot[::subsample], dphi_mean[i, ::subsample], linewidth=2, label=rf'$\dot{{\phi}}_{i+1}$')
    if show_uncertainty:
        axs[0].fill_between(time_plot[::subsample], dphi_mean[i, ::subsample] - dphi_std[i, ::subsample],
                            dphi_mean[i, ::subsample] + dphi_std[i, ::subsample], alpha=0.3)

axs[0].set_xlabel(r'Time ($\mu s$)', fontsize=22)
axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
if N_lasers <= 4:
    axs[0].legend(loc='upper right', fontsize=14)
# axs[0].set_ylim(-12,5)
axs[0].grid(True, alpha=0.2)
axs[0].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[0].set_title(
    rf'$\kappa_c$: $0 \rightarrow {kappa_c_final*1e-9:.0f}$ ns$^{{-1}}$',
    fontsize=24,
    pad=20,
)


# -------- 2) Wrapped pairwise phase differences, all trajectories --------
pair_colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(N_lasers, 1)))
for pair_index, (i, j) in enumerate(pairs[:N_lasers-1]):
    for case_index in range(n_iterations):
        axs[1].plot(
            time_plot,
            wrapped_delta_phi_pairs[pair_index, case_index, :],
            color=pair_colors[pair_index],
            linewidth=0.6,
            alpha=1,
        )
    axs[1].plot(
        [],
        [],
        color=pair_colors[pair_index],
        linewidth=2,
        label=rf'$\Delta \phi_{{{j+1},{i+1}}}=\phi_{j+1}-\phi_{i+1}$',
    )

axs[1].set_xlabel(r'Time ($\mu s$)', fontsize=22)
axs[1].set_ylabel(r"wrapped $\Delta \phi$ (rad)", fontsize=22)
axs[1].set_ylim(-np.pi, np.pi)
axs[1].set_yticks([-np.pi, 0.0, np.pi])
axs[1].set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
axs[1].grid(True, alpha=0.2)
axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
axs[1].tick_params(axis='both', which='major', labelsize=18)

if N_lasers <= 4:
    axs[1].legend(loc='lower right', fontsize=14)

# -------- 3) Photon numbers (mean ± std) --------
# Convert nondimensional intensities to optical power (mW) using free-running outcoupling scale 1/tau_p.
intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)

for i in range(N_lasers):
    S_mean_mW = S_mean[i, :] * intensity_to_mW
    S_std_mW = S_std[i, :] * intensity_to_mW
    axs[2].plot(time_plot, S_mean_mW, linewidth=2, label=f'$P_{i+1}$')
    if show_uncertainty:
        axs[2].fill_between(time_plot, S_mean_mW - S_std_mW, S_mean_mW + S_std_mW, alpha=0.3)

# Total field (mean plus min/max envelope)
E_all_cases = np.sqrt(S_all) * (np.cos(phi_all) + 1j*np.sin(phi_all))
E_tot_cases = np.sum(E_all_cases, axis=1)
E_tot_power = np.abs(E_tot_cases)**2
E_tot_mean = np.mean(E_tot_power, axis=0)
E_tot_cases_mW = E_tot_power * intensity_to_mW
E_tot_mean_mW = E_tot_mean * intensity_to_mW
E_tot_min_mW = np.min(E_tot_cases_mW, axis=0)
E_tot_max_mW = np.max(E_tot_cases_mW, axis=0)
if show_uncertainty:
    axs[2].plot(time_plot, E_tot_min_mW, color='g', linestyle='--', linewidth=1.4, alpha=0.75, label=r'$\min P_{\rm tot}$')
    axs[2].plot(time_plot, E_tot_max_mW, color='g', linestyle=':', linewidth=1.8, alpha=0.75, label=r'$\max P_{\rm tot}$')
axs[2].plot(time_plot, E_tot_mean_mW, 'g', linewidth=2.5, label=r'$\langle P_{\rm tot}\rangle$')

axs[2].set_xlabel(r'Time ($\mu s$)', fontsize=22)
if N_lasers <= 4:
    axs[2].legend(loc='upper right', fontsize=14)
axs[2].grid(True, alpha=0.2)
axs[2].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].set_ylabel(r'Output Power (mW)', fontsize=22)

# Optional twin axis
ax2 = axs[2].twinx()
ax2.set_ylabel(r'Coupling Power ($\mu$W)', color='black', fontsize=20)
ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
time_idx = np.round(t / (nd["dt"] * nd["tau_p"])).astype(int)
time_idx = np.clip(time_idx, 0, kappa_arr.shape[0] - 1)
kappa_plot = kappa_arr[time_idx, 0, 1]
P_c = hbar * omega0 * kappa_plot * nd['sbar'] / (g0 * tau_n)
P_c_uW = P_c * 1e6
ax2.plot(time_plot, P_c_uW, 'k--', alpha=0.5, linewidth=2)
ax2.set_ylim(0, P_c_uW.max() * 1.5)

plt.tight_layout()
summary_plot_path = plot_dir / "simple_example_summary.png"
fig.savefig(summary_plot_path, bbox_inches="tight")
plt.show()
plt.close(fig)
