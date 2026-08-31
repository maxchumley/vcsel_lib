#%%
import numpy as np
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.constants import hbar, c

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# --- Physical Parameters ---
alpha = 4
tau_p = 7.15e-12
tau_n = 0.33e-9
g0 = 1.13e4
N0 = 8.2e6
s = 0.0
q = 1.602e-19
beta = 3.54e-5
tau = 5e-9  # delay (s)
eta = 1.0
current_threshold = 4

I = eta * current_threshold * q / tau_n * (N0 + 1 / (g0 * tau_p))

# --- Simulation Parameters ---
self_feedback = 0.0
coupling = 1.0
noise_amplitude = 1.0
N_lasers = 2
coupling_scheme = 'CUSTOM'
dx = 0.7

lam = 910e-9
omega0 = 2 * np.pi * c / lam

# --- Time and Detuning ---
dt = 1 * tau_p
Tmax = 5e-7
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)

detuning = 0.3
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
delta_dist = np.sort(np.concatenate([delta / 2 * np.linspace(0, 2, N_lasers)]))

n_cases = 100
plot_stride = 1

# --- Coupling Ramp Parameters ---
ramp_start = 3
ramp_shape = 0.1
final_kappa_ns_inv = 0.5  # ns^-1
final_kappa = final_kappa_ns_inv * 1e9

# --- Build coupling matrix with ramp ---
aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)
kappa_arr = VCSEL.build_coupling_matrix(
    time_arr=time_arr,
    kappa_initial=0,
    kappa_final=final_kappa,
    N_lasers=N_lasers,
    ramp_start=ramp_start,
    ramp_shape=ramp_shape,
    tau=tau,
    scheme=coupling_scheme,
    plot=False,
    dx=dx,
    aMAT=aMAT,
)

# --- Physics Dictionary ---
phi_p_vals = np.array([0.0])
phys = {
    'tau_p': tau_p,
    'tau_n': tau_n,
    'g0': g0,
    'N0': N0,
    'N_bar': N0 + 1 / (g0 * tau_p),
    's': s,
    'beta': beta,
    'kappa_c_mat': kappa_arr[-1, :, :],
    'phi_p_mat': np.ones(shape=(n_cases, N_lasers, N_lasers)) * phi_p_vals[:, None, None],
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
    'sparse': False,
    'save_every': 1,
    'smooth_freqs': True,
}

# --- Initialize VCSEL and generate history ---
vcsel = VCSEL(phys)
nd = vcsel.scale_params()

history, freq_history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)

# --- Set phase and integrate ---
nd['phi_p'] = phys['phi_p_mat']
t, y, freqs = vcsel.integrate(history, nd=nd, progress=True, theta=0.5, max_iter=1, smooth_freqs=True)

# Map time indices
t_idx = np.clip(np.rint(t / dt).astype(int), 0, len(time_arr) - 1)

# Extract intensity and phase indices
S_idx = [3 * i + 1 for i in range(N_lasers)]
phi_idx = [3 * i + 2 for i in range(N_lasers)]

S = y[:, S_idx, :]
phi = y[:, phi_idx, :]

# --- Compute frequency derivatives ---
dphi = freqs * 1e-9 / (2 * np.pi * tau_p)
hist_mask = t_idx < freq_history.shape[2]
if np.any(hist_mask):
    dphi[:, :, hist_mask] = freq_history[:, :, t_idx[hist_mask]]

dphi_mean = np.mean(dphi[:, :, :-1:plot_stride], axis=0)
dphi_std = np.std(dphi[:, :, :-1:plot_stride], axis=0)

# --- Compute phase differences ---
cos_pd_pair_mean = np.zeros((N_lasers - 1, phi.shape[2]))
cos_pd_pair_std = np.zeros((N_lasers - 1, phi.shape[2]))
for i in range(N_lasers - 1):
    cos_ij = np.cos(phi[:, i, :] - phi[:, i + 1, :])
    cos_pd_pair_mean[i, :] = np.mean(cos_ij, axis=0)
    cos_pd_pair_std[i, :] = np.std(cos_ij, axis=0)

# --- Compute total power ---
time_plot_full = t[::plot_stride] * 1e6
intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
E = np.sqrt(S) * np.exp(1j * phi)
E_tot_cases_mW = (np.abs(E.sum(axis=1)) ** 2) * intensity_to_mW
E_tot_mean = np.mean(E_tot_cases_mW, axis=0)[::plot_stride]
E_tot_std = np.std(E_tot_cases_mW, axis=0)[::plot_stride]
E_i_mean_arr = np.mean(S, axis=0) * intensity_to_mW
E_i_std_arr = np.std(S, axis=0) * intensity_to_mW

# --- PLOTTING ---
fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)

line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X']
laser_colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#000000']
pair_colors = ['#332288', '#117733', '#CC6677', '#AA4499', '#44AA99', '#999933']

time_plot = t[:-1:plot_stride] * 1e6
marker_every = max(1, len(time_plot) // 24)

# --- Panel 0: Frequency derivatives ---
for i in range(N_lasers):
    style = line_styles[i % len(line_styles)]
    color_i = laser_colors[i % len(laser_colors)]
    mean_dphi = dphi_mean[i, :]
    std_dphi = dphi_std[i, :]
    axs[0].plot(
        time_plot,
        mean_dphi,
        linestyle=style,
        color=color_i,
        linewidth=2.4,
        marker=marker_styles[i % len(marker_styles)],
        markevery=marker_every,
        markersize=4,
        markerfacecolor='white',
        markeredgewidth=1.0,
        label=fr'$\dot{{\phi}}_{i+1}$',
    )
    axs[0].fill_between(
        time_plot,
        mean_dphi - std_dphi,
        mean_dphi + std_dphi,
        color=color_i,
        alpha=0.10,
    )

if N_lasers <= 6:
    axs[0].legend(loc='upper right', fontsize=22)
axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=28)
axs[0].grid(True, alpha=0.2)
axs[0].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
axs[0].tick_params(axis='both', which='major', labelsize=24)
axs[0].set_title(rf'$\kappa_c = {final_kappa_ns_inv:.2f}\,\mathrm{{ns}}^{{-1}}$', fontsize=30, pad=20)

# --- Panel 1: Phase differences ---
for i in range(N_lasers - 1):
    mean_cos = cos_pd_pair_mean[i, :-1:plot_stride]
    std_cos = cos_pd_pair_std[i, :-1:plot_stride]
    style = line_styles[(i + 1) % len(line_styles)]
    color_i = pair_colors[i % len(pair_colors)]
    axs[1].plot(
        time_plot,
        mean_cos,
        linestyle=style,
        color=color_i,
        linewidth=2.4,
        label=fr'$\cos(\phi_{i+1}-\phi_{i+2})$',
    )
    axs[1].fill_between(
        time_plot,
        mean_cos - std_cos,
        mean_cos + std_cos,
        color=color_i,
        alpha=0.10,
    )

axs[1].set_ylim(-1.1, 1.1)
axs[1].grid(True, alpha=0.2)
axs[1].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
axs[1].set_ylabel(r'$\cos(\Delta\phi)$', fontsize=28)
axs[1].tick_params(axis='both', which='major', labelsize=24)
if N_lasers <= 6:
    axs[1].legend(loc='lower right', fontsize=22)

# --- Panel 2: Total and individual power ---
axs[2].plot(
    time_plot_full,
    E_tot_mean,
    color='black',
    linestyle='-',
    linewidth=2.8,
    alpha=0.95,
    label=r'$P_{\rm tot}$',
    zorder=3,
)
axs[2].fill_between(
    time_plot_full,
    E_tot_mean - E_tot_std,
    E_tot_mean + E_tot_std,
    color='black',
    alpha=0.08,
    zorder=1,
)

for i in range(N_lasers):
    E_i_mean = E_i_mean_arr[i, ::plot_stride]
    E_i_std = E_i_std_arr[i, ::plot_stride]
    style = line_styles[i % len(line_styles)]
    color_i = laser_colors[i % len(laser_colors)]
    axs[2].plot(
        time_plot_full,
        E_i_mean,
        color=color_i,
        linestyle=style,
        linewidth=2.2,
        label=f'$P_{i+1}$',
        zorder=3,
    )
    axs[2].fill_between(
        time_plot_full,
        E_i_mean - E_i_std,
        E_i_mean + E_i_std,
        color=color_i,
        alpha=0.08,
        zorder=1,
    )

axs[2].set_xlabel('Time ($\mu$s)', fontsize=28)
axs[2].set_ylabel('Power (mW)', fontsize=28)
axs[2].grid(True, alpha=0.2)
axs[2].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
axs[2].tick_params(axis='both', which='major', labelsize=24)

if N_lasers <= 6:
    h1, l1 = axs[2].get_legend_handles_labels()
    axs[2].legend(h1, l1, loc='upper right', fontsize=18, ncol=2, frameon=True)

plt.tight_layout()
plt.show()
