#%%
# Time series plotting for a VCSEL model

import numpy as np
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
from itertools import combinations
from IPython.display import clear_output
from scipy.constants import hbar, c


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')

# Physical parameters
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
I = eta*current_threshold * q/ tau_n * (N0 + 1/(g0*tau_p))
lam = 910e-9
omega0 = 2*np.pi*c/lam



self_feedback = 0.0
coupling = 1.0



N_lasers = 4
detuning = 4 # detuning (GHz)
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
delta_dist = delta/2 * np.linspace(-1, 1, N_lasers)  # detuning distribution for 2 lasers



dt = 1*tau_p # 1 ps
Tmax = 2e-6
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
# Kappa ramp
kappa_c = 10e9


ramp_start = 100
ramp_shape = 400

noise_amplitude = 0.0
noise_start = 1.0
noise_ramp_10_90_tau = 10.0


n_iterations = 1

kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme='ATA')




phi_p_vals = np.array([0.0])


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
smooth_freqs = True


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
delta_phi_all = phi_all[:, None, :, :] - phi_all[:, :, None, :]
cos_pd_all = np.cos(delta_phi_all)
cos_pd_mean = np.mean(cos_pd_all, axis=0)
cos_pd_std = np.std(cos_pd_all, axis=0)


# ----------------- PLOTTING -----------------
# clear_output(wait=True)
fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)
time_plot = t * 1e6
subsample = 1

# -------- 1) Phase derivatives (mean ± std) --------
for i in range(N_lasers):
    axs[0].plot(time_plot[::subsample], dphi_mean[i, ::subsample], linewidth=2, label=f'$\dot{{\phi}}_{i+1}$')
    axs[0].fill_between(time_plot[::subsample], dphi_mean[i, ::subsample] - dphi_std[i, ::subsample], 
                        dphi_mean[i, ::subsample] + dphi_std[i, ::subsample], alpha=0.3)

axs[0].set_xlabel('Time ($\mu s$)', fontsize=22)
axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
axs[0].legend(loc='upper right', fontsize=14)
# axs[0].set_ylim(-12,5)
axs[0].grid(True, alpha=0.2)
axs[0].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[0].set_title(f'kappa_c = {kappa_c*1e-9:.2f} ns$^{{-1}}$', fontsize=24, pad=20)

# -------- 2) cos(Δφ) (mean ± std) --------
pairs = list(combinations(range(N_lasers), 2))
for l, (i, j) in enumerate(pairs):
    axs[1].plot(time_plot, cos_pd_mean[i, j, :], linewidth=2, label=f'$ \\cos{{(\\Delta \\phi_{{{i+1},{j+1}}})}}$')
    axs[1].fill_between(time_plot, cos_pd_mean[i, j, :] - cos_pd_std[i, j, :], 
                        cos_pd_mean[i, j, :] + cos_pd_std[i, j, :], alpha=0.3)

axs[1].set_xlabel('Time ($\mu s$)', fontsize=22)
axs[1].set_ylabel(r"$\cos{(\Delta \phi)}$", fontsize=22)
axs[1].set_ylim(-1.1, 1.1)
axs[1].grid(True, alpha=0.2)
axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[1].legend(loc='lower right', fontsize=14)

# -------- 3) Photon numbers (mean ± std) --------
# Convert nondimensional intensities to optical power (mW) using free-running outcoupling scale 1/tau_p.
intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)

for i in range(N_lasers):
    S_mean_mW = S_mean[i, :] * intensity_to_mW
    S_std_mW = S_std[i, :] * intensity_to_mW
    axs[2].plot(time_plot, S_mean_mW, linewidth=2, label=f'$P_{i+1}$')
    axs[2].fill_between(time_plot, S_mean_mW - S_std_mW, S_mean_mW + S_std_mW, alpha=0.3)

# Total field (mean ± std)
E_all_cases = np.sqrt(S_all) * (np.cos(phi_all) + 1j*np.sin(phi_all))
E_tot_cases = np.sum(E_all_cases, axis=1)
E_tot_power = np.abs(E_tot_cases)**2
E_tot_mean = np.mean(E_tot_power, axis=0)
E_tot_std = np.std(E_tot_power, axis=0)
E_tot_mean_mW = E_tot_mean * intensity_to_mW
E_tot_std_mW = E_tot_std * intensity_to_mW
axs[2].plot(time_plot, E_tot_mean_mW, 'g', linewidth=2, label=r'$P_{\rm tot}$')
axs[2].fill_between(time_plot, E_tot_mean_mW - E_tot_std_mW, E_tot_mean_mW + E_tot_std_mW, color='g', alpha=0.3)

axs[2].set_xlabel('Time ($\mu s$)', fontsize=22)
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
plt.show()
plt.close(fig)

# %%
