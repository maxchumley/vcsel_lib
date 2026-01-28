#%%
# Time series plotting for a VCSEL model

import numpy as np
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
from sympy import symbols, Eq, solve
from itertools import combinations
from IPython.display import clear_output


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
# kappa_c = 12e9
tau = 1e-9  # delay (s)
eta = 0.9
current_threshold = 3
I = eta*current_threshold * q/ tau_n * (N0 + 1/(g0*tau_p))



self_feedback = 0.0
coupling = 1.0


# np.array([1.0])



    
detuning = 4.0 # detuning (GHz)
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s

phi_p = 0.0

dt = 1*tau_p # 1 ps
Tmax = 2e-6
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
segment_len = int(steps/2)
segment_start = int(steps/2)

# Kappa ramp
kappa_c = 8e9


ramp_start = 2
ramp_shape = 100

noise_amplitude = 1#0 * np.exp(-((time_arr - 150*tau) ** 2) / (2 * 10*tau ** 2))#np.hstack([VCSEL.cosine_ramp(time_arr, 2*tau, 50*tau, kappa_initial=0, kappa_final=1),VCSEL.cosine_ramp(time_arr, 2*tau, 50*tau, kappa_initial=1, kappa_final=0)])




N_lasers = 2


n_iterations = 100

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
    'delta': np.sort(np.concatenate([[0.0],[delta]])),
    'coupling': coupling,
    'self_feedback': self_feedback,
    'noise_amplitude': noise_amplitude,
    'dt': dt,
    'Tmax': Tmax,
    'tau': tau,
    'N_lasers': N_lasers
}

N_bar_sym, S_bar = symbols('N_bar S_bar')


kappa_max = 20e9

vcsel = VCSEL(phys)
nd = vcsel.scale_params()
n_cases = len(nd['phi_p'])
nd['N_lasers'] = N_lasers

nd['kappa'] = nd['kappa'][-1]

history, freq_hist, eq = vcsel.generate_history(nd, shape='FR', n_cases=n_iterations)




vcsel = VCSEL(phys)
nd = vcsel.scale_params()


t, y, freqs = vcsel.integrate(history, nd=nd, progress=True, max_iter=100)


import numpy as np
import matplotlib.pyplot as plt

# ----------------- INITIALIZATION -----------------
N_lasers = y.shape[1] // 3
n_all = y[:, 0::3, :]     # (n_cases, N_lasers, steps)
S_all = y[:, 1::3, :]     # (n_cases, N_lasers, steps)
phi_all = y[:, 2::3, :]   # (n_cases, N_lasers, steps)

# Average photon numbers over cases
S_avg = np.mean(S_all, axis=0)  # (N_lasers, steps)
S_std = np.std(S_all, axis=0)

# Average frequencies + std
freqs_avg = np.mean(freqs, axis=0)  # (N_lasers, steps)
freqs_std = np.std(freqs, axis=0)

dphi = freqs_avg * 1e-9 / (2*np.pi*tau_p)
dphi_std = freqs_std * 1e-9 / (2*np.pi*tau_p)

# Apply previous values for initial delay steps
dphi[:, :2*delay_steps] = freq_hist[0] 
dphi_std[:, :2*delay_steps] = 0   # std = 0 for forced region

# Complex fields for all lasers
phi_mean = np.mean(phi_all, axis=0)
E_all = np.sqrt(S_avg) * (np.cos(phi_mean) + 1j*np.sin(phi_mean))
E_tot = np.sum(E_all, axis=0)

# Photon number std (via |E|²)
E_power_all = np.abs(E_all)**2     # (N_lasers, steps)
E_power_std = np.std(E_power_all, axis=0)

# ----------------- PHASE DIFFERENCES -----------------
phase_diff = np.unwrap(phi_all[:, None, :, :] - phi_all[:, :, None, :], axis=-1)
avg_cos_pd = np.mean(np.cos(phase_diff), axis=0)
std_cos_pd = np.std(np.cos(phase_diff), axis=0)

pairwise_cos_pd_avg = avg_cos_pd[np.triu_indices(N_lasers,k=1)[0],np.triu_indices(N_lasers,k=1)[1],:]
pairwise_cos_pd_std = std_cos_pd[np.triu_indices(N_lasers,k=1)[0],np.triu_indices(N_lasers,k=1)[1],:]




# ----------------- PLOTTING -----------------
clear_output(wait=True)
fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)
time_plot = time_arr * 1e6
subsample = 1

# -------- 1) Phase derivatives --------
for i in range(N_lasers):
    mu = dphi[i, ::subsample]
    sd = dphi_std[i, ::subsample]

    axs[0].plot(time_plot[::subsample], mu, label=f'$\dot{{\phi}}_{i+1}$', linewidth=2)
    axs[0].fill_between(time_plot[::subsample], mu - sd, mu + sd, alpha=0.3)

axs[0].set_xlabel('Time ($\mu s$)', fontsize=22)
axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
axs[0].legend(loc='upper right',fontsize=14)
axs[0].set_ylim(-1,5)
axs[0].grid(True, alpha=0.2)
axs[0].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[0].set_title(f'kappa_c = {kappa_c*1e-9:.2f} ns$^{{-1}}$', fontsize=24, pad=20)

pairs = list(combinations(range(N_lasers), 2))
# -------- 2) cos(Δφ) (laser 0 vs 1) --------
for l, row in enumerate(pairwise_cos_pd_avg):
    std = pairwise_cos_pd_std[l]
    axs[1].plot(time_plot, row, linewidth=2, label=f'$ \cos{{(\Delta \phi_{{{pairs[l][0]+1},{pairs[l][1]+1}}})}}$')
    axs[1].fill_between(time_plot, row - std, row + std, alpha=0.15)

axs[1].set_xlabel('Time ($\mu s$)', fontsize=22)
axs[1].set_ylabel(r"$\cos{(\Delta \phi)}$", fontsize=22)
axs[1].set_ylim(-1.1, 1.1)
axs[1].grid(True, alpha=0.2)
axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[1].legend(loc='lower right',fontsize=14)

# -------- 3) Photon numbers (|E|²) --------
for i in range(N_lasers):
    mu = np.abs(E_all[i])**2
    sd = E_power_std[i]

    axs[2].plot(time_plot, mu, label=f'$|E_{i+1}|^2$', linewidth=2)
    axs[2].fill_between(time_plot, mu - sd, mu + sd, alpha=0.15)


# ----------------- TOTAL FIELD -----------------
E_all_cases = (
    np.sqrt(S_all) *
    (np.cos(phi_all) + 1j*np.sin(phi_all))
)                           # (n_cases, N, steps)
E_tot_cases = np.sum(E_all_cases, axis=1)    # (n_cases, steps)
E_tot_power = np.abs(E_tot_cases)**2         # (n_cases, steps)
E_tot_power_mean = np.mean(E_tot_power, axis=0)
E_tot_power_std  = np.std(E_tot_power, axis=0)
axs[2].plot(time_plot, E_tot_power_mean, 'g', linewidth=2, alpha=0.8, label=r'$|E_{\rm tot}|^2$')
axs[2].fill_between(
    time_plot,
    E_tot_power_mean - E_tot_power_std,
    E_tot_power_mean + E_tot_power_std,
    color='green',
    alpha=0.15
)

axs[2].set_xlabel('Time ($\mu s$)', fontsize=22)
axs[2].legend(loc='upper right',fontsize=14)
axs[2].grid(True, alpha=0.2)
axs[2].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
axs[2].set_ylim(-5, 100)#np.max(E_tot_power_mean)*1.2)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].set_ylabel(r'$|E_i|^2$', fontsize=22)

# Optional twin axis
ax2 = axs[2].twinx()
ax2.set_ylabel('kappa ($ns^{-1}$)', color='blue', fontsize=20)
ax2.set_ylim(0, kappa_max*1e-9)
ax2.tick_params(axis='y', labelcolor='blue', labelsize=18)
ax2.plot(time_plot, kappa_arr[-len(time_plot):, 0,1]*1e-9, 'b--', alpha=0.5, linewidth=2)

plt.tight_layout()
plt.show()
plt.close(fig)

