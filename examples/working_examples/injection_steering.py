#%%

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
# matplotlib.use('Agg')

from IPython.display import clear_output
import gc
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelextrema
from joblib import Parallel, delayed
import time 
from scipy.constants import hbar, c

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
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
# kappa_c = 12e9
tau = 1e-9  # delay (s)
eta = 0.9
current_threshold = 3

I = eta*current_threshold * q/ tau_n * (N0 + 1/(g0*tau_p))

# print(f"p={g0*tau_p * (I*tau_n/(q) - N0) - 1:.3f}")


self_feedback = 0.0
coupling = 1.0
noise_amplitude = 0.0

N_lasers = 2
coupling_scheme = 'ATA'
dx = 0.7

lam = 910e-9
omega0 = 2*np.pi*c/lam



results = None


# for detuning in np.linspace(4,5,50):
detuning = 4.0
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
# Create evenly distributed detuning for both even and odd N_lasers
delta_dist = np.sort(np.concatenate([delta/2 * np.linspace(-1, 1, N_lasers)]))




phi_p = 0#np.pi

dt = 1*tau_p# 1 ps
Tmax = 1.5e-6
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
segment_len = int(steps/2)
segment_start = int(steps/2)
plot_stride = 1
cos_smooth_tau = 0.5

n_kappa = 50
ramp_start = 10
ramp_shape = 50

final_kappa_arr = np.linspace(0e9,20e9,50)

kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=final_kappa_arr[0], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)





n_cases = 1

phi_p_vals = np.array([0.0])

phys = {
    'tau_p': tau_p,
    'tau_n': tau_n,
    'g0': g0,
    'N0': N0,
    'N_bar': N0 + 1/(g0*tau_p),
    's': s,
    'beta': beta,
    'kappa_c_mat': None,
    'phi_p_mat': np.ones(shape=(n_cases,N_lasers,N_lasers))*phi_p_vals[:,None,None],
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
    'save_every':1
}

phys['kappa_c_mat'] = kappa_arr[-1,:,:]




phys['injection'] = False

# n_kappa = len(kappa_c)-1

# inj_freqs = np.linspace(-3,3,n_kappa)

# Gaussian kappa injection centered at peak_time with controllable width

peak_time = 200*tau                     # center (s), e.g. peak = 3*tau above
#30e9


extrema = []
S_idx   = [3*i + 1 for i in range(N_lasers)]
phi_idx = [3*i + 2 for i in range(N_lasers)]
# --- Loop over segments of kappa --- 
# max_tau_inj_width = 200

inj_phases = np.linspace(0,2*np.pi,n_cases)



for k in range(0, n_kappa):
    k=49

    final_kappa = final_kappa_arr[k]


    kappa_inj_width = 10 * tau          # width (s) — change this to control the Gaussian spread 
    kappa_inj_amp_peak = 10 *final_kappa
    kappa_inj_amp = np.linspace(kappa_inj_amp_peak, kappa_inj_amp_peak, n_cases)

    # Slice the ramp for this segment
    if k > 0:
        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=final_kappa, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)

    phys['kappa_c_mat'] = kappa_arr[-1,:,:]
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()


    # if k == 0:
    if results is not None:
        results = np.unique(results, axis=0)
        for eq_pt in results:
            if np.any(np.isnan(eq_pt)):
                continue
            guesses.append(np.concatenate([
                eq_pt[1::2][:N_lasers],  # S1, S2, ...
                eq_pt[2*N_lasers:3*N_lasers-1],                      # φ1, φ2, ...
                np.array([eq_pt[-1]])                      # ω
            ]))
    else:
        guesses = []
        # guesses.append(np.concatenate([
        #         eq_pt[1::2][:N_lasers],  # S1, S2, ...
        #         eq_pt[2*N_lasers:3*N_lasers-1],                      # φ1, φ2, ...
        #         np.array([eq_pt[-1]])                      # ω
        #     ]))

    history, freq_history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)
    # eq_history, freq_hist = vcsel.generate_history(nd, shape='FR', n_cases=n_cases, des_phase_diff = 0*np.pi)
    # nd['phi_p'] = np.array([phys['phi_p_mat'][0]])*n_iterations

    nd['phi_p'] = np.array([phys['phi_p_mat'][0]])[0,:,:]
    counts = {'phase_count': 20, 'freq_count': 200}
    eq, results, E_tot = vcsel.solve_equilibria(nd, counts=counts, guesses=guesses)
    guesses = []

    if eq is None:
        print(f"No equilibria found for kappa = {final_kappa*1e-9:.2f} ns^-1")
        

    nd['phi_p'] = phys['phi_p_mat'][0]



    if eq is not None:
        phys['injection'] = True

        injection_array = np.zeros(N_lasers)
        center_idx = (N_lasers-1) // 2
        injection_array[center_idx] = 1
        phys['injection_topology'] = injection_array

        phys['injected_strength'] = nd['sbar']  # baseline amplitude

        tmp_stable = []
        N = 30
        n_eigenvalues = N*3*N_lasers - 1
        # print(len(eqs), n_eigenvalues)
        tmp_stable = Parallel(n_jobs=-1)(
            delayed(vcsel.compute_stability)(eq_pt, nd, N=N, newton_maxit=10000, threshold=1e-10, sparse=phys['sparse'], spectral_shift=0.01+0.01j, n_eigenvalues=n_eigenvalues)
            for eq_pt in results
        )
        tmp_stable = [result[0] for result in tmp_stable]
        if tmp_stable.count(1) > 0:
            stable_indices = np.where(np.array(tmp_stable) == 1.0)[0]
            # Initialize injection arrays
            phys['kappa_injection'] = np.zeros((n_cases, len(time_arr)))
            phys['injected_frequency'] = np.zeros(len(time_arr))
            
            # Create Gaussian peaks for each stable equilibrium
            stable_indices = stable_indices[np.argsort(E_tot[stable_indices])]
            for peak_idx, stable_idx in enumerate(stable_indices):
                eq = results[stable_idx]
                
                # Target setpoints from equilibrium
                phi_diff_target = eq[-2]
                omega_target = eq[-1]/(2*np.pi*1e9*tau_p)
                
                # Peak time for this equilibrium (separated by 100*tau)
                current_peak_time = peak_time + peak_idx * 300 * tau
                
                # Add Gaussian peak centered at current_peak_time
                gaussian_peak = kappa_inj_amp[:,None] * np.exp(-((time_arr - current_peak_time) ** 2) / (2 * kappa_inj_width ** 2))
                phys['kappa_injection'] += gaussian_peak
                
                # Set omega_target for the time region starting at 50tau + peak_idx*100tau
                jump_time = 100 * tau + peak_idx * 300 * tau
                jump_end_time = jump_time +300 * tau
                time_mask = (time_arr >= jump_time) & (time_arr < jump_end_time)
                phys['injected_frequency'][time_mask] = omega_target
                # phys['injected_frequency'] = omega_target * np.ones(len(time_arr))
            
            # Use the last stable equilibrium for phase target
            if len(stable_indices) > 0:
                phys['injected_phase_diff'] = 0.0

            kappa = kappa_inj_amp_peak   # ns^-1 → s^-1
            g0_si = g0                   # ns^-1 → s^-1

            P_inj = hbar * omega0 * phys['kappa_injection'] * nd['sbar'] / (g0_si * tau_n)

            injection_power_uW = P_inj * 1e6  # Convert to microwatts


        nd['phi_p'] = phys['phi_p_mat']

        


    
        phys['kappa_c_mat'] = kappa_arr
        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()

        nd['injected_phase_diff'] = 0.0#np.linspace(0,2*np.pi,n_cases)
        # 



        t, y, freqs = vcsel.integrate(history, nd=nd, progress=True, theta=0.5, max_iter=1, smooth_freqs=True)


        # intensities S[i,:]
        S = np.abs(y[:, S_idx, :])

        # phases phi[i,:]
        phi = y[:, phi_idx, :]

        # instantaneous freq derivatives dphi[i,:]
        dphi = freqs * 1e-9/(2*np.pi*tau_p)
        dphi[:, :, :2*delay_steps] = freq_history

        # Insert previous values for the initial delay window
        # for i in range(N_lasers):
        #     dphi[i, :2*delay_steps] = prev_dphi[i]

        # phase differences across all laser pairs (same approach as simple_example)
        delta_phi_all = phi[:, None, :, :] - phi[:, :, None, :]
        cos_pd_all = np.cos(delta_phi_all)
        cos_pd_mean = np.mean(cos_pd_all, axis=0)
        cos_pd_std = np.std(cos_pd_all, axis=0)

        # ---------------------------------------------------------
        # ---------               PLOTTING               ----------
        # ---------------------------------------------------------

        # clear_output(wait=True)
        fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)


        time_plot = time_arr[:-1:plot_stride]*1e6

        if eq is not None:
            axs[0].plot(time_plot, phys['injected_frequency'][:-1:plot_stride], 'k--', linewidth=2, label=r'$\dot{\phi}_{inj}$', alpha=0.5, zorder=10)
            axs[0].legend(loc='upper right', fontsize=18)

        # --- dphi for each laser ---

        for i in range(N_lasers):
            style = '--' if i % 2 else '-'
            mean_dphi = np.mean(dphi[:,i, :-1:plot_stride], axis=0)
            std_dphi = np.std(dphi[:,i, :-1:plot_stride], axis=0)
            axs[0].plot(time_plot, mean_dphi, style, linewidth=2, label=fr'$\dot{{\phi}}_{i+1}$')
            axs[0].fill_between(time_plot, mean_dphi - std_dphi, mean_dphi + std_dphi, alpha=0.15)

        if N_lasers <= 6:
            axs[0].legend(loc='upper right', fontsize=18)
        axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
        # dphi_min = np.min(dphi[:, :, :-1])
        # dphi_max = np.max(dphi[:, :, :-1])
        # dphi_range = dphi_max - dphi_min
        # axs[0].set_ylim(dphi_min - 0.1 * dphi_range, dphi_max + 0.1 * dphi_range)
        axs[0].set_ylim(-5,5) 
        axs[0].grid(True, alpha=0.2)
        axs[0].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
        axs[0].tick_params(axis='both', which='major', labelsize=18)


        kappa_ratio = kappa_inj_amp[-1] / final_kappa if final_kappa != 0 else 0
        axs[0].set_title(
        rf'$\kappa_c = {final_kappa*1e-9:.2f}\,\mathrm{{ns}}^{{-1}},\ \kappa_{{\mathrm{{ratio}}}} = {kappa_ratio:.2f}$',
        fontsize=24,
        pad=20
        )

        # --- nearest-neighbor phase differences ---
        for i in range(N_lasers-1):
            mean_cos = cos_pd_mean[i, i + 1, :-1:plot_stride]
            std_cos = cos_pd_std[i, i + 1, :-1:plot_stride]
            # no smoothing for cos(Δφ)
            axs[1].plot(time_plot, mean_cos, linewidth=2, label=fr'$\cos(\phi_{i+1}-\phi_{i+2})$')
            axs[1].fill_between(time_plot, mean_cos - std_cos, mean_cos + std_cos, alpha=0.15)

        axs[1].set_ylim(-1.1,1.1)
        axs[1].grid(True, alpha=0.2)
        axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
        axs[1].set_xlabel('Time ($\mu$s)', fontsize=22)
        axs[1].set_ylabel(r'$\cos(\Delta\phi)$', fontsize=22)
        axs[1].tick_params(axis='both', which='major', labelsize=18)
        if N_lasers <= 6:
            axs[1].legend(loc='upper left', fontsize=18)

        # --- total field ---
        E = np.sqrt(S) * np.exp(1j*phi)
        time_plot_full = time_arr[::plot_stride] * 1e6
        intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
        E_tot_mean = np.mean(np.abs(E.sum(axis=1))**2, axis=0)[::plot_stride] * intensity_to_mW
        E_tot_std = np.std(np.abs(E.sum(axis=1))**2, axis=0)[::plot_stride] * intensity_to_mW

        axs[2].plot(time_plot_full, E_tot_mean, linewidth=2, alpha=0.5, label=r'$P_{\rm tot}$')
        axs[2].fill_between(time_plot_full, E_tot_mean - E_tot_std, E_tot_mean + E_tot_std, alpha=0.15)

        for i in range(N_lasers):
            E_i_mean = np.mean(np.abs(E[:, i, :]**2), axis=0)[::plot_stride] * intensity_to_mW
            E_i_std = np.std(np.abs(E[:, i, :]**2), axis=0)[::plot_stride] * intensity_to_mW
            axs[2].plot(time_plot_full, E_i_mean, linewidth=1.5, label=f'$P_{i+1}$')
            axs[2].fill_between(time_plot_full, E_i_mean - E_i_std, E_i_mean + E_i_std, alpha=0.15)

        

        axs[2].set_xlabel('Time ($\mu$s)', fontsize=22)
        axs[2].set_ylabel('Power (mW)', fontsize=22)
        axs[2].grid(True, alpha=0.2)
        axs[2].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
        axs[2].tick_params(axis='both', which='major', labelsize=18)
        if N_lasers <= 6:
            axs[2].legend(loc='upper left', fontsize=16, ncol=2)

        ax2 = axs[2].twinx()
        P_c = hbar * omega0 * kappa_arr[-len(time_plot_full):, 0,0] * nd['sbar'] / (g0 * tau_n) * 1e6
        # ax2.plot(time_plot_full, kappa_arr[-len(time_plot_full):, 0,0]*1e-9, 'b--', alpha=0.5, linewidth=2)
        ax2.plot(time_plot_full, P_c, 'b--', alpha=0.5, linewidth=2)
        if phys['injection']:
            for i in range(n_cases):
                ax2.plot(time_plot_full, injection_power_uW[i][::plot_stride], 'b--', alpha=0.5, linewidth=2)
        # ax2.set_ylabel('kappa ($ns^{-1}$)', color='blue', fontsize=24)
        ax2.set_ylabel('Injection Power ($\\mu$W)', color='blue', fontsize=24)
        # ax2.set_ylim(0, 40e9*1e-9)
        # ax2.set_ylim(0, 1000)
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=20)

        plt.tight_layout()
        # plt.savefig(f'../injection_tests/injection_steering_plots/{k}.png')
        # plt.savefig(f'./injection_tests/detuning_test/injection_time_series_kappa{final_kappa/1e9:.1f}_detuning{detuning:.1f}ghz.png', dpi=300)
        plt.show()
        plt.close(fig)
        break






#%%

    
    

    # history = y[:,:,-2*delay_steps:].copy()
    # freq_history = dphi[:, -2*delay_steps:].copy()
    # prev_dphi1 = dphi1[-2*delay_steps:].copy()
    # prev_dphi2 = dphi2[-2*delay_steps:].copy()



    # prev_dphi1 = np.zeros(2*delay_steps)
    # prev_dphi2 = np.ones(2*delay_steps) * delta /(2*np.pi*1e9)

    
    # # take second half of the series
    # tail = np.mean(np.abs(E_tot)**2, axis=0)[-int(len(E_tot[0,:]) / 2):]

    # order = max(1, int(len(tail) * 0.01))
 
    # if len(tail) >= 3:
    #     local_max_idx = argrelextrema(tail, np.greater, order=order)[0]
    #     local_min_idx = argrelextrema(tail, np.less, order=order)[0]
    # else:
    #     local_max_idx = np.array([], dtype=int)
    #     local_min_idx = np.array([], dtype=int)

    # # remove duplicate values (unique extrema)
    # max_vals, unique_max_idx = np.unique(np.round(tail[local_max_idx]/1e-2)*1e-2, return_index=True)
    # min_vals, unique_min_idx = np.unique(np.round(tail[local_min_idx]/1e-2)*1e-2, return_index=True)

    # # recover original indices
    # local_max_idx = local_max_idx[unique_max_idx]
    # local_min_idx = local_min_idx[unique_min_idx]

    # # convert to indices relative to the full E_tot array
    # offset = len(E_tot) - len(tail)
    # max_idx_global = (local_max_idx + offset).tolist()
    # min_idx_global = (local_min_idx + offset).tolist()

    
    # # for val in max_vals:
    # #     extrema.append((kappa_c[k]*1e-9, val))
    # # for val in min_vals:
    # #     extrema.append((kappa_c[k]*1e-9, val))

    # vals = []
    # for val in max_vals:
    #     vals.append(val)
    # for val in min_vals:
    #     vals.append(val)

    # extrema.append((kappa_c[k]*1e-9, np.mean(vals)))


    


    


#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = cm.get_cmap('jet')
forward_extrema_array = np.array(extrema)
# backward_extrema_array = np.array(extrema)


fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
plt.scatter(forward_extrema_array[:, 0], forward_extrema_array[:, 1], color='red', label='Forward', s=1)
# plt.scatter(backward_extrema_array[:, 0], backward_extrema_array[:, 1], color='blue', label='Backward', s=1)

ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', fontsize=20)
ax.set_ylabel(r'$|E_1 + E_2|^2$', fontsize=20)

# increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(alpha=0.25) 

# colorbar for order parameter with larger font
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Order parameter', fontsize=20)
cbar.ax.tick_params(labelsize=20)


plt.xlim(0,20.5)
plt.ylim(-0.5,20)
plt.legend(fontsize=16, loc='upper left')
 
plt.tight_layout()
plt.savefig('./forward_extrema_FR_cont_gaussian_0.2ghz.png', transparent=True)
plt.show()
