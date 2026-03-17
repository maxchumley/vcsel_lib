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
from matplotlib.lines import Line2D
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

N_lasers = 3
coupling_scheme = 'ATA'
dx = 0.7

detuning = 3.0# detuning (GHz) 

lam = 910e-9
omega0 = 2*np.pi*c/lam



results = None


for detuning in np.linspace(4,5,50):
    delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
    # Create evenly distributed detuning for both even and odd N_lasers
    delta_dist = np.sort(np.concatenate([delta/2 * np.linspace(-1, 1, N_lasers)]))




    phi_p = 0#np.pi

    dt = .01*tau_p# 1 ps
    Tmax = 3e-9
    steps = int(Tmax / dt)
    time_arr = np.linspace(0, Tmax, steps)
    delay_steps = int(tau / dt)
    segment_len = int(steps/2)
    segment_start = int(steps/2)

    n_kappa = 1
    ramp_start = 0
    ramp_shape = 0.00001

    kappa_max = 20e9

    kappa_c = np.linspace(5e9,5e9,n_kappa)

    final_kappa = 20e9#np.linspace(0e9,20e9,500)[448]

    kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=final_kappa, kappa_final=final_kappa, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)





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
        'sparse': False
    }

    phys['kappa_c_mat'] = kappa_arr#[-1,:,:]



    phys['injection'] = False
    peak_time = 200*tau                     # center (s), e.g. peak = 3*tau above
    #30e9


    extrema = []
    S_idx   = [3*i + 1 for i in range(N_lasers)]
    phi_idx = [3*i + 2 for i in range(N_lasers)]
    # --- Loop over segments of kappa --- 
    # max_tau_inj_width = 200
    n_iterations = 1

    inj_phases = np.linspace(0,2*np.pi,n_cases)



    for k in range(0, n_kappa):

        kappa_inj_width = 5 * tau          # width (s) — change this to control the Gaussian spread 
        kappa_inj_amp_peak = 3*final_kappa
        kappa_inj_amp = np.linspace(kappa_inj_amp_peak, kappa_inj_amp_peak, n_cases)

        # Slice the ramp for this segment
        if k > 0:
            kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=final_kappa, kappa_final=final_kappa, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)

        # phys['kappa_c_mat'] = kappa_arr[-1,:,:]
        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()


        if k == 0:
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

            counts = {'phase_count': 5, 'freq_count': 500}
            history, freq_history, eq, results = vcsel.generate_history(nd, shape='EQ', n_cases=1, counts=counts)

        if eq is not None:
            phys['injection'] = False

            injection_array = np.zeros(N_lasers)
            center_idx = (N_lasers-1) // 2
            injection_array[center_idx] = 1
            phys['injection_topology'] = injection_array

            phys['injected_strength'] = nd['sbar']  # baseline amplitude

            tmp_stable = []
            N = 30
            n_eigenvalues = N*3*N_lasers - 1
            tmp_stable = Parallel(n_jobs=-1)(
                delayed(vcsel.compute_stability)(eq_pt, nd, N=N, newton_maxit=10000, threshold=1e-10, sparse=phys['sparse'], spectral_shift=0.01+0.01j, n_eigenvalues=n_eigenvalues)
                for eq_pt in results
            )


            tmp_stable = [result[0] for result in tmp_stable]

        
            phys['kappa_c_mat'] = kappa_arr
            phys['phi_p_mat'] = np.ones(shape=(n_cases,N_lasers,N_lasers))*phi_p_vals[:,None,None]
            
            vcsel = VCSEL(phys)
            nd = vcsel.scale_params()
            nd['injected_phase_diff'] = 0.0#np.linspace(0,2*np.pi,n_cases)


            length = 2*delay_steps
                
            n_cases = results.shape[0]
                

            
            
            nd['phi_p'] = np.ones(shape=(n_cases,N_lasers,N_lasers))*phi_p_vals[:,None,None]
            t, y, freqs = vcsel.integrate(history, nd=nd, progress=True, theta=0.5, max_iter=1)

            # intensities S[i,:]
            S = np.abs(y[:, S_idx, :])

            # phases phi[i,:]
            phi = y[:, phi_idx, :]

            # instantaneous freq derivatives dphi[i,:]
            dphi = freqs * 1e-9/(2*np.pi*tau_p)
            dphi[:, :, :2*delay_steps] = freq_history

            # nearest-neighbor phase differences
            phase_diff = np.unwrap(phi[:, :-1, :] - phi[:, 1:, :], axis=1)  # shape (N_lasers-1, time)

            # ---------------------------------------------------------
            # ---------               PLOTTING               ----------
            # ---------------------------------------------------------

            # clear_output(wait=True)

        else:
            print(f"No equilibria found for kappa={final_kappa*1e-9:.2f} ns^-1.")
            continue




        


        
    if eq is not None:
        fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)

        time_plot = time_arr[:-1]*1e6

        # --- dphi for each laser ---
        style =  '-'
        colors = ['blue' if tmp_stable[traj] else 'red' for traj in range(dphi.shape[0])]
        for traj in range(dphi.shape[0]):
            zorder = 10 if colors[traj] == 'blue' else 1
            axs[0].plot(time_plot, dphi[traj, 0, :-1], style, linewidth=1, alpha=0.7, color=colors[traj], zorder=zorder)
        axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
        axs[0].grid(True, alpha=0.2)
        axs[0].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
        axs[0].tick_params(axis='both', which='major', labelsize=18)
        axs[0].set_ylim(-12,5)

        kappa_ratio = kappa_inj_amp[-1] / final_kappa if final_kappa != 0 else 0
        axs[0].set_title(
            rf'$\kappa_c = {final_kappa*1e-9:.2f}\,\mathrm{{ns}}^{{-1}},\ \kappa_{{\mathrm{{ratio}}}} = {kappa_ratio:.2f}$',
            fontsize=24,
            pad=20
        )

        # --- nearest-neighbor phase differences ---
        for i in range(N_lasers-1):
            for traj in range(phase_diff.shape[0]):
                zorder = 10 if colors[traj] == 'blue' else 1
                axs[1].plot(time_plot, np.cos(phase_diff[traj, i, :-1]), linewidth=1, alpha=0.7, color=colors[traj], zorder=zorder)
        axs[1].set_ylim(-1.1,1.1)
        axs[1].grid(True, alpha=0.2)
        axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
        axs[1].set_xlabel('Time ($\mu$s)', fontsize=22)
        axs[1].set_ylabel(r'$\cos(\Delta\phi)$', fontsize=22)
        axs[1].tick_params(axis='both', which='major', labelsize=18)

        # --- total field ---
        E = np.sqrt(S) * np.exp(1j*phi)
        time_plot_full = time_arr[:] * 1e6
        for traj in range(E.shape[0]):
            zorder = 10 if colors[traj] == 'blue' else 1
            axs[2].plot(time_plot_full, np.abs(E[traj].sum(axis=0))**2, linewidth=1, alpha=0.7, color=colors[traj], label='$|E_{tot}|^2$' if traj == 0 else None, zorder=zorder)
        axs[2].set_xlabel('Time ($\mu$s)', fontsize=22)
        axs[2].grid(True, alpha=0.2)
        axs[2].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
        # axs[2].set_ylim(-5, 40)
        axs[2].tick_params(axis='both', which='major', labelsize=18)

        # Add custom legend for stable/unstable
        legend_elements = [Line2D([0], [0], color='red', lw=2, label='Stable'),
                        Line2D([0], [0], color='blue', lw=2, label='Unstable')]
        axs[2].legend(handles=legend_elements, loc='upper left', fontsize=16)

        plt.tight_layout()
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
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
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

