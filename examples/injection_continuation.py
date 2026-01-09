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
matplotlib.use('Agg')

from IPython.display import clear_output
import gc
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelextrema

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
coupling_scheme = 'DECAYED'
dx = 0.7
n_cases = 1

detuning = 0.5 # detuning (GHz) 
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
phi_p = 0#np.pi

dt = .5*tau_p# 1 ps
Tmax = 2e-7
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
segment_len = int(steps/2)
segment_start = int(steps/2)

n_kappa = 500
ramp_start = 2
ramp_shape = 10

kappa_max = 20e9

kappa_c = np.linspace(0e9,20e9,n_kappa)

kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c[0], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)




n_iterations = 1


phys = {
    'tau_p': tau_p,
    'tau_n': tau_n,
    'g0': g0,
    'N0': N0,
    'N_bar': N0 + 1/(g0*tau_p),
    's': s,
    'beta': beta,
    'kappa_c_mat': None,
    'phi_p_mat': np.ones(shape=(N_lasers,N_lasers))*np.pi,
    'I': I,
    'q': q,
    'alpha': alpha,
    'delta': np.sort(np.concatenate([delta*np.linspace(-1,1,N_lasers)])),
    'coupling': coupling,     
    'self_feedback': self_feedback, 
    'noise_amplitude': noise_amplitude,
    'dt': dt,
    'Tmax': Tmax,
    'tau': tau,
    'N_lasers': N_lasers
}

phys['kappa_c_mat'] = kappa_arr[-1,:,:]



phys['injection'] = False


# Gaussian kappa injection centered at peak_time with controllable width
kappa_inj_width = 10 * tau          # width (s) — change this to control the Gaussian spread
peak_time = 50*tau                     # center (s), e.g. peak = 3*tau above
kappa_inj_amp = 30e9


%matplotlib inline
extrema = []
S_idx   = [3*i + 1 for i in range(N_lasers)]
phi_idx = [3*i + 2 for i in range(N_lasers)]
# --- Loop over segments of kappa --- 
for k in range(0, len(kappa_c)-1): 

    

    # Slice the ramp for this segment
    if k > 0:
        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=kappa_c[k-1], kappa_final=kappa_c[k], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)

    phys['kappa_c_mat'] = kappa_arr[-1,:,:]
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    nd['phi_p'] = np.array(phys['phi_p_mat'])
    if k ==0:
        history, freq_history, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)
    # eq_history, freq_hist = vcsel.generate_history(nd, shape='FR', n_cases=n_cases, des_phase_diff = 0*np.pi)
    # nd['phi_p'] = np.array([phys['phi_p_mat'][0]])*n_iterations

    eq, results = vcsel.solve_equilibria(nd)
    

    if eq is not None:
        phys['injection'] = True
        phys['injected_strength'] = nd['sbar']  # baseline amplitude

        # Target setpoints from equilibrium
        phi_diff_target = eq[-2]
        omega_target = eq[-1]/(2*np.pi*1e9*tau_p)

        # Proportional control updates
        phys['injected_phase_diff'] = np.random.uniform(0,2*np.pi)#-phi_diff_target 
        



        phys['injected_frequency'] = omega_target * np.ones_like(time_arr)



        # prev_dphi1[-1] + (omega_target - prev_dphi1[-1])** np.exp(-((time_arr - peak_time) ** 2) / (2 * kappa_inj_width ** 2))
        

        phys['kappa_injection'] = kappa_inj_amp * np.exp(-((time_arr - peak_time) ** 2) / (2 * kappa_inj_width ** 2))


   
    
    
    phys['kappa_c_mat'] = kappa_arr
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    nd['injected_phase_diff'] = np.random.uniform(0,2*np.pi)
    # 



    t, y, freqs = vcsel.integrate(history, nd=nd, progress=True )


    # intensities S[i,:]
    S = np.abs(y[0, S_idx, :])

    # phases phi[i,:]
    phi = y[0, phi_idx, :]

    # instantaneous freq derivatives dphi[i,:]
    dphi = freqs[0, :, :] * 1e-9/(2*np.pi*tau_p)
    dphi[:, :2*delay_steps] = freq_history

    # Insert previous values for the initial delay window
    # for i in range(N_lasers):
    #     dphi[i, :2*delay_steps] = prev_dphi[i]

    # nearest-neighbor phase differences
    phase_diff = np.unwrap(phi[:-1, :] - phi[1:, :], axis=1)  # shape (N_lasers-1, time)

    # ---------------------------------------------------------
    # ---------               PLOTTING               ----------
    # ---------------------------------------------------------
    clear_output(wait=True)
    fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)


    if phys['injection']:
        axs[0].plot(time_plot, phys['injected_frequency'][:-1], 'k--', linewidth=2, label=r'$\dot{\phi}_{inj}$', alpha=0.5)
        axs[0].legend(loc='upper right', fontsize=18)

    # --- dphi for each laser ---
    time_plot = time_arr[:-1]*1e6
    for i in range(N_lasers):
        style = '--' if i % 2 else '-'
        axs[0].plot(time_plot, dphi[i, :-1], style, linewidth=2, label=fr'$\dot{{\phi}}_{i+1}$')

    if N_lasers <= 6:
        axs[0].legend(loc='upper right', fontsize=18)
    axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
    axs[0].set_ylim(-np.max(np.abs(dphi))*1.2, np.max(np.abs(dphi))*1.2)
    axs[0].set_ylim(-10,1)
    axs[0].grid(True, alpha=0.2)
    axs[0].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
    axs[0].tick_params(axis='both', which='major', labelsize=18)
    axs[0].set_title(f'kappa_c = {kappa_c[k+1]*1e-9:.2f} ns$^{{-1}}$', fontsize=24)

    

    # --- nearest-neighbor phase differences ---
    for i in range(N_lasers-1):
        axs[1].plot(time_plot, np.cos(phase_diff[i, :-1]), linewidth=2, label=fr'$\cos(\phi_{i+1}-\phi_{i+2})$')

    axs[1].set_ylim(-1.1,1.1)
    axs[1].grid(True, alpha=0.2)
    axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
    axs[1].set_xlabel('Time ($\mu$s)', fontsize=22)
    axs[1].set_ylabel(r'$\cos(\Delta\phi)$', fontsize=22)
    axs[1].tick_params(axis='both', which='major', labelsize=18)
    if N_lasers <= 6:
        axs[1].legend(loc='upper left', fontsize=18)

    # --- total field ---
    # build complex fields for each laser
    E = np.sqrt(S) * np.exp(1j*phi)     # shape (N_lasers, time)
    time_plot_full = time_arr[:] * 1e6
    E_tot = np.abs(E.sum(axis=0))**2

    axs[2].plot(time_plot_full, E_tot, linewidth=2, alpha=0.5, label='$|E_{tot}|^2$')
    for i in range(N_lasers):
        axs[2].plot(time_plot_full, np.abs(E[i])**2, linewidth=1.5, label=f'$|E_{i+1}|^2$')

    axs[2].set_xlabel('Time ($\mu$s)', fontsize=22)
    axs[2].grid(True, alpha=0.2)
    axs[2].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
    
    
    axs[2].set_ylim(-5, 100)

    axs[2].tick_params(axis='both', which='major', labelsize=18)
    if N_lasers <= 6:
        axs[2].legend(loc='upper left', fontsize=16, ncol=2)

    ax2 = axs[2].twinx()
    ax2.plot(time_plot_full, kappa_arr[-len(time_plot_full):, 0,0]*1e-9, 'b--', alpha=0.5, linewidth=2)
    if phys['injection']:
        ax2.plot(time_plot_full, phys['kappa_injection']*1e-9, 'k--', alpha=0.5, linewidth=2)
    ax2.set_ylabel('kappa ($ns^{-1}$)', color='blue', fontsize=24)
    ax2.set_ylim(0, 30e9*1e-9)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=20)

    plt.tight_layout()
    # plt.savefig(f'./{N_lasers}_laser_injection_continuation/{k}.png')
    plt.show()
    plt.close(fig)
    

    
    

    history = y[:,:,-2*delay_steps:].copy()
    freq_history = dphi[:, -2*delay_steps:].copy()
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

