#%%
# Example usage
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
from IPython.display import clear_output
import gc
from scipy.ndimage import uniform_filter1d
from IPython.display import clear_output
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

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


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

detuning = 0.5  # detuning (GHz)
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s


dt = 0.5*tau_p # 1 ps
Tmax = 5e-8

steps = int(Tmax / dt)

time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
segment_len = int(steps/2)
segment_start = int(steps/2)

resolution = 1000
N_lasers = 3
coupling_scheme = 'DECAYED'  # 'ATA', 'NN' or 'RANDOM'
dx=0.7
ramp_start = 2


kappa_max = 20e9
kappa_c = np.linspace(0e9,20e9,resolution)



phi_p_vals = np.array([np.pi])#np.linspace(0,2*np.pi,resolution)

n_iterations = 100

phys = {
    'tau_p': tau_p,
    'tau_n': tau_n,
    'g0': g0,
    'N0': N0,
    'N_bar': N0 + 1/(g0*tau_p),
    's': s,
    'beta': beta,
    'kappa_c_mat': None,
    'phi_p_mat': np.ones(shape=(1,N_lasers,N_lasers))*phi_p_vals[:,None,None],
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



# t, y, freqs = vcsel.integrate(history, nd=nd, progress=True)



ramp_start = 1
ramp_shape = 25


kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c[1], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=True, dx=dx)

phys['kappa_c_mat'] = kappa_arr

vcsel = VCSEL(phys)
nd = vcsel.scale_params()



n_cases = len(nd['phi_p'])


history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)








#%%
extrema = []

# indices for each laser inside your state vector
# assuming your state layout is: [ ... S_i , phi_i ... ] per laser
# If your state format differs, adjust these two lists.
S_idx   = [3*i + 1 for i in range(N_lasers)]
phi_idx = [3*i + 2 for i in range(N_lasers)]

reverse = False

# --- Loop over segments of kappa ---
for k in range(len(kappa_c)-1):

    if reverse:
        k = len(kappa_c)-1 - k

    # Slice the ramp for this segment
    # if k > 0:
    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=kappa_c[k],
        kappa_final=kappa_c[k+1],
        N_lasers=N_lasers,
        ramp_start=ramp_start,
        ramp_shape=ramp_shape,
        tau=tau,
        scheme=coupling_scheme,
        dx=dx,
        plot=False
    )
    
    phys['kappa_c_mat'] = kappa_arr

    vcsel = VCSEL(phys)
    nd    = vcsel.scale_params()
    if k == 0:
        prev_dphi = np.ones((N_lasers, 2*delay_steps))*phys['delta'].reshape((N_lasers,1)) /(2*np.pi*1e9)
        history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)
    t, y_scaled, freqs = vcsel.integrate(history, nd=nd, progress=True)

    y = y_scaled.copy()

    # ---------------------------------------------------------
    # ---------    EXTRACT ALL FIELDS & PHASES     ------------
    # ---------------------------------------------------------

    # intensities S[i,:]
    S = np.abs(y[0, S_idx, :])

    # phases phi[i,:]
    phi = y[0, phi_idx, :]

    # instantaneous freq derivatives dphi[i,:]
    dphi = freqs[0, :, :] * 1e-9/(2*np.pi*tau_p)

    # Insert previous values for the initial delay window
    for i in range(N_lasers):
        dphi[i, :2*delay_steps] = prev_dphi[i]

    # nearest-neighbor phase differences
    phase_diff = np.unwrap(phi[:-1, :] - phi[1:, :], axis=1)  # shape (N_lasers-1, time)

    # ---------------------------------------------------------
    # ---------               PLOTTING               ----------
    # ---------------------------------------------------------
    clear_output(wait=True)
    fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)

    # --- dphi for each laser ---
    time_plot = time_arr[:-1]*1e6
    for i in range(N_lasers):
        style = '--' if i % 2 else '-'
        axs[0].plot(time_plot, dphi[i, :-1], style, linewidth=2, label=fr'$\dot{{\phi}}_{i+1}$')

    if N_lasers <= 6:
        axs[0].legend(loc='upper left', fontsize=18)
    axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
    axs[0].set_ylim(-np.max(np.abs(dphi))*1.2, np.max(np.abs(dphi))*1.2)
    axs[0].set_ylim(-5,5)
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
    
    
    axs[2].set_ylim(-5, 60)

    axs[2].tick_params(axis='both', which='major', labelsize=18)
    if N_lasers <= 6:
        axs[2].legend(loc='upper left', fontsize=16, ncol=2)

    ax2 = axs[2].twinx()
    ax2.plot(time_plot_full, kappa_arr[-len(time_plot_full):, 0,0]*1e-9, 'b--', alpha=0.5, linewidth=2)
    ax2.set_ylabel('kappa ($ns^{-1}$)', color='blue', fontsize=24)
    ax2.set_ylim(0, kappa_max*1e-9)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=20)

    plt.tight_layout()
    # plt.savefig(f'./{N_lasers}_laser_continuation/{k}.png')
    plt.show()
    plt.close(fig)

    # ---------------------------------------------------------
    # ---------            Update history            ----------
    # ---------------------------------------------------------

    history = y_scaled[:, :, -2*delay_steps:]
    prev_dphi = dphi[:, -2*delay_steps:]

    # ---------------------------------------------------------
    # ---------            Compute extrema           ----------
    # ---------------------------------------------------------

    tail = E_tot[-len(E_tot)//2:]
    order = max(1, int(len(tail)*0.01))

    if len(tail) >= 3:
        local_max_idx = argrelextrema(tail, np.greater, order=order)[0]
        local_min_idx = argrelextrema(tail, np.less, order=order)[0]
    else:
        local_max_idx = np.array([], dtype=int)
        local_min_idx = np.array([], dtype=int)

    max_vals, unique_max_idx = np.unique(
        np.round(tail[local_max_idx]/1e-2)*1e-2,
        return_index=True
    )
    min_vals, unique_min_idx = np.unique(
        np.round(tail[local_min_idx]/1e-2)*1e-2,
        return_index=True
    )

    offset = len(E_tot) - len(tail)
    for val in max_vals:
        extrema.append((kappa_c[k+1]*1e-9, val))
    for val in min_vals:
        extrema.append((kappa_c[k+1]*1e-9, val))

    # break

    


    


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
ax.set_ylabel(r'$|E_{\mathrm{tot}}|^2$', fontsize=20)

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
plt.ylim(-0.5,60)
plt.legend(fontsize=16, loc='upper left')
 
plt.tight_layout()
# plt.savefig('./backward_extrema_3_DECAYED.png', transparent=True)
plt.show()

