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

detuning = 4.0  # detuning (GHz)
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s


dt = 0.5*tau_p # 1 ps
Tmax = 2e-7


steps = int(Tmax / dt)

time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
segment_len = int(steps/2)
segment_start = int(steps/2)

resolution = 20
N_lasers = 2
coupling_scheme = 'ATA'  # 'ATA', 'NN' or 'RANDOM'
ramp_start = 2

kappa_c = np.linspace(0e9,20e9,resolution)



phi_p_vals = np.linspace(0,2*np.pi,resolution)


phys = {
    'tau_p': tau_p,
    'tau_n': tau_n,
    'g0': g0,
    'N0': N0,
    'N_bar': N0 + 1/(g0*tau_p),
    's': s,
    'beta': beta,
    'kappa_c_mat': None,
    'phi_p_mat': np.ones(shape=(resolution,N_lasers,N_lasers))*phi_p_vals[:,None,None],
    'I': I,
    'q': q,
    'alpha': alpha,
    'delta': np.sort(np.concatenate([[0], [delta]])),
    'coupling': coupling,     
    'self_feedback': self_feedback, 
    'noise_amplitude': noise_amplitude,
    'dt': dt,
    'Tmax': Tmax,
    'tau': tau,
    'N_lasers': N_lasers
}



# t, y, freqs = vcsel.integrate(history, nd=nd, progress=True)



ramp_start = 10
ramp_shape = 100


kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c[1], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme)

phys['kappa_c_mat'] = kappa_arr

vcsel = VCSEL(phys)
nd = vcsel.scale_params()



n_cases = len(nd['phi_p'])


history, freq_hist,_ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)

#%%
 
reverse = False

order_param = np.zeros(shape=(resolution, resolution))
intensity = np.zeros(shape=(resolution, resolution))

for k in range(0, len(kappa_c)):
    if reverse:
        k = len(kappa_c)-1 - k
    # Slice the ramp for this segment
    if k > 0:
        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=kappa_c[k-1], kappa_final=kappa_c[k], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme)
    
    phys['kappa_c_mat'] = kappa_arr
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    t, y_scaled, freqs = vcsel.integrate(history, nd=nd, progress=True )

    y = y_scaled.copy()#vcsel.invert_scaling(y_scaled.copy(), phys)
    S = []
    phi = []
    dphi = []

    for i in range(N_lasers):
        # Your original indexing used:
        #   Laser 1 → y[:, 1, :]
        #   Laser 2 → y[:, 4, :]
        #   Laser 1 phase → y[:, 2, :]
        #   Laser 2 phase → y[:, 5, :]
        #
        # Pattern is (per laser): intensity index = 3*i + 1 , phase index = 3*i + 2
        S.append(np.abs(y[:, 3*i + 1, :]))
        phi.append(y[:, 3*i + 2, :])
        dphi.append(freqs[:, i, :] * 1e-9/(2*np.pi*tau_p))

    S = np.stack(S, axis=1)        # shape: (time, N, traj)
    phi = np.stack(phi, axis=1)
    dphi = np.stack(dphi, axis=1)

    # Pairwise phase differences (using laser 0 as reference)
    phase_diff = np.unwrap(phi - phi[:, 0:1, :], axis=0)

    # Order parameter now works for N lasers
    order_param[k,:] = vcsel.order_parameter(y[:,:,-int(len(t)/2):])

    # Build fields E_i
    E_list = []
    for i in range(N_lasers):
        Ei = np.sqrt(S[:, i, :]) * (np.cos(phi[:, i, :]) + 1j*np.sin(phi[:, i, :]))
        E_list.append(Ei)

    E_all = np.stack(E_list, axis=1)     # shape: (time, N, traj)

    # Total intensity across lasers
    E_tot = np.abs(np.sum(E_all, axis=1))**2
    intensity[k,:] = np.mean(E_tot[:, -int(len(t)/2):], axis=1)


    if k % 1 == 0:

        clear_output(wait=True)

        fig = plt.figure(figsize=(8, 6), dpi=200)
        im = plt.imshow(order_param, aspect='auto', origin='lower',
                        extent=[phi_p_vals[0]/(np.pi), phi_p_vals[-1]/(np.pi),
                                kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
                        cmap='jet') 

        cbar = plt.colorbar(im, pad=0.02)
        cbar.set_label('Order Parameter', fontsize=24, labelpad=16)
        cbar.ax.tick_params(labelsize=22)    
        plt.clim(0,1)

        plt.xlabel("Coupling Phase $\phi_p/\pi$", fontsize=24, labelpad=14)
        plt.ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=24, labelpad=14)
        plt.title(rf"$\delta=${detuning:.1f} GHz", fontsize=28, pad=20)
        plt.yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6), fontsize=22)
        plt.xticks(np.linspace(phi_p_vals[0]/(np.pi), phi_p_vals[-1]/(np.pi), 5), fontsize=22)
        plt.xlim(phi_p_vals[0]/(np.pi), phi_p_vals[-1]/(np.pi))
        plt.ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
        plt.tick_params(axis='both', labelsize=22)

        plt.tight_layout()
        # plt.savefig(f'./order_parameter/order_parameter_continuation_delta4.0_alpha{alpha:.1f}_{N_lasers}laser_test.png')
        plt.show()
        plt.close(fig)

    history = y_scaled[:,:,-2*delay_steps:]










