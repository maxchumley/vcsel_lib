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
noise_amplitude = 1.0

detuning = 0.5 # detuning (GHz)
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s


dt = 0.5*tau_p # 1 ps
Tmax = 1e-6


steps = int(Tmax / dt)

time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
segment_len = int(steps/2)
segment_start = int(steps/2)

resolution = 100
n_iterations = 1
N_lasers = 2
ramp_start = 2

kappa_c = np.linspace(0e9,30e9,resolution)



phi_p_vals = np.repeat(np.linspace(0,2*np.pi,resolution),n_iterations)



phys = {
    'tau_p': tau_p,
    'tau_n': tau_n,
    'g0': g0,
    'N0': N0,
    'N_bar': N0 + 1/(g0*tau_p),
    's': s,
    'beta': beta,
    'kappa_c_mat': None,
    'phi_p_mat': np.ones(shape=(resolution*n_iterations,N_lasers,N_lasers))*phi_p_vals[:,None,None],
    'I': I,
    'q': q,
    'alpha': alpha,
    'delta': np.sort(np.concatenate([[0], [1*delta]])),
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


kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c[1], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme='ATA')

phys['kappa_c_mat'] = kappa_arr

vcsel = VCSEL(phys)
nd = vcsel.scale_params()



n_cases = len(nd['phi_p'])


history, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)

#%%

prev_E1 = []
prev_E2 = []
prev_dphi1_list = []  
prev_dphi2_list = []


reverse = False 

order_param = np.ones(shape=(resolution, resolution))*0.0
locking_param = np.zeros(shape=(resolution, resolution))
 
n_detuning = 4
detuning_arr = np.array([0.5,2.0,4.0,6.0])#np.linspace(0.0,5.0,n_detuning)
critical_kappa_vals = np.zeros(shape=(resolution,n_detuning)) 

for di, detuning in enumerate(detuning_arr): 
    crit_threshold = 0.8
    order_param = np.ones(shape=(resolution, resolution))*0.0
    locking_param = np.zeros(shape=(resolution, resolution))
    kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c[1], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme='ATA')

    phys['kappa_c_mat'] = kappa_arr
    delta = detuning * 2 * np.pi * 1e9
    phys['delta'] = np.sort(np.concatenate([[0], [1*delta]]))
    vcsel = VCSEL(phys)
    
    nd = vcsel.scale_params()
    history, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)

    for k in range(0, len(kappa_c)):
        if reverse: 
            k = len(kappa_c)-1 - k
        # Slice the ramp for this segment
        if k > 0:
            kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=kappa_c[k-1], kappa_final=kappa_c[k], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme='ATA')
        
        phys['kappa_c_mat'] = kappa_arr
        delta = detuning * 2 * np.pi * 1e9
        phys['delta'] = np.sort(np.concatenate([[0], [1*delta]]))
        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()
        t, y, freqs = vcsel.integrate(history, nd=nd, progress=True )

        S = []
        phi = []
        dphi = []

        for i in range(N_lasers): 
            S.append(np.abs(y[:, 3*i + 1, :]))
            phi.append(y[:, 3*i + 2, :])
            dphi.append(freqs[:, i, :] * 1e-9/(2*np.pi*tau_p))

        S = np.stack(S, axis=1)        # shape: (time, N, traj)
        phi = np.stack(phi, axis=1)
        dphi = np.stack(dphi, axis=1)


        # Order parameter now works for N lasers
        order_param[k,:] = np.mean(vcsel.order_parameter(y[:,:,-int(len(t)/2):]).reshape(resolution, n_iterations),axis=1)

        # locking_param = 4*np.square(order_param-0.5)
        locking_param[k,:] = np.mean(2*np.abs(vcsel.order_parameter(y[:,:,-int(len(t)/2):]).reshape(resolution, n_iterations)-0.5), axis=1)

        
        for row in locking_param[:k,:]:
            for idx, row in enumerate(locking_param[:k, :]):
                mask = (row > crit_threshold) & (critical_kappa_vals[:,di] == 0)
                critical_kappa_vals[mask, di] = kappa_c[idx]
 

        if np.all(critical_kappa_vals[:,di] > 0):
            fig = plt.figure(figsize=(8,6), dpi=200)
            mean_vals = np.mean(critical_kappa_vals, axis=0) * 1e-9
            std_vals = np.std(critical_kappa_vals, axis=0) * 1e-9
            plt.plot(detuning_arr, mean_vals, 'r-', linewidth=2)
            plt.fill_between(detuning_arr, mean_vals - std_vals, mean_vals + std_vals, 
                   color='r', alpha=0.3)
            plt.xlabel("Detuning (GHz)", fontsize=24, labelpad=14)
            plt.ylabel(r"Critical Coupling $\kappa_c~(\mathrm{ns}^{-1})$", fontsize=24, labelpad=14)
            plt.title(rf"With Noise", fontsize=28, pad=20)
            plt.ylim(0,30)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.tight_layout()
            plt.savefig(f'./critical_coupling_scaling_with_noise.png')
            plt.show()
            plt.close(fig)
            break
            
            

        

        if k % 1 == 0:

            clear_output(wait=True)

            fig = plt.figure(figsize=(8, 6), dpi=200)
            # locking_param
            im = plt.imshow(locking_param, aspect='auto', origin='lower',
                            extent=[phi_p_vals[0]/(np.pi), phi_p_vals[-1]/(np.pi),
                                    kappa_c[0]*1e-9, kappa_c[-1]*1e-9],
                            cmap='jet') 

            cbar = plt.colorbar(im, pad=0.02)
            cbar.set_label('Locking Parameter', fontsize=24, labelpad=16)
            cbar.ax.tick_params(labelsize=22)    
            plt.clim(0,1)

            plt.xlabel("Coupling Phase $\phi_p/\pi$", fontsize=24, labelpad=14)
            plt.ylabel(r"$\kappa_c~(\mathrm{ns}^{-1})$", fontsize=24, labelpad=14)
            plt.title(rf"$\delta=${detuning:.2f} GHz", fontsize=28, pad=20)
            plt.yticks(np.linspace(kappa_c[0]*1e-9, kappa_c[-1]*1e-9, 6), fontsize=22)
            plt.xticks(np.linspace(phi_p_vals[0]/(np.pi), phi_p_vals[-1]/(np.pi), 5), fontsize=22)
            plt.xlim(phi_p_vals[0]/(np.pi), phi_p_vals[-1]/(np.pi))
            plt.ylim(kappa_c[0]*1e-9, kappa_c[-1]*1e-9)
            plt.tick_params(axis='both', labelsize=22)

            plt.tight_layout()
            plt.savefig(f'./coupling_scaling_noise/order_parameter_continuation_delta{detuning:.3f}.png')
            plt.show()
            plt.close(fig)
        
        

        history = y[:,:,-2*delay_steps:]
    

    
#%%

np.save('./coupling_scaling/mean_vals.npy', mean_vals)
np.save('./coupling_scaling/std_vals.npy', std_vals)