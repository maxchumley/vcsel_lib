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

detuning = 0.2# detuning (GHz) 


i = 1
n_cases = 100
running_avg_mean = np.zeros(shape=(n_cases,n_cases))
running_avg_std = np.zeros(shape=(n_cases,n_cases))

for detuning in np.linspace(0.0,5,11):
    delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
    # Create evenly distributed detuning for both even and odd N_lasers
    delta_dist = np.sort(np.concatenate([delta/2 * np.linspace(-1, 1, N_lasers)]))

    phi_p = 0#np.pi

    dt = 2*tau_p# 1 ps
    Tmax = 5e-7
    steps = int(Tmax / dt)
    time_arr = np.linspace(0, Tmax, steps)
    delay_steps = int(tau / dt)
    segment_len = int(steps/3)
    segment_start = int(2*steps/3)

    n_kappa = 1
    ramp_start = 10
    ramp_shape = 20

    kappa_max = 20e9

    kappa_c = np.linspace(5e9,5e9,n_kappa)

    # final_kappa = np.linspace(0e9,20e9,500)

    for final_kappa in np.linspace(0.001e9,20e9,11):

        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=final_kappa, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)


        

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

        phys['kappa_c_mat'] = kappa_arr[-1,:,:]




 
        phys['injection'] = False

        peak_time = 200*tau                     # center (s), e.g. peak = 3*tau above
        #30e9


        extrema = []
        S_idx   = [3*i + 1 for i in range(N_lasers)]
        phi_idx = [3*i + 2 for i in range(N_lasers)]
        # --- Loop over segments of kappa --- 
        # max_tau_inj_width = 200
        n_iterations = 100

        inj_phases = np.linspace(0,2*np.pi,n_cases)
        n_widths = n_cases
        width_factors = np.linspace(1,10,n_widths)


        avg_freq_diff = []
        std_freq_diff = []


        for k in range(0, n_widths):

            kappa_inj_width = width_factors[k] * tau          # width (s) — change this to control the Gaussian spread 
            

            # Slice the ramp for this segment
            if k > 0:
                kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=final_kappa, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)

            phys['kappa_c_mat'] = kappa_arr[-1,:,:]
            vcsel = VCSEL(phys)
            nd = vcsel.scale_params()


            if k == 0:
                history, freq_history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)
                # eq_history, freq_hist = vcsel.generate_history(nd, shape='FR', n_cases=n_cases, des_phase_diff = 0*np.pi)
                # nd['phi_p'] = np.array([phys['phi_p_mat'][0]])*n_iterations

                nd['phi_p'] = np.array([phys['phi_p_mat'][0]])[0,:,:]
                counts = {'phase_count': 20, 'freq_count': 200}
                eq, results, E_tot = vcsel.solve_equilibria(nd, counts=counts)

                nd['phi_p'] = phys['phi_p_mat'][0]
                tmp_stable = []
                N = 30
                n_eigenvalues = N*3*N_lasers - 1
                # print(len(eqs), n_eigenvalues)
                nd['phi_p'] = phys['phi_p_mat'][0]
                tmp_stable = Parallel(n_jobs=-1)(
                    delayed(vcsel.compute_stability)(eq_pt, nd, N=N, newton_maxit=10000, threshold=1e-10, sparse=phys['sparse'], spectral_shift=0.01+0.01j, n_eigenvalues=n_eigenvalues)
                    for eq_pt in results
                )
                tmp_stable = [result[0] for result in tmp_stable]



            if eq is not None:
                phys['injection'] = True

                injection_array = np.zeros(N_lasers)
                center_idx = (N_lasers-1) // 2
                injection_array[center_idx] = 1
                phys['injection_topology'] = injection_array

                phys['injected_strength'] = nd['sbar']  # baseline amplitude


                if tmp_stable.count(1) > 0:
                    stable_eq_index = np.argmax(E_tot[np.array(tmp_stable)==1.0])
                    eq = results[np.array(tmp_stable)==1.0][stable_eq_index]

                    # Target setpoints from equilibrium
                    phi_diff_target = eq[-2]
                    omega_target = eq[-1]/(2*np.pi*1e9*tau_p)

                    # Proportional control updates
                    phys['injected_phase_diff'] = np.random.uniform(-np.pi,np.pi,n_cases)#-phi_diff_target 
                

                    phys['injected_frequency'] = omega_target #* np.ones_like(time_arr)
                
                    kappa_inj_amp = 4.0*final_kappa
                    kappa_inj_amp = np.linspace(0, kappa_inj_amp, n_cases)
                    phys['kappa_injection'] = kappa_inj_amp[:,None] * np.exp(-((time_arr - peak_time) ** 2) / (2 * kappa_inj_width ** 2))
                
                nd['phi_p'] = phys['phi_p_mat']

                


            
                phys['kappa_c_mat'] = kappa_arr
                vcsel = VCSEL(phys)
                nd = vcsel.scale_params()

                nd['injected_phase_diff'] = 0.0#np.linspace(0,2*np.pi,n_cases)
                # 



                t, y, freqs = vcsel.integrate(history, nd=nd, progress=True, theta=0.5, max_iter=1)


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

                # nearest-neighbor phase differences
                phase_diff = np.unwrap(phi[:, :-1, :] - phi[:, 1:, :], axis=1)  # shape (N_lasers-1, time)


                avg_freq_diff.append(np.mean(np.abs(dphi[:,:, segment_start:]-omega_target), axis=(1,2)))
                std_freq_diff.append(np.std(np.abs(dphi[:,:, segment_start:]-omega_target), axis=(1,2)))
            else:
                break





        avg_freq_diff = np.array(avg_freq_diff)
        std_freq_diff = np.array(std_freq_diff)



        


        if avg_freq_diff.shape == (n_cases, n_cases):
            running_avg_mean += avg_freq_diff
            running_avg_std += std_freq_diff

            avg_color_max = np.ceil(np.max(running_avg_mean/i) * 10) / 10
            std_color_max = np.ceil(np.max(running_avg_std/i) * 10) / 10

            fig = plt.figure(figsize=(12,5), dpi=200)

            plt.subplot(1,2,1)
            im1 = plt.imshow(running_avg_mean/i , aspect='auto', cmap='viridis', origin='lower', extent=[kappa_inj_amp[0]/final_kappa, kappa_inj_amp[-1]/final_kappa, width_factors[0], width_factors[-1]], vmin=0, vmax=avg_color_max)
            cbar1 = plt.colorbar(im1)
            cbar1.set_ticks([0, avg_color_max])
            cbar1.ax.tick_params(labelsize=14)
            cbar1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            cbar1.set_label(r'$\mu$', fontsize=14)
            plt.xlabel(r'Injection Strength ($\kappa_{inj}/\kappa_c$)', fontsize=16)
            plt.ylabel(r'Injection Width ($\tau_w/\tau$)', fontsize=16)
            # plt.title(r'$\mu(|\dot{\phi}-\omega|)~~$ $\kappa_c=$' + f'{final_kappa/1e9:.1f}' + r' $ns^{-1}$', fontsize=18)
            plt.title(r'Average $\mu(|\dot{\phi}-\omega|)~~$', fontsize=18)
            plt.xticks(np.linspace(kappa_inj_amp[0]/final_kappa, kappa_inj_amp[-1]/final_kappa, 5))
            plt.yticks(np.linspace(width_factors[0], width_factors[-1], 5))
            plt.tick_params(axis='both', labelsize=14)
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            plt.subplot(1,2,2)
            im2 = plt.imshow(running_avg_std/i, aspect='auto', cmap='viridis', origin='lower', extent=[kappa_inj_amp[0]/final_kappa, kappa_inj_amp[-1]/final_kappa, width_factors[0], width_factors[-1]], vmin=0, vmax=std_color_max)
            cbar2 = plt.colorbar(im2)
            cbar2.set_ticks([0, std_color_max])
            cbar2.ax.tick_params(labelsize=14)
            cbar2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            cbar2.set_label(r'$\sigma$', fontsize=14)
            plt.xlabel(r'Injection Strength ($\kappa_{inj}/\kappa_c$)', fontsize=16)
            plt.ylabel(r'Injection Width ($\tau_w/\tau$)', fontsize=16)
            plt.title(r'Average $\sigma(|\dot{\phi}-\omega|)~~$', fontsize=18)
            plt.xticks(np.linspace(kappa_inj_amp[0]/final_kappa, kappa_inj_amp[-1]/final_kappa, 5))
            plt.yticks(np.linspace(width_factors[0], width_factors[-1], 5))
            plt.tick_params(axis='both', labelsize=14)
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            plt.tight_layout()
            # plt.savefig(f'injection_tests/injection_shape/injection_shape_heatmaps_kappa{final_kappa/1e9:.1f}_detuning{detuning:.1f}GHZ.png', dpi=300)
            plt.savefig(f'injection_tests/avg_results.png', dpi=300)
            plt.show()
            plt.close(fig)
            i += 1
            print(f"TRIAL: {i}, Detuning: {detuning:.1f} GHz, final_kappa: {final_kappa/1e9:.1f} ns^-1")
        else:
            print(f"Skipping plot for final_kappa={final_kappa/1e9:.1f} due to shape mismatch: avg_freq_diff.shape={avg_freq_diff.shape}")

        


                
        
