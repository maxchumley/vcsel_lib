#%%
# Time series plotting for a VCSEL model

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import rc
from vcsel_lib import VCSEL
import matplotlib
# matplotlib.use("Agg")  # disable GUI backend
# %matplotlib inline
import matplotlib.pyplot as plt

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')

cmap = plt.colormaps['jet']
 
detuning = 0.01

for phi_p_loop in np.linspace(0.0,2,11):


    all_eqs = []
    all_order_params = []
    sorted_indices_arr = []
    max_num_eqs = 0
    kappa_vals = np.linspace(0e9, 20e9, 50)

 
    # folder_name = f"./all_eq_2laser_symmetric_detuning_noise"
    # os.makedirs(f"{folder_name}/detuning_{detuning:.2f}_0self_phi_p{phi_p_loop:.2f}pi_ramp_noise_injection", exist_ok=True)
    eqs = None

    for kappa_ind, kappa_c in enumerate(kappa_vals):


        # Physical parameters
        alpha = 2
        tau_p = 5.4e-12
        tau_n = 0.25e-9
        g0 = 8.75e-4 * 1e9
        N0 = 2.86e5
        s = 4e-6
        q = 1.602e-19
        beta = 1.e-3
        tau = 1e-9
        eta = 0.9
        current_threshold = 3

        I = eta * current_threshold * q / tau_n * (N0 + 1/(g0*tau_p))

        # Coupling / feedback / detuning
        # self_feedback = 0.5
        coupling = 1.0
        
        # detuning = 0.5
        delta = detuning * 2 * np.pi * 1e9

        # Time discretization
        dt = 0.5  * tau_p#.1e-11
        Tmax = 2.1e-8
        steps = int(Tmax / dt)
        time_arr = np.linspace(0, Tmax, steps)
        delay_steps = int(tau / dt)

        noise_amplitude = 0.0#VCSEL.cosine_ramp(time_arr, 50*tau, 50*tau, kappa_initial=0, kappa_final=1)
        

        N_lasers = 2
        coupling_scheme = 'ATA'
        ramp_start = 0
        ramp_shape = 0.00000001
        dx = 1.0
        n_iterations = 1

        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=kappa_c, kappa_final=kappa_c, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)

        self_feedback = 0.0

        phys = {
            'tau_p': tau_p,
            'tau_n': tau_n,
            'g0': g0,
            'N0': N0,
            'N_bar': N0 + 1/(g0*tau_p),
            's': s,
            'beta': beta,
            'kappa_c_mat': kappa_arr[-1,:,:],
            'phi_p_mat': np.ones(shape=(N_lasers,N_lasers))*phi_p_loop*np.pi,
            'I': I,
            'q': q,
            'alpha': alpha,
            'delta': np.sort(np.concatenate([delta*np.linspace(0,1,N_lasers)])),  # detuning for each laser
            'coupling': coupling,
            'self_feedback': self_feedback,
            'noise_amplitude': noise_amplitude,
            'dt': dt,
            'Tmax': Tmax,
            'tau': tau,
            'N_lasers': N_lasers
        }
    

        kappa_max = 20e9

        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()
        n_cases = len(nd['phi_p'])



        if eqs is not None:
            for eq_pt in eqs:
                guesses.append(np.concatenate([
                    eq_pt[0:N_lasers],  # n1, S1, n2, S2, ...
                    eq_pt[N_lasers:(2*N_lasers -1)],                      # φ1, φ2, ...
                    np.array([eq_pt[-1]])                      # ω
                ]))
        else:
            guesses = []

        # print("Additional guesses...", len(guesses))

        counts = {'phase_count': 50, 'freq_count': 500}
        history, freq_history, eq_max, eqs = vcsel.generate_history(nd, shape='EQ', n_cases=1, counts=counts, guesses=guesses)
        guesses = []



        if len(eqs) > max_num_eqs:
            max_num_eqs = len(eqs)


        if len(eqs) > 0:

            phys['injection'] = False

            phys['injection_topology'] = np.array([1,1])

            phys['injected_strength'] = nd['sbar']  # baseline amplitude

            # Target setpoints from equilibrium
            omega_target = eq_max[-1]/(2*np.pi*1e9*tau_p)

            phys['injected_phase_diff'] = np.linspace(0,2*np.pi,n_cases)#-phi_diff_target 

            phys['injected_frequency'] = omega_target * np.ones_like(time_arr)
        
            peak_time = 50*tau
            kappa_inj_width = 10*tau
            kappa_inj_amp = 2 * kappa_c
            phys['kappa_injection'] = kappa_inj_amp * np.exp(-((time_arr - peak_time) ** 2) / (2 * kappa_inj_width ** 2))




            all_eqs.append(eqs)
    
            length = 2*delay_steps
            
            n_cases = eqs.shape[0]*n_iterations

            phys['kappa_c_mat'] = kappa_arr
            vcsel = VCSEL(phys)
            nd = vcsel.scale_params()
            nd['injected_phase_diff'] = np.random.uniform(0, 2*np.pi, size=n_cases)#np.linspace(0,2*np.pi,n_cases)

            

            t, y, freqs = vcsel.integrate(history*1.0, nd=nd, progress=True, theta=0.8, max_iter=100)

            y = y.reshape(eqs.shape[0], n_iterations, 3*N_lasers, y.shape[-1])

            freqs = freqs.reshape(eqs.shape[0], n_iterations, N_lasers, freqs.shape[-1])

            freqs_std = np.std(freqs, axis=1)
            freqs = freqs.mean(axis=1)

            S = y[:,:, 1::3,:]   # Extract photon numbers for all N_lasers
            phi = y[:, :, 2::3, :]    # Extract phases for all N_lasers


            # Extract photon numbers and phases for all N_lasers
            # S = y[:, 1::3, :]  # shape (n_cases, N_lasers, length)
            # phi = y[:, 2::3, :]  # shape (n_cases, N_lasers, length)
            
            dphi = freqs * 1e-9/(2*np.pi*tau_p)  # shape (n_cases, N_lasers, length)
            dphi_std = freqs_std * 1e-9/(2*np.pi*tau_p)
            
            # Set initial phase derivatives from equilibrium
            factor = 1e-9 / (2 * np.pi * tau_p)
            prev_vals = (eqs[:, -1] * factor)[:, np.newaxis]
            prev_dphi = np.repeat(prev_vals, 2 * delay_steps, axis=1)
            dphi[:, :, :2*delay_steps] = prev_dphi[:, np.newaxis, :]
            dphi_std[:, :, :2*delay_steps] = 0.0
            # Compute phase differences relative to first laser
            phase_diff = np.unwrap(phi - phi[:, :, 0:1, :], axis=3)
            
            # Construct E-fields for all lasers
            E = np.sqrt(S) * (np.cos(phi) + 1j*np.sin(phi))  # shape (n_cases, N_lasers, length)
        

            
            
            # # Total field intensity
            full_E_tot = np.abs(np.sum(E, axis=2))**2  # shape (n_cases, length)

            full_E_tot_std = full_E_tot.std(axis=1)

            full_E_tot = full_E_tot.mean(axis=1)



            S_hist = history[:, 1::3, :]
            phi_hist = history[:, 2::3, :]
            E_hist = np.sqrt(S_hist) * (np.cos(phi_hist) + 1j*np.sin(phi_hist))
            E_tot_hist = np.abs(np.sum(E_hist, axis=1))**2

            
            
            
            num_traj = eqs.shape[0]
            cmap = plt.colormaps['jet']
            
            # Use first laser's phase derivative for plotting
            dphi1 = dphi[:, 0, :]
            dphi1_std = dphi_std[:, 0, :]
            # Use phase difference between second and first laser
            cos_pd = np.cos(np.unwrap(phi - phi[:, :, 0:1, :], axis=3)[:,:,1,:])
            cos_pd_std = np.std(cos_pd, axis=1)
            cos_pd = np.mean(cos_pd, axis=1)
            

            order_param = vcsel.order_parameter(y[:,:,:int(steps/2)].mean(axis=1))

            order_param = []

            for i in range(n_iterations):
                order_param.append(vcsel.order_parameter(y[:,i,:int(steps/2)]))

            order_param = np.array(order_param).mean(axis=0)


            order_param_eqs = vcsel.order_parameter(history)
            
            sorted_indices = np.argsort(np.abs(order_param-0.5))
            sorted_indices_arr.append(sorted_indices)
            order_param = order_param#[sorted_indices]
            full_E_tot = full_E_tot#[sorted_indices, :]
            dphi1 = dphi1#[sorted_indices, :]
            phase_diff = phase_diff
            # cos_pd = np.cos(phase_diff)#[sorted_indices, :])

            all_order_params.append(order_param)

            
            colors = cmap(order_param)



            from IPython.display import clear_output
            # clear_output(wait=True)
            fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True, clear=True)
            plot_flag = 0
            subsample = 10  # subsample factor



            del history
            del nd
            del phys
            # del eqs  
            import gc
            gc.collect()
            vcsel.f_nd = None
            vcsel.compute_noise_sample = None
            vcsel.__dict__.clear()
            del vcsel
            del (
                y, freqs, freqs_std, dphi, dphi_std,
                phi, S, E, phase_diff, E_hist
                
            )

            for j in range(num_traj):

        
                k = sorted_indices[j]

                if not plot_flag:
                    axs[0].set_xlabel('Time ($\mu s$)', fontsize=22)
                    axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
                    # axs[0].legend(fontsize=15, loc='lower left')
                    axs[0].set_ylim(-5,5)
                    axs[0].grid(True, alpha=0.2)
                    axs[0].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
                    axs[0].tick_params(axis='both', which='major', labelsize=18)
                    axs[0].set_title(f'$\kappa_f = {kappa_c*1e-9:.2f} \, ns^{{-1}}$', fontsize=24, pad=20)
                    axs[1].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
                    axs[2].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
                    plot_flag = 1

                

                time_plot = time_arr[:-1][::subsample] * 1e6
                E_tot = full_E_tot[k, ::subsample]
                E_tot_std = full_E_tot_std[k, ::subsample]

                # Transparency: lower for oscillatory traces
                color = colors[k]
                alpha_t=1.0

                # --- Plot phase derivatives ---

                
                # axs[0].fill_between(time_plot, dphi1[k, :-1] - dphi1_std[k, :-1], 
                #                      dphi1[k, :-1] + dphi1_std[k, :-1], 
                #                      color=color, alpha=0.1, label='_nolegend_', zorder=j-1)
                axs[0].plot(time_plot, dphi1[k, :-1][::subsample], color=color, alpha=alpha_t,
                                label='_nolegend_', linewidth=2, zorder=j)

                # --- Plot phase difference ---
                # axs[1].fill_between(time_plot, cos_pd[k,:-1] - cos_pd_std[k,:-1], 
                #                      cos_pd[k,:-1] + cos_pd_std[k,:-1], 
                #                      color=color, alpha=0.1, label='_nolegend_', zorder=j-1)
                axs[1].plot(time_plot, cos_pd[k,:-1][::subsample], color=color,
                            label='_nolegend_', linewidth=2, alpha=alpha_t, zorder=j)
                
                axs[1].set_xlabel('Time ($\mu s$)', fontsize=22)
                axs[1].set_ylabel(r'$\cos(\Delta \phi)$', fontsize=22)
                axs[1].set_ylim(-1,1)
                axs[1].grid(True, alpha=0.2)
                    
                axs[1].tick_params(axis='both', which='major', labelsize=18)

                # --- Plot intensity ---
                time_plot_full = time_arr[::subsample] * 1e6
                # axs[2].fill_between(time_plot_full, E_tot - E_tot_std, E_tot + E_tot_std, 
                #                      color=color, alpha=0.1, label='_nolegend_', zorder=j-1)
                axs[2].plot(time_plot_full, E_tot, color=color, alpha=alpha_t,
                                label='_nolegend_', linewidth=2, zorder=j)

                axs[2].set_xlabel('Time ($\mu s$)', fontsize=22)
                axs[2].grid(True, alpha=0.2)
                    
                axs[2].set_ylim(-5, 20)
                axs[2].tick_params(axis='both', which='major', labelsize=18)
                axs[2].set_ylabel(r'$|E_{tot}|^2$', fontsize=22)
                
            plt.tight_layout()
            # plt.savefig(f"{folder_name}/detuning_{detuning:.2f}_0self_phi_p{phi_p_loop:.2f}pi_ramp_noise_injection/{kappa_ind}.png")
            plt.show()
            fig.clf()
            plt.close(fig)
            del fig
            plt.close('all')

            del (full_E_tot, full_E_tot_std,
                colors)
        

        else:
            print(f"No valid equilibria found for kappa_c = {kappa_c*1e-9:.2f} ns^-1") 
            all_eqs.append(np.nan*np.ones((1, 2*N_lasers + (N_lasers - 1) + 1), dtype=np.float64))
            all_order_params.append(np.array([np.nan], dtype=np.float64))

    break

#%%
# %matplotlib inline
import matplotlib.pyplot as plt

for k, eqs in enumerate(all_eqs):
    # pad eqs with nan rows if needed so all entries have length max_num_eqs 
    if eqs.shape[0] < max_num_eqs:
        eq_diff = max_num_eqs - eqs.shape[0]
        expected_cols = 2*N_lasers + (N_lasers - 1) + 1
        pad_rows = np.nan*np.ones((eq_diff, expected_cols), dtype=eqs.dtype)
        eqs = np.vstack([eqs, pad_rows]) 

        pad_order_params = np.nan*np.ones((eq_diff,), dtype=np.float64)
        all_order_params[k] = np.hstack([all_order_params[k], pad_order_params])
        all_eqs[k] = eqs



all_eqs = np.array(all_eqs)
all_order_params = np.array(all_order_params)

# n1, S1, n2, S2, phase_diff, omega
S = all_eqs[:,:,1::2][:,:,:N_lasers]

omega = all_eqs[:,:,-1]

phi = omega[:,:,None]*tau + np.concatenate((np.zeros(shape=(len(kappa_vals), max_num_eqs, 1)), all_eqs[:,:,2*N_lasers:3*N_lasers-1]), axis=2)


E = np.sqrt(S) * (np.cos(phi) + 1j*np.sin(phi))



E_tot = np.abs(np.sum(E, axis=2))**2



import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl


n_kappa = all_eqs.shape[0]
n_branches = all_eqs.shape[1]

# reconstruct kappa values used in the loop (in Hz) and convert to ns^-1 for plotting
# kappa_vals = np.linspace(1e9, 5e9, n_kappa)        # Hz
kappa_plot = kappa_vals * 1e-9                     # ns^-1



fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

for b in range(len(kappa_plot)):
    Et = E_tot[b,:]#omega[b,:]/(2*np.pi*1e9*tau_p)
    E_mask =  np.isfinite(Et)
    if not np.any(E_mask):
        continue
    Et = Et[E_mask]
    ordp = all_order_params[b,:]
    mask =  np.isfinite(ordp)
    alpha_vals = (2*(ordp - 0.5)**2).astype(float)
    if not np.any(mask):
        continue
    ax.scatter(kappa_plot[b]*np.ones_like(Et), Et,
            c=ordp[mask], cmap=cmap, vmin=0.0, vmax=1.0,
            s=5, edgecolors='none')
    #, alpha=alpha_vals[mask]

ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', fontsize=20)
ax.set_ylabel(r'$|E_{\mathrm{tot}}|^2$', fontsize=20)
ax.set_title(rf'$\delta = {detuning:.2f}$ GHz', fontsize=22, pad=20)
# increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(alpha=0.25)

# colorbar for order parameter with larger font
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Order parameter', fontsize=20)
cbar.ax.tick_params(labelsize=20)
plt.xlim(-0.1,20.5)
# plt.ylim(-12,12)

plt.tight_layout()
# plt.savefig(f'{folder_name}/branches/detuning_{detuning:.2f}_equilibrium_branches_self_{self_feedback:.2f}_phi_p{phi_p_loop:.2f}pi.png')
plt.show()
plt.close(fig)
# break
            