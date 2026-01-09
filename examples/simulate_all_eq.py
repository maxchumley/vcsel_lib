#%%
# Time series plotting for a VCSEL model

import numpy as np
from vcsel_lib import VCSEL
import matplotlib.pyplot as plt
from matplotlib import rc
import os
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')


def unique_roots(results, nd, vcsel, residuals_func, tol=1e-1, max_residual=1e-4):
    """
    Filter and return unique steady-state roots with small residuals.

    Parameters
    ----------
    results : list or np.ndarray
        List/array of root vectors (shape (N, 6)).
    nd : dict
        Nondimensional parameters (must include 'tau' and 'delta_p').
    vcsel : object
        VCSEL instance with an `order_parameter` method.
    residuals_func : callable
        Function that computes residuals: residuals_func(x, nd=...).
    tol : float, optional
        Maximum element-wise difference to consider roots identical.
    max_residual : float, optional
        Discard roots with ||f(x)|| > max_residual.

    Returns
    -------
    np.ndarray
        Array of unique valid roots, shape (n_unique, 6).
    """
    N_lasers = nd['N_lasers']

    if len(results) == 0:
        return np.empty((0, 2*N_lasers + (N_lasers - 1) + 1))

    roots = np.array(results)
    unique = []

    for r in roots:
        # wrap phase difference into [0, 2π)
        r[2*N_lasers:3*N_lasers-1] = r[2*N_lasers:3*N_lasers-1] % (2 * np.pi)

        # evaluate residual norm
        res_norm = np.linalg.norm(residuals_func(r, nd))

        # filter by residual and detuning range
        if res_norm >= max_residual:
            continue

        # check uniqueness
        if any(np.allclose(r, u, atol=tol, rtol=0) for u in unique):
            continue

        # # optional: compute order parameter for diagnostics
        # ss = np.array([[r[0], r[1], r[5] * nd['tau'], r[2], r[3], r[5] * nd['tau'] + r[4]]]).reshape((1, 6, 1))
        # order_param = vcsel.order_parameter(ss)
        # # print(f"Accepted root: |res|={res_norm:.2e}, order={order_param:.3f}")

        unique.append(r)

    return np.array(unique)



import matplotlib
# matplotlib.use("Agg")  # disable GUI backend
%matplotlib inline
import matplotlib.pyplot as plt

for detuning in np.linspace(0.0,10,50):


    all_eqs = []
    all_order_params = []
    sorted_indices_arr = []
    max_num_eqs = 0
    kappa_vals = np.linspace(0e9, 20e9, 500)








    folder_name = f"./all_eq_3laser_symmetric_detuning_test"
    os.makedirs(f"{folder_name}/detuning_{detuning:.2f}", exist_ok=True)
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
        self_feedback = 0.0
        coupling = 1.0
        noise_amplitude = 0.0
        # detuning = 0.5
        delta = detuning * 2 * np.pi * 1e9

        # Time discretization
        dt = 0.5 * tau_p#.1e-11
        Tmax = 2e-7
        steps = int(Tmax / dt)
        time_arr = np.linspace(0, Tmax, steps)
        delay_steps = int(tau / dt)

        # Kappa ramp


        # kappa_arr = VCSEL.cosine_ramp(time_arr, 5*tau, 100*tau, kappa_initial=kappa_c, kappa_final=kappa_c)

        N_lasers = 3
        coupling_scheme = 'DECAYED'
        ramp_start = 0
        ramp_shape = 0.00000001
        dx = 0.7

        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=kappa_c, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)

        phys = {
            'tau_p': tau_p,
            'tau_n': tau_n,
            'g0': g0,
            'N0': N0,
            'N_bar': N0 + 1/(g0*tau_p),
            's': s,
            'beta': beta,
            'kappa_c_mat': kappa_arr[-1,:,:],
            'phi_p_mat': np.ones(shape=(N_lasers,N_lasers))*np.pi,
            'I': I,
            'q': q,
            'alpha': alpha,
            'delta': np.sort(np.concatenate([delta*np.linspace(-1,1,N_lasers)])),  # detuning for each laser
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








        # Example guesses — vary phase difference and omega to find multiple branches
        # generate guesses over a range of phase differences (and a couple of omega seeds)
        phase_count = 20
        max_omg = 10 * 2 * np.pi * 1e9 * tau_p#np.max(np.abs(nd['delta_p']))
        phase_vals = np.linspace(-2*np.pi, 2*np.pi, phase_count, endpoint=False)
        omega_seeds = np.linspace(-max_omg, max_omg, 200)#np.linspace(-1*nd['delta_p'], 1*nd['delta_p'], 50)  # try zero and the detuning as initial omega guesses

        # guesses = []
        # for phi in phase_vals:
        #     for omega_guess in omega_seeds:
        #         guesses.append(np.concatenate([
        #             np.tile([nd['nbar'], nd['sbar']], N_lasers),  # n1, S1, n2, S2, ...
        #             np.full(N_lasers-1, phi),                      # φ1, φ2, ...
        #             np.array([omega_guess])                      # ω
        #         ]))

        if eqs is not None:
            for eq_pt in eqs:
                guesses.append(np.concatenate([
                    eq_pt[0:2*N_lasers],  # n1, S1, n2, S2, ...
                    eq_pt[2*N_lasers:(3*N_lasers -1)],                      # φ1, φ2, ...
                    np.array([eq_pt[-1]])                      # ω
                ]))
        else:
            guesses = []

        print("Additional guesses...", len(guesses))
        eq, results = vcsel.solve_equilibria(nd, guesses=guesses)
        guesses = []

        results = np.array(results)



        eqs = unique_roots(results, nd, vcsel, vcsel.residuals, tol=1e-3, max_residual=1e-6)


        if len(eqs) > max_num_eqs:
            max_num_eqs = len(eqs)


        if len(eqs) > 0:

            all_eqs.append(eqs)
    


            # history = eqs[:,:,None].repeat(2*delay_steps, axis=2)  # repeat equilibrium across time/history axis
            length = 2*delay_steps
            n_cases = eqs.shape[0]
            
            history = np.zeros((n_cases, 3*N_lasers, length))
            for k, eq in enumerate(eqs):
                omega = eq[-1]
                phase_diff = np.concatenate([[0.0], eq[2*N_lasers:(3*N_lasers -1)]])[:N_lasers]#np.concatenate([[0.0], eq[2*N_lasers:3*N_lasers-1]])

                history[k, 0::3, :] = eq[0::2][:N_lasers].reshape(-1,1)*np.ones(shape=(N_lasers,length))         # n1
                history[k, 1::3, :] = eq[1::2][:N_lasers].reshape(-1,1)*np.ones(shape=(N_lasers,length))            # S1 (nondimensional)
                history[k, 2::3, :] = omega * nd['dt'] * np.arange(length)*np.ones(shape=(N_lasers,length))       # phi1
                history[k, 2::3, :] += phase_diff.reshape(-1,1)    

            phys['kappa_c_mat'] = kappa_arr
            vcsel = VCSEL(phys)
            nd = vcsel.scale_params()
            t, y, freqs = vcsel.integrate(history, nd=nd, progress=True)

            # Extract photon numbers and phases for all N_lasers
            S = y[:, 1::3, :]  # shape (n_cases, N_lasers, length)
            phi = y[:, 2::3, :]  # shape (n_cases, N_lasers, length)
            
            dphi = freqs[:, 0::3, :] * 1e-9/(2*np.pi*tau_p)  # shape (n_cases, N_lasers, length)
            
            # Set initial phase derivatives from equilibrium
            factor = 1e-9 / (2 * np.pi * tau_p)
            prev_vals = (eqs[:, -1] * factor)[:, np.newaxis]
            prev_dphi = np.repeat(prev_vals, 2 * delay_steps, axis=1)
            dphi[:, :, :2*delay_steps] = prev_dphi[:, np.newaxis, :]
            
            # Compute phase differences relative to first laser
            phase_diff = np.unwrap(phi - phi[:, 0:1, :], axis=1)
            
            # Construct E-fields for all lasers
            E = np.sqrt(S) * (np.cos(phi) + 1j*np.sin(phi))  # shape (n_cases, N_lasers, length)
            
            # Total field intensity
            full_E_tot = np.abs(np.sum(E, axis=1))**2  # shape (n_cases, length)


            first_pt = full_E_tot[:, 0]
            change = np.max(100*np.abs(full_E_tot - first_pt[:, np.newaxis]) / first_pt[:, np.newaxis], axis=1)
            stable_indices = np.where(change < 5)[0]
            
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.colors as colors
            import numpy as np
            
            num_traj = eqs.shape[0]
            cmap = plt.colormaps['jet']
            
            # Use first laser's phase derivative for plotting
            dphi1 = dphi[:, 0, :]
            # Use phase difference between second and first laser
            cos_pd = np.cos(phase_diff[:, 1, :]) if N_lasers > 1 else np.cos(phase_diff[:, 0, :])

            # Compute the standard deviation of each row

            # Get indices that sort by std
            

            

            

            # compute time gradient of total intensity (d/dt |E1+E2|^2)
            # point-to-point absolute differences and average per trajectory
            # max_pt = np.max(full_E_tot[:, -100000:], axis=1)
            # min_pt = np.min(full_E_tot[:, -100000:], axis=1)
            # pt_range = max_pt - min_pt

            order_param = vcsel.order_parameter(y[:,:,-int(steps/2):])
            sorted_indices = np.argsort((order_param-0.5)**2)
            sorted_indices_arr.append(sorted_indices)
            order_param = order_param#[sorted_indices]
            full_E_tot = full_E_tot#[sorted_indices, :]
            dphi1 = dphi1#[sorted_indices, :]
            # phase_diff = phase_diff
            # cos_pd = np.cos(phase_diff)#[sorted_indices, :])

            all_order_params.append(order_param)

            
            colors = cmap(order_param)



            from IPython.display import clear_output
            clear_output(wait=True)
            fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True, clear=True)
            plot_flag = 0
            for j in range(num_traj):
            # for j in stable_indices:

                k = sorted_indices[j]

                if not plot_flag:
                    axs[0].set_xlabel('Time ($\mu s$)', fontsize=22)
                    axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
                    # axs[0].legend(fontsize=15, loc='lower left')
                    axs[0].set_ylim(-10,10)
                    axs[0].grid(True, alpha=0.2)
                    axs[0].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
                    axs[0].tick_params(axis='both', which='major', labelsize=18)
                    axs[0].set_title(f'$\kappa_f = {kappa_c*1e-9:.2f} \, ns^{{-1}}$', fontsize=24, pad=20)
                    axs[1].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
                    axs[2].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
                    plot_flag = 1

                

                time_plot = time_arr[:-1] * 1e6
                E_tot = full_E_tot[k, :]

                # Transparency: lower for oscillatory traces
                color = colors[k]
                alpha_t=1

                # --- Plot phase derivatives ---

                axs[0].plot(time_plot, dphi1[k, :-1], color=color, alpha=alpha_t,
                                label='_nolegend_', linewidth=2, zorder=j)

                # --- Plot phase difference ---
                axs[1].plot(time_plot, cos_pd[k,:-1], color=color,
                            label='_nolegend_', linewidth=2, alpha=alpha_t, zorder=j)
                axs[1].set_xlabel('Time ($\mu s$)', fontsize=22)
                axs[1].set_ylabel('Phase Difference', fontsize=22)
                axs[1].set_ylim(-1,1)
                axs[1].grid(True, alpha=0.2)
                    
                axs[1].tick_params(axis='both', which='major', labelsize=18)

                # --- Plot intensity ---
                time_plot_full = time_arr[:] * 1e6

                axs[2].plot(time_plot_full, E_tot, color=color, alpha=alpha_t,
                                label='_nolegend_', linewidth=2, zorder=j)

                axs[2].set_xlabel('Time ($\mu s$)', fontsize=22)
                axs[2].grid(True, alpha=0.2)
                    
                axs[2].set_ylim(-5, 50)
                axs[2].tick_params(axis='both', which='major', labelsize=18)
                axs[2].set_ylabel(r'$|E_{tot}|^2$', fontsize=22)
                # # Twin axis for kappa
                # ax2 = axs[2].twinx()
                # ax2.plot(time_plot_full, kappa_arr[-len(time_plot_full):] * 1e-9,
                #          'b--', alpha=0.5, linewidth=2)
                # ax2.set_ylabel('kappa ($ns^{-1}$)', color='blue', fontsize=28, labelpad=18)
                # ax2.set_ylim(0, kappa_max * 1e-9)
                # ax2.tick_params(axis='y', labelcolor='blue', labelsize=24, width=2, length=8)
                # ax2.tick_params(axis='x', labelsize=24, width=2, length=8)
                
            plt.tight_layout()
            # plt.savefig(f'./all_eq_delta0.5_3laser_symmetric_decayed0.7/{kappa_ind}.png')
            # plt.savefig(f'./stable_eqs_delta0.5/{kappa_ind}.png')
            plt.savefig(f"{folder_name}/detuning_{detuning:.2f}/{kappa_ind}.png")
            plt.show()
            plt.close(fig)
            # plt.close('all')
            # del vcsel
            # del history
            # del y
            # del freqs
            # del nd
            # del phys
            # del results
            # del eqs
            # import gc
            # gc.collect()

            

        else:
            print(f"No valid equilibria found for kappa_c = {kappa_c*1e-9:.2f} ns^-1") 
            all_eqs.append(np.nan*np.ones((1, 2*N_lasers + (N_lasers - 1) + 1), dtype=np.float64))
            all_order_params.append(np.array([np.nan], dtype=np.float64))



    
    %matplotlib inline
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


    cmap = plt.colormaps['jet']
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    for b in range(len(kappa_plot)):
        Et = E_tot[b,:]
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
    plt.ylim(-0.5,60)
    
    plt.tight_layout()
    plt.savefig(f'{folder_name}/branches/detuning_{detuning:.2f}_equilibrium_branches.png')
    plt.show()
    plt.close(fig)
            