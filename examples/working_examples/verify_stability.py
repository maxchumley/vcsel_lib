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
from joblib import Parallel, delayed
import time 
from scipy.constants import hbar, c

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# --- Parameters ---
use_fig4_params = False

alpha = 2
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4e-6 
q = 1.602e-19
beta = 1.e-3
eta = 0.9
current_threshold = 3

# Keep a copy of the original parameters
params_original = {
    "tau": 1e-9,
    "detuning": 4.0,
    "final_kappa": np.linspace(0e9, 20e9, 50)[25],
    "phi_p": 0.0,
}

# Figure 4 parameters (Appl. Sci. 2019, 9, 1436)
params_fig4 = {
    "alpha": 4.0,
    "tau_p": 7.15e-12,
    "tau_n": 0.33e-9,
    "g0": 1.13e4,
    "N0": 8.2e6,
    "beta": 3.54e-5,
    "eta": 1.0,
    "current_threshold": 4.0,
    "tau": 5e-9,           # delay time (s)
    "detuning": 0.3,       # GHz
    "final_kappa": 5e8,    # 1/s
    "phi_p": 0.0,

}

if use_fig4_params:
    alpha = params_fig4["alpha"]
    tau_p = params_fig4["tau_p"]
    tau_n = params_fig4["tau_n"]
    g0 = params_fig4["g0"]
    N0 = params_fig4["N0"]
    beta = params_fig4["beta"]
    eta = params_fig4["eta"]
    current_threshold = params_fig4["current_threshold"]
    tau = params_fig4["tau"]
    detuning = params_fig4["detuning"]
    final_kappa_override = params_fig4["final_kappa"]
    phi_p_override = params_fig4["phi_p"]
    s = 0
else:
    tau = params_original["tau"]
    detuning = params_original["detuning"]
    final_kappa_override = params_original["final_kappa"]
    phi_p_override = params_original["phi_p"]

I = eta*current_threshold * q/ tau_n * (N0 + 1/(g0*tau_p))

# print(f"p={g0*tau_p * (I*tau_n/(q) - N0) - 1:.3f}")


self_feedback = 0.0
coupling = 1.0
noise_amplitude = 1.0
noise_start = 5.0
noise_ramp_10_90_tau = 40.0
n_realizations = 100
use_injection = True
inj_peak_time_us = 1.0
kappa_inj_width_tau = 10.0
kappa_inj_amp_ratio = 10.0
n_inj_pulses = 3
inj_pulse_spacing_us = 1.0

N_lasers = 2
coupling_scheme = 'ATA'
dx = 0.7

detuning = detuning  # detuning (GHz)

# Plotting controls
plot_only_stable = True

# Stability diagnostics
print_stability_spectrum = True
stability_tol = 1e-6
print_init_alignment = True

lam = 910e-9
omega0 = 2*np.pi*c/lam



results = None
kappa_vals = np.linspace(0e9, 20e9, 200)
extrema = []
plot_enabled = True
inj_freq_plot = None

delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
# Create evenly distributed detuning for both even and odd N_lasers
delta_dist = np.sort(np.concatenate([delta/2 * np.linspace(-1, 1, N_lasers)]))
kappa_ind = 0
for final_kappa in kappa_vals:




    phi_p = phi_p_override

    dt = 1*tau_p# 1 ps
    Tmax = 5e-6
    interpolation = 'cubic'
    steps = int(Tmax / dt)
    time_arr = np.array([0.0])
    delay_steps = int(tau / dt)
    segment_len = int(steps/2)
    segment_start = int(steps/2)

    n_kappa = 1
    ramp_start = 0
    ramp_shape = 0.00001

    kappa_max = 20e9

    kappa_c = np.linspace(5e9,5e9,n_kappa)

    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=final_kappa,
        kappa_final=final_kappa,
        N_lasers=N_lasers,
        ramp_start=ramp_start,
        ramp_shape=ramp_shape,
        tau=tau,
        scheme=coupling_scheme,
        plot=False,
        dx=dx,
    )[0]





    n_cases = 1

    phi_p_vals = np.array([0.0])

    max_output_gb = 1.0
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
        'save_every':1,
        'max_output_gb': max_output_gb
    }

    phys['kappa_c_mat'] = kappa_arr#[-1,:,:]



    phys['injection'] = False
    peak_time = 200*tau                     # center (s), e.g. peak = 3*tau above
    #30e9


    S_idx   = [3*i + 1 for i in range(N_lasers)]
    phi_idx = [3*i + 2 for i in range(N_lasers)]
    # --- Loop over segments of kappa --- 
    # max_tau_inj_width = 200
    n_iterations = 1

    inj_phases = np.linspace(0,2*np.pi,n_cases)



    for k in range(0, n_kappa):

        kappa_inj_width = kappa_inj_width_tau * tau          # width (s) — change this to control the Gaussian spread 
        kappa_inj_amp_peak = kappa_inj_amp_ratio * final_kappa
        kappa_inj_amp = np.linspace(kappa_inj_amp_peak, kappa_inj_amp_peak, n_cases)

        # Slice the ramp for this segment
        if k > 0:
            kappa_arr = VCSEL.build_coupling_matrix(
                time_arr=time_arr,
                kappa_initial=final_kappa,
                kappa_final=final_kappa,
                N_lasers=N_lasers,
                ramp_start=ramp_start,
                ramp_shape=ramp_shape,
                tau=tau,
                scheme=coupling_scheme,
                plot=False,
                dx=dx,
            )[0]

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

            counts = {'phase_count': 5, 'freq_count': 50, 'max_refine':2, 'refine_factor':2}
            history, freq_history, eq, results = vcsel.generate_history(nd, shape='EQ', n_cases=1, counts=counts, guesses=guesses)

        if eq is not None:
            phys['injection'] = use_injection

            injection_array = np.zeros(N_lasers)
            center_idx = (N_lasers-1) // 2
            injection_array[center_idx] = 1
            phys['injection_topology'] = injection_array

            phys['injected_strength'] = nd['sbar']  # baseline amplitude

            tmp_stable = []
            tmp_eigs = []
            N = 100
            n_eigenvalues = N*3*N_lasers - 1
            tmp_results = Parallel(n_jobs=-1)(
                delayed(vcsel.compute_stability)(eq_pt, nd, N=N, newton_maxit=10000, threshold=1e-10, sparse=phys['sparse'], spectral_shift=0.01+0.01j, n_eigenvalues=n_eigenvalues)
                for eq_pt in results
            )

            tmp_stable = [result[0] for result in tmp_results]
            tmp_eigs = [result[1] for result in tmp_results]


        
            phys['kappa_c_mat'] = kappa_arr
            phys['phi_p_mat'] = np.ones(shape=(n_cases,N_lasers,N_lasers))*phi_p_vals[:,None,None]
            
            vcsel = VCSEL(phys)
            nd = vcsel.scale_params()
            nd['injected_phase_diff'] = 0.0#np.linspace(0,2*np.pi,n_cases)


            length = 2*delay_steps

            stable_indices = [i for i, is_stable in enumerate(tmp_stable) if is_stable]
            if plot_only_stable:
                selected_indices = stable_indices
            else:
                selected_indices = list(range(results.shape[0]))

            if not selected_indices:
                print("No equilibria selected for integration.")
                continue

            results_sel = results[selected_indices]
            history_sel = history[selected_indices]
            freq_history_sel = freq_history[selected_indices]
            tmp_stable_sel = [tmp_stable[i] for i in selected_indices]

            n_eqs = results_sel.shape[0]
            n_cases_total = n_eqs * n_realizations

            history_batch = np.repeat(history_sel, n_realizations, axis=0)
            freq_history_batch = np.repeat(freq_history_sel, n_realizations, axis=0)

            if use_injection:
                S_eq = results_sel[:, 1:2 * N_lasers:2]
                phi_eq = np.zeros((results_sel.shape[0], N_lasers))
                phi_eq[:, 1:] = results_sel[:, 2 * N_lasers:3 * N_lasers - 1]
                E_eq = np.sqrt(S_eq) * np.exp(1j * phi_eq)
                E_tot_eq = np.abs(E_eq.sum(axis=1)) ** 2
                target_idx = int(np.argmax(E_tot_eq))
                omega_target = results_sel[target_idx][-1] / (2 * np.pi * 1e9 * tau_p)

                inj_peak_time = inj_peak_time_us * 1e-6
                kappa_inj_width = kappa_inj_width_tau * tau
                kappa_inj_amp_peak = kappa_inj_amp_ratio * final_kappa
                time_inj = np.arange(steps) * dt
                pulse_centers = inj_peak_time + np.arange(n_inj_pulses) * (inj_pulse_spacing_us * 1e-6)
                kappa_inj_profile = np.zeros_like(time_inj)
                for center in pulse_centers:
                    kappa_inj_profile += kappa_inj_amp_peak * np.exp(
                        -((time_inj - center) ** 2) / (2 * kappa_inj_width ** 2)
                    )
                omega_target_nd = omega_target * 1e9 * (2 * np.pi * tau_p)
                inj_freq_ghz = np.full(steps, omega_target, dtype=float)
                save_every = int(max(1, nd.get('save_every', 1)))
                g0_si = g0
                P_inj = hbar * omega0 * kappa_inj_profile * nd['sbar'] / (g0_si * tau_n)
                inj_power_full = P_inj * 1e6

                inj_time_full = time_inj
                inj_freq_full = inj_freq_ghz

                nd['injection'] = True
                nd['injection_topology'] = injection_array
                nd['injected_strength'] = nd['sbar']
                nd['injected_phase_diff'] = 0.0
                nd['kappa_inj'] = np.repeat((kappa_inj_profile * tau_p)[None, :], n_cases_total, axis=0)
                nd['injected_frequency'] = np.full((n_cases_total, steps), omega_target_nd)
            else:
                nd['injection'] = False
                inj_time_full = None
                inj_freq_full = None
                inj_power_full = None

            nd['phi_p'] = np.ones(shape=(n_cases_total,N_lasers,N_lasers))*phi_p_vals[:,None,None]
            nd['max_output_gb'] = max_output_gb
            def noise_ramp(t):
                return VCSEL.cosine_ramp(
                    np.array([t]),
                    t_start=noise_start * tau,
                    rise_10_90=noise_ramp_10_90_tau * tau,
                    kappa_initial=0.0,
                    kappa_final=noise_amplitude,
                )[0]
            nd['noise_amplitude'] = noise_ramp


            nd['delay_interp'] = interpolation
            t, y, freqs = vcsel.integrate(history_batch, nd=nd, progress=True, theta=0.5, max_iter=1, smooth_freqs=True)

            y_stack = y.reshape(n_eqs, n_realizations, y.shape[1], y.shape[2])
            freqs_stack = freqs.reshape(n_eqs, n_realizations, freqs.shape[1], freqs.shape[2])
            y = y_stack.mean(axis=1)
            freqs = freqs_stack.mean(axis=1)
            tmp_stable = tmp_stable_sel

            freq_history = freq_history_sel

            # intensities S[i,:]
            S = np.abs(y[:, S_idx, :])

            # phases phi[i,:]
            phi = y[:, phi_idx, :]

            # instantaneous freq derivatives dphi[i,:]
            dphi = freqs * 1e-9/(2*np.pi*tau_p)
            save_every = int(max(1, nd.get('save_every', 1)))
            if save_every > 1:
                freq_history_plot = freq_history[:, :, ::save_every]
            else:
                freq_history_plot = freq_history
            hist_len = min(freq_history_plot.shape[2], dphi.shape[2])
            dphi[:, :, :hist_len] = freq_history_plot[:, :, :hist_len]
            dphi_std = freqs_stack.std(axis=1) * 1e-9/(2*np.pi*tau_p)
            dphi_std[:, :, :hist_len] = 0.0

            # nearest-neighbor phase differences (cosine, bounded in [-1, 1])
            phi_stack = y_stack[:, :, phi_idx, :]
            delta_phi = phi_stack[:, :, :-1, :] - phi_stack[:, :, 1:, :]
            cos_phase_diff_stack = np.cos(delta_phi)
            cos_phase_diff = cos_phase_diff_stack.mean(axis=1)
            cos_phase_diff_std = cos_phase_diff_stack.std(axis=1)

            E_stack = np.sqrt(np.abs(y_stack[:, :, S_idx, :])) * np.exp(1j * phi_stack)
            E_tot_stack = np.abs(E_stack.sum(axis=2))**2
            E_tot_mean = E_tot_stack.mean(axis=1)
            E_tot_std = E_tot_stack.std(axis=1)

            # ---------------------------------------------------------
            # ---------               PLOTTING               ----------
            # ---------------------------------------------------------

            # clear_output(wait=True)

        else:
            print(f"No equilibria found for kappa={final_kappa*1e-9:.2f} ns^-1.")
            continue




        


        
        if plot_enabled and eq is not None and 't' in locals():
            fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)

            time_plot = t[:-1]*1e6

            if use_injection and inj_time_full is not None:
                inj_time_plot = t * 1e6
                inj_freq_plot = np.interp(t, inj_time_full, inj_freq_full)
                axs[0].plot(inj_time_plot, inj_freq_plot, 'k--', linewidth=2, label=r'$\dot{\phi}_{inj}$', alpha=0.5, zorder=10)
                axs[0].legend(loc='upper right', fontsize=18)

            # --- dphi for each laser ---
            colors = plt.cm.tab10(np.linspace(0.0, 1.0, dphi.shape[0]))
            plot_indices = [i for i, is_stable in enumerate(tmp_stable) if is_stable] if plot_only_stable else list(range(dphi.shape[0]))
            for traj in plot_indices:
                style = '-' if tmp_stable[traj] else '--'
                zorder = 10 if tmp_stable[traj] else 1
                axs[0].plot(time_plot, dphi[traj, 0, :-1], style, linewidth=2, alpha=0.7, color=colors[traj], zorder=zorder)
                axs[0].fill_between(
                    time_plot,
                    dphi[traj, 0, :-1] - dphi_std[traj, 0, :-1],
                    dphi[traj, 0, :-1] + dphi_std[traj, 0, :-1],
                    color=colors[traj],
                    alpha=0.15,
                    linewidth=0,
                    zorder=zorder - 1,
                )
            axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=22)
            axs[0].grid(True, alpha=0.2)
            axs[0].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
            axs[0].tick_params(axis='both', which='major', labelsize=18)
            axs[0].set_ylim(-5,1)

            kappa_ratio = kappa_inj_amp[-1] / final_kappa if final_kappa != 0 else 0
            axs[0].set_title(
                rf'$\kappa_c = {final_kappa*1e-9:.2f}\,\mathrm{{ns}}^{{-1}},\ \kappa_{{\mathrm{{ratio}}}} = {kappa_ratio:.2f}$',
                fontsize=24,
                pad=20
            )

            # --- nearest-neighbor phase differences (cosine) ---
            for i in range(N_lasers-1):
                for traj in plot_indices:
                    style = '-' if tmp_stable[traj] else '--'
                    zorder = 10 if tmp_stable[traj] else 1
                    cos_mean = cos_phase_diff[traj, i, :-1]
                    cos_std = cos_phase_diff_std[traj, i, :-1]
                    axs[1].plot(time_plot, cos_mean, style, linewidth=2, alpha=0.7, color=colors[traj], zorder=zorder)
                    axs[1].fill_between(
                        time_plot,
                        np.clip(cos_mean - cos_std, -1.0, 1.0),
                        np.clip(cos_mean + cos_std, -1.0, 1.0),
                        color=colors[traj],
                        alpha=0.15,
                        linewidth=0,
                        zorder=zorder - 1,
                    )
            axs[1].set_ylim(-1.1,1.1)
            axs[1].grid(True, alpha=0.2)
            axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
            axs[1].set_xlabel('Time ($\mu$s)', fontsize=22)
            axs[1].set_ylabel(r'$\cos(\Delta\phi)$', fontsize=22)
            axs[1].tick_params(axis='both', which='major', labelsize=18)

            # --- total field ---
            E = np.sqrt(S) * np.exp(1j*phi)
            time_plot_full = t[:] * 1e6
            for traj in plot_indices:
                style = '-' if tmp_stable[traj] else '--'
                zorder = 10 if tmp_stable[traj] else 1
                axs[2].plot(time_plot_full, E_tot_mean[traj], style, linewidth=2, alpha=0.7, color=colors[traj], label='$|E_{tot}|^2$' if traj == 0 else None, zorder=zorder)
                axs[2].fill_between(
                    time_plot_full,
                    E_tot_mean[traj] - E_tot_std[traj],
                    E_tot_mean[traj] + E_tot_std[traj],
                    color=colors[traj],
                    alpha=0.15,
                    linewidth=0,
                    zorder=zorder - 1,
                )
            if use_injection and inj_time_full is not None:
                inj_time_plot_full = t * 1e6
                inj_power_plot = np.interp(t, inj_time_full, inj_power_full)
                ax2 = axs[2].twinx()
                ax2.plot(inj_time_plot_full, inj_power_plot, 'k--', linewidth=1.8, alpha=0.5, label=r'$P_{inj}$')
                ax2.set_ylabel(r'$P_{inj}$ ($\mu$W)', fontsize=20)
                ax2.tick_params(axis='y', which='major', labelsize=18)
                ax2.set_ylim(0, 500)
            axs[2].set_xlabel('Time ($\mu$s)', fontsize=22)
            axs[2].grid(True, alpha=0.2)
            axs[2].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
            axs[2].set_ylim(-2, 20)
            axs[2].tick_params(axis='both', which='major', labelsize=18)

            # Add custom legend for stable/unstable
            if plot_only_stable:
                legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle='-', label='Stable')]
            else:
                legend_elements = [
                    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Stable'),
                    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Unstable'),
                ]
            axs[2].legend(handles=legend_elements, loc='upper left', fontsize=16)

            plt.tight_layout()
            plt.savefig(f'../injection_tests/noisy_branches_injection/{kappa_ind}.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            kappa_ind += 1


    
        # Compare equilibrium points (no noise) to stationary mean under noise
        tail_len = int(len(E_tot_mean[0, :]) / 2)
        stationary_tail = E_tot_stack[:, :, -tail_len:]
        stationary_mean_per_real = stationary_tail.mean(axis=2)
        stationary_mean = stationary_mean_per_real.mean(axis=1)
        stationary_std = stationary_mean_per_real.std(axis=1)
        S_eq = results_sel[:, 1:2 * N_lasers:2]
        phi_eq = np.zeros((results_sel.shape[0], N_lasers))
        phi_eq[:, 1:] = results_sel[:, 2 * N_lasers:3 * N_lasers - 1]
        E_eq = np.sqrt(S_eq) * np.exp(1j * phi_eq)
        E_tot_eq = np.abs(E_eq.sum(axis=1)) ** 2
        kappa_ns = final_kappa * 1e-9
        for eq_val, stat_val, stat_std in zip(E_tot_eq, stationary_mean, stationary_std):
            extrema.append((kappa_ns, eq_val, stat_val, stat_std))


    


#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

cmap = cm.get_cmap('jet')
extrema = [e for e in extrema if len(e) == 4]
forward_extrema_array = np.array(extrema, dtype=float)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

# Plot with error bars
ax.errorbar(
    forward_extrema_array[:, 0],
    np.abs(forward_extrema_array[:, 2]),
    yerr=forward_extrema_array[:, 3],
    fmt='o',
    color=(0.0, 0.0, 1.0, 0.5),
    markersize=3,
    markeredgecolor=(0.0, 0.0, 1.0, 0.5),
    markerfacecolor=(0.0, 0.0, 1.0, 0.5),
    capsize=3,
    capthick=0,
    elinewidth=1,
    ecolor=(0.0, 0.0, 1.0, 0.3),
    label='Stationary Mean ± Std (Noise)'
)
ax.scatter(
    forward_extrema_array[:, 0],
    forward_extrema_array[:, 1],
    color=(1.0, 0.0, 0.0, 0.7),
    s=6,
    label='Stable Equilibrium',
)


ax.set_xlim(0,20.1)
ax.set_ylim(0,22)



ax.set_xlabel(r'$\kappa$ ($ns^{-1}$)', fontsize=20)
ax.set_ylabel(r'$|E_{tot}|^2$', fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(alpha=0.25) 
ax.legend(loc='upper left', fontsize=16)

plt.tight_layout()
plt.savefig('../injection_tests/noisy_branches_inj1.png', dpi=300, bbox_inches='tight')
plt.show()
