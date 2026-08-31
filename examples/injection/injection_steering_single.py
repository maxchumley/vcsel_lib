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
from itertools import combinations
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelextrema
from joblib import Parallel, delayed
import time 
from scipy.constants import hbar, c
from pathlib import Path

try:
    from examples._paths import INJECTION_RESULTS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples._paths import INJECTION_RESULTS_DIR

INJECTION_TESTS_DIR = INJECTION_RESULTS_DIR / "injection_tests"

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

N_lasers = 2
coupling_scheme = 'CUSTOM'
dx = 0.7

lam = 910e-9
omega0 = 2*np.pi*c/lam



results = None

reverse = False







phi_p = 0#np.pi

dt = 1*tau_p# 1 ps
Tmax = 3e-6
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
segment_len = int(steps/2)
segment_start = int(steps/2)
plot_stride = 1
cos_smooth_tau = 0.5
plot_all_trajectories = True
plot_mean_with_all_trajectories = False
all_trajectory_alpha = 0.14

n_kappa = 50
ramp_start = 10
ramp_shape = 50


final_kappa_ns_inv = 32.0  # single coupling value to simulate (ns^-1)
final_kappa_arr = np.array([final_kappa_ns_inv * 1e9])

aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)

kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=final_kappa_arr[0], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx, aMAT=aMAT)



#%%

# for detuning in np.linspace(4,5,50):
detuning = 4.0
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/
# Create evenly distributed detuning for both even and odd N_lasers
delta_dist = np.sort(np.concatenate([delta/2 * np.linspace(-1, 1, N_lasers)]))

n_cases = 50

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
peak_spacing_tau = 500                 # separation between Gaussian peaks (in units of tau)
midpoint_window_tau = 5.0              # averaging half-window (in units of tau) around each midpoint
cache_frame_data = False               # single-run script; no frame-data caching
#30e9


extrema = []
S_idx   = [3*i + 1 for i in range(N_lasers)]
phi_idx = [3*i + 2 for i in range(N_lasers)]
# --- Loop over segments of kappa --- 
# max_tau_inj_width = 200

inj_phases = np.linspace(0,2*np.pi,n_cases)*0.0



for k in range(0, len(final_kappa_arr)):
    final_kappa = final_kappa_arr[k]

 
    kappa_inj_width = 10 * tau          # width (s) — change this to control the Gaussian spread 
    kappa_inj_amp_peak = 3 * np.sum(kappa_arr[-1,:,:])
    kappa_inj_amp = np.linspace(kappa_inj_amp_peak, kappa_inj_amp_peak, n_cases)

    # Slice the ramp for this segment
    if k > 0:
        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=final_kappa, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx,aMAT=aMAT)

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
        # break
        

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
            peak_times = []
            omega_targets = []
            
            # Create Gaussian peaks for each stable equilibrium:
            # first from lowest->highest E_tot, then reverse back down,
            # but do not repeat the highest E_tot twice in a row.
            stable_indices_sorted = stable_indices[np.argsort(E_tot[stable_indices])]
            if stable_indices_sorted.size > 1:
                reverse_without_peak_repeat = stable_indices_sorted[-2::-1]
                if reverse:
                    stable_indices_injection = np.concatenate(
                    [stable_indices_sorted, reverse_without_peak_repeat]
                )
                else:
                    stable_indices_injection = np.concatenate(
                    [stable_indices_sorted]
                )
                
            else:
                stable_indices_injection = stable_indices_sorted.copy()
            for peak_idx, stable_idx in enumerate(stable_indices_injection):
                eq = results[stable_idx]
                
                # Target setpoints from equilibrium
                phi_diff_target = eq[-2]
                omega_target = eq[-1]/(2*np.pi*1e9*tau_p)
                
                # Peak time for this equilibrium
                current_peak_time = peak_time + peak_idx * peak_spacing_tau * tau
                
                # Add Gaussian peak centered at current_peak_time
                gaussian_env = np.exp(-((time_arr - current_peak_time) ** 2) / (2 * kappa_inj_width ** 2))
                gaussian_peak = kappa_inj_amp[:, None] * gaussian_env[None, :]
                phys['kappa_injection'] += gaussian_peak
                peak_times.append(current_peak_time)
                omega_targets.append(omega_target)

            # Piecewise-constant injected frequency:
            # jump to the next omega_target at the midpoint between adjacent peaks.
            if omega_targets:
                phys['injected_frequency'][:] = omega_targets[0]
                if len(omega_targets) > 1:
                    for idx in range(len(omega_targets) - 1):
                        jump_time = 0.5 * (peak_times[idx] + peak_times[idx + 1])
                        phys['injected_frequency'][time_arr >= jump_time] = omega_targets[idx + 1]
            
            # Use the last stable equilibrium for phase target
            if len(stable_indices_injection) > 0:
                phys['injected_phase_diff'] = 0.0

            kappa = kappa_inj_amp_peak   # ns^-1 → s^-1
            g0_si = g0                   # ns^-1 → s^-1

            P_inj = hbar * omega0 * phys['kappa_injection'] * nd['sbar'] / (g0_si * tau_n)

            injection_power_uW = P_inj * 1e6  # Convert to microwatts


            nd['phi_p'] = phys['phi_p_mat']

            


        
            phys['kappa_c_mat'] = kappa_arr
            vcsel = VCSEL(phys)
            nd = vcsel.scale_params()

            nd['injected_phase_diff'] = np.linspace(0,2*np.pi,n_cases)
            # 



            t, y, freqs = vcsel.integrate(history, nd=nd, progress=True, theta=0.5, max_iter=1, smooth_freqs=True)
            # Map returned time vector to nearest base-grid indices so plotting
            # and auxiliary profiles stay consistent when save_every > 1.
            t_idx = np.clip(np.rint(t / dt).astype(int), 0, len(time_arr) - 1)


            # intensities S[i,:] (view into y to avoid an extra large copy)
            S = y[:, S_idx, :]

            # phases phi[i,:]
            phi = y[:, phi_idx, :]

            # instantaneous freq derivatives dphi[i,:]
            # Reuse freqs in-place to avoid another large allocation.
            dphi = freqs
            dphi *= 1e-9/(2*np.pi*tau_p)
            hist_mask = t_idx < freq_history.shape[2]
            if np.any(hist_mask):
                dphi[:, :, hist_mask] = freq_history[:, :, t_idx[hist_mask]]
            dphi_mean = np.mean(dphi[:, :, :-1:plot_stride], axis=0)
            dphi_std = np.std(dphi[:, :, :-1:plot_stride], axis=0)

            # Insert previous values for the initial delay window
            # for i in range(N_lasers):
            #     dphi[i, :2*delay_steps] = prev_dphi[i]

            # Pairwise wrapped phase differences, plotted trajectory-by-trajectory.
            pairs = list(combinations(range(N_lasers), 2))
            wrapped_delta_phi_pairs = []
            for i, j in pairs:
                wrapped_delta_phi = np.angle(
                    np.exp(1j * (phi[:, j, :] - phi[:, i, :]))
                )
                wrapped_delta_phi_pairs.append(wrapped_delta_phi)
            wrapped_delta_phi_pairs = np.asarray(wrapped_delta_phi_pairs)

            # --- total field summaries ---
            time_plot_full = t[::plot_stride] * 1e6
            intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
            E = np.sqrt(S) * np.exp(1j*phi)
            E_tot_cases_mW = (np.abs(E.sum(axis=1))**2) * intensity_to_mW
            E_tot_mean = np.mean(E_tot_cases_mW, axis=0)[::plot_stride]
            E_tot_min = np.min(E_tot_cases_mW, axis=0)[::plot_stride]
            E_tot_max = np.max(E_tot_cases_mW, axis=0)[::plot_stride]
            E_i_mean_arr = np.mean(S, axis=0) * intensity_to_mW
            E_i_std_arr = np.std(S, axis=0) * intensity_to_mW
            eq_power_sorted_mW = np.asarray(E_tot[stable_indices_sorted], dtype=float) * intensity_to_mW
            eq_power_injection_mW = np.asarray(E_tot[stable_indices_injection], dtype=float) * intensity_to_mW

            # Piecewise-constant equilibrium power target with jumps at
            # midpoints between injection pulses (same timing logic as
            # injected_frequency in the top panel).
            eq_power_plot = None
            if eq_power_injection_mW.size > 0:
                eq_power_profile = np.full(len(time_arr), eq_power_injection_mW[0], dtype=float)
                n_levels = min(eq_power_injection_mW.size, len(peak_times))
                if n_levels > 1:
                    for idx in range(n_levels - 1):
                        jump_time = 0.5 * (peak_times[idx] + peak_times[idx + 1])
                        eq_power_profile[time_arr >= jump_time] = eq_power_injection_mW[idx + 1]
                eq_power_plot = np.asarray(eq_power_profile[t_idx[::plot_stride]], dtype=float).copy()
                jump_idx_eq = np.where(np.abs(np.diff(eq_power_plot)) > 1e-12)[0]
                if jump_idx_eq.size > 0:
                    eq_power_plot[jump_idx_eq + 1] = np.nan

            # ---------------------------------------------------------
            # ---------               PLOTTING               ----------
            # ---------------------------------------------------------

            # clear_output(wait=True)
            fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)

            line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
            marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X']
            # Colorblind-safe palette, with linestyle/markers so curves remain
            # distinguishable even in grayscale printouts.
            laser_colors = [
                '#0072B2', '#D55E00', '#009E73', '#CC79A7',
                '#56B4E9', '#E69F00', '#000000'
            ]
            pair_colors = ['#332288', '#117733', '#CC6677', '#AA4499', '#44AA99', '#999933']

            time_plot = t[:-1:plot_stride]*1e6
            marker_every = max(1, len(time_plot) // 24)

            if eq is not None:
                inj_freq_plot = np.asarray(phys['injected_frequency'][t_idx[:-1:plot_stride]], dtype=float).copy()
                # Break the line at jump points so discontinuities are not
                # connected by vertical segments.
                jump_idx = np.where(np.abs(np.diff(inj_freq_plot)) > 1e-12)[0]
                if jump_idx.size > 0:
                    inj_freq_plot[jump_idx + 1] = np.nan
                axs[0].plot(
                    time_plot,
                    inj_freq_plot,
                    color='#6A3D9A',
                    linestyle=(0, (1, 1)),
                    linewidth=5,
                    label=r'$\dot{\phi}_{inj}$',
                    alpha=0.8,
                    zorder=10,
                )
                axs[0].legend(loc='upper right', fontsize=22)

            # --- dphi for each laser ---

            for i in range(N_lasers):
                style = line_styles[i % len(line_styles)]
                color_i = laser_colors[i % len(laser_colors)]
                mean_dphi = dphi_mean[i, :]
                std_dphi = dphi_std[i, :]
                if plot_all_trajectories:
                    for case_index in range(n_cases):
                        axs[0].plot(
                            time_plot,
                            dphi[case_index, i, :-1:plot_stride],
                            color=color_i,
                            linewidth=0.55,
                            alpha=all_trajectory_alpha,
                        )
                    axs[0].plot(
                        [],
                        [],
                        color=color_i,
                        linewidth=2.4,
                        label=fr'all $\dot{{\phi}}_{i+1}$',
                    )
                    if plot_mean_with_all_trajectories:
                        axs[0].plot(
                            time_plot,
                            mean_dphi,
                            linestyle=style,
                            color=color_i,
                            linewidth=2.4,
                            marker=marker_styles[i % len(marker_styles)],
                            markevery=marker_every,
                            markersize=4,
                            markerfacecolor='white',
                            markeredgewidth=1.0,
                        )
                else:
                    axs[0].plot(
                        time_plot,
                        mean_dphi,
                        linestyle=style,
                        color=color_i,
                        linewidth=2.4,
                        marker=marker_styles[i % len(marker_styles)],
                        markevery=marker_every,
                        markersize=4,
                        markerfacecolor='white',
                        markeredgewidth=1.0,
                        label=fr'$\dot{{\phi}}_{i+1}$',
                    )
                    axs[0].fill_between(
                        time_plot,
                        mean_dphi - std_dphi,
                        mean_dphi + std_dphi,
                        color=color_i,
                        alpha=0.10,
                    )

            if N_lasers <= 6:
                axs[0].legend(loc='upper right', fontsize=22)
            axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=28)
            # dphi_min = np.min(dphi[:, :, :-1])
            # dphi_max = np.max(dphi[:, :, :-1])
            # dphi_range = dphi_max - dphi_min
            # axs[0].set_ylim(dphi_min - 0.1 * dphi_range, dphi_max + 0.1 * dphi_range)
            axs[0].set_ylim(-5,2.5) 
            axs[0].grid(True, alpha=0.2)
            axs[0].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
            axs[0].tick_params(axis='both', which='major', labelsize=24)


            kappa_ratio = kappa_inj_amp[-1] / np.sum(kappa_arr[-1,:,:]) if np.sum(kappa_arr[-1,:,:]) != 0 else 0
            axs[0].set_title(
            rf'$\kappa_c = {final_kappa*1e-9:.2f}\,\mathrm{{ns}}^{{-1}},\ \kappa_{{\mathrm{{ratio}}}} = {kappa_ratio:.2f}$',
            fontsize=30,
            pad=20
            )

            # --- wrapped pairwise phase differences, all trajectories ---
            for pair_index, (i, j) in enumerate(pairs):
                color_i = pair_colors[pair_index % len(pair_colors)]
                for case_index in range(n_cases):
                    axs[1].plot(
                        time_plot,
                        wrapped_delta_phi_pairs[
                            pair_index,
                            case_index,
                            :-1:plot_stride,
                        ],
                        color=color_i,
                        linewidth=0.6,
                        alpha=0.18,
                    )
                axs[1].plot(
                    [],
                    [],
                    color=color_i,
                    linewidth=2.4,
                    label=fr'$\Delta \phi_{{{j+1},{i+1}}}=\phi_{j+1}-\phi_{i+1}$',
                )

            axs[1].set_ylim(-np.pi, np.pi)
            axs[1].set_yticks([-np.pi, 0.0, np.pi])
            axs[1].set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
            axs[1].grid(True, alpha=0.2)
            axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
            axs[1].set_xlabel(r'Time ($\mu$s)', fontsize=28)
            axs[1].set_ylabel(r'wrapped $\Delta\phi$ (rad)', fontsize=28)
            axs[1].tick_params(axis='both', which='major', labelsize=24)
            if N_lasers <= 6:
                axs[1].legend(loc='lower right', fontsize=22)

            # --- total field ---

            # Record power statistics at pulse midpoints and one point after the
            # final pulse.
            if len(peak_times) >= 1:
                sample_times = []
                if len(peak_times) >= 2:
                    midpoint_times = 0.5 * (np.array(peak_times[:-1]) + np.array(peak_times[1:]))
                    sample_times.extend(midpoint_times.tolist())
                    post_last_time = peak_times[-1] + 0.5 * (peak_times[-1] - peak_times[-2])
                else:
                    post_last_time = peak_times[-1] + 0.5 * peak_spacing_tau * tau

                # Keep the post-last sample inside simulation time.
                if post_last_time > t[-1]:
                    post_last_time = 0.5 * (peak_times[-1] + t[-1])
                sample_times.append(float(post_last_time))

                # Equilibrium total power values in the same injection order
                # used to build peak_times (forward then reverse).
                dt_eff = np.median(np.diff(t)) if len(t) > 1 else dt
                window_half_steps = max(1, int(round((midpoint_window_tau * tau) / dt_eff)))
                for mid_idx, t_mid in enumerate(sample_times):
                    idx_mid = int(np.argmin(np.abs(t - t_mid)))
                    i0 = max(0, idx_mid - window_half_steps)
                    i1 = min(E_tot_cases_mW.shape[1], idx_mid + window_half_steps + 1)
                    if i1 <= i0:
                        continue
                    per_case_power = np.mean(E_tot_cases_mW[:, i0:i1], axis=1)
                    # mid_idx follows the same injection order as
                    # stable_indices_injection / peak_times.
                    eq_rank = int(mid_idx)
                    eq_power_mW = float(eq_power_injection_mW[min(eq_rank, len(eq_power_injection_mW) - 1)])
                    extrema.append(
                        (
                            final_kappa * 1e-9,      # kappa in ns^-1
                            float(np.mean(per_case_power)),
                            float(np.std(per_case_power)),
                            t_mid * 1e6,             # sample time in us
                            float(mid_idx),          # sample order index
                            float(eq_rank),          # equilibrium rank (E_tot-sorted)
                            eq_power_mW,             # equilibrium total power (mW)
                        )
                    )

            if plot_all_trajectories:
                for case_index in range(n_cases):
                    axs[2].plot(
                        time_plot_full,
                        E_tot_cases_mW[case_index, ::plot_stride],
                        color='black',
                        linewidth=0.55,
                        alpha=all_trajectory_alpha,
                        zorder=1,
                    )
                axs[2].plot(
                    [],
                    [],
                    color='black',
                    linewidth=2.4,
                    label=r'all $P_{\rm tot}$',
                )
                if plot_mean_with_all_trajectories:
                    axs[2].plot(
                        time_plot_full,
                        E_tot_mean,
                        color='black',
                        linestyle='-',
                        linewidth=2.8,
                        alpha=0.95,
                        label=r'$\langle P_{\rm tot}\rangle$',
                        zorder=3,
                    )
            else:
                axs[2].plot(
                    time_plot_full,
                    E_tot_mean,
                    color='black',
                    linestyle='-',
                    linewidth=2.8,
                    alpha=0.95,
                    label=r'$\langle P_{\rm tot}\rangle$',
                    zorder=3,
                )
                axs[2].plot(
                    time_plot_full,
                    E_tot_min,
                    color='black',
                    linestyle='--',
                    linewidth=1.6,
                    alpha=0.75,
                    label=r'$\min P_{\rm tot}$',
                    zorder=2,
                )
                axs[2].plot(
                    time_plot_full,
                    E_tot_max,
                    color='black',
                    linestyle=':',
                    linewidth=1.9,
                    alpha=0.75,
                    label=r'$\max P_{\rm tot}$',
                    zorder=2,
                )

            for i in range(N_lasers):
                E_i_mean = E_i_mean_arr[i, ::plot_stride]
                E_i_std = E_i_std_arr[i, ::plot_stride]
                style = line_styles[i % len(line_styles)]
                color_i = laser_colors[i % len(laser_colors)]
                if plot_all_trajectories:
                    for case_index in range(n_cases):
                        axs[2].plot(
                            time_plot_full,
                            S[case_index, i, ::plot_stride] * intensity_to_mW,
                            color=color_i,
                            linewidth=0.45,
                            alpha=all_trajectory_alpha,
                            zorder=1,
                        )
                    axs[2].plot(
                        [],
                        [],
                        color=color_i,
                        linewidth=2.2,
                        label=fr'all $P_{i+1}$',
                    )
                    if plot_mean_with_all_trajectories:
                        axs[2].plot(
                            time_plot_full,
                            E_i_mean,
                            color=color_i,
                            linestyle=style,
                            linewidth=2.2,
                            zorder=3,
                        )
                else:
                    axs[2].plot(
                        time_plot_full,
                        E_i_mean,
                        color=color_i,
                        linestyle=style,
                        linewidth=2.2,
                        label=f'$P_{i+1}$',
                        zorder=3,
                    )
                    axs[2].fill_between(
                        time_plot_full,
                        E_i_mean - E_i_std,
                        E_i_mean + E_i_std,
                        color=color_i,
                        alpha=0.08,
                        zorder=1,
                    )

            if eq_power_plot is not None:
                axs[2].plot(
                    time_plot_full,
                    eq_power_plot,
                    color='#6A3D9A',
                    linestyle=(0, (1, 1)),
                    linewidth=5,
                    alpha=0.8,
                    zorder=10,
                    label=r'$P_{\mathrm{eq,target}}$',
                )

            

            axs[2].set_xlabel(r'Time ($\mu$s)', fontsize=28)
            axs[2].set_ylabel('Power (mW)', fontsize=28)
            axs[2].grid(True, alpha=0.2)
            axs[2].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
            axs[2].tick_params(axis='both', which='major', labelsize=24)
            axs[2].set_ylim(0, 10)

            ax2 = axs[2].twinx()
            # Keep twin-axis artists above primary-axis artists.
            ax2.set_zorder(axs[2].get_zorder() + 1)
            axs[2].patch.set_visible(False)
            ax2.patch.set_visible(False)
            P_c = hbar * omega0 * np.sum(kappa_arr[t_idx[::plot_stride], :, :], axis=(1,2)) * nd['sbar'] / (g0 * tau_n) * 1e6
            # ax2.plot(time_plot_full, kappa_arr[-len(time_plot_full):, 0,0]*1e-9, 'b--', alpha=0.5, linewidth=2)
            ax2.plot(
                time_plot_full,
                P_c,
                color='#1F77B4',
                linestyle=(0, (1,1)),
                alpha=0.95,
                linewidth=2.6,
                zorder=20,
                label='Coupling',
            )
            inj_series_for_frames = None
            if phys['injection']:
                inj_label_added = False
                for i in range(n_cases):
                    inj_series = injection_power_uW[i][t_idx[::plot_stride]]
                    if not np.any(inj_series > 10.0):
                        continue
                    if inj_series_for_frames is None:
                        inj_series_for_frames = inj_series.copy()
                    ax2.plot(
                        time_plot_full,
                        inj_series,
                        color='#1F77B4',
                        linestyle=(0, (6, 2, 1, 2)),
                        alpha=0.9,
                        linewidth=2,
                        zorder=20,
                        label='Injection' if not inj_label_added else None,
                    )
                    inj_label_added = True
            # ax2.set_ylabel('kappa ($ns^{-1}$)', color='blue', fontsize=24)
            ax2.set_ylabel('Injection Power ($\\mu$W)', color='black', fontsize=28)
            # ax2.set_ylim(0, 40e9*1e-9)
            ax2.set_ylim(0, 1000)
            ax2.tick_params(axis='y', labelcolor='black', labelsize=24)
            if N_lasers <= 6:
                h1, l1 = axs[2].get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                legend_items = {}
                for h, l in zip(h1 + h2, l1 + l2):
                    if l and l not in legend_items:
                        legend_items[l] = h
                axs[2].legend(
                    list(legend_items.values()),
                    list(legend_items.keys()),
                    loc='upper right',
                    fontsize=18,
                    ncol=2,
                    frameon=True,
                )

            if cache_frame_data:
                steering_frame_data = {
                    "time_plot": time_plot.copy(),
                    "time_plot_full": time_plot_full.copy(),
                    "dphi_mean": dphi_mean[:, :].copy(),
                    "dphi_std": dphi_std[:, :].copy(),
                    "wrapped_delta_phi_pairs": wrapped_delta_phi_pairs[:, :, :-1:plot_stride].copy(),
                    "pairs": np.asarray(pairs, dtype=int),
                    "E_tot_mean": E_tot_mean.copy(),
                    "E_tot_min": E_tot_min.copy(),
                    "E_tot_max": E_tot_max.copy(),
                    "E_i_mean": E_i_mean_arr[:, ::plot_stride].copy(),
                    "E_i_std": E_i_std_arr[:, ::plot_stride].copy(),
                    "inj_freq_plot": None if eq is None else inj_freq_plot.copy(),
                    "P_c": P_c.copy(),
                    "inj_series": None if inj_series_for_frames is None else inj_series_for_frames.copy(),
                    "delay_shade_end_us": float(2 * delay_steps * dt * 1e6),
                    "kappa_title": rf'$\kappa_c = {final_kappa*1e-9:.2f}\,\mathrm{{ns}}^{{-1}},\ \kappa_{{\mathrm{{ratio}}}} = {kappa_ratio:.2f}$',
                    "N_lasers": int(N_lasers),
                }

            # Cache raw power-panel arrays for the custom re-plot cell.
            power_plot_data = {
                "time_us": time_plot_full.copy(),
                "E_tot_mean_mW": E_tot_mean.copy(),
                "E_tot_min_mW": E_tot_min.copy(),
                "E_tot_max_mW": E_tot_max.copy(),
                "E_i_mean_mW": E_i_mean_arr[:, ::plot_stride].copy(),
                "E_i_std_mW": E_i_std_arr[:, ::plot_stride].copy(),
                "P_c_uW": P_c.copy(),
                "P_inj_uW": None if inj_series_for_frames is None else inj_series_for_frames.copy(),
                "P_eq_target_mW": None if eq_power_plot is None else eq_power_plot.copy(),
                "N_lasers": int(N_lasers),
            }

            plt.tight_layout()
            plt.show()
            plt.close(fig)
            
        else:
            print(f"No stable equilibria found for kappa = {final_kappa*1e-9:.2f} ns^-1")





#%%
# Re-plot power-panel data directly from arrays (easy to customize).
if "power_plot_data" in globals():
    pzd = power_plot_data
else:
    _required = ["time_plot_full", "E_tot_mean", "E_tot_min", "E_tot_max", "E_i_mean_arr", "E_i_std_arr", "P_c"]
    if not all(name in globals() for name in _required):
        raise RuntimeError("No power data found. Run the simulation cell first.")
    pzd = {
        "time_us": time_plot_full.copy(),
        "E_tot_mean_mW": E_tot_mean.copy(),
        "E_tot_min_mW": E_tot_min.copy(),
        "E_tot_max_mW": E_tot_max.copy(),
        "E_i_mean_mW": E_i_mean_arr[:, ::plot_stride].copy(),
        "E_i_std_mW": E_i_std_arr[:, ::plot_stride].copy(),
        "P_c_uW": P_c.copy(),
        "P_inj_uW": inj_series_for_frames.copy() if "inj_series_for_frames" in globals() and inj_series_for_frames is not None else None,
        "P_eq_target_mW": eq_power_plot.copy() if "eq_power_plot" in globals() and eq_power_plot is not None else None,
        "N_lasers": int(N_lasers),
    }
    power_plot_data = pzd

# Plot-window controls (microseconds)
apply_zoom = True
zoom_t_min_us = 0
zoom_t_max_us = 4
if apply_zoom and zoom_t_max_us <= zoom_t_min_us:
    raise ValueError("zoom_t_max_us must be greater than zoom_t_min_us.")

time_us = np.asarray(pzd["time_us"], dtype=float)
if apply_zoom:
    mask = (time_us >= zoom_t_min_us) & (time_us <= zoom_t_max_us)
else:
    mask = np.ones_like(time_us, dtype=bool)
if not np.any(mask):
    raise RuntimeError("No power samples in the requested time window.")

line_styles = ['-', '-', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
laser_colors = ['#0072B2', "#D50000", '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#000000']

fig, ax = plt.subplots(figsize=(12, 5), dpi=220)

# Total power
E_tot_mean_mW = np.asarray(pzd["E_tot_mean_mW"], dtype=float)
E_tot_min_mW = np.asarray(pzd["E_tot_min_mW"], dtype=float)
E_tot_max_mW = np.asarray(pzd["E_tot_max_mW"], dtype=float)
ax.plot(
    time_us[mask],
    E_tot_mean_mW[mask],
    color='black',
    linestyle='-',
    linewidth=2.8,
    alpha=0.9,
    label=r'$\langle P_{\rm tot}\rangle$',
    zorder=3,
)
ax.plot(
    time_us[mask],
    E_tot_min_mW[mask],
    color='black',
    linestyle='--',
    linewidth=1.6,
    alpha=0.75,
    label=r'$\min P_{\rm tot}$',
    zorder=2,
)
ax.plot(
    time_us[mask],
    E_tot_max_mW[mask],
    color='black',
    linestyle=':',
    linewidth=1.9,
    alpha=0.75,
    label=r'$\max P_{\rm tot}$',
    zorder=2,
)

# Per-laser power
E_i_mean_mW = np.asarray(pzd["E_i_mean_mW"], dtype=float)
E_i_std_mW = np.asarray(pzd["E_i_std_mW"], dtype=float)
for i in range(E_i_mean_mW.shape[0]):
    ax.plot(
        time_us[mask],
        E_i_mean_mW[i, mask],
        color=laser_colors[i % len(laser_colors)],
        linestyle=line_styles[i % len(line_styles)],
        linewidth=2.2,
        label=f'$P_{i+1}$',
        zorder=3,
    )
    ax.fill_between(
        time_us[mask],
        (E_i_mean_mW[i, :] - E_i_std_mW[i, :])[mask],
        (E_i_mean_mW[i, :] + E_i_std_mW[i, :])[mask],
        color=laser_colors[i % len(laser_colors)],
        alpha=0.08,
        zorder=1,
    )

# Target equilibrium-power trace
if pzd.get("P_eq_target_mW") is not None:
    P_eq_target_mW = np.asarray(pzd["P_eq_target_mW"], dtype=float)
    ax.plot(
        time_us[mask],
        P_eq_target_mW[mask],
        color="#00FF15",
        linestyle=(0, (1, 1)),
        linewidth=2,
        alpha=0.8,
        zorder=10,
        label=r'$P_{\mathrm{eq,target}}$',
    )

ax.set_xlabel('Time ($\\mu$s)', fontsize=20)
ax.set_ylabel('Power (mW)', fontsize=20)
if apply_zoom:
    ax.set_xlim(zoom_t_min_us, zoom_t_max_us)
ax.grid(True, alpha=0.2)
ax.tick_params(axis='both', which='major', labelsize=16)

# Coupling / injection powers
ax2 = ax.twinx()
P_c_uW = np.asarray(pzd["P_c_uW"], dtype=float)
# ax2.plot(
#     time_us[mask],
#     P_c_uW[mask],
#     color='#1F77B4',
#     linestyle=(0, (1, 1)),
#     alpha=0.95,
#     linewidth=2.6,
#     label='Coupling',
#     zorder=20,
# )
if pzd.get("P_inj_uW") is not None:
    P_inj_uW = np.asarray(pzd["P_inj_uW"], dtype=float)
    ax2.plot(
        time_us[mask],
        P_inj_uW[mask],
        color="#006F23",
        linestyle='--',
        alpha=0.9,
        linewidth=2,
        label='Injection',
        zorder=20,
    )
ax2.set_ylabel('Injection Power ($\\mu$W)', fontsize=20)
ax2.tick_params(axis='y', labelsize=16)
ax2.set_ylim(0,500)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
legend_items = {}
for h, l in zip(h1 + h2, l1 + l2):
    if l and l not in legend_items:
        legend_items[l] = h
ax.legend(
    list(legend_items.values()),
    list(legend_items.keys()),
    loc='upper right',
    fontsize=20,
    ncol=2,
    frameon=True,
)

ax.set_ylim(-0.5,7)

plt.tight_layout()
plt.show()

#%%
# Generate animation frames of the end plot with increasing time.
from pathlib import Path
import gc

required = [
    "t", "plot_stride", "N_lasers", "dphi_mean", "dphi_std",
    "wrapped_delta_phi_pairs", "pairs", "E_tot_mean", "E_tot_min", "E_tot_max",
    "E_i_mean_arr", "E_i_std_arr", "delay_steps", "dt",
    "kappa_arr", "t_idx", "nd", "g0", "tau_n", "hbar", "omega0",
    "phys", "n_cases",
]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(f"Missing required data for animation: {missing}")

frames_dir = INJECTION_TESTS_DIR / "injection_steering_frames"
frames_dir.mkdir(parents=True, exist_ok=True)

n_frames = 200
frame_dpi = 200
frame_prefix = "frame_"
min_points = 10
use_fill_between = False

line_styles = ['-', '-', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X']
laser_colors = ['#0072B2', "#D50000", '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#000000']
pair_colors = ['#332288', '#117733', '#CC6677', '#AA4499', '#44AA99', '#999933']

time_plot = t[:-1:plot_stride] * 1e6
time_plot_full = t[::plot_stride] * 1e6
if len(time_plot) < min_points or len(time_plot_full) < min_points:
    raise RuntimeError("Not enough points to build animation frames.")

wrapped_delta_phi_plot = wrapped_delta_phi_pairs[:, :, :-1:plot_stride]
E_i_mean_plot = E_i_mean_arr[:, ::plot_stride]
E_i_std_plot = E_i_std_arr[:, ::plot_stride]
P_c = hbar * omega0 * np.sum(kappa_arr[t_idx[::plot_stride], :, :], axis=(1, 2)) * nd['sbar'] / (g0 * tau_n) * 1e6

inj_freq_plot = None
if "eq" in globals() and eq is not None:
    inj_freq_plot = np.asarray(phys['injected_frequency'][t_idx[:-1:plot_stride]], dtype=float).copy()
    jump_idx = np.where(np.abs(np.diff(inj_freq_plot)) > 1e-12)[0]
    if jump_idx.size > 0:
        inj_freq_plot[jump_idx + 1] = np.nan

inj_series = None
if phys['injection'] and "injection_power_uW" in globals():
    for i in range(n_cases):
        tmp = injection_power_uW[i][t_idx[::plot_stride]]
        if np.any(tmp > 10.0):
            inj_series = tmp.copy()
            break

kappa_ratio = kappa_inj_amp[-1] / np.sum(kappa_arr[-1, :, :]) if np.sum(kappa_arr[-1, :, :]) != 0 else 0
title_text = rf'$\kappa_c = {final_kappa*1e-9:.2f}\,\mathrm{{ns}}^{{-1}},\ \kappa_{{\mathrm{{ratio}}}} = {kappa_ratio:.2f}$'

frame_end_idx = np.unique(np.linspace(min_points, len(time_plot), n_frames, dtype=int))
plt.ioff()

for frame_id, end_idx in enumerate(frame_end_idx):
    # end_idx = 555554
    end_full = min(end_idx + 1, len(time_plot_full))
    marker_every = max(1, end_idx // 24)

    fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=frame_dpi, sharex=True)

    # --- Top panel: frequencies ---
    if inj_freq_plot is not None:
        axs[0].plot(
            time_plot[:end_idx],
            inj_freq_plot[:end_idx],
            color='#6A3D9A',
            linestyle=(0, (1, 1)),
            linewidth=5,
            label=r'$\dot{\phi}_{inj}$',
            alpha=0.8,
            zorder=10,
        )

    for i in range(N_lasers):
        color_i = laser_colors[i % len(laser_colors)]
        axs[0].plot(
            time_plot[:end_idx],
            dphi_mean[i, :end_idx],
            linestyle=line_styles[i % len(line_styles)],
            color=color_i,
            linewidth=2.4,
            label=fr'$\dot{{\phi}}_{i+1}$',
        )
        # marker=marker_styles[i % len(marker_styles)],
        # markevery=marker_every,
        # markersize=4,
        # markerfacecolor='white',
        # markeredgewidth=1.0,
        if use_fill_between:
            axs[0].fill_between(
                time_plot[:end_idx],
                dphi_mean[i, :end_idx] - dphi_std[i, :end_idx],
                dphi_mean[i, :end_idx] + dphi_std[i, :end_idx],
                color=color_i,
                alpha=0.10,
            )

    if N_lasers <= 6:
        axs[0].legend(loc='upper right', fontsize=22)
    axs[0].set_ylabel(r'$\dot{\phi}$ (GHz)', fontsize=28)
    axs[0].set_ylim(-5, 2.5)
    axs[0].grid(True, alpha=0.2)
    axs[0].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
    axs[0].tick_params(axis='both', which='major', labelsize=24)
    axs[0].set_title(title_text, fontsize=30, pad=20)

    # --- Middle panel: wrapped phase differences ---
    for pair_index, (i, j) in enumerate(pairs):
        color_i = pair_colors[pair_index % len(pair_colors)]
        for case_index in range(n_cases):
            axs[1].plot(
                time_plot[:end_idx],
                wrapped_delta_phi_plot[pair_index, case_index, :end_idx],
                color=color_i,
                linewidth=0.6,
                alpha=0.18,
            )
        axs[1].plot(
            [],
            [],
            color=color_i,
            linewidth=2.4,
            label=fr'$\Delta \phi_{{{j+1},{i+1}}}=\phi_{j+1}-\phi_{i+1}$',
        )

    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].set_yticks([-np.pi, 0.0, np.pi])
    axs[1].set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    axs[1].grid(True, alpha=0.2)
    axs[1].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
    axs[1].set_xlabel(r'Time ($\mu$s)', fontsize=28)
    axs[1].set_ylabel(r'wrapped $\Delta\phi$ (rad)', fontsize=28)
    axs[1].tick_params(axis='both', which='major', labelsize=24)
    if N_lasers <= 6:
        axs[1].legend(loc='lower right', fontsize=22)

    # --- Bottom panel: power ---
    axs[2].plot(
        time_plot_full[:end_full],
        E_tot_mean[:end_full],
        color='black',
        linestyle='-',
        linewidth=2.8,
        alpha=0.95,
        label=r'$\langle P_{\rm tot}\rangle$',
        zorder=3,
    )
    axs[2].plot(
        time_plot_full[:end_full],
        E_tot_min[:end_full],
        color='black',
        linestyle='--',
        linewidth=1.6,
        alpha=0.75,
        label=r'$\min P_{\rm tot}$',
        zorder=2,
    )
    axs[2].plot(
        time_plot_full[:end_full],
        E_tot_max[:end_full],
        color='black',
        linestyle=':',
        linewidth=1.9,
        alpha=0.75,
        label=r'$\max P_{\rm tot}$',
        zorder=2,
    )

    for i in range(N_lasers):
        color_i = laser_colors[i % len(laser_colors)]
        axs[2].plot(
            time_plot_full[:end_full],
            E_i_mean_plot[i, :end_full],
            color=color_i,
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2.2,
            label=f'$P_{i+1}$',
            zorder=3,
        )
        if use_fill_between:
            axs[2].fill_between(
                time_plot_full[:end_full],
                (E_i_mean_plot[i, :] - E_i_std_plot[i, :])[:end_full],
                (E_i_mean_plot[i, :] + E_i_std_plot[i, :])[:end_full],
                color=color_i,
                alpha=0.08,
                zorder=1,
            )

    axs[2].set_xlabel('Time ($\\mu$s)', fontsize=28)
    axs[2].set_ylabel('Power (mW)', fontsize=28)
    axs[2].grid(True, alpha=0.2)
    axs[2].axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
    axs[2].tick_params(axis='both', which='major', labelsize=24)
    axs[2].set_ylim(-0.5, 8)

    ax2 = axs[2].twinx()
    ax2.set_zorder(axs[2].get_zorder() + 1)
    axs[2].patch.set_visible(False)
    ax2.patch.set_visible(False)
    if inj_series is not None:
        ax2.plot(
            time_plot_full[:end_full],
            inj_series[:end_full],
            color="#1FB421",
            linestyle='--',
            alpha=0.9,
            linewidth=2,
            zorder=20,
            label='Injection',
        )
    ax2.set_ylabel('Injection Power ($\\mu$W)', color='black', fontsize=28)
    ax2.set_ylim(0, 800)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=24)

    if N_lasers <= 6:
        h1, l1 = axs[2].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend_items = {}
        for h, l in zip(h1 + h2, l1 + l2):
            if l and l not in legend_items:
                legend_items[l] = h
        axs[2].legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc='upper right',
            fontsize=18,
            ncol=2,
            frameon=True,
        )

    axs[0].set_xlim(time_plot[0], time_plot[-1])
    plt.tight_layout()
    fig.savefig(frames_dir / f"{frame_prefix}{frame_id:05d}.png")
    # Explicitly break references created during this frame so memory
    # does not accumulate across the loop in notebook kernels.
    if N_lasers <= 6:
        del h1, l1, h2, l2, legend_items
    for _ax in axs:
        _ax.cla()
    ax2.cla()
    fig.clf()
    plt.close(fig)
    plt.close("all")
    del ax2, axs, fig, end_full, marker_every
    gc.collect()
    # break

plt.ion()

print(f"Saved {len(frame_end_idx)} frames to: {frames_dir}")
print("Example ffmpeg command:")
print(f"ffmpeg -framerate 30 -i {frames_dir}/{frame_prefix}%05d.png -pix_fmt yuv420p injection_steering_single.mp4")

#%%




print(", ".join(f"{omega:.3f}" for omega in np.sort(omega_targets)[::-1]))
