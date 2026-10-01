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
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import argrelextrema
from joblib import Parallel, delayed
import time 
from scipy.constants import hbar, c
from pathlib import Path
import sys

# Notebook kernels commonly start in ``examples/injection`` (or another
# directory outside the repository root), and do not define a useful
# ``__file__``.  Find the root from the current working directory first, then
# from this source file when it is being run as a script.
_path_search_starts = [Path.cwd().resolve()]
if "__file__" in globals():
    _path_search_starts.append(Path(__file__).resolve().parent)
for _path_start in _path_search_starts:
    for _candidate in (_path_start, *_path_start.parents):
        if (_candidate / "examples" / "_paths.py").is_file():
            if str(_candidate) not in sys.path:
                sys.path.insert(0, str(_candidate))
            break
    else:
        continue
    break
else:
    raise ModuleNotFoundError(
        "Could not locate the vcsel_lib repository root containing examples/_paths.py."
    )

# Prefer this repository's package if a notebook previously imported another
# installed package named ``examples``.
for _module_name in tuple(sys.modules):
    if _module_name == "examples" or _module_name.startswith("examples."):
        del sys.modules[_module_name]

from examples._paths import INJECTION_DATA_DIR, INJECTION_RESULTS_DIR

DATA_DIR = INJECTION_DATA_DIR
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
noise_amplitude = 0.0
# The history generator uses NumPy's legacy global RNG.  Set it explicitly so
# this target plot and its derived frames are reproducible across reruns.
random_seed = 0
np.random.seed(random_seed)

N_lasers = 2
coupling_scheme = 'CUSTOM'
dx = 0.7

lam = 910e-9
omega0 = 2*np.pi*c/lam



results = None







phi_p = 0#np.pi

dt = 1*tau_p# 1 ps
# Four monotonically increasing branch-steering pulses, followed by a settling
# interval.  No reverse (high-to-low) sweep is performed.
Tmax = 3e-6
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)
segment_len = int(steps/2)
segment_start = int(steps/2)
plot_stride = 1
cos_smooth_tau = 0.5

n_kappa = 50
ramp_start = 10
ramp_shape = 50


# Default to one coupling value: this notebook is configured to produce the
# injection-steering / far-field animation, not a continuation sweep.  Change
# `run_full_continuation_sweep` only when branch data across the whole range is
# explicitly needed.
selected_final_kappa = 40e9  # s^-1; 40 ns^-1, matching the existing animation
run_full_continuation_sweep = False
final_kappa_arr = (
    np.linspace(0e9, 40e9, 500)
    if run_full_continuation_sweep
    else np.array([selected_final_kappa])
)
start_kappa_index = 0

if not (0 <= start_kappa_index < len(final_kappa_arr)):
    raise ValueError(
        f"start_kappa_index={start_kappa_index} is out of bounds for "
        f"final_kappa_arr of length {len(final_kappa_arr)}"
    )

aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)

kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=final_kappa_arr[start_kappa_index], N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=True, dx=dx, aMAT=aMAT)



#%%

# for detuning in np.linspace(4,5,50):
detuning = 4.0
delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
# Create evenly distributed detuning for both even and odd N_lasers
delta_dist = np.sort(np.concatenate([delta/2 * np.linspace(-1, 1, N_lasers)]))

# A single deterministic realization: no stochastic noise or ensemble envelope.
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
    'sparse': False,
    'save_every':100
}

phys['kappa_c_mat'] = kappa_arr[-1,:,:]




run_injection_steering = True
phys['injection'] = run_injection_steering
# Target every stable equilibrium once in ascending total-power order; the
# sequence ends at the highest-power branch and does not steer back down.
inject_max_etot_only = False

# n_kappa = len(kappa_c)-1

# inj_freqs = np.linspace(-3,3,n_kappa)

# Gaussian kappa injection centered at peak_time with controllable width

peak_time = 200*tau                    # first pulse center: 0.2 us
peak_spacing_tau = 500                 # separation between Gaussian peaks (in units of tau)
midpoint_window_tau = 10.0              # averaging half-window (in units of tau) around each midpoint
cache_frame_data = True                # cache one solved case for frame generation cell
use_saved_midpoint_power_data = False  # True: reload the summary plot data instead of rerunning/using live arrays
save_midpoint_power_data = False  # not needed for a single animation run
injection_label = "injection_on" if phys.get("injection", False) else "injection_off"
noise_label = "noise_on" if not np.isclose(noise_amplitude, 0.0) else "noise_off"
steering_label = "steering_max_etot" if inject_max_etot_only else "steering_all_eq"
midpoint_power_data_path = Path(
    DATA_DIR
    / f"injection_steering_midpoint_power_{injection_label}_{steering_label}_{noise_label}.npz"
)
percent_difference_overlay_paths = [
    DATA_DIR / "injection_steering_midpoint_power_injection_on_steering_max_etot_noise_on.npz",
    DATA_DIR / "injection_steering_midpoint_power_injection_off_steering_max_etot_noise_on.npz",
]
#30e9


extrema = []
all_eq_points = []
INJECTION_TESTS_DIR.mkdir(parents=True, exist_ok=True)
(INJECTION_TESTS_DIR / "injection_steering_plots").mkdir(parents=True, exist_ok=True)
S_idx   = [3*i + 1 for i in range(N_lasers)]
phi_idx = [3*i + 2 for i in range(N_lasers)]
# --- Loop over segments of kappa --- 
# max_tau_inj_width = 200

inj_phases = np.linspace(0,2*np.pi,n_cases)*0.0



for k in range(start_kappa_index, len(final_kappa_arr)):
    final_kappa = final_kappa_arr[k]

    if k > start_kappa_index:
        kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=0, kappa_final=final_kappa, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx,aMAT=aMAT)


    kappa_inj_width = 10 * tau          # width (s) — change this to control the Gaussian spread 
    kappa_inj_amp_peak = 5 * np.sum(kappa_arr[-1,:,:])
    kappa_inj_amp = np.linspace(kappa_inj_amp_peak, kappa_inj_amp_peak, n_cases)

    # Slice the ramp for this segment


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
            phase_diff_targets = []
            
            # Sort stable equilibria by total equilibrium intensity.
            stable_indices = stable_indices[np.argsort(E_tot[stable_indices])]
            if inject_max_etot_only:
                injection_indices = stable_indices[-1:]
            else:
                injection_indices = stable_indices
            if phys['injection']:
                # Create Gaussian peaks for each stable equilibrium.
                for peak_idx, stable_idx in enumerate(injection_indices):
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
                    phase_diff_targets.append(phi_diff_target)

                # Piecewise-constant injected frequency:
                # jump to the next omega_target at the midpoint between adjacent peaks.
                if omega_targets:
                    phys['injected_frequency'][:] = omega_targets[0]
                    if len(omega_targets) > 1:
                        for idx in range(len(omega_targets) - 1):
                            jump_time = 0.5 * (peak_times[idx] + peak_times[idx + 1])
                            phys['injected_frequency'][time_arr >= jump_time] = omega_targets[idx + 1]
            
            # Use the last stable equilibrium for phase target
            if len(stable_indices) > 0:
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
            dphi_plot_cases = dphi[:, :, :-1:plot_stride]
            dphi_mean = np.mean(dphi_plot_cases, axis=0)
            dphi_min = np.min(dphi_plot_cases, axis=0)
            dphi_max = np.max(dphi_plot_cases, axis=0)
            dphi_std = np.std(dphi_plot_cases, axis=0)

            # Insert previous values for the initial delay window
            # for i in range(N_lasers):
            #     dphi[i, :2*delay_steps] = prev_dphi[i]

            # nearest-neighbor phase differences, computed pair-by-pair to
            # avoid allocating a large (n_cases, N, N, steps) tensor.
            cos_pd_pair_mean = np.zeros((N_lasers - 1, phi.shape[2]))
            cos_pd_pair_min = np.zeros((N_lasers - 1, phi.shape[2]))
            cos_pd_pair_max = np.zeros((N_lasers - 1, phi.shape[2]))
            cos_pd_pair_std = np.zeros((N_lasers - 1, phi.shape[2]))
            for i in range(N_lasers - 1):
                cos_ij = np.cos(phi[:, i, :] - phi[:, i + 1, :])
                cos_pd_pair_mean[i, :] = np.mean(cos_ij, axis=0)
                cos_pd_pair_min[i, :] = np.min(cos_ij, axis=0)
                cos_pd_pair_max[i, :] = np.max(cos_ij, axis=0)
                cos_pd_pair_std[i, :] = np.std(cos_ij, axis=0)

            # --- total field summaries ---
            time_plot_full = t[::plot_stride] * 1e6
            intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
            E = np.sqrt(S) * np.exp(1j*phi)
            E_tot_cases_mW = (np.abs(E.sum(axis=1))**2) * intensity_to_mW
            E_tot_mean = np.mean(E_tot_cases_mW, axis=0)[::plot_stride]
            E_tot_std = np.std(E_tot_cases_mW, axis=0)[::plot_stride]
            E_i_mean_arr = np.mean(S, axis=0) * intensity_to_mW
            E_i_std_arr = np.std(S, axis=0) * intensity_to_mW

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
                '#0072B2', '#D60000', '#009E73', '#CC79A7',
                '#56B4E9', '#E69F00', '#000000'
            ]
            pair_colors = ['#332288', '#117733', '#CC6677', '#AA4499', '#44AA99', '#999933']

            time_plot = t[:-1:plot_stride]*1e6
            marker_every = max(1, len(time_plot) // 24)

            inj_freq_plot = None
            if eq is not None and phys['injection']:
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
                min_dphi = dphi_min[i, :]
                max_dphi = dphi_max[i, :]
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
                    min_dphi,
                    max_dphi,
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

            # --- nearest-neighbor phase differences ---
            for i in range(N_lasers-1):
                mean_cos = cos_pd_pair_mean[i, :-1:plot_stride]
                min_cos = cos_pd_pair_min[i, :-1:plot_stride]
                max_cos = cos_pd_pair_max[i, :-1:plot_stride]
                # no smoothing for cos(Δφ)
                style = line_styles[(i + 1) % len(line_styles)]
                color_i = pair_colors[i % len(pair_colors)]
                axs[1].plot(
                    time_plot,
                    mean_cos,
                    linestyle=style,
                    color=color_i,
                    linewidth=2.4,
                    label=fr'$\cos(\phi_{i+1}-\phi_{i+2})$',
                )
                axs[1].fill_between(
                    time_plot,
                    min_cos,
                    max_cos,
                    color=color_i,
                    alpha=0.10,
                )

            axs[1].set_ylim(-1.1,1.1)
            axs[1].grid(True, alpha=0.2)
            axs[1].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
            axs[1].set_xlabel('Time ($\mu$s)', fontsize=28)
            axs[1].set_ylabel(r'$\cos(\Delta\phi)$', fontsize=28)
            axs[1].tick_params(axis='both', which='major', labelsize=24)
            if N_lasers <= 6:
                axs[1].legend(loc='lower right', fontsize=22)

            # --- total field ---

            # Record power statistics at pulse midpoints plus one post-last
            # point when injection is enabled. If injection is disabled, record
            # exactly one sample at the end of the trajectory.
            sample_times = []
            if phys['injection'] and len(peak_times) >= 1:
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
            else:
                sample_times = [float(t[-1])]

            # Equilibrium total power values in the same E_tot-sorted order
            # used to build stable_indices / peaks.
            all_eq_power_sorted_mW = np.asarray(E_tot[stable_indices], dtype=float) * intensity_to_mW
            for eq_power_mW in all_eq_power_sorted_mW:
                all_eq_points.append((final_kappa * 1e-9, float(eq_power_mW)))
            eq_power_sorted_mW = np.asarray(E_tot[injection_indices], dtype=float) * intensity_to_mW
            dt_eff = np.median(np.diff(t)) if len(t) > 1 else dt
            window_half_steps = max(1, int(round((midpoint_window_tau * tau) / dt_eff)))
            for mid_idx, t_mid in enumerate(sample_times):
                idx_mid = int(np.argmin(np.abs(t - t_mid)))
                i0 = max(0, idx_mid - window_half_steps)
                i1 = min(E_tot_cases_mW.shape[1], idx_mid + window_half_steps + 1)
                if i1 <= i0:
                    continue
                # By default, branch statistics are taken across cases
                # (e.g., different noise realizations).
                per_case_power = np.mean(E_tot_cases_mW[:, i0:i1], axis=1)
                mean_power = float(np.mean(per_case_power))
                std_power = float(np.std(per_case_power))
                min_power = float(np.min(per_case_power))
                max_power = float(np.max(per_case_power))

                # If there is no noise and only one case, report
                # trajectory variability in the time window so oscillatory
                # behavior still appears in the branch error bars.
                if n_cases == 1 and np.isclose(noise_amplitude, 0.0):
                    window_power = E_tot_cases_mW[0, i0:i1]
                    mean_power = float(np.mean(window_power))
                    std_power = float(np.std(window_power))
                    min_power = float(np.min(window_power))
                    max_power = float(np.max(window_power))
                # mid_idx follows the same order as stable_indices after
                # sorting by E_tot (line where stable_indices is sorted).
                eq_rank = int(mid_idx)
                eq_power_mW = float(eq_power_sorted_mW[min(eq_rank, len(eq_power_sorted_mW) - 1)])
                extrema.append(
                    (
                        final_kappa * 1e-9,      # kappa in ns^-1
                        mean_power,
                        std_power,
                        t_mid * 1e6,             # sample time in us
                        float(mid_idx),          # sample order index
                        float(eq_rank),          # equilibrium rank (E_tot-sorted)
                        eq_power_mW,             # equilibrium total power (mW)
                        min_power,               # minimum sampled power (mW)
                        max_power,               # maximum sampled power (mW)
                    )
                )

            axs[2].plot(
                time_plot_full,
                E_tot_mean,
                color='black',
                linestyle='-',
                linewidth=2.8,
                alpha=0.95,
                label=r'$P_{\rm tot}$',
                zorder=3,
            )
            axs[2].fill_between(
                time_plot_full,
                E_tot_mean - E_tot_std,
                E_tot_mean + E_tot_std,
                color='black',
                alpha=0.08,
                zorder=1,
            )

            for i in range(N_lasers):
                E_i_mean = E_i_mean_arr[i, ::plot_stride]
                E_i_std = E_i_std_arr[i, ::plot_stride]
                style = line_styles[i % len(line_styles)]
                color_i = laser_colors[i % len(laser_colors)]
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



            axs[2].set_xlabel('Time ($\mu$s)', fontsize=28)
            axs[2].set_ylabel('Power (mW)', fontsize=28)
            axs[2].grid(True, alpha=0.2)
            axs[2].axvspan(0, 2*delay_steps*dt*1e6, color='gray', alpha=0.2)
            axs[2].tick_params(axis='both', which='major', labelsize=24)
            axs[2].set_ylim(0, 8)

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
                        color='#20B221',
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
                    "dphi_min": dphi_min[:, :].copy(),
                    "dphi_max": dphi_max[:, :].copy(),
                    "dphi_std": dphi_std[:, :].copy(),
                    "cos_pd_pair_mean": cos_pd_pair_mean[:, :-1:plot_stride].copy(),
                    "cos_pd_pair_min": cos_pd_pair_min[:, :-1:plot_stride].copy(),
                    "cos_pd_pair_max": cos_pd_pair_max[:, :-1:plot_stride].copy(),
                    "cos_pd_pair_std": cos_pd_pair_std[:, :-1:plot_stride].copy(),
                    "E_tot_mean": E_tot_mean.copy(),
                    "E_tot_std": E_tot_std.copy(),
                    "E_i_mean": E_i_mean_arr[:, ::plot_stride].copy(),
                    "E_i_std": E_i_std_arr[:, ::plot_stride].copy(),
                    "inj_freq_plot": None if inj_freq_plot is None else inj_freq_plot.copy(),
                    "P_c": P_c.copy(),
                    "inj_series": None if inj_series_for_frames is None else inj_series_for_frames.copy(),
                    # Keep the physical fields, rather than only their summary
                    # statistics, so a synchronized far-field animation can be
                    # rendered after this plotting cell has released `y`.
                    "field_time_s": t.copy(),
                    "field_S": S.copy(),
                    "field_phi": phi.copy(),
                    "injection_peak_times_us": np.asarray(peak_times, dtype=float) * 1e6,
                    "injection_target_frequencies_ghz": np.asarray(omega_targets, dtype=float),
                    "injection_target_phase_diffs": np.asarray(phase_diff_targets, dtype=float),
                    "delay_shade_end_us": float(2 * delay_steps * dt * 1e6),
                    "kappa_title": rf'$\kappa_c = {final_kappa*1e-9:.2f}\,\mathrm{{ns}}^{{-1}},\ \kappa_{{\mathrm{{ratio}}}} = {kappa_ratio:.2f}$',
                    "N_lasers": int(N_lasers),
                }

            plt.tight_layout()
            plt.savefig(INJECTION_TESTS_DIR / "injection_steering_plots" / f"{k}.png")
            # plt.savefig(f'./injection_tests/detuning_test/injection_time_series_kappa{final_kappa/1e9:.1f}_detuning{detuning:.1f}ghz.png', dpi=300)
            # plt.show()
            plt.close(fig)
            # Explicitly release heavy integration arrays as soon as plotting is done.
            del y, freqs, S, phi, dphi, dphi_plot_cases, dphi_mean, dphi_min, dphi_max, dphi_std, E, E_tot_cases_mW, E_i_mean_arr, E_i_std_arr
            gc.collect()

        else:
            print(f"No stable equilibria found for kappa = {final_kappa*1e-9:.2f} ns^-1")
    # Explicitly drop large iteration-local arrays so long sweeps don't
    # accumulate memory in notebook kernels.
    _mem_cleanup_names = [
        "history", "freq_history", "eq", "E_tot",
        "tmp_stable", "stable_indices", "peak_times", "omega_targets",
        "P_inj", "injection_power_uW",
        "t", "y", "freqs",
        "S", "phi", "dphi",
        "cos_ij", "cos_pd_pair_mean", "cos_pd_pair_std",
        "E", "E_tot_cases_mW", "E_tot_mean", "E_tot_std",
        "time_plot", "time_plot_full", "inj_freq_plot",
        "fig", "axs", "ax2", "line_styles", "marker_styles",
    ]
    for _name in _mem_cleanup_names:
        if _name in globals():
            del globals()[_name]
    plt.close("all")
    gc.collect()









def compute_percent_difference_from_mid_stats(mid_stats_array):
    if mid_stats_array.shape[1] <= 6:
        raise RuntimeError("`mid_stats` does not include equilibrium power in column 6.")
    y_mean = mid_stats_array[:, 1]
    y_eq = mid_stats_array[:, 6]
    with np.errstate(divide="ignore", invalid="ignore"):
        percent_diff = 100.0 * np.abs(y_mean - y_eq) / np.abs(y_eq)
    percent_diff[~np.isfinite(percent_diff)] = np.nan
    return percent_diff

if save_midpoint_power_data and len(extrema) > 0:
    mid_stats_to_save = np.asarray(extrema, dtype=float)
    percent_diff_to_save = compute_percent_difference_from_mid_stats(mid_stats_to_save)
    midpoint_power_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        midpoint_power_data_path,
        mid_stats=mid_stats_to_save,
        all_eq_points=np.asarray(all_eq_points, dtype=float),
        percent_diff_power=percent_diff_to_save,
        injection=np.asarray(bool(phys.get("injection", False))),
        injection_label=np.asarray(injection_label),
        injection_steering=np.asarray(bool(phys.get("injection", False))),
        inject_max_etot_only=np.asarray(bool(inject_max_etot_only)),
        steering_label=np.asarray(steering_label),
        noise_used=np.asarray(not np.isclose(noise_amplitude, 0.0)),
        noise_label=np.asarray(noise_label),
        noise_amplitude=np.asarray(noise_amplitude),
        N_lasers=np.asarray(N_lasers),
        coupling_scheme=np.asarray(coupling_scheme),
        alpha=np.asarray(alpha),
        tau=np.asarray(tau),
        tau_p=np.asarray(tau_p),
    )
    print(f"Saved midpoint power data to {midpoint_power_data_path}")


#%%
# Plot midpoint power statistics: mean with min/max envelope versus coupling strength.
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if "midpoint_power_data_path" not in globals():
    midpoint_power_data_path = DATA_DIR / "injection_steering_midpoint_power_injection_on_steering_max_etot_noise_on.npz"
if "use_saved_midpoint_power_data" not in globals():
    use_saved_midpoint_power_data = True

if "compute_percent_difference_from_mid_stats" not in globals():
    def compute_percent_difference_from_mid_stats(mid_stats_array):
        if mid_stats_array.shape[1] <= 6:
            raise RuntimeError("`mid_stats` does not include equilibrium power in column 6.")
        y_mean = mid_stats_array[:, 1]
        y_eq = mid_stats_array[:, 6]
        with np.errstate(divide="ignore", invalid="ignore"):
            percent_diff = 100.0 * np.abs(y_mean - y_eq) / np.abs(y_eq)
        percent_diff[~np.isfinite(percent_diff)] = np.nan
        return percent_diff

if use_saved_midpoint_power_data:
    if not midpoint_power_data_path.exists():
        raise FileNotFoundError(
            f"No saved midpoint power data found at {midpoint_power_data_path}. "
            "Run the simulation cell once with save_midpoint_power_data=True."
        )
    loaded_midpoint_data = np.load(midpoint_power_data_path, allow_pickle=True)
    mid_stats = np.asarray(loaded_midpoint_data["mid_stats"], dtype=float)
    all_eq_points_plot = np.asarray(loaded_midpoint_data["all_eq_points"], dtype=float)
    if "injection" in loaded_midpoint_data.files:
        injection_for_plot = bool(np.asarray(loaded_midpoint_data["injection"]).item())
    else:
        injection_for_plot = bool(phys.get("injection", False))
    print(f"Loaded midpoint power data from {midpoint_power_data_path}")
else:
    if len(extrema) == 0:
        raise RuntimeError("No midpoint statistics were recorded. Check pulse count and stability.")
    mid_stats = np.array(extrema, dtype=float)
    all_eq_points_plot = np.asarray(all_eq_points, dtype=float)
    injection_for_plot = bool(phys.get("injection", False))

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

if mid_stats.shape[1] <= 6:
    raise RuntimeError(
        "Current `extrema` does not include equilibrium power for color mapping. "
        "Re-run the simulation cell once with the updated script."
    )

x_all = mid_stats[:, 0]        # kappa (ns^-1)
y_all = mid_stats[:, 1]        # steady-state mean power (mW)
eq_power_all = mid_stats[:, 6] # equilibrium power (mW)
if mid_stats.shape[1] >= 9:
    y_min_all = mid_stats[:, 7]  # minimum sampled steady-state power (mW)
    y_max_all = mid_stats[:, 8]  # maximum sampled steady-state power (mW)
    yerr_all = np.vstack([
        np.maximum(y_all - y_min_all, 0.0),
        np.maximum(y_max_all - y_all, 0.0),
    ])
else:
    yerr_all = mid_stats[:, 2]   # legacy fallback: steady-state std (mW)

if use_saved_midpoint_power_data and "percent_diff_power" in loaded_midpoint_data.files:
    percent_diff_power_all = np.asarray(loaded_midpoint_data["percent_diff_power"], dtype=float)
else:
    percent_diff_power_all = compute_percent_difference_from_mid_stats(mid_stats)

base_cmap = plt.get_cmap('jet')
# Keep only a darker segment of the colormap to avoid very light colors.
dark_cmap_min = 0.2
dark_cmap_max = 0.9
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "viridis_dark",
    base_cmap(np.linspace(dark_cmap_min, dark_cmap_max, 256)),
)
vmin = 0.0
vmax = 100.0
norm = plt.Normalize(vmin=vmin, vmax=vmax)
color_values_all = np.nan_to_num(percent_diff_power_all, nan=vmax, posinf=vmax, neginf=vmax)

for idx, (x_i, y_i, d_i) in enumerate(zip(x_all, y_all, color_values_all)):
    color = cmap(norm(float(d_i)))
    yerr_i = yerr_all[:, idx:idx + 1] if np.ndim(yerr_all) == 2 else yerr_all[idx]
    ax.errorbar(
        x_i,
        y_i,
        yerr=yerr_i,
        fmt='none',
        capsize=0,
        elinewidth=1.0,
        alpha=0.6,
        ecolor=color,
        zorder=2,
    )
ax.scatter(
    x_all,
    y_all,
    s=15,
    marker='o',
    c=color_values_all,
    cmap=cmap,
    norm=norm,
    alpha=0.98,
    zorder=3,
)
if all_eq_points_plot.size > 0:
    eq_points_arr = np.asarray(all_eq_points_plot, dtype=float)
    x_eq_all = eq_points_arr[:, 0]
    y_eq_all = eq_points_arr[:, 1]
else:
    x_eq_all = x_all
    y_eq_all = eq_power_all
ax.scatter(
    x_eq_all,
    y_eq_all,
    s=10,
    marker='o',
    color='k',
    alpha=0.5,
    zorder=0,
)

ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', fontsize=26)
ax.set_ylabel(r'$P_{\mathrm{tot}}$ (mW)', fontsize=26)
ax.set_xlim(0, 40)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.grid(alpha=0.25)
ax.set_ylim(0,4)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'$100|P_{\mathrm{ss,mean}} - P_{\mathrm{eq}}|/|P_{\mathrm{eq}}|$ (\%)', fontsize=21)
cbar.ax.tick_params(labelsize=20)


plt.tight_layout()
plt.show()


#%%
# Plot percent difference from the selected equilibrium branch versus coupling.
from pathlib import Path

if "compute_percent_difference_from_mid_stats" not in globals():
    def compute_percent_difference_from_mid_stats(mid_stats_array):
        if mid_stats_array.shape[1] <= 6:
            raise RuntimeError("`mid_stats` does not include equilibrium power in column 6.")
        y_mean = mid_stats_array[:, 1]
        y_eq = mid_stats_array[:, 6]
        with np.errstate(divide="ignore", invalid="ignore"):
            percent_diff = 100.0 * np.abs(y_mean - y_eq) / np.abs(y_eq)
        percent_diff[~np.isfinite(percent_diff)] = np.nan
        return percent_diff

if "percent_difference_overlay_paths" not in globals():
    percent_difference_overlay_paths = [
        DATA_DIR / "injection_steering_midpoint_power_injection_on_steering_max_etot_noise_on.npz",
        DATA_DIR / "injection_steering_midpoint_power_injection_off_steering_max_etot_noise_on.npz",
    ]

def load_percent_difference_curve(path):
    data = np.load(path, allow_pickle=True)
    data_mid_stats = np.asarray(data["mid_stats"], dtype=float)
    if "percent_diff_power" in data.files:
        data_y_pct = np.asarray(data["percent_diff_power"], dtype=float)
    else:
        data_y_pct = compute_percent_difference_from_mid_stats(data_mid_stats)

    if "injection_label" in data.files:
        injection_label_loaded = str(np.asarray(data["injection_label"]).item())
        run_label = "With Injection" if injection_label_loaded == "injection_on" else "Without Injection"
    else:
        injection_loaded = bool(np.asarray(data["injection"]).item()) if "injection" in data.files else False
        run_label = "With Injection" if injection_loaded else "Without Injection"
    injection_on = run_label == "With Injection"
    return data_mid_stats[:, 0], data_y_pct, run_label, injection_on


percent_difference_curves = []
for data_path in percent_difference_overlay_paths:
    if Path(data_path).exists():
        percent_difference_curves.append(load_percent_difference_curve(Path(data_path)))

if len(percent_difference_curves) == 0:
    if "mid_stats" not in globals():
        raise RuntimeError("Run the midpoint power plotting cell first to load or build `mid_stats`.")
    if "percent_diff_power_all" in globals():
        y_pct = percent_diff_power_all
    else:
        y_pct = compute_percent_difference_from_mid_stats(mid_stats)
    percent_difference_curves.append((mid_stats[:, 0], y_pct, "current run", bool(phys.get("injection", False))))

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.axvspan(0, 12.75, color='0.85', alpha=0.6, zorder=0)
ax.axhspan(0, 5, color='lightgreen', alpha=0.5, zorder=0)
line_styles_pct = ['-', '--', ':', '-.']
run_handles = []
for curve_idx, (x_pct, y_pct, run_label, injection_on) in enumerate(percent_difference_curves):
    finite_pct = np.isfinite(x_pct) & np.isfinite(y_pct)
    line_style = line_styles_pct[curve_idx % len(line_styles_pct)]
    run_color = 'blue' if injection_on else 'red'
    ax.scatter(
        x_pct[finite_pct],
        y_pct[finite_pct],
        s=18,
        marker='o',
        color=run_color,
        alpha=0.8,
        zorder=3,
    )
    ax.plot(
        x_pct[finite_pct],
        y_pct[finite_pct],
        color=run_color,
        linestyle=line_style,
        linewidth=1.2,
        alpha=0.55,
        zorder=2,
        label=run_label,
    )
    run_handles.append(
        plt.Line2D(
            [0], [0],
            color=run_color,
            linestyle=line_style,
            marker='o',
            markerfacecolor=run_color,
            markeredgecolor=run_color,
            markersize=6,
            linewidth=1.2,
            label=run_label,
        )
    )
success_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.5, edgecolor='none',
                              label=r'Success ($<5\%$)')
ax.legend(handles=run_handles + [success_patch], loc='upper left', fontsize=13, frameon=True)

ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', fontsize=26)
ax.set_ylabel(r'$100|P_{\mathrm{ss,mean}} - P_{\mathrm{eq}}|/|P_{\mathrm{eq}}|$ (\%)', fontsize=22)
ax.set_xlim(0, 40)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


#%% ANIMATION CODE

# Generate animation frames of the steering time-series plot.
from pathlib import Path

if "steering_frame_data" in globals():
    fd = steering_frame_data
else:
    # Fallback for sessions that still have plotting arrays in memory.
    _required = [
        "time_plot", "time_plot_full",
        "dphi_mean", "dphi_std",
        "cos_pd_pair_mean", "cos_pd_pair_std",
        "E_tot_mean", "E_tot_std",
        "E_i_mean_arr", "E_i_std_arr",
        "P_c",
    ]
    if all(name in globals() for name in _required):
        fd = {
            "time_plot": time_plot.copy(),
            "time_plot_full": time_plot_full.copy(),
            "dphi_mean": dphi_mean.copy(),
            "dphi_min": dphi_min.copy() if "dphi_min" in globals() else None,
            "dphi_max": dphi_max.copy() if "dphi_max" in globals() else None,
            "dphi_std": dphi_std.copy(),
            "cos_pd_pair_mean": cos_pd_pair_mean[:, :-1:plot_stride].copy(),
            "cos_pd_pair_min": cos_pd_pair_min[:, :-1:plot_stride].copy() if "cos_pd_pair_min" in globals() else None,
            "cos_pd_pair_max": cos_pd_pair_max[:, :-1:plot_stride].copy() if "cos_pd_pair_max" in globals() else None,
            "cos_pd_pair_std": cos_pd_pair_std[:, :-1:plot_stride].copy(),
            "E_tot_mean": E_tot_mean.copy(),
            "E_tot_std": E_tot_std.copy(),
            "E_i_mean": E_i_mean_arr[:, ::plot_stride].copy(),
            "E_i_std": E_i_std_arr[:, ::plot_stride].copy(),
            "inj_freq_plot": inj_freq_plot.copy() if "inj_freq_plot" in globals() else None,
            "P_c": P_c.copy(),
            "inj_series": inj_series_for_frames.copy() if "inj_series_for_frames" in globals() and inj_series_for_frames is not None else None,
            "delay_shade_end_us": float(2 * delay_steps * dt * 1e6),
            "kappa_title": "",
            "N_lasers": int(N_lasers),
        }
    else:
        raise RuntimeError(
            "No cached steering data found. Run the simulation cell once with cache_frame_data=True."
        )

# Composite time-series + far-field frames for the single animation.
frames_dir = INJECTION_TESTS_DIR / "far_field_inj_steering"
frames_dir.mkdir(parents=True, exist_ok=True)
# Regeneration must not leave stale trailing frames from a previous schedule.
for _old_frame_path in frames_dir.glob("frame_*.png"):
    _old_frame_path.unlink()

# Frame controls. Render the initial transient much more slowly than the later
# branch-steering sequence. Duplicate endpoint indices are intentional: saved
# simulation samples are coarser than 500 frames in the first 0.25 us.
slow_window_end_us = 0.25
slow_window_frames = 500
remaining_window_frames = 500
min_points = 10
frame_plot_stride = 1  # additional decimation for frame rendering
frame_dpi = 180

# Style (match main steering plot)
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
# Palette sampled from the original steering animation: blue/red laser traces,
# deep-purple relative phase, and green injection pulses.
laser_colors = ['#0072B2', '#D60000', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#000000']
pair_colors = ['#332288', '#117733', '#CC6677', '#AA4499', '#44AA99', '#999933']
injection_color = '#20B221'

time_top = fd["time_plot"][::frame_plot_stride]
time_full = fd["time_plot_full"][::frame_plot_stride]
dphi_mean_f = fd["dphi_mean"][:, ::frame_plot_stride]
dphi_std_f = fd["dphi_std"][:, ::frame_plot_stride]
if fd.get("dphi_min") is not None and fd.get("dphi_max") is not None:
    dphi_min_f = fd["dphi_min"][:, ::frame_plot_stride]
    dphi_max_f = fd["dphi_max"][:, ::frame_plot_stride]
else:
    dphi_min_f = dphi_mean_f - dphi_std_f
    dphi_max_f = dphi_mean_f + dphi_std_f
cos_mean_f = fd["cos_pd_pair_mean"][:, ::frame_plot_stride]
cos_std_f = fd["cos_pd_pair_std"][:, ::frame_plot_stride]
if fd.get("cos_pd_pair_min") is not None and fd.get("cos_pd_pair_max") is not None:
    cos_min_f = fd["cos_pd_pair_min"][:, ::frame_plot_stride]
    cos_max_f = fd["cos_pd_pair_max"][:, ::frame_plot_stride]
else:
    cos_min_f = cos_mean_f - cos_std_f
    cos_max_f = cos_mean_f + cos_std_f
E_tot_mean_f = fd["E_tot_mean"][::frame_plot_stride]
E_tot_std_f = fd["E_tot_std"][::frame_plot_stride]
E_i_mean_f = fd["E_i_mean"][:, ::frame_plot_stride]
E_i_std_f = fd["E_i_std"][:, ::frame_plot_stride]
P_c_f = fd["P_c"][::frame_plot_stride]
inj_freq_f = None if fd["inj_freq_plot"] is None else fd["inj_freq_plot"][::frame_plot_stride]
inj_series_f = None if fd["inj_series"] is None else fd["inj_series"][::frame_plot_stride]

n_top = len(time_top)
if n_top < min_points:
    raise RuntimeError("Not enough time points to generate animation frames.")

slow_window_end_idx = int(
    np.searchsorted(time_top, slow_window_end_us, side="right")
)
slow_window_end_idx = min(max(slow_window_end_idx, min_points), n_top)
if slow_window_end_idx >= n_top:
    raise RuntimeError(
        f"The {slow_window_end_us:g} us slow window reaches the end of the simulation."
    )
frame_end_idx = np.concatenate(
    (
        np.rint(
            np.linspace(min_points, slow_window_end_idx, slow_window_frames)
        ).astype(int),
        np.rint(
            np.linspace(slow_window_end_idx + 1, n_top, remaining_window_frames)
        ).astype(int),
    )
)

# Far-field data for the lower animation panel.  Each frame shows one
# instantaneous horizontal angular slice; its normalization is global so the
# beam remains comparable from frame to frame.
if not {"field_time_s", "field_S", "field_phi"}.issubset(fd):
    raise RuntimeError(
        "The combined time-series/far-field frames require cached fields. "
        "Re-run the simulation cell with cache_frame_data=True."
    )

# Display geometry: an effective one-wavelength pitch produces a clear
# in-phase central lobe and out-of-phase central null/side-lobe pair.  The
# dynamical coupling model is unchanged; this affects only the plotted field.
far_field_emitter_spacing_m = lam
far_field_theta_range_deg = (-180.0, 180.0)
far_field_n_theta = 801
far_field_element_fwhm_deg = 100.0
far_field_blur_fwhm_deg = 1.0
field_time_s = np.asarray(fd["field_time_s"], dtype=float)
field_S = np.asarray(fd["field_S"], dtype=float)
field_phi = np.asarray(fd["field_phi"], dtype=float)
far_field_theta_deg = np.linspace(
    *far_field_theta_range_deg, far_field_n_theta
)
far_field_positions_m = (
    np.arange(field_S.shape[1]) - 0.5 * (field_S.shape[1] - 1)
) * far_field_emitter_spacing_m
far_field_steering = np.exp(
    1j * (2.0 * np.pi / lam) * np.outer(
        np.sin(np.deg2rad(far_field_theta_deg)), far_field_positions_m
    )
)
far_field_element_envelope = np.exp(
    -4.0 * np.log(2.0)
    * (far_field_theta_deg / far_field_element_fwhm_deg) ** 2
)
far_field_theta_step_deg = far_field_theta_deg[1] - far_field_theta_deg[0]
far_field_blur_sigma_samples = (
    far_field_blur_fwhm_deg
    / (2.0 * np.sqrt(2.0 * np.log(2.0)) * far_field_theta_step_deg)
)
far_field_sample_indices = np.array(
    [
        int(np.argmin(np.abs(field_time_s - time_top[end_idx - 1] * 1e-6)))
        for end_idx in frame_end_idx
    ],
    dtype=int,
)


def instantaneous_far_field_slice(sample_index):
    fields = np.sqrt(np.clip(field_S[:, :, sample_index], 0.0, None)) * np.exp(
        1j * np.remainder(field_phi[:, :, sample_index], 2.0 * np.pi)
    )
    array_field = np.einsum("ae,ce->ac", far_field_steering, fields, optimize=False)
    return np.mean(np.abs(array_field) ** 2, axis=1) * far_field_element_envelope


def smooth_far_field_profile(profile):
    """Apply the optional display-only angular blur."""
    if far_field_blur_fwhm_deg <= 0.0:
        return np.asarray(profile, dtype=float).copy()
    return gaussian_filter1d(
        profile,
        sigma=far_field_blur_sigma_samples,
        mode="nearest",
    )


far_field_slices = np.vstack(
    [instantaneous_far_field_slice(sample_index) for sample_index in far_field_sample_indices]
)
far_field_slice_max = float(np.max(far_field_slices))
if not np.isfinite(far_field_slice_max) or far_field_slice_max <= 0.0:
    raise RuntimeError("Far-field slices have zero or non-finite intensity.")
far_field_slices /= far_field_slice_max
zero_angle_index = int(np.argmin(np.abs(far_field_theta_deg)))

# Color the far-field curve continuously from the simulated instantaneous
# frequency of laser 1.  Anchor jet_r to the injection-target range so the
# lowest target frequency is red and the highest is blue.
if "injection_target_frequencies_ghz" not in fd:
    raise RuntimeError(
        "Frequency coloring requires cached injection targets. Re-run the simulation cell."
    )
injection_target_frequencies_ghz = np.asarray(
    fd["injection_target_frequencies_ghz"], dtype=float
)
if injection_target_frequencies_ghz.size == 0:
    raise RuntimeError("Cached injection-target frequency array is empty.")
_frame_time_indices = np.clip(frame_end_idx - 1, 0, len(time_top) - 1)
frame_dphi_1_ghz = dphi_mean_f[0, _frame_time_indices]
frequency_color_min = float(np.min(injection_target_frequencies_ghz))
frequency_color_max = float(np.max(injection_target_frequencies_ghz))
if np.isclose(frequency_color_min, frequency_color_max):
    frequency_color_max = frequency_color_min + 1.0
frequency_to_color = matplotlib.colors.Normalize(
    frequency_color_min, frequency_color_max, clip=True
)
frequency_colormap = plt.get_cmap("jet_r")
frame_far_field_colors = [
    frequency_colormap(frequency_to_color(_frequency))
    for _frequency in frame_dphi_1_ghz
]

for frame_id, end_idx in enumerate(frame_end_idx):
    end_full = min(end_idx + 1, len(time_full))
    marker_every = max(1, end_idx // 24)

    fig = plt.figure(figsize=(16, 18), dpi=frame_dpi)
    grid = fig.add_gridspec(
        4, 1,
        height_ratios=(1.0, 1.0, 1.12, 1.30),
        hspace=0.42,
    )
    # Use the canvas efficiently: GridSpec defaults reserve wide outer margins
    # that become very noticeable in animation frames.
    fig.subplots_adjust(left=0.095, right=0.955, bottom=0.045, top=0.975)
    axs = [fig.add_subplot(grid[row, 0]) for row in range(3)]
    far_field_ax = fig.add_subplot(grid[3, 0])

    if inj_freq_f is not None:
        axs[0].plot(
            time_top[:end_idx],
            inj_freq_f[:end_idx],
            color="#6A3D9A",
            linestyle=(0, (1, 1)),
            linewidth=5,
            alpha=0.8,
            zorder=10,
            label=r"$\dot{\phi}_{inj}$",
        )

    for i in range(fd["N_lasers"]):
        mean_dphi = dphi_mean_f[i, :end_idx]
        min_dphi = dphi_min_f[i, :end_idx]
        max_dphi = dphi_max_f[i, :end_idx]
        axs[0].plot(
            time_top[:end_idx],
            mean_dphi,
            linestyle=line_styles[i % len(line_styles)],
            color=laser_colors[i % len(laser_colors)],
            linewidth=2.4,
            markevery=marker_every,
            label=fr"$\dot{{\phi}}_{i+1}$",
        )
        axs[0].fill_between(
            time_top[:end_idx],
            min_dphi,
            max_dphi,
            color=laser_colors[i % len(laser_colors)],
            alpha=0.10,
        )

    axs[0].set_ylabel(r"$\dot{\phi}$ (GHz)", fontsize=28)
    axs[0].set_ylim(-5, 2.5)
    axs[0].grid(True, alpha=0.2)
    axs[0].axvspan(0, fd["delay_shade_end_us"], color="gray", alpha=0.2)
    axs[0].tick_params(axis="both", which="major", labelsize=24)
    if fd.get("kappa_title"):
        axs[0].set_title(fd["kappa_title"], fontsize=30, pad=8)
    if fd["N_lasers"] <= 6:
        axs[0].legend(loc="upper right", fontsize=16)

    for i in range(fd["N_lasers"] - 1):
        mean_cos = cos_mean_f[i, :end_idx]
        min_cos = cos_min_f[i, :end_idx]
        max_cos = cos_max_f[i, :end_idx]
        axs[1].plot(
            time_top[:end_idx],
            mean_cos,
            linestyle=line_styles[(i + 1) % len(line_styles)],
            color=pair_colors[i % len(pair_colors)],
            linewidth=2.4,
            label=fr"$\cos(\phi_{i+1}-\phi_{i+2})$",
        )
        axs[1].fill_between(
            time_top[:end_idx],
            min_cos,
            max_cos,
            color=pair_colors[i % len(pair_colors)],
            alpha=0.10,
        )

    axs[1].set_ylim(-1.1, 1.1)
    axs[1].grid(True, alpha=0.2)
    axs[1].axvspan(0, fd["delay_shade_end_us"], color="gray", alpha=0.2)
    axs[1].set_xlabel("Time ($\\mu$s)", fontsize=28)
    axs[1].set_ylabel(r"$\cos(\Delta\phi)$", fontsize=28)
    axs[1].tick_params(axis="both", which="major", labelsize=24)
    if fd["N_lasers"] <= 6:
        axs[1].legend(loc="lower right", fontsize=16)

    axs[2].plot(
        time_full[:end_full],
        E_tot_mean_f[:end_full],
        color="black",
        linestyle="-",
        linewidth=2.8,
        alpha=0.95,
        label=r"$P_{\rm tot}$",
        zorder=3,
    )
    axs[2].fill_between(
        time_full[:end_full],
        (E_tot_mean_f - E_tot_std_f)[:end_full],
        (E_tot_mean_f + E_tot_std_f)[:end_full],
        color="black",
        alpha=0.08,
        zorder=1,
    )

    for i in range(fd["N_lasers"]):
        axs[2].plot(
            time_full[:end_full],
            E_i_mean_f[i, :end_full],
            color=laser_colors[i % len(laser_colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2.2,
            label=f"$P_{i+1}$",
            zorder=3,
        )
        axs[2].fill_between(
            time_full[:end_full],
            (E_i_mean_f[i, :] - E_i_std_f[i, :])[:end_full],
            (E_i_mean_f[i, :] + E_i_std_f[i, :])[:end_full],
            color=laser_colors[i % len(laser_colors)],
            alpha=0.08,
            zorder=1,
        )

    axs[2].set_xlabel("Time ($\\mu$s)", fontsize=28)
    axs[2].set_ylabel("Power (mW)", fontsize=28)
    axs[2].grid(True, alpha=0.2)
    axs[2].axvspan(0, fd["delay_shade_end_us"], color="gray", alpha=0.2)
    axs[2].tick_params(axis="both", which="major", labelsize=24)
    axs[2].set_ylim(0, 10)

    ax2 = axs[2].twinx()
    ax2.set_zorder(axs[2].get_zorder() + 1)
    axs[2].patch.set_visible(False)
    ax2.patch.set_visible(False)
    ax2.plot(
        time_full[:end_full],
        P_c_f[:end_full],
        color="#1F77B4",
        linestyle=(0, (1, 1)),
        alpha=0.95,
        linewidth=2.6,
        zorder=20,
        label="Coupling",
    )
    if inj_series_f is not None:
        ax2.plot(
            time_full[:end_full],
            inj_series_f[:end_full],
            color=injection_color,
            linestyle=(0, (6, 2, 1, 2)),
            alpha=0.9,
            linewidth=2.0,
            zorder=20,
            label="Injection",
        )
    ax2.set_ylabel("Injection Power ($\\mu$W)", color="black", fontsize=28)
    ax2.set_ylim(0, 1000)
    ax2.tick_params(axis="y", labelcolor="black", labelsize=24)

    if fd["N_lasers"] <= 6:
        h1, l1 = axs[2].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend_items = {}
        for h, l in zip(h1 + h2, l1 + l2):
            if l and l not in legend_items:
                legend_items[l] = h
        axs[2].legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper right",
            fontsize=14,
            ncol=2,
            frameon=True,
        )

    # The first two panels contain the decimated frequency/phase timeline and
    # would otherwise autoscale to the animation's current endpoint.  Keep all
    # time-series axes on the full run window, like the far-field panel below.
    for time_series_ax in axs:
        time_series_ax.set_xlim(time_full[0], time_full[-1])

    current_far_field_global = far_field_slices[frame_id]
    current_on_axis_intensity = current_far_field_global[zero_angle_index]
    far_field_color = frame_far_field_colors[frame_id]
    current_far_field = smooth_far_field_profile(current_far_field_global)
    # Normalize each displayed angular slice to its own peak: this emphasizes
    # the central-lobe versus central-null modal structure, while the curve
    # color continues to convey the absolute on-axis intensity.
    current_far_field /= np.max(current_far_field)
    far_field_ax.plot(
        far_field_theta_deg,
        current_far_field,
        color=far_field_color,
        linewidth=3.2,
    )
    far_field_ax.fill_between(
        far_field_theta_deg,
        0.0,
        current_far_field,
        color=far_field_color,
        alpha=0.14,
    )
    far_field_ax.set(
        xlim=far_field_theta_range_deg,
        ylim=(0.0, 1.05),
        xlabel=r"Far-field angle $\theta$ (degrees)",
        ylabel="Globally normalized intensity",
        title=(
            rf"Instantaneous far-field slice  |  "
            rf"$\dot{{\phi}}_1={frame_dphi_1_ghz[frame_id]:.2f}\,\mathrm{{GHz}},\ "
            rf"I(\theta=0^\circ)={current_on_axis_intensity:.3f}$"
        ),
    )
    far_field_ax.set_xlabel(r"Far-field angle $\theta$ (degrees)", fontsize=20, labelpad=8)
    far_field_ax.set_ylabel("Globally normalized intensity", fontsize=20, labelpad=10)
    far_field_ax.set_title(far_field_ax.get_title(), fontsize=20, pad=13)
    far_field_ax.tick_params(axis="both", which="major", labelsize=17)
    far_field_ax.grid(alpha=0.25)

    fig.savefig(frames_dir / f"frame_{frame_id:05d}.png")
    plt.close(fig)

print(f"Saved {len(frame_end_idx)} frames to: {frames_dir}")
print("Example ffmpeg command:")
print(f"ffmpeg -framerate 30 -i {frames_dir}/frame_%05d.png -pix_fmt yuv420p injection_steering.mp4")


#%% SYNCHRONIZED FAR-FIELD FRAMES
#
# Render one instantaneous far-field frame for each steering time-series
# animation frame.  The time coordinate is taken from `frame_end_idx`, so
# far_field_inj_steering/frame_00042.png corresponds exactly to
# injection_steering_frames/frame_00042.png.  This cell deliberately only
# writes PNGs; it does not assemble a movie.
if not {"field_time_s", "field_S", "field_phi"}.issubset(fd):
    raise RuntimeError(
        "Far-field frames require cached fields. Re-run the simulation cell "
        "with cache_frame_data=True after updating this script."
    )

from examples.visualization.far_field_intensity import angle_grid, emitter_positions

# Optional far-field-only frames.  Keep them distinct from the composite
# animation frames above so running this cell cannot overwrite those PNGs.
far_field_frames_dir = INJECTION_TESTS_DIR / "far_field_inj_steering_only"
far_field_frames_dir.mkdir(parents=True, exist_ok=True)

# Optical geometry.  These match the repository's two-VCSEL far-field example;
# adjust `far_field_emitter_spacing_m` to match the fabricated array.
far_field_emitter_spacing_m = 10e-6
far_field_theta_range_deg = (-30.0, 30.0)
far_field_n_theta = 801
far_field_element_fwhm_deg = 20.0
far_field_dpi = 180

field_time_s = np.asarray(fd["field_time_s"], dtype=float)
field_S = np.asarray(fd["field_S"], dtype=float)
field_phi = np.asarray(fd["field_phi"], dtype=float)
if field_S.ndim != 3 or field_phi.shape != field_S.shape:
    raise ValueError("Cached field arrays must have shape (case, laser, time).")
if field_S.shape[1] != fd["N_lasers"]:
    raise ValueError("Cached field laser count does not match steering_frame_data.")

theta_deg = angle_grid(far_field_theta_range_deg, n_theta=far_field_n_theta)
positions_m = emitter_positions(field_S.shape[1], d=far_field_emitter_spacing_m)
steering_matrix = np.exp(
    1j * (2.0 * np.pi / lam) * np.outer(np.sin(np.deg2rad(theta_deg)), positions_m)
)
element_envelope = np.exp(
    -4.0 * np.log(2.0) * (theta_deg / far_field_element_fwhm_deg) ** 2
)

# `end_idx - 1` is the latest plotted sample in the matching time-series
# frame.  Locate it in the raw fields rather than assuming save_every=1.
far_field_sample_indices = np.array(
    [
        int(np.argmin(np.abs(field_time_s - time_top[end_idx - 1] * 1e-6)))
        for end_idx in frame_end_idx
    ],
    dtype=int,
)
far_field_frame_times_us = field_time_s[far_field_sample_indices] * 1e6


def instantaneous_far_field(sample_index):
    """Ensemble-average instantaneous intensity at one saved simulation time."""
    fields = np.sqrt(np.clip(field_S[:, :, sample_index], 0.0, None)) * np.exp(
        1j * np.remainder(field_phi[:, :, sample_index], 2.0 * np.pi)
    )
    array_field = np.einsum("ae,ce->ac", steering_matrix, fields, optimize=False)
    return np.mean(np.abs(array_field) ** 2, axis=1) * element_envelope


far_field_intensities = np.vstack(
    [instantaneous_far_field(sample_index) for sample_index in far_field_sample_indices]
)
global_far_field_max = float(np.max(far_field_intensities))
if not np.isfinite(global_far_field_max) or global_far_field_max <= 0.0:
    raise RuntimeError("The synchronized far-field frames have zero or non-finite intensity.")
far_field_intensities /= global_far_field_max

# A solution is identified by the circular-mean relative phase, binned into
# eight phase sectors.  Require a sector to persist for two frames so numerical
# noise does not recolor the plot.  Each accepted solution transition advances
# the curve color slightly, making steering events immediately visible.
relative_phases = np.angle(
    np.mean(np.exp(1j * (field_phi[:, 0, :][:, far_field_sample_indices]
                         - field_phi[:, 1, :][:, far_field_sample_indices])), axis=0)
)
phase_sectors = np.floor((relative_phases + np.pi) / (2.0 * np.pi) * 8).astype(int) % 8
solution_ids = np.zeros(len(phase_sectors), dtype=int)
active_sector = int(phase_sectors[0])
pending_sector = active_sector
pending_count = 0
for i, sector in enumerate(phase_sectors[1:], start=1):
    solution_changed = False
    if sector == active_sector:
        pending_sector, pending_count = active_sector, 0
    elif sector == pending_sector:
        pending_count += 1
        if pending_count >= 2:
            active_sector, pending_count = int(sector), 0
            solution_changed = True
    else:
        pending_sector, pending_count = int(sector), 1
    solution_ids[i] = solution_ids[i - 1] + int(solution_changed)

solution_colors = plt.get_cmap("viridis")(
    0.22 + 0.62 * ((solution_ids % 9) / 8.0)
)
for frame_id, (intensity, time_us, phase_rad, solution_id, color) in enumerate(
    zip(far_field_intensities, far_field_frame_times_us, relative_phases, solution_ids, solution_colors)
):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=far_field_dpi)
    ax.plot(theta_deg, intensity, color=color, linewidth=3.0)
    ax.fill_between(theta_deg, 0.0, intensity, color=color, alpha=0.16)
    ax.set(
        xlim=far_field_theta_range_deg,
        ylim=(0.0, 1.05),
        xlabel=r"Far-field angle $\theta$ (degrees)",
        ylabel="Globally normalized intensity",
        title=(rf"Injection steering far field  |  $t={time_us:.4f}\,\mu$s$"
               rf"  |  $\Delta\phi={phase_rad:.2f}$ rad  |  solution {solution_id}"),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(far_field_frames_dir / f"frame_{frame_id:05d}.png")
    plt.close(fig)

np.savetxt(
    far_field_frames_dir / "frame_times_us.csv",
    np.column_stack((np.arange(len(far_field_frame_times_us)), far_field_frame_times_us)),
    delimiter=",",
    header="frame_id,time_us",
    comments="",
)
print(f"Saved {len(far_field_sample_indices)} synchronized far-field frames to: {far_field_frames_dir}")
