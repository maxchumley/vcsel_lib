#%%
import gc
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import rc
from scipy.constants import c, hbar

from vcsel_lib import VCSEL

try:
    from examples._paths import INJECTION_DATA_DIR, INJECTION_RESULTS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples._paths import INJECTION_DATA_DIR, INJECTION_RESULTS_DIR


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


#%%
# Run controls
run_beta_sweep = True
use_saved_beta_data = False
save_beta_data = True

beta_arr = np.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=float)
data_dir = INJECTION_DATA_DIR / "beta_dependence"
figure_path = INJECTION_RESULTS_DIR / "injection_tests" / "beta_dependence_success.png"


#%%
# Physical and numerical parameters, matching injection_steering.py.
alpha = 2
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4e-6
q = 1.602e-19
tau = 1e-9
eta = 0.9
current_threshold = 3

I = eta * current_threshold * q / tau_n * (N0 + 1 / (g0 * tau_p))

self_feedback = 0.0
coupling = 1.0
noise_amplitude = 1.0

N_lasers = 2
coupling_scheme = 'CUSTOM'
dx = 0.7
aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)

lam = 910e-9
omega0 = 2 * np.pi * c / lam

phi_p = 0.0
dt = 1 * tau_p
Tmax = 2e-7
steps = int(Tmax / dt)
time_arr = np.linspace(0, Tmax, steps)
delay_steps = int(tau / dt)

detuning = 4.0
delta = detuning * 2 * np.pi * 1e9
delta_dist = np.sort(np.concatenate([delta / 2 * np.linspace(-1, 1, N_lasers)]))

n_cases = 50
phi_p_vals = np.array([0.0])
final_kappa_arr = np.linspace(0e9, 40e9, 100)
start_kappa_index = 0

peak_time = 50 * tau
peak_spacing_tau = 500
midpoint_window_tau = 10.0
inject_max_etot_only = True
kappa_inj_width = 10 * tau

S_idx = [3 * i + 1 for i in range(N_lasers)]
phi_idx = [3 * i + 2 for i in range(N_lasers)]


def beta_label(beta_value):
    exponent = int(np.round(np.log10(beta_value)))
    return rf"$\beta=10^{{{exponent}}}$"


def beta_file_path(beta_value):
    return data_dir / f"beta_dependence_beta_{beta_value:.0e}_injection_on_noise_on.npz"


def compute_percent_difference(mid_stats):
    y_mean = mid_stats[:, 1]
    y_eq = mid_stats[:, 6]
    with np.errstate(divide="ignore", invalid="ignore"):
        percent_diff = 100.0 * np.abs(y_mean - y_eq) / np.abs(y_eq)
    percent_diff[~np.isfinite(percent_diff)] = np.nan
    return percent_diff


def make_phys(beta_value, kappa_c_mat):
    return {
        'tau_p': tau_p,
        'tau_n': tau_n,
        'g0': g0,
        'N0': N0,
        'N_bar': N0 + 1 / (g0 * tau_p),
        's': s,
        'beta': beta_value,
        'kappa_c_mat': kappa_c_mat,
        'phi_p_mat': np.ones(shape=(n_cases, N_lasers, N_lasers)) * phi_p_vals[:, None, None],
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
        'save_every': 100,
        'injection': True,
    }


def run_one_beta(beta_value):
    extrema = []
    all_eq_points = []
    results = None
    guesses = []

    for k, final_kappa in enumerate(final_kappa_arr[start_kappa_index:], start=start_kappa_index):
        print(f"beta={beta_value:.0e}, kappa index {k + 1}/{len(final_kappa_arr)}")
        kappa_arr = VCSEL.build_coupling_matrix(
            time_arr=time_arr,
            kappa_initial=0,
            kappa_final=final_kappa,
            N_lasers=N_lasers,
            ramp_start=10,
            ramp_shape=50,
            tau=tau,
            scheme=coupling_scheme,
            plot=False,
            dx=dx,
            aMAT=aMAT,
        )

        phys = make_phys(beta_value, kappa_arr[-1, :, :])
        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()

        if results is not None:
            results = np.unique(results, axis=0)
            for eq_pt in results:
                if np.any(np.isnan(eq_pt)):
                    continue
                guesses.append(np.concatenate([
                    eq_pt[1::2][:N_lasers],
                    eq_pt[2 * N_lasers:3 * N_lasers - 1],
                    np.array([eq_pt[-1]]),
                ]))

        history, freq_history, _, _ = vcsel.generate_history(nd, shape='FR', n_cases=n_cases)
        nd['phi_p'] = np.array([phys['phi_p_mat'][0]])[0, :, :]
        eq, results, E_tot = vcsel.solve_equilibria(nd, counts={'phase_count': 20, 'freq_count': 200}, guesses=guesses)
        guesses = []

        if eq is None:
            print(f"No equilibria found for kappa={final_kappa * 1e-9:.2f} ns^-1")
            continue

        nd['phi_p'] = phys['phi_p_mat'][0]
        injection_array = np.zeros(N_lasers)
        injection_array[(N_lasers - 1) // 2] = 1
        phys['injection_topology'] = injection_array
        phys['injected_strength'] = nd['sbar']

        n_delay = 30
        n_eigenvalues = n_delay * 3 * N_lasers - 1
        tmp_stable = Parallel(n_jobs=-1)(
            delayed(vcsel.compute_stability)(
                eq_pt,
                nd,
                N=n_delay,
                newton_maxit=10000,
                threshold=1e-10,
                sparse=phys['sparse'],
                spectral_shift=0.01 + 0.01j,
                n_eigenvalues=n_eigenvalues,
            )
            for eq_pt in results
        )
        tmp_stable = [result[0] for result in tmp_stable]
        if tmp_stable.count(1) == 0:
            print(f"No stable equilibria found for kappa={final_kappa * 1e-9:.2f} ns^-1")
            continue

        stable_indices = np.where(np.asarray(tmp_stable) == 1.0)[0]
        stable_indices = stable_indices[np.argsort(E_tot[stable_indices])]
        injection_indices = stable_indices[-1:] if inject_max_etot_only else stable_indices

        phys['kappa_injection'] = np.zeros((n_cases, len(time_arr)))
        phys['injected_frequency'] = np.zeros(len(time_arr))
        peak_times = []
        omega_targets = []

        kappa_inj_amp_peak = 5 * np.sum(kappa_arr[-1, :, :])
        kappa_inj_amp = np.full(n_cases, kappa_inj_amp_peak)

        for peak_idx, stable_idx in enumerate(injection_indices):
            eq_pt = results[stable_idx]
            omega_target = eq_pt[-1] / (2 * np.pi * 1e9 * tau_p)
            current_peak_time = peak_time + peak_idx * peak_spacing_tau * tau
            gaussian_env = np.exp(-((time_arr - current_peak_time) ** 2) / (2 * kappa_inj_width ** 2))
            phys['kappa_injection'] += kappa_inj_amp[:, None] * gaussian_env[None, :]
            peak_times.append(current_peak_time)
            omega_targets.append(omega_target)

        if omega_targets:
            phys['injected_frequency'][:] = omega_targets[0]
            for idx in range(len(omega_targets) - 1):
                jump_time = 0.5 * (peak_times[idx] + peak_times[idx + 1])
                phys['injected_frequency'][time_arr >= jump_time] = omega_targets[idx + 1]

        phys['injected_phase_diff'] = 0.0
        phys['kappa_c_mat'] = kappa_arr
        vcsel = VCSEL(phys)
        nd = vcsel.scale_params()
        nd['injected_phase_diff'] = np.linspace(0, 2 * np.pi, n_cases)
        nd['phi_p'] = phys['phi_p_mat']

        t, y, freqs = vcsel.integrate(history, nd=nd, progress=True, theta=0.5, max_iter=1, smooth_freqs=True)
        t_idx = np.clip(np.rint(t / dt).astype(int), 0, len(time_arr) - 1)
        S = y[:, S_idx, :]
        phi = y[:, phi_idx, :]

        intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
        E = np.sqrt(S) * np.exp(1j * phi)
        E_tot_cases_mW = (np.abs(E.sum(axis=1)) ** 2) * intensity_to_mW

        sample_times = []
        if len(peak_times) >= 1:
            if len(peak_times) >= 2:
                midpoint_times = 0.5 * (np.asarray(peak_times[:-1]) + np.asarray(peak_times[1:]))
                sample_times.extend(midpoint_times.tolist())
                post_last_time = peak_times[-1] + 0.5 * (peak_times[-1] - peak_times[-2])
            else:
                post_last_time = peak_times[-1] + 0.5 * peak_spacing_tau * tau
            if post_last_time > t[-1]:
                post_last_time = 0.5 * (peak_times[-1] + t[-1])
            sample_times.append(float(post_last_time))

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
            per_case_power = np.mean(E_tot_cases_mW[:, i0:i1], axis=1)
            mean_power = float(np.mean(per_case_power))
            std_power = float(np.std(per_case_power))
            eq_rank = int(mid_idx)
            eq_power_mW = float(eq_power_sorted_mW[min(eq_rank, len(eq_power_sorted_mW) - 1)])
            extrema.append((
                final_kappa * 1e-9,
                mean_power,
                std_power,
                t_mid * 1e6,
                float(mid_idx),
                float(eq_rank),
                eq_power_mW,
            ))

        del y, freqs, S, phi, E, E_tot_cases_mW, history, freq_history
        gc.collect()

    mid_stats = np.asarray(extrema, dtype=float)
    percent_diff = compute_percent_difference(mid_stats) if mid_stats.size else np.array([])
    all_eq_points = np.asarray(all_eq_points, dtype=float)
    return mid_stats, all_eq_points, percent_diff


#%%
if run_beta_sweep and not use_saved_beta_data:
    data_dir.mkdir(parents=True, exist_ok=True)
    for beta_value in beta_arr:
        mid_stats, all_eq_points, percent_diff = run_one_beta(beta_value)
        if save_beta_data:
            save_path = beta_file_path(beta_value)
            np.savez(
                save_path,
                beta=np.asarray(beta_value),
                beta_arr=beta_arr,
                mid_stats=mid_stats,
                all_eq_points=all_eq_points,
                percent_diff_power=percent_diff,
                successful=percent_diff < 5.0,
                N_lasers=np.asarray(N_lasers),
                alpha=np.asarray(alpha),
                detuning_GHz=np.asarray(detuning),
                noise_amplitude=np.asarray(noise_amplitude),
                injection=np.asarray(True),
                inject_max_etot_only=np.asarray(inject_max_etot_only),
                final_kappa_arr=final_kappa_arr,
            )
            print(f"Saved beta dependence data to {save_path}")


#%%
# Four-panel beta dependence plot.
fig, axes = plt.subplots(1, len(beta_arr), figsize=(18, 4.6), dpi=300, sharex=True, sharey=True)
axes = np.atleast_1d(axes)

base_cmap = plt.get_cmap('jet')
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "beta_percent_diff",
    base_cmap(np.linspace(0.2, 0.9, 256)),
)
norm = plt.Normalize(vmin=0.0, vmax=100.0)

for ax, beta_value in zip(axes, beta_arr):
    save_path = beta_file_path(beta_value)
    if not save_path.exists():
        ax.set_title(beta_label(beta_value), fontsize=18)
        ax.text(0.5, 0.5, "No saved data", transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        continue

    data = np.load(save_path, allow_pickle=True)
    mid_stats = np.asarray(data["mid_stats"], dtype=float)
    all_eq_points = np.asarray(data["all_eq_points"], dtype=float)
    percent_diff = np.asarray(data["percent_diff_power"], dtype=float)
    x = mid_stats[:, 0]
    y = mid_stats[:, 1]
    yerr = mid_stats[:, 2]
    color_values = np.nan_to_num(percent_diff, nan=100.0, posinf=100.0, neginf=100.0)

    if all_eq_points.size > 0:
        ax.scatter(
            all_eq_points[:, 0],
            all_eq_points[:, 1],
            s=10,
            marker='o',
            color='black',
            alpha=0.5,
            linewidths=0,
            zorder=1,
        )
    for x_i, y_i, yerr_i, c_i in zip(x, y, yerr, color_values):
        ax.errorbar(
            x_i,
            y_i,
            yerr=yerr_i,
            fmt='none',
            capsize=0,
            elinewidth=0.9,
            alpha=0.55,
            ecolor=cmap(norm(float(c_i))),
            zorder=2,
        )
    ax.scatter(
        x,
        y,
        s=15,
        c=color_values,
        cmap=cmap,
        norm=norm,
        alpha=0.98,
        zorder=3,
    )

    ax.set_title(beta_label(beta_value), fontsize=18)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 4)
    ax.grid(alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=14)

axes[0].set_ylabel(r'$P_{\mathrm{ss,mean}}$ (mW)', fontsize=18)
for ax in axes:
    ax.set_xlabel(r'$\kappa_c$ (ns$^{-1}$)', fontsize=16)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, pad=0.015, shrink=0.9)
cbar.set_label(r'$100|P_{\mathrm{ss,mean}} - P_{\mathrm{eq}}|/|P_{\mathrm{eq}}|$ (\%)', fontsize=16)
cbar.ax.tick_params(labelsize=14)

data_dir.mkdir(parents=True, exist_ok=True)
figure_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
