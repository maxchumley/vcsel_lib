#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.constants import c
from vcsel_lib import VCSEL

from matplotlib import rc
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

detuning = 3.0# detuning (GHz) 

# Basin sampling controls
n_perturb = 1000
omega_span_ghz = 10.0
return_tol_ghz = 0.1
tail_fraction = 0.2
plot_mode = "range"  # "trajectories", "std", or "range"

lam = 910e-9
omega0 = 2*np.pi*c/lam



results = None


for detuning in np.linspace(4, 5, 50):
    delta = detuning * 2 * np.pi * 1e9  # convert GHz to rad/s
    # Create evenly distributed detuning for both even and odd N_lasers
    delta_dist = np.sort(np.concatenate([delta/2 * np.linspace(-1, 1, N_lasers)]))




    phi_p = 0#np.pi

    dt = .5*tau_p# 1 ps
    Tmax = 1e-6
    interpolation = 'cubic'
    steps = int(Tmax / dt)
    time_arr = np.linspace(0, Tmax, steps)
    delay_steps = int(tau / dt)
    ramp_start = 0
    ramp_shape = 0.00001

    final_kappa = np.linspace(0e9, 20e9, 50)[31] * 1.00

    kappa_arr = VCSEL.build_coupling_matrix(time_arr=time_arr, kappa_initial=final_kappa, kappa_final=final_kappa, N_lasers=N_lasers, ramp_start=ramp_start, ramp_shape=ramp_shape, tau=tau, scheme=coupling_scheme, plot=False, dx=dx)





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
        'max_output_gb': 50
    }

    phys['kappa_c_mat'] = kappa_arr#[-1,:,:]



    phys['injection'] = False
    # --- Solve equilibria once ---
    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    counts = {'phase_count': 5, 'freq_count': 5}
    history, freq_history, eq, results = vcsel.generate_history(nd, shape='EQ', n_cases=1, counts=counts)

    if results is None or results.size == 0:
        print(f"No equilibria found for kappa={final_kappa*1e-9:.2f} ns^-1.")
        continue

    # --- Compute stability of equilibria ---
    N = 100
    n_eigenvalues = N * 3 * N_lasers - 1
    stab_results = Parallel(n_jobs=-1)(
        delayed(vcsel.compute_stability)(
            eq_pt,
            nd,
            N=N,
            newton_maxit=10000,
            threshold=1e-10,
            sparse=phys['sparse'],
            spectral_shift=0.01 + 0.01j,
            n_eigenvalues=n_eigenvalues,
        )
        for eq_pt in results
    )
    stable_mask = np.array([res[0] for res in stab_results], dtype=bool)

    stable_indices = np.where(stable_mask)[0].tolist()
    if not stable_indices:
        print("No stable equilibria found.")
        continue

    # --- Prepare plot ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    colors = matplotlib.colormaps["jet"](np.linspace(0.0, 1.0, len(stable_indices)))
    colors[:, :3] = np.clip(colors[:, :3] * 0.7, 0.0, 1.0)
    time_plot = time_arr * 1e6
    t_arr = (np.arange(history.shape[2]) - (history.shape[2] - 1)) * nd["dt"]

    # --- Simulate perturbed trajectories around each stable equilibrium ---
    omega_offsets_ghz = np.linspace(-omega_span_ghz, omega_span_ghz, n_perturb)
    if not np.any(np.isclose(omega_offsets_ghz, 0.0)):
        omega_offsets_ghz = np.sort(np.concatenate([omega_offsets_ghz, [0.0]]))

    for color, eq_idx in zip(colors, stable_indices):
        eq_pt = results[eq_idx]
        omega_eq = eq_pt[-1]
        omega_eq_ghz = omega_eq / (2 * np.pi * 1e9 * tau_p)
        # Build batch of histories with omega perturbations
        n_cases = len(omega_offsets_ghz)
        phase_diff = np.concatenate([[0.0], eq_pt[2 * N_lasers : 3 * N_lasers - 1]])[:N_lasers]
        hist_batch = np.repeat(history[eq_idx:eq_idx + 1, :, :], n_cases, axis=0)
        freq_hist_batch = np.repeat(freq_history[eq_idx:eq_idx + 1, :, :], n_cases, axis=0)
        for i, offset_ghz in enumerate(omega_offsets_ghz):
            omega_target = omega_eq + offset_ghz * 2 * np.pi * 1e9 * tau_p
            phi = omega_target * t_arr + phase_diff.reshape(-1, 1)
            hist_batch[i, 2::3, :] = phi
            freq_hist_batch[i, :, :] = omega_target / (2 * np.pi * 1e9 * tau_p)

        nd_local = dict(nd)
        nd_local["phi_p"] = np.ones(shape=(n_cases, N_lasers, N_lasers)) * phi_p_vals[:, None, None]
        nd_local["delay_interp"] = interpolation
        t, y, freqs = vcsel.integrate(hist_batch, nd=nd_local, progress=True, theta=0.5, max_iter=1)

        dphi = freqs * 1e-9 / (2 * np.pi * tau_p)
        dphi[:, :, :2 * delay_steps] = freq_hist_batch
        dphi_arr = dphi[:, 0, :]

        tail_len = min(1000, dphi_arr.shape[1])
        tail = dphi_arr[:, -tail_len:]
        return_mask = np.all(np.abs(tail - omega_eq_ghz) <= return_tol_ghz, axis=1)

        idx0 = int(np.argmin(np.abs(omega_offsets_ghz)))
        if return_mask[idx0]:
            left = idx0
            right = idx0
            while left - 1 >= 0 and return_mask[left - 1]:
                left -= 1
            while right + 1 < len(return_mask) and return_mask[right + 1]:
                right += 1
            freq_min = omega_offsets_ghz[left]
            freq_max = omega_offsets_ghz[right]
            print(
                f"eq omega={omega_eq_ghz:+.3f} GHz stable for offsets "
                f"[{freq_min:+.3f}, {freq_max:+.3f}] GHz"
            )
        else:
            print(f"eq omega={omega_eq_ghz:+.3f} GHz has no returning offsets.")

        # Plot equilibrium and either all trajectories or the std envelope.
        eq_mask = np.isclose(omega_offsets_ghz, 0.0)
        eq_idx_local = int(np.where(eq_mask)[0][0])
        if return_mask[idx0]:
            range_mask = np.zeros_like(return_mask, dtype=bool)
            range_mask[left:right + 1] = True
        else:
            range_mask = np.zeros_like(return_mask, dtype=bool)
        pert_mask = (~eq_mask) & range_mask
        if np.any(pert_mask):
            if plot_mode == "std":
                pert = dphi_arr[pert_mask, :]
                pert_mean = pert.mean(axis=0)
                pert_std = pert.std(axis=0)
                ax.fill_between(
                    time_plot,
                    pert_mean - pert_std,
                    pert_mean + pert_std,
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    time_plot,
                    pert_mean,
                    color=color,
                    alpha=0.6,
                    linewidth=1.6,
                    zorder=2,
                )
            elif plot_mode == "range":
                pert = dphi_arr[pert_mask, :]
                pert_min = pert.min(axis=0)
                pert_max = pert.max(axis=0)
                ax.fill_between(
                    time_plot,
                    pert_min,
                    pert_max,
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    time_plot,
                    pert_min,
                    color=color,
                    alpha=0.4,
                    linewidth=1.0,
                    zorder=2,
                )
                ax.plot(
                    time_plot,
                    pert_max,
                    color=color,
                    alpha=0.4,
                    linewidth=1.0,
                    zorder=2,
                )
            else:
                for traj in np.where(pert_mask)[0]:
                    ax.plot(
                        time_plot,
                        dphi_arr[traj, :],
                        color=color,
                        alpha=0.6,
                        linewidth=1.0,
                        zorder=1,
                    )
        ax.plot(
            time_plot,
            dphi_arr[eq_idx_local, :],
            color=color,
            alpha=1.0,
            linewidth=2.8,
            zorder=3,
        )
    
    ax.set_ylim(-3,0)
    ax.set_xlabel("Time ($\\mu$s)", fontsize=22)
    ax.set_ylabel(r"$\dot{\phi}$ (GHz)", fontsize=22)
    ax.grid(True, alpha=0.2)
    ax.axvspan(0, 2 * delay_steps * dt * 1e6, color='gray', alpha=0.2)
    ax.set_title(
        rf'$\kappa_c = {final_kappa*1e-9:.2f}\,\mathrm{{ns}}^{{-1}}$',
        fontsize=24,
    )
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    break
