#%%
# Direct simulation of the Ma et al. 2019 mutually injection locked laser model.
#
# This script intentionally does not use vcsel_lib. It implements Equation (1)
# and Equation (2) from:
# Ma et al., "Linewidth Narrowing of Mutually Injection Locked Semiconductor
# Lasers with Short and Long Delay", Applied Sciences 2019.
#
# Target: Figure 4 long-delay case:
# tau_d = 5 ns, kappa_c = 5e8 s^-1, phi_p = 0, delta = 0.3 GHz.

from pathlib import Path
# Make the repositorys ``examples`` package available to scripts and notebooks.
import sys

_path_search_starts = [Path.cwd().resolve()]
if "__file__" in globals():
    _path_search_starts.append(Path(__file__).resolve().parent)
_repo_root = next(
    (candidate for start in _path_search_starts for candidate in (start, *start.parents)
     if (candidate / "examples" / "_paths.py").is_file()),
    None,
)
if _repo_root is None:
    raise ModuleNotFoundError(
        "Could not locate the vcsel_lib repository root containing examples/_paths.py."
    )
sys.path.insert(0, str(_repo_root))
for _module_name in tuple(sys.modules):
    if _module_name == "examples" or _module_name.startswith("examples."):
        del sys.modules[_module_name]


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.optimize import root
from tqdm.auto import tqdm

try:
    from examples._paths import PAPER_DATA_DIR, PAPER_RESULTS_DIR
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples._paths import PAPER_DATA_DIR, PAPER_RESULTS_DIR


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('font', family='serif')


# Table 1 parameters and Figure 4 settings.
q = 1.602e-19
tau_p = 7.15e-12
tau_n = 0.33e-9
g = 1.13e4
N0 = 8.2e6
alpha = 4.0
beta = 3.54e-5

tau_d = 5.0e-9
kappa_c_fig4 = 5.0e8
phi_p = 0.0
detuning_GHz = 0.3
delta = np.array([0.0, 2.0 * np.pi * detuning_GHz * 1e9])



# Controls
run_simulations = True
use_saved_data = False
save_data = True

def get_script_dir():
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    cwd = Path.cwd().resolve()
    if (cwd / "paper_system_test.py").exists():
        return cwd

    repo_style_dir = cwd / "examples"
    if (repo_style_dir / "paper_system_test.py").exists():
        return repo_style_dir

    return cwd


script_dir = get_script_dir()
data_dir = PAPER_DATA_DIR / "paper_system_test"
plot_dir = PAPER_RESULTS_DIR
data_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# The paper's Fig. 4 uses 50 us. Keep this shorter while matching all other
# stated Fig. 4 settings.
Tmax = 5.0e-6

dt = 1.0 * tau_p
max_internal_dt = 0.25 * tau_p
save_every = 100

n_cases = 1
seed = 3
show_progress = True
use_noise = True
noise_start_us = 0.0
noise_expression = "paper_force"  # "paper_force" or "em_increment"
integrator_name = "AB4_EM"
frequency_observable = "phase_gate"  # "phase_gate", "phase_increment", "phase_noise_increment", or "rhs"
frequency_gate_ns = 150.0

plot_ensemble_mean = True
plot_case_index = 0
smooth_frequency_data = False
frequency_smooth_window_ns = 5.0
frequency_smooth_window = max(
    1,
    int(round(frequency_smooth_window_ns * 1e-9 / (save_every * dt))),
)
xlim_us = (0.0, Tmax * 1e6)
mil_initial_history = "locked_equilibrium"


N_th = N0 + 1.0 / (g * tau_p)
I_th = q * N_th / tau_n
I = 4.0 * I_th


def solitary_steady_state():
    """Free-running steady state of the paper equations, including beta."""
    gamma_n = 1.0 / tau_n
    gamma_p = 1.0 / tau_p
    pump = I / q

    # From dN/dt = 0 and dS/dt = 0 after eliminating S:
    # (G - gamma_p)(pump - gamma_n N) + beta gamma_n N G = 0,
    # where G = g(N - N0).
    coeffs = [
        gamma_n * g * (beta - 1.0),
        g * pump + gamma_n * (g * N0 + gamma_p) - beta * gamma_n * g * N0,
        -(g * N0 + gamma_p) * pump,
    ]
    roots = np.roots(coeffs)
    real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-6)].real
    if real_roots.size == 0:
        raise RuntimeError("Could not find a real solitary-laser steady-state carrier.")

    carrier = real_roots[np.argmin(np.abs(real_roots - N_th))]
    gain = g * (carrier - N0)
    photons = (pump - gamma_n * carrier) / gain
    if photons <= 0:
        raise RuntimeError("Solitary-laser steady state has non-positive photon number.")
    return carrier, photons


# Free-running steady state and phase reference from the same paper equations.
N_fr, S_fr = solitary_steady_state()
N_ref = N_fr


def solve_locked_equilibrium(kappa_c):
    """Deterministic two-laser locked state for Fig. 4 initial history."""
    pump = I / q
    omega_target = 2.0 * np.pi * 157.0e6

    def residual(u):
        N = np.array([u[0], u[1]])
        S = np.exp(np.array([u[2], u[3]]))
        theta = u[4]
        omega = u[5]

        phase_offsets = np.array([0.0, theta])
        phase_arg = np.empty(2)
        phase_arg[0] = phase_offsets[1] - phase_offsets[0] - omega * tau_d - phi_p
        phase_arg[1] = phase_offsets[0] - phase_offsets[1] - omega * tau_d - phi_p

        other = np.array([1, 0])
        gain = g * (N - N0)
        sqrt_SS = np.sqrt(S[other] * S)
        sqrt_ratio = np.sqrt(S[other] / S)

        dN = pump - N / tau_n - gain * S
        dS = (
            gain * S
            - S / tau_p
            + beta * N / tau_n
            + 2.0 * kappa_c * sqrt_SS * np.cos(phase_arg)
        )
        dphi = (
            0.5 * alpha * g * (N - N_ref)
            + kappa_c * sqrt_ratio * np.sin(phase_arg)
            + delta
        )

        return np.concatenate([
            dN / pump,
            dS / np.maximum(S / tau_p, 1.0),
            (dphi - omega) / (2.0 * np.pi * 1e9),
        ])

    guesses = []
    for omega_MHz in np.linspace(80.0, 300.0, 23):
        for theta in np.linspace(-np.pi, np.pi, 25, endpoint=False):
            guesses.append(np.array([
                N_fr,
                N_fr,
                np.log(S_fr),
                np.log(S_fr),
                theta,
                2.0 * np.pi * omega_MHz * 1e6,
            ]))

    candidates = []
    for guess in guesses:
        sol = root(residual, guess, method="hybr", tol=1e-10)
        err = np.linalg.norm(residual(sol.x))
        if sol.success and err < 1e-7:
            candidates.append((abs(sol.x[5] - omega_target), err, sol.x))

    if not candidates:
        raise RuntimeError("Could not solve the deterministic MIL equilibrium for Fig. 4.")

    _, best_err, best = min(candidates, key=lambda item: (item[0], item[1]))
    N_eq = np.array([best[0], best[1]])
    S_eq = np.exp(np.array([best[2], best[3]]))
    phase_offsets = np.array([0.0, best[4]])
    omega = best[5]

    print(
        "MIL deterministic initial state: "
        f"omega={omega / (2.0 * np.pi * 1e6):.3f} MHz, "
        f"residual={best_err:.3e}"
    )
    return N_eq, S_eq, phase_offsets, omega


def moving_average(x, window):
    if window <= 1:
        return x
    window = int(window)
    left_pad = (window - 1) // 2
    right_pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(x, (left_pad, right_pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_frequency_array(freq_MHz, window):
    """Smooth frequency data with shape (n_cases, n_lasers, n_times)."""
    if not smooth_frequency_data or window <= 1:
        return np.asarray(freq_MHz, dtype=float).copy()
    freq_MHz = np.asarray(freq_MHz, dtype=float)
    smoothed = np.empty_like(freq_MHz)
    for case_idx in range(freq_MHz.shape[0]):
        for laser_idx in range(freq_MHz.shape[1]):
            smoothed[case_idx, laser_idx, :] = moving_average(freq_MHz[case_idx, laser_idx, :], window)
    return smoothed


def simulate_ma_system(kappa_c, label, initial_history="free_running"):
    rng = np.random.default_rng(seed)

    internal_substeps = max(1, int(np.ceil(dt / max_internal_dt)))
    dt_internal = dt / internal_substeps
    delay_steps = int(round(tau_d / dt_internal))
    tau_d_eff = delay_steps * dt_internal
    if not np.isclose(tau_d_eff, tau_d, rtol=0.0, atol=0.25 * dt_internal):
        print(f"{label}: using effective delay {tau_d_eff * 1e9:.6f} ns")

    external_intervals = int(round(Tmax / dt))
    internal_intervals = external_intervals * internal_substeps
    save_indices = np.arange(0, external_intervals, save_every, dtype=int)
    t_save = (save_indices + 1) * dt
    gate_steps = max(1, int(round(frequency_gate_ns * 1e-9 / dt)))
    gate_time = gate_steps * dt

    freq_save = np.empty((n_cases, 2, save_indices.size), dtype=float)

    if initial_history == "free_running":
        N_initial = np.array([N_fr, N_fr])
        S_initial = np.array([S_fr, S_fr])
        phase_offsets_initial = np.array([0.0, 0.0])
        phase_rates_initial = delta.copy()
    elif initial_history == "zero_field":
        N_initial = np.array([N_fr, N_fr])
        S_initial = np.array([1.0, 1.0])
        phase_offsets_initial = np.array([0.0, 0.0])
        phase_rates_initial = delta.copy()
    elif initial_history == "locked_equilibrium":
        if kappa_c == 0.0:
            raise ValueError("locked_equilibrium history requires nonzero coupling.")
        N_initial, S_initial, phase_offsets_initial, omega_locked = solve_locked_equilibrium(kappa_c)
        phase_rates_initial = np.array([omega_locked, omega_locked])
    else:
        raise ValueError(f"Unknown initial_history: {initial_history}")

    N_curr = np.tile(N_initial[None, :], (n_cases, 1)).astype(float)
    S_curr = np.tile(S_initial[None, :], (n_cases, 1)).astype(float)
    phi_curr = np.tile(phase_offsets_initial[None, :], (n_cases, 1)).astype(float)

    save_ptr = 0
    eps_S = 1.0
    other = np.array([1, 0])
    drift_history = []
    buffer_len = delay_steps + 1
    S_buffer = np.empty((n_cases, 2, buffer_len), dtype=float)
    phi_buffer = np.empty((n_cases, 2, buffer_len), dtype=float)
    S_buffer[:, :, 0] = S_curr
    phi_buffer[:, :, 0] = phi_curr
    phi_gate_buffer = np.empty((n_cases, 2, gate_steps + 1), dtype=float)
    phi_gate_buffer[:, :, 0] = phi_curr

    def compute_drift(Nn, Sn, phin, S_tau, phi_tau):
        phase_arg = phi_tau[:, other] - phi_p - phin
        sqrt_SS = np.sqrt(np.maximum(S_tau[:, other], eps_S) * Sn)
        ratio = np.sqrt(np.maximum(S_tau[:, other], eps_S) / Sn)

        dN = I / q - Nn / tau_n - g * (Nn - N0) * Sn
        dS = (
            g * (Nn - N0) * Sn
            - Sn / tau_p
            + beta * Nn / tau_n
            + 2.0 * kappa_c * sqrt_SS * np.cos(phase_arg)
        )
        dphi = (
            0.5 * alpha * g * (Nn - N_ref)
            + kappa_c * ratio * np.sin(phase_arg)
            + delta[None, :]
        )
        return dN, dS, dphi

    def ab_drift(current_drift):
        drift_history.insert(0, current_drift)
        del drift_history[4:]
        if len(drift_history) == 1:
            coeffs = [1.0]
        elif len(drift_history) == 2:
            coeffs = [3.0 / 2.0, -1.0 / 2.0]
        elif len(drift_history) == 3:
            coeffs = [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0]
        else:
            coeffs = [55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0]

        dN_ab = np.zeros_like(current_drift[0])
        dS_ab = np.zeros_like(current_drift[1])
        dphi_ab = np.zeros_like(current_drift[2])
        for coeff, drift in zip(coeffs, drift_history):
            dN_ab += coeff * drift[0]
            dS_ab += coeff * drift[1]
            dphi_ab += coeff * drift[2]
        return dN_ab, dS_ab, dphi_ab

    def delayed_state(n_internal):
        if n_internal < delay_steps:
            hist_t = (n_internal - delay_steps) * dt_internal
            S_tau = np.tile(S_initial[None, :], (n_cases, 1))
            phi_tau = phase_offsets_initial[None, :] + phase_rates_initial[None, :] * hist_t
        else:
            delay_idx = (n_internal - delay_steps) % buffer_len
            S_tau = S_buffer[:, :, delay_idx]
            phi_tau = phi_buffer[:, :, delay_idx]
        return S_tau, phi_tau

    def gated_phase_start(external_endpoint_idx):
        gate_start_idx = external_endpoint_idx - gate_steps
        if gate_start_idx >= 0:
            return phi_gate_buffer[:, :, gate_start_idx % (gate_steps + 1)]

        gate_start_time = gate_start_idx * dt
        return phase_offsets_initial[None, :] + phase_rates_initial[None, :] * gate_start_time

    print(
        f"{label}: dt={dt / tau_p:.3g} tau_p, "
        f"internal_substeps={internal_substeps}, "
        f"dt_internal={dt_internal / tau_p:.3g} tau_p, "
        f"frequency gate={gate_time * 1e9:.3g} ns ({gate_steps} external steps)"
    )

    external_iter = tqdm(
        range(external_intervals),
        desc=f"Integrating {label}",
        disable=not show_progress,
    )

    for external_idx in external_iter:
        phi_external_start = phi_curr.copy()
        Fp_external_inc = np.zeros_like(phi_curr)
        dphi_det_external_start = None

        for sub_idx in range(internal_substeps):
            n_internal = external_idx * internal_substeps + sub_idx
            S_tau, phi_tau = delayed_state(n_internal)

            Nn = np.maximum(N_curr, 0.0)
            Sn = np.maximum(S_curr, eps_S)
            phin = phi_curr

            dN_det, dS_det, dphi_det = compute_drift(Nn, Sn, phin, S_tau, phi_tau)
            if sub_idx == 0:
                dphi_det_external_start = dphi_det.copy()
            dN_step, dS_step, dphi_step = ab_drift((dN_det, dS_det, dphi_det))

            # Equation (2). The "paper_force" mode writes the noise exactly as
            # the paper does: F = prefactor / sqrt(dt_internal) * xi in the
            # rate equations. The timestep update then uses F * dt_internal.
            noise_active = use_noise and (n_internal * dt_internal >= noise_start_us * 1e-6)
            if noise_active:
                x1 = rng.standard_normal((n_cases, 2))
                x2 = rng.standard_normal((n_cases, 2))
                x3 = rng.standard_normal((n_cases, 2))

                if noise_expression == "paper_force":
                    Fn_force = (
                        np.sqrt(2.0 * Nn / (tau_n * dt_internal)) * x1
                        - np.sqrt(2.0 * beta * Nn * Sn / (tau_n * dt_internal)) * x2
                    )
                    Fs_force = np.sqrt(2.0 * beta * Nn * Sn / (tau_n * dt_internal)) * x2
                    Fp_force = np.sqrt(beta * Nn / (2.0 * tau_n * Sn * dt_internal)) * x3

                    Fn_inc = Fn_force * dt_internal
                    Fs_inc = Fs_force * dt_internal
                    Fp_inc = Fp_force * dt_internal
                elif noise_expression == "em_increment":
                    Fn_inc = (
                        np.sqrt(2.0 * Nn / tau_n) * x1
                        - np.sqrt(2.0 * beta * Nn * Sn / tau_n) * x2
                    ) * np.sqrt(dt_internal)
                    Fs_inc = np.sqrt(2.0 * beta * Nn * Sn / tau_n) * x2 * np.sqrt(dt_internal)
                    Fp_inc = np.sqrt(beta * Nn / (2.0 * tau_n * Sn)) * x3 * np.sqrt(dt_internal)
                else:
                    raise ValueError(f"Unknown noise_expression: {noise_expression}")
            else:
                Fn_inc = np.zeros_like(Nn)
                Fs_inc = np.zeros_like(Sn)
                Fp_inc = np.zeros_like(phin)

            N_next = Nn + dN_step * dt_internal + Fn_inc
            S_next = np.maximum(Sn + dS_step * dt_internal + Fs_inc, eps_S)
            phi_next = phin + dphi_step * dt_internal + Fp_inc

            if not (np.all(np.isfinite(N_next)) and np.all(np.isfinite(S_next)) and np.all(np.isfinite(phi_next))):
                raise FloatingPointError(
                    f"{label}: non-finite state at external step {external_idx}, "
                    f"internal substep {sub_idx}. Try decreasing max_internal_dt."
                )

            N_curr = N_next
            S_curr = S_next
            phi_curr = phi_next
            Fp_external_inc += Fp_inc

            write_idx = (n_internal + 1) % buffer_len
            S_buffer[:, :, write_idx] = S_curr
            phi_buffer[:, :, write_idx] = phi_curr

        external_endpoint_idx = external_idx + 1
        phi_gate_buffer[:, :, external_endpoint_idx % (gate_steps + 1)] = phi_curr

        if save_ptr < save_indices.size and external_idx == save_indices[save_ptr]:
            if frequency_observable == "phase_gate":
                phi_gate_start = gated_phase_start(external_endpoint_idx)
                freq_save[:, :, save_ptr] = (
                    (phi_curr - phi_gate_start) / gate_time / (2.0 * np.pi) * 1e-6
                )
            elif frequency_observable == "phase_increment":
                freq_save[:, :, save_ptr] = (
                    (phi_curr - phi_external_start) / dt / (2.0 * np.pi) * 1e-6
                )
            elif frequency_observable == "phase_noise_increment":
                freq_save[:, :, save_ptr] = Fp_external_inc / dt / (2.0 * np.pi) * 1e-6
            elif frequency_observable == "rhs":
                freq_save[:, :, save_ptr] = dphi_det_external_start / (2.0 * np.pi) * 1e-6
            else:
                raise ValueError(f"Unknown frequency_observable: {frequency_observable}")
            save_ptr += 1

    sigma_MHz = np.std(freq_save, axis=2)
    return {
        "label": label,
        "t": t_save,
        "freq_MHz": freq_save,
        "sigma_MHz": sigma_MHz,
        "kappa_c": kappa_c,
        "dt": dt,
        "dt_internal": dt_internal,
        "max_internal_dt": max_internal_dt,
        "internal_substeps": internal_substeps,
        "frequency_gate_ns": frequency_gate_ns,
        "frequency_gate_steps": gate_steps,
        "frequency_gate_time": gate_time,
        "save_every": save_every,
        "delay_steps": delay_steps,
        "internal_intervals": internal_intervals,
        "integrator": integrator_name,
        "frequency_observable": frequency_observable,
        "use_noise": use_noise,
        "noise_start_us": noise_start_us,
        "noise_expression": noise_expression,
        "initial_history": initial_history,
        "N_ref": N_ref,
        "N_fr": N_fr,
        "S_fr": S_fr,
    }



noise_label = "noise_off"
if use_noise:
    noise_label = f"noise_start_{noise_start_us:g}us".replace(".", "p")
phase_ref_label = "solitary_ref"
noise_expression_label = noise_expression.replace(".", "p")
internal_dt_label = f"maxint_{max_internal_dt / tau_p:g}taup".replace(".", "p")
frequency_gate_label = f"gate_{frequency_gate_ns:g}ns".replace(".", "p")
mil_history_label = f"mil_{mil_initial_history}"
data_path = data_dir / (
    f"fig4_direct_{integrator_name}_{frequency_observable}_Tmax_{Tmax:.1e}_dt_{dt:.1e}_"
    f"saveevery_{save_every}_cases_{n_cases}_{noise_label}_{noise_expression_label}_"
    f"{frequency_gate_label}_{internal_dt_label}_{mil_history_label}_{phase_ref_label}.npz"
)

if use_saved_data and data_path.exists():
    loaded = np.load(data_path, allow_pickle=True)
    free_data = loaded["free"].item()
    mil_data = loaded["mil"].item()
elif run_simulations:
    free_data = simulate_ma_system(0.0, "free_running")
    mil_data = simulate_ma_system(
        kappa_c_fig4,
        "mutually_injection_locked",
        initial_history=mil_initial_history,
    )
    if save_data:
        np.savez(data_path, free=free_data, mil=mil_data)
        print(f"Saved {data_path.resolve()}")
else:
    raise RuntimeError("No data available. Set run_simulations=True or use_saved_data=True.")






#%%

# Figure 4-style three-panel plot.
t_free_us = free_data["t"] * 1e6
t_mil_us = mil_data["t"] * 1e6

# Smooth each noise realization first; only then average across cases.
free_freq_smoothed = smooth_frequency_array(free_data["freq_MHz"], frequency_smooth_window)
mil_freq_smoothed = smooth_frequency_array(mil_data["freq_MHz"], frequency_smooth_window)
free_sigma_data = np.std(free_freq_smoothed, axis=2)
mil_sigma_data = np.std(mil_freq_smoothed, axis=2)

if plot_ensemble_mean:
    free_freq = np.mean(free_freq_smoothed, axis=0)
    mil_freq = np.mean(mil_freq_smoothed, axis=0)
    plotted_sigma_free = np.std(free_freq, axis=1)
    plotted_sigma_mil = np.std(mil_freq, axis=1)
    plotted_label = f"ensemble mean over {free_data['freq_MHz'].shape[0]} cases"
else:
    free_freq = free_freq_smoothed[plot_case_index].copy()
    mil_freq = mil_freq_smoothed[plot_case_index].copy()
    plotted_sigma_free = free_sigma_data[plot_case_index]
    plotted_sigma_mil = mil_sigma_data[plot_case_index]
    plotted_label = f"case {plot_case_index}"
if smooth_frequency_data:
    effective_smooth_ns = frequency_smooth_window * save_every * dt * 1e9
    plotted_label += (
        f", smoothed over {effective_smooth_ns:g} ns "
        f"({frequency_smooth_window} saved samples)"
    )

fig, axs = plt.subplots(3, 1, figsize=(8.0, 4.8), dpi=300, sharex=True)

axs[0].plot(t_free_us, free_freq[0], color="blue", linestyle="-", linewidth=1.0, label="Laser\\#1 free-running")
axs[1].plot(t_free_us, free_freq[1], color="green", linestyle="-", linewidth=1.0, label="Laser\\#2 free-running")
axs[2].plot(t_mil_us, mil_freq[0], color="black", linestyle="-", linewidth=1.0, label="Laser\\#1 MIL")
axs[2].plot(t_mil_us, mil_freq[1], color="red", linestyle=":", linewidth=1.2, label="Laser\\#2 MIL")

fig.supylabel("Frequency Fluctuation (MHz)", fontsize=22)
axs[2].set_xlabel(r"Time ($\mu$s)", fontsize=22)

axs[0].set_ylim(-10, 10)
axs[0].set_yticks([-5, 0, 5])
axs[1].set_ylim(290, 310)
axs[1].set_yticks([295, 300, 305])
axs[2].set_ylim(155, 159)
axs[2].set_yticks([156, 157, 158])


for ax in axs:
    ax.set_xlim(*xlim_us)
    ax.tick_params(axis="both", which="major", labelsize=16, direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.legend(
        loc="lower left",
        fontsize=13,
        frameon=True,
        facecolor="white",
        framealpha=0.5,
        edgecolor="none",
    )

fig.subplots_adjust(hspace=0.0, left=0.16, right=0.98, bottom=0.16, top=0.98)

print(f"Temporal sigma of plotted direct-paper frequency observable ({frequency_observable}, {plotted_label}):")
print(f"  Free-running DFB#1: {plotted_sigma_free[0]:.3g} MHz")
print(f"  Free-running DFB#2: {plotted_sigma_free[1]:.3g} MHz")
print(f"  MIL DFB#1:          {plotted_sigma_mil[0]:.3g} MHz")
print(f"  MIL DFB#2:          {plotted_sigma_mil[1]:.3g} MHz")

print("Mean temporal sigma of individual noise realizations:")
print(f"  Free-running DFB#1: {np.mean(free_sigma_data[:, 0]):.3g} MHz")
print(f"  Free-running DFB#2: {np.mean(free_sigma_data[:, 1]):.3g} MHz")
print(f"  MIL DFB#1:          {np.mean(mil_sigma_data[:, 0]):.3g} MHz")
print(f"  MIL DFB#2:          {np.mean(mil_sigma_data[:, 1]):.3g} MHz")
print("Paper Fig. 4 reported temporal sigma:")
print("  Free-running either laser: 1.92 MHz")
print("  MIL:                       0.136 MHz")

fig_path = plot_dir / "paper_system_test_fig4_direct.png"
fig.savefig(fig_path, bbox_inches="tight", facecolor="white")
print(f"Saved {fig_path.resolve()}")
plt.show()
