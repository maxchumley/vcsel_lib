#%%
"""Validate the effect of free-running initial phase offsets for two VCSELs.

This is intentionally close to ``simple_example.py``.  The simulator first
receives a free-running history, then the mutual coupling is smoothly ramped
from zero to its final value.  Set ``initial_phase_offsets`` below and rerun
the simulation cell to test a different initial relative phase.

The offsets are in radians.  A common offset applied to both lasers is only a
global phase reference and should not change the dynamics; the relative
offset, ``initial_phase_offsets[1] - initial_phase_offsets[0]``, is the
quantity that can change which transient or locked branch is reached.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.constants import c, hbar

try:
    from vcsel_lib import VCSEL
except ModuleNotFoundError:  # Allow direct execution from examples/basics.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vcsel_lib import VCSEL

# Derive the output directory locally so direct execution cannot be affected
# by an unrelated installed package also named ``examples``.
BASICS_RESULTS_DIR = Path(__file__).resolve().parent / "results"


rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rc("font", family="serif")


# ---------------------------- Parameters ----------------------------
# Physical VCSEL parameters (the same values used by simple_example.py).
alpha = 2.0
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4e-6
q = 1.602e-19
beta = 1.0e-3
tau = 1e-9
eta = 0.9
current_threshold = 3.0
I = eta * current_threshold * q / tau_n * (N0 + 1.0 / (g0 * tau_p))

lam = 910e-9
omega0 = 2.0 * np.pi * c / lam

# This validation is deliberately a two-laser experiment.
N_lasers = 2

# Detuning is specified in GHz relative to the 0-GHz mean frequency.
detuning_span_ghz = 0.5
delta_dist_ghz = 0.5 * detuning_span_ghz * np.linspace(-1.0, 1.0, N_lasers)
delta_dist = 2.0 * np.pi * 1e9 * delta_dist_ghz  # rad/s for vcsel_lib

# Set these values (radians) before running the simulation cell.
# Examples: [0, 0] is equal phase; [0, pi] is an initial anti-phase offset.
initial_phase_offsets = np.array([0.0, np.pi], dtype=float)

dt = 1.0 * tau_p
Tmax_requested = 2.0e-7
kappa_c_final = 20e9
noise_amplitude = 0.0

# The coupling is zero during the free-running history and then follows a
# smooth cosine ramp.  Both times are measured in units of the delay tau.
ramp_start_delays = 3.5
ramp_rise_delays = 100.0
coupling_scheme = "CUSTOM"
aMAT = np.ones((N_lasers, N_lasers), dtype=float) - np.eye(N_lasers)
phi_p_values = np.zeros((N_lasers, N_lasers), dtype=float)


def run_simulation(phase_offsets=None):
    """Run one two-laser free-running-to-coupled simulation.

    Parameters
    ----------
    phase_offsets : array_like, shape (2,)
        Additive phase offsets (radians) for lasers 1 and 2.  The offsets are
        applied to every sample of the free-running history, not just its last
        sample, so the delayed state is internally consistent.

    Returns
    -------
    dict
        Time series, coupling ramp, and quantities needed by the plotting cell.
    """
    if phase_offsets is None:
        phase_offsets = initial_phase_offsets
    phase_offsets = np.asarray(phase_offsets, dtype=float)
    if phase_offsets.shape != (N_lasers,):
        raise ValueError(
            f"phase_offsets must have shape ({N_lasers},), got {phase_offsets.shape}"
        )

    # Match the simple example while ensuring the coupling and time arrays
    # cover the same physical interval.
    steps = int(np.round(Tmax_requested / dt)) + 1
    time_arr = np.arange(steps, dtype=float) * dt
    Tmax = float(time_arr[-1])
    delay_steps = int(np.round(tau / dt))

    kappa_arr = VCSEL.build_coupling_matrix(
        time_arr=time_arr,
        kappa_initial=0.0,
        kappa_final=kappa_c_final,
        N_lasers=N_lasers,
        ramp_start=ramp_start_delays,
        ramp_shape=ramp_rise_delays,
        tau=tau,
        scheme=coupling_scheme,
        aMAT=aMAT,
    )

    phys = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "N_bar": N0 + 1.0 / (g0 * tau_p),
        "s": s,
        "beta": beta,
        "kappa_c_mat": kappa_arr,
        "phi_p_mat": phi_p_values[None, :, :],
        "I": I,
        "q": q,
        "alpha": alpha,
        "delta": delta_dist,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "noise_amplitude": noise_amplitude,
        "dt": dt,
        "Tmax": Tmax,
        "tau": tau,
        "N_lasers": N_lasers,
        "save_every": 1,
    }

    vcsel = VCSEL(phys)
    nd = vcsel.scale_params()
    history, freq_hist, _, _ = vcsel.generate_history(
        nd, shape="FR", n_cases=1
    )

    # generate_history creates phi_i(t) = delta_i*t.  Add the requested
    # offsets to all history samples so both delayed rounds carry the offset.
    history[:, 2::3, :] += phase_offsets[None, :, None]

    # Suppress only floating-point warnings emitted by unstable trial states;
    # the returned arrays are still checked for non-finite values below.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        t, y, freqs = vcsel.integrate(
            history,
            nd=nd,
            progress=True,
            max_iter=1,
            smooth_freqs=False,
        )

    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(freqs))):
        raise FloatingPointError(
            "The simulation returned a non-finite state. Reduce kappa_c_final "
            "or the detuning span and try again."
        )

    # State layout is [n_1, S_1, phi_1, n_2, S_2, phi_2].
    S_all = y[:, 1::3, :]
    phi_all = y[:, 2::3, :]

    # VCSEL.integrate returns nondimensional dphi/dt. Convert it to GHz and
    # restore the known free-running frequency over the history interval.
    dphi_all = freqs * 1e-9 / (2.0 * np.pi * tau_p)
    history_freq_ghz = freq_hist[:, :, : dphi_all.shape[2]]
    dphi_all[:, :, : history_freq_ghz.shape[2]] = history_freq_ghz

    wrapped_relative_phase = np.angle(
        np.exp(1j * (phi_all[:, 1, :] - phi_all[:, 0, :]))
    )

    intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
    individual_power_mW = S_all * intensity_to_mW
    E_all = np.sqrt(np.maximum(S_all, 0.0)) * np.exp(1j * phi_all)
    total_power_mW = np.abs(np.sum(E_all, axis=1)) ** 2 * intensity_to_mW

    # The integrator may return one fewer sample than the matrix ramp because
    # vcsel_lib computes its step count from Tmax/dt.  Slice to the time axis.
    kappa_plot_ns_inv = kappa_arr[: len(t), 0, 1] * 1e-9

    return {
        "t": t,
        "y": y,
        "freqs_ghz": dphi_all,
        "phases": phi_all,
        "relative_phase": wrapped_relative_phase,
        "individual_power_mW": individual_power_mW,
        "total_power_mW": total_power_mW,
        "kappa_ns_inv": kappa_plot_ns_inv,
        "delay_steps": delay_steps,
        "dt": dt,
        "phase_offsets": phase_offsets,
        "delta_dist_ghz": delta_dist_ghz,
    }


# ----------------------------- Simulation ----------------------------
# Change `initial_phase_offsets` above and rerun this cell to compare cases.
result = run_simulation(initial_phase_offsets)


#  ------------------------------- Plotting -----------------------------
def plot_result(result):
    """Make a simple_example-style three-panel trajectory plot."""
    t = result["t"]
    time_plot = t * 1e9  # ns
    dphi_ghz = result["freqs_ghz"][0]
    relative_phase = result["relative_phase"][0]
    individual_power_mW = result["individual_power_mW"][0]
    total_power_mW = result["total_power_mW"][0]
    kappa_plot_ns_inv = result["kappa_ns_inv"]
    delay_steps = result["delay_steps"]
    dt = result["dt"]
    offsets = result["phase_offsets"]
    detunings = result["delta_dist_ghz"]

    fig, axs = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)

    # 1) Instantaneous phase derivatives.
    for laser in range(N_lasers):
        axs[0].plot(
            time_plot,
            dphi_ghz[laser],
            linewidth=2,
            label=rf"$\dot{{\phi}}_{laser + 1}$",
        )
    axs[0].set_ylabel(r"$\dot{\phi}$ (GHz)", fontsize=22)
    axs[0].set_title(
        rf"Free-running phase offsets: $(\phi_1(0),\phi_2(0))="
        rf"({offsets[0] / np.pi:.2g}\pi,{offsets[1] / np.pi:.2g}\pi)$; "
        rf"$\delta=({detunings[0]:.2g},{detunings[1]:.2g})$ GHz",
        fontsize=23,
        pad=16,
    )
    axs[0].legend(loc="upper right", fontsize=14)
    axs[0].grid(True, alpha=0.2)

    # 2) Wrapped relative phase, with the free-running history highlighted.
    axs[1].plot(
        time_plot,
        relative_phase,
        color="tab:purple",
        linewidth=2,
        label=r"$\mathrm{wrap}(\phi_2-\phi_1)$",
    )
    axs[1].set_ylabel(r"$\phi_2-\phi_1$ (rad)", fontsize=22)
    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].set_yticks([-np.pi, 0.0, np.pi])
    axs[1].set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    axs[1].legend(loc="upper right", fontsize=14)
    axs[1].grid(True, alpha=0.2)

    # 3) Individual and coherent total output power, plus the coupling ramp.
    for laser in range(N_lasers):
        axs[2].plot(
            time_plot,
            individual_power_mW[laser],
            linewidth=2,
            label=rf"$P_{laser + 1}$",
        )
    axs[2].plot(
        time_plot,
        total_power_mW,
        color="green",
        linewidth=2.5,
        label=r"$P_{\rm total}$",
    )
    axs[2].set_xlabel(r"Time (ns)", fontsize=22)
    axs[2].set_ylabel("Output power (mW)", fontsize=22)
    axs[2].legend(loc="upper right", fontsize=14)
    axs[2].grid(True, alpha=0.2)

    # The first 2*tau is the supplied delay history.  Coupling starts later,
    # then rises smoothly according to the cosine ramp.
    history_end_ns = 2.0 * delay_steps * dt * 1e9
    ramp_start_ns = ramp_start_delays * tau * 1e9
    axs[0].axvspan(0.0, history_end_ns, color="gray", alpha=0.2)
    axs[1].axvspan(0.0, history_end_ns, color="gray", alpha=0.2)
    axs[2].axvspan(0.0, history_end_ns, color="gray", alpha=0.2)
    for ax in axs:
        ax.axvline(
            ramp_start_ns,
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.65,
        )
        ax.tick_params(axis="both", which="major", labelsize=18)

    coupling_axis = axs[2].twinx()
    coupling_axis.plot(
        time_plot,
        kappa_plot_ns_inv,
        "k--",
        linewidth=2,
        alpha=0.7,
        label=r"$\kappa_{12}$",
    )
    coupling_axis.set_ylabel(r"Coupling $\kappa_{12}$ (ns$^{-1}$)", fontsize=20)
    coupling_axis.tick_params(axis="y", labelsize=18)
    coupling_axis.set_ylim(0.0, max(1.0, 1.1 * np.max(kappa_plot_ns_inv)))

    fig.suptitle(
        "Two-laser free-running initialization with a coupling ramp",
        fontsize=26,
        y=0.995,
    )
    plt.tight_layout()

    plot_dir = BASICS_RESULTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plot_dir / "initial_phase_validation_summary.png"
    fig.savefig(figure_path, bbox_inches="tight")
    return fig, figure_path


fig, figure_path = plot_result(result)
print(f"Saved figure to {figure_path}")
plt.show()
