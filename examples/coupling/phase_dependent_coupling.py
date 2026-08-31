# %%
"""Small VCSEL-array example with a simple phase-mask coupling model.

The phase mask is represented by a pairwise optical transfer phase phi_ij and
an optional pairwise phase-dependent coupling magnitude.  This replaces the old
"global order" feedback rule.

Coupling term implemented through vcsel_lib's existing LK form:

    sum_j kappa_ij(t) E_j(t - tau) exp(-1j * phi_ij(t))

Modes:
    static_mask:
        kappa_ij(t) = ramp(t) * kappa0 * A_ij
        phi_ij(t)  = theta_star[j] - theta_star[i]

    dynamic_phase_mask:
        kappa_ij(t) = ramp(t) * kappa0 * A_ij
        phi_ij(t)  = theta_star[j] - theta_star[i]
                     + eps_phi * sin(Omega*t + psi_ij)

    phase_dependent_coupling:
        mismatch_ij(t) = wrap(phi_j(t-tau) - phi_i(t)
                              - (theta_star[j] - theta_star[i]))
        kappa_ij(t) = ramp(t) * kappa0 * A_ij
                      * [floor + (1-floor)
                         * exp(-mismatch_ij(t)^2/(2*sigma_phi^2))]
        phi_ij(t)  = theta_star[j] - theta_star[i]

For N=2:
    THETA_STAR = [0, 0]     favors in-phase locking.
    THETA_STAR = [0, np.pi] favors out-of-phase locking.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


rc("text", usetex=True)
rc("font", family="serif")


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from examples._paths import COUPLING_RESULTS_DIR

RESULTS_DIR = COUPLING_RESULTS_DIR / "phase_dependent_coupling"

from vcsel_lib import VCSEL


# ------------------------- knobs to play with -------------------------
N_LASERS = 6
N_NOISE_ITERATIONS = 50
RANDOM_SEED = None
KAPPA_MAX = 1.0e9          # strongest individual directed link, s^-1
COUPLING_RULE = "phase_dependent_coupling"  # "static_mask", "dynamic_phase_mask", "phase_dependent_coupling"
COUPLING_FLOOR = 0.0      # fraction of KAPPA_MAX for phase_dependent_coupling
PHASE_WIDTH = 1.0e10       # radians for phase_dependent_coupling Gaussian

# Target locked phase pattern.  For N=2, use [0, 0] or [0, np.pi].
THETA_STAR = np.linspace(0, np.pi, N_LASERS, endpoint=False)
THETA_STAR = 0*np.pi*np.ones(N_LASERS)  # uncomment to favor all in-phase
# THETA_STAR[:N_LASERS // 2] = np.pi/2

# Optional time-dependent target phase program.  Leave as None to use the fixed
# THETA_STAR above for the full simulation.  Each listed time begins a smooth
# transition to the new target, which is then held until the next listed time.
#
# Example:
THETA_STAR_PROGRAM = [
    (0.0e-9, np.zeros(N_LASERS)),
    (1000.0e-9, np.linspace(-np.pi, np.pi, N_LASERS, endpoint=False)),
    (3000.0e-9, np.r_[np.full(N_LASERS // 2, np.pi), np.zeros(N_LASERS - N_LASERS // 2)]),
]
# THETA_STAR_PROGRAM = None
THETA_STAR_TRANSITION_TIME = 25.0e-9  # raised-cosine transition; use 0 for steps



# Optional time-varying phase mask settings.
EPS_PHI = 0.10             # radians
OMEGA_MASK_GHZ = 0.25      # modulation frequency in GHz cycles/s, not rad/s

# Smooth coupling ramp.  This avoids an artificial DDE startup jump.
RAMP_START = 10.0e-9
RAMP_TIME = 5.0e-8

DETUNING_SPAN_GHZ = 0.0
NOISE_AMPLITUDE = 1.0
TMAX = 50.0e-7 
SAVE_EVERY = 2

# Phase-circle frame export.  Frames are sampled from the saved integration
# output, not from every internal DDE step.
PHASE_CIRCLE_FRAME_DT_NS = 10.0
PHASE_CIRCLE_FRAME_DPI = 140


def wrap_phase(x):
    """Wrap angles to [-pi, pi]."""
    return np.angle(np.exp(1j * x))


def cosine_ramp(t, start, rise_time):
    """Smooth ramp from 0 to 1 over rise_time."""
    if rise_time <= 0.0:
        return (t >= start).astype(float)
    u = np.clip((t - start) / rise_time, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * u))


def target_phase_matrix(theta_star):
    """target[i,j] = desired phase of source j relative to receiver i."""
    theta_star = np.asarray(theta_star, dtype=float)
    return wrap_phase(theta_star[None, :] - theta_star[:, None])


def target_phase_matrix_time_series(theta_star_t):
    """Vectorized target matrices for theta_star_t with shape (Nt, N_lasers)."""
    theta_star_t = np.asarray(theta_star_t, dtype=float)
    return wrap_phase(theta_star_t[:, None, :] - theta_star_t[:, :, None])


def theta_star_time_series(time_arr):
    """Return smoothly programmed target phases with shape (Nt, N_lasers)."""
    if THETA_STAR_PROGRAM is None:
        theta_star = np.asarray(THETA_STAR, dtype=float)
        if theta_star.shape != (N_LASERS,):
            raise ValueError("THETA_STAR must have shape (N_LASERS,).")
        return np.repeat(theta_star[None, :], time_arr.size, axis=0)

    program = sorted(
        [(float(start_time), np.asarray(theta, dtype=float)) for start_time, theta in THETA_STAR_PROGRAM],
        key=lambda item: item[0],
    )
    if not program:
        raise ValueError("THETA_STAR_PROGRAM cannot be empty.")
    for _, theta in program:
        if theta.shape != (N_LASERS,):
            raise ValueError("Each THETA_STAR_PROGRAM pattern must have shape (N_LASERS,).")
    if THETA_STAR_TRANSITION_TIME < 0.0:
        raise ValueError("THETA_STAR_TRANSITION_TIME cannot be negative.")

    theta_t = np.repeat(program[0][1][None, :], time_arr.size, axis=0)
    for start_time, theta in program[1:]:
        start_index = int(np.searchsorted(time_arr, start_time, side="left"))
        if start_index >= time_arr.size:
            continue

        # Begin from the pattern already active at this instant.  This also
        # keeps the program continuous if a new transition starts before the
        # preceding transition has completely finished.
        theta_before = theta_t[start_index].copy()
        theta_t[time_arr >= start_time] = theta
        if THETA_STAR_TRANSITION_TIME > 0.0:
            in_transition = (
                (time_arr >= start_time)
                & (time_arr < start_time + THETA_STAR_TRANSITION_TIME)
            )
            blend = cosine_ramp(
                time_arr[in_transition],
                start_time,
                THETA_STAR_TRANSITION_TIME,
            )
            shortest_delta = wrap_phase(theta - theta_before)
            theta_t[in_transition] = (
                theta_before[None, :]
                + blend[:, None] * shortest_delta[None, :]
            )
    return theta_t


class PhaseMaskCouplingVCSEL(VCSEL):
    """VCSEL model with a simple pairwise phase-mask transfer matrix."""

    def f_nd(self, x, x_tau, x_2tau, j, phi_p, nd=None):
        if nd is None:
            nd = self.nd

        n_lasers = x.shape[1] // 3
        rule = nd.get("phase_mask_rule", "static_mask")
        theta_ij = np.asarray(nd["phase_mask_theta_ij"])
        if theta_ij.ndim == 3:
            theta_ij = theta_ij[min(j, theta_ij.shape[0] - 1)]

        # Base kappa is already scaled by vcsel_lib.  It may be a time series.
        base_kappa = np.asarray(nd["kappa"])
        if base_kappa.ndim == 3:
            base_kappa = base_kappa[min(j, base_kappa.shape[0] - 1)]
        elif base_kappa.ndim == 0:
            base_kappa = np.full((n_lasers, n_lasers), float(base_kappa))

        # Smooth startup ramp in dimensionless time index space.
        ramp = np.asarray(nd["phase_mask_ramp"])
        ramp_j = float(ramp[min(j, ramp.size - 1)])
        kappa_eff = ramp_j * base_kappa
        phi_eff = np.array(theta_ij, copy=True)

        if rule == "dynamic_phase_mask":
            eps_phi = float(nd.get("phase_mask_eps_phi", 0.0))
            omega = float(nd.get("phase_mask_omega", 0.0))
            psi = np.asarray(nd.get("phase_mask_psi_ij", 0.0))
            t_now = j * float(nd["dt"])
            phi_eff = wrap_phase(theta_ij + eps_phi * np.sin(omega * t_now + psi))

        elif rule == "phase_dependent_coupling":
            phase_i = x[:, 2::3]
            phase_j_tau = x_tau[:, 2::3]

            # mismatch[:, i, j] = delayed source phase j - receiver phase i - target_ij
            mismatch = wrap_phase(phase_j_tau[:, None, :] - phase_i[:, :, None] - theta_ij)
            sigma = float(nd.get("phase_mask_phase_width", 1.0))
            floor = float(nd.get("phase_mask_floor", 0.0))
            if sigma <= 0.0:
                raise ValueError("PHASE_WIDTH must be positive.")

            weight = floor + (1.0 - floor) * np.exp(-0.5 * (mismatch / sigma) ** 2)
            kappa_eff = kappa_eff[None, :, :] * weight
            nd_dynamic = dict(nd)
            nd_dynamic["kappa"] = kappa_eff
            nd_dynamic["kappa_case_dependent"] = True
            return super().f_nd(x, x_tau, x_2tau, j, phi_eff, nd=nd_dynamic)

        elif rule != "static_mask":
            raise ValueError(
                "COUPLING_RULE must be 'static_mask', 'dynamic_phase_mask', "
                "or 'phase_dependent_coupling'."
            )

        nd_dynamic = dict(nd)
        nd_dynamic["kappa"] = kappa_eff
        nd_dynamic["kappa_case_dependent"] = False
        return super().f_nd(x, x_tau, x_2tau, j, phi_eff, nd=nd_dynamic)


def compute_plot_coupling(t, phases, final_kappa, theta_ij, tau):
    """Recreate the effective kappa and phi used by the model for plotting.

    Accepts either one trajectory with shape (N_lasers, Nt), or a vectorized
    ensemble with shape (n_cases, N_lasers, Nt).
    """
    phases = np.asarray(phases)
    single_case = phases.ndim == 2
    if single_case:
        phases = phases[None, :, :]

    ramp = cosine_ramp(t, RAMP_START, RAMP_TIME)
    n_cases, n_lasers, n_t = phases.shape
    theta_ij = np.asarray(theta_ij)
    if theta_ij.ndim == 2:
        theta_plot = np.broadcast_to(theta_ij[None, :, :], (n_t, n_lasers, n_lasers))
    elif theta_ij.ndim == 3:
        if theta_ij.shape[0] == n_t:
            theta_plot = theta_ij
        else:
            source_t = np.linspace(t[0], t[-1], theta_ij.shape[0])
            source_indices = np.searchsorted(source_t, t, side="right") - 1
            source_indices = np.clip(source_indices, 0, theta_ij.shape[0] - 1)
            theta_plot = theta_ij[source_indices]
    else:
        raise ValueError("theta_ij must have shape (N,N) or (Nt,N,N).")

    phi_eff = np.broadcast_to(
        theta_plot[None, :, :, :].transpose(0, 2, 3, 1),
        (n_cases, n_lasers, n_lasers, n_t),
    ).copy()
    kappa_eff = np.broadcast_to(
        final_kappa[None, :, :, None] * ramp[None, None, None, :],
        (n_cases, n_lasers, n_lasers, n_t),
    ).copy()

    if COUPLING_RULE == "dynamic_phase_mask":
        omega = 2.0 * np.pi * OMEGA_MASK_GHZ * 1e9
        psi_ij = np.zeros((n_lasers, n_lasers))
        phi_eff = wrap_phase(
            theta_plot[None, :, :, :].transpose(0, 2, 3, 1)
            + EPS_PHI
            * np.sin(omega * t[None, None, None, :] + psi_ij[None, :, :, None])
        )

    elif COUPLING_RULE == "phase_dependent_coupling":
        dt_save = t[1] - t[0]
        delay_steps = int(round(tau / dt_save))
        delayed = np.empty_like(phases)
        if delay_steps <= 0:
            delayed[...] = phases
        else:
            delayed[:, :, : min(delay_steps, n_t)] = phases[:, :, [0]]
            if delay_steps < n_t:
                delayed[:, :, delay_steps:] = phases[:, :, :-delay_steps]

        mismatch = wrap_phase(
            delayed[:, None, :, :]
            - phases[:, :, None, :]
            - theta_plot[None, :, :, :].transpose(0, 2, 3, 1)
        )
        weight = COUPLING_FLOOR + (1.0 - COUPLING_FLOOR) * np.exp(
            -0.5 * (mismatch / PHASE_WIDTH) ** 2
        )
        kappa_eff = (
            final_kappa[None, :, :, None]
            * ramp[None, None, None, :]
            * weight
        )

    if single_case:
        return kappa_eff[0], phi_eff[0]
    return kappa_eff, phi_eff


def mean_and_std(samples, axis=0):
    """Return nan-safe mean/std for a noise ensemble."""
    samples = np.asarray(samples)
    return np.nanmean(samples, axis=axis), np.nanstd(samples, axis=axis)


def circular_mean_and_std(angles, axis=0):
    """Circular mean and circular standard deviation for wrapped phases."""
    phasor = np.nanmean(np.exp(1j * np.asarray(angles)), axis=axis)
    mean = np.angle(phasor)
    resultant_length = np.clip(np.abs(phasor), np.finfo(float).tiny, 1.0)
    std = np.sqrt(-2.0 * np.log(resultant_length))
    return mean, std


def plot_mean_std(
    ax,
    t_ns,
    samples,
    label,
    color=None,
    linestyle="-",
    alpha=0.18,
    line_alpha=1.0,
    linewidth=1.5,
    zorder=None,
):
    """Plot mean ± one standard deviation across noise realizations."""
    mean, std = mean_and_std(samples, axis=0)
    (line,) = ax.plot(
        t_ns,
        mean,
        label=label,
        color=color,
        linestyle=linestyle,
        alpha=line_alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
    fill_color = line.get_color() if color is None else color
    ax.fill_between(
        t_ns,
        mean - std,
        mean + std,
        color=fill_color,
        alpha=alpha,
        linewidth=0,
    )
    return mean, std


def plot_circular_mean_std(ax, t_ns, samples, label, color=None, alpha=0.18):
    """Plot circular mean ± circular standard deviation for wrapped phases."""
    mean, std = circular_mean_and_std(samples, axis=0)
    (line,) = ax.plot(t_ns, mean, label=label, color=color)
    fill_color = line.get_color() if color is None else color
    ax.fill_between(
        t_ns,
        np.clip(mean - std, -np.pi, np.pi),
        np.clip(mean + std, -np.pi, np.pi),
        color=fill_color,
        alpha=alpha,
        linewidth=0,
    )
    return mean, std


def laser_color(index):
    """Readable repeated color cycle for larger arrays."""
    return plt.get_cmap("tab20")(index % 20)


def save_phase_circle_frames(
    output_dir,
    t,
    phases,
    theta_star_saved,
    frame_dt_ns=10.0,
    dpi=140,
):
    """Save polar frames of mean phase differences and circular std envelopes.

    The plotted angles are phase differences relative to laser 1.  Each laser's
    radial spoke points to the circular mean over noise realizations, and the
    translucent outer arc spans mean ± one circular standard deviation.
    """
    if frame_dt_ns <= 0.0:
        raise ValueError("PHASE_CIRCLE_FRAME_DT_NS must be positive.")

    frame_dir = output_dir / "phase_circle_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    relative = wrap_phase(phases - phases[:, 0:1, :])
    mean_rel, std_rel = circular_mean_and_std(relative, axis=0)
    target_rel = wrap_phase(theta_star_saved - theta_star_saved[:, 0:1])

    t_ns = t * 1e9
    frame_times_ns = np.arange(t_ns[0], t_ns[-1] + 0.5 * frame_dt_ns, frame_dt_ns)
    frame_indices = np.searchsorted(t_ns, frame_times_ns, side="left")
    frame_indices = np.clip(frame_indices, 0, t.size - 1)
    frame_indices = np.unique(frame_indices)

    theta_grid = np.linspace(0.0, 2.0 * np.pi, 720)
    unit = np.ones_like(theta_grid)

    for frame_number, time_index in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(6.0, 6.0), subplot_kw={"projection": "polar"})
        ax.plot(theta_grid, unit, color="0.78", linewidth=0.8)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_ylim(0.0, 1.18)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["", ""])
        ax.set_xticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$-\pi/2$"])
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Mean phase differences at t = {t_ns[time_index]:.1f} ns")

        for laser in range(N_LASERS):
            color = laser_color(laser)
            angle = mean_rel[laser, time_index]
            spread = std_rel[laser, time_index]
            target_angle = target_rel[time_index, laser]

            arc = np.linspace(angle - spread, angle + spread, 80)
            ax.fill_between(
                arc,
                0.88,
                1.06,
                color=color,
                alpha=0.18 if laser != 0 else 0.10,
                linewidth=0,
            )
            ax.plot(
                [angle, angle],
                [0.0, 1.0],
                color=color,
                linewidth=2.0 if laser != 0 else 2.8,
                alpha=0.90,
                label=f"laser {laser + 1}",
            )
            ax.scatter(
                [angle],
                [1.0],
                color=[color],
                s=36 if laser != 0 else 52,
                zorder=5,
            )
            ax.scatter(
                [target_angle],
                [1.10],
                marker="x",
                color=[color],
                s=34,
                linewidths=1.4,
                alpha=0.85,
            )

        ax.text(
            0.02,
            0.02,
            "spokes: mean phase vs laser 1\nbands: ±1 circular std\nx: target THETA_STAR difference",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.23, 1.12),
            fontsize=8,
            framealpha=0.9,
        )
        fig.tight_layout()
        fig.savefig(frame_dir / f"phase_circle_{frame_number:05d}.png", dpi=dpi)
        plt.close(fig)

    return frame_dir, frame_indices.size


def main():
    # Standard device parameters used by the other vcsel_lib examples.
    tau_p = 5.4e-12
    tau_n = 0.25e-9
    g0 = 8.75e-4 * 1e9
    N0 = 2.86e5
    gain_saturation = 4e-6
    beta = 1e-3
    alpha = 2.0
    q = 1.602e-19
    tau = 1.0e-9
    eta = 0.9
    current_threshold = 3.0
    current = eta * current_threshold * q / tau_n * (N0 + 1.0 / (g0 * tau_p))

    dt = tau_p
    steps = int(TMAX / dt)
    time_arr = np.arange(steps) * dt

    adjacency = np.ones((N_LASERS, N_LASERS)) - np.eye(N_LASERS)
    final_kappa = adjacency * KAPPA_MAX
    theta_star_t = theta_star_time_series(time_arr)
    theta_ij_t = target_phase_matrix_time_series(theta_star_t)
    theta_ij_initial = theta_ij_t[0]
    print(theta_ij_initial)

    # Store the maximum/base kappa as a time series because vcsel_lib examples
    # expect kappa_c_mat.  The subclass applies the ramp and any phase rule.
    kappa_time_series = np.repeat(final_kappa[None, :, :], time_arr.size, axis=0)

    detuning_rad_s = (
        np.linspace(-0.5, 0.5, N_LASERS) * DETUNING_SPAN_GHZ * 2.0 * np.pi * 1e9
    )

    phys = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "s": gain_saturation,
        "beta": beta,
        "kappa_c_mat": kappa_time_series,
        "phi_p_mat": theta_ij_initial,
        "I": current,
        "q": q,
        "alpha": alpha,
        "delta": detuning_rad_s,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "noise_amplitude": NOISE_AMPLITUDE,
        "dt": dt,
        "Tmax": TMAX,
        "tau": tau,
        "N_lasers": N_LASERS,
        "save_every": SAVE_EVERY,
    }

    vcsel = PhaseMaskCouplingVCSEL(phys)
    nd = vcsel.scale_params()
    nd["phase_mask_rule"] = COUPLING_RULE
    nd["phase_mask_theta_ij"] = theta_ij_t
    nd["phase_mask_ramp"] = cosine_ramp(time_arr, RAMP_START, RAMP_TIME)
    nd["phase_mask_floor"] = COUPLING_FLOOR
    nd["phase_mask_phase_width"] = PHASE_WIDTH
    nd["phase_mask_eps_phi"] = EPS_PHI
    nd["phase_mask_omega"] = 2.0 * np.pi * OMEGA_MASK_GHZ * 1e9 * tau_p
    nd["phase_mask_psi_ij"] = np.zeros_like(theta_ij_initial)

    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    history, _, _, _ = vcsel.generate_history(
        nd,
        shape="FR",
        n_cases=N_NOISE_ITERATIONS,
    )
    t, y, _ = vcsel.integrate(history, nd=nd, progress=True)
    saved_indices = np.clip(np.rint(t / dt).astype(int), 0, time_arr.size - 1)
    theta_star_saved = theta_star_t[saved_indices]
    theta_ij_saved = theta_ij_t[saved_indices]

    photons = y[:, 1::3, :]
    phases = y[:, 2::3, :]
    relative_phase = wrap_phase(phases - phases[:, 0:1, :])
    phase_coherence = np.abs(np.mean(np.exp(1j * phases), axis=1))
    fields = np.sqrt(np.maximum(photons, 1e-12)) * np.exp(1j * phases)
    total_field_intensity = np.abs(np.sum(fields, axis=1)) ** 2
    global_order = np.abs(np.sum(fields, axis=1)) / np.maximum(
        np.sum(np.abs(fields), axis=1), 1e-12
    )

    # Pairwise target-locking error; this replaces the old R_eff variable while
    # keeping the plotting layout the same.
    pair_errors = []
    for i in range(N_LASERS):
        for j in range(i + 1, N_LASERS):
            target_delta = theta_star_saved[:, j] - theta_star_saved[:, i]
            pair_errors.append(
                wrap_phase(
                    (phases[:, j, :] - phases[:, i, :])
                    - target_delta[None, :]
                )
            )
    if pair_errors:
        locking_error = np.mean(np.abs(np.stack(pair_errors, axis=1)), axis=1)
    else:
        locking_error = np.zeros((N_NOISE_ITERATIONS, t.size), dtype=float)
    effective_global_order = 1.0 - np.clip(locking_error / np.pi, 0.0, 1.0)
    reference_order_plot = 0.0

    dynamic_kappa, dynamic_phi = compute_plot_coupling(
        t,
        phases,
        final_kappa,
        theta_ij_saved,
        tau,
    )
    dynamic_kappa_mean, dynamic_kappa_std = mean_and_std(dynamic_kappa, axis=0)
    dynamic_phi_mean, _ = circular_mean_and_std(dynamic_phi, axis=0)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "simulation.npz",
        time_s=t,
        photons=photons,
        phases=phases,
        total_field_intensity=total_field_intensity,
        phase_coherence=phase_coherence,
        global_order=global_order,
        effective_global_order=effective_global_order,
        locking_error=locking_error,
        theta_star=theta_star_saved,
        theta_star_program=np.asarray(THETA_STAR_PROGRAM, dtype=object)
        if THETA_STAR_PROGRAM is not None
        else np.asarray([], dtype=object),
        phi_p=dynamic_phi_mean.astype(np.float32, copy=False),
        coupling_rule=COUPLING_RULE,
        n_noise_iterations=N_NOISE_ITERATIONS,
        dynamic_kappa_mean_rad_s=dynamic_kappa_mean.astype(np.float32, copy=False),
        dynamic_kappa_std_rad_s=dynamic_kappa_std.astype(np.float32, copy=False),
    )

    # Keep the plotting structure the same as the original example.
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    t_ns = t * 1e9
    top_max = 0.0
    for laser in range(N_LASERS):
        color = laser_color(laser)
        mean, std = plot_mean_std(
            axes[0],
            t_ns,
            photons[:, laser, :],
            label=f"laser {laser + 1}",
            color=color,
            alpha=0.08,
            line_alpha=0.78,
            linewidth=1.1,
            zorder=2,
        )
        top_max = max(top_max, float(np.nanmax(mean + std)))
        plot_mean_std(
            axes[1],
            t_ns,
            np.abs(relative_phase[:, laser, :]),
            label=rf"$|\phi_{{{laser + 1}}}-\phi_1|$",
            color=color,
            alpha=0.08,
            line_alpha=0.78,
            linewidth=1.1,
            zorder=2,
        )
    total_mean, total_std = plot_mean_std(
        axes[0],
        t_ns,
        total_field_intensity,
        label=r"$|\sum_i E_i|^2$",
        color="black",
        alpha=0.10,
        line_alpha=1.0,
        linewidth=2.0,
        zorder=5,
    )
    top_max = max(top_max, float(np.nanmax(total_mean + total_std)))
    axes[0].set_ylabel("Photon state S")
    axes[0].legend(ncol=3, fontsize=8, framealpha=0.9)
    axes[0].set_ylim(0.0, 1.05 * top_max)
    axes[1].set_ylabel(r"$|\phi_i-\phi_1|$ (rad)")
    axes[1].set_ylim(-0.05, 1.05*np.pi)
    axes[1].set_yticks([0.0, 0.5 * np.pi, np.pi])
    axes[1].set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
    axes[1].legend(ncol=3, fontsize=8, framealpha=0.9)
    plot_mean_std(
        axes[2],
        t_ns,
        effective_global_order,
        color="black",
        label="target score",
        alpha=0.10,
        linewidth=2.0,
        zorder=5,
    )
    plot_mean_std(
        axes[2],
        t_ns,
        global_order,
        color="tab:blue",
        linestyle="--",
        alpha=0.10,
        linewidth=1.5,
        label=rf"$R$; $R_0={reference_order_plot:.3f}$",
    )
    plot_mean_std(
        axes[2],
        t_ns,
        phase_coherence,
        color="gray",
        linestyle=":",
        alpha=0.10,
        linewidth=1.5,
        label="phase only",
    )
    axes[2].set_ylabel("Order")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(loc="lower right")
    for source in range(1, N_LASERS):
        plot_mean_std(
            axes[3],
            t_ns,
            dynamic_kappa[:, 0, source, :] * 1e-9,
            label=rf"$\kappa_{{1,{source + 1}}}$",
            color=laser_color(source),
            alpha=0.08,
            line_alpha=0.85,
            linewidth=1.1,
        )
    axes[3].set_ylabel(r"$\kappa_{1j}$ (ns$^{-1}$)")
    axes[3].set_xlabel("Time (ns)")
    axes[3].legend(ncol=max(1, min(4, N_LASERS - 1)), fontsize=8, framealpha=0.9)
    # axes[3].set_ylim(0.9995,1.0005)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "time_series.png", dpi=180)
    plt.show()

    print("Final instantaneous coupling strengths (ns^-1):")
    print(np.round(dynamic_kappa_mean[:, :, -1] * 1e-9, 3))
    if N_NOISE_ITERATIONS > 1:
        print("Final instantaneous coupling std across noise cases (ns^-1):")
        print(np.round(dynamic_kappa_std[:, :, -1] * 1e-9, 3))
    print("Final phase-mask phases phi_ij (rad):")
    print(np.round(dynamic_phi_mean[:, :, -1], 3))
    final_error_mean, final_error_std = mean_and_std(locking_error[:, -1])
    print(
        "Final mean target locking error: "
        f"{final_error_mean:.4f} ± {final_error_std:.4f} rad"
    )
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()


# %% Save phase-circle frames from the last saved simulation
# Run this cell after the simulation cell above.  It loads simulation.npz and
# writes PNG frames to results/phase_dependent_coupling/phase_circle_frames/.
if "get_ipython" in globals():
    simulation_file = RESULTS_DIR / "simulation.npz"
    if simulation_file.exists():
        with np.load(simulation_file) as saved:
            frame_dir, n_frames = save_phase_circle_frames(
                RESULTS_DIR,
                saved["time_s"],
                saved["phases"],
                saved["theta_star"],
                frame_dt_ns=PHASE_CIRCLE_FRAME_DT_NS,
                dpi=PHASE_CIRCLE_FRAME_DPI,
            )
        print(f"Saved {n_frames} phase-circle frames to {frame_dir}")
    else:
        print(f"No saved simulation found yet: {simulation_file}")


# %% Plot phase-dependent coupling magnitude
# Small explanatory plot for slides: coupling is strongest when the actual
# delayed phase mismatch Delta_ij approaches zero.
if "get_ipython" in globals():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    demo_delta = np.linspace(-np.pi, np.pi, 600)
    demo_floor = COUPLING_FLOOR+0.1
    demo_sigma = PHASE_WIDTH
    if demo_sigma > 2.0 * np.pi:
        demo_sigma = 0.35 * np.pi

    demo_weight = demo_floor + (1.0 - demo_floor) * np.exp(
        -0.5 * (demo_delta / demo_sigma) ** 2
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(
        demo_delta,
        demo_weight,
        color="tab:blue",
        linewidth=2.5,
        label=(
            r"$f+(1-f)\exp[-\Delta_{ij}^2/(2\sigma_\phi^2)]$"
        ),
    )
    ax.axvline(0.0, color="black", linewidth=1.2, alpha=0.75)
    ax.scatter([0.0], [1.0], color="tab:red", s=55, zorder=5)
    ax.annotate(
        r"maximum coupling at $\Delta_{ij}=0$",
        xy=(0.0, 1.0),
        xytext=(0.25 * np.pi, 0.9),
        arrowprops={"arrowstyle": "->", "color": "0.25"},
        fontsize=10,
    )
    sigma_y = demo_floor + (1.0 - demo_floor) * np.exp(-0.5)
    # ax.axvline(demo_sigma, color="tab:orange", linestyle=":", linewidth=1.6)
    ax.annotate(
        "",
        xy=(-demo_sigma, sigma_y),
        xytext=(demo_sigma, sigma_y),
        arrowprops={
            "arrowstyle": "<->",
            "color": "tab:red",
            "linewidth": 2.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    ax.text(
        -0.4,
        sigma_y - 0.07,
        rf"$2\sigma_\phi$",
        color="tab:red",
        fontsize=10,
        ha="center",
        va="top",
    )
    ax.axhline(demo_floor, color="0.45", linestyle="--", linewidth=1.1)
    ax.text(
        -0.98 * np.pi,
        demo_floor - 0.05,
        rf"floor $f={demo_floor:.2f}$",
        color="0.35",
        fontsize=9,
    )
    ax.set_xlabel(r"phase mismatch $\Delta_{ij}$ (rad)")
    ax.set_ylabel(r"normalized coupling $\kappa_{ij}/(r\kappa_0 A_{ij})$")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9)
    ax.set_title(r"Phase-selective coupling magnitude")
    fig.tight_layout()
    # fig.savefig(RESULTS_DIR / "phase_dependent_coupling_curve.png", dpi=180)
    plt.show()
