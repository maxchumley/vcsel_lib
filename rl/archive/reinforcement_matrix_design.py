# %% Imports and settings
"""Minimal REINFORCE example for designing kappa and phi_p matrices."""

from contextlib import nullcontext
import multiprocessing as mp
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from vcsel_lib import VCSEL

from rl.paths import DATA_DIR


# ----------------- Desired behavior -----------------
N_lasers = 5

# Phases relative to laser 1. Use [0, 0] for in-phase, [0, pi] for anti-phase,
# or arange(N_lasers)*2*pi/N_lasers for a splay state.
# target_phases = np.linspace(-np.pi,np.pi, N_lasers)#np.array(5*[0.0, 0.0])
target_phases = (
    np.arange(N_lasers) * 2.0 * np.pi / N_lasers
)

phase_weight = 0.7
frequency_lock_weight = 0.3
frequency_tolerance_ghz = 0.10
# Within the phase term, split the score between the population-wide average
# and the weakest non-reference laser. Increase this if one bad laser remains.
worst_laser_phase_weight = 0.5


# ----------------- Reinforcement learning -----------------
n_restarts = 1
parallel_restarts = 1
n_iterations = 300                    # iterations per independent restart
population_size = 80                     # must be even
random_seed = 7
learning_rate = 0.05
policy_std = 0.8
policy_std_decay = 0.98
minimum_policy_std = 0.10

maximum_kappa_ns = 25.0
initial_kappa_fraction = 0.25
tail_fraction = 0.35
coupling_ramp_start_tau = 2.0
coupling_ramp_time_tau = 5.0

# Entry [i,j] sends delayed source j into receiver i. The zero diagonal avoids
# self-coupling; set other entries to zero to remove links from the design.
aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)

show_inline_plot = True
output_file = DATA_DIR / "reinforcement_matrix_design.npz"


# ----------------- Physical setup -----------------
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
I = eta * 3.0 * q / tau_n * (N0 + 1.0 / (g0 * tau_p))

detuning_span_ghz = 2.0
delta = (
    np.linspace(-0.5, 0.5, N_lasers)
    * detuning_span_ghz
    * 2.0
    * np.pi
    * 1e9
)

dt = tau_p
Tmax = 500.0 * tau
save_every = 10


# Two small helpers
def decode_actions(actions, adjacency, kappa_max_ns):
    """Convert policy actions to bounded kappa and wrapped phi_p matrices."""
    actions = np.asarray(actions, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    link_i, link_j = np.where(adjacency > 0)
    n_links = len(link_i)
    if actions.ndim != 2 or actions.shape[1] != 2 * n_links:
        raise ValueError(f"actions must have shape (batch, {2 * n_links})")

    kappa_links = kappa_max_ns / (
        1.0 + np.exp(-np.clip(actions[:, :n_links], -30.0, 30.0))
    )
    phi_links = np.angle(np.exp(1j * actions[:, n_links:]))

    n_cases = actions.shape[0]
    n_lasers = adjacency.shape[0]
    kappa_ns = np.zeros((n_cases, n_lasers, n_lasers))
    phi_p = np.zeros_like(kappa_ns)
    kappa_ns[:, link_i, link_j] = kappa_links
    phi_p[:, link_i, link_j] = phi_links
    return kappa_ns, phi_p


def trajectory_rewards(states, frequencies_ghz, tail_start, reward_config=None):
    """Reward target phases, including the weakest non-reference laser."""
    phases = states[:, 2::3, tail_start:]
    frequencies = frequencies_ghz[:, :, tail_start:]

    if reward_config is None:
        desired_phases = np.asarray(target_phases, dtype=float)
        worst_weight = worst_laser_phase_weight
        phase_reward_weight = phase_weight
        frequency_reward_weight = frequency_lock_weight
        tolerance_ghz = frequency_tolerance_ghz
    else:
        desired_phases = np.asarray(reward_config["target_phases"], dtype=float)
        worst_weight = reward_config["worst_laser_phase_weight"]
        phase_reward_weight = reward_config["phase_weight"]
        frequency_reward_weight = reward_config["frequency_lock_weight"]
        tolerance_ghz = reward_config["frequency_tolerance_ghz"]
    if desired_phases.ndim == 1:
        desired = np.angle(
            np.exp(1j * (desired_phases - desired_phases[0]))
        )[None, :]
    elif (
        desired_phases.ndim == 2
        and desired_phases.shape == phases.shape[:2]
    ):
        desired = np.angle(
            np.exp(1j * (desired_phases - desired_phases[:, :1]))
        )
    else:
        raise ValueError(
            "target_phases must have shape (N_lasers,) or (n_cases, N_lasers)"
        )
    relative_phases = np.angle(np.exp(1j * (phases - phases[:, :1])))
    phase_error = np.angle(
        np.exp(1j * (relative_phases - desired[:, :, None]))
    )
    per_laser_phase_score = np.mean(
        0.5 * (1.0 + np.cos(phase_error[:, 1:])), axis=2
    )
    mean_phase_score = np.mean(per_laser_phase_score, axis=1)
    worst_phase_score = np.min(per_laser_phase_score, axis=1)
    phase_score = (
        (1.0 - worst_weight) * mean_phase_score
        + worst_weight * worst_phase_score
    )

    common_frequency = frequencies.mean(axis=1, keepdims=True)
    frequency_spread = np.sqrt(
        np.mean((frequencies - common_frequency) ** 2, axis=(1, 2))
    )
    frequency_score = np.exp(
        -np.minimum((frequency_spread / tolerance_ghz) ** 2, 100.0)
    )
    reward = (
        phase_reward_weight * phase_score
        + frequency_reward_weight * frequency_score
    ) / (phase_reward_weight + frequency_reward_weight)

    finite = np.all(np.isfinite(states[:, :, tail_start:]), axis=(1, 2))
    finite &= np.all(np.isfinite(frequencies), axis=(1, 2))
    reward = np.where(finite & np.isfinite(reward), reward, -1.0)
    return (
        reward,
        phase_score,
        mean_phase_score,
        worst_phase_score,
        frequency_score,
        frequency_spread,
    )


def _evaluate_population(
    phys,
    nd,
    history,
    training_coupling_ramp,
    kappa_ns,
    phi_p,
    config,
):
    """Simulate and score one restart's candidate population."""
    vcsel = VCSEL(phys)
    nd_run = dict(nd)
    nd_run["kappa"] = (
        kappa_ns * 1e9 * config["tau_p"]
    )  # ns^-1 -> nondimensional
    nd_run["kappa_case_dependent"] = True
    nd_run["phi_p"] = phi_p
    nd_run["kappa_ramp"] = training_coupling_ramp
    t, states, frequencies = vcsel.integrate(
        history.copy(), nd=nd_run, progress=False, max_iter=1
    )
    frequencies_ghz = frequencies / (2.0 * np.pi * config["tau_p"] * 1e9)
    ramp_end_time = (
        config["coupling_ramp_start_tau"]
        + config["coupling_ramp_time_tau"] / 0.8
    ) * config["tau"]
    tail_start = max(
        int((1.0 - config["tail_fraction"]) * states.shape[2]),
        int(np.searchsorted(t, ramp_end_time)),
    )
    if tail_start >= states.shape[2] - 1:
        raise ValueError("Tmax must include time after the coupling ramp")
    return trajectory_rewards(
        states, frequencies_ghz, tail_start, reward_config=config
    )


def _simulate_population_outcomes(
    phys,
    nd,
    history,
    coupling_ramp,
    kappa_ns,
    phi_p,
    config,
):
    """Simulate coupling designs and return settled phase/frequency features.

    This importable worker is also used by the model-based neural example so
    multiprocessing remains reliable when that example is run from IPython.
    """
    vcsel = VCSEL(phys)
    nd_run = dict(nd)
    nd_run["kappa"] = kappa_ns * 1e9 * config["tau_p"]
    nd_run["kappa_case_dependent"] = True
    nd_run["phi_p"] = phi_p
    nd_run["kappa_ramp"] = coupling_ramp
    t, states, frequencies = vcsel.integrate(
        history.copy(), nd=nd_run, progress=False, max_iter=1
    )
    frequencies_ghz = frequencies / (
        2.0 * np.pi * config["tau_p"] * 1e9
    )
    ramp_end_time = (
        config["coupling_ramp_start_tau"]
        + config["coupling_ramp_time_tau"] / 0.8
    ) * config["tau"]
    tail_start = max(
        int((1.0 - config["tail_fraction"]) * states.shape[2]),
        int(np.searchsorted(t, ramp_end_time)),
    )
    if tail_start >= states.shape[2] - 1:
        raise ValueError("Tmax must include time after the coupling ramp")

    phases = states[:, 2::3, tail_start:]
    relative = np.angle(
        np.exp(1j * (phases - phases[:, :1]))
    )[:, 1:]
    circular_mean = np.mean(np.exp(1j * relative), axis=2)
    frequency_tail = frequencies_ghz[:, :, tail_start:]
    phase_coherence = np.abs(circular_mean)
    unit_phase = circular_mean / np.maximum(phase_coherence, 1e-8)
    settled_frequency = frequency_tail.mean(axis=2)
    relative_frequency = settled_frequency[:, 1:] - settled_frequency[:, :1]
    # Each value is signed and continuous. Zero means that laser is locked to
    # laser 1; values approach +/-1 as its frequency mismatch grows.
    normalized_frequency_difference = relative_frequency / (
        np.abs(relative_frequency) + config["frequency_tolerance_ghz"]
    )
    outcomes = np.c_[
        unit_phase.real,
        unit_phase.imag,
        phase_coherence,
        normalized_frequency_difference,
    ]
    finite = np.all(
        np.isfinite(states[:, :, tail_start:]), axis=(1, 2)
    )
    finite &= np.all(np.isfinite(frequency_tail), axis=(1, 2))
    outcomes[~finite] = 0.0
    return outcomes


def _plot_parallel_progress(results):
    """Plot all completed restarts on their shared iteration axis."""
    ordered = sorted(results, key=lambda result: result["restart"])
    best_result = max(ordered, key=lambda result: result["best_reward"])
    completed_iterations = len(ordered[0]["mean_reward_history"])
    iteration_axis = np.arange(1, completed_iterations + 1)
    restart_best = np.vstack(
        [result["best_reward_history"] for result in ordered]
    )
    winner = np.argmax(restart_best, axis=0)
    column = np.arange(completed_iterations)
    global_best = restart_best[winner, column]
    worst_phase = np.vstack(
        [result["best_worst_phase_history"] for result in ordered]
    )[winner, column]

    clear_output(wait=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for result in ordered:
        line = axes[0].plot(
            iteration_axis,
            result["mean_reward_history"],
            alpha=0.65,
            label=f"restart {result['restart']} mean",
        )[0]
        axes[0].plot(
            iteration_axis,
            result["best_reward_history"],
            color=line.get_color(),
            linestyle=":",
            alpha=0.7,
        )
    axes[0].plot(iteration_axis, global_best, "k", linewidth=2, label="global best")
    axes[0].plot(
        iteration_axis,
        worst_phase,
        "k--",
        linewidth=1.5,
        label="global-best worst phase",
    )
    axes[0].set(
        xlabel="iteration within restart",
        ylabel="score",
        title="RL progress (dotted = restart best)",
    )
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    kappa_plot = axes[1].imshow(
        best_result["best_kappa_ns"].T,
        origin="upper",
        vmin=0,
        vmax=maximum_kappa_ns,
    )
    axes[1].set(
        title=r"Best $\kappa$ (ns$^{-1}$)",
        xlabel="receiver",
        ylabel="source",
    )
    fig.colorbar(kappa_plot, ax=axes[1])

    phi_plot = axes[2].imshow(
        best_result["best_phi_p"].T,
        origin="upper",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[2].set(
        title=r"Best $\phi_p$ (rad)",
        xlabel="receiver",
        ylabel="source",
    )
    fig.colorbar(phi_plot, ax=axes[2])
    fig.suptitle(
        f"iteration {completed_iterations}/{n_iterations} | "
        f"{len(ordered)} parallel restarts | "
        f"best restart {best_result['restart']} | "
        f"mean phase {best_result['best_mean_phase_score']:.3f} | "
        f"worst phase {best_result['best_worst_phase_score']:.3f} | "
        f"frequency {best_result['best_frequency_score']:.3f}"
    )
    if "agg" not in plt.get_backend().lower():
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


# Direct simulation and REINFORCE loop
def run_training():
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if population_size < 2 or population_size % 2:
        raise ValueError("population_size must be an even number of at least 2")
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")
    if parallel_restarts < 1:
        raise ValueError("parallel_restarts must be at least 1")
    if np.asarray(target_phases).shape != (N_lasers,):
        raise ValueError("target_phases must have shape (N_lasers,)")
    if phase_weight + frequency_lock_weight <= 0:
        raise ValueError("At least one reward weight must be positive")
    if not 0.0 <= worst_laser_phase_weight <= 1.0:
        raise ValueError("worst_laser_phase_weight must be between 0 and 1")

    link_i, link_j = np.where(aMAT > 0)
    n_links = len(link_i)
    initial_kappa_logit = np.log(
        initial_kappa_fraction / (1.0 - initial_kappa_fraction)
    )
    initial_policy_mean = np.r_[
        np.full(n_links, initial_kappa_logit),
        np.zeros(n_links),
    ]

    phys = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "s": s,
        "beta": beta,
        "kappa_c_mat": np.zeros((N_lasers, N_lasers)),
        "phi_p_mat": np.zeros((N_lasers, N_lasers)),
        "I": I,
        "q": q,
        "alpha": alpha,
        "delta": delta,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "noise_amplitude": 0.0,
        "dt": dt,
        "Tmax": Tmax,
        "tau": tau,
        "N_lasers": N_lasers,
        "save_every": save_every,
    }

    setup_vcsel = VCSEL(phys)
    nd = setup_vcsel.scale_params()
    history, _, _, _ = setup_vcsel.generate_history(
        nd, shape="FR", n_cases=population_size
    )
    training_time = np.arange(nd["steps"]) * dt
    training_coupling_ramp = VCSEL.cosine_ramp(
        training_time,
        t_start=coupling_ramp_start_tau * tau,
        rise_10_90=coupling_ramp_time_tau * tau,
        kappa_initial=0.0,
        kappa_final=1.0,
    )
    restart_seeds = np.arange(
        random_seed, random_seed + n_restarts, dtype=np.int64
    )
    training_config = {
        "n_iterations": n_iterations,
        "population_size": population_size,
        "policy_std": policy_std,
        "policy_std_decay": policy_std_decay,
        "minimum_policy_std": minimum_policy_std,
        "learning_rate": learning_rate,
        "adjacency": np.asarray(aMAT),
        "maximum_kappa_ns": maximum_kappa_ns,
        "tau_p": tau_p,
        "tau": tau,
        "tail_fraction": tail_fraction,
        "coupling_ramp_start_tau": coupling_ramp_start_tau,
        "coupling_ramp_time_tau": coupling_ramp_time_tau,
        "target_phases": np.asarray(target_phases),
        "worst_laser_phase_weight": worst_laser_phase_weight,
        "phase_weight": phase_weight,
        "frequency_lock_weight": frequency_lock_weight,
        "frequency_tolerance_ghz": frequency_tolerance_ghz,
    }

    worker_count = min(parallel_restarts, n_restarts)
    print(
        f"Launching {n_restarts} independent restarts on "
        f"{worker_count} worker processes."
    )
    rngs = [np.random.default_rng(seed) for seed in restart_seeds]
    policy_means = [initial_policy_mean.copy() for _ in range(n_restarts)]
    policy_stds = np.full(n_restarts, policy_std, dtype=float)
    restart_best = [None] * n_restarts
    mean_reward_matrix = np.empty((n_restarts, n_iterations))
    restart_best_matrix = np.empty_like(mean_reward_matrix)
    restart_mean_phase_matrix = np.empty_like(mean_reward_matrix)
    restart_worst_phase_matrix = np.empty_like(mean_reward_matrix)

    parallel_context = (
        mp.get_context("fork").Pool(processes=worker_count)
        if worker_count > 1
        else nullcontext()
    )
    with parallel_context as parallel:
        for iteration in range(n_iterations):
            populations = []
            for restart in range(n_restarts):
                half_epsilon = rngs[restart].standard_normal(
                    (population_size // 2, 2 * n_links)
                )
                epsilon = np.vstack([half_epsilon, -half_epsilon])
                actions = policy_means[restart] + policy_stds[restart] * epsilon
                kappa_ns, phi_p = decode_actions(
                    actions, aMAT, maximum_kappa_ns
                )
                populations.append((epsilon, kappa_ns, phi_p))

            if parallel is None:
                evaluations = [
                    _evaluate_population(
                        phys,
                        nd,
                        history,
                        training_coupling_ramp,
                        kappa_ns,
                        phi_p,
                        training_config,
                    )
                    for _, kappa_ns, phi_p in populations
                ]
            else:
                evaluations = parallel.starmap(
                    _evaluate_population,
                    (
                        (
                            phys,
                            nd,
                            history,
                            training_coupling_ramp,
                            kappa_ns,
                            phi_p,
                            training_config,
                        )
                        for _, kappa_ns, phi_p in populations
                    ),
                )

            for restart, evaluation in enumerate(evaluations):
                (
                    rewards,
                    phase_scores,
                    mean_phase_scores,
                    worst_phase_scores,
                    frequency_scores,
                    frequency_spreads,
                ) = evaluation
                epsilon, kappa_ns, phi_p = populations[restart]

                advantages = (rewards - rewards.mean()) / (
                    rewards.std() + 1e-8
                )
                policy_means[restart] += learning_rate * np.mean(
                    advantages[:, None]
                    * epsilon
                    / policy_stds[restart],
                    axis=0,
                )
                policy_means[restart][n_links:] = np.angle(
                    np.exp(1j * policy_means[restart][n_links:])
                )
                policy_stds[restart] = max(
                    minimum_policy_std,
                    policy_stds[restart] * policy_std_decay,
                )

                candidate = int(np.argmax(rewards))
                candidate_reward = float(rewards[candidate])
                if (
                    restart_best[restart] is None
                    or candidate_reward > restart_best[restart]["best_reward"]
                ):
                    restart_best[restart] = {
                        "restart": restart + 1,
                        "seed": int(restart_seeds[restart]),
                        "best_reward": candidate_reward,
                        "best_kappa_ns": kappa_ns[candidate].copy(),
                        "best_phi_p": phi_p[candidate].copy(),
                        "best_phase_score": float(phase_scores[candidate]),
                        "best_mean_phase_score": float(
                            mean_phase_scores[candidate]
                        ),
                        "best_worst_phase_score": float(
                            worst_phase_scores[candidate]
                        ),
                        "best_frequency_score": float(
                            frequency_scores[candidate]
                        ),
                        "best_frequency_spread": float(
                            frequency_spreads[candidate]
                        ),
                        "best_iteration": iteration + 1,
                    }

                mean_reward_matrix[restart, iteration] = rewards.mean()
                restart_best_matrix[restart, iteration] = restart_best[restart][
                    "best_reward"
                ]
                restart_mean_phase_matrix[restart, iteration] = restart_best[
                    restart
                ]["best_mean_phase_score"]
                restart_worst_phase_matrix[restart, iteration] = restart_best[
                    restart
                ]["best_worst_phase_score"]

            results = []
            for restart, best in enumerate(restart_best):
                result = dict(best)
                result["mean_reward_history"] = mean_reward_matrix[
                    restart, : iteration + 1
                ].copy()
                result["best_reward_history"] = restart_best_matrix[
                    restart, : iteration + 1
                ].copy()
                result["best_mean_phase_history"] = restart_mean_phase_matrix[
                    restart, : iteration + 1
                ].copy()
                result["best_worst_phase_history"] = restart_worst_phase_matrix[
                    restart, : iteration + 1
                ].copy()
                results.append(result)

            current_best = max(results, key=lambda item: item["best_reward"])
            print(
                f"iteration {iteration + 1:3d}/{n_iterations} | "
                f"global best {current_best['best_reward']:.3f} | "
                f"worst phase {current_best['best_worst_phase_score']:.3f} | "
                f"best restart {current_best['restart']}"
            )
            if show_inline_plot:
                _plot_parallel_progress(results)
    for result in results:
        print(
            f"restart {result['restart']}/{n_restarts} complete | "
            f"restart best {result['best_reward']:.3f} | "
            f"worst phase {result['best_worst_phase_score']:.3f}"
        )

    results.sort(key=lambda result: result["restart"])
    best_result = max(results, key=lambda result: result["best_reward"])
    best_reward = best_result["best_reward"]
    best_kappa_ns = best_result["best_kappa_ns"]
    best_phi_p = best_result["best_phi_p"]
    best_phase_score = best_result["best_phase_score"]
    best_mean_phase_score = best_result["best_mean_phase_score"]
    best_worst_phase_score = best_result["best_worst_phase_score"]
    best_frequency_score = best_result["best_frequency_score"]
    best_frequency_spread = best_result["best_frequency_spread"]
    best_restart = best_result["restart"]
    best_iteration = best_result["best_iteration"]

    mean_rewards = np.concatenate(
        [result["mean_reward_history"] for result in results]
    )
    restart_best_rewards = np.concatenate(
        [result["best_reward_history"] for result in results]
    )
    restart_indices = np.repeat(np.arange(1, n_restarts + 1), n_iterations)
    iterations_in_restart = np.tile(np.arange(1, n_iterations + 1), n_restarts)
    restart_start_steps = np.arange(n_restarts) * n_iterations

    # Reconstruct the global-best histories in deterministic restart order.
    best_rewards = []
    best_mean_phase_scores = []
    best_worst_phase_scores = []
    running_reward = -np.inf
    running_mean_phase = np.nan
    running_worst_phase = np.nan
    for result in results:
        for iteration in range(n_iterations):
            if result["best_reward_history"][iteration] > running_reward:
                running_reward = result["best_reward_history"][iteration]
                running_mean_phase = result["best_mean_phase_history"][iteration]
                running_worst_phase = result["best_worst_phase_history"][iteration]
            best_rewards.append(running_reward)
            best_mean_phase_scores.append(running_mean_phase)
            best_worst_phase_scores.append(running_worst_phase)

    np.savez(
        output_file,
        kappa_ns=best_kappa_ns,
        kappa_s=best_kappa_ns * 1e9,
        phi_p=best_phi_p,
        reward=best_reward,
        mean_reward_history=mean_rewards,
        restart_best_reward_history=restart_best_rewards,
        best_reward_history=best_rewards,
        best_mean_phase_history=best_mean_phase_scores,
        best_worst_phase_history=best_worst_phase_scores,
        restart_index_history=restart_indices,
        iteration_in_restart_history=iterations_in_restart,
        restart_start_steps=restart_start_steps,
        restart_seeds=restart_seeds,
        n_restarts=n_restarts,
        parallel_restarts=worker_count,
        iterations_per_restart=n_iterations,
        population_size=population_size,
        random_seed=random_seed,
        best_restart=best_restart,
        best_iteration=best_iteration,
        target_phases=target_phases,
        phase_weight=phase_weight,
        frequency_lock_weight=frequency_lock_weight,
        frequency_tolerance_ghz=frequency_tolerance_ghz,
        phase_score=best_phase_score,
        mean_phase_score=best_mean_phase_score,
        worst_phase_score=best_worst_phase_score,
        worst_laser_phase_weight=worst_laser_phase_weight,
        frequency_score=best_frequency_score,
        frequency_spread_ghz=best_frequency_spread,
        coupling_ramp_start_tau=coupling_ramp_start_tau,
        coupling_ramp_time_tau=coupling_ramp_time_tau,
        training_Tmax_tau=Tmax / tau,
        tail_fraction=tail_fraction,
    )
    print("\nBest kappa matrix (ns^-1):")
    print(best_kappa_ns)
    print("\nBest phi_p matrix (radians):")
    print(best_phi_p)
    print(
        f"\nBest design came from restart {best_restart}, "
        f"iteration {best_iteration}."
    )
    return best_kappa_ns, best_phi_p, best_reward


if __name__ == "__main__":
    run_training()


# %% Simulate and visualize the final learned design
# This validation rollout uses the same kappa ramp as the RL training rollouts.
def simulate_final_design():
    from scipy.constants import c, hbar

    validation_Tmax = 1000.0 * tau
    validation_wavelength = 910e-9

    with np.load(output_file) as learned:
        final_kappa_s = learned["kappa_s"]
        final_phi_p = learned["phi_p"]
        validation_ramp_start_tau = (
            float(learned["coupling_ramp_start_tau"])
            if "coupling_ramp_start_tau" in learned.files
            else coupling_ramp_start_tau
        )
        validation_ramp_time_tau = (
            float(learned["coupling_ramp_time_tau"])
            if "coupling_ramp_time_tau" in learned.files
            else coupling_ramp_time_tau
        )

    validation_steps = int(validation_Tmax / dt)
    validation_time = np.arange(validation_steps) * dt
    coupling_ramp = VCSEL.cosine_ramp(
        validation_time,
        t_start=validation_ramp_start_tau * tau,
        rise_10_90=validation_ramp_time_tau * tau,
        kappa_initial=0.0,
        kappa_final=1.0,
    )

    validation_phys = {
        "tau_p": tau_p,
        "tau_n": tau_n,
        "g0": g0,
        "N0": N0,
        "s": s,
        "beta": beta,
        "kappa_c_mat": final_kappa_s,
        "phi_p_mat": final_phi_p,
        "I": I,
        "q": q,
        "alpha": alpha,
        "delta": delta,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "noise_amplitude": 0.0,
        "dt": dt,
        "Tmax": validation_Tmax,
        "tau": tau,
        "N_lasers": N_lasers,
        "save_every": save_every,
    }

    validation_vcsel = VCSEL(validation_phys)
    validation_nd = validation_vcsel.scale_params()
    validation_nd["kappa_ramp"] = coupling_ramp
    validation_history, validation_freq_history, _, _ = (
        validation_vcsel.generate_history(
            validation_nd, shape="FR", n_cases=1
        )
    )
    validation_t, validation_y, validation_freqs = validation_vcsel.integrate(
        validation_history,
        nd=validation_nd,
        progress=True,
        max_iter=1,
        smooth_freqs=True,
    )

    photons = validation_y[0, 1::3]
    phases = validation_y[0, 2::3]
    frequencies_ghz = validation_freqs[0] / (
        2.0 * np.pi * tau_p * 1e9
    )

    # Fill the initial two-delay history with its known free-running frequency.
    saved_history = validation_freq_history[0, :, ::save_every]
    history_points = min(saved_history.shape[1], frequencies_ghz.shape[1])
    frequencies_ghz[:, :history_points] = saved_history[:, :history_points]

    time_ns = validation_t * 1e9
    omega0 = 2.0 * np.pi * c / validation_wavelength
    intensity_to_mW = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
    laser_power_mW = photons * intensity_to_mW
    total_field = np.sum(np.sqrt(photons) * np.exp(1j * phases), axis=0)
    total_power_mW = np.abs(total_field) ** 2 * intensity_to_mW

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), dpi=200, sharex=True)

    # 1) Lasing frequencies
    for laser in range(N_lasers):
        axes[0].plot(
            time_ns,
            frequencies_ghz[laser],
            linewidth=2,
            label=rf"$\dot{{\phi}}_{laser + 1}$",
        )
    axes[0].set_ylabel(r"$\dot{\phi}$ (GHz)", fontsize=22)
    axes[0].set_title("Validation with the learned coupling matrices", fontsize=24)
    axes[0].grid(True, alpha=0.2)
    if N_lasers <= 4:
        axes[0].legend(fontsize=14)

    # 2) One wrapped phase per laser, all relative to laser 1
    relative_phases = np.angle(np.exp(1j * (phases - phases[:1])))
    for laser in range(N_lasers):
        axes[1].plot(
            time_ns,
            relative_phases[laser],
            linewidth=1.2,
            label=rf"$\phi_{laser + 1}-\phi_1$",
            alpha=0.4
        )
    axes[1].set_ylabel(r"wrapped $\phi_i-\phi_1$ (rad)", fontsize=22)
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].set_yticks([-1.1*np.pi, 0.0, 1.1*np.pi])
    axes[1].set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    axes[1].grid(True, alpha=0.2)
    if N_lasers <= 4:
        axes[1].legend(fontsize=14)

    # 3) Individual and coherent total output powers
    for laser in range(N_lasers):
        axes[2].plot(
            time_ns,
            laser_power_mW[laser],
            linewidth=2,
            label=rf"$P_{laser + 1}$",
        )
    axes[2].plot(
        time_ns,
        total_power_mW,
        color="green",
        linewidth=2.5,
        label=r"$P_{\rm tot}$",
    )
    axes[2].set_xlabel("Time (ns)", fontsize=22)
    axes[2].set_ylabel("Output power (mW)", fontsize=22)
    axes[2].grid(True, alpha=0.2)
    if N_lasers <= 4:
        axes[2].legend(fontsize=14)

    # Match simple_example.py by showing the initial history and coupling power.
    for axis in axes:
        axis.axvspan(0.0, 2.0 * tau * 1e9, color="gray", alpha=0.2)
        axis.tick_params(axis="both", which="major", labelsize=18)

    active_links = final_kappa_s > 0.0
    mean_kappa_time = coupling_ramp * np.mean(final_kappa_s[active_links])
    coupling_power_uW = (
        hbar
        * omega0
        * mean_kappa_time
        * validation_nd["sbar"]
        / (g0 * tau_n)
        * 1e6
    )
    saved_indices = np.rint(validation_t / dt).astype(int)
    saved_indices = np.clip(saved_indices, 0, validation_steps - 1)
    coupling_axis = axes[2].twinx()
    coupling_axis.plot(
        time_ns,
        coupling_power_uW[saved_indices],
        "k--",
        alpha=0.5,
        linewidth=2,
    )
    coupling_axis.set_ylabel(r"Mean-link coupling power ($\mu$W)", fontsize=18)
    coupling_axis.tick_params(axis="y", labelsize=16)

    validation_figure = Path(output_file).with_name(
        "reinforcement_matrix_design_validation.png"
    )
    validation_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(validation_figure, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return validation_t, validation_y, validation_freqs


if __name__ == "__main__":
    simulate_final_design()
