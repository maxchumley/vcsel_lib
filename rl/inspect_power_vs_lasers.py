#%% Run sweep and helper functions
"""Evaluate a variable-M BiGRU checkpoint on in-phase targets.

This is a small companion to ``inspect_saved_coupling_model_variable_m.py``.
It keeps one physical detuning span fixed, designs an in-phase solution for
each requested array size, and plots the resulting settled total output
power.  The policy is variable-size, so sizes not used during training can be
evaluated as zero-shot cases.
"""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import c, hbar

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (
    DesignerConfig,
    decode_action,
    encode_for_policy,
    load_checkpoint,
    normalized_magnitude_symmetry_penalty,
    simulate_with_vcsel,
    simulate_candidates_parallel,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Settings to edit
# ---------------------------------------------------------------------------

CHECKPOINT_KIND = "best"  # "best", "current", or "final"
MODEL_SUFFIX = "_bigru"
TRAINING_SIZE_LABEL = "2to10"  # label used in the checkpoint filename
CHECKPOINT_DIRECTORY = MODEL_DIR

LASER_COUNTS = tuple(range(2, 11))
NUMBER_OF_CANDIDATES = 32
SIMULATION_TIME_NS = 500.0
TAIL_FRACTION = 0.35
NUMBER_OF_DETUNING_DISTRIBUTIONS = 8
RANDOM_SEED = 1234
# Target-level multiprocessing is optional.  Regardless of this value, all
# candidates for a worker's target chunk are integrated in one vectorized call.
# Keep this at 1 in notebooks if process spawning is inconvenient.
NUMBER_OF_WORKERS = 1

# The simulator receives these values in GHz; only the neural-network input
# is normalized internally by the policy code.  With ``random`` selected,
# every M is evaluated on several independent distributions, giving the
# output-power plot meaningful across-distribution error bars.
DETUNING_DISTRIBUTION_MODE = "random"  # "random" or "evenly_spaced"
DETUNING_SPAN_GHZ = 1.0

# This script is intended to compare deterministic physical responses.
NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1

FIGURE_DIRECTORY = RESULTS_DIR
FIGURE_SUFFIX = "power_vs_lasers"


def checkpoint_path() -> Path:
    """Return the selected variable-M checkpoint path."""
    suffix = MODEL_SUFFIX
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return CHECKPOINT_DIRECTORY / (
        f"conditional_coupling_variable_m_{TRAINING_SIZE_LABEL}_lasers_"
        f"{CHECKPOINT_KIND}{suffix}.pt"
    )


def centered_detunings(
    n_lasers: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create sorted detunings relative to the 0-GHz mean frequency."""
    if n_lasers < 2:
        raise ValueError("n_lasers must be at least 2")
    if DETUNING_DISTRIBUTION_MODE == "random":
        if rng is None:
            raise ValueError("rng is required for random detunings")
        detunings = rng.uniform(
            -0.5 * DETUNING_SPAN_GHZ,
            0.5 * DETUNING_SPAN_GHZ,
            size=n_lasers,
        )
    elif DETUNING_DISTRIBUTION_MODE == "evenly_spaced":
        detunings = np.linspace(
            -0.5 * DETUNING_SPAN_GHZ,
            0.5 * DETUNING_SPAN_GHZ,
            n_lasers,
        )
    else:
        raise ValueError(
            "DETUNING_DISTRIBUTION_MODE must be 'random' or "
            "'evenly_spaced'"
        )
    return detunings - np.mean(detunings)


def total_power_from_simulation(
    states: np.ndarray,
    config: DesignerConfig,
    time_count: int,
) -> tuple[float, float]:
    """Return mean and temporal standard deviation of settled total power."""
    # VCSEL states are [carrier, photon number, phase] for every laser.
    photons = np.asarray(states)[:, 1::3]
    optical_angular_frequency = 2.0 * np.pi * c / 910.0e-9
    intensity_to_mw = (
        1.0e3
        * hbar
        * optical_angular_frequency
        / (
            config.gain_per_second
            * config.carrier_lifetime_seconds
            * config.photon_lifetime_seconds
        )
    )
    phases = np.asarray(states)[:, 2::3]
    total_field = np.sum(
        np.sqrt(np.maximum(photons, 0.0)) * np.exp(1j * phases),
        axis=1,
    )
    total_power_mw = np.abs(total_field) ** 2 * intensity_to_mw
    tail_start = max(
        0,
        int((1.0 - TAIL_FRACTION) * time_count),
    )
    settled = total_power_mw[:, tail_start:]
    finite_values = settled[np.isfinite(settled)]
    if finite_values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite_values)), float(np.std(finite_values))


def settled_power_per_case(
    states: np.ndarray,
    config: DesignerConfig,
    time_count: int,
) -> np.ndarray:
    """Return one settled total-power mean for every simulated case."""
    photons = np.asarray(states)[:, 1::3]
    optical_angular_frequency = 2.0 * np.pi * c / 910.0e-9
    intensity_to_mw = (
        1.0e3
        * hbar
        * optical_angular_frequency
        / (
            config.gain_per_second
            * config.carrier_lifetime_seconds
            * config.photon_lifetime_seconds
        )
    )
    phases = np.asarray(states)[:, 2::3]
    total_field = np.sum(
        np.sqrt(np.maximum(photons, 0.0)) * np.exp(1j * phases),
        axis=1,
    )
    total_power_mw = np.abs(total_field) ** 2 * intensity_to_mw
    tail_start = max(0, int((1.0 - TAIL_FRACTION) * time_count))
    settled = total_power_mw[:, tail_start:]
    with np.errstate(invalid="ignore"):
        samples = np.nanmean(
            np.where(np.isfinite(settled), settled, np.nan),
            axis=1,
        )
    return samples


def total_coupling_strength_per_case(
    kappa_per_ns: np.ndarray,
    symmetric: bool,
) -> np.ndarray:
    """Return the total physical coupling strength for each matrix.

    Directed matrices sum every off-diagonal entry.  For a symmetric matrix,
    the upper triangle counts each physical link once rather than twice.
    """
    kappa = np.asarray(kappa_per_ns, dtype=float)
    if kappa.ndim < 2 or kappa.shape[-1] != kappa.shape[-2]:
        raise ValueError("kappa_per_ns must end with a square matrix")
    n_lasers = kappa.shape[-1]
    if symmetric:
        mask = np.triu(
            np.ones((n_lasers, n_lasers), dtype=bool),
            k=1,
        )
    else:
        mask = ~np.eye(n_lasers, dtype=bool)
    return np.sum(np.where(mask, kappa, 0.0), axis=(-2, -1))


def design_detuning_batch(
    policy,
    target_phases: np.ndarray,
    detuning_distributions: np.ndarray,
    config: DesignerConfig,
    *,
    number_of_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design all detuning conditions and simulate candidates in one batch.

    The candidate tensor is ``(D, C, M, M)``.  The existing simulator helper
    then partitions only the target dimension across workers, while each
    worker integrates its complete ``targets * candidates`` block at once.
    """
    detuning_distributions = np.asarray(detuning_distributions, dtype=float)
    target_batch = np.repeat(
        np.asarray(target_phases, dtype=float)[None, :],
        len(detuning_distributions),
        axis=0,
    )
    encoded = encode_for_policy(
        target_batch,
        policy,
        config,
        detuning_distributions,
    )
    policy.eval()
    with torch.no_grad():
        distribution = policy.distribution(encoded)
        # PyTorch samples as (C, D, actions); transpose to (D, C, actions).
        sampled = distribution.sample((number_of_candidates,)).detach()
        actions = sampled.permute(1, 0, 2).contiguous()
        # Include one deterministic policy action for every detuning condition.
        actions[:, 0, :] = distribution.mean
    actions_numpy = actions.cpu().numpy()
    d_count, candidate_count, action_width = actions_numpy.shape
    kappa_flat, phi_flat = decode_action(
        actions_numpy.reshape(d_count * candidate_count, action_width),
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
    )
    kappa = kappa_flat.reshape(
        d_count, candidate_count, config.n_lasers, config.n_lasers
    )
    phi_p = phi_flat.reshape(
        d_count, candidate_count, config.n_lasers, config.n_lasers
    )
    simulation = simulate_candidates_parallel(
        target_batch,
        kappa,
        phi_p,
        config,
        detuning_distributions_ghz=detuning_distributions,
    )
    symmetry = normalized_magnitude_symmetry_penalty(
        kappa_flat,
        config.maximum_kappa_per_ns,
    ).reshape(d_count, candidate_count)
    ranking = (
        simulation["phase_reward"]
        - config.magnitude_symmetry_weight * symmetry
    )
    best_indices = np.argmax(ranking, axis=1)
    return (
        kappa[np.arange(d_count), best_indices],
        phi_p[np.arange(d_count), best_indices],
        simulation["phase_reward"][np.arange(d_count), best_indices],
    )


def evaluate_power_vs_lasers() -> dict:
    """Design and simulate one in-phase solution for every requested M."""
    if NUMBER_OF_DETUNING_DISTRIBUTIONS < 1:
        raise ValueError("NUMBER_OF_DETUNING_DISTRIBUTIONS must be positive")
    if NUMBER_OF_WORKERS < 1:
        raise ValueError("NUMBER_OF_WORKERS must be positive")
    checkpoint = checkpoint_path()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Check MODEL_SUFFIX and TRAINING_SIZE_LABEL."
        )

    policy, _, saved_config, metadata = load_checkpoint(
        checkpoint,
        device="cpu",
        architecture="bigru",
    )
    print(
        f"Loaded {CHECKPOINT_KIND} checkpoint {checkpoint.name} "
        f"(training iteration {metadata['iteration']})."
    )

    results: list[dict[str, float | int | dict]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for n_lasers in LASER_COUNTS:
        if n_lasers < 2:
            raise ValueError("LASER_COUNTS must contain values >= 2")
        run_config = replace(
            saved_config,
            n_lasers=n_lasers,
            n_jobs=min(max(1, NUMBER_OF_WORKERS), NUMBER_OF_DETUNING_DISTRIBUTIONS),
            jupyter_mode=False,
            simulation_time_seconds=SIMULATION_TIME_NS * 1.0e-9,
            noise_amplitude=NOISE_AMPLITUDE,
        )
        target_phases = np.zeros(n_lasers, dtype=float)
        detuning_batch = np.asarray(
            [centered_detunings(n_lasers, rng)
             for _ in range(NUMBER_OF_DETUNING_DISTRIBUTIONS)],
            dtype=float,
        )
        # Generate and evaluate every distribution/candidate in one vectorized
        # candidate simulation.  Multiprocessing, when enabled, happens above
        # this vectorized level by splitting the target/distribution axis.
        selected_kappa, selected_phi, phase_rewards_array = (
            design_detuning_batch(
                policy,
                target_phases,
                detuning_batch,
                run_config,
                number_of_candidates=NUMBER_OF_CANDIDATES,
            )
        )
        kappa_cases = np.repeat(
            selected_kappa,
            NOISE_ITERATIONS,
            axis=0,
        )
        phi_cases = np.repeat(
            selected_phi,
            NOISE_ITERATIONS,
            axis=0,
        )
        detuning_cases = np.repeat(
            detuning_batch,
            NOISE_ITERATIONS,
            axis=0,
        )
        _, states, _, _ = simulate_with_vcsel(
            kappa_cases,
            phi_cases,
            run_config,
            progress=False,
            detuning_distribution_ghz=detuning_cases,
        )
        per_case_power = settled_power_per_case(
            states,
            run_config,
            states.shape[-1],
        )
        power_samples = per_case_power.reshape(
            NUMBER_OF_DETUNING_DISTRIBUTIONS,
            NOISE_ITERATIONS,
        ).mean(axis=1)
        phase_rewards = np.asarray(phase_rewards_array, dtype=float)
        coupling_samples = total_coupling_strength_per_case(
            selected_kappa,
            run_config.force_symmetric_kappa,
        )
        valid_efficiency = (
            np.isfinite(power_samples)
            & np.isfinite(coupling_samples)
            & (coupling_samples > 0.0)
        )
        power_per_coupling_samples = (
            power_samples[valid_efficiency]
            / coupling_samples[valid_efficiency]
        )

        finite_power = np.asarray(power_samples, dtype=float)
        finite_power = finite_power[np.isfinite(finite_power)]
        if finite_power.size == 0:
            power_mean = float("nan")
            power_std = float("nan")
        else:
            power_mean = float(np.mean(finite_power))
            power_std = float(
                np.std(finite_power, ddof=1) if finite_power.size > 1 else 0.0
            )
        phase_reward = float(np.mean(phase_rewards))
        results.append(
            {
                "n_lasers": n_lasers,
                "phase_reward": phase_reward,
                "power_mean_mw": power_mean,
                "power_std_mw": power_std,
                "power_samples_mw": finite_power.copy(),
                "coupling_strength_mean_per_ns": float(
                    np.mean(coupling_samples)
                ),
                "coupling_strength_std_per_ns": float(
                    np.std(coupling_samples, ddof=1)
                    if coupling_samples.size > 1
                    else 0.0
                ),
                "coupling_strength_samples_per_ns": coupling_samples.copy(),
                "power_per_coupling_samples": (
                    power_per_coupling_samples.copy()
                ),
                "power_per_coupling_mean": float(
                    np.mean(power_per_coupling_samples)
                    if power_per_coupling_samples.size
                    else float("nan")
                ),
                "power_per_coupling_std": float(
                    np.std(power_per_coupling_samples, ddof=1)
                    if power_per_coupling_samples.size > 1
                    else 0.0
                ),
                "symmetric_kappa": bool(run_config.force_symmetric_kappa),
                "phase_reward_std": float(np.std(phase_rewards)),
                "design": {
                    "kappa_per_ns": selected_kappa[0].copy(),
                    "phi_p_rad": selected_phi[0].copy(),
                    "phase_reward": float(phase_rewards[0]),
                    "detuning_distribution_ghz": detuning_batch[0].copy(),
                    "total_coupling_strength_per_ns": float(
                        coupling_samples[0]
                    ),
                },
            }
        )
        print(
            f"M={n_lasers:2d} | phase reward={phase_reward:.4f} | "
            f"settled total power={power_mean:.3f} +/- {power_std:.3f} mW "
            f"({finite_power.size}/{NUMBER_OF_DETUNING_DISTRIBUTIONS} "
            "detuning distributions)"
        )

    plot_power_results(results)
    return {
        "policy": policy,
        "checkpoint": checkpoint,
        "metadata": metadata,
        "results": results,
    }


def plot_power_results(
    results: list[dict[str, float | int | dict]],
) -> plt.Figure:
    """Plot settled total output power as a function of laser count."""
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([result["n_lasers"] for result in results], dtype=int)
    means = np.asarray(
        [result["power_mean_mw"] for result in results], dtype=float
    )
    spreads = np.asarray(
        [result["power_std_mw"] for result in results], dtype=float
    )

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.errorbar(
        counts,
        means,
        yerr=spreads,
        marker="o",
        capsize=3,
        linewidth=1.8,
        label="settled total power",
    )
    axis.set(
        xlabel="number of lasers $M$",
        ylabel="total output power (mW)",
        title="In-phase design: output power versus array size",
        xticks=counts,
    )
    axis.grid(alpha=0.3)
    axis.legend()
    output_path = FIGURE_DIRECTORY / f"output_power_vs_lasers_{FIGURE_SUFFIX}.png"
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    print(f"Saved {output_path}")
    return figure


def plot_power_per_coupling(
    results: list[dict[str, float | int | dict]] | dict,
    maximum_lasers: int | None = None,
) -> plt.Figure:
    """Plot settled power per total coupling strength.

    New sweeps use the ratio for every detuning distribution, preserving
    distribution-to-distribution error bars.  Older in-memory sweep results
    fall back to their representative design, so this replot does not require
    another simulation.
    """
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    if not isinstance(results, list):
        raise TypeError("results must be the sweep list or its result wrapper")

    counts = []
    means = []
    spreads = []
    for result in results:
        count = float(result["n_lasers"])
        if maximum_lasers is not None and count > maximum_lasers:
            continue
        if "power_per_coupling_mean" in result:
            efficiency_mean = float(result["power_per_coupling_mean"])
            efficiency_std = float(result.get("power_per_coupling_std", 0.0))
        else:
            # Backward-compatible fallback for results created before this
            # metric was added.  It uses the representative stored design.
            design = result.get("design", {})
            kappa = np.asarray(design.get("kappa_per_ns"), dtype=float)
            if kappa.ndim != 2:
                continue
            symmetric = bool(result.get("symmetric_kappa", False))
            coupling = float(
                total_coupling_strength_per_case(kappa[None, ...], symmetric)[0]
            )
            if coupling <= 0.0:
                continue
            efficiency_mean = float(result["power_mean_mw"]) / coupling
            efficiency_std = float(result["power_std_mw"]) / coupling
        counts.append(count)
        means.append(efficiency_mean)
        spreads.append(efficiency_std)

    counts_array = np.asarray(counts, dtype=float)
    means_array = np.asarray(means, dtype=float)
    spreads_array = np.asarray(spreads, dtype=float)
    finite = (
        np.isfinite(counts_array)
        & np.isfinite(means_array)
        & np.isfinite(spreads_array)
        & (means_array > 0.0)
    )
    if not np.any(finite):
        raise ValueError("No finite power-per-coupling results are available")

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.errorbar(
        counts_array[finite],
        means_array[finite],
        yerr=spreads_array[finite],
        marker="o",
        capsize=3,
        linewidth=1.8,
        label=r"$P_{\mathrm{total}}/\sum\kappa_{ij}$",
    )
    axis.set(
        xlabel="number of lasers $M$",
        ylabel=r"power per total coupling (mW ns)",
        title="In-phase design: power per total coupling",
        xticks=counts_array[finite],
    )
    axis.grid(alpha=0.3)
    axis.legend()
    output_path = FIGURE_DIRECTORY / (
        f"output_power_vs_lasers_{FIGURE_SUFFIX}_per_coupling.png"
    )
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    print(f"Saved {output_path}")
    return figure


def plot_log_power_fit(
    results: list[dict[str, float | int | dict]] | dict,
    maximum_lasers: int | None = 10,
) -> plt.Figure:
    """Replot the sweep on log-log axes with a power-law fit.

    The error bars remain the standard deviation across the independent
    detuning distributions.  A straight line in log-log coordinates is a
    power-law fit in the original variables: ``power = a * M**b``.
    This function is intentionally separate so it can be rerun in a notebook
    after changing only the plotting choices.
    """
    # evaluate_power_vs_lasers returns a wrapper containing metadata and the
    # per-size records.  Accept that wrapper directly in the notebook cell.
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    if not isinstance(results, list):
        raise TypeError("results must be the sweep list or its result wrapper")

    counts = np.asarray(
        [result["n_lasers"] for result in results], dtype=float
    )
    means = np.asarray(
        [result["power_mean_mw"] for result in results], dtype=float
    )
    spreads = np.asarray(
        [result["power_std_mw"] for result in results], dtype=float
    )
    finite = np.isfinite(counts) & np.isfinite(means) & (means > 0.0)
    if maximum_lasers is not None:
        finite &= counts <= maximum_lasers
    if np.count_nonzero(finite) < 3:
        raise ValueError(
            "At least three finite, positive-power array sizes are required"
        )

    counts = counts[finite]
    means = means[finite]
    spreads = np.nan_to_num(spreads[finite], nan=0.0)
    log_coefficients = np.polyfit(
        np.log(counts),
        np.log(means),
        deg=1,
    )
    scaling_exponent = float(log_coefficients[0])
    prefactor = float(np.exp(log_coefficients[1]))
    fit_x = np.geomspace(np.min(counts), np.max(counts), 300)
    fit_y = prefactor * fit_x**scaling_exponent

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    # Log axes require positive error-bar endpoints.  The means themselves
    # were already filtered above, while this clipping only protects plotting
    # when a distribution standard deviation is larger than its mean.
    lower_error = np.minimum(spreads, 0.99 * means)
    upper_error = spreads
    axis.errorbar(
        counts,
        means,
        yerr=np.vstack((lower_error, upper_error)),
        marker="o",
        capsize=3,
        linewidth=1.8,
        label="settled total power",
    )
    axis.plot(
        fit_x,
        fit_y,
        "--",
        linewidth=2.0,
        label=rf"linear log-log fit ($P={prefactor:.3g}M^{{{scaling_exponent:.3g}}}$)",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="number of lasers $M$ (log scale)",
        ylabel="total output power (mW, log scale)",
        title="In-phase design: log-log power fit",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    output_path = FIGURE_DIRECTORY / (
        f"output_power_vs_lasers_{FIGURE_SUFFIX}_log_fit.png"
    )
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    print(
        "Power-law fit (power = a * M**b): "
        f"a={prefactor:.6g}, b={scaling_exponent:.6g}"
    )
    print(f"Saved {output_path}")
    return figure


def plot_normalized_power(
    results: list[dict[str, float | int | dict]] | dict,
    maximum_lasers: int | None = 10,
) -> plt.Figure:
    """Compare power normalized by M and M² using existing sweep results.

    No simulations are run here.  ``P_total/M`` is useful for comparing
    against an incoherent sum of independent laser powers, while
    ``P_total/M²`` tests ideal coherent-summed-field scaling.
    """
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    if not isinstance(results, list):
        raise TypeError("results must be the sweep list or its result wrapper")

    counts = np.asarray(
        [result["n_lasers"] for result in results], dtype=float
    )
    means = np.asarray(
        [result["power_mean_mw"] for result in results], dtype=float
    )
    spreads = np.asarray(
        [result["power_std_mw"] for result in results], dtype=float
    )
    finite = np.isfinite(counts) & np.isfinite(means) & (means > 0.0)
    if maximum_lasers is not None:
        finite &= counts <= maximum_lasers
    if not np.any(finite):
        raise ValueError("No finite power results are available")

    counts = counts[finite]
    means = means[finite]
    spreads = np.nan_to_num(spreads[finite], nan=0.0)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )
    axes[0].errorbar(
        counts,
        means / counts,
        yerr=spreads / counts,
        marker="o",
        capsize=3,
        linewidth=1.8,
    )
    axes[0].set_title(r"$P_{\mathrm{total}}/M$")
    axes[0].set_ylabel("normalized power (mW)")
    axes[1].errorbar(
        counts,
        means / counts**2,
        yerr=spreads / counts**2,
        marker="o",
        capsize=3,
        linewidth=1.8,
    )
    axes[1].set_title(r"$P_{\mathrm{total}}/M^2$")
    axes[1].set_ylabel("coherent-normalized power (mW)")
    for axis in axes:
        axis.set_xlabel("number of lasers $M$")
        axis.grid(alpha=0.3)

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIRECTORY / (
        f"output_power_vs_lasers_{FIGURE_SUFFIX}_normalized.png"
    )
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    print(f"Saved {output_path}")
    return figure

def main() -> dict:
    """Run the power sweep and save its four summary figures."""
    results = evaluate_power_vs_lasers()
    sweep_records = (
        results["results"]
        if isinstance(results, dict) and "results" in results
        else results
    )
    power_figure = plot_power_results(sweep_records)
    power_per_coupling_figure = plot_power_per_coupling(sweep_records)
    log_fit_figure = plot_log_power_fit(
        sweep_records,
        maximum_lasers=max(LASER_COUNTS),
    )
    normalized_power_figure = plot_normalized_power(
        sweep_records,
        maximum_lasers=max(LASER_COUNTS),
    )
    return {
        "results": sweep_records,
        "power_figure": power_figure,
        "power_per_coupling_figure": power_per_coupling_figure,
        "log_fit_figure": log_fit_figure,
        "normalized_power_figure": normalized_power_figure,
    }


if __name__ == "__main__":
    results = main()
