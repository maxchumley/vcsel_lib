"""Test whether a trained variable-M policy undercouples the M=2 case.

The policy is evaluated deterministically on the same held-out M=2 targets
and detuning templates used during training.  Its coupling phases are held
fixed while every off-diagonal coupling magnitude is multiplied by a small
set of diagnostic gains.  This script never changes or saves the model.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Support direct script execution and VS Code/Jupyter cells started in rl/.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (
    decode_action,
    detuning_curriculum_half_span_at,
    effective_maximum_kappa_per_link,
    encode_for_policy,
    load_checkpoint,
    sample_detuning_distributions,
    sample_target_phases,
    simulate_candidates_parallel,
    sort_by_detuning,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


MODEL_SUFFIX = (
    "_bigru_M2-9_span1to5_edge_std_degree_scaled_stratified_span_"
    "long_detuning_curriculum"
)
DEFAULT_CHECKPOINT = MODEL_DIR / (
    "conditional_coupling_variable_m_2to9_lasers_current"
    f"{MODEL_SUFFIX}.pt"
)
DEFAULT_GAINS = (1.0, 1.5, 2.0, 3.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Current variable-M checkpoint to inspect.",
    )
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=DEFAULT_GAINS,
        help="Multipliers applied to the decoded M=2 coupling magnitudes.",
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=8,
        help="Held-out targets simulated together on the single worker.",
    )
    return parser.parse_args()


def held_out_m2_cases(
    config,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recreate the fixed held-out M=2 cases at the current span."""
    m2_config = replace(config, n_lasers=2)
    targets = sample_target_phases(
        config.held_out_target_count,
        np.random.default_rng(config.random_seed + 10_000 + 2),
        2,
        config.structured_target_fraction,
        config.structured_target_jitter_rad,
    )
    full_span_detunings = sample_detuning_distributions(
        config.held_out_target_count,
        np.random.default_rng(config.random_seed + 20_000 + 2),
        m2_config,
        iteration=config.detuning_curriculum_iterations,
    )
    current_half_span = detuning_curriculum_half_span_at(
        iteration, config
    )
    detunings = full_span_detunings * (
        current_half_span / (0.5 * config.detuning_span_ghz)
    )
    targets, detunings = sort_by_detuning(targets, detunings)
    return targets, detunings, 2.0 * current_half_span


def simulate_gain_sweep(
    checkpoint: Path,
    gains: np.ndarray,
    target_chunk_size: int,
) -> dict[str, np.ndarray | float | int | Path]:
    if target_chunk_size < 1:
        raise ValueError("target_chunk_size must be positive")
    if gains.ndim != 1 or len(gains) < 1:
        raise ValueError("gains must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(gains)) or np.any(gains <= 0.0):
        raise ValueError("all gains must be finite and positive")

    policy, _, config, metadata = load_checkpoint(
        checkpoint, device="cpu", architecture="auto"
    )
    checkpoint_iteration = int(metadata["iteration"])
    schedule_iteration = max(checkpoint_iteration - 1, 0)
    targets, detunings, validation_span_ghz = held_out_m2_cases(
        config, schedule_iteration
    )
    m2_config = replace(
        config,
        n_lasers=2,
        n_jobs=1,
        noise_amplitude=0.0,
        jupyter_mode=False,
    )

    policy.eval()
    encoded = encode_for_policy(
        targets, policy, m2_config, detunings
    )
    with torch.no_grad():
        actions = policy.action_mean(encoded).cpu().numpy()
    base_kappa, base_phi = decode_action(
        actions,
        m2_config.maximum_kappa_per_ns,
        2,
        m2_config.force_symmetric_kappa,
        m2_config.force_symmetric_phi_p,
        m2_config.balanced_coupling,
        normalize_incoming_coupling_by_degree=(
            m2_config.normalize_incoming_coupling_by_degree
        ),
    )
    maximum_link_kappa = effective_maximum_kappa_per_link(
        m2_config.maximum_kappa_per_ns,
        2,
        m2_config.normalize_incoming_coupling_by_degree,
    )
    gain_count = len(gains)
    kappa_by_gain = np.minimum(
        base_kappa[:, None, :, :] * gains[None, :, None, None],
        maximum_link_kappa,
    )
    phi_by_gain = np.broadcast_to(
        base_phi[:, None, :, :], kappa_by_gain.shape
    ).copy()

    rewards = np.empty((len(targets), gain_count), dtype=float)
    for start in range(0, len(targets), target_chunk_size):
        stop = min(start + target_chunk_size, len(targets))
        result = simulate_candidates_parallel(
            targets[start:stop],
            kappa_by_gain[start:stop],
            phi_by_gain[start:stop],
            m2_config,
            detuning_distributions_ghz=detunings[start:stop],
        )
        rewards[start:stop] = result["phase_reward"]
        print(f"Simulated held-out targets {start + 1}--{stop}/{len(targets)}")

    return {
        "checkpoint": checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "checkpoint_held_out_reward": float(metadata["held_out_reward"]),
        "validation_span_ghz": validation_span_ghz,
        "gains": gains,
        "targets_rad": targets,
        "detunings_ghz": detunings,
        "detuning_spans_ghz": np.ptp(detunings, axis=1),
        "base_kappa_per_ns": base_kappa,
        "base_phi_p_rad": base_phi,
        "rewards": rewards,
    }


def save_results(
    result: dict[str, np.ndarray | float | int | Path],
) -> tuple[Path, Path, Path]:
    gains = np.asarray(result["gains"])
    rewards = np.asarray(result["rewards"])
    spans = np.asarray(result["detuning_spans_ghz"])
    output_directory = RESULTS_DIR / "analyses" / "m2_diagnostics"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = output_directory / "m2_current_model_coupling_gain_sweep"
    summary_path = output_stem.with_suffix(".csv")
    cases_path = output_stem.with_name(
        f"{output_stem.name}_cases.csv"
    )
    figure_path = output_stem.with_suffix(".png")

    with summary_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "gain",
                "mean_phase_reward",
                "std_phase_reward",
                "median_phase_reward",
                "tenth_percentile_phase_reward",
                "fraction_at_or_above_0p98",
            )
        )
        for index, gain in enumerate(gains):
            writer.writerow(
                (
                    gain,
                    np.mean(rewards[:, index]),
                    np.std(rewards[:, index]),
                    np.median(rewards[:, index]),
                    np.quantile(rewards[:, index], 0.10),
                    np.mean(rewards[:, index] >= 0.98),
                )
            )

    with cases_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ("case", "detuning_span_ghz")
            + tuple(f"reward_g_{gain:g}" for gain in gains)
        )
        for case_index, span in enumerate(spans):
            writer.writerow(
                (case_index, span) + tuple(rewards[case_index])
            )

    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    axes[0].errorbar(
        gains,
        np.mean(rewards, axis=0),
        yerr=np.std(rewards, axis=0),
        marker="o",
        capsize=3,
    )
    axes[0].set(
        xlabel=r"coupling gain $g$",
        ylabel="M=2 phase reward",
        title="Mean and standard deviation",
        ylim=(0.0, 1.02),
    )
    axes[0].grid(alpha=0.25)

    order = np.argsort(spans)
    for index, gain in enumerate(gains):
        axes[1].plot(
            spans[order],
            rewards[order, index],
            marker=".",
            linewidth=1.0,
            label=rf"$g={gain:g}$",
        )
    axes[1].set(
        xlabel="realized detuning span (GHz)",
        ylabel="M=2 phase reward",
        title="Reward versus held-out difficulty",
        ylim=(0.0, 1.02),
    )
    axes[1].legend(loc="lower left", ncol=2)
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        "M=2 coupling-gain diagnostic at "
        rf"$D_{{\max}}={float(result['validation_span_ghz']):.2f}$ GHz"
    )
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return summary_path, cases_path, figure_path


def print_summary(
    result: dict[str, np.ndarray | float | int | Path],
) -> None:
    gains = np.asarray(result["gains"])
    rewards = np.asarray(result["rewards"])
    print()
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Checkpoint iteration: {result['checkpoint_iteration']}")
    print(
        "Current validation maximum span: "
        f"{float(result['validation_span_ghz']):.3f} GHz"
    )
    print()
    print(" gain   mean    std   median    p10   fraction>=0.98")
    for index, gain in enumerate(gains):
        values = rewards[:, index]
        print(
            f"{gain:5.2f}  {np.mean(values):6.3f}  {np.std(values):6.3f}  "
            f"{np.median(values):6.3f}  {np.quantile(values, 0.10):6.3f}  "
            f"{np.mean(values >= 0.98):8.3f}"
        )


def main() -> dict[str, np.ndarray | float | int | Path]:
    args = parse_args()
    result = simulate_gain_sweep(
        args.checkpoint,
        np.asarray(args.gains, dtype=float),
        args.target_chunk_size,
    )
    print_summary(result)
    summary_path, cases_path, figure_path = save_results(result)
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-case results: {cases_path}")
    print(f"Saved figure: {figure_path}")
    return result


if __name__ == "__main__":
    main()
