"""Separate M=2 coupling-magnitude and coupling-phase failure modes.

The current mixed-M policy and a successful M=2-only expert are evaluated on
the same held-out cases. Their deterministic magnitude and phase matrices are
then recombined into four hybrids and simulated with one CPU worker. Neither
checkpoint is modified.
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
    encode_for_policy,
    load_checkpoint,
    simulate_candidates_parallel,
)
from rl.inspect_m2_coupling_gain_sweep import (
    DEFAULT_CHECKPOINT as DEFAULT_CURRENT_CHECKPOINT,
    held_out_m2_cases,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


DEFAULT_EXPERT_CHECKPOINT = MODEL_DIR / (
    "conditional_coupling_variable_m_2to2_lasers_best_"
    "bigru_2_lasers_test.pt"
)
HYBRID_LABELS = (
    "current K + current phase",
    "expert K + current phase",
    "current K + expert phase",
    "expert K + expert phase",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-checkpoint",
        type=Path,
        default=DEFAULT_CURRENT_CHECKPOINT,
    )
    parser.add_argument(
        "--expert-checkpoint",
        type=Path,
        default=DEFAULT_EXPERT_CHECKPOINT,
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=8,
        help="Held-out targets simulated together on the single worker.",
    )
    return parser.parse_args()


def deterministic_matrices(
    policy,
    config,
    targets: np.ndarray,
    detunings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one deterministic M=2 design per target."""
    m2_config = replace(config, n_lasers=2)
    policy.eval()
    encoded = encode_for_policy(
        targets, policy, m2_config, detunings
    )
    with torch.no_grad():
        actions = policy.action_mean(encoded).cpu().numpy()
    return decode_action(
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


def simulate_hybrids(
    current_checkpoint: Path,
    expert_checkpoint: Path,
    target_chunk_size: int,
) -> dict[str, object]:
    if target_chunk_size < 1:
        raise ValueError("target_chunk_size must be positive")

    current_policy, _, current_config, current_metadata = load_checkpoint(
        current_checkpoint, device="cpu", architecture="auto"
    )
    expert_policy, _, expert_config, expert_metadata = load_checkpoint(
        expert_checkpoint, device="cpu", architecture="auto"
    )
    current_iteration = int(current_metadata["iteration"])
    targets, detunings, validation_span_ghz = held_out_m2_cases(
        current_config, max(current_iteration - 1, 0)
    )

    current_kappa, current_phi = deterministic_matrices(
        current_policy,
        current_config,
        targets,
        detunings,
    )
    expert_kappa, expert_phi = deterministic_matrices(
        expert_policy,
        expert_config,
        targets,
        detunings,
    )

    # Axis 1 is the vectorized candidate/hybrid dimension.
    kappa_hybrids = np.stack(
        (
            current_kappa,
            expert_kappa,
            current_kappa,
            expert_kappa,
        ),
        axis=1,
    )
    phi_hybrids = np.stack(
        (
            current_phi,
            current_phi,
            expert_phi,
            expert_phi,
        ),
        axis=1,
    )
    simulation_config = replace(
        current_config,
        n_lasers=2,
        n_jobs=1,
        noise_amplitude=0.0,
        jupyter_mode=False,
    )
    rewards = np.empty((len(targets), len(HYBRID_LABELS)), dtype=float)
    for start in range(0, len(targets), target_chunk_size):
        stop = min(start + target_chunk_size, len(targets))
        result = simulate_candidates_parallel(
            targets[start:stop],
            kappa_hybrids[start:stop],
            phi_hybrids[start:stop],
            simulation_config,
            detuning_distributions_ghz=detunings[start:stop],
        )
        rewards[start:stop] = result["phase_reward"]
        print(f"Simulated held-out targets {start + 1}--{stop}/{len(targets)}")

    return {
        "current_checkpoint": current_checkpoint,
        "expert_checkpoint": expert_checkpoint,
        "current_iteration": current_iteration,
        "expert_iteration": int(expert_metadata["iteration"]),
        "validation_span_ghz": validation_span_ghz,
        "labels": HYBRID_LABELS,
        "targets_rad": targets,
        "detunings_ghz": detunings,
        "detuning_spans_ghz": np.ptp(detunings, axis=1),
        "current_kappa_per_ns": current_kappa,
        "current_phi_p_rad": current_phi,
        "expert_kappa_per_ns": expert_kappa,
        "expert_phi_p_rad": expert_phi,
        "rewards": rewards,
    }


def print_summary(result: dict[str, object]) -> None:
    rewards = np.asarray(result["rewards"])
    print()
    print(f"Current checkpoint: {result['current_checkpoint']}")
    print(f"Current checkpoint iteration: {result['current_iteration']}")
    print(f"Expert checkpoint: {result['expert_checkpoint']}")
    print(f"Expert checkpoint iteration: {result['expert_iteration']}")
    print(
        "Current validation maximum span: "
        f"{float(result['validation_span_ghz']):.3f} GHz"
    )
    print()
    print("hybrid                            mean    std  median    p10  frac>=.98")
    for index, label in enumerate(result["labels"]):
        values = rewards[:, index]
        print(
            f"{label:32s} {np.mean(values):6.3f} {np.std(values):6.3f} "
            f"{np.median(values):6.3f} {np.quantile(values, 0.10):6.3f} "
            f"{np.mean(values >= 0.98):9.3f}"
        )


def save_results(result: dict[str, object]) -> tuple[Path, Path, Path]:
    labels = tuple(result["labels"])
    rewards = np.asarray(result["rewards"])
    spans = np.asarray(result["detuning_spans_ghz"])
    output_directory = RESULTS_DIR / "analyses" / "m2_diagnostics"
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "m2_current_vs_expert_hybrid_test"
    summary_path = stem.with_suffix(".csv")
    cases_path = stem.with_name(f"{stem.name}_cases.csv")
    figure_path = stem.with_suffix(".png")

    with summary_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "hybrid",
                "mean_phase_reward",
                "std_phase_reward",
                "median_phase_reward",
                "tenth_percentile_phase_reward",
                "fraction_at_or_above_0p98",
            )
        )
        for index, label in enumerate(labels):
            values = rewards[:, index]
            writer.writerow(
                (
                    label,
                    np.mean(values),
                    np.std(values),
                    np.median(values),
                    np.quantile(values, 0.10),
                    np.mean(values >= 0.98),
                )
            )

    with cases_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ("case", "detuning_span_ghz")
            + tuple(label.replace(" ", "_") for label in labels)
        )
        for case_index, span in enumerate(spans):
            writer.writerow(
                (case_index, span) + tuple(rewards[case_index])
            )

    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True
    )
    positions = np.arange(len(labels))
    axes[0].errorbar(
        positions,
        np.mean(rewards, axis=0),
        yerr=np.std(rewards, axis=0),
        marker="o",
        linestyle="none",
        capsize=4,
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        (
            "current/current",
            "expert K/current phase",
            "current K/expert phase",
            "expert/expert",
        ),
        rotation=18,
        ha="right",
    )
    axes[0].set(
        ylabel="M=2 phase reward",
        title="Mean and standard deviation",
        ylim=(0.0, 1.02),
    )
    axes[0].grid(alpha=0.25)

    order = np.argsort(spans)
    for index, label in enumerate(labels):
        axes[1].plot(
            spans[order],
            rewards[order, index],
            marker=".",
            linewidth=1.0,
            label=label,
        )
    axes[1].set(
        xlabel="realized detuning span (GHz)",
        ylabel="M=2 phase reward",
        title="Reward versus held-out difficulty",
        ylim=(0.0, 1.02),
    )
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        "M=2 current/expert hybrid diagnostic at "
        rf"$D_{{\max}}={float(result['validation_span_ghz']):.2f}$ GHz"
    )
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return summary_path, cases_path, figure_path


def main() -> dict[str, object]:
    args = parse_args()
    result = simulate_hybrids(
        args.current_checkpoint,
        args.expert_checkpoint,
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
