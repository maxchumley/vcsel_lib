#%%
"""Measure policy-gradient alignment between array sizes without training.

The diagnostic loads one atomic checkpoint snapshot, evaluates independent
REINFORCE batches for M=2,...,9 at the full detuning curriculum span, and
compares their flattened gradients. It never calls ``optimizer.step`` and
never writes a model checkpoint.
"""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (
    decode_action,
    effective_maximum_kappa_per_link,
    encode_for_policy,
    load_checkpoint,
    maximum_log_std_at,
    minimum_log_std_at,
    normalized_magnitude_symmetry_penalty,
    sample_coupling_designs,
    sample_detuning_distributions,
    sample_target_phases,
    simulate_candidates_parallel,
    sort_by_detuning,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


MODEL_SUFFIX = (
    "_gnn_M2-9_span5_edge_std_degree_scaled_stratified_span_no_M_features"
)
CHECKPOINT_KIND = "current"
TRAINING_N_LASERS = tuple(range(2, 10))
DIAGNOSTIC_DETUNING_SPAN_GHZ = 5.0
TARGETS_PER_M = 2
CANDIDATES_PER_TARGET = 64
REPEATS = 3
N_JOBS = 1
RANDOM_SEED = 91_731


def checkpoint_path() -> Path:
    """Resolve the compact checkpoint name, with legacy GNN fallback."""
    compact_label = (
        f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
    )
    verbose_label = "-".join(str(value) for value in TRAINING_N_LASERS)
    filenames = [
        MODEL_DIR
        / (
            f"conditional_coupling_variable_m_{label}_lasers_"
            f"{CHECKPOINT_KIND}{MODEL_SUFFIX}.pt"
        )
        for label in (compact_label, verbose_label)
    ]
    for filename in filenames:
        if filename.exists():
            return filename
    raise FileNotFoundError(
        "GNN checkpoint not found. Checked:\n"
        + "\n".join(f"  {filename}" for filename in filenames)
    )


def reinforce_gradient_for_size(
    policy: torch.nn.Module,
    config,
    n_lasers: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Return the unmodified policy gradient and batch diagnostics for M."""
    size_config = replace(
        config,
        n_lasers=n_lasers,
        targets_per_batch=TARGETS_PER_M,
        candidates_per_target=CANDIDATES_PER_TARGET,
        n_jobs=N_JOBS,
        detuning_span_ghz=DIAGNOSTIC_DETUNING_SPAN_GHZ,
    )
    targets_rad = sample_target_phases(
        TARGETS_PER_M,
        rng,
        n_lasers,
        size_config.structured_target_fraction,
        size_config.structured_target_jitter_rad,
    )
    # Sampling at the curriculum end gives every M the same stratified
    # distribution of peak-to-peak spans from 0 to the requested 5 GHz.
    detunings_ghz = sample_detuning_distributions(
        TARGETS_PER_M,
        rng,
        size_config,
        iteration=size_config.detuning_curriculum_iterations,
    )
    targets_rad, detunings_ghz = sort_by_detuning(
        targets_rad, detunings_ghz
    )
    node_features = encode_for_policy(
        targets_rad,
        policy,
        size_config,
        detunings_ghz,
    )
    actions, log_probabilities = sample_coupling_designs(
        policy,
        node_features,
        CANDIDATES_PER_TARGET,
    )
    action_size = policy.action_size_for(n_lasers)
    flat_actions = actions.detach().cpu().numpy().reshape(-1, action_size)
    flat_kappa, flat_phi_p = decode_action(
        flat_actions,
        size_config.maximum_kappa_per_ns,
        n_lasers,
        size_config.force_symmetric_kappa,
        size_config.force_symmetric_phi_p,
        size_config.balanced_coupling,
        normalize_incoming_coupling_by_degree=(
            size_config.normalize_incoming_coupling_by_degree
        ),
    )
    matrix_shape = (
        TARGETS_PER_M,
        CANDIDATES_PER_TARGET,
        n_lasers,
        n_lasers,
    )
    simulation = simulate_candidates_parallel(
        targets_rad,
        flat_kappa.reshape(matrix_shape),
        flat_phi_p.reshape(matrix_shape),
        size_config,
        detuning_distributions_ghz=detunings_ghz,
    )
    phase_rewards = simulation["phase_reward"].reshape(-1)
    symmetry_penalty = normalized_magnitude_symmetry_penalty(
        flat_kappa,
        effective_maximum_kappa_per_link(
            size_config.maximum_kappa_per_ns,
            n_lasers,
            size_config.normalize_incoming_coupling_by_degree,
        ),
    )
    training_rewards = (
        phase_rewards
        - size_config.magnitude_symmetry_weight * symmetry_penalty
    ).reshape(TARGETS_PER_M, CANDIDATES_PER_TARGET)
    advantages = training_rewards - training_rewards.mean(
        axis=1, keepdims=True
    )
    advantages /= training_rewards.std(axis=1, keepdims=True) + 1.0e-8
    advantages_tensor = torch.as_tensor(
        advantages,
        dtype=log_probabilities.dtype,
        device=log_probabilities.device,
    )
    loss = -(log_probabilities * advantages_tensor).mean()

    parameters = [
        parameter for parameter in policy.parameters()
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        allow_unused=True,
    )
    flat_gradient = torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    ).cpu().numpy()
    return (
        flat_gradient,
        float(np.mean(phase_rewards)),
        float(loss.detach().cpu()),
    )


def cosine_matrix(gradients: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarity for one gradient per row."""
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    normalized = gradients / np.maximum(norms, 1.0e-15)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def main() -> dict[str, np.ndarray | Path | int]:
    checkpoint = checkpoint_path()
    policy, _, config, metadata = load_checkpoint(
        checkpoint,
        device="cpu",
        architecture="gnn",
    )
    policy.train()
    checkpoint_iteration = int(metadata["iteration"])
    schedule_iteration = max(checkpoint_iteration - 1, 0)
    policy.minimum_log_std = minimum_log_std_at(schedule_iteration, config)
    policy.maximum_log_std = maximum_log_std_at(schedule_iteration, config)

    all_cosines = []
    all_rewards = []
    all_losses = []
    all_norms = []
    for repeat in range(REPEATS):
        rng = np.random.default_rng(RANDOM_SEED + repeat)
        torch.manual_seed(RANDOM_SEED + repeat)
        gradients = []
        rewards = []
        losses = []
        print(f"Repeat {repeat + 1}/{REPEATS}")
        for n_lasers in TRAINING_N_LASERS:
            gradient, reward, loss = reinforce_gradient_for_size(
                policy,
                config,
                n_lasers,
                rng,
            )
            gradients.append(gradient)
            rewards.append(reward)
            losses.append(loss)
            print(
                f"  M={n_lasers}: sampled phase={reward:.3f}, "
                f"gradient norm={np.linalg.norm(gradient):.3e}"
            )
        gradient_array = np.stack(gradients)
        all_cosines.append(cosine_matrix(gradient_array))
        all_rewards.append(rewards)
        all_losses.append(losses)
        all_norms.append(np.linalg.norm(gradient_array, axis=1))

    cosine_values = np.stack(all_cosines)
    reward_values = np.asarray(all_rewards)
    loss_values = np.asarray(all_losses)
    norm_values = np.asarray(all_norms)
    mean_cosine = cosine_values.mean(axis=0)
    std_cosine = cosine_values.std(axis=0)

    print("\nMean gradient cosine similarity:")
    print(np.array2string(mean_cosine, precision=3, suppress_small=True))
    print("\nM=2 cosine similarity by partner (mean +/- std):")
    for index, n_lasers in enumerate(TRAINING_N_LASERS[1:], start=1):
        print(
            f"  M=2 vs M={n_lasers}: "
            f"{mean_cosine[0, index]:+.3f} +/- "
            f"{std_cosine[0, index]:.3f}"
        )

    output_directory = RESULTS_DIR / "analyses" / "gradient_diagnostics"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = output_directory / (
        "gnn_gradient_conflicts_current_span5"
    )
    np.savez(
        output_stem.with_suffix(".npz"),
        n_lasers=np.asarray(TRAINING_N_LASERS),
        cosine_similarity=cosine_values,
        mean_cosine_similarity=mean_cosine,
        std_cosine_similarity=std_cosine,
        sampled_phase_reward=reward_values,
        reinforce_loss=loss_values,
        gradient_norm=norm_values,
        checkpoint_iteration=checkpoint_iteration,
    )

    figure, axis = plt.subplots(figsize=(7.2, 6.1), constrained_layout=True)
    image = axis.imshow(mean_cosine, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    tick_labels = [f"M={value}" for value in TRAINING_N_LASERS]
    axis.set_xticks(range(len(tick_labels)), tick_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(tick_labels)), tick_labels)
    axis.set_title(
        "GNN policy-gradient cosine similarity\n"
        f"5 GHz span distribution, checkpoint iteration {checkpoint_iteration}"
    )
    for row in range(len(TRAINING_N_LASERS)):
        for column in range(len(TRAINING_N_LASERS)):
            value = mean_cosine[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="gradient cosine similarity")
    figure_path = output_stem.with_suffix(".png")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(f"\nSaved {figure_path}")

    return {
        "checkpoint": checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "cosine_similarity": cosine_values,
        "mean_cosine_similarity": mean_cosine,
        "std_cosine_similarity": std_cosine,
        "sampled_phase_reward": reward_values,
        "gradient_norm": norm_values,
        "figure": figure_path,
    }


if __name__ == "__main__":
    diagnostic_results = main()
