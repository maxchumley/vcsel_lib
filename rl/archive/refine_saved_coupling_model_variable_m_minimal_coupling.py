#%%
"""Refine a saved variable-M policy to remove phase-safe coupling links.

The saved dense checkpoint is loaded read-only and upgraded with one learned
Bernoulli gate per decoded link.  Candidate zero keeps every link active and
provides a same-target phase reference.  Sparse candidates receive a small
bonus only when their phase reward remains close to that dense reference, so
phase performance remains the primary objective.

All executable work lives in :func:`main` so multiprocessing remains safe on
macOS and Windows.
"""

from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (  # noqa: E402
    DesignerConfig,
    GNNPolicyNetwork,
    _initialize_simulation_worker,
    add_sparse_gates_to_policy,
    combine_task_gradients_pcgrad,
    decode_action,
    encode_for_policy,
    effective_maximum_kappa_per_link,
    evaluate_policy,
    load_checkpoint,
    normalized_magnitude_symmetry_penalty,
    sample_coupling_designs,
    sample_detuning_distributions,
    sample_sparse_coupling_designs,
    sample_target_phases,
    save_checkpoint,
    set_random_seed,
    simulate_candidates_parallel,
    simulate_indexed_target_chunk,
    split_simulation_batch,
    sort_by_detuning,
)
from rl.paths import MODEL_DIR, RESULTS_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Source model and output names
# ---------------------------------------------------------------------------

SOURCE_CHECKPOINT_KIND = "best"  # "best", "current", or "final"
SOURCE_TRAINING_N_LASERS = tuple(range(3, 11))
SOURCE_MODEL_SUFFIX = (
    "_gnn_M3-10_span5_edge_std_degree_scaled_stratified_span_"
    "no_M_features_lr1e-4_pcgrad"
)

# The source checkpoint is never overwritten.
SPARSE_MODEL_SUFFIX = (
    f"{SOURCE_MODEL_SUFFIX}_phase_relative_gates_eps5e-3_"
    "lambda2e-3_lr1e-5_pcgrad"
)


def checkpoint_path(kind: str, suffix: str) -> Path:
    """Return the checkpoint name used by the sparse workflow."""
    if kind not in {"best", "current", "final"}:
        raise ValueError("checkpoint kind must be 'best', 'current', or 'final'")
    size_label = (
        f"{SOURCE_TRAINING_N_LASERS[0]}to"
        f"{SOURCE_TRAINING_N_LASERS[-1]}"
    )
    normalized_suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return MODEL_DIR / (
        f"conditional_coupling_variable_m_{size_label}_lasers_"
        f"{kind}{normalized_suffix}.pt"
    )


SOURCE_CHECKPOINT = checkpoint_path(
    SOURCE_CHECKPOINT_KIND, SOURCE_MODEL_SUFFIX
)
SPARSE_BEST_CHECKPOINT = checkpoint_path("best", SPARSE_MODEL_SUFFIX)
SPARSE_CURRENT_CHECKPOINT = checkpoint_path("current", SPARSE_MODEL_SUFFIX)
SPARSE_FINAL_CHECKPOINT = checkpoint_path("final", SPARSE_MODEL_SUFFIX)


# ---------------------------------------------------------------------------
# Refinement controls
# ---------------------------------------------------------------------------

REFINEMENT_ITERATIONS = 1500
REFINEMENT_LEARNING_RATE = 1.0e-5
REFINEMENT_TARGETS_PER_BATCH = 14
TRAIN_ALL_SIZES_EACH_ITERATION = True
GRADIENT_COMBINATION_MODE = "pcgrad"

# Suspend resource pressure whenever any array size falls below the floor.
ENFORCE_PHASE_REWARD_FLOOR = True

# Give the gate head a short phase-only adaptation period, then test hard
# missing-link candidates for the remainder of this topology-only run.
GATE_SAMPLING_START_ITERATION = 100
COUPLING_STAGE_START_ITERATION = REFINEMENT_ITERATIONS
COUPLING_PENALTY_FINAL = 0.0
COUPLING_PENALTY_RAMP_ITERATIONS = 400

# The source model is already trained at a total 5 GHz detuning span, so begin
# refinement at the full span rather than repeating its detuning curriculum.
REFINEMENT_DETUNING_SPAN_GHZ = 5.0
DETUNING_INITIAL_HALF_SPAN_GHZ = 2.5
DETUNING_WARMUP_ITERATIONS = 0
DETUNING_CURRICULUM_ITERATIONS = 0

# Phase reward remains in the objective at full strength.  A candidate earns
# the much smaller sparsity bonus only if it stays within the stated tolerance
# of candidate zero, which is the all-links-active reference for that target.
USE_PHASE_CONSTRAINED_TOPOLOGY_REWARD = True
PHASE_RETENTION_TOLERANCE = 0.005
TOPOLOGY_SPARSITY_REWARD_INITIAL = 0.001
TOPOLOGY_SPARSITY_REWARD_FINAL = 0.002
TOPOLOGY_SPARSITY_REWARD_RAMP_ITERATIONS = 500

# Deliberately test missing links instead of relying on initially near-one
# Bernoulli probabilities. Candidate zero remains the dense reference; other
# reserved candidates probe one removed edge or randomized removal rates.
INCLUDE_DENSE_GATE_REFERENCE = True
SINGLE_EDGE_PROBE_FRACTION = 0.25
EXPLORATORY_MASK_FRACTION = 0.50
EXPLORATION_REMOVAL_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
EXPLORATION_STRENGTH = 0.75

# When any held-out M falls below the floor, remove sparsity pressure entirely
# until phase performance recovers.
RECOVERY_TOPOLOGY_REWARD_FRACTION = 0.0

# This remains an absolute safety floor.  Checkpointing and recovery also
# require every M to remain within PHASE_RETENTION_TOLERANCE of its own dense
# source-model validation reward.
PHASE_REWARD_FLOOR = 0.98

# Use narrower exploration during refinement than during dense training.
REFINEMENT_MAXIMUM_LOG_STD = -1.0

RUN_REFINEMENT = True
LIVE_PLOT = True


def linear_ramp(
    iteration: int,
    *,
    start: int,
    duration: int,
    final_value: float,
) -> float:
    """Ramp a nonnegative coefficient from zero to its requested value."""
    if iteration < start:
        return 0.0
    if duration <= 0:
        return float(final_value)
    progress = np.clip((iteration - start + 1) / duration, 0.0, 1.0)
    return float(final_value * progress)


def detuning_span_at(iteration: int, config: DesignerConfig) -> float:
    """Return the total physical detuning span used for this update."""
    final_half_span = 0.5 * config.detuning_span_ghz
    if iteration < config.detuning_curriculum_warmup_iterations:
        current_half_span = config.detuning_curriculum_initial_half_span_ghz
    else:
        ramp_iterations = max(
            config.detuning_curriculum_iterations
            - config.detuning_curriculum_warmup_iterations,
            1,
        )
        progress = np.clip(
            (
                iteration
                - config.detuning_curriculum_warmup_iterations
            )
            / ramp_iterations,
            0.0,
            1.0,
        )
        current_half_span = (
            config.detuning_curriculum_initial_half_span_ghz
            + progress
            * (
                final_half_span
                - config.detuning_curriculum_initial_half_span_ghz
            )
        )
    return float(2.0 * current_half_span)


def phase_relative_topology_rewards(
    phase_reward_matrix: np.ndarray,
    active_fraction_matrix: np.ndarray,
    *,
    sparsity_weight: float,
    phase_reward_floor: float,
    phase_retention_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reward sparsity only when it preserves the dense candidate's phase.

    Candidate zero must be the all-links-active reference.  Phase reward is
    never down-weighted or replaced; an eligible sparse candidate simply gets
    a small additive bonus proportional to its fraction of disabled links.
    """
    phase_reward_matrix = np.asarray(phase_reward_matrix, dtype=float)
    active_fraction_matrix = np.asarray(
        active_fraction_matrix, dtype=float
    )
    if phase_reward_matrix.ndim != 2:
        raise ValueError("phase_reward_matrix must be two-dimensional")
    if active_fraction_matrix.shape != phase_reward_matrix.shape:
        raise ValueError(
            "active_fraction_matrix must match phase_reward_matrix"
        )
    if phase_retention_tolerance < 0.0:
        raise ValueError("phase retention tolerance must be nonnegative")
    if sparsity_weight < 0.0:
        raise ValueError("sparsity weight must be nonnegative")

    dense_phase_reference = phase_reward_matrix[:, [0]]
    required_phase = np.maximum(
        phase_reward_floor,
        dense_phase_reference - phase_retention_tolerance,
    )
    phase_eligible = phase_reward_matrix >= required_phase
    sparsity_bonus = (
        sparsity_weight
        * (1.0 - active_fraction_matrix)
        * phase_eligible
    )
    return (
        phase_reward_matrix + sparsity_bonus,
        phase_eligible,
        required_phase,
    )


def sparse_refinement_batch(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DesignerConfig,
    rng: np.random.Generator,
    iteration: int,
    *,
    dense_total_coupling_reference: float,
    coupling_weight: float,
    sparsity_weight: float,
    sample_gates: bool,
    simulation_pool: Any | None,
    use_deterministic_gates: bool = False,
    use_phase_constrained_topology_reward: bool = False,
    phase_reward_floor: float = PHASE_REWARD_FLOOR,
    phase_retention_tolerance: float = PHASE_RETENTION_TOLERANCE,
    include_dense_gate_reference: bool = INCLUDE_DENSE_GATE_REFERENCE,
    single_edge_probe_fraction: float = SINGLE_EDGE_PROBE_FRACTION,
    exploratory_mask_fraction: float = EXPLORATORY_MASK_FRACTION,
    exploration_removal_fractions: tuple[float, ...] = (
        EXPLORATION_REMOVAL_FRACTIONS
    ),
    exploration_strength: float = EXPLORATION_STRENGTH,
    apply_optimizer_step: bool = True,
) -> dict[str, Any]:
    """Run one coupling-budget REINFORCE update for one array size."""
    if dense_total_coupling_reference <= 0.0:
        raise ValueError("dense coupling reference must be positive")
    if (
        use_phase_constrained_topology_reward
        and not include_dense_gate_reference
    ):
        raise ValueError(
            "phase-relative topology reward requires a dense candidate"
        )
    policy.train()
    policy.maximum_log_std = config.maximum_log_std
    targets_rad = sample_target_phases(
        config.targets_per_batch,
        rng,
        config.n_lasers,
        config.structured_target_fraction,
        config.structured_target_jitter_rad,
    )
    detunings_ghz = sample_detuning_distributions(
        config.targets_per_batch,
        rng,
        config,
        iteration=iteration,
    )
    targets_rad, detunings_ghz = sort_by_detuning(
        targets_rad, detunings_ghz
    )
    node_features = encode_for_policy(
        targets_rad, policy, config, detunings_ghz
    )
    has_sparse_gates = bool(getattr(policy, "enable_sparse_gates", False))
    if not has_sparse_gates:
        actions, log_probabilities = sample_coupling_designs(
            policy,
            node_features,
            config.candidates_per_target,
        )
        gates = None
        expected_active_probability = torch.ones(
            (), device=node_features.device, dtype=node_features.dtype
        )
    elif use_deterministic_gates:
        actions, log_probabilities = sample_coupling_designs(
            policy,
            node_features,
            config.candidates_per_target,
        )
        gates = policy.deterministic_gates(node_features)[:, None, :].expand(
            -1, config.candidates_per_target, -1
        )
        expected_active_probability = (
            policy.gate_distribution(node_features).probs.mean()
        )
    else:
        actions, gates, log_probabilities = sample_sparse_coupling_designs(
            policy,
            node_features,
            config.candidates_per_target,
            sample_gates=sample_gates,
            include_dense_reference=include_dense_gate_reference,
            single_edge_probe_fraction=single_edge_probe_fraction,
            exploratory_mask_fraction=exploratory_mask_fraction,
            exploration_removal_fractions=(
                exploration_removal_fractions
            ),
            exploration_strength=exploration_strength,
        )
        expected_active_probability = (
            policy.gate_distribution(node_features).probs.mean()
        )

    action_size = policy.action_size_for(config.n_lasers)
    magnitude_link_count = policy.link_counts(config.n_lasers)[1]
    flat_actions = actions.detach().cpu().numpy().reshape(-1, action_size)
    flat_gates = None
    if gates is not None:
        flat_gates = gates.detach().cpu().numpy().reshape(
            -1, magnitude_link_count
        )
    flat_kappa, flat_phi_p = decode_action(
        flat_actions,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
        config.balanced_coupling,
        magnitude_gates=flat_gates,
        normalize_incoming_coupling_by_degree=(
            config.normalize_incoming_coupling_by_degree
        ),
    )
    matrix_shape = (
        config.targets_per_batch,
        config.candidates_per_target,
        config.n_lasers,
        config.n_lasers,
    )
    simulation = simulate_candidates_parallel(
        targets_rad,
        flat_kappa.reshape(matrix_shape),
        flat_phi_p.reshape(matrix_shape),
        config,
        detuning_distributions_ghz=detunings_ghz,
        simulation_pool=simulation_pool,
    )
    phase_reward = simulation["phase_reward"].reshape(-1)
    symmetry_penalty = normalized_magnitude_symmetry_penalty(
        flat_kappa, config.maximum_kappa_per_ns
    )
    coupling_budget = np.sum(flat_kappa, axis=(1, 2)) / (
        dense_total_coupling_reference
    )
    active_fraction = (
        np.ones(len(flat_actions), dtype=float)
        if flat_gates is None
        else np.mean(flat_gates, axis=1)
    )

    if use_phase_constrained_topology_reward:
        phase_reward_matrix = phase_reward.reshape(
            config.targets_per_batch, config.candidates_per_target
        )
        active_fraction_matrix = active_fraction.reshape(
            config.targets_per_batch, config.candidates_per_target
        )
        candidate_reward_matrix, phase_eligible, _ = (
            phase_relative_topology_rewards(
                phase_reward_matrix,
                active_fraction_matrix,
                sparsity_weight=sparsity_weight,
                phase_reward_floor=phase_reward_floor,
                phase_retention_tolerance=phase_retention_tolerance,
            )
        )
        reinforce_reward = (
            candidate_reward_matrix.reshape(-1)
            - config.magnitude_symmetry_weight * symmetry_penalty
        )
        phase_eligible_fraction = float(np.mean(phase_eligible))
    else:
        reinforce_reward = (
            phase_reward
            - config.magnitude_symmetry_weight * symmetry_penalty
            - coupling_weight * coupling_budget
        )
        phase_eligible_fraction = float("nan")
    reward_matrix = reinforce_reward.reshape(
        config.targets_per_batch, config.candidates_per_target
    )
    advantages = reward_matrix - reward_matrix.mean(axis=1, keepdims=True)
    advantages /= reward_matrix.std(axis=1, keepdims=True) + 1.0e-8
    advantages_tensor = torch.as_tensor(
        advantages,
        dtype=torch.float32,
        device=log_probabilities.device,
    )
    reinforce_loss = -(log_probabilities * advantages_tensor).mean()
    if use_phase_constrained_topology_reward:
        # The realized active fraction is already inside each candidate's
        # REINFORCE reward, so no separate mean-probability penalty is needed.
        loss = reinforce_loss
    else:
        loss = reinforce_loss + sparsity_weight * expected_active_probability

    gradient_norm = torch.tensor(float("nan"), device=node_features.device)
    if apply_optimizer_step:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), config.gradient_clip
        )
        optimizer.step()
    with torch.no_grad():
        if apply_optimizer_step:
            policy.log_std_kappa.clamp_(
                config.minimum_log_std, config.maximum_log_std
            )
            policy.log_std_phi.clamp_(
                config.minimum_log_std, config.maximum_log_std
            )
        mean_action_std = float(
            policy.distribution(node_features).stddev.mean().cpu()
        )
        mean_gate_probability = (
            float(policy.gate_distribution(node_features).probs.mean().cpu())
            if has_sparse_gates
            else 1.0
        )

    metrics: dict[str, Any] = {
        "loss": float(loss.detach().cpu()),
        "phase_reward": float(np.mean(phase_reward)),
        "training_reward": float(
            np.mean(reinforce_reward)
            if use_phase_constrained_topology_reward
            else np.mean(reinforce_reward)
            - sparsity_weight
            * float(expected_active_probability.detach().cpu())
        ),
        "coupling_budget": float(np.mean(coupling_budget)),
        "active_fraction": float(np.mean(active_fraction)),
        "mean_gate_probability": mean_gate_probability,
        "mean_action_std": mean_action_std,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "coupling_weight": float(coupling_weight),
        "sparsity_weight": float(sparsity_weight),
        "phase_eligible_fraction": phase_eligible_fraction,
    }
    if not apply_optimizer_step:
        metrics["_loss_tensor"] = loss
    return metrics


def multisize_pcgrad_refinement_batch(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DesignerConfig,
    rng: np.random.Generator,
    iteration: int,
    *,
    dense_coupling_reference_by_m: dict[int, float],
    coupling_weight: float,
    sparsity_weight: float,
    sample_gates: bool,
    simulation_pool: Any | None,
    use_deterministic_gates: bool = False,
    use_phase_constrained_topology_reward: bool = False,
    phase_reward_floor: float = PHASE_REWARD_FLOOR,
    phase_retention_tolerance: float = PHASE_RETENTION_TOLERANCE,
) -> dict[str, Any]:
    """Apply one all-M sparse-refinement update using shared-queue PCGrad."""
    if (
        use_phase_constrained_topology_reward
        and not INCLUDE_DENSE_GATE_REFERENCE
    ):
        raise ValueError(
            "phase-relative topology reward requires a dense candidate"
        )
    policy.train()
    policy.minimum_log_std = config.minimum_log_std
    policy.maximum_log_std = config.maximum_log_std
    training_sizes = tuple(config.training_n_lasers)
    size_count = len(training_sizes)
    schedule_offset = (iteration * config.targets_per_batch) % size_count
    target_sizes = [
        training_sizes[(schedule_offset + index) % size_count]
        for index in range(config.targets_per_batch)
    ]
    target_counts = {
        n_lasers: target_sizes.count(n_lasers)
        for n_lasers in training_sizes
        if n_lasers in target_sizes
    }

    # Match the regular all-M trainer: prepare every size first, then place
    # every target/candidate chunk into one shared queue. Sorting by descending
    # M prevents a large-array simulation from becoming the final straggler.
    prepared_batches: list[dict[str, Any]] = []
    indexed_work_items: list[tuple[int, int, int, tuple[Any, ...]]] = []
    for size_batch_index, (n_lasers, target_count) in enumerate(
        target_counts.items()
    ):
        size_config = replace(config, n_lasers=n_lasers)
        targets_rad = sample_target_phases(
            target_count,
            rng,
            n_lasers,
            config.structured_target_fraction,
            config.structured_target_jitter_rad,
        )
        detunings_ghz = sample_detuning_distributions(
            target_count,
            rng,
            size_config,
            iteration=iteration,
        )
        targets_rad, detunings_ghz = sort_by_detuning(
            targets_rad, detunings_ghz
        )
        node_features = encode_for_policy(
            targets_rad, policy, size_config, detunings_ghz
        )
        has_sparse_gates = bool(
            getattr(policy, "enable_sparse_gates", False)
        )
        if not has_sparse_gates:
            actions, log_probabilities = sample_coupling_designs(
                policy, node_features, config.candidates_per_target
            )
            gates = None
            expected_active_probability = torch.ones(
                (), device=node_features.device, dtype=node_features.dtype
            )
        elif use_deterministic_gates:
            actions, log_probabilities = sample_coupling_designs(
                policy, node_features, config.candidates_per_target
            )
            gates = policy.deterministic_gates(node_features)[
                :, None, :
            ].expand(-1, config.candidates_per_target, -1)
            expected_active_probability = (
                policy.gate_distribution(node_features).probs.mean()
            )
        else:
            actions, gates, log_probabilities = (
                sample_sparse_coupling_designs(
                    policy,
                    node_features,
                    config.candidates_per_target,
                    sample_gates=sample_gates,
                    include_dense_reference=INCLUDE_DENSE_GATE_REFERENCE,
                    single_edge_probe_fraction=(
                        SINGLE_EDGE_PROBE_FRACTION
                    ),
                    exploratory_mask_fraction=EXPLORATORY_MASK_FRACTION,
                    exploration_removal_fractions=(
                        EXPLORATION_REMOVAL_FRACTIONS
                    ),
                    exploration_strength=EXPLORATION_STRENGTH,
                )
            )
            expected_active_probability = (
                policy.gate_distribution(node_features).probs.mean()
            )
        flat_actions = actions.detach().cpu().numpy().reshape(
            -1, policy.action_size_for(n_lasers)
        )
        magnitude_link_count = policy.link_counts(n_lasers)[1]
        flat_gates = None
        if gates is not None:
            flat_gates = gates.detach().cpu().numpy().reshape(
                -1, magnitude_link_count
            )
        flat_kappa, flat_phi_p = decode_action(
            flat_actions,
            config.maximum_kappa_per_ns,
            n_lasers,
            config.force_symmetric_kappa,
            config.force_symmetric_phi_p,
            config.balanced_coupling,
            magnitude_gates=flat_gates,
            normalize_incoming_coupling_by_degree=(
                config.normalize_incoming_coupling_by_degree
            ),
        )
        grouped_shape = (
            target_count,
            config.candidates_per_target,
            n_lasers,
            n_lasers,
        )
        grouped_kappa = flat_kappa.reshape(grouped_shape)
        grouped_phi_p = flat_phi_p.reshape(grouped_shape)
        candidate_chunk_size = min(
            config.candidates_per_worker_task,
            config.candidates_per_target,
        )
        for candidate_start in range(
            0, config.candidates_per_target, candidate_chunk_size
        ):
            candidate_stop = min(
                candidate_start + candidate_chunk_size,
                config.candidates_per_target,
            )
            work_items = split_simulation_batch(
                targets_rad,
                grouped_kappa[:, candidate_start:candidate_stop],
                grouped_phi_p[:, candidate_start:candidate_stop],
                size_config,
                detuning_distributions_ghz=detunings_ghz,
                n_jobs=target_count,
            )
            indexed_work_items.extend(
                (
                    size_batch_index,
                    candidate_start,
                    candidate_stop,
                    work_item,
                )
                for work_item in work_items
            )
        prepared_batches.append(
            {
                "n_lasers": n_lasers,
                "target_count": target_count,
                "log_probabilities": log_probabilities,
                "flat_kappa": flat_kappa,
                "flat_gates": flat_gates,
                "expected_active_probability": (
                    expected_active_probability
                ),
                "chunk_results": [],
                "mean_action_std": float(
                    policy.distribution(node_features).stddev.mean()
                    .detach().cpu()
                ),
            }
        )

    indexed_work_items.sort(
        key=lambda item: prepared_batches[item[0]]["n_lasers"],
        reverse=True,
    )
    if config.n_jobs == 1:
        chunk_results = [
            simulate_indexed_target_chunk(item)
            for item in indexed_work_items
        ]
    elif simulation_pool is not None:
        chunk_results = list(
            simulation_pool.imap_unordered(
                simulate_indexed_target_chunk, indexed_work_items
            )
        )
    else:
        spawn_context = mp.get_context("spawn")
        temporary_pool = spawn_context.Pool(
            processes=min(config.n_jobs, len(indexed_work_items)),
            initializer=_initialize_simulation_worker,
        )
        try:
            chunk_results = list(
                temporary_pool.imap_unordered(
                    simulate_indexed_target_chunk, indexed_work_items
                )
            )
        finally:
            temporary_pool.close()
            temporary_pool.join()
    for result in chunk_results:
        prepared_batches[int(result["size_batch_index"])][
            "chunk_results"
        ].append(result)

    size_metrics: dict[int, dict[str, float]] = {}
    task_losses: list[torch.Tensor] = []
    for prepared in prepared_batches:
        target_count = int(prepared["target_count"])
        n_lasers = int(prepared["n_lasers"])
        phase_reward_matrix = np.empty(
            (target_count, config.candidates_per_target), dtype=float
        )
        candidate_coverage = np.zeros_like(
            phase_reward_matrix, dtype=int
        )
        for result in prepared["chunk_results"]:
            start = int(result["start_index"])
            stop = int(result["stop_index"])
            candidate_start = int(result["candidate_start_index"])
            candidate_stop = int(result["candidate_stop_index"])
            phase_reward_matrix[
                start:stop, candidate_start:candidate_stop
            ] = result["phase_reward"]
            candidate_coverage[
                start:stop, candidate_start:candidate_stop
            ] += 1
        if not np.all(candidate_coverage == 1):
            raise RuntimeError(
                "multi-size refinement did not return every candidate "
                f"exactly once for M={n_lasers}"
            )

        phase_reward = phase_reward_matrix.reshape(-1)
        symmetry_penalty = normalized_magnitude_symmetry_penalty(
            prepared["flat_kappa"],
            effective_maximum_kappa_per_link(
                config.maximum_kappa_per_ns,
                n_lasers,
                config.normalize_incoming_coupling_by_degree,
            ),
        )
        coupling_budget = np.sum(
            prepared["flat_kappa"], axis=(1, 2)
        ) / dense_coupling_reference_by_m[n_lasers]
        active_fraction = (
            np.ones(len(phase_reward), dtype=float)
            if prepared["flat_gates"] is None
            else np.mean(prepared["flat_gates"], axis=1)
        )
        if use_phase_constrained_topology_reward:
            active_fraction_matrix = active_fraction.reshape(
                target_count, config.candidates_per_target
            )
            candidate_reward_matrix, phase_eligible, _ = (
                phase_relative_topology_rewards(
                    phase_reward_matrix,
                    active_fraction_matrix,
                    sparsity_weight=sparsity_weight,
                    phase_reward_floor=phase_reward_floor,
                    phase_retention_tolerance=(
                        phase_retention_tolerance
                    ),
                )
            )
            training_reward = (
                candidate_reward_matrix.reshape(-1)
                - config.magnitude_symmetry_weight * symmetry_penalty
            )
            phase_eligible_fraction = float(np.mean(phase_eligible))
        else:
            training_reward = (
                phase_reward
                - config.magnitude_symmetry_weight * symmetry_penalty
                - coupling_weight * coupling_budget
            )
            phase_eligible_fraction = float("nan")
        reward_matrix = training_reward.reshape(
            target_count, config.candidates_per_target
        )
        advantages = reward_matrix - reward_matrix.mean(
            axis=1, keepdims=True
        )
        advantages /= reward_matrix.std(axis=1, keepdims=True) + 1.0e-8
        advantages_tensor = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=prepared["log_probabilities"].device,
        )
        reinforce_loss = -(
            prepared["log_probabilities"] * advantages_tensor
        ).mean()
        if use_phase_constrained_topology_reward:
            size_loss = reinforce_loss
        else:
            size_loss = (
                reinforce_loss
                + sparsity_weight
                * prepared["expected_active_probability"]
            )
        task_losses.append(size_loss)
        size_metrics[n_lasers] = {
            "phase_reward": float(np.mean(phase_reward)),
            "training_reward": float(np.mean(training_reward)),
            "coupling_budget": float(np.mean(coupling_budget)),
            "active_fraction": float(np.mean(active_fraction)),
            "mean_gate_probability": float(
                prepared["expected_active_probability"].detach().cpu()
            ),
            "mean_action_std": prepared["mean_action_std"],
            "loss": float(size_loss.detach().cpu()),
            "phase_eligible_fraction": phase_eligible_fraction,
        }

    parameters = [
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    flattened_task_gradients = []
    for task_loss in task_losses:
        task_gradient = torch.autograd.grad(
            task_loss,
            parameters,
            allow_unused=True,
        )
        flattened_task_gradients.append(
            torch.cat(
                [
                    torch.zeros_like(parameter).reshape(-1)
                    if gradient is None
                    else gradient.detach().reshape(-1)
                    for parameter, gradient in zip(parameters, task_gradient)
                ]
            )
        )
    combined_gradient, diagnostics = combine_task_gradients_pcgrad(
        torch.stack(flattened_task_gradients),
        config.per_size_gradient_clip,
        rng,
    )
    offset = 0
    for parameter in parameters:
        parameter_size = parameter.numel()
        parameter.grad = combined_gradient[
            offset : offset + parameter_size
        ].reshape_as(parameter).clone()
        offset += parameter_size
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, config.gradient_clip
    )
    optimizer.step()
    with torch.no_grad():
        policy.log_std_kappa.clamp_(
            config.minimum_log_std, config.maximum_log_std
        )
        policy.log_std_phi.clamp_(
            config.minimum_log_std, config.maximum_log_std
        )

    def mean_size_metric(name: str) -> float:
        return float(
            np.mean([metrics[name] for metrics in size_metrics.values()])
        )

    return {
        "loss": float(torch.stack(task_losses).mean().detach().cpu()),
        "phase_reward": mean_size_metric("phase_reward"),
        "training_reward": mean_size_metric("training_reward"),
        "coupling_budget": mean_size_metric("coupling_budget"),
        "active_fraction": mean_size_metric("active_fraction"),
        "mean_gate_probability": mean_size_metric(
            "mean_gate_probability"
        ),
        "mean_action_std": mean_size_metric("mean_action_std"),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "coupling_weight": float(coupling_weight),
        "sparsity_weight": float(sparsity_weight),
        "phase_eligible_fraction": mean_size_metric(
            "phase_eligible_fraction"
        ),
        "size_metrics": size_metrics,
        "negative_gradient_pair_fraction": float(
            diagnostics["negative_gradient_pair_fraction"]
        ),
        "mean_gradient_cosine": float(
            diagnostics["mean_gradient_cosine"]
        ),
    }


def make_held_out_sets(
    config: DesignerConfig,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Recreate the deterministic validation conditions used in training."""
    held_out_sets = {}
    for n_lasers in config.training_n_lasers:
        size_config = replace(config, n_lasers=n_lasers)
        targets = sample_target_phases(
            config.held_out_target_count,
            np.random.default_rng(config.random_seed + 10_000 + n_lasers),
            n_lasers,
            config.structured_target_fraction,
            config.structured_target_jitter_rad,
        )
        detunings = sample_detuning_distributions(
            config.held_out_target_count,
            np.random.default_rng(config.random_seed + 20_000 + n_lasers),
            size_config,
            iteration=config.detuning_curriculum_iterations,
        )
        held_out_sets[n_lasers] = sort_by_detuning(targets, detunings)
    return held_out_sets


def validate_sparse_policy(
    policy: torch.nn.Module,
    config: DesignerConfig,
    held_out_sets: dict[int, tuple[np.ndarray, np.ndarray]],
    simulation_pool: Any | None,
    dense_coupling_reference_by_m: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Measure phase retention, dense-relative coupling, and active edges."""
    phase_by_m = {}
    coupling_by_m = {}
    total_coupling_by_m = {}
    active_by_m = {}
    gate_probability_by_m = {}
    for n_lasers, (targets, detunings) in held_out_sets.items():
        validation = evaluate_policy(
            policy,
            targets,
            replace(config, n_lasers=n_lasers),
            detuning_distributions_ghz=detunings,
            simulation_pool=simulation_pool,
        )
        phase_by_m[n_lasers] = float(np.mean(validation["phase_reward"]))
        total_coupling_by_m[n_lasers] = float(
            np.mean(np.sum(validation["kappa_per_ns"], axis=(1, 2)))
        )
        reference = (
            total_coupling_by_m[n_lasers]
            if dense_coupling_reference_by_m is None
            else dense_coupling_reference_by_m[n_lasers]
        )
        if reference <= 0.0:
            raise ValueError(
                f"dense coupling reference for M={n_lasers} must be positive"
            )
        coupling_by_m[n_lasers] = (
            total_coupling_by_m[n_lasers] / reference
        )
        active_by_m[n_lasers] = float(
            np.mean(validation["active_connection_fraction"])
        )
        gate_probability_by_m[n_lasers] = float(
            np.mean(validation["gate_probabilities"])
        )
    return {
        "phase_by_m": phase_by_m,
        "coupling_by_m": coupling_by_m,
        "total_coupling_by_m": total_coupling_by_m,
        "active_by_m": active_by_m,
        "gate_probability_by_m": gate_probability_by_m,
        "phase_reward": float(np.mean(list(phase_by_m.values()))),
        "minimum_phase_reward": float(min(phase_by_m.values())),
        "coupling_budget": float(np.mean(list(coupling_by_m.values()))),
        "active_fraction": float(np.mean(list(active_by_m.values()))),
        "mean_gate_probability": float(
            np.mean(list(gate_probability_by_m.values()))
        ),
    }


def update_refinement_figure(
    figure: plt.Figure,
    axes: np.ndarray,
    history: dict[str, Any],
) -> None:
    """Update phase preservation and the two resource objectives."""
    for axis in axes:
        axis.clear()
    axes[0].plot(history["phase_reward"], label="sampled phase reward")
    axes[0].plot(history["training_reward"], label="optimization reward")
    axes[0].axhline(
        PHASE_REWARD_FLOOR,
        color="black",
        linestyle="--",
        label=(
            "phase floor (phase only)"
            if ENFORCE_PHASE_REWARD_FLOOR
            else "phase floor (checkpoint only)"
        ),
    )
    reward_ceiling = max(
        1.0,
        float(np.max(history["phase_reward"])),
        float(np.max(history["training_reward"])),
    )
    axes[0].set(
        xlabel="iteration",
        ylabel="reward",
        ylim=(0.0, 1.05 * reward_ceiling),
    )
    axes[0].legend(loc="lower left")

    for n_lasers, values in history["held_out_reward_by_m"].items():
        line = axes[1].plot(values, label=f"M={n_lasers}")[0]
        if "phase_floor_by_m" in history:
            axes[1].axhline(
                history["phase_floor_by_m"][n_lasers],
                color=line.get_color(),
                linestyle="--",
                alpha=0.35,
            )
    axes[1].set(
        xlabel="iteration",
        ylabel=(
            "held-out phase reward "
            f"({REFINEMENT_DETUNING_SPAN_GHZ:g} GHz span)"
        ),
        ylim=(0.0, 1.0),
    )
    axes[1].legend(loc="lower left", ncol=2)

    axes[2].plot(
        history["held_out_coupling_budget"],
        label="coupling / source coupling",
    )
    axes[2].plot(
        history["held_out_active_fraction"], label="active fraction"
    )
    axes[2].plot(
        history["held_out_gate_probability"],
        label="mean gate probability",
    )
    axes[2].set(xlabel="iteration", ylabel="resource fraction")
    axes[2].set_ylim(bottom=0.0)
    axes[2].legend(loc="lower left")
    for axis in axes:
        axis.grid(alpha=0.25)


def plot_refinement_history(history: dict[str, Any]) -> plt.Figure:
    """Plot the final refinement history outside the live training loop."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    update_refinement_figure(figure, axes, history)
    return figure


def refine_sparse_policy(
    policy: torch.nn.Module,
    config: DesignerConfig,
    *,
    live_plot: bool = True,
    freeze_shared_topology_features: bool = True,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    """Fine-tune all trained array sizes using coupling-aware PCGrad."""
    has_sparse_gates = bool(getattr(policy, "enable_sparse_gates", False))
    rng = set_random_seed(config.random_seed + 50_000)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=REFINEMENT_LEARNING_RATE
    )
    held_out_sets = make_held_out_sets(config)
    training_sizes = tuple(config.training_n_lasers)
    history: dict[str, Any] = {
        "n_lasers": [],
        "phase_reward": [],
        "training_reward": [],
        "coupling_budget": [],
        "active_fraction": [],
        "mean_gate_probability": [],
        "phase_eligible_fraction": [],
        "coupling_weight": [],
        "sparsity_weight": [],
        "held_out_reward": [],
        "held_out_coupling_budget": [],
        "held_out_active_fraction": [],
        "held_out_gate_probability": [],
        "held_out_reward_by_m": {size: [] for size in training_sizes},
        "phase_recovery_mode": [],
        "negative_gradient_pair_fraction": [],
        "mean_gradient_cosine": [],
    }
    figure_and_axes = None
    if live_plot:
        plt.ion()
        figure_and_axes = plt.subplots(
            1, 3, figsize=(16, 4), constrained_layout=True
        )
        if config.jupyter_mode:
            plt.close(figure_and_axes[0])

    active_jobs = min(config.n_jobs, config.targets_per_batch)
    simulation_pool = None
    if active_jobs > 1:
        simulation_pool = mp.get_context("spawn").Pool(
            processes=active_jobs,
            initializer=_initialize_simulation_worker,
        )
    print(
        f"Coupling-budget refinement: {active_jobs} simulation workers; "
        f"M={training_sizes}; lr={REFINEMENT_LEARNING_RATE:.1e}"
    )

    latest_validation = validate_sparse_policy(
        policy, config, held_out_sets, simulation_pool
    )
    source_phase_reference_by_m = dict(latest_validation["phase_by_m"])
    phase_floor_by_m = {
        size: max(
            PHASE_REWARD_FLOOR,
            source_phase_reference_by_m[size]
            - PHASE_RETENTION_TOLERANCE,
        )
        for size in training_sizes
    }
    history["source_phase_reference_by_m"] = dict(
        source_phase_reference_by_m
    )
    history["phase_floor_by_m"] = dict(phase_floor_by_m)
    dense_coupling_reference_by_m = dict(
        latest_validation["total_coupling_by_m"]
    )
    history["dense_coupling_reference_by_m"] = dict(
        dense_coupling_reference_by_m
    )
    dense_reference_text = " | ".join(
        f"M={size}: {dense_coupling_reference_by_m[size]:.3f} ns^-1"
        for size in training_sizes
    )
    print(f"Source coupling references | {dense_reference_text}")
    phase_reference_text = " | ".join(
        f"M={size}: source={source_phase_reference_by_m[size]:.4f}, "
        f"floor={phase_floor_by_m[size]:.4f}"
        for size in training_sizes
    )
    print(f"Phase-retention constraints | {phase_reference_text}")
    feasible_best_cost = (np.inf, np.inf)
    best_infeasible_phase = -np.inf

    def maybe_save_best(iteration: int) -> None:
        nonlocal feasible_best_cost, best_infeasible_phase
        feasible = all(
            latest_validation["phase_by_m"][size]
            >= phase_floor_by_m[size]
            for size in training_sizes
        )
        # Among phase-feasible checkpoints, prefer fewer active links and use
        # dense-relative coupling only as a secondary resource tiebreaker.
        resource_cost = (
            latest_validation["active_fraction"],
            latest_validation["coupling_budget"],
        )
        should_save = False
        if feasible and resource_cost < feasible_best_cost:
            feasible_best_cost = resource_cost
            should_save = True
        elif (
            not np.isfinite(feasible_best_cost[0])
            and latest_validation["phase_reward"] > best_infeasible_phase
        ):
            best_infeasible_phase = latest_validation["phase_reward"]
            should_save = True
        if should_save:
            save_checkpoint(
                config.best_checkpoint_file,
                policy,
                optimizer,
                config,
                iteration=iteration,
                held_out_reward=latest_validation["phase_reward"],
            )

    maybe_save_best(0)
    topology_frozen = False
    coupling_ramp_start_iteration: int | None = None
    try:
        for iteration in range(REFINEMENT_ITERATIONS):
            topology_stage = iteration < COUPLING_STAGE_START_ITERATION
            if not topology_stage and not topology_frozen:
                # An existing input-dependent sparse topology requires its
                # gate path (and optionally its shared features) to remain
                # fixed. Dense policies have no gate modules to freeze.
                topology_modules = (
                    (policy.edge_gate_head,) if has_sparse_gates else ()
                )
                if has_sparse_gates and freeze_shared_topology_features:
                    encoder_modules: tuple[torch.nn.Module, ...]
                    if isinstance(policy, GNNPolicyNetwork):
                        encoder_modules = (
                            policy.node_encoder,
                            *tuple(policy.message_layers),
                        )
                    else:
                        encoder_modules = (
                            policy.node_encoder,
                            policy.bigru,
                        )
                    topology_modules = (
                        *encoder_modules,
                        *tuple(policy.edge_decoder.children())[:-1],
                        policy.edge_gate_head,
                    )
                for module in topology_modules:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
                topology_frozen = True

            if iteration < GATE_SAMPLING_START_ITERATION:
                scheduled_sparsity_weight = 0.0
            else:
                scheduled_sparsity_weight = (
                    TOPOLOGY_SPARSITY_REWARD_INITIAL
                    + linear_ramp(
                        iteration,
                        start=GATE_SAMPLING_START_ITERATION,
                        duration=TOPOLOGY_SPARSITY_REWARD_RAMP_ITERATIONS,
                        final_value=(
                            TOPOLOGY_SPARSITY_REWARD_FINAL
                            - TOPOLOGY_SPARSITY_REWARD_INITIAL
                        ),
                    )
                )
            phase_recovery_mode = (
                ENFORCE_PHASE_REWARD_FLOOR
                and any(
                    latest_validation["phase_by_m"][size]
                    < phase_floor_by_m[size]
                    for size in training_sizes
                )
            )
            if (
                not topology_stage
                and
                not phase_recovery_mode
                and coupling_ramp_start_iteration is None
            ):
                # Start the resource clock only after every trained M works
                # at the final detuning span. This avoids an abrupt full
                # penalty if wide-range recovery takes many iterations.
                coupling_ramp_start_iteration = iteration
            scheduled_coupling_weight = (
                0.0
                if coupling_ramp_start_iteration is None
                else linear_ramp(
                    iteration,
                    start=coupling_ramp_start_iteration,
                    duration=COUPLING_PENALTY_RAMP_ITERATIONS,
                    final_value=COUPLING_PENALTY_FINAL,
                )
            )
            if topology_stage:
                coupling_weight = 0.0
                sparsity_weight = scheduled_sparsity_weight
            else:
                coupling_weight = scheduled_coupling_weight
                sparsity_weight = 0.0

            if phase_recovery_mode:
                if topology_stage:
                    sparsity_weight *= RECOVERY_TOPOLOGY_REWARD_FRACTION
                else:
                    coupling_weight = 0.0
            sample_gates = (
                GATE_SAMPLING_START_ITERATION
                <= iteration
                < COUPLING_STAGE_START_ITERATION
            )
            if config.train_all_sizes_each_iteration:
                metrics = multisize_pcgrad_refinement_batch(
                    policy,
                    optimizer,
                    config,
                    rng,
                    iteration,
                    dense_coupling_reference_by_m=(
                        dense_coupling_reference_by_m
                    ),
                    coupling_weight=coupling_weight,
                    sparsity_weight=sparsity_weight,
                    sample_gates=sample_gates,
                    simulation_pool=simulation_pool,
                    use_deterministic_gates=(not topology_stage),
                    use_phase_constrained_topology_reward=topology_stage,
                )
                selected_size: int | tuple[int, ...] = training_sizes
            else:
                selected_size = int(rng.choice(training_sizes))
                metrics = sparse_refinement_batch(
                    policy,
                    optimizer,
                    replace(config, n_lasers=selected_size),
                    rng,
                    iteration,
                    dense_total_coupling_reference=(
                        dense_coupling_reference_by_m[selected_size]
                    ),
                    coupling_weight=coupling_weight,
                    sparsity_weight=sparsity_weight,
                    sample_gates=sample_gates,
                    simulation_pool=simulation_pool,
                    use_deterministic_gates=(not topology_stage),
                    use_phase_constrained_topology_reward=topology_stage,
                )
                metrics["negative_gradient_pair_fraction"] = float("nan")
                metrics["mean_gradient_cosine"] = float("nan")
            history["n_lasers"].append(selected_size)
            history["phase_recovery_mode"].append(phase_recovery_mode)
            for name in (
                "phase_reward",
                "training_reward",
                "coupling_budget",
                "active_fraction",
                "mean_gate_probability",
                "phase_eligible_fraction",
                "coupling_weight",
                "sparsity_weight",
            ):
                history[name].append(metrics[name])
            history["negative_gradient_pair_fraction"].append(
                metrics["negative_gradient_pair_fraction"]
            )
            history["mean_gradient_cosine"].append(
                metrics["mean_gradient_cosine"]
            )

            should_validate = (
                iteration == 0
                or (iteration + 1) % config.validation_interval == 0
                or iteration + 1 == REFINEMENT_ITERATIONS
            )
            if should_validate:
                latest_validation = validate_sparse_policy(
                    policy,
                    config,
                    held_out_sets,
                    simulation_pool,
                    dense_coupling_reference_by_m,
                )
                maybe_save_best(iteration + 1)

            history["held_out_reward"].append(
                latest_validation["phase_reward"]
            )
            history["held_out_coupling_budget"].append(
                latest_validation["coupling_budget"]
            )
            history["held_out_active_fraction"].append(
                latest_validation["active_fraction"]
            )
            history["held_out_gate_probability"].append(
                latest_validation["mean_gate_probability"]
            )
            for size in training_sizes:
                history["held_out_reward_by_m"][size].append(
                    latest_validation["phase_by_m"][size]
                )

            save_checkpoint(
                config.current_checkpoint_file,
                policy,
                optimizer,
                config,
                iteration=iteration + 1,
                held_out_reward=latest_validation["phase_reward"],
            )
            if config.jupyter_mode:
                clear_output(wait=True)
            print(
                f"Iteration {iteration + 1:4d}/{REFINEMENT_ITERATIONS} | "
                f"M={'all' if isinstance(selected_size, tuple) else selected_size} | "
                f"phase={metrics['phase_reward']:.3f} | "
                f"span={detuning_span_at(iteration, config):.2f} GHz | "
                f"K/K_source={metrics['coupling_budget']:.3f} | "
                f"active={metrics['active_fraction']:.3f} | "
                f"p(gate)={metrics['mean_gate_probability']:.3f} | "
                f"phase-safe={metrics['phase_eligible_fraction']:.3f} | "
                f"coupling_weight={coupling_weight:.3f} | "
                f"topology_weight={sparsity_weight:.3f} | "
                "gradient conflicts="
                f"{metrics['negative_gradient_pair_fraction']:.2f} | "
                f"stage={'topology' if topology_stage else 'coupling'} | "
                f"mode={'phase recovery' if phase_recovery_mode else 'optimize'}"
            )
            if should_validate:
                per_size = " | ".join(
                    f"M={size}: {latest_validation['phase_by_m'][size]:.3f}"
                    for size in training_sizes
                )
                print(
                    f"Validation | {per_size} | "
                    f"K/K_source={latest_validation['coupling_budget']:.3f} | "
                    f"active={latest_validation['active_fraction']:.3f}"
                )
            if figure_and_axes is not None:
                figure, axes = figure_and_axes
                update_refinement_figure(figure, axes, history)
                figure.savefig(
                    RESULTS_DIR
                    / f"{Path(config.current_checkpoint_file).stem}_progress.png",
                    dpi=150,
                )
                if config.jupyter_mode:
                    display(figure)
                else:
                    figure.canvas.draw_idle()
                    plt.pause(0.001)

        save_checkpoint(
            config.final_checkpoint_file,
            policy,
            optimizer,
            config,
            iteration=REFINEMENT_ITERATIONS,
            held_out_reward=latest_validation["phase_reward"],
        )
    finally:
        if simulation_pool is not None:
            simulation_pool.close()
            simulation_pool.join()
        if figure_and_axes is not None:
            plt.close(figure_and_axes[0])
    return policy, optimizer, history


def main() -> dict[str, Any]:
    """Load the source checkpoint and optionally refine its coupling budget."""
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Source checkpoint not found: {SOURCE_CHECKPOINT}"
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    source_policy, _, source_config, metadata = load_checkpoint(
        SOURCE_CHECKPOINT, architecture="auto"
    )
    if tuple(source_config.training_n_lasers) != SOURCE_TRAINING_N_LASERS:
        raise ValueError(
            "Checkpoint training sizes do not match "
            "SOURCE_TRAINING_N_LASERS"
        )
    source_has_sparse_gates = bool(
        getattr(source_policy, "enable_sparse_gates", False)
    )
    if source_has_sparse_gates:
        sparse_policy = source_policy
        sparse_source_config = source_config
    else:
        sparse_policy, sparse_source_config = add_sparse_gates_to_policy(
            source_policy,
            source_config,
            # A probability of exactly 0.5 still evaluates as all-open at the
            # deterministic >=0.5 threshold, while giving stochastic gate
            # exploration a useful gradient from the first sparse update.
            initial_gate_logit=0.0,
        )

    refinement_config = replace(
        sparse_source_config,
        training_n_lasers=SOURCE_TRAINING_N_LASERS,
        training_iterations=REFINEMENT_ITERATIONS,
        learning_rate=REFINEMENT_LEARNING_RATE,
        learning_rate_switch_iteration=REFINEMENT_ITERATIONS + 1,
        learning_rate_after_switch=REFINEMENT_LEARNING_RATE,
        targets_per_batch=REFINEMENT_TARGETS_PER_BATCH,
        train_all_sizes_each_iteration=TRAIN_ALL_SIZES_EACH_ITERATION,
        gradient_combination_mode=GRADIENT_COMBINATION_MODE,
        initial_log_std=REFINEMENT_MAXIMUM_LOG_STD,
        maximum_log_std=REFINEMENT_MAXIMUM_LOG_STD,
        enable_exploration_annealing=False,
        detuning_span_ghz=REFINEMENT_DETUNING_SPAN_GHZ,
        detuning_curriculum_initial_half_span_ghz=(
            DETUNING_INITIAL_HALF_SPAN_GHZ
        ),
        detuning_curriculum_warmup_iterations=(
            DETUNING_WARMUP_ITERATIONS
        ),
        detuning_curriculum_iterations=DETUNING_CURRICULUM_ITERATIONS,
        best_checkpoint_file=str(SPARSE_BEST_CHECKPOINT),
        current_checkpoint_file=str(SPARSE_CURRENT_CHECKPOINT),
        final_checkpoint_file=str(SPARSE_FINAL_CHECKPOINT),
    )
    print(f"Loaded source model: {SOURCE_CHECKPOINT}")
    print(
        f"Source checkpoint iteration={metadata['iteration']}, "
        f"held-out reward={metadata['held_out_reward']:.4f}"
    )
    print(
        "Refinement objective: preserve dense-source phase locking at "
        f"{REFINEMENT_DETUNING_SPAN_GHZ:g} GHz while learning which links "
        "can be set exactly to zero"
    )
    if refinement_config.train_all_sizes_each_iteration:
        print(
            "Gradient update: all trained M values every iteration with "
            f"{refinement_config.gradient_combination_mode} "
            f"(per-task clip={refinement_config.per_size_gradient_clip:g})."
        )
    else:
        print("Gradient update: one uniformly sampled M per Adam update.")
    print(
        "Phase-floor penalty switch: "
        f"{'enabled' if ENFORCE_PHASE_REWARD_FLOOR else 'disabled'}"
    )
    if source_has_sparse_gates:
        print("Existing sparse gates retained and frozen.")
    else:
        print(
            "Dense GNN source upgraded with p=0.5 learned gates; "
            "dense, single-edge, and randomized sparse candidates enabled."
        )
    print(
        "Topology reward: full phase reward plus at most "
        f"{TOPOLOGY_SPARSITY_REWARD_FINAL:g} sparsity bonus, only within "
        f"{PHASE_RETENTION_TOLERANCE:g} of the same-target dense candidate."
    )
    print(
        f"Schedule: {GATE_SAMPLING_START_ITERATION} phase-only iterations; "
        "hard gate sampling thereafter; "
        "continuous coupling-budget stage disabled."
    )
    print(f"Refined outputs will use suffix: {SPARSE_MODEL_SUFFIX}")

    optimizer = torch.optim.Adam(
        sparse_policy.parameters(), lr=REFINEMENT_LEARNING_RATE
    )
    history = None
    if RUN_REFINEMENT:
        sparse_policy, optimizer, history = refine_sparse_policy(
            sparse_policy,
            refinement_config,
            live_plot=LIVE_PLOT,
            freeze_shared_topology_features=True,
        )
    return {
        "policy": sparse_policy,
        "optimizer": optimizer,
        "config": refinement_config,
        "source_metadata": metadata,
        "history": history,
    }

#%%
# The process-name check is needed in addition to ``__name__`` when this file
# is launched through a VS Code/Jupyter cell runner. Spawned workers can
# otherwise re-execute the cell as ``__main__`` and recursively create pools.
if (
    __name__ == "__main__"
    and mp.current_process().name == "MainProcess"
):
    sparse_refinement_result = main()
