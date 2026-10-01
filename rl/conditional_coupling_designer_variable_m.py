"""Variable-size conditional reinforcement learning for VCSEL coupling design.

This is a one-step contextual bandit:

* Context/state: a requested relative phase pattern.
* Action: every directed coupling magnitude and coupling phase.
* Environment: one simulation performed by :mod:`vcsel_lib`.
* Reward: agreement between requested and achieved relative phases.
* Update: REINFORCE increases the probability of above-average designs.

There is only one action before the simulator returns a reward. Consequently,
there are no multi-step trajectories and no discounted returns. For each
target, the other sampled coupling designs provide a reward baseline.

The file is intentionally organized in the same order as the data flow:
targets -> per-laser features -> pooled policy -> matrices -> VCSEL -> reward.
It depends only on NumPy, PyTorch, Matplotlib, IPython, and ``vcsel_lib``.

Unlike :mod:`conditional_coupling_designer`, the learned layer dimensions do
not depend on the number of lasers. One policy can therefore train on several
array sizes and be evaluated on a new size without rebuilding its layers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from torch import nn

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.paths import MODEL_DIR, RESULTS_DIR


DEFAULT_N_LASERS = 5
CHECKPOINT_FORMAT = "conditional_coupling_variable_m_pooled_v1"
BIGRU_CHECKPOINT_FORMAT = "conditional_coupling_variable_m_bigru_v1"
BIGRU_M_CONDITIONED_CHECKPOINT_FORMAT = (
    "conditional_coupling_variable_m_bigru_m_conditioned_v2"
)
BIGRU_EDGE_STD_CHECKPOINT_FORMAT = (
    "conditional_coupling_variable_m_bigru_edge_std_v3"
)
BIGRU_SPARSE_CHECKPOINT_FORMAT = (
    "conditional_coupling_variable_m_bigru_sparse_v4"
)
BIGRU_MODULATED_CHECKPOINT_FORMAT = (
    "conditional_coupling_variable_m_bigru_modulated_v5"
)
GNN_CHECKPOINT_FORMAT = "conditional_coupling_variable_m_gnn_v1"


def smoothly_bound_log_std(
    raw_log_std: torch.Tensor,
    minimum_log_std: float,
    maximum_log_std: float,
) -> torch.Tensor:
    """Smoothly constrain log standard deviations to the configured range.

    Unlike ``torch.clamp``, this transformation retains a nonzero gradient
    when a decoder prediction passes a bound, so exploration can recover.
    """
    return (
        minimum_log_std
        + torch.nn.functional.softplus(raw_log_std - minimum_log_std)
        - torch.nn.functional.softplus(raw_log_std - maximum_log_std)
    )


def raw_log_std_for_smooth_bound(
    bounded_log_std: float,
    minimum_log_std: float,
    maximum_log_std: float,
) -> float:
    """Invert :func:`smoothly_bound_log_std` for decoder initialization."""
    if not minimum_log_std < bounded_log_std < maximum_log_std:
        raise ValueError("bounded_log_std must lie strictly inside the bounds")
    log_numerator = np.log(np.expm1(bounded_log_std - minimum_log_std))
    log_denominator = np.log1p(-np.exp(bounded_log_std - maximum_log_std))
    return float(minimum_log_std + log_numerator - log_denominator)


def _load_vcsel_class() -> type:
    """Import VCSEL only when a process actually starts a simulation."""
    try:
        # Script/notebook use from this repository.
        from vcsel_lib import VCSEL
    except ImportError:
        # Package use, where ``vcsel_lib`` resolves to the package directory
        # before the sibling ``vcsel_lib.py`` module.
        from vcsel_lib.vcsel_lib import VCSEL
    return VCSEL


def _initialize_simulation_worker() -> None:
    """Prevent joblib from starting nested process machinery in a worker."""
    os.environ["JOBLIB_MULTIPROCESSING"] = "0"


def directed_link_indices(
    n_lasers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return off-diagonal ``(receiver, source)`` indices in row-major order."""
    adjacency = ~np.eye(n_lasers, dtype=bool)
    return np.where(adjacency)


def coupling_link_indices(
    n_lasers: int,
    force_symmetric_kappa: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the independently learned coupling-magnitude entries."""
    if force_symmetric_kappa:
        return np.triu_indices(n_lasers, k=1)
    return directed_link_indices(n_lasers)


def action_size_for(
    n_lasers: int,
    force_symmetric_kappa: bool = False,
    force_symmetric_phi_p: bool = False,
) -> int:
    """Return the independently learned magnitude and phase actions."""
    magnitude_link_count = (
        n_lasers * (n_lasers - 1) // 2
        if force_symmetric_kappa
        else n_lasers * (n_lasers - 1)
    )
    phase_link_count = (
        n_lasers * (n_lasers - 1) // 2
        if force_symmetric_phi_p
        else n_lasers * (n_lasers - 1)
    )
    return magnitude_link_count + phase_link_count

# PHASE_TICKS = np.array(
#     [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
# )
# PHASE_TICK_LABELS = [
#     r"$-\pi$",
#     r"$-\pi/2$",
#     r"$0$",
#     r"$\pi/2$",
#     r"$\pi$",
# ]

PHASE_TICKS = np.array(
    [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
)
PHASE_TICK_LABELS = [
    r"$-\pi$",
    r"$-\pi/2$",
    r"$0$",
    r"$\pi/2$",
    r"$\pi$",
]


@dataclass
class DesignerConfig:
    """Settings a user is reasonably likely to change.

    The physical defaults are the exact values used by the successful
    five-laser run preserved in ``working_model_backup``. ``n_lasers`` may be
    changed; all policy and matrix dimensions are derived from it.
    """

    # ``n_lasers`` is the size used by direct evaluation/design calls.
    # Training randomly selects one of these array sizes for each update.
    n_lasers: int = 5
    training_n_lasers: tuple[int, ...] = (3, 4, 5, 6, 7)
    # Optional sampling weights for the variable-size training loop.  When
    # omitted, all configured sizes are sampled uniformly.
    training_n_laser_weights: tuple[float, ...] | None = None
    # Optionally begin with a prefix of ``training_n_lasers`` and append one
    # additional size at a fixed iteration interval.
    enable_array_size_curriculum: bool = False
    array_size_curriculum_initial_count: int = 1
    array_size_curriculum_start_iteration: int = 500
    array_size_curriculum_add_interval: int = 100

    # Neural network and REINFORCE training
    training_iterations: int = 700
    # Number of same-size targets in one optimizer update.
    targets_per_batch: int = 16
    candidates_per_target: int = 128
    # Training-only worker granularity. Candidate rewards are reassembled
    # before the REINFORCE loss is evaluated.
    candidates_per_worker_task: int = 16
    node_hidden_sizes: tuple[int, ...] = (64,)
    node_embedding_dim: int = 64
    # Select the pooled, recurrent, or graph message-passing encoder without
    # changing either of the other implementations.
    encoder_architecture: str = "pooled"
    # Retained so older configuration cells can still be loaded. The pooled
    # policy below no longer uses a recurrent GRU hidden state.
    gru_hidden_size: int = 64
    gnn_message_passing_steps: int = 3
    gnn_message_hidden_size: int = 64
    edge_hidden_sizes: tuple[int, ...] = (128, 128)
    # Give the shared edge decoder the normalized array size and inverse
    # neighbor count, and let exploration depend on those same two features.
    condition_on_n_lasers: bool = True
    condition_log_std_on_n_lasers: bool = True
    # Give the GNN message and edge networks only bounded degree features:
    # [1 / (M - 1), 1 / (M - 1)^2].  Unlike normalized M, these features
    # remain in [0, 1] and approach zero for unseen larger arrays.
    condition_on_bounded_degree: bool = False
    # Apply an M-conditioned scale and shift after every hidden edge-decoder
    # activation. This gives array size direct control over the coordinated
    # kappa/phase solution while preserving the shared decoder weights.
    modulate_edge_decoder_by_n_lasers: bool = False
    # Variable-M analogue of the fixed-M policy's per-action exploration:
    # predict separate kappa/phi log standard deviations for every edge.
    edge_conditioned_log_std: bool = False
    # Replace the edge log-standard-deviation hard clamp with a differentiable
    # soft bound. This lets an edge recover after its raw decoder output falls
    # below the configured minimum instead of receiving exactly zero gradient.
    smooth_edge_log_std_bounds: bool = False
    # Sparse refinement adds one Bernoulli gate per independently decoded
    # coupling magnitude. Dense policies leave this disabled.
    enable_sparse_gates: bool = False
    initial_gate_logit: float = 5.0
    gate_probability_threshold: float = 0.5
    # Exact-budget sparse training keeps the existing gate head but treats
    # its outputs as link-selection scores.  Every sampled mask contains a
    # directed spanning tree plus enough additional links to meet the sampled
    # budget, so disconnected laser arrays are impossible by construction.
    enable_connected_edge_budgets: bool = False
    condition_on_edge_budget: bool = False
    edge_budget_warmup_iterations: int = 200
    edge_budget_curriculum_iterations: int = 1000
    # Let the gate head choose both the connected topology and its link count.
    # A spanning tree is always sampled first; every remaining directed link
    # is then a Bernoulli policy action.  Successful sparse candidates receive
    # an additional reward, so REINFORCE learns how many links to retain.
    enable_learned_connected_sparsity: bool = False
    sparsity_phase_reward_threshold: float = 0.98
    # Replace the absolute phase threshold with a per-target reference formed
    # from the best current training candidate.  Sparsity is eligible only
    # within ``phase_retention_tolerance`` of that reference.
    use_relative_phase_retention: bool = False
    phase_retention_tolerance: float = 0.005
    # Use ordinal candidate ranks instead of weighted scalar resource bonuses.
    # Eligible candidates outrank ineligible candidates; eligible candidates
    # are then ordered by link count, coupling cost, and phase reward.
    use_lexicographic_rank_advantages: bool = False
    # Reserve the first training candidate as an all-links-active reference.
    # Its continuous actions remain on-policy, while its gates are not used
    # for a Bernoulli gate update because they are intentionally forced on.
    include_dense_reference_candidate: bool = False
    successful_sparsity_reward_weight: float = 0.05
    # Successful candidates are ranked first by active-link fraction and
    # then, among otherwise equally sparse solutions, by normalized squared
    # coupling cost.  This fraction is the largest possible coupling-cost
    # bonus expressed as a fraction of the reward for removing one link.
    # Keeping it below one makes the ordering strictly lexicographic:
    # phase success -> fewer links -> lower coupling cost.
    coupling_cost_tiebreak_fraction: float = 0.0
    # Learn one global retained-link density per target/candidate.  The
    # sampled density is converted into a deterministic, weakly connected
    # relative-coupling backbone before simulation.
    enable_learned_backbone_density: bool = False
    # Treat the connected link budget as an input condition instead of a
    # policy action. The GNN predicts only kappa/phi under that hard budget.
    enable_conditional_backbone_budget: bool = False
    # Learn phase-error-versus-budget and direct allowable-error-to-budget
    # mappings from the simulations already used by REINFORCE.
    enable_budget_error_selector: bool = False
    selector_budget_levels_per_target: int = 8
    selector_labels_per_target: int = 8
    monotonic_budget_selector: bool = False
    # Reserve one policy-mean candidate at every sampled budget. These
    # deterministic candidates supervise the direct selector with maximum
    # per-laser phase error, matching one-candidate deterministic inference.
    # All remaining candidates continue to train the coupling policy through
    # the unchanged phase-only REINFORCE objective.
    selector_use_deterministic_max_error: bool = False
    selector_training_minimum_error_deg: float = 1.0
    selector_training_maximum_error_deg: float = 30.0
    selector_validation_error_deg: float = 5.0
    initial_backbone_density: float = 1.0
    backbone_density_log_std: float = 0.75
    learning_rate: float = 1.0e-3
    learning_rate_switch_iteration: int = 1500
    learning_rate_after_switch: float = 1.0e-4
    # The default preserves the existing one-array-size-per-update behavior.
    # The optional balanced mode spreads one fixed-size target budget across
    # every configured M and combines their gradients in one optimizer step.
    train_all_sizes_each_iteration: bool = False
    gradient_combination_mode: str = "mean"
    per_size_gradient_clip: float = 1.0
    force_symmetric_kappa: bool = False
    force_symmetric_phi_p: bool = False
    # Keep individual links asymmetric while balancing total upper/lower
    # triangular coupling strength after action decoding.
    balanced_coupling: bool = False
    # Scale every decoded edge by 1/(M-1), bounding the total possible
    # incoming coupling at each receiver independently of array size.
    normalize_incoming_coupling_by_degree: bool = False
    initial_log_std: float = -0.25
    minimum_log_std: float = -3.0
    maximum_log_std: float = 0.75
    # Optionally hold a higher exploration floor while the detuning
    # curriculum expands, then lower it linearly to ``minimum_log_std``.
    enable_minimum_log_std_annealing: bool = False
    initial_minimum_log_std: float = -1.5
    minimum_log_std_anneal_start_iteration: int = 1000
    minimum_log_std_anneal_iterations: int = 500
    enable_exploration_annealing: bool = False
    exploration_anneal_start_iteration: int = 200
    exploration_anneal_iterations: int = 250
    final_maximum_log_std: float = -1.5
    gradient_clip: float = 1.0
    random_seed: int = 7
    device: str = "cpu"
    n_jobs: int = 1

    # Target generation
    structured_target_fraction: float = 0.20
    structured_target_jitter_rad: float = 0.10
    allow_global_phase_conjugate: bool = True
    worst_laser_reward_weight: float = 0.50

    # Softly encourage reciprocal coupling magnitudes from iteration 1.
    magnitude_symmetry_weight: float = 0.0

    # VCSEL physical settings (SI units unless the name says otherwise)
    noise_amplitude: float = 0.0
    maximum_kappa_per_ns: float = 100.0
    initial_kappa_fraction: float = 0.25
    alpha: float = 2.0
    photon_lifetime_seconds: float = 5.4e-12
    carrier_lifetime_seconds: float = 2.5e-10
    gain_per_second: float = 875000.0
    transparency_carriers: float = 286000.0
    gain_saturation: float = 4.0e-6
    electron_charge_coulomb: float = 1.602e-19
    spontaneous_emission_factor: float = 1.0e-3
    delay_seconds: float = 1.0e-9
    pump_current_amp: float = 0.0008609971885714285
    # Detunings are specified relative to the 0-GHz mean-frequency reference.
    # The default range is the full -5 to +5 GHz interval.
    detuning_span_ghz: float = 10.0
    # During training, begin with a narrow random detuning distribution and
    # grow its physical span until the full configured range is reached.
    detuning_curriculum_initial_half_span_ghz: float = 0.05
    detuning_curriculum_warmup_iterations: int = 200
    detuning_curriculum_iterations: int = 1500
    # ``independent`` preserves the original per-laser uniform sampler.
    # ``stratified_span`` first assigns each row a peak-to-peak width from a
    # randomized stratification of the current curriculum interval, then
    # places the lasers within that width. This makes span coverage
    # independent of M.
    detuning_span_sampling_mode: str = "independent"
    time_step_seconds: float = 5.4e-12
    simulation_time_seconds: float = 500.0e-9
    save_every: int = 10
    coupling_ramp_start_delays: float = 2.0
    coupling_ramp_rise_delays: float = 100.0
    reward_tail_fraction: float = 0.35

    # Validation, plotting, and saving
    held_out_target_count: int = 64
    validation_interval: int = 10
    # When enabled, keep the held-out phase targets and normalized detuning
    # patterns fixed while scaling their physical detunings to the maximum
    # span currently allowed by the training curriculum.
    validation_follows_detuning_curriculum: bool = False
    plot_update_interval: int = 1
    best_checkpoint_file: str | None = None
    current_checkpoint_file: str | None = None
    final_checkpoint_file: str | None = None
    jupyter_mode: bool = True

    def __post_init__(self) -> None:
        if self.best_checkpoint_file is None:
            sizes = "-".join(str(size) for size in self.training_n_lasers)
            self.best_checkpoint_file = str(
                MODEL_DIR
                / f"conditional_coupling_variable_m_{sizes}_lasers_best.pt"
            )
        if self.current_checkpoint_file is None:
            sizes = "-".join(str(size) for size in self.training_n_lasers)
            self.current_checkpoint_file = str(
                MODEL_DIR
                / f"conditional_coupling_variable_m_{sizes}_lasers_current.pt"
            )
        if self.final_checkpoint_file is None:
            sizes = "-".join(str(size) for size in self.training_n_lasers)
            self.final_checkpoint_file = str(
                MODEL_DIR
                / f"conditional_coupling_variable_m_{sizes}_lasers_final.pt"
            )

        if self.n_lasers < 2:
            raise ValueError("n_lasers must be at least 2")
        if not self.training_n_lasers:
            raise ValueError("training_n_lasers must not be empty")
        if any(size < 2 for size in self.training_n_lasers):
            raise ValueError("every training_n_lasers value must be at least 2")
        if len(set(self.training_n_lasers)) != len(self.training_n_lasers):
            raise ValueError("training_n_lasers must not contain duplicates")
        if self.training_n_laser_weights is not None:
            if len(self.training_n_laser_weights) != len(self.training_n_lasers):
                raise ValueError(
                    "training_n_laser_weights must have one value per "
                    "training_n_lasers entry"
                )
            if any(
                not np.isfinite(weight) or weight <= 0.0
                for weight in self.training_n_laser_weights
            ):
                raise ValueError(
                    "training_n_laser_weights must contain finite positive "
                    "values"
                )
        if not (
            1
            <= self.array_size_curriculum_initial_count
            <= len(self.training_n_lasers)
        ):
            raise ValueError(
                "array_size_curriculum_initial_count must be between one "
                "and the number of training sizes"
            )
        if self.array_size_curriculum_start_iteration < 0:
            raise ValueError(
                "array_size_curriculum_start_iteration must be nonnegative"
            )
        if self.array_size_curriculum_add_interval < 1:
            raise ValueError(
                "array_size_curriculum_add_interval must be positive"
            )
        if self.targets_per_batch < 1:
            raise ValueError("targets_per_batch must be positive")
        if self.candidates_per_target < 2:
            raise ValueError("candidates_per_target must be at least 2")
        if self.candidates_per_worker_task < 1:
            raise ValueError("candidates_per_worker_task must be positive")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if len(self.node_hidden_sizes) < 1:
            raise ValueError("node_hidden_sizes must contain at least one layer")
        if self.node_embedding_dim < 1:
            raise ValueError("node_embedding_dim must be positive")
        if self.encoder_architecture not in {"pooled", "bigru", "gnn"}:
            raise ValueError(
                "encoder_architecture must be 'pooled', 'bigru', or 'gnn'"
            )
        if (
            self.edge_conditioned_log_std
            and self.encoder_architecture not in {"bigru", "gnn"}
        ):
            raise ValueError(
                "edge_conditioned_log_std is supported only by the bigru "
                "and gnn architectures"
            )
        if (
            self.modulate_edge_decoder_by_n_lasers
            and self.encoder_architecture != "bigru"
        ):
            raise ValueError(
                "modulate_edge_decoder_by_n_lasers is currently supported "
                "only by the bigru architecture"
            )
        if (
            self.condition_on_bounded_degree
            and self.encoder_architecture != "gnn"
        ):
            raise ValueError(
                "condition_on_bounded_degree is supported only by the gnn "
                "architecture"
            )
        if self.condition_on_bounded_degree and self.condition_on_n_lasers:
            raise ValueError(
                "condition_on_bounded_degree and condition_on_n_lasers "
                "cannot both be enabled"
            )
        if (
            self.edge_conditioned_log_std
            and self.condition_log_std_on_n_lasers
        ):
            raise ValueError(
                "edge_conditioned_log_std and "
                "condition_log_std_on_n_lasers cannot both be enabled"
            )
        if (
            self.smooth_edge_log_std_bounds
            and not self.edge_conditioned_log_std
        ):
            raise ValueError(
                "smooth_edge_log_std_bounds requires "
                "edge_conditioned_log_std=True"
            )
        if (
            self.enable_sparse_gates
            and self.encoder_architecture not in {"bigru", "gnn"}
        ):
            raise ValueError(
                "enable_sparse_gates is supported only by the bigru and "
                "gnn architectures"
            )
        if not np.isfinite(self.initial_gate_logit):
            raise ValueError("initial_gate_logit must be finite")
        if not 0.0 < self.gate_probability_threshold < 1.0:
            raise ValueError(
                "gate_probability_threshold must lie strictly between 0 and 1"
            )
        if self.enable_connected_edge_budgets:
            if not self.enable_sparse_gates:
                raise ValueError(
                    "connected edge budgets require enable_sparse_gates=True"
                )
            if self.encoder_architecture != "gnn":
                raise ValueError(
                    "connected edge budgets are currently supported only "
                    "by the GNN architecture"
                )
            if self.force_symmetric_kappa:
                raise ValueError(
                    "connected edge budgets currently require directed "
                    "coupling magnitudes"
                )
            if self.force_symmetric_phi_p:
                raise ValueError(
                    "connected edge budgets currently require directed "
                    "coupling phases"
                )
            if not self.condition_on_edge_budget:
                raise ValueError(
                    "connected edge budgets require "
                    "condition_on_edge_budget=True"
                )
        if self.enable_learned_connected_sparsity:
            if self.enable_connected_edge_budgets:
                raise ValueError(
                    "learned connected sparsity and exact edge budgets are "
                    "mutually exclusive"
                )
            if not self.enable_sparse_gates:
                raise ValueError(
                    "learned connected sparsity requires "
                    "enable_sparse_gates=True"
                )
            if self.encoder_architecture != "gnn":
                raise ValueError(
                    "learned connected sparsity is currently supported only "
                    "by the GNN architecture"
                )
            if self.force_symmetric_kappa or self.force_symmetric_phi_p:
                raise ValueError(
                    "learned connected sparsity currently requires directed "
                    "coupling magnitudes and phases"
                )
            if self.condition_on_edge_budget:
                raise ValueError(
                    "learned connected sparsity chooses its own link count, "
                    "so condition_on_edge_budget must be False"
                )
        if not 0.0 <= self.sparsity_phase_reward_threshold <= 1.0:
            raise ValueError(
                "sparsity_phase_reward_threshold must lie in [0, 1]"
            )
        if self.phase_retention_tolerance < 0.0:
            raise ValueError(
                "phase_retention_tolerance must be nonnegative"
            )
        if (
            self.use_relative_phase_retention
            and not self.enable_learned_connected_sparsity
        ):
            raise ValueError(
                "relative phase retention requires "
                "enable_learned_connected_sparsity=True"
            )
        if (
            self.include_dense_reference_candidate
            and not self.use_relative_phase_retention
        ):
            raise ValueError(
                "a dense reference candidate requires relative phase "
                "retention"
            )
        if (
            self.use_lexicographic_rank_advantages
            and not self.use_relative_phase_retention
        ):
            raise ValueError(
                "lexicographic rank advantages require relative phase "
                "retention"
            )
        if self.successful_sparsity_reward_weight < 0.0:
            raise ValueError(
                "successful_sparsity_reward_weight must be nonnegative"
            )
        if not 0.0 <= self.coupling_cost_tiebreak_fraction < 1.0:
            raise ValueError(
                "coupling_cost_tiebreak_fraction must lie in [0, 1)"
            )
        if not 0.0 < self.initial_backbone_density <= 1.0:
            raise ValueError("initial_backbone_density must lie in (0, 1]")
        if not np.isfinite(self.backbone_density_log_std):
            raise ValueError("backbone_density_log_std must be finite")
        if self.enable_learned_backbone_density:
            if self.encoder_architecture != "gnn":
                raise ValueError(
                    "learned backbone density is currently supported only "
                    "by the GNN architecture"
                )
            if self.enable_sparse_gates or self.enable_connected_edge_budgets:
                raise ValueError(
                    "learned backbone density cannot be combined with edge "
                    "gates or exact edge budgets"
                )
        if self.enable_conditional_backbone_budget:
            if self.encoder_architecture != "gnn":
                raise ValueError(
                    "conditional backbone budgets require the GNN architecture"
                )
            if not self.condition_on_edge_budget:
                raise ValueError(
                    "conditional backbone budgets require "
                    "condition_on_edge_budget=True"
                )
            if (
                self.enable_sparse_gates
                or self.enable_connected_edge_budgets
                or self.enable_learned_backbone_density
            ):
                raise ValueError(
                    "conditional backbone budgets cannot be combined with "
                    "edge gates or learned rho"
                )
        if self.enable_budget_error_selector:
            if not self.enable_conditional_backbone_budget:
                raise ValueError(
                    "the budget error selector requires conditional budgets"
                )
            if not (
                0.0 < self.selector_training_minimum_error_deg
                < self.selector_training_maximum_error_deg
                <= 180.0
            ):
                raise ValueError(
                    "selector training errors must satisfy "
                    "0 < minimum < maximum <= 180 degrees"
                )
            if not 0.0 < self.selector_validation_error_deg <= 180.0:
                raise ValueError(
                    "selector_validation_error_deg must lie in (0, 180]"
                )
            if self.selector_budget_levels_per_target < 2:
                raise ValueError(
                    "selector_budget_levels_per_target must be at least 2"
                )
            if self.selector_labels_per_target < 1:
                raise ValueError(
                    "selector_labels_per_target must be positive"
                )
        if self.monotonic_budget_selector and not self.enable_budget_error_selector:
            raise ValueError(
                "monotonic_budget_selector requires the budget error selector"
            )
        if (
            self.selector_use_deterministic_max_error
            and not self.enable_budget_error_selector
        ):
            raise ValueError(
                "deterministic maximum-error selector labels require the "
                "budget error selector"
            )
        if (
            self.coupling_cost_tiebreak_fraction > 0.0
            and not self.enable_learned_connected_sparsity
        ):
            raise ValueError(
                "coupling-cost tie-breaking requires "
                "enable_learned_connected_sparsity=True"
            )
        if self.condition_on_edge_budget and self.encoder_architecture != "gnn":
            raise ValueError(
                "edge-budget conditioning is currently supported only by "
                "the GNN architecture"
            )
        if self.edge_budget_warmup_iterations < 0:
            raise ValueError(
                "edge_budget_warmup_iterations must be nonnegative"
            )
        if self.edge_budget_curriculum_iterations < 0:
            raise ValueError(
                "edge_budget_curriculum_iterations must be nonnegative"
            )
        if self.gru_hidden_size < 1:
            raise ValueError("gru_hidden_size must be positive")
        if self.gnn_message_passing_steps < 1:
            raise ValueError("gnn_message_passing_steps must be positive")
        if self.gnn_message_hidden_size < 1:
            raise ValueError("gnn_message_hidden_size must be positive")
        if len(self.edge_hidden_sizes) < 1:
            raise ValueError("edge_hidden_sizes must contain at least one layer")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.learning_rate_switch_iteration < 0:
            raise ValueError(
                "learning_rate_switch_iteration must be nonnegative"
            )
        if self.learning_rate_after_switch <= 0.0:
            raise ValueError("learning_rate_after_switch must be positive")
        if self.gradient_combination_mode not in {"mean", "pcgrad"}:
            raise ValueError(
                "gradient_combination_mode must be 'mean' or 'pcgrad'"
            )
        if self.per_size_gradient_clip <= 0.0:
            raise ValueError("per_size_gradient_clip must be positive")
        if self.train_all_sizes_each_iteration:
            if self.targets_per_batch < len(self.training_n_lasers):
                raise ValueError(
                    "all-size updates require at least one target per "
                    "training_n_lasers entry"
                )
            if self.enable_array_size_curriculum:
                raise ValueError(
                    "all-size updates and the array-size curriculum cannot "
                    "be enabled together"
                )
        if not (
            self.minimum_log_std
            <= self.initial_log_std
            <= self.maximum_log_std
        ):
            raise ValueError(
                "initial_log_std must be within the log-std bounds"
            )
        if self.enable_minimum_log_std_annealing:
            if not (
                self.minimum_log_std
                <= self.initial_minimum_log_std
                <= self.maximum_log_std
            ):
                raise ValueError(
                    "initial_minimum_log_std must be within the log-std "
                    "bounds"
                )
            if self.initial_log_std < self.initial_minimum_log_std:
                raise ValueError(
                    "initial_log_std must not be below "
                    "initial_minimum_log_std"
                )
        if self.minimum_log_std_anneal_start_iteration < 0:
            raise ValueError(
                "minimum_log_std_anneal_start_iteration must be nonnegative"
            )
        if self.minimum_log_std_anneal_iterations < 0:
            raise ValueError(
                "minimum_log_std_anneal_iterations must be nonnegative"
            )
        if not (
            self.minimum_log_std
            <= self.final_maximum_log_std
            <= self.maximum_log_std
        ):
            raise ValueError(
                "final_maximum_log_std must be within the log-std bounds"
            )
        if self.exploration_anneal_start_iteration < 0:
            raise ValueError(
                "exploration_anneal_start_iteration must be nonnegative"
            )
        if self.exploration_anneal_iterations < 0:
            raise ValueError(
                "exploration_anneal_iterations must be nonnegative"
            )
        if not 0.0 <= self.structured_target_fraction <= 1.0:
            raise ValueError("structured_target_fraction must be in [0, 1]")
        if not 0.0 <= self.worst_laser_reward_weight <= 1.0:
            raise ValueError("worst_laser_reward_weight must be in [0, 1]")
        if self.magnitude_symmetry_weight < 0.0:
            raise ValueError("magnitude_symmetry_weight must be nonnegative")
        if self.maximum_kappa_per_ns <= 0.0:
            raise ValueError("maximum_kappa_per_ns must be positive")
        if self.time_step_seconds <= 0.0:
            raise ValueError("time_step_seconds must be positive")
        if self.simulation_time_seconds <= 0.0:
            raise ValueError("simulation_time_seconds must be positive")
        if self.detuning_span_ghz <= 0.0:
            raise ValueError("detuning_span_ghz must be positive")
        if self.detuning_span_sampling_mode not in {
            "independent",
            "stratified_span",
        }:
            raise ValueError(
                "detuning_span_sampling_mode must be 'independent' or "
                "'stratified_span'"
            )
        if not (
            0.0
            <= self.detuning_curriculum_initial_half_span_ghz
            <= 0.5 * self.detuning_span_ghz
        ):
            raise ValueError(
                "detuning_curriculum_initial_half_span_ghz must be "
                "between zero and half the configured detuning span"
            )
        if self.detuning_curriculum_iterations < 0:
            raise ValueError(
                "detuning_curriculum_iterations must be nonnegative"
            )
        if not (
            0
            <= self.detuning_curriculum_warmup_iterations
            <= self.detuning_curriculum_iterations
        ):
            raise ValueError(
                "detuning_curriculum_warmup_iterations must be between "
                "zero and detuning_curriculum_iterations"
            )
        if self.held_out_target_count < 1:
            raise ValueError("held_out_target_count must be positive")


def set_random_seed(seed: int) -> np.random.Generator:
    """Seed NumPy and PyTorch and return the NumPy random generator."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 1. Requested phase patterns
# ---------------------------------------------------------------------------


def make_relative_to_laser_1(phases_rad: np.ndarray) -> np.ndarray:
    """Reference phases to laser 1 and wrap them to ``[-pi, pi]``.

    Input shape is ``(N,)`` or ``(batch, N)``. The first output phase is zero.
    """
    phases_rad = np.asarray(phases_rad, dtype=float)
    if phases_rad.ndim < 1 or phases_rad.shape[-1] < 2:
        raise ValueError("phase arrays must contain at least two lasers")
    relative = phases_rad - phases_rad[..., :1]
    return np.angle(np.exp(1j * relative))


def canonicalize_global_phase_conjugate(
    target_phases_rad: np.ndarray,
) -> np.ndarray:
    """Give a target and its complete sign reversal one representation.

    If global phase conjugacy is enabled, ``[0, a, b, ...]`` and
    ``[0, -a, -b, ...]`` describe the same requested relationship. One sign
    is chosen for the *whole* pattern based on its first nonzero entry. Signs
    are never chosen independently for different lasers.
    """
    targets = make_relative_to_laser_1(target_phases_rad)
    single_target = targets.ndim == 1
    targets_2d = targets[None, :] if single_target else targets.copy()
    nonreference = targets_2d[:, 1:]
    nonzero = np.abs(nonreference) > 1.0e-7
    first_nonzero_index = np.argmax(nonzero, axis=1)
    first_nonzero_value = nonreference[
        np.arange(len(targets_2d)), first_nonzero_index
    ]
    has_nonzero = np.any(nonzero, axis=1)
    sign = np.where(has_nonzero & (first_nonzero_value < 0.0), -1.0, 1.0)
    canonical = np.angle(np.exp(1j * sign[:, None] * targets_2d))
    canonical[:, 0] = 0.0
    return canonical[0] if single_target else canonical


def named_phase_targets(
    n_lasers: int = DEFAULT_N_LASERS,
) -> dict[str, np.ndarray]:
    """Return readable targets for an ``n_lasers`` validation system."""
    if n_lasers < 2:
        raise ValueError("n_lasers must be at least 2")
    binary = np.pi * (np.arange(n_lasers) % 2)
    multicluster = 2.0 * np.pi * (np.arange(n_lasers) % 3) / 3.0
    if n_lasers == 5:
        arbitrary = np.array(
            [0.0, 0.4 * np.pi, -0.7 * np.pi, 0.9 * np.pi, -0.2 * np.pi]
        )
    else:
        arbitrary = 0.7 * np.pi * np.arange(n_lasers)
    return {
        "in-phase": np.zeros(n_lasers),
        "splay": 2.0 * np.pi * np.arange(n_lasers) / n_lasers,
        "binary": make_relative_to_laser_1(binary),
        "multicluster": make_relative_to_laser_1(multicluster),
        "arbitrary": make_relative_to_laser_1(arbitrary),
    }


def sample_target_phases(
    count: int,
    rng: np.random.Generator,
    n_lasers: int = DEFAULT_N_LASERS,
    structured_fraction: float = 0.20,
    structured_jitter_rad: float = 0.10,
) -> np.ndarray:
    """Sample ordinary random targets plus a few structured examples.

    The output has shape ``(count, n_lasers)`` and is relative to laser 1.
    """
    if count < 1:
        raise ValueError("count must be positive")
    targets = rng.uniform(-np.pi, np.pi, size=(count, n_lasers))
    structured = list(named_phase_targets(n_lasers).values())[:4]
    for index in range(count):
        if rng.random() < structured_fraction:
            targets[index] = structured[rng.integers(len(structured))]
            targets[index, 1:] += rng.normal(
                0.0, structured_jitter_rad, size=n_lasers - 1
            )
    return make_relative_to_laser_1(targets)


def detuning_curriculum_half_span_at(
    iteration: int | None,
    config: DesignerConfig,
) -> float:
    """Return the maximum absolute detuning allowed by the curriculum."""
    final_half_span = 0.5 * config.detuning_span_ghz
    if iteration is None or config.detuning_curriculum_iterations == 0:
        return final_half_span
    if iteration < config.detuning_curriculum_warmup_iterations:
        span_fraction = 0.0
    else:
        ramp_iterations = max(
            config.detuning_curriculum_iterations
            - config.detuning_curriculum_warmup_iterations,
            1,
        )
        span_fraction = np.clip(
            float(iteration - config.detuning_curriculum_warmup_iterations)
            / float(ramp_iterations),
            0.0,
            1.0,
        )
    return float(
        config.detuning_curriculum_initial_half_span_ghz
        + span_fraction
        * (
            final_half_span
            - config.detuning_curriculum_initial_half_span_ghz
        )
    )


def sample_detuning_distributions(
    count: int,
    rng: np.random.Generator,
    config: DesignerConfig,
    iteration: int | None = None,
) -> np.ndarray:
    """Sample one ordered detuning vector per target, in GHz.

    The returned values are detunings relative to the mean optical frequency,
    whose reference is 0 GHz. Training holds a narrow random span for the
    configured warm-up, then grows that span to the full configured range.
    With ``stratified_span`` sampling, each row is assigned one randomized
    stratum of peak-to-peak width so every batch covers the current curriculum
    interval consistently for every M. Rows are centered and sorted so matrix
    indices always run from the lowest to the highest detuning.
    """
    if count < 1:
        raise ValueError("count must be positive")
    half_span = 0.5 * config.detuning_span_ghz
    current_half_span = detuning_curriculum_half_span_at(iteration, config)

    if config.detuning_span_sampling_mode == "stratified_span":
        # Draw exactly one randomized width from each equal-width stratum,
        # then shuffle their association with target phase patterns. For a
        # one-row design/evaluation call this reduces to an ordinary uniform
        # draw over the current peak-to-peak interval.
        current_total_span = 2.0 * current_half_span
        desired_spans = current_total_span * (
            np.arange(count, dtype=float) + rng.random(count)
        ) / float(count)
        rng.shuffle(desired_spans)

        # Random order statistics give an irregular array while explicitly
        # retaining both ends of the requested interval. After subtracting
        # the row mean, the optical reference is zero and the peak-to-peak
        # width remains exactly desired_spans.
        unit_positions = rng.uniform(
            0.0, 1.0, size=(count, config.n_lasers)
        )
        unit_positions[:, 0] = 0.0
        unit_positions[:, -1] = 1.0
        unit_positions.sort(axis=1)
        detunings = unit_positions * desired_spans[:, None]
        detunings -= detunings.mean(axis=1, keepdims=True)
        return detunings.astype(np.float64)

    random_detunings = rng.uniform(
        -half_span, half_span, size=(count, config.n_lasers)
    )
    random_detunings -= random_detunings.mean(axis=1, keepdims=True)
    random_abs_max = np.max(np.abs(random_detunings), axis=1, keepdims=True)
    random_detunings *= np.minimum(
        1.0, half_span / np.maximum(random_abs_max, 1.0e-12)
    )
    detunings = random_detunings * (current_half_span / half_span)
    # Keep the physical reference at the mean frequency and make the ordering
    # explicit after scaling.
    detunings -= detunings.mean(axis=1, keepdims=True)
    return np.sort(detunings, axis=1).astype(np.float64)


def sort_by_detuning(
    target_phases_rad: np.ndarray,
    detuning_distribution_ghz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize a target/condition pair by increasing detuning.

    This makes the coupling matrix representation permutation-aware: a
    permutation of physically labelled lasers produces the same ordered
    policy input, while the corresponding target phases are permuted with
    their lasers.
    """
    targets = np.asarray(target_phases_rad, dtype=float)
    detunings = np.asarray(detuning_distribution_ghz, dtype=float)
    single = targets.ndim == 1
    if single:
        targets = targets[None, :]
    if detunings.ndim == 1:
        detunings = detunings[None, :]
    if targets.ndim != 2 or detunings.ndim != 2:
        raise ValueError("targets and detunings must be one- or two-dimensional")
    if targets.shape != detunings.shape:
        raise ValueError("targets and detunings must have the same shape")
    order = np.argsort(detunings, axis=1, kind="stable")
    ordered_targets = np.take_along_axis(targets, order, axis=1)
    ordered_detunings = np.take_along_axis(detunings, order, axis=1)
    return (
        make_relative_to_laser_1(ordered_targets),
        ordered_detunings,
    ) if not single else (
        make_relative_to_laser_1(ordered_targets[0]),
        ordered_detunings[0],
    )


# ---------------------------------------------------------------------------
# 2. Target encoding and policy network
# ---------------------------------------------------------------------------


def encode_target_phases(target_phases_rad: np.ndarray) -> np.ndarray:
    """Encode the nonreference phases as cosine/sine features.

    ``(batch, 5) -> (batch, 8)`` and ``(5,) -> (8,)``.
    """
    relative = make_relative_to_laser_1(target_phases_rad)
    nonreference = relative[..., 1:]
    encoded = np.concatenate(
        (np.cos(nonreference), np.sin(nonreference)), axis=-1
    )
    return encoded.astype(np.float32)


def encode_detuning_distributions(
    detuning_distribution_ghz: np.ndarray,
    config: DesignerConfig,
) -> np.ndarray:
    """Encode detunings relative to the mean-frequency reference (0 GHz)."""
    detunings = np.asarray(detuning_distribution_ghz, dtype=float)
    single = detunings.ndim == 1
    if single:
        detunings = detunings[None, :]
    if detunings.shape != (detunings.shape[0], config.n_lasers):
        raise ValueError(
            "detuning distributions must have shape "
            f"(batch, {config.n_lasers})"
        )
    if not np.all(np.isfinite(detunings)):
        raise ValueError("detuning distributions must be finite")
    # Canonicalize laser permutations before encoding.  The target phases are
    # reordered alongside this vector by ``sort_by_detuning`` at the public
    # training/evaluation boundaries.
    detunings = np.sort(detunings, axis=1)
    # The simulator continues to receive ``detunings`` in GHz.  Only this
    # copy is normalized, using a fixed scale so the network sees roughly
    # [-1, 1] over the configured -span/2..+span/2 interval.
    scale = max(0.5 * config.detuning_span_ghz, 1.0e-12)
    encoded = (detunings / scale).astype(np.float32)
    return encoded[0] if single else encoded


class PolicyNetwork(nn.Module):
    """Size-independent Gaussian policy with a pooled global context."""

    def __init__(self, config: DesignerConfig):
        super().__init__()
        self.force_symmetric_kappa = config.force_symmetric_kappa
        self.force_symmetric_phi_p = config.force_symmetric_phi_p
        self.node_embedding_dim = config.node_embedding_dim

        # The same small encoder processes every laser independently.
        node_layers: list[nn.Module] = []
        input_width = 3
        for output_width in config.node_hidden_sizes:
            linear = nn.Linear(input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            node_layers.extend((linear, nn.SiLU()))
            input_width = output_width
        node_output = nn.Linear(input_width, self.node_embedding_dim)
        nn.init.orthogonal_(node_output.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(node_output.bias)
        node_layers.extend((node_output, nn.SiLU()))
        self.node_encoder = nn.Sequential(*node_layers)

        # A mean/max pool summarizes the whole array without introducing a
        # preferred laser index or dependence on the sorted sequence order.
        self.global_embedding_dim = 2 * self.node_embedding_dim
        global_input_width = 2 * self.node_embedding_dim
        global_layers: list[nn.Module] = []
        for output_width in (
            self.global_embedding_dim,
            self.global_embedding_dim,
        ):
            linear = nn.Linear(global_input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            global_layers.extend((linear, nn.SiLU()))
            global_input_width = output_width
        self.global_encoder = nn.Sequential(*global_layers)

        # Each directed edge sees source and receiver context plus the same
        # four local physical features used by the fixed-size policy.
        edge_layers: list[nn.Module] = []
        edge_input_width = (
            self.global_embedding_dim + 2 * self.node_embedding_dim + 4
        )
        for output_width in config.edge_hidden_sizes:
            linear = nn.Linear(edge_input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            edge_layers.extend((linear, nn.SiLU()))
            edge_input_width = output_width
        edge_output = nn.Linear(edge_input_width, 2)
        nn.init.orthogonal_(edge_output.weight, gain=0.01)
        nn.init.zeros_(edge_output.bias)

        initial_fraction = config.initial_kappa_fraction
        initial_kappa_logit = np.log(
            initial_fraction / (1.0 - initial_fraction)
        )
        with torch.no_grad():
            edge_output.bias.copy_(
                torch.tensor(
                    [initial_kappa_logit, 0.0],
                    dtype=edge_output.bias.dtype,
                )
            )
        edge_layers.append(edge_output)
        self.edge_decoder = nn.Sequential(*edge_layers)

        # Two scalar exploration parameters work for any number of edges.
        self.log_std_kappa = nn.Parameter(
            torch.tensor(float(config.initial_log_std))
        )
        self.log_std_phi = nn.Parameter(
            torch.tensor(float(config.initial_log_std))
        )
        self.minimum_log_std = config.minimum_log_std
        self.maximum_log_std = config.maximum_log_std

    def link_counts(self, n_lasers: int) -> tuple[int, int, int]:
        """Return directed, magnitude-action, and phase-action counts."""
        directed = n_lasers * (n_lasers - 1)
        magnitude = directed // 2 if self.force_symmetric_kappa else directed
        phase = directed // 2 if self.force_symmetric_phi_p else directed
        return directed, magnitude, phase

    def action_size_for(self, n_lasers: int) -> int:
        """Return this policy's action width for ``n_lasers``."""
        _, magnitude, phase = self.link_counts(n_lasers)
        return magnitude + phase

    def action_mean(self, node_features: torch.Tensor) -> torch.Tensor:
        """Return action means for a ``(batch, M, 3)`` feature tensor."""
        if node_features.ndim != 3 or node_features.shape[2] != 3:
            raise ValueError("node_features must have shape (batch, M, 3)")
        batch_size, n_lasers, _ = node_features.shape
        if n_lasers < 2:
            raise ValueError("node_features must contain at least two lasers")

        # (batch, M, node_embedding_dim)
        node_embeddings = self.node_encoder(node_features)
        # Pool independently encoded lasers into an order-independent global
        # context. The max term preserves strong/extreme node features that a
        # mean alone would dilute.
        pooled_embeddings = torch.cat(
            (
                node_embeddings.mean(dim=1),
                node_embeddings.amax(dim=1),
            ),
            dim=1,
        )
        global_embedding = self.global_encoder(pooled_embeddings)

        receiver_numpy, source_numpy = directed_link_indices(n_lasers)
        link_receivers = torch.as_tensor(
            receiver_numpy, dtype=torch.long, device=node_features.device
        )
        link_sources = torch.as_tensor(
            source_numpy, dtype=torch.long, device=node_features.device
        )
        n_directed_links, _, _ = self.link_counts(n_lasers)

        source_embeddings = node_embeddings.index_select(
            1, link_sources
        )
        receiver_embeddings = node_embeddings.index_select(
            1, link_receivers
        )
        global_edge_embedding = global_embedding.unsqueeze(1).expand(
            -1, n_directed_links, -1
        )

        cosine = node_features[:, :, 0]
        sine = node_features[:, :, 1]
        normalized_detuning = node_features[:, :, 2]
        local_cosine = (
            cosine[:, link_sources] * cosine[:, link_receivers]
            + sine[:, link_sources] * sine[:, link_receivers]
        )
        local_sine = (
            sine[:, link_sources] * cosine[:, link_receivers]
            - cosine[:, link_sources] * sine[:, link_receivers]
        )
        local_detuning_difference = 0.5 * (
            normalized_detuning[:, link_sources]
            - normalized_detuning[:, link_receivers]
        )
        local_detuning_features = torch.stack(
            (
                local_detuning_difference,
                torch.abs(local_detuning_difference),
            ),
            dim=2,
        )
        local_phase_features = torch.stack(
            (local_cosine, local_sine), dim=2
        )
        edge_features = torch.cat(
            (local_phase_features, local_detuning_features), dim=2
        )
        edge_inputs = torch.cat(
            (
                global_edge_embedding,
                source_embeddings,
                receiver_embeddings,
                edge_features,
            ),
            dim=2,
        )
        edge_outputs = self.edge_decoder(
            edge_inputs.reshape(batch_size * n_directed_links, -1)
        ).reshape(batch_size, n_directed_links, 2)

        directed_magnitude_means = 6.0 * torch.tanh(
            edge_outputs[:, :, 0] / 6.0
        )
        directed_phase_means = edge_outputs[:, :, 1]
        upper_triangle = torch.as_tensor(
            np.flatnonzero(receiver_numpy < source_numpy),
            dtype=torch.long,
            device=node_features.device,
        )
        magnitude_means = (
            directed_magnitude_means.index_select(1, upper_triangle)
            if self.force_symmetric_kappa
            else directed_magnitude_means
        )
        phase_means = (
            directed_phase_means.index_select(1, upper_triangle)
            if self.force_symmetric_phi_p
            else directed_phase_means
        )
        return torch.cat((magnitude_means, phase_means), dim=1)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Return means shaped ``(batch, action_size_for(M))``."""
        return self.action_mean(node_features)

    def distribution(
        self, node_features: torch.Tensor
    ) -> torch.distributions.Normal:
        mean = self(node_features)
        n_lasers = node_features.shape[1]
        _, n_magnitude_links, n_phase_links = self.link_counts(n_lasers)
        log_std_kappa = torch.clamp(
            self.log_std_kappa, self.minimum_log_std, self.maximum_log_std
        ).expand(n_magnitude_links)
        log_std_phi = torch.clamp(
            self.log_std_phi, self.minimum_log_std, self.maximum_log_std
        ).expand(n_phase_links)
        log_std = torch.cat((log_std_kappa, log_std_phi))
        return torch.distributions.Normal(mean, torch.exp(log_std))


class BiGRUPolicyNetwork(nn.Module):
    """Legacy variable-size policy used by the archived ``*_bigru`` runs.

    This is intentionally kept small and separate from :class:`PolicyNetwork`:
    it allows old checkpoints to be inspected or reproduced without changing
    the current pooled architecture.
    """

    def __init__(self, config: DesignerConfig):
        super().__init__()
        self.force_symmetric_kappa = config.force_symmetric_kappa
        self.force_symmetric_phi_p = config.force_symmetric_phi_p
        self.node_embedding_dim = config.node_embedding_dim
        self.gru_hidden_size = config.gru_hidden_size
        self.minimum_training_n_lasers = min(config.training_n_lasers)
        self.maximum_training_n_lasers = max(config.training_n_lasers)
        self.condition_on_n_lasers = config.condition_on_n_lasers
        self.condition_log_std_on_n_lasers = (
            config.condition_log_std_on_n_lasers
        )
        self.modulate_edge_decoder_by_n_lasers = (
            config.modulate_edge_decoder_by_n_lasers
        )
        self.edge_conditioned_log_std = config.edge_conditioned_log_std
        self.smooth_edge_log_std_bounds = (
            config.smooth_edge_log_std_bounds
        )
        self.enable_sparse_gates = config.enable_sparse_gates
        self.gate_probability_threshold = config.gate_probability_threshold

        node_layers: list[nn.Module] = []
        input_width = 3
        for output_width in config.node_hidden_sizes:
            linear = nn.Linear(input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            node_layers.extend((linear, nn.SiLU()))
            input_width = output_width
        node_output = nn.Linear(input_width, self.node_embedding_dim)
        nn.init.orthogonal_(node_output.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(node_output.bias)
        node_layers.extend((node_output, nn.SiLU()))
        self.node_encoder = nn.Sequential(*node_layers)

        self.bigru = nn.GRU(
            input_size=self.node_embedding_dim,
            hidden_size=self.gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Each edge uses source/receiver BiGRU states, four local physical
        # features, and (for new models) two explicit array-size features.
        edge_input_width = 4 * self.gru_hidden_size + 4
        if self.condition_on_n_lasers:
            edge_input_width += 2
        edge_layers: list[nn.Module] = []
        self.edge_hidden_sizes = tuple(config.edge_hidden_sizes)
        for output_width in config.edge_hidden_sizes:
            linear = nn.Linear(edge_input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            edge_layers.extend((linear, nn.SiLU()))
            edge_input_width = output_width
        edge_output_width = 4 if self.edge_conditioned_log_std else 2
        edge_output = nn.Linear(edge_input_width, edge_output_width)
        nn.init.orthogonal_(edge_output.weight, gain=0.01)
        nn.init.zeros_(edge_output.bias)
        initial_fraction = config.initial_kappa_fraction
        initial_kappa_logit = np.log(
            initial_fraction / (1.0 - initial_fraction)
        )
        with torch.no_grad():
            edge_output.bias[:2].copy_(
                torch.tensor(
                    [initial_kappa_logit, 0.0],
                    dtype=edge_output.bias.dtype,
                )
            )
            if self.edge_conditioned_log_std:
                # Begin with exactly the same shared exploration used before;
                # the two extra rows then learn edge-specific deviations.
                edge_output.weight[2:].zero_()
                initial_raw_log_std = float(config.initial_log_std)
                if self.smooth_edge_log_std_bounds:
                    initial_raw_log_std = raw_log_std_for_smooth_bound(
                        config.initial_log_std,
                        config.minimum_log_std,
                        config.maximum_log_std,
                    )
                edge_output.bias[2:].fill_(initial_raw_log_std)
        edge_layers.append(edge_output)
        self.edge_decoder = nn.Sequential(*edge_layers)
        if self.modulate_edge_decoder_by_n_lasers:
            # One small conditioning layer produces a scale correction and a
            # shift for every hidden decoder channel. Zero initialization
            # makes scale=1 and shift=0, exactly recovering the unmodulated
            # decoder at initialization.
            modulation_width = 2 * sum(self.edge_hidden_sizes)
            self.edge_size_modulation = nn.Linear(2, modulation_width)
            nn.init.zeros_(self.edge_size_modulation.weight)
            nn.init.zeros_(self.edge_size_modulation.bias)
        if self.enable_sparse_gates:
            # A separate head preserves every dense decoder parameter exactly
            # when upgrading an existing checkpoint. Its zero weights and
            # positive bias initially leave almost every connection active.
            self.edge_gate_head = nn.Linear(edge_input_width, 1)
            nn.init.zeros_(self.edge_gate_head.weight)
            nn.init.constant_(
                self.edge_gate_head.bias, float(config.initial_gate_logit)
            )
        self.log_std_kappa = nn.Parameter(
            torch.tensor(float(config.initial_log_std))
        )
        self.log_std_phi = nn.Parameter(
            torch.tensor(float(config.initial_log_std))
        )
        if self.condition_log_std_on_n_lasers:
            # The two outputs are learned corrections to log(sigma_kappa)
            # and log(sigma_phi). Zero initialization exactly reproduces the
            # original shared exploration scale at the start of training.
            self.size_log_std_head = nn.Linear(2, 2)
            nn.init.zeros_(self.size_log_std_head.weight)
            nn.init.zeros_(self.size_log_std_head.bias)
        self.minimum_log_std = config.minimum_log_std
        self.maximum_log_std = config.maximum_log_std

    def array_size_features(
        self,
        n_lasers: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return normalized M and inverse neighbor count for one array."""
        size_range = (
            self.maximum_training_n_lasers
            - self.minimum_training_n_lasers
        )
        normalized_size = (
            0.0
            if size_range == 0
            else 2.0
            * (n_lasers - self.minimum_training_n_lasers)
            / size_range
            - 1.0
        )
        return torch.tensor(
            [normalized_size, 1.0 / (n_lasers - 1)],
            device=device,
            dtype=dtype,
        )

    def link_counts(self, n_lasers: int) -> tuple[int, int, int]:
        directed = n_lasers * (n_lasers - 1)
        magnitude = directed // 2 if self.force_symmetric_kappa else directed
        phase = directed // 2 if self.force_symmetric_phi_p else directed
        return directed, magnitude, phase

    def action_size_for(self, n_lasers: int) -> int:
        _, magnitude, phase = self.link_counts(n_lasers)
        return magnitude + phase

    def _action_parameters(
        self, node_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if node_features.ndim != 3 or node_features.shape[2] != 3:
            raise ValueError("node_features must have shape (batch, M, 3)")
        batch_size, n_lasers, _ = node_features.shape
        if n_lasers < 2:
            raise ValueError("node_features must contain at least two lasers")

        node_embeddings = self.node_encoder(node_features)
        sequence_embeddings, _ = self.bigru(node_embeddings)
        receiver_numpy, source_numpy = directed_link_indices(n_lasers)
        link_receivers = torch.as_tensor(
            receiver_numpy, dtype=torch.long, device=node_features.device
        )
        link_sources = torch.as_tensor(
            source_numpy, dtype=torch.long, device=node_features.device
        )
        n_directed_links, _, _ = self.link_counts(n_lasers)
        source_embeddings = sequence_embeddings.index_select(1, link_sources)
        receiver_embeddings = sequence_embeddings.index_select(
            1, link_receivers
        )

        cosine = node_features[:, :, 0]
        sine = node_features[:, :, 1]
        normalized_detuning = node_features[:, :, 2]
        local_cosine = (
            cosine[:, link_sources] * cosine[:, link_receivers]
            + sine[:, link_sources] * sine[:, link_receivers]
        )
        local_sine = (
            sine[:, link_sources] * cosine[:, link_receivers]
            - cosine[:, link_sources] * sine[:, link_receivers]
        )
        local_detuning_difference = 0.5 * (
            normalized_detuning[:, link_sources]
            - normalized_detuning[:, link_receivers]
        )
        edge_features = torch.stack(
            (
                local_cosine,
                local_sine,
                local_detuning_difference,
                torch.abs(local_detuning_difference),
            ),
            dim=2,
        )
        size_feature_vector = None
        if (
            self.condition_on_n_lasers
            or self.modulate_edge_decoder_by_n_lasers
        ):
            size_feature_vector = self.array_size_features(
                n_lasers,
                device=node_features.device,
                dtype=node_features.dtype,
            )
        if self.condition_on_n_lasers:
            size_features = size_feature_vector.view(1, 1, 2).expand(
                batch_size, n_directed_links, 2
            )
            edge_features = torch.cat((edge_features, size_features), dim=2)
        edge_inputs = torch.cat(
            (source_embeddings, receiver_embeddings, edge_features), dim=2
        )
        flat_edge_inputs = edge_inputs.reshape(
            batch_size * n_directed_links, -1
        )
        if self.modulate_edge_decoder_by_n_lasers:
            modulation = self.edge_size_modulation(size_feature_vector)
            modulation_width = sum(self.edge_hidden_sizes)
            scale_corrections = modulation[:modulation_width]
            shifts = modulation[modulation_width:]
            edge_latent = flat_edge_inputs
            channel_start = 0
            for hidden_index, hidden_width in enumerate(
                self.edge_hidden_sizes
            ):
                linear = self.edge_decoder[2 * hidden_index]
                activation = self.edge_decoder[2 * hidden_index + 1]
                edge_latent = activation(linear(edge_latent))
                channel_stop = channel_start + hidden_width
                edge_latent = (
                    1.0
                    + scale_corrections[channel_start:channel_stop]
                ) * edge_latent + shifts[channel_start:channel_stop]
                channel_start = channel_stop
        else:
            edge_latent = self.edge_decoder[:-1](flat_edge_inputs)
        edge_outputs = self.edge_decoder[-1](edge_latent).reshape(
            batch_size, n_directed_links, -1
        )
        directed_magnitude_means = 6.0 * torch.tanh(
            edge_outputs[:, :, 0] / 6.0
        )
        directed_phase_means = edge_outputs[:, :, 1]
        upper_triangle = torch.as_tensor(
            np.flatnonzero(receiver_numpy < source_numpy),
            dtype=torch.long,
            device=node_features.device,
        )
        magnitude_means = (
            directed_magnitude_means.index_select(1, upper_triangle)
            if self.force_symmetric_kappa
            else directed_magnitude_means
        )
        phase_means = (
            directed_phase_means.index_select(1, upper_triangle)
            if self.force_symmetric_phi_p
            else directed_phase_means
        )
        means = torch.cat((magnitude_means, phase_means), dim=1)
        gate_logits = None
        if self.enable_sparse_gates:
            directed_gate_logits = self.edge_gate_head(edge_latent).reshape(
                batch_size, n_directed_links
            )
            gate_logits = (
                directed_gate_logits.index_select(1, upper_triangle)
                if self.force_symmetric_kappa
                else directed_gate_logits
            )
        if not self.edge_conditioned_log_std:
            return means, None, gate_logits

        directed_kappa_log_std = edge_outputs[:, :, 2]
        directed_phi_log_std = edge_outputs[:, :, 3]
        kappa_log_std = (
            directed_kappa_log_std.index_select(1, upper_triangle)
            if self.force_symmetric_kappa
            else directed_kappa_log_std
        )
        phi_log_std = (
            directed_phi_log_std.index_select(1, upper_triangle)
            if self.force_symmetric_phi_p
            else directed_phi_log_std
        )
        return (
            means,
            torch.cat((kappa_log_std, phi_log_std), dim=1),
            gate_logits,
        )

    def action_mean(self, node_features: torch.Tensor) -> torch.Tensor:
        return self._action_parameters(node_features)[0]

    def gate_distribution(
        self, node_features: torch.Tensor
    ) -> torch.distributions.Bernoulli:
        """Return one learned on/off distribution per magnitude link."""
        gate_logits = self._action_parameters(node_features)[2]
        if gate_logits is None:
            raise RuntimeError("this policy does not have sparse gates enabled")
        return torch.distributions.Bernoulli(logits=gate_logits)

    def deterministic_gates(self, node_features: torch.Tensor) -> torch.Tensor:
        """Threshold gate probabilities for deterministic evaluation."""
        probabilities = self.gate_distribution(node_features).probs
        return (probabilities >= self.gate_probability_threshold).to(
            probabilities.dtype
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.action_mean(node_features)

    def distribution(
        self, node_features: torch.Tensor
    ) -> torch.distributions.Normal:
        mean, edge_log_std, _ = self._action_parameters(node_features)
        if edge_log_std is not None:
            if self.smooth_edge_log_std_bounds:
                log_std = smoothly_bound_log_std(
                    edge_log_std,
                    self.minimum_log_std,
                    self.maximum_log_std,
                )
            else:
                log_std = torch.clamp(
                    edge_log_std,
                    self.minimum_log_std,
                    self.maximum_log_std,
                )
            return torch.distributions.Normal(mean, torch.exp(log_std))

        n_lasers = node_features.shape[1]
        _, n_magnitude_links, n_phase_links = self.link_counts(n_lasers)
        size_log_std_adjustment = torch.zeros(
            2, device=mean.device, dtype=mean.dtype
        )
        if self.condition_log_std_on_n_lasers:
            size_features = self.array_size_features(
                n_lasers,
                device=mean.device,
                dtype=mean.dtype,
            )
            size_log_std_adjustment = self.size_log_std_head(size_features)
        kappa_log_std = torch.clamp(
            self.log_std_kappa + size_log_std_adjustment[0],
            self.minimum_log_std,
            self.maximum_log_std,
        )
        phi_log_std = torch.clamp(
            self.log_std_phi + size_log_std_adjustment[1],
            self.minimum_log_std,
            self.maximum_log_std,
        )
        log_std = torch.cat(
            (
                kappa_log_std.expand(n_magnitude_links),
                phi_log_std.expand(n_phase_links),
            )
        )
        return torch.distributions.Normal(mean, torch.exp(log_std))

class GraphMessagePassingLayer(nn.Module):
    """One degree-normalized residual update on a complete directed graph."""

    def __init__(
        self,
        node_width: int,
        hidden_width: int,
        edge_feature_width: int = 4,
    ):
        super().__init__()
        # A message depends on both endpoints and the four physical pair
        # features. The same MLP is applied to every directed edge.
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_width + edge_feature_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, node_width),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, node_width),
        )
        self.normalization = nn.LayerNorm(node_width)
        for module in (self.message_mlp, self.update_mlp):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_features: torch.Tensor,
        link_receivers: torch.Tensor,
        link_sources: torch.Tensor,
    ) -> torch.Tensor:
        """Update ``(batch, M, width)`` node states equivariantly."""
        batch_size, n_lasers, node_width = node_states.shape
        source_states = node_states.index_select(1, link_sources)
        receiver_states = node_states.index_select(1, link_receivers)
        messages = self.message_mlp(
            torch.cat(
                (source_states, receiver_states, edge_features), dim=2
            )
        )
        receiver_indices = link_receivers.view(1, -1, 1).expand(
            batch_size, -1, node_width
        )
        aggregated_messages = torch.zeros_like(node_states)
        aggregated_messages.scatter_add_(
            1, receiver_indices, messages
        )
        # Every receiver has M-1 incoming neighbors. Taking their mean keeps
        # the message scale comparable for both trained and unseen M.
        aggregated_messages = aggregated_messages / float(n_lasers - 1)
        update = self.update_mlp(
            torch.cat((node_states, aggregated_messages), dim=2)
        )
        return self.normalization(node_states + update)


class GNNPolicyNetwork(nn.Module):
    """Permutation-equivariant, variable-size message-passing policy."""

    def __init__(self, config: DesignerConfig):
        super().__init__()
        if config.modulate_edge_decoder_by_n_lasers:
            raise ValueError(
                "the GNN comparison requires M modulation to be disabled"
            )
        self.force_symmetric_kappa = config.force_symmetric_kappa
        self.force_symmetric_phi_p = config.force_symmetric_phi_p
        self.node_embedding_dim = config.node_embedding_dim
        self.minimum_training_n_lasers = min(config.training_n_lasers)
        self.maximum_training_n_lasers = max(config.training_n_lasers)
        self.condition_on_n_lasers = config.condition_on_n_lasers
        self.condition_on_bounded_degree = (
            config.condition_on_bounded_degree
        )
        self.condition_log_std_on_n_lasers = (
            config.condition_log_std_on_n_lasers
        )
        self.modulate_edge_decoder_by_n_lasers = False
        self.edge_conditioned_log_std = config.edge_conditioned_log_std
        self.smooth_edge_log_std_bounds = (
            config.smooth_edge_log_std_bounds
        )
        self.enable_sparse_gates = config.enable_sparse_gates
        self.enable_learned_backbone_density = (
            config.enable_learned_backbone_density
        )
        self.enable_budget_error_selector = config.enable_budget_error_selector
        self.monotonic_budget_selector = config.monotonic_budget_selector
        self.condition_on_edge_budget = config.condition_on_edge_budget
        self.gate_probability_threshold = config.gate_probability_threshold

        node_layers: list[nn.Module] = []
        input_width = 4 if self.condition_on_edge_budget else 3
        for output_width in config.node_hidden_sizes:
            linear = nn.Linear(input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            node_layers.extend((linear, nn.SiLU()))
            input_width = output_width
        node_output = nn.Linear(input_width, self.node_embedding_dim)
        nn.init.orthogonal_(node_output.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(node_output.bias)
        node_layers.extend((node_output, nn.SiLU()))
        self.node_encoder = nn.Sequential(*node_layers)

        message_edge_feature_width = (
            6 if self.condition_on_bounded_degree else 4
        )
        self.message_layers = nn.ModuleList(
            GraphMessagePassingLayer(
                self.node_embedding_dim,
                config.gnn_message_hidden_size,
                edge_feature_width=message_edge_feature_width,
            )
            for _ in range(config.gnn_message_passing_steps)
        )

        edge_input_width = 2 * self.node_embedding_dim + 4
        if self.condition_on_bounded_degree:
            edge_input_width += 2
        if self.condition_on_n_lasers:
            edge_input_width += 2
        edge_layers: list[nn.Module] = []
        for output_width in config.edge_hidden_sizes:
            linear = nn.Linear(edge_input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            edge_layers.extend((linear, nn.SiLU()))
            edge_input_width = output_width
        edge_output_width = 4 if self.edge_conditioned_log_std else 2
        edge_output = nn.Linear(edge_input_width, edge_output_width)
        nn.init.orthogonal_(edge_output.weight, gain=0.01)
        nn.init.zeros_(edge_output.bias)
        initial_fraction = config.initial_kappa_fraction
        initial_kappa_logit = np.log(
            initial_fraction / (1.0 - initial_fraction)
        )
        with torch.no_grad():
            edge_output.bias[:2].copy_(
                torch.tensor(
                    [initial_kappa_logit, 0.0],
                    dtype=edge_output.bias.dtype,
                )
            )
            if self.edge_conditioned_log_std:
                edge_output.weight[2:].zero_()
                initial_raw_log_std = float(config.initial_log_std)
                if self.smooth_edge_log_std_bounds:
                    initial_raw_log_std = raw_log_std_for_smooth_bound(
                        config.initial_log_std,
                        config.minimum_log_std,
                        config.maximum_log_std,
                    )
                edge_output.bias[2:].fill_(initial_raw_log_std)
        edge_layers.append(edge_output)
        self.edge_decoder = nn.Sequential(*edge_layers)
        if self.enable_sparse_gates:
            # This head reads the existing edge latent state, so adding it to
            # a dense checkpoint leaves every GNN encoder/decoder parameter
            # unchanged.  A positive bias initially keeps all links open.
            self.edge_gate_head = nn.Linear(edge_input_width, 1)
            nn.init.zeros_(self.edge_gate_head.weight)
            nn.init.constant_(
                self.edge_gate_head.bias, float(config.initial_gate_logit)
            )
        if self.enable_learned_backbone_density:
            self.backbone_density_head = nn.Sequential(
                nn.Linear(5, config.node_embedding_dim), nn.SiLU(),
                nn.Linear(config.node_embedding_dim, 1),
            )
            nn.init.orthogonal_(self.backbone_density_head[0].weight, gain=np.sqrt(2.0))
            nn.init.zeros_(self.backbone_density_head[0].bias)
            nn.init.zeros_(self.backbone_density_head[2].weight)
            initial_rho = min(config.initial_backbone_density, 1.0 - 1.0e-4)
            nn.init.constant_(
                self.backbone_density_head[2].bias,
                float(np.log(initial_rho / (1.0 - initial_rho))),
            )
            self.backbone_density_log_std = nn.Parameter(
                torch.tensor(float(config.backbone_density_log_std))
            )
        if self.enable_budget_error_selector:
            context_width = 2 * config.node_embedding_dim + 2
            self.budget_error_context_encoder = nn.Sequential(
                nn.Linear(3, config.node_embedding_dim),
                nn.SiLU(),
                nn.Linear(config.node_embedding_dim, config.node_embedding_dim),
                nn.SiLU(),
            )
            self.budget_error_head = nn.Sequential(
                nn.Linear(context_width + 1, config.node_embedding_dim),
                nn.SiLU(),
                nn.Linear(config.node_embedding_dim, 1),
            )
            selector_input_width = (
                context_width
                if self.monotonic_budget_selector
                else context_width + 1
            )
            selector_output_width = (
                4 if self.monotonic_budget_selector else 1
            )
            self.budget_selector_head = nn.Sequential(
                nn.Linear(selector_input_width, config.node_embedding_dim),
                nn.SiLU(),
                nn.Linear(config.node_embedding_dim, selector_output_width),
            )
            for module in (
                self.budget_error_context_encoder,
                self.budget_error_head,
                self.budget_selector_head,
            ):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                        nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.budget_selector_head[-1].weight)
            if self.monotonic_budget_selector:
                # An unbiased q=0.5 starting point avoids carrying the old
                # near-dense selector collapse into a fresh run. The three
                # negative raw coefficients become positive through softplus
                # and guarantee dq/d(epsilon) <= 0.
                with torch.no_grad():
                    self.budget_selector_head[-1].bias[0].zero_()
                    self.budget_selector_head[-1].bias[1:].fill_(-2.0)
            else:
                # Preserve compatibility with the first selector experiment.
                nn.init.constant_(
                    self.budget_selector_head[-1].bias,
                    float(np.log(0.95 / 0.05)),
                )

        self.log_std_kappa = nn.Parameter(
            torch.tensor(float(config.initial_log_std))
        )
        self.log_std_phi = nn.Parameter(
            torch.tensor(float(config.initial_log_std))
        )
        if self.condition_log_std_on_n_lasers:
            self.size_log_std_head = nn.Linear(2, 2)
            nn.init.zeros_(self.size_log_std_head.weight)
            nn.init.zeros_(self.size_log_std_head.bias)
        self.minimum_log_std = config.minimum_log_std
        self.maximum_log_std = config.maximum_log_std

    def array_size_features(
        self,
        n_lasers: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        size_range = (
            self.maximum_training_n_lasers
            - self.minimum_training_n_lasers
        )
        normalized_size = (
            0.0
            if size_range == 0
            else 2.0
            * (n_lasers - self.minimum_training_n_lasers)
            / size_range
            - 1.0
        )
        return torch.tensor(
            [normalized_size, 1.0 / (n_lasers - 1)],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def bounded_degree_features(
        n_lasers: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return smooth degree features that remain bounded for every M."""
        inverse_degree = 1.0 / float(n_lasers - 1)
        return torch.tensor(
            [inverse_degree, inverse_degree**2],
            device=device,
            dtype=dtype,
        )

    def link_counts(self, n_lasers: int) -> tuple[int, int, int]:
        directed = n_lasers * (n_lasers - 1)
        magnitude = directed // 2 if self.force_symmetric_kappa else directed
        phase = directed // 2 if self.force_symmetric_phi_p else directed
        return directed, magnitude, phase

    def action_size_for(self, n_lasers: int) -> int:
        _, magnitude, phase = self.link_counts(n_lasers)
        return magnitude + phase

    def _action_parameters(
        self, node_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        expected_feature_count = 4 if self.condition_on_edge_budget else 3
        if (
            node_features.ndim != 3
            or node_features.shape[2] != expected_feature_count
        ):
            raise ValueError(
                "node_features must have shape "
                f"(batch, M, {expected_feature_count})"
            )
        batch_size, n_lasers, _ = node_features.shape
        if n_lasers < 2:
            raise ValueError("node_features must contain at least two lasers")

        receiver_numpy, source_numpy = directed_link_indices(n_lasers)
        link_receivers = torch.as_tensor(
            receiver_numpy, dtype=torch.long, device=node_features.device
        )
        link_sources = torch.as_tensor(
            source_numpy, dtype=torch.long, device=node_features.device
        )
        n_directed_links, _, _ = self.link_counts(n_lasers)

        cosine = node_features[:, :, 0]
        sine = node_features[:, :, 1]
        normalized_detuning = node_features[:, :, 2]
        local_cosine = (
            cosine[:, link_sources] * cosine[:, link_receivers]
            + sine[:, link_sources] * sine[:, link_receivers]
        )
        local_sine = (
            sine[:, link_sources] * cosine[:, link_receivers]
            - cosine[:, link_sources] * sine[:, link_receivers]
        )
        local_detuning_difference = 0.5 * (
            normalized_detuning[:, link_sources]
            - normalized_detuning[:, link_receivers]
        )
        edge_features = torch.stack(
            (
                local_cosine,
                local_sine,
                local_detuning_difference,
                torch.abs(local_detuning_difference),
            ),
            dim=2,
        )
        if self.condition_on_bounded_degree:
            degree_features = self.bounded_degree_features(
                n_lasers,
                device=node_features.device,
                dtype=node_features.dtype,
            ).view(1, 1, 2).expand(
                batch_size, n_directed_links, 2
            )
            edge_features = torch.cat(
                (edge_features, degree_features), dim=2
            )

        node_states = self.node_encoder(node_features)
        for message_layer in self.message_layers:
            node_states = message_layer(
                node_states,
                edge_features,
                link_receivers,
                link_sources,
            )

        source_states = node_states.index_select(1, link_sources)
        receiver_states = node_states.index_select(1, link_receivers)
        decoder_features = edge_features
        if self.condition_on_n_lasers:
            size_features = self.array_size_features(
                n_lasers,
                device=node_features.device,
                dtype=node_features.dtype,
            ).view(1, 1, 2).expand(
                batch_size, n_directed_links, 2
            )
            decoder_features = torch.cat(
                (decoder_features, size_features), dim=2
            )
        edge_inputs = torch.cat(
            (source_states, receiver_states, decoder_features), dim=2
        )
        flat_edge_inputs = edge_inputs.reshape(
            batch_size * n_directed_links, -1
        )
        edge_latent = self.edge_decoder[:-1](flat_edge_inputs)
        edge_outputs = self.edge_decoder[-1](edge_latent).reshape(
            batch_size, n_directed_links, -1
        )

        directed_magnitude_means = 6.0 * torch.tanh(
            edge_outputs[:, :, 0] / 6.0
        )
        directed_phase_means = edge_outputs[:, :, 1]
        upper_triangle = torch.as_tensor(
            np.flatnonzero(receiver_numpy < source_numpy),
            dtype=torch.long,
            device=node_features.device,
        )
        magnitude_means = (
            directed_magnitude_means.index_select(1, upper_triangle)
            if self.force_symmetric_kappa
            else directed_magnitude_means
        )
        phase_means = (
            directed_phase_means.index_select(1, upper_triangle)
            if self.force_symmetric_phi_p
            else directed_phase_means
        )
        means = torch.cat((magnitude_means, phase_means), dim=1)
        gate_logits = None
        if self.enable_sparse_gates:
            directed_gate_logits = self.edge_gate_head(edge_latent).reshape(
                batch_size, n_directed_links
            )
            gate_logits = (
                directed_gate_logits.index_select(1, upper_triangle)
                if self.force_symmetric_kappa
                else directed_gate_logits
            )
        if not self.edge_conditioned_log_std:
            return means, None, gate_logits

        directed_kappa_log_std = edge_outputs[:, :, 2]
        directed_phi_log_std = edge_outputs[:, :, 3]
        kappa_log_std = (
            directed_kappa_log_std.index_select(1, upper_triangle)
            if self.force_symmetric_kappa
            else directed_kappa_log_std
        )
        phi_log_std = (
            directed_phi_log_std.index_select(1, upper_triangle)
            if self.force_symmetric_phi_p
            else directed_phi_log_std
        )
        return (
            means,
            torch.cat((kappa_log_std, phi_log_std), dim=1),
            gate_logits,
        )

    def action_mean(self, node_features: torch.Tensor) -> torch.Tensor:
        return self._action_parameters(node_features)[0]

    def gate_distribution(
        self, node_features: torch.Tensor
    ) -> torch.distributions.Bernoulli:
        """Return one learned on/off distribution per magnitude link."""
        gate_logits = self._action_parameters(node_features)[2]
        if gate_logits is None:
            raise RuntimeError("this policy does not have sparse gates enabled")
        return torch.distributions.Bernoulli(logits=gate_logits)

    def deterministic_gates(self, node_features: torch.Tensor) -> torch.Tensor:
        """Threshold gate probabilities for deterministic evaluation."""
        probabilities = self.gate_distribution(node_features).probs
        return (probabilities >= self.gate_probability_threshold).to(
            probabilities.dtype
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.action_mean(node_features)

    def distribution(
        self, node_features: torch.Tensor
    ) -> torch.distributions.Normal:
        mean, edge_log_std, _ = self._action_parameters(node_features)
        if edge_log_std is not None:
            if self.smooth_edge_log_std_bounds:
                log_std = smoothly_bound_log_std(
                    edge_log_std,
                    self.minimum_log_std,
                    self.maximum_log_std,
                )
            else:
                log_std = torch.clamp(
                    edge_log_std,
                    self.minimum_log_std,
                    self.maximum_log_std,
                )
            return torch.distributions.Normal(mean, torch.exp(log_std))

        n_lasers = node_features.shape[1]
        _, n_magnitude_links, n_phase_links = self.link_counts(n_lasers)
        size_log_std_adjustment = torch.zeros(
            2, device=mean.device, dtype=mean.dtype
        )
        if self.condition_log_std_on_n_lasers:
            size_features = self.array_size_features(
                n_lasers,
                device=mean.device,
                dtype=mean.dtype,
            )
            size_log_std_adjustment = self.size_log_std_head(size_features)
        kappa_log_std = torch.clamp(
            self.log_std_kappa + size_log_std_adjustment[0],
            self.minimum_log_std,
            self.maximum_log_std,
        )
        phi_log_std = torch.clamp(
            self.log_std_phi + size_log_std_adjustment[1],
            self.minimum_log_std,
            self.maximum_log_std,
        )
        log_std = torch.cat(
            (
                kappa_log_std.expand(n_magnitude_links),
                phi_log_std.expand(n_phase_links),
            )
        )
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def backbone_density_distribution(
        self, node_features: torch.Tensor
    ) -> torch.distributions.Normal:
        """Distribution over a logit mapped to retained density by sigmoid."""
        if not self.enable_learned_backbone_density:
            raise RuntimeError("learned backbone density is not enabled")
        n_lasers = node_features.shape[1]
        pooled = node_features[:, :, :3].mean(dim=1)
        degree = self.bounded_degree_features(
            n_lasers, device=pooled.device, dtype=pooled.dtype
        ).view(1, 2).expand(pooled.shape[0], 2)
        mean = self.backbone_density_head(torch.cat((pooled, degree), dim=1))
        log_std = torch.clamp(
            self.backbone_density_log_std,
            self.minimum_log_std,
            self.maximum_log_std,
        )
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def budget_error_context(self, node_features: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant context for budget prediction heads."""
        if not self.enable_budget_error_selector:
            raise RuntimeError("budget error selector is not enabled")
        encoded = self.budget_error_context_encoder(node_features[:, :, :3])
        pooled = torch.cat((encoded.mean(dim=1), encoded.amax(dim=1)), dim=1)
        degree = self.bounded_degree_features(
            node_features.shape[1], device=pooled.device, dtype=pooled.dtype
        ).view(1, 2).expand(pooled.shape[0], 2)
        return torch.cat((pooled, degree), dim=1)

    def predict_normalized_phase_error(
        self, node_features: torch.Tensor, retained_density: torch.Tensor
    ) -> torch.Tensor:
        """Predict circular RMS phase error divided by pi."""
        context = self.budget_error_context(node_features)
        density = retained_density.to(context).reshape(-1, 1)
        if density.shape[0] != context.shape[0]:
            raise ValueError("retained_density must have one value per target")
        return torch.sigmoid(
            self.budget_error_head(torch.cat((context, density), dim=1))
        ).squeeze(1)

    def predict_normalized_budget(
        self, node_features: torch.Tensor, allowable_error_fraction: torch.Tensor
    ) -> torch.Tensor:
        """Predict q=0 spanning-tree through q=1 all-to-all budget."""
        context = self.budget_error_context(node_features)
        tolerance = allowable_error_fraction.to(context).reshape(-1, 1)
        if tolerance.shape[0] != context.shape[0]:
            raise ValueError(
                "allowable_error_fraction must have one value per target"
            )
        if self.monotonic_budget_selector:
            parameters = self.budget_selector_head(context)
            intercept = parameters[:, 0]
            positive_coefficients = torch.nn.functional.softplus(
                parameters[:, 1:]
            )
            powers = torch.cat(
                (tolerance, tolerance.square(), tolerance.pow(3)), dim=1
            )
            return torch.sigmoid(
                intercept - torch.sum(positive_coefficients * powers, dim=1)
            )
        return torch.sigmoid(
            self.budget_selector_head(torch.cat((context, tolerance), dim=1))
        ).squeeze(1)


def make_policy(config: DesignerConfig) -> nn.Module:
    """Construct the architecture selected by ``config.encoder_architecture``."""
    if config.encoder_architecture == "bigru":
        return BiGRUPolicyNetwork(config)
    if config.encoder_architecture == "gnn":
        return GNNPolicyNetwork(config)
    return PolicyNetwork(config)


def encode_for_policy(
    targets_rad: np.ndarray,
    policy: PolicyNetwork,
    config: DesignerConfig,
    detuning_distribution_ghz: np.ndarray | None = None,
    active_link_counts: np.ndarray | int | None = None,
) -> torch.Tensor:
    """Prepare per-laser phase/detuning features on the policy's device.

    Detunings are supplied in GHz relative to the mean-frequency reference
    (0 GHz).  A zero vector is used only for backwards-compatible callers
    that do not provide detunings explicitly.
    """
    if np.asarray(targets_rad).shape[-1] != config.n_lasers:
        raise ValueError(
            f"targets must contain {config.n_lasers} phases"
        )
    policy_targets = (
        canonicalize_global_phase_conjugate(targets_rad)
        if config.allow_global_phase_conjugate
        else make_relative_to_laser_1(targets_rad)
    )
    relative_phases = make_relative_to_laser_1(policy_targets)
    if relative_phases.ndim == 1:
        relative_phases = relative_phases[None, :]
    if detuning_distribution_ghz is None:
        detuning_features = np.zeros(
            (relative_phases.shape[0], config.n_lasers), dtype=np.float32
        )
    else:
        detuning_features = encode_detuning_distributions(
            detuning_distribution_ghz, config
        )
        if detuning_features.ndim == 1:
            detuning_features = detuning_features[None, :]
        if detuning_features.shape[0] != relative_phases.shape[0]:
            raise ValueError(
                "targets and detuning distributions must have the same "
                "batch size"
            )
    feature_columns = [
        np.cos(relative_phases),
        np.sin(relative_phases),
        detuning_features,
    ]
    if getattr(policy, "condition_on_edge_budget", False):
        maximum_links = config.n_lasers * (config.n_lasers - 1)
        if active_link_counts is None:
            active_counts = np.full(
                relative_phases.shape[0], maximum_links, dtype=float
            )
        else:
            active_counts = np.asarray(active_link_counts, dtype=float)
            if active_counts.ndim == 0:
                active_counts = np.full(
                    relative_phases.shape[0], float(active_counts)
                )
            if active_counts.shape != (relative_phases.shape[0],):
                raise ValueError(
                    "active_link_counts must be scalar or have one value "
                    "per target"
                )
            if np.any(active_counts < config.n_lasers - 1) or np.any(
                active_counts > maximum_links
            ):
                raise ValueError(
                    "active_link_counts must lie between M-1 and M*(M-1)"
                )
        normalized_budget = (
            active_counts / float(maximum_links)
        )[:, None]
        feature_columns.append(
            np.broadcast_to(normalized_budget, relative_phases.shape)
        )
    # One feature row per laser: target angle on the unit circle, normalized
    # detuning, and (for the connected-sparse policy) active-link budget.
    node_features = np.stack(feature_columns, axis=2).astype(np.float32)
    return torch.as_tensor(
        node_features,
        dtype=torch.float32,
        device=next(policy.parameters()).device,
    )


def sample_coupling_designs(
    policy: PolicyNetwork,
    encoded_targets: torch.Tensor,
    candidates_per_target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample continuous actions and their log probabilities.

    Returns:
        actions: ``(targets, candidates, policy.action_size)``
        log_probabilities: ``(targets, candidates)``
    """
    distribution = policy.distribution(encoded_targets)
    # PyTorch samples as (candidates, targets, actions).
    sampled = distribution.sample((candidates_per_target,))
    # The simulator is not differentiable. REINFORCE instead differentiates
    # log pi(action | target), treating the sampled action as observed data.
    actions = sampled.detach().permute(1, 0, 2)
    log_probabilities = distribution.log_prob(sampled.detach())
    # Average over action dimensions so variable-size arrays have comparable
    # policy-gradient scale while every sampled action still contributes.
    log_probabilities = log_probabilities.mean(dim=2).permute(1, 0)
    return actions, log_probabilities


def sample_coupling_designs_with_backbone_density(
    policy: GNNPolicyNetwork,
    encoded_targets: torch.Tensor,
    candidates_per_target: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample continuous matrices plus one retained-link density ``rho``."""
    actions, action_log_probability = sample_coupling_designs(
        policy, encoded_targets, candidates_per_target
    )
    distribution = policy.backbone_density_distribution(encoded_targets)
    sampled_logits = distribution.sample((candidates_per_target,))
    rho_log_probability = distribution.log_prob(sampled_logits.detach())
    rho = torch.sigmoid(sampled_logits.detach()).permute(1, 0, 2).squeeze(2)
    # The continuous score is already averaged over its action dimensions.
    # Add the scalar rho score so both parts of the joint action can learn.
    joint_log_probability = (
        action_log_probability
        + rho_log_probability.squeeze(2).permute(1, 0)
    )
    return actions, rho, joint_log_probability


def relative_coupling_backbone_mask(
    kappa_per_ns: np.ndarray,
    retained_density: float,
) -> np.ndarray:
    """Select a deterministic weakly connected directed backbone.

    Links are ranked by receiver-normalized coupling.  A maximum spanning
    tree is selected first, followed by each node's strongest local links and
    then the globally strongest remaining directions.
    """
    kappa = np.asarray(kappa_per_ns, dtype=float)
    if kappa.ndim != 2 or kappa.shape[0] != kappa.shape[1]:
        raise ValueError("kappa_per_ns must be a square matrix")
    m = kappa.shape[0]
    maximum_links = m * (m - 1)
    link_limit = max(m - 1, int(np.ceil(float(retained_density) * maximum_links)))
    link_limit = min(link_limit, maximum_links)
    totals = kappa.sum(axis=1, keepdims=True)
    relative = np.divide(
        kappa, totals, out=np.zeros_like(kappa), where=totals > 1.0e-15
    )
    np.fill_diagonal(relative, 0.0)

    # Kruskal maximum spanning tree over reciprocal pair strengths.
    parent = list(range(m))
    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    selected: set[tuple[int, int]] = set()
    pairs = sorted(
        (
            (max(relative[b, a], relative[a, b]), a, b)
            for a in range(m) for b in range(a + 1, m)
        ),
        reverse=True,
    )
    for strength, a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a == root_b or strength <= 0.0:
            continue
        parent[root_b] = root_a
        selected.add((a, b) if relative[b, a] >= relative[a, b] else (b, a))
        if len(selected) == m - 1:
            break

    local: dict[tuple[int, int], float] = {}
    for receiver in range(m):
        source = int(np.argmax(relative[receiver]))
        if source != receiver and relative[receiver, source] > 0.0:
            local[(source, receiver)] = float(relative[receiver, source])
    for source in range(m):
        receiver = int(np.argmax(relative[:, source]))
        if receiver != source and relative[receiver, source] > 0.0:
            local[(source, receiver)] = float(relative[receiver, source])
    ranked = list(local.items())
    ranked.extend(
        ((source, receiver), float(relative[receiver, source]))
        for source in range(m) for receiver in range(m)
        if source != receiver and (source, receiver) not in local
    )
    for direction, _ in sorted(ranked, key=lambda item: item[1], reverse=True):
        if len(selected) >= link_limit:
            break
        selected.add(direction)
    mask = np.zeros((m, m), dtype=bool)
    for source, receiver in selected:
        mask[receiver, source] = True
    return mask


def prune_coupling_batch_to_backbone(
    kappa_per_ns: np.ndarray,
    phi_p_rad: np.ndarray,
    retained_densities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prune a flattened candidate batch and return realized densities."""
    pruned_kappa = np.zeros_like(kappa_per_ns)
    pruned_phi = np.zeros_like(phi_p_rad)
    realized = np.empty(len(kappa_per_ns), dtype=float)
    for index, rho in enumerate(np.asarray(retained_densities).reshape(-1)):
        mask = relative_coupling_backbone_mask(kappa_per_ns[index], float(rho))
        pruned_kappa[index] = np.where(mask, kappa_per_ns[index], 0.0)
        pruned_phi[index] = np.where(mask, phi_p_rad[index], 0.0)
        m = mask.shape[0]
        realized[index] = np.count_nonzero(mask) / (m * (m - 1))
    return pruned_kappa, pruned_phi, realized


def prune_coupling_batch_to_link_budget(
    kappa_per_ns: np.ndarray,
    phi_p_rad: np.ndarray,
    active_link_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one exact connected directed-link budget per candidate."""
    kappa = np.asarray(kappa_per_ns, dtype=float)
    phi = np.asarray(phi_p_rad, dtype=float)
    counts = np.asarray(active_link_counts, dtype=np.int64).reshape(-1)
    if len(counts) != len(kappa):
        raise ValueError("active_link_counts must have one value per candidate")
    m = kappa.shape[1]
    maximum_links = m * (m - 1)
    if np.any(counts < m - 1) or np.any(counts > maximum_links):
        raise ValueError("active link budgets must lie between M-1 and M*(M-1)")
    # Subtract a tiny amount so ceil(rho*maximum_links) reproduces the exact
    # integer budget despite floating-point roundoff.
    rho = np.maximum(counts - 1.0e-9, m - 1) / maximum_links
    return prune_coupling_batch_to_backbone(kappa, phi, rho)


def sample_sparse_coupling_designs(
    policy: nn.Module,
    encoded_targets: torch.Tensor,
    candidates_per_target: int,
    *,
    sample_gates: bool = True,
    include_dense_reference: bool = False,
    single_edge_probe_fraction: float = 0.0,
    exploratory_mask_fraction: float = 0.0,
    exploration_removal_fractions: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.30,
    ),
    exploration_strength: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample continuous actions, hard gates, and their joint log probability.

    In addition to ordinary Bernoulli samples, sparse refinement may reserve
    candidates for three explicit exploration roles: an all-open reference,
    exactly-one-edge removal probes, and low-sparsity random masks.  The
    latter use a differentiable mixture of the learned gate probabilities and
    requested removal rates, so their score-function gradient remains valid.
    During dense recovery, gates are held open and omitted from the score.
    """
    if not policy.enable_sparse_gates:
        raise ValueError("sample_sparse_coupling_designs requires sparse gates")
    if candidates_per_target < 1:
        raise ValueError("candidates_per_target must be positive")
    for name, fraction in (
        ("single_edge_probe_fraction", single_edge_probe_fraction),
        ("exploratory_mask_fraction", exploratory_mask_fraction),
    ):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if single_edge_probe_fraction + exploratory_mask_fraction > 1.0:
        raise ValueError(
            "single-edge and exploratory-mask fractions must sum to at "
            "most one"
        )
    if not 0.0 <= exploration_strength <= 1.0:
        raise ValueError("exploration_strength must lie in [0, 1]")
    if not exploration_removal_fractions:
        raise ValueError("exploration_removal_fractions must not be empty")
    if any(
        not 0.0 < fraction < 1.0
        for fraction in exploration_removal_fractions
    ):
        raise ValueError(
            "every exploration removal fraction must lie strictly in (0, 1)"
        )

    continuous_distribution = policy.distribution(encoded_targets)
    sampled_actions = continuous_distribution.sample((candidates_per_target,))
    continuous_log_probability = continuous_distribution.log_prob(
        sampled_actions.detach()
    )
    action_count = sampled_actions.shape[2]

    gate_distribution = policy.gate_distribution(encoded_targets)
    gate_count = gate_distribution.probs.shape[1]
    if sample_gates:
        learned_gate_samples = gate_distribution.sample(
            (candidates_per_target,)
        )
        gate_log_probability = gate_distribution.log_prob(
            learned_gate_samples.detach()
        ).sum(dim=2)
        # Exploration roles overwrite selected masks below. Keep the samples
        # saved by Bernoulli.log_prob untouched for autograd.
        sampled_gates = learned_gate_samples.clone()
        gate_decision_count = torch.full(
            (candidates_per_target, 1),
            float(gate_count),
            device=sampled_actions.device,
            dtype=sampled_actions.dtype,
        )

        candidate_start = 0
        if include_dense_reference:
            # Use the deterministic policy mean for the reference rather than
            # a noisy Gaussian action.  This candidate is evaluated only as
            # the same-target dense baseline and therefore carries no policy
            # score of its own.
            sampled_actions[0] = continuous_distribution.mean.detach()
            continuous_log_probability[0].zero_()
            sampled_gates[0].fill_(1.0)
            gate_log_probability[0].zero_()
            gate_decision_count[0].zero_()
            candidate_start = 1

        available_candidates = candidates_per_target - candidate_start
        single_probe_count = min(
            int(round(candidates_per_target * single_edge_probe_fraction)),
            available_candidates,
        )
        if single_probe_count:
            # This is the learned Bernoulli distribution conditioned on
            # exactly one edge being off: P(edge=e) is proportional to
            # (1-p_e)/p_e = exp(-logit_e).
            removal_distribution = torch.distributions.Categorical(
                logits=-gate_distribution.logits
            )
            removed_edges = removal_distribution.sample(
                (single_probe_count,)
            )
            probe_slice = slice(
                candidate_start, candidate_start + single_probe_count
            )
            probe_gates = torch.ones(
                (
                    single_probe_count,
                    encoded_targets.shape[0],
                    gate_count,
                ),
                device=sampled_actions.device,
                dtype=sampled_actions.dtype,
            )
            probe_gates.scatter_(2, removed_edges.unsqueeze(2), 0.0)
            sampled_gates[probe_slice] = probe_gates
            gate_log_probability[probe_slice] = (
                removal_distribution.log_prob(removed_edges)
            )
            gate_decision_count[probe_slice] = 1.0
            candidate_start += single_probe_count

        available_candidates = candidates_per_target - candidate_start
        exploratory_count = min(
            int(round(candidates_per_target * exploratory_mask_fraction)),
            available_candidates,
        )
        if exploratory_count:
            removal_rates = torch.as_tensor(
                [
                    exploration_removal_fractions[
                        index % len(exploration_removal_fractions)
                    ]
                    for index in range(exploratory_count)
                ],
                device=sampled_actions.device,
                dtype=sampled_actions.dtype,
            ).view(exploratory_count, 1, 1)
            learned_probabilities = gate_distribution.probs.unsqueeze(0)
            exploratory_probabilities = (
                (1.0 - exploration_strength) * learned_probabilities
                + exploration_strength * (1.0 - removal_rates)
            )
            exploratory_distribution = torch.distributions.Bernoulli(
                probs=exploratory_probabilities
            )
            exploratory_gates = exploratory_distribution.sample()
            exploratory_slice = slice(
                candidate_start, candidate_start + exploratory_count
            )
            sampled_gates[exploratory_slice] = exploratory_gates
            gate_log_probability[exploratory_slice] = (
                exploratory_distribution.log_prob(
                    exploratory_gates.detach()
                ).sum(dim=2)
            )
            candidate_start += exploratory_count

        joint_log_probability = (
            continuous_log_probability.sum(dim=2) + gate_log_probability
        ) / (float(action_count) + gate_decision_count)
    else:
        sampled_gates = torch.ones(
            (
                candidates_per_target,
                encoded_targets.shape[0],
                gate_count,
            ),
            device=encoded_targets.device,
            dtype=encoded_targets.dtype,
        )
        joint_log_probability = continuous_log_probability.mean(dim=2)

    return (
        sampled_actions.detach().permute(1, 0, 2),
        sampled_gates.detach().permute(1, 0, 2),
        joint_log_probability.permute(1, 0),
    )


def minimum_active_links_at(
    iteration: int,
    n_lasers: int,
    config: DesignerConfig,
) -> int:
    """Return the sparsest budget exposed by the current curriculum."""
    maximum_links = n_lasers * (n_lasers - 1)
    minimum_links = n_lasers - 1
    if iteration < config.edge_budget_warmup_iterations:
        return maximum_links
    if config.edge_budget_curriculum_iterations == 0:
        progress = 1.0
    else:
        progress = np.clip(
            (
                iteration
                - config.edge_budget_warmup_iterations
                + 1
            )
            / config.edge_budget_curriculum_iterations,
            0.0,
            1.0,
        )
    removable_links = maximum_links - minimum_links
    return maximum_links - int(round(progress * removable_links))


def sample_active_link_counts(
    count: int,
    n_lasers: int,
    rng: np.random.Generator,
    iteration: int,
    config: DesignerConfig,
) -> np.ndarray:
    """Sample exact connected-link budgets for one training batch."""
    if count < 1:
        raise ValueError("count must be positive")
    maximum_links = n_lasers * (n_lasers - 1)
    minimum_links = minimum_active_links_at(iteration, n_lasers, config)
    return rng.integers(
        minimum_links,
        maximum_links + 1,
        size=count,
        dtype=np.int64,
    )


def sample_conditional_link_counts(
    count: int,
    n_lasers: int,
    iteration: int,
) -> np.ndarray:
    """Sweep exact connected budgets evenly without extra simulations.

    Each target still receives one budget and all of its stochastic coupling
    candidates share that budget.  A size-specific cyclic sequence visits
    every feasible integer link count; targets within one update are offset
    across that sequence.  This gives the error model examples ranging from
    a spanning tree through all-to-all coupling without changing the number
    of VCSEL simulations.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if n_lasers < 2:
        raise ValueError("n_lasers must be at least 2")
    minimum_links = n_lasers - 1
    maximum_links = n_lasers * (n_lasers - 1)
    feasible_count = maximum_links - minimum_links + 1
    # Choose a size-dependent stride coprime to the number of budgets. This
    # makes the base index traverse every exact count before repeating.
    stride = max(1, int(round(0.61803398875 * feasible_count)))
    while np.gcd(stride, feasible_count) != 1:
        stride += 1
    base_index = (iteration * stride + n_lasers) % feasible_count
    offsets = np.floor(
        np.arange(count, dtype=float) * feasible_count / count
    ).astype(np.int64)
    budget_indices = (base_index + offsets) % feasible_count
    return minimum_links + budget_indices


def sample_multibudget_candidate_counts(
    target_count: int,
    candidates_per_target: int,
    n_lasers: int,
    budget_levels_per_target: int,
    rng: np.random.Generator,
    iteration: int,
) -> np.ndarray:
    """Assign one exact connected budget to every existing candidate.

    Each target receives up to ``budget_levels_per_target`` levels including
    the spanning-tree and all-to-all endpoints. Candidate counts are balanced
    across levels, and their ordering is shuffled. The returned shape is
    ``(target_count, candidates_per_target)`` and therefore does not change
    the number of VCSEL simulations.
    """
    if target_count < 1 or candidates_per_target < 1:
        raise ValueError("target and candidate counts must be positive")
    minimum_links = n_lasers - 1
    maximum_links = n_lasers * (n_lasers - 1)
    feasible_budget_count = maximum_links - minimum_links + 1
    level_count = min(
        budget_levels_per_target,
        feasible_budget_count,
        candidates_per_target,
    )
    levels = np.unique(
        np.rint(
            np.linspace(minimum_links, maximum_links, level_count)
        ).astype(np.int64)
    )
    if len(levels) != level_count:
        raise RuntimeError("failed to construct distinct exact budget levels")
    assignments = np.empty(
        (target_count, candidates_per_target), dtype=np.int64
    )
    base_count, remainder = divmod(candidates_per_target, level_count)
    for target_index in range(target_count):
        repeats = np.full(level_count, base_count, dtype=np.int64)
        start = (iteration + target_index) % level_count
        repeats[(start + np.arange(remainder)) % level_count] += 1
        row = np.repeat(levels, repeats)
        rng.shuffle(row)
        assignments[target_index] = row
    return assignments


def deterministic_budget_candidate_mask(
    active_link_count_matrix: np.ndarray,
) -> np.ndarray:
    """Select exactly one deterministic candidate for every target/budget."""
    budgets = np.asarray(active_link_count_matrix, dtype=np.int64)
    if budgets.ndim != 2 or budgets.shape[1] < 1:
        raise ValueError("active_link_count_matrix must be a nonempty 2-D array")
    deterministic = np.zeros_like(budgets, dtype=bool)
    for target_index, row in enumerate(budgets):
        _, first_indices = np.unique(row, return_index=True)
        deterministic[target_index, first_indices] = True
    return deterministic


def nonincreasing_isotonic_fit(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted least-squares isotonic fit constrained to be non-increasing."""
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or len(observations) < 1:
        raise ValueError("values must be a nonempty one-dimensional array")
    if weights is None:
        sample_weights = np.ones_like(observations)
    else:
        sample_weights = np.asarray(weights, dtype=float)
        if sample_weights.shape != observations.shape:
            raise ValueError("weights must match values")
        if np.any(sample_weights <= 0.0):
            raise ValueError("weights must be positive")

    # Pool adjacent violators. A block ordering a < b violates the requested
    # non-increasing relation and is replaced by its weighted mean.
    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(
        zip(observations, sample_weights)
    ):
        blocks.append([index, index + 1, float(weight), float(value)])
        while len(blocks) >= 2 and blocks[-2][3] < blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            combined_weight = float(left[2]) + float(right[2])
            combined_value = (
                float(left[2]) * float(left[3])
                + float(right[2]) * float(right[3])
            ) / combined_weight
            blocks.append(
                [int(left[0]), int(right[1]), combined_weight, combined_value]
            )
    fitted = np.empty_like(observations)
    for start, stop, _weight, value in blocks:
        fitted[int(start) : int(stop)] = float(value)
    return fitted


def budget_group_standardized_advantages(
    reward_matrix: np.ndarray,
    active_link_count_matrix: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Standardize rewards among eligible candidates at the same budget."""
    rewards = np.asarray(reward_matrix, dtype=float)
    budgets = np.asarray(active_link_count_matrix, dtype=np.int64)
    if rewards.shape != budgets.shape or rewards.ndim != 2:
        raise ValueError("rewards and budgets must have matching 2-D shapes")
    if candidate_mask is None:
        eligible = np.ones_like(rewards, dtype=bool)
    else:
        eligible = np.asarray(candidate_mask, dtype=bool)
        if eligible.shape != rewards.shape:
            raise ValueError("candidate_mask must match reward_matrix")
    advantages = np.zeros_like(rewards)
    for target_index in range(rewards.shape[0]):
        for link_count in np.unique(budgets[target_index]):
            mask = (
                (budgets[target_index] == link_count)
                & eligible[target_index]
            )
            group = rewards[target_index, mask]
            if len(group) == 0:
                continue
            advantages[target_index, mask] = (
                group - np.mean(group)
            ) / (np.std(group) + 1.0e-8)
    return advantages


def budget_error_selector_loss(
    policy: GNNPolicyNetwork,
    node_features: torch.Tensor,
    targets_rad: np.ndarray,
    active_link_counts: np.ndarray,
    aligned_achieved_phases_rad: np.ndarray,
    n_lasers: int,
    config: DesignerConfig,
    rng: np.random.Generator,
    deterministic_candidate_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, float, float]:
    """Train auxiliary heads from measured same-target multi-budget curves."""
    target_relative = make_relative_to_laser_1(targets_rad)
    circular_error = np.angle(
        np.exp(
            1.0j
            * (
                aligned_achieved_phases_rad
                - target_relative[:, None, :]
            )
        )
    )[:, :, 1:]
    if config.selector_use_deterministic_max_error:
        measured_error_fraction = np.max(
            np.abs(circular_error), axis=2
        ) / np.pi
    else:
        measured_error_fraction = np.sqrt(
            np.mean(circular_error**2, axis=2)
        ) / np.pi
    target_count, candidate_count = measured_error_fraction.shape
    budget_matrix = np.asarray(active_link_counts, dtype=np.int64)
    if budget_matrix.ndim == 1:
        if budget_matrix.shape != (target_count,):
            raise ValueError("active_link_counts must have one row per target")
        budget_matrix = np.broadcast_to(
            budget_matrix[:, None], (target_count, candidate_count)
        )
    if budget_matrix.shape != (target_count, candidate_count):
        raise ValueError(
            "active_link_counts must match target and candidate dimensions"
        )
    deterministic_mask = None
    if config.selector_use_deterministic_max_error:
        if deterministic_candidate_mask is None:
            raise ValueError(
                "deterministic_candidate_mask is required for deterministic "
                "maximum-error selector labels"
            )
        deterministic_mask = np.asarray(
            deterministic_candidate_mask, dtype=bool
        )
        if deterministic_mask.shape != budget_matrix.shape:
            raise ValueError(
                "deterministic_candidate_mask must match candidate dimensions"
            )
    maximum_links = n_lasers * (n_lasers - 1)
    minimum_links = n_lasers - 1
    device = node_features.device

    error_context_rows: list[torch.Tensor] = []
    error_density_labels: list[float] = []
    error_value_labels: list[float] = []
    selector_context_rows: list[torch.Tensor] = []
    selector_tolerance_labels: list[float] = []
    selector_q_labels: list[float] = []
    for target_index in range(target_count):
        counts = np.unique(budget_matrix[target_index])
        counts.sort()
        group_sizes: list[float] = []
        group_errors: list[float] = []
        for count in counts:
            group_mask = budget_matrix[target_index] == count
            if deterministic_mask is not None:
                group_mask &= deterministic_mask[target_index]
                if np.count_nonzero(group_mask) != 1:
                    raise ValueError(
                        "each target and budget must contain exactly one "
                        "deterministic selector candidate"
                    )
            group_sizes.append(float(np.count_nonzero(group_mask)))
            group_errors.append(
                float(
                    np.mean(
                        measured_error_fraction[target_index, group_mask]
                    )
                )
            )
        group_sizes_array = np.asarray(group_sizes, dtype=float)
        group_errors_array = np.asarray(group_errors, dtype=float)
        if config.selector_use_deterministic_max_error:
            selector_curve_errors = group_errors_array
        else:
            selector_curve_errors = nonincreasing_isotonic_fit(
                group_errors_array, group_sizes_array
            )
        for count, curve_error in zip(counts, selector_curve_errors):
            error_context_rows.append(node_features[target_index])
            error_density_labels.append(float(count) / maximum_links)
            error_value_labels.append(float(curve_error))

        error_high = float(np.max(selector_curve_errors))
        error_low = float(np.min(selector_curve_errors))
        if error_high - error_low <= 1.0e-8:
            continue
        label_count = config.selector_labels_per_target
        stratified_positions = (
            np.arange(label_count, dtype=float) + rng.random(label_count)
        ) / label_count
        tolerances = error_low + stratified_positions * (
            error_high - error_low
        )
        normalized_q = (counts - minimum_links) / (
            maximum_links - minimum_links
        )
        for tolerance in tolerances:
            feasible_indices = np.flatnonzero(
                selector_curve_errors <= tolerance
            )
            if len(feasible_indices) == 0:
                # Do not repeat the original failure mode by converting an
                # infeasible tolerance into an ordinary all-to-all label.
                continue
            selector_context_rows.append(node_features[target_index])
            selector_tolerance_labels.append(float(tolerance))
            selector_q_labels.append(
                float(normalized_q[int(feasible_indices[0])])
            )

    error_context = torch.stack(error_context_rows)
    error_density = torch.as_tensor(
        error_density_labels, dtype=torch.float32, device=device
    )
    error_targets = torch.as_tensor(
        error_value_labels, dtype=torch.float32, device=device
    )
    predicted_error = policy.predict_normalized_phase_error(
        error_context, error_density
    )
    error_loss = torch.nn.functional.mse_loss(predicted_error, error_targets)

    if selector_context_rows:
        selector_context = torch.stack(selector_context_rows)
        selector_tolerances = torch.as_tensor(
            selector_tolerance_labels, dtype=torch.float32, device=device
        )
        teacher_q = torch.as_tensor(
            selector_q_labels, dtype=torch.float32, device=device
        )
        predicted_q = policy.predict_normalized_budget(
            selector_context, selector_tolerances
        )
        selector_loss = torch.nn.functional.mse_loss(predicted_q, teacher_q)
        selector_budget_mae = float(
            torch.mean(torch.abs(predicted_q - teacher_q)).detach().cpu()
        )
    else:
        # Retain a valid graph-connected scalar without training the selector
        # from an uninformative or infeasible curve.
        selector_loss = sum(
            parameter.sum() * 0.0
            for parameter in policy.budget_selector_head.parameters()
        )
        selector_budget_mae = float("nan")
    error_prediction_mae_deg = float(
        torch.mean(torch.abs(predicted_error - error_targets))
        .detach()
        .cpu()
        * 180.0
    )
    return (
        error_loss + selector_loss,
        error_prediction_mae_deg,
        selector_budget_mae,
    )


def sample_connected_coupling_designs(
    policy: nn.Module,
    encoded_targets: torch.Tensor,
    candidates_per_target: int,
    active_link_counts: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Gaussian actions and exact-budget weakly connected masks.

    A score-weighted growing tree first supplies ``M-1`` directed links whose
    undirected projection spans every laser. Additional directed links are
    sampled without replacement until each target's requested budget is met.
    The categorical log probabilities are combined with the existing
    Gaussian score, preserving the current REINFORCE training method.
    """
    if not getattr(policy, "enable_sparse_gates", False):
        raise ValueError("connected sampling requires sparse gates")
    if candidates_per_target < 1:
        raise ValueError("candidates_per_target must be positive")
    target_count, n_lasers, _ = encoded_targets.shape
    maximum_links = n_lasers * (n_lasers - 1)
    active_link_counts = np.asarray(active_link_counts, dtype=np.int64)
    if active_link_counts.shape != (target_count,):
        raise ValueError("active_link_counts must have one value per target")
    if np.any(active_link_counts < n_lasers - 1) or np.any(
        active_link_counts > maximum_links
    ):
        raise ValueError("active link budgets must lie in [M-1, M*(M-1)]")

    continuous_distribution = policy.distribution(encoded_targets)
    sampled_actions = continuous_distribution.sample((candidates_per_target,))
    continuous_log_probability = continuous_distribution.log_prob(
        sampled_actions.detach()
    )
    action_count = sampled_actions.shape[2]
    gate_logits = policy.gate_distribution(encoded_targets).logits
    if gate_logits.shape != (target_count, maximum_links):
        raise ValueError(
            "connected sampling currently requires one gate per directed link"
        )
    receivers, sources = directed_link_indices(n_lasers)

    candidate_gates: list[torch.Tensor] = []
    candidate_gate_log_probabilities: list[torch.Tensor] = []
    for candidate_index in range(candidates_per_target):
        target_gates: list[torch.Tensor] = []
        target_log_probabilities: list[torch.Tensor] = []
        for target_index, requested_count in enumerate(active_link_counts):
            requested_count = int(requested_count)
            if requested_count == maximum_links:
                target_gates.append(torch.ones_like(gate_logits[target_index]))
                target_log_probabilities.append(
                    gate_logits[target_index].sum() * 0.0
                )
                continue

            selected: list[int] = []
            selected_set: set[int] = set()
            connected_nodes = {
                int((candidate_index + target_index) % n_lasers)
            }
            decision_log_probabilities: list[torch.Tensor] = []

            while len(connected_nodes) < n_lasers:
                crossing = [
                    edge_index
                    for edge_index, (receiver, source) in enumerate(
                        zip(receivers, sources)
                    )
                    if (
                        (int(receiver) in connected_nodes)
                        != (int(source) in connected_nodes)
                    )
                ]
                crossing_tensor = torch.as_tensor(
                    crossing,
                    dtype=torch.long,
                    device=encoded_targets.device,
                )
                distribution = torch.distributions.Categorical(
                    logits=gate_logits[target_index].index_select(
                        0, crossing_tensor
                    )
                )
                local_choice = distribution.sample()
                edge_index = crossing[int(local_choice)]
                decision_log_probabilities.append(
                    distribution.log_prob(local_choice)
                )
                selected.append(edge_index)
                selected_set.add(edge_index)
                connected_nodes.add(int(receivers[edge_index]))
                connected_nodes.add(int(sources[edge_index]))

            while len(selected) < requested_count:
                remaining = [
                    edge_index
                    for edge_index in range(maximum_links)
                    if edge_index not in selected_set
                ]
                remaining_tensor = torch.as_tensor(
                    remaining,
                    dtype=torch.long,
                    device=encoded_targets.device,
                )
                distribution = torch.distributions.Categorical(
                    logits=gate_logits[target_index].index_select(
                        0, remaining_tensor
                    )
                )
                local_choice = distribution.sample()
                edge_index = remaining[int(local_choice)]
                decision_log_probabilities.append(
                    distribution.log_prob(local_choice)
                )
                selected.append(edge_index)
                selected_set.add(edge_index)

            gate = torch.zeros_like(gate_logits[target_index])
            gate[torch.as_tensor(
                selected,
                dtype=torch.long,
                device=encoded_targets.device,
            )] = 1.0
            target_gates.append(gate)
            target_log_probabilities.append(
                torch.stack(decision_log_probabilities).sum()
            )
        candidate_gates.append(torch.stack(target_gates))
        candidate_gate_log_probabilities.append(
            torch.stack(target_log_probabilities)
        )

    sampled_gates = torch.stack(candidate_gates)
    gate_log_probability = torch.stack(candidate_gate_log_probabilities)
    if action_count != 2 * maximum_links:
        raise ValueError(
            "connected sampling requires one magnitude and one phase action "
            "per directed link"
        )
    # Disabled links do not affect the simulator, so exclude their otherwise
    # pure-noise magnitude and phase scores from the REINFORCE objective.
    active_action_mask = torch.cat(
        (sampled_gates, sampled_gates), dim=2
    )
    active_continuous_log_probability = (
        continuous_log_probability * active_action_mask
    ).sum(dim=2)
    decision_counts = torch.as_tensor(
        active_link_counts,
        dtype=sampled_actions.dtype,
        device=sampled_actions.device,
    ).unsqueeze(0)
    topology_decision_counts = torch.where(
        decision_counts == float(maximum_links),
        torch.zeros_like(decision_counts),
        decision_counts,
    )
    joint_log_probability = (
        active_continuous_log_probability + gate_log_probability
    ) / (2.0 * decision_counts + topology_decision_counts)
    return (
        sampled_actions.detach().permute(1, 0, 2),
        sampled_gates.detach().permute(1, 0, 2),
        joint_log_probability.permute(1, 0),
    )


def sample_learned_connected_coupling_designs(
    policy: nn.Module,
    encoded_targets: torch.Tensor,
    candidates_per_target: int,
    *,
    include_dense_reference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample connected masks whose link counts are policy decisions.

    The gate logits first define a categorical policy over the directed edges
    that can grow a spanning tree.  Once all lasers are connected, every
    remaining directed edge is sampled from its learned Bernoulli policy.
    Thus connectivity is guaranteed, while both topology and active-link
    count remain stochastic actions credited by the REINFORCE loss. When
    requested, candidate zero is forced dense to provide a same-condition
    phase reference; only its normally sampled continuous actions contribute
    a score-function gradient.
    """
    if not getattr(policy, "enable_sparse_gates", False):
        raise ValueError("learned connected sampling requires sparse gates")
    if candidates_per_target < 1:
        raise ValueError("candidates_per_target must be positive")

    target_count, n_lasers, _ = encoded_targets.shape
    maximum_links = n_lasers * (n_lasers - 1)
    continuous_distribution = policy.distribution(encoded_targets)
    sampled_actions = continuous_distribution.sample((candidates_per_target,))
    continuous_log_probability = continuous_distribution.log_prob(
        sampled_actions.detach()
    )
    if sampled_actions.shape[2] != 2 * maximum_links:
        raise ValueError(
            "learned connected sampling requires one magnitude and one "
            "phase action per directed link"
        )
    gate_logits = policy.gate_distribution(encoded_targets).logits
    if gate_logits.shape != (target_count, maximum_links):
        raise ValueError(
            "learned connected sampling requires one gate per directed link"
        )
    receivers, sources = directed_link_indices(n_lasers)

    candidate_gates: list[torch.Tensor] = []
    candidate_gate_log_probabilities: list[torch.Tensor] = []
    for candidate_index in range(candidates_per_target):
        target_gates: list[torch.Tensor] = []
        target_log_probabilities: list[torch.Tensor] = []
        for target_index in range(target_count):
            if include_dense_reference and candidate_index == 0:
                target_gates.append(
                    torch.ones_like(gate_logits[target_index])
                )
                # The topology is deliberately forced rather than sampled,
                # so it supplies no Bernoulli score-function gradient.  Its
                # normally sampled continuous actions remain trainable.
                target_log_probabilities.append(
                    gate_logits[target_index].sum() * 0.0
                )
                continue
            selected: list[int] = []
            selected_set: set[int] = set()
            connected_nodes = {
                int((candidate_index + target_index) % n_lasers)
            }
            decision_log_probabilities: list[torch.Tensor] = []

            while len(connected_nodes) < n_lasers:
                crossing = [
                    edge_index
                    for edge_index, (receiver, source) in enumerate(
                        zip(receivers, sources)
                    )
                    if (
                        (int(receiver) in connected_nodes)
                        != (int(source) in connected_nodes)
                    )
                ]
                crossing_tensor = torch.as_tensor(
                    crossing,
                    dtype=torch.long,
                    device=encoded_targets.device,
                )
                tree_distribution = torch.distributions.Categorical(
                    logits=gate_logits[target_index].index_select(
                        0, crossing_tensor
                    )
                )
                local_choice = tree_distribution.sample()
                edge_index = crossing[int(local_choice)]
                decision_log_probabilities.append(
                    tree_distribution.log_prob(local_choice)
                )
                selected.append(edge_index)
                selected_set.add(edge_index)
                connected_nodes.add(int(receivers[edge_index]))
                connected_nodes.add(int(sources[edge_index]))

            remaining = [
                edge_index
                for edge_index in range(maximum_links)
                if edge_index not in selected_set
            ]
            remaining_tensor = torch.as_tensor(
                remaining,
                dtype=torch.long,
                device=encoded_targets.device,
            )
            extra_distribution = torch.distributions.Bernoulli(
                logits=gate_logits[target_index].index_select(
                    0, remaining_tensor
                )
            )
            extra_samples = extra_distribution.sample()
            decision_log_probabilities.append(
                extra_distribution.log_prob(extra_samples).sum()
            )
            for local_index in torch.nonzero(
                extra_samples.detach() > 0.5, as_tuple=False
            ).flatten().tolist():
                selected.append(remaining[int(local_index)])

            gate = torch.zeros_like(gate_logits[target_index])
            gate[
                torch.as_tensor(
                    selected,
                    dtype=torch.long,
                    device=encoded_targets.device,
                )
            ] = 1.0
            target_gates.append(gate)
            target_log_probabilities.append(
                torch.stack(decision_log_probabilities).sum()
            )
        candidate_gates.append(torch.stack(target_gates))
        candidate_gate_log_probabilities.append(
            torch.stack(target_log_probabilities)
        )

    sampled_gates = torch.stack(candidate_gates)
    gate_log_probability = torch.stack(candidate_gate_log_probabilities)
    active_action_mask = torch.cat((sampled_gates, sampled_gates), dim=2)
    active_continuous_log_probability = (
        continuous_log_probability * active_action_mask
    ).sum(dim=2)
    # Use a fixed denominator so choosing fewer links does not itself rescale
    # the score-function estimator.
    joint_log_probability = (
        active_continuous_log_probability + gate_log_probability
    ) / float(3 * maximum_links)
    return (
        sampled_actions.detach().permute(1, 0, 2),
        sampled_gates.detach().permute(1, 0, 2),
        joint_log_probability.permute(1, 0),
    )


def deterministic_connected_gates(
    gate_logits: np.ndarray,
    active_link_counts: np.ndarray,
    n_lasers: int,
) -> np.ndarray:
    """Select a maximum-score spanning tree and exact-budget extra links."""
    gate_logits = np.asarray(gate_logits, dtype=float)
    active_link_counts = np.asarray(active_link_counts, dtype=np.int64)
    maximum_links = n_lasers * (n_lasers - 1)
    if gate_logits.ndim != 2 or gate_logits.shape[1] != maximum_links:
        raise ValueError("gate_logits must have shape (batch, M*(M-1))")
    if active_link_counts.shape != (gate_logits.shape[0],):
        raise ValueError("active_link_counts must have one value per row")
    receivers, sources = directed_link_indices(n_lasers)
    edge_lookup = {
        (int(receiver), int(source)): edge_index
        for edge_index, (receiver, source) in enumerate(
            zip(receivers, sources)
        )
    }
    gates = np.zeros_like(gate_logits)
    for row_index, requested_count in enumerate(active_link_counts):
        requested_count = int(requested_count)
        if not n_lasers - 1 <= requested_count <= maximum_links:
            raise ValueError("active link budgets must lie in [M-1, M*(M-1)]")
        parent = list(range(n_lasers))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        pair_candidates = []
        for first in range(n_lasers):
            for second in range(first + 1, n_lasers):
                first_receives = edge_lookup[(first, second)]
                second_receives = edge_lookup[(second, first)]
                selected_direction = (
                    first_receives
                    if gate_logits[row_index, first_receives]
                    >= gate_logits[row_index, second_receives]
                    else second_receives
                )
                pair_candidates.append(
                    (
                        gate_logits[row_index, selected_direction],
                        first,
                        second,
                        selected_direction,
                    )
                )
        selected: list[int] = []
        for _, first, second, edge_index in sorted(
            pair_candidates, reverse=True
        ):
            first_root = find(first)
            second_root = find(second)
            if first_root == second_root:
                continue
            parent[first_root] = second_root
            selected.append(edge_index)
            if len(selected) == n_lasers - 1:
                break
        selected_set = set(selected)
        remaining = sorted(
            (
                edge_index
                for edge_index in range(maximum_links)
                if edge_index not in selected_set
            ),
            key=lambda edge_index: gate_logits[row_index, edge_index],
            reverse=True,
        )
        selected.extend(remaining[: requested_count - len(selected)])
        gates[row_index, selected] = 1.0
    return gates


def deterministic_learned_connected_gates(
    gate_logits: np.ndarray,
    n_lasers: int,
    probability_threshold: float = 0.5,
) -> np.ndarray:
    """Threshold learned gates while always retaining a spanning tree."""
    gate_logits = np.asarray(gate_logits, dtype=float)
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must lie strictly in (0, 1)")
    spanning_tree = deterministic_connected_gates(
        gate_logits,
        np.full(gate_logits.shape[0], n_lasers - 1, dtype=np.int64),
        n_lasers,
    )
    threshold_logit = np.log(
        probability_threshold / (1.0 - probability_threshold)
    )
    return np.maximum(spanning_tree, gate_logits >= threshold_logit).astype(
        float
    )


def run_architecture_sanity_checks(
    config: DesignerConfig,
    n_lasers_values: tuple[int, ...] = (2, 3, 5, 7, 10),
) -> dict[int, tuple[int, int]]:
    """Check dynamic forward shapes and shared-parameter gradients.

    This intentionally performs no VCSEL simulations. A fresh policy instance
    is used so calling the check cannot alter a trained model.
    """
    policy = make_policy(config).to(torch.device(config.device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    parameter_shapes_before = {
        name: tuple(parameter.shape)
        for name, parameter in policy.named_parameters()
    }
    output_shapes: dict[int, tuple[int, int]] = {}
    losses = []
    for n_lasers in n_lasers_values:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.random_seed + n_lasers)
        feature_count = 4 if config.condition_on_edge_budget else 3
        node_features = torch.randn(
            (2, n_lasers, feature_count),
            generator=generator,
            dtype=torch.float32,
        ).to(torch.device(config.device))
        if config.condition_on_edge_budget:
            # The fourth feature is a broadcast retained-link fraction and
            # must represent the same requested budget for every node.
            budget_fraction = torch.tensor(
                [1.0 / n_lasers, 1.0],
                dtype=node_features.dtype,
                device=node_features.device,
            )
            node_features[:, :, 3] = budget_fraction[:, None]
        action_means = policy(node_features)
        expected_width = policy.action_size_for(n_lasers)
        if action_means.shape != (2, expected_width):
            raise AssertionError(
                f"M={n_lasers}: got {tuple(action_means.shape)}, "
                f"expected (2, {expected_width})"
            )
        if (
            not config.force_symmetric_kappa
            and not config.force_symmetric_phi_p
            and expected_width != 2 * n_lasers * (n_lasers - 1)
        ):
            raise AssertionError("fully directed action width is incorrect")
        output_shapes[n_lasers] = tuple(action_means.shape)
        losses.append(action_means.square().mean())

    optimizer.zero_grad()
    torch.stack(losses).sum().backward()
    if isinstance(policy, PolicyNetwork):
        encoder_gradient = policy.global_encoder[0].weight.grad
    elif isinstance(policy, BiGRUPolicyNetwork):
        encoder_gradient = policy.bigru.weight_ih_l0.grad
    else:
        encoder_gradient = (
            policy.message_layers[0].message_mlp[0].weight.grad
        )
    required_gradients = (
        policy.node_encoder[0].weight.grad,
        encoder_gradient,
        policy.edge_decoder[-1].weight.grad,
    )
    if any(gradient is None for gradient in required_gradients):
        raise AssertionError(
            "encoder and edge decoder must share gradients across all tested "
            "M values"
        )
    optimizer.step()
    parameter_shapes_after = {
        name: tuple(parameter.shape)
        for name, parameter in policy.named_parameters()
    }
    if parameter_shapes_after != parameter_shapes_before:
        raise AssertionError("policy parameter shapes changed with M")
    print("Variable-M architecture sanity checks passed:")
    for n_lasers, shape in output_shapes.items():
        print(f"  M={n_lasers}: action means {shape}")
    return output_shapes


# ---------------------------------------------------------------------------
# 3. Action vector -> complete coupling matrices
# ---------------------------------------------------------------------------


def effective_maximum_kappa_per_link(
    maximum_kappa_per_ns: float,
    n_lasers: int,
    normalize_incoming_coupling_by_degree: bool = False,
) -> float:
    """Return the physical upper bound for one directed coupling link."""
    if maximum_kappa_per_ns <= 0.0:
        raise ValueError("maximum_kappa_per_ns must be positive")
    if n_lasers < 2:
        raise ValueError("n_lasers must be at least 2")
    if normalize_incoming_coupling_by_degree:
        return maximum_kappa_per_ns / float(n_lasers - 1)
    return maximum_kappa_per_ns


def decode_action(
    actions: np.ndarray,
    maximum_kappa_per_ns: float,
    n_lasers: int = DEFAULT_N_LASERS,
    force_symmetric_kappa: bool = False,
    force_symmetric_phi_p: bool = False,
    balanced_coupling: bool = False,
    magnitude_gates: np.ndarray | None = None,
    normalize_incoming_coupling_by_degree: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode latent actions into complete ``N x N`` coupling matrices.

    By default both matrices have ``N*(N-1)`` directed off-diagonal actions.
    Each matrix can independently use either ``N*(N-1)`` directed actions or
    ``N*(N-1)/2`` unique actions that are mirrored across the diagonal:

    * magnitude logits come first;
    * coupling-phase actions follow the magnitude logits;
    * both matrix diagonals are always zero;
    * ``kappa[i, j]`` injects source laser ``j`` into receiver laser ``i``.

    Args:
        actions: Shape ``(number_of_cases, action_size_for(...))``.
        maximum_kappa_per_ns: Upper magnitude bound in inverse nanoseconds.
        magnitude_gates: Optional hard/soft gates shaped
            ``(number_of_cases, n_magnitude_links)``.
        normalize_incoming_coupling_by_degree: Divide every physical edge by
            ``n_lasers - 1`` so a receiver's maximum incoming sum remains
            ``maximum_kappa_per_ns`` for every array size.

    Returns:
        ``kappa_per_ns`` and ``phi_p_rad``, each shaped
        ``(number_of_cases, N, N)``.
    """
    actions = np.asarray(actions, dtype=float)
    n_directed_links = n_lasers * (n_lasers - 1)
    n_magnitude_links = (
        n_lasers * (n_lasers - 1) // 2
        if force_symmetric_kappa
        else n_directed_links
    )
    n_phase_links = (
        n_lasers * (n_lasers - 1) // 2
        if force_symmetric_phi_p
        else n_directed_links
    )
    action_size = n_magnitude_links + n_phase_links
    if actions.ndim != 2 or actions.shape[1] != action_size:
        raise ValueError(
            f"actions must have shape (number_of_cases, {action_size})"
        )

    # Keep the sigmoid numerically safe while retaining a useful near-maximum
    # action for explicit high-logit designs.
    magnitude_logits = np.clip(
        actions[:, :n_magnitude_links], -8.0, 8.0
    )
    maximum_kappa_per_link = effective_maximum_kappa_per_link(
        maximum_kappa_per_ns,
        n_lasers,
        normalize_incoming_coupling_by_degree,
    )
    link_kappa_per_ns = maximum_kappa_per_link / (
        1.0 + np.exp(-magnitude_logits)
    )
    if magnitude_gates is not None:
        magnitude_gates = np.asarray(magnitude_gates, dtype=float)
        if magnitude_gates.shape != link_kappa_per_ns.shape:
            raise ValueError(
                "magnitude_gates must have shape "
                f"{link_kappa_per_ns.shape}, got {magnitude_gates.shape}"
            )
        if not np.all(np.isfinite(magnitude_gates)):
            raise ValueError("magnitude_gates must be finite")
        if np.any((magnitude_gates < 0.0) | (magnitude_gates > 1.0)):
            raise ValueError("magnitude_gates must lie in [0, 1]")
        link_kappa_per_ns *= magnitude_gates
    link_phi_p_rad = np.angle(
        np.exp(1j * actions[:, n_magnitude_links:])
    )

    cases = len(actions)
    magnitude_receivers, magnitude_sources = coupling_link_indices(
        n_lasers, force_symmetric_kappa
    )
    phase_receivers, phase_sources = coupling_link_indices(
        n_lasers, force_symmetric_phi_p
    )
    kappa_per_ns = np.zeros((cases, n_lasers, n_lasers))
    phi_p_rad = np.zeros_like(kappa_per_ns)
    kappa_per_ns[:, magnitude_receivers, magnitude_sources] = link_kappa_per_ns
    phi_p_rad[:, phase_receivers, phase_sources] = link_phi_p_rad
    if force_symmetric_kappa:
        kappa_per_ns[:, magnitude_sources, magnitude_receivers] = (
            link_kappa_per_ns
        )
    if balanced_coupling and not force_symmetric_kappa:
        # Balance total upper- and lower-triangular coupling without making
        # corresponding links equal.  Scaling only the larger side preserves
        # the per-link cap and the relative strengths within each side.
        upper_receivers, upper_sources = np.triu_indices(n_lasers, k=1)
        lower_receivers, lower_sources = np.tril_indices(n_lasers, k=-1)
        upper_sum = np.sum(
            kappa_per_ns[:, upper_receivers, upper_sources], axis=1
        )
        lower_sum = np.sum(
            kappa_per_ns[:, lower_receivers, lower_sources], axis=1
        )
        common_sum = np.minimum(upper_sum, lower_sum)
        upper_scale = np.divide(
            common_sum,
            upper_sum,
            out=np.ones_like(common_sum),
            where=upper_sum > 0.0,
        )
        lower_scale = np.divide(
            common_sum,
            lower_sum,
            out=np.ones_like(common_sum),
            where=lower_sum > 0.0,
        )
        kappa_per_ns[:, upper_receivers, upper_sources] *= upper_scale[:, None]
        kappa_per_ns[:, lower_receivers, lower_sources] *= lower_scale[:, None]
    if force_symmetric_phi_p:
        phi_p_rad[:, phase_sources, phase_receivers] = link_phi_p_rad
    return kappa_per_ns, phi_p_rad


# ---------------------------------------------------------------------------
# 4. Explicit vcsel_lib environment
# ---------------------------------------------------------------------------


def make_vcsel_physical_parameters(
    config: DesignerConfig,
    *,
    detuning_distribution_ghz: np.ndarray | None = None,
) -> dict[str, object]:
    """Build the ``n_lasers`` physical parameter dictionary used by VCSEL.

    When ``detuning_distribution_ghz`` is omitted, each simulation receives
    the usual independent uniform detunings.  Supplying an array gives an
    explicit detuning for each laser, in GHz relative to the reference
    frequency.
    """
    if detuning_distribution_ghz is None:
        detuning_ghz = np.random.uniform(-0.5, 0.5, config.n_lasers)
        detuning_ghz *= config.detuning_span_ghz
    else:
        detuning_ghz = np.asarray(detuning_distribution_ghz, dtype=float)
        valid_shape = (
            detuning_ghz.shape == (config.n_lasers,)
            or (
                detuning_ghz.ndim == 2
                and detuning_ghz.shape[1] == config.n_lasers
            )
        )
        if not valid_shape:
            raise ValueError(
                "detuning_distribution_ghz must have shape "
                f"({config.n_lasers},) or (cases, {config.n_lasers}), "
                f"got {detuning_ghz.shape}"
            )
        if not np.all(np.isfinite(detuning_ghz)):
            raise ValueError("detuning_distribution_ghz must be finite")
    detuning_per_second = (
        detuning_ghz * 2.0 * np.pi * 1.0e9
    )
    return {
        "tau_p": config.photon_lifetime_seconds,
        "tau_n": config.carrier_lifetime_seconds,
        "g0": config.gain_per_second,
        "N0": config.transparency_carriers,
        "s": config.gain_saturation,
        "beta": config.spontaneous_emission_factor,
        "kappa_c_mat": np.zeros((config.n_lasers, config.n_lasers)),
        "phi_p_mat": np.zeros((config.n_lasers, config.n_lasers)),
        "I": config.pump_current_amp,
        "q": config.electron_charge_coulomb,
        "alpha": config.alpha,
        "delta": detuning_per_second,
        "coupling": 1.0,
        "self_feedback": 0.0,
        "noise_amplitude": config.noise_amplitude,
        "dt": config.time_step_seconds,
        "Tmax": config.simulation_time_seconds,
        "tau": config.delay_seconds,
        "N_lasers": config.n_lasers,
        "save_every": config.save_every,
        "show_output_size_message": False,
    }


def simulate_with_vcsel(
    kappa_per_ns: np.ndarray,
    phi_p_rad: np.ndarray,
    config: DesignerConfig,
    *,
    progress: bool = False,
    smooth_frequencies: bool = False,
    detuning_distribution_ghz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate candidate matrices from free-running initial conditions.

    Shapes:
        coupling matrices: ``(cases, N, N)``
        returned ``states``: ``(cases, 3*N, saved_times)``

    Units:
        input kappa is ns^-1; VCSEL internally receives nondimensional
        ``kappa_per_second * photon_lifetime_seconds``.
    """
    kappa_per_ns = np.asarray(kappa_per_ns, dtype=float)
    phi_p_rad = np.asarray(phi_p_rad, dtype=float)
    if kappa_per_ns.shape != phi_p_rad.shape:
        raise ValueError("kappa_per_ns and phi_p_rad must have matching shapes")
    if kappa_per_ns.ndim != 3 or kappa_per_ns.shape[1:] != (
        config.n_lasers,
        config.n_lasers,
    ):
        raise ValueError(
            "coupling matrices must have shape "
            f"(cases, {config.n_lasers}, {config.n_lasers})"
        )

    cases = len(kappa_per_ns)
    physical_parameters = make_vcsel_physical_parameters(
        config,
        detuning_distribution_ghz=detuning_distribution_ghz,
    )

    # These are the main vcsel_lib steps, kept together intentionally.
    vcsel_class = _load_vcsel_class()
    vcsel = vcsel_class(physical_parameters)
    nondimensional_parameters = vcsel.scale_params()

    # Convert ns^-1 -> s^-1 -> the nondimensional coupling used internally.
    kappa_per_second = kappa_per_ns * 1.0e9
    nondimensional_parameters["kappa"] = (
        kappa_per_second * config.photon_lifetime_seconds
    )
    nondimensional_parameters["kappa_case_dependent"] = True
    nondimensional_parameters["phi_p"] = phi_p_rad

    full_time_grid_seconds = (
        np.arange(nondimensional_parameters["steps"])
        * config.time_step_seconds
    )
    nondimensional_parameters["kappa_ramp"] = vcsel_class.cosine_ramp(
        full_time_grid_seconds,
        t_start=(
            config.coupling_ramp_start_delays * config.delay_seconds
        ),
        rise_10_90=(
            config.coupling_ramp_rise_delays * config.delay_seconds
        ),
        kappa_initial=0.0,
        kappa_final=1.0,
    )

    # Every candidate starts from the same kind of free-running history.
    history, initial_frequency_ghz, _, _ = vcsel.generate_history(
        nondimensional_parameters,
        shape="FR",
        n_cases=cases,
    )
    # Strong exploratory couplings can make an individual candidate diverge.
    # Keep those expected floating-point warnings out of notebook output; the
    # reward calculation below detects nonfinite trajectories and assigns them
    # the explicit failure reward.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        time_seconds, states, frequencies_nondimensional = vcsel.integrate(
            history,
            nd=nondimensional_parameters,
            progress=progress,
            max_iter=1,
            smooth_freqs=smooth_frequencies,
        )
    return (
        time_seconds,
        states,
        frequencies_nondimensional,
        initial_frequency_ghz,
    )


# ---------------------------------------------------------------------------
# 5. Phase reward
# ---------------------------------------------------------------------------


def calculate_phase_reward(
    time_seconds: np.ndarray,
    states: np.ndarray,
    requested_phases_rad: np.ndarray,
    config: DesignerConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculate settled phase agreement for every simulated case.

    For nonreference laser ``i`` at saved time ``t``:

    ``score_i(t) = 0.5 * (1 + cos(achieved_i(t) - requested_i))``.

    Scores are averaged over the settled tail. The final phase reward is an
    equal-weight blend (by default) of the mean laser score and the weakest
    laser score. If conjugacy is enabled, direct and globally sign-reversed
    targets are compared, and one sign is selected for the complete pattern.

    Returns phase reward, mean score, worst score, raw achieved phases,
    globally aligned achieved phases, and the selected orientation.
    """
    targets = make_relative_to_laser_1(requested_phases_rad)
    if targets.ndim == 1:
        targets = targets[None, :]
    if states.shape[0] != len(targets):
        raise ValueError("one requested target is required per simulated case")
    if targets.shape[1] != config.n_lasers:
        raise ValueError(
            f"requested targets must contain {config.n_lasers} phases"
        )
    if states.shape[1] != 3 * config.n_lasers:
        raise ValueError(
            "states do not match the configured number of lasers"
        )

    ramp_end_seconds = (
        config.coupling_ramp_start_delays
        + config.coupling_ramp_rise_delays / 0.8
    ) * config.delay_seconds
    tail_start = max(
        int((1.0 - config.reward_tail_fraction) * states.shape[2]),
        int(np.searchsorted(time_seconds, ramp_end_seconds)),
    )
    if tail_start >= states.shape[2]:
        raise ValueError(
            "simulation is too short to contain a settled reward interval"
        )

    phases_rad = states[:, 2::3, tail_start:]
    relative_phases_rad = np.angle(
        np.exp(1j * (phases_rad - phases_rad[:, :1]))
    )

    def score_orientation(sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        error_rad = np.angle(
            np.exp(
                1j
                * (
                    relative_phases_rad
                    - sign * targets[:, :, None]
                )
            )
        )
        per_laser_score = np.mean(
            0.5 * (1.0 + np.cos(error_rad[:, 1:])), axis=2
        )
        mean_score = np.mean(per_laser_score, axis=1)
        worst_score = np.min(per_laser_score, axis=1)
        weight = config.worst_laser_reward_weight
        reward = (1.0 - weight) * mean_score + weight * worst_score
        return reward, mean_score, worst_score

    direct = score_orientation(+1.0)
    if config.allow_global_phase_conjugate:
        conjugate = score_orientation(-1.0)
        use_conjugate = conjugate[0] > direct[0]
        phase_reward = np.where(use_conjugate, conjugate[0], direct[0])
        mean_score = np.where(use_conjugate, conjugate[1], direct[1])
        worst_score = np.where(use_conjugate, conjugate[2], direct[2])
        orientation = np.where(use_conjugate, -1.0, 1.0)
    else:
        phase_reward, mean_score, worst_score = direct
        orientation = np.ones(len(targets))

    circular_mean = np.mean(np.exp(1j * relative_phases_rad), axis=2)
    achieved_phases_rad = np.angle(circular_mean)
    achieved_phases_rad[:, 0] = 0.0
    aligned_achieved_phases_rad = np.angle(
        np.exp(1j * orientation[:, None] * achieved_phases_rad)
    )

    finite = np.all(np.isfinite(states[:, :, tail_start:]), axis=(1, 2))
    phase_reward = np.where(finite, phase_reward, -1.0)
    return (
        phase_reward,
        mean_score,
        worst_score,
        achieved_phases_rad,
        aligned_achieved_phases_rad,
        orientation,
    )


def normalized_magnitude_symmetry_penalty(
    kappa_per_ns: np.ndarray,
    maximum_kappa_per_ns: float,
) -> np.ndarray:
    """Return the mean reciprocal-link magnitude mismatch in [0, 1]."""
    kappa_per_ns = np.asarray(kappa_per_ns, dtype=float)
    if (
        kappa_per_ns.ndim != 3
        or kappa_per_ns.shape[1] != kappa_per_ns.shape[2]
    ):
        raise ValueError("kappa_per_ns must have shape (cases, N, N)")
    if maximum_kappa_per_ns <= 0.0:
        raise ValueError("maximum_kappa_per_ns must be positive")
    n_lasers = kappa_per_ns.shape[1]
    if n_lasers < 2:
        raise ValueError("coupling matrices must contain at least two lasers")
    difference = np.abs(
        kappa_per_ns - np.swapaxes(kappa_per_ns, 1, 2)
    )
    return np.sum(difference, axis=(1, 2)) / (
        n_lasers * (n_lasers - 1) * maximum_kappa_per_ns
    )


def normalized_squared_coupling_cost(
    kappa_per_ns: np.ndarray,
    maximum_kappa_per_ns: float,
) -> np.ndarray:
    """Return mean squared active coupling, normalized to ``[0, 1]``.

    The zero diagonal is excluded through the ``N*(N-1)`` denominator.  A
    disabled link contributes zero, while an active link at the per-link
    physical cap contributes one before averaging over all possible directed
    links.
    """
    kappa_per_ns = np.asarray(kappa_per_ns, dtype=float)
    if (
        kappa_per_ns.ndim != 3
        or kappa_per_ns.shape[1] != kappa_per_ns.shape[2]
    ):
        raise ValueError("kappa_per_ns must have shape (cases, N, N)")
    if maximum_kappa_per_ns <= 0.0:
        raise ValueError("maximum_kappa_per_ns must be positive")
    n_lasers = kappa_per_ns.shape[1]
    if n_lasers < 2:
        raise ValueError("coupling matrices must contain at least two lasers")
    normalized = kappa_per_ns / maximum_kappa_per_ns
    return np.sum(normalized * normalized, axis=(1, 2)) / (
        n_lasers * (n_lasers - 1)
    )


def successful_sparse_reward_components(
    phase_reward_matrix: np.ndarray,
    active_fraction_matrix: np.ndarray,
    coupling_cost_matrix: np.ndarray,
    *,
    n_lasers: int,
    phase_threshold: float,
    sparsity_weight: float,
    coupling_cost_tiebreak_fraction: float,
    phase_reference: np.ndarray | None = None,
    phase_retention_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the phase-first, links-second, coupling-cost-third reward.

    ``phase_reference`` optionally replaces the fixed phase threshold with
    one reference value per target. The maximum coupling-cost bonus is a
    configurable fraction below one of the reward increment produced by
    removing a single directed link. Thus coupling magnitude can break ties
    between equally sparse eligible designs, but can never make an extra-link
    design preferable.
    """
    phase_reward_matrix = np.asarray(phase_reward_matrix, dtype=float)
    active_fraction_matrix = np.asarray(active_fraction_matrix, dtype=float)
    coupling_cost_matrix = np.asarray(coupling_cost_matrix, dtype=float)
    if (
        active_fraction_matrix.shape != phase_reward_matrix.shape
        or coupling_cost_matrix.shape != phase_reward_matrix.shape
    ):
        raise ValueError(
            "phase rewards, active fractions, and coupling costs must have "
            "matching shapes"
        )
    if n_lasers < 2:
        raise ValueError("n_lasers must be at least 2")
    if not 0.0 <= phase_threshold <= 1.0:
        raise ValueError("phase_threshold must lie in [0, 1]")
    if phase_retention_tolerance < 0.0:
        raise ValueError("phase_retention_tolerance must be nonnegative")
    if sparsity_weight < 0.0:
        raise ValueError("sparsity_weight must be nonnegative")
    if not 0.0 <= coupling_cost_tiebreak_fraction < 1.0:
        raise ValueError(
            "coupling_cost_tiebreak_fraction must lie in [0, 1)"
        )

    if phase_reference is None:
        required_phase_reward = np.full(
            (phase_reward_matrix.shape[0], 1), phase_threshold
        )
    else:
        phase_reference = np.asarray(phase_reference, dtype=float)
        if phase_reference.shape == (phase_reward_matrix.shape[0],):
            phase_reference = phase_reference[:, None]
        if phase_reference.shape != (phase_reward_matrix.shape[0], 1):
            raise ValueError(
                "phase_reference must have one value per target"
            )
        required_phase_reward = (
            phase_reference - phase_retention_tolerance
        )
    successful = phase_reward_matrix >= required_phase_reward
    sparsity_bonus = (
        sparsity_weight
        * successful
        * (1.0 - active_fraction_matrix)
    )
    maximum_links = n_lasers * (n_lasers - 1)
    coupling_cost_bonus = (
        sparsity_weight
        * coupling_cost_tiebreak_fraction
        / float(maximum_links)
        * successful
        * (1.0 - np.clip(coupling_cost_matrix, 0.0, 1.0))
    )
    phase_or_sparse_reward = np.where(
        successful,
        1.0 + sparsity_bonus + coupling_cost_bonus,
        phase_reward_matrix,
    )
    return (
        phase_or_sparse_reward,
        sparsity_bonus,
        coupling_cost_bonus,
        successful,
    )


def lexicographic_resource_rank_advantages(
    phase_reward_matrix: np.ndarray,
    active_fraction_matrix: np.ndarray,
    coupling_cost_matrix: np.ndarray,
    *,
    phase_reference: np.ndarray,
    phase_retention_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weight-free ordinal advantages for hierarchical objectives.

    Every phase-eligible candidate outranks every ineligible candidate.
    Ineligible candidates are ordered only by phase reward. Eligible
    candidates are ordered by fewer active links, lower coupling cost, then
    higher phase reward. Rank zero is worst and rank ``C-1`` is best.
    """
    phase_reward_matrix = np.asarray(phase_reward_matrix, dtype=float)
    active_fraction_matrix = np.asarray(active_fraction_matrix, dtype=float)
    coupling_cost_matrix = np.asarray(coupling_cost_matrix, dtype=float)
    if phase_reward_matrix.ndim != 2 or phase_reward_matrix.shape[1] < 1:
        raise ValueError(
            "phase_reward_matrix must have shape (targets, candidates >= 1)"
        )
    if (
        active_fraction_matrix.shape != phase_reward_matrix.shape
        or coupling_cost_matrix.shape != phase_reward_matrix.shape
    ):
        raise ValueError(
            "phase rewards, active fractions, and coupling costs must have "
            "matching shapes"
        )
    phase_reference = np.asarray(phase_reference, dtype=float)
    if phase_reference.shape == phase_reward_matrix.shape:
        # A per-candidate dense counterpart can serve as the reference when
        # every sparse action is obtained by pruning that same continuous
        # kappa/phi action.
        pass
    elif phase_reference.shape == (phase_reward_matrix.shape[0],):
        phase_reference = phase_reference[:, None]
    if phase_reference.shape not in {
        phase_reward_matrix.shape,
        (phase_reward_matrix.shape[0], 1),
    }:
        raise ValueError(
            "phase_reference must have one value per target or candidate"
        )
    if phase_retention_tolerance < 0.0:
        raise ValueError("phase_retention_tolerance must be nonnegative")

    eligible = phase_reward_matrix >= (
        phase_reference - phase_retention_tolerance
    )
    candidate_count = phase_reward_matrix.shape[1]
    if candidate_count == 1:
        # Ranking is only meaningful when candidates compete.  A single
        # inference candidate is selected directly, so give it a neutral
        # advantage while still reporting its eligibility.
        zeros = np.zeros_like(phase_reward_matrix, dtype=float)
        return zeros, eligible, zeros.copy()

    ranks = np.empty_like(phase_reward_matrix, dtype=float)
    candidate_index = np.arange(candidate_count)
    for target_index in range(phase_reward_matrix.shape[0]):
        target_eligible = eligible[target_index]
        # ``np.lexsort`` treats the last key as primary and sorts ascending.
        # These keys therefore list candidates from worst to best.
        order = np.lexsort(
            (
                candidate_index,
                phase_reward_matrix[target_index],
                np.where(
                    target_eligible,
                    -coupling_cost_matrix[target_index],
                    0.0,
                ),
                np.where(
                    target_eligible,
                    -active_fraction_matrix[target_index],
                    0.0,
                ),
                target_eligible.astype(np.int8),
            )
        )
        ranks[target_index, order] = np.arange(candidate_count, dtype=float)

    advantages = 2.0 * ranks / float(candidate_count - 1) - 1.0
    return advantages, eligible, ranks


def active_training_n_lasers_at(
    iteration: int,
    config: DesignerConfig,
) -> tuple[int, ...]:
    """Return the array sizes eligible for sampling at one iteration."""
    training_sizes = tuple(config.training_n_lasers)
    if not config.enable_array_size_curriculum:
        return training_sizes

    active_count = config.array_size_curriculum_initial_count
    if iteration >= config.array_size_curriculum_start_iteration:
        additions = 1 + (
            iteration - config.array_size_curriculum_start_iteration
        ) // config.array_size_curriculum_add_interval
        active_count += additions
    return training_sizes[: min(active_count, len(training_sizes))]


def minimum_log_std_at(
    iteration: int,
    config: DesignerConfig,
) -> float:
    """Return the scheduled lower bound on policy exploration."""
    if not config.enable_minimum_log_std_annealing:
        return config.minimum_log_std
    start = config.minimum_log_std_anneal_start_iteration
    if iteration <= start:
        return config.initial_minimum_log_std
    duration = config.minimum_log_std_anneal_iterations
    if duration <= 0:
        return config.minimum_log_std
    progress = float(np.clip((iteration - start) / duration, 0.0, 1.0))
    return (
        (1.0 - progress) * config.initial_minimum_log_std
        + progress * config.minimum_log_std
    )


def maximum_log_std_at(
    iteration: int,
    config: DesignerConfig,
) -> float:
    """Return the scheduled upper bound on policy exploration."""
    if not config.enable_exploration_annealing:
        return config.maximum_log_std
    if iteration < config.exploration_anneal_start_iteration:
        return config.maximum_log_std
    if config.exploration_anneal_iterations <= 0:
        return config.final_maximum_log_std
    progress = (
        iteration - config.exploration_anneal_start_iteration + 1
    ) / config.exploration_anneal_iterations
    progress = float(np.clip(progress, 0.0, 1.0))
    return (
        (1.0 - progress) * config.maximum_log_std
        + progress * config.final_maximum_log_std
    )


# ---------------------------------------------------------------------------
# 6. Target-level multiprocessing above the vectorized VCSEL simulator
# ---------------------------------------------------------------------------


def split_simulation_batch(
    targets_rad: np.ndarray,
    kappa_per_ns: np.ndarray,
    phi_p_rad: np.ndarray,
    config: DesignerConfig,
    *,
    detuning_distributions_ghz: np.ndarray | None = None,
    n_jobs: int | None = None,
) -> list[
    tuple[
        int,
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        DesignerConfig,
    ]
]:
    """Split a simulation batch only along its target dimension.

    Each returned work item contains every candidate belonging to its target
    range. For example, ten targets and three jobs produce chunks of sizes
    ``[4, 3, 3]``. No candidate dimension is ever split.
    """
    targets = np.asarray(targets_rad, dtype=float)
    kappa = np.asarray(kappa_per_ns, dtype=float)
    phi_p = np.asarray(phi_p_rad, dtype=float)
    detunings = None
    if detuning_distributions_ghz is not None:
        detunings = np.asarray(detuning_distributions_ghz, dtype=float)
        if detunings.shape != (targets.shape[0], config.n_lasers):
            raise ValueError(
                "detuning_distributions_ghz must have shape "
                f"({targets.shape[0]}, {config.n_lasers})"
            )
    expected_matrix_tail = (config.n_lasers, config.n_lasers)

    if targets.ndim != 2 or targets.shape[1] != config.n_lasers:
        raise ValueError(
            "targets_rad must have shape "
            f"(targets, {config.n_lasers})"
        )
    if targets.shape[0] < 1:
        raise ValueError("at least one target is required")
    if (
        kappa.ndim != 4
        or kappa.shape[0] != targets.shape[0]
        or kappa.shape[2:] != expected_matrix_tail
    ):
        raise ValueError(
            "kappa_per_ns must have shape "
            f"(targets, candidates, {config.n_lasers}, "
            f"{config.n_lasers})"
        )
    if phi_p.shape != kappa.shape:
        raise ValueError(
            "phi_p_rad must have the same "
            "(targets, candidates, N, N) shape as kappa_per_ns"
        )
    if kappa.shape[1] < 1:
        raise ValueError("at least one candidate per target is required")

    requested_jobs = config.n_jobs if n_jobs is None else n_jobs
    if requested_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    active_jobs = min(int(requested_jobs), targets.shape[0])
    target_index_chunks = np.array_split(
        np.arange(targets.shape[0]), active_jobs
    )

    work_items = []
    for chunk_index, target_indices in enumerate(target_index_chunks):
        # array_split returns contiguous, nonempty ranges because active_jobs
        # never exceeds the number of targets.
        start_index = int(target_indices[0])
        stop_index = int(target_indices[-1]) + 1
        work_items.append(
            (
                chunk_index,
                start_index,
                stop_index,
                np.ascontiguousarray(targets[start_index:stop_index]),
                np.ascontiguousarray(kappa[start_index:stop_index]),
                np.ascontiguousarray(phi_p[start_index:stop_index]),
                (
                    None
                    if detunings is None
                    else np.ascontiguousarray(
                        detunings[start_index:stop_index]
                    )
                ),
                config,
            )
        )
    return work_items


def simulate_target_chunk(
    work_item: tuple[
        int,
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        DesignerConfig,
    ],
) -> dict[str, int | np.ndarray]:
    """Simulate one target chunk with one vectorized integrator call.

    This module-level function is spawn-picklable. It receives NumPy arrays
    shaped ``(chunk_targets, candidates, N, N)``, flattens only the first two
    dimensions, and calls :func:`simulate_with_vcsel` exactly once.

    Worker seeding is unnecessary: target/action sampling stays in the parent,
    the free-running history is analytic, and simulator noise is disabled.
    """
    (
        chunk_index,
        start_index,
        stop_index,
        targets,
        kappa,
        phi_p,
        detunings,
        config,
    ) = work_item
    chunk_shape_text = (
        f"targets={targets.shape}, kappa={kappa.shape}, phi_p={phi_p.shape}"
    )

    try:
        chunk_target_count = stop_index - start_index
        if targets.shape != (chunk_target_count, config.n_lasers):
            raise ValueError("target chunk shape does not match its index range")
        if (
            kappa.ndim != 4
            or kappa.shape[0] != chunk_target_count
            or kappa.shape[2:] != (config.n_lasers, config.n_lasers)
        ):
            raise ValueError("kappa chunk has an invalid shape")
        if phi_p.shape != kappa.shape:
            raise ValueError("phi_p chunk must match the kappa chunk")

        candidates_per_target = kappa.shape[1]
        number_of_systems = chunk_target_count * candidates_per_target
        flat_kappa = kappa.reshape(
            number_of_systems, config.n_lasers, config.n_lasers
        )
        flat_phi_p = phi_p.reshape(
            number_of_systems, config.n_lasers, config.n_lasers
        )
        repeated_targets = np.repeat(
            targets, candidates_per_target, axis=0
        )
        repeated_detunings = (
            None
            if detunings is None
            else np.repeat(detunings, candidates_per_target, axis=0)
        )

        # The expensive integrator remains vectorized within this process.
        simulation_kwargs = {}
        if repeated_detunings is not None:
            simulation_kwargs["detuning_distribution_ghz"] = (
                repeated_detunings
            )
        time_seconds, states, _, _ = simulate_with_vcsel(
            flat_kappa, flat_phi_p, config, **simulation_kwargs
        )
        reward = calculate_phase_reward(
            time_seconds, states, repeated_targets, config
        )
        scalar_shape = (chunk_target_count, candidates_per_target)
        phase_shape = (
            chunk_target_count,
            candidates_per_target,
            config.n_lasers,
        )
        return {
            "chunk_index": chunk_index,
            "start_index": start_index,
            "stop_index": stop_index,
            "phase_reward": reward[0].reshape(scalar_shape),
            "mean_phase_score": reward[1].reshape(scalar_shape),
            "worst_phase_score": reward[2].reshape(scalar_shape),
            "achieved_phases_rad": reward[3].reshape(phase_shape),
            "aligned_achieved_phases_rad": reward[4].reshape(phase_shape),
            "orientation": reward[5].reshape(scalar_shape),
        }
    except Exception as error:
        raise RuntimeError(
            f"VCSEL simulation failed for chunk {chunk_index}, "
            f"targets [{start_index}:{stop_index}); {chunk_shape_text}"
        ) from error


def simulate_indexed_target_chunk(
    indexed_work_item: tuple[int, int, int, tuple[Any, ...]],
) -> dict[str, int | np.ndarray]:
    """Simulate one target/candidate chunk and retain both index ranges."""
    (
        size_batch_index,
        candidate_start_index,
        candidate_stop_index,
        work_item,
    ) = indexed_work_item
    result = simulate_target_chunk(work_item)
    result["size_batch_index"] = size_batch_index
    result["candidate_start_index"] = candidate_start_index
    result["candidate_stop_index"] = candidate_stop_index
    return result


def simulate_candidates_parallel(
    targets_rad: np.ndarray,
    kappa_per_ns: np.ndarray,
    phi_p_rad: np.ndarray,
    config: DesignerConfig,
    *,
    detuning_distributions_ghz: np.ndarray | None = None,
    simulation_pool: Any | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate grouped candidates serially or with target-level workers.

    The returned arrays always retain target-major ordering, regardless of
    worker completion order. Scalar diagnostics have shape ``(T, C)`` and
    achieved phase arrays have shape ``(T, C, N)``.
    """
    targets_array = np.asarray(targets_rad)
    if detuning_distributions_ghz is None:
        # Draw once in the parent so serial and multiprocessing paths use
        # identical physical conditions. Workers then receive the same
        # per-target detuning vector for every candidate.
        detuning_distributions_ghz = sample_detuning_distributions(
            len(targets_array),
            np.random.default_rng(config.random_seed + 40_000),
            config,
        )
    work_items = split_simulation_batch(
        targets_rad,
        kappa_per_ns,
        phi_p_rad,
        config,
        detuning_distributions_ghz=detuning_distributions_ghz,
        n_jobs=config.n_jobs,
    )

    if config.n_jobs == 1:
        chunk_results = [simulate_target_chunk(work_items[0])]
    elif simulation_pool is not None:
        chunk_results = list(
            simulation_pool.imap_unordered(
                simulate_target_chunk, work_items
            )
        )
    else:
        # Public evaluation can be called outside train_reinforce. In that
        # case create one short-lived spawn pool. The training loop always
        # supplies its persistent pool and never creates a pool per iteration.
        active_jobs = min(config.n_jobs, len(work_items))
        spawn_context = mp.get_context("spawn")
        temporary_pool = spawn_context.Pool(
            processes=active_jobs,
            initializer=_initialize_simulation_worker,
        )
        try:
            chunk_results = list(
                temporary_pool.imap_unordered(
                    simulate_target_chunk, work_items
                )
            )
        finally:
            temporary_pool.close()
            temporary_pool.join()

    targets = np.asarray(targets_rad)
    kappa = np.asarray(kappa_per_ns)
    target_count = targets.shape[0]
    candidates_per_target = kappa.shape[1]
    scalar_shape = (target_count, candidates_per_target)
    phase_shape = (
        target_count,
        candidates_per_target,
        config.n_lasers,
    )
    combined = {
        "phase_reward": np.empty(scalar_shape, dtype=float),
        "mean_phase_score": np.empty(scalar_shape, dtype=float),
        "worst_phase_score": np.empty(scalar_shape, dtype=float),
        "achieved_phases_rad": np.empty(phase_shape, dtype=float),
        "aligned_achieved_phases_rad": np.empty(
            phase_shape, dtype=float
        ),
        "orientation": np.empty(scalar_shape, dtype=float),
    }
    target_coverage = np.zeros(target_count, dtype=int)
    for result in chunk_results:
        start_index = int(result["start_index"])
        stop_index = int(result["stop_index"])
        if np.any(target_coverage[start_index:stop_index]):
            raise RuntimeError(
                "parallel simulation returned a duplicate target range "
                f"[{start_index}:{stop_index})"
            )
        target_coverage[start_index:stop_index] += 1
        for key in combined:
            combined[key][start_index:stop_index] = result[key]

    if not np.all(target_coverage == 1):
        missing = np.flatnonzero(target_coverage == 0).tolist()
        raise RuntimeError(
            f"parallel simulation did not return targets {missing}"
        )
    return combined


# ---------------------------------------------------------------------------
# 7. One visible REINFORCE update
# ---------------------------------------------------------------------------


def run_training_batch(
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: DesignerConfig,
    rng: np.random.Generator,
    iteration: int,
    *,
    simulation_pool: Any | None = None,
) -> dict[str, float]:
    """Run one contextual-bandit batch and update the policy once."""
    policy.train()
    current_minimum_log_std = minimum_log_std_at(iteration, config)
    current_maximum_log_std = maximum_log_std_at(iteration, config)
    policy.minimum_log_std = current_minimum_log_std
    policy.maximum_log_std = current_maximum_log_std
    targets_rad = sample_target_phases(
        config.targets_per_batch,
        rng,
        config.n_lasers,
        config.structured_target_fraction,
        config.structured_target_jitter_rad,
    )
    detuning_distributions_ghz = sample_detuning_distributions(
        config.targets_per_batch, rng, config, iteration=iteration
    )
    targets_rad, detuning_distributions_ghz = sort_by_detuning(
        targets_rad, detuning_distributions_ghz
    )

    # targets/detunings: (T, M); node features: (T, M, 3)
    active_link_counts = None
    if config.enable_connected_edge_budgets:
        active_link_counts = sample_active_link_counts(
            config.targets_per_batch,
            config.n_lasers,
            rng,
            iteration,
            config,
        )
    elif config.enable_conditional_backbone_budget:
        active_link_counts = sample_conditional_link_counts(
            config.targets_per_batch,
            config.n_lasers,
            iteration,
        )
    node_features = encode_for_policy(
        targets_rad,
        policy,
        config,
        detuning_distributions_ghz,
        active_link_counts=active_link_counts,
    )
    if config.enable_connected_edge_budgets:
        actions, gates, log_probabilities = sample_connected_coupling_designs(
            policy,
            node_features,
            config.candidates_per_target,
            active_link_counts,
        )
    elif config.enable_learned_connected_sparsity:
        actions, gates, log_probabilities = (
            sample_learned_connected_coupling_designs(
                policy,
                node_features,
                config.candidates_per_target,
                include_dense_reference=(
                    config.include_dense_reference_candidate
                ),
            )
        )
    else:
        actions, log_probabilities = sample_coupling_designs(
            policy, node_features, config.candidates_per_target
        )
        gates = None

    # actions: (T, C, 2*N*(N-1)) -> flattened candidate actions
    actions_numpy = actions.detach().cpu().numpy()
    action_size = policy.action_size_for(config.n_lasers)
    flat_actions = actions_numpy.reshape(-1, action_size)
    flat_gates = (
        None
        if gates is None
        else gates.detach().cpu().numpy().reshape(
            -1, policy.link_counts(config.n_lasers)[1]
        )
    )
    flat_kappa_per_ns, flat_phi_p_rad = decode_action(
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
    if config.enable_conditional_backbone_budget:
        flat_kappa_per_ns, flat_phi_p_rad, _ = (
            prune_coupling_batch_to_link_budget(
                flat_kappa_per_ns,
                flat_phi_p_rad,
                np.repeat(
                    active_link_counts, config.candidates_per_target
                ),
            )
        )
    grouped_matrix_shape = (
        config.targets_per_batch,
        config.candidates_per_target,
        config.n_lasers,
        config.n_lasers,
    )
    grouped_kappa_per_ns = flat_kappa_per_ns.reshape(grouped_matrix_shape)
    grouped_phi_p_rad = flat_phi_p_rad.reshape(grouped_matrix_shape)
    simulation = simulate_candidates_parallel(
        targets_rad,
        grouped_kappa_per_ns,
        grouped_phi_p_rad,
        config,
        detuning_distributions_ghz=detuning_distributions_ghz,
        simulation_pool=simulation_pool,
    )
    phase_reward_matrix = simulation["phase_reward"]
    phase_reward = phase_reward_matrix.reshape(-1)
    magnitude_symmetry_penalty = normalized_magnitude_symmetry_penalty(
        flat_kappa_per_ns,
        effective_maximum_kappa_per_link(
            config.maximum_kappa_per_ns,
            config.n_lasers,
            config.normalize_incoming_coupling_by_degree,
        ),
    )

    # Keep the physical objective and engineering preference visibly separate.
    coupling_cost = normalized_squared_coupling_cost(
        flat_kappa_per_ns,
        effective_maximum_kappa_per_link(
            config.maximum_kappa_per_ns,
            config.n_lasers,
            config.normalize_incoming_coupling_by_degree,
        ),
    )
    coupling_cost_matrix = coupling_cost.reshape(
        config.targets_per_batch, config.candidates_per_target
    )
    sparsity_bonus_matrix = np.zeros_like(phase_reward_matrix)
    coupling_cost_bonus_matrix = np.zeros_like(phase_reward_matrix)
    successful = np.zeros_like(phase_reward_matrix, dtype=bool)
    phase_or_sparse_reward = phase_reward.copy()
    rank_advantages = None
    if config.enable_learned_connected_sparsity:
        active_fraction_matrix = flat_gates.reshape(
            config.targets_per_batch,
            config.candidates_per_target,
            -1,
        ).mean(axis=2)
        phase_reference = np.max(phase_reward_matrix, axis=1)
        if config.use_lexicographic_rank_advantages:
            rank_advantages, successful, _ = (
                lexicographic_resource_rank_advantages(
                    phase_reward_matrix,
                    active_fraction_matrix,
                    coupling_cost_matrix,
                    phase_reference=phase_reference,
                    phase_retention_tolerance=(
                        config.phase_retention_tolerance
                    ),
                )
            )
        else:
            (
                phase_or_sparse_reward_matrix,
                sparsity_bonus_matrix,
                coupling_cost_bonus_matrix,
                successful,
            ) = successful_sparse_reward_components(
                phase_reward_matrix,
                active_fraction_matrix,
                coupling_cost_matrix,
                n_lasers=config.n_lasers,
                phase_threshold=config.sparsity_phase_reward_threshold,
                sparsity_weight=config.successful_sparsity_reward_weight,
                coupling_cost_tiebreak_fraction=(
                    config.coupling_cost_tiebreak_fraction
                ),
                phase_reference=(
                    phase_reference
                    if config.use_relative_phase_retention
                    else None
                ),
                phase_retention_tolerance=(
                    config.phase_retention_tolerance
                ),
            )
            phase_or_sparse_reward = phase_or_sparse_reward_matrix.reshape(-1)
    else:
        successful = (
            phase_reward_matrix >= config.sparsity_phase_reward_threshold
        )
    training_reward = (
        phase_or_sparse_reward
        - config.magnitude_symmetry_weight * magnitude_symmetry_penalty
    )

    auxiliary_loss = None
    if config.enable_budget_error_selector:
        auxiliary_loss, _, _ = budget_error_selector_loss(
            policy,
            node_features,
            targets_rad,
            active_link_counts,
            simulation["aligned_achieved_phases_rad"],
            config.n_lasers,
            config,
            rng,
        )

    # Compare candidates only with candidates generated for the same target.
    if rank_advantages is None:
        reward_matrix = training_reward.reshape(
            config.targets_per_batch, config.candidates_per_target
        )
        advantages = reward_matrix - reward_matrix.mean(
            axis=1, keepdims=True
        )
        advantages /= reward_matrix.std(axis=1, keepdims=True) + 1.0e-8
        advantages_tensor = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=log_probabilities.device,
        )
        loss = -(log_probabilities * advantages_tensor).mean()
    else:
        advantages_tensor = torch.as_tensor(
            rank_advantages,
            dtype=torch.float32,
            device=log_probabilities.device,
        )
        loss = -(log_probabilities * advantages_tensor).mean()
    if auxiliary_loss is not None:
        loss = loss + auxiliary_loss

    optimizer.zero_grad()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config.gradient_clip
    )
    optimizer.step()
    with torch.no_grad():
        policy.log_std_kappa.clamp_(
            current_minimum_log_std,
            current_maximum_log_std,
        )
        policy.log_std_phi.clamp_(
            current_minimum_log_std,
            current_maximum_log_std,
        )
        mean_action_std = float(
            policy.distribution(node_features).stddev.mean().cpu()
        )

    return {
        "loss": float(loss.detach().cpu()),
        "phase_reward": float(np.mean(phase_reward)),
        "magnitude_symmetry_penalty": float(
            np.mean(magnitude_symmetry_penalty)
        ),
        "training_reward": float(np.mean(training_reward)),
        "sparsity_bonus": float(np.mean(sparsity_bonus_matrix)),
        "normalized_coupling_cost": float(np.mean(coupling_cost_matrix)),
        "coupling_cost_bonus": float(
            np.mean(coupling_cost_bonus_matrix)
        ),
        "phase_reference_reward": float(
            np.mean(np.max(phase_reward_matrix, axis=1))
        ),
        "phase_success_fraction": float(
            np.mean(successful)
        ),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "minimum_log_std": current_minimum_log_std,
        "maximum_log_std": current_maximum_log_std,
        "mean_action_std": mean_action_std,
        "n_lasers": float(config.n_lasers),
        "active_connection_fraction": (
            1.0
            if flat_gates is None
            else float(np.mean(flat_gates))
        ),
    }


def combine_task_gradients_pcgrad(
    task_gradients: torch.Tensor,
    maximum_task_norm: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Clip, conflict-project, and average flattened task gradients.

    Rows correspond to tasks (one array size here). Each row is clipped
    independently before PCGrad so one task cannot dominate merely because
    its raw gradient norm is much larger. PCGrad then removes only components
    that oppose another task gradient; no array size receives a fixed weight.
    """
    if task_gradients.ndim != 2 or task_gradients.shape[0] < 2:
        raise ValueError("task_gradients must have shape (tasks >= 2, values)")
    if maximum_task_norm <= 0.0:
        raise ValueError("maximum_task_norm must be positive")

    raw_norms = torch.linalg.vector_norm(task_gradients, dim=1)
    clip_scales = torch.clamp(
        maximum_task_norm / torch.clamp(raw_norms, min=1.0e-15),
        max=1.0,
    )
    clipped = task_gradients * clip_scales[:, None]
    clipped_norms = torch.linalg.vector_norm(clipped, dim=1)
    cosine_denominator = torch.clamp(
        clipped_norms[:, None] * clipped_norms[None, :],
        min=1.0e-15,
    )
    cosine = torch.clamp(
        (clipped @ clipped.T) / cosine_denominator,
        -1.0,
        1.0,
    )

    projected = clipped.clone()
    task_count = clipped.shape[0]
    for task_index in range(task_count):
        for other_index_value in rng.permutation(task_count):
            other_index = int(other_index_value)
            if other_index == task_index:
                continue
            other_gradient = clipped[other_index]
            dot_product = torch.dot(projected[task_index], other_gradient)
            if float(dot_product.detach().cpu()) < 0.0:
                other_squared_norm = torch.dot(
                    other_gradient, other_gradient
                ).clamp_min(1.0e-15)
                projected[task_index] = (
                    projected[task_index]
                    - dot_product / other_squared_norm * other_gradient
                )

    upper_triangle = torch.triu_indices(
        task_count,
        task_count,
        offset=1,
        device=task_gradients.device,
    )
    pairwise_cosines = cosine[
        upper_triangle[0], upper_triangle[1]
    ]
    diagnostics: dict[str, torch.Tensor | float] = {
        "raw_task_norms": raw_norms,
        "clipped_task_norms": clipped_norms,
        "gradient_cosine_matrix": cosine,
        "negative_gradient_pair_fraction": float(
            torch.mean((pairwise_cosines < 0.0).to(torch.float32)).cpu()
        ),
        "mean_gradient_cosine": float(pairwise_cosines.mean().cpu()),
    }
    return projected.mean(dim=0), diagnostics


def run_multisize_training_batch(
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: DesignerConfig,
    rng: np.random.Generator,
    iteration: int,
    *,
    simulation_pool: Any | None = None,
) -> dict[str, Any]:
    """Update once from a fixed total of targets spread across array sizes.

    The cyclic schedule is balanced over successive iterations. With 14
    targets and sizes M=2,...,10, every update contains every size once and
    five sizes receive one rotating extra target. Each target's candidates
    are split into small vectorized worker tasks for dynamic load balancing.
    """
    policy.train()
    current_minimum_log_std = minimum_log_std_at(iteration, config)
    current_maximum_log_std = maximum_log_std_at(iteration, config)
    policy.minimum_log_std = current_minimum_log_std
    policy.maximum_log_std = current_maximum_log_std
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
        detuning_distributions_ghz = sample_detuning_distributions(
            target_count,
            rng,
            size_config,
            iteration=iteration,
        )
        targets_rad, detuning_distributions_ghz = sort_by_detuning(
            targets_rad, detuning_distributions_ghz
        )
        active_link_counts = None
        active_link_count_matrix = None
        deterministic_candidate_mask = None
        reinforce_candidate_mask = None
        if config.enable_connected_edge_budgets:
            active_link_counts = sample_active_link_counts(
                target_count, n_lasers, rng, iteration, size_config
            )
        elif config.enable_budget_error_selector:
            active_link_count_matrix = sample_multibudget_candidate_counts(
                target_count,
                config.candidates_per_target,
                n_lasers,
                config.selector_budget_levels_per_target,
                rng,
                iteration,
            )
            if config.selector_use_deterministic_max_error:
                deterministic_candidate_mask = (
                    deterministic_budget_candidate_mask(
                        active_link_count_matrix
                    )
                )
                reinforce_candidate_mask = ~deterministic_candidate_mask
        elif config.enable_conditional_backbone_budget:
            active_link_counts = sample_conditional_link_counts(
                target_count,
                n_lasers,
                iteration,
            )
        context_link_counts = active_link_counts
        if active_link_count_matrix is not None:
            context_link_counts = np.full(
                target_count,
                n_lasers * (n_lasers - 1),
                dtype=np.int64,
            )
        node_features = encode_for_policy(
            targets_rad,
            policy,
            size_config,
            detuning_distributions_ghz,
            active_link_counts=context_link_counts,
        )
        action_node_features = node_features
        if active_link_count_matrix is not None:
            repeated_targets = np.repeat(
                targets_rad, config.candidates_per_target, axis=0
            )
            repeated_detunings = np.repeat(
                detuning_distributions_ghz,
                config.candidates_per_target,
                axis=0,
            )
            action_node_features = encode_for_policy(
                repeated_targets,
                policy,
                size_config,
                repeated_detunings,
                active_link_counts=active_link_count_matrix.reshape(-1),
            )
            action_distribution = policy.distribution(action_node_features)
            sampled_actions = action_distribution.sample()
            if deterministic_candidate_mask is not None:
                flat_deterministic_mask = torch.as_tensor(
                    deterministic_candidate_mask.reshape(-1),
                    dtype=torch.bool,
                    device=sampled_actions.device,
                )
                sampled_actions = sampled_actions.clone()
                sampled_actions[flat_deterministic_mask] = (
                    action_distribution.mean[flat_deterministic_mask]
                )
            actions = sampled_actions.reshape(
                target_count,
                config.candidates_per_target,
                policy.action_size_for(n_lasers),
            )
            log_probabilities = action_distribution.log_prob(
                sampled_actions.detach()
            ).mean(dim=1).reshape(
                target_count, config.candidates_per_target
            )
            gates = None
        elif config.enable_connected_edge_budgets:
            actions, gates, log_probabilities = (
                sample_connected_coupling_designs(
                    policy,
                    node_features,
                    config.candidates_per_target,
                    active_link_counts,
                )
            )
        elif config.enable_learned_connected_sparsity:
            actions, gates, log_probabilities = (
                sample_learned_connected_coupling_designs(
                    policy,
                    node_features,
                    config.candidates_per_target,
                    include_dense_reference=(
                        config.include_dense_reference_candidate
                    ),
                )
            )
        elif config.enable_learned_backbone_density:
            actions, sampled_rho, log_probabilities = (
                sample_coupling_designs_with_backbone_density(
                    policy,
                    node_features,
                    config.candidates_per_target,
                )
            )
            gates = None
        else:
            actions, log_probabilities = sample_coupling_designs(
                policy, node_features, config.candidates_per_target
            )
            gates = None
        actions_numpy = actions.detach().cpu().numpy()
        action_size = policy.action_size_for(n_lasers)
        flat_actions = actions_numpy.reshape(-1, action_size)
        flat_gates = (
            None
            if gates is None
            else gates.detach().cpu().numpy().reshape(
                -1, policy.link_counts(n_lasers)[1]
            )
        )
        flat_kappa_per_ns, flat_phi_p_rad = decode_action(
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
        realized_rho = None
        if config.enable_learned_backbone_density:
            flat_kappa_per_ns, flat_phi_p_rad, realized_rho = (
                prune_coupling_batch_to_backbone(
                    flat_kappa_per_ns,
                    flat_phi_p_rad,
                    sampled_rho.detach().cpu().numpy().reshape(-1),
                )
            )
        elif config.enable_conditional_backbone_budget:
            candidate_link_counts = (
                active_link_count_matrix.reshape(-1)
                if active_link_count_matrix is not None
                else np.repeat(
                    active_link_counts, config.candidates_per_target
                )
            )
            flat_kappa_per_ns, flat_phi_p_rad, realized_rho = (
                prune_coupling_batch_to_link_budget(
                    flat_kappa_per_ns,
                    flat_phi_p_rad,
                    candidate_link_counts,
                )
            )
        grouped_shape = (
            target_count,
            config.candidates_per_target,
            n_lasers,
            n_lasers,
        )
        grouped_kappa_per_ns = flat_kappa_per_ns.reshape(grouped_shape)
        grouped_phi_p_rad = flat_phi_p_rad.reshape(grouped_shape)
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
                grouped_kappa_per_ns[:, candidate_start:candidate_stop],
                grouped_phi_p_rad[:, candidate_start:candidate_stop],
                size_config,
                detuning_distributions_ghz=detuning_distributions_ghz,
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
                "config": size_config,
                "targets_rad": targets_rad,
                "node_features": node_features,
                "active_link_counts": (
                    active_link_count_matrix
                    if active_link_count_matrix is not None
                    else active_link_counts
                ),
                "deterministic_candidate_mask": (
                    deterministic_candidate_mask
                ),
                "reinforce_candidate_mask": reinforce_candidate_mask,
                "log_probabilities": log_probabilities,
                "flat_kappa_per_ns": flat_kappa_per_ns,
                "flat_gates": flat_gates,
                "realized_rho": realized_rho,
                "active_connection_fraction": (
                    1.0
                    if flat_gates is None
                    else float(np.mean(flat_gates))
                ),
                "chunk_results": [],
                "mean_action_std": float(
                    policy.distribution(action_node_features).stddev.mean()
                    .detach().cpu()
                ),
            }
        )

    # Start the expensive large-M chunks first. Shorter small-M chunks then
    # fill gaps near the end instead of leaving one large-array straggler.
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

    losses = []
    size_metrics: dict[int, dict[str, float]] = {}
    for prepared in prepared_batches:
        phase_reward_matrix = np.empty(
            (prepared["target_count"], config.candidates_per_target),
            dtype=float,
        )
        candidate_coverage = np.zeros(
            (prepared["target_count"], config.candidates_per_target),
            dtype=int,
        )
        aligned_phase_matrix = None
        if config.enable_budget_error_selector:
            aligned_phase_matrix = np.empty(
                (
                    prepared["target_count"],
                    config.candidates_per_target,
                    prepared["n_lasers"],
                ),
                dtype=float,
            )
        for result in prepared["chunk_results"]:
            start_index = int(result["start_index"])
            stop_index = int(result["stop_index"])
            candidate_start = int(result["candidate_start_index"])
            candidate_stop = int(result["candidate_stop_index"])
            phase_reward_matrix[
                start_index:stop_index, candidate_start:candidate_stop
            ] = result["phase_reward"]
            if aligned_phase_matrix is not None:
                aligned_phase_matrix[
                    start_index:stop_index,
                    candidate_start:candidate_stop,
                ] = result["aligned_achieved_phases_rad"]
            candidate_coverage[
                start_index:stop_index, candidate_start:candidate_stop
            ] += 1
        if not np.all(candidate_coverage == 1):
            raise RuntimeError(
                "multi-size simulation did not return every candidate exactly "
                f"once for M={prepared['n_lasers']}"
            )

        phase_reward = phase_reward_matrix.reshape(-1)
        symmetry_penalty = normalized_magnitude_symmetry_penalty(
            prepared["flat_kappa_per_ns"],
            effective_maximum_kappa_per_link(
                config.maximum_kappa_per_ns,
                prepared["n_lasers"],
                config.normalize_incoming_coupling_by_degree,
            ),
        )
        coupling_cost = normalized_squared_coupling_cost(
            prepared["flat_kappa_per_ns"],
            effective_maximum_kappa_per_link(
                config.maximum_kappa_per_ns,
                prepared["n_lasers"],
                config.normalize_incoming_coupling_by_degree,
            ),
        )
        coupling_cost_matrix = coupling_cost.reshape(
            prepared["target_count"], config.candidates_per_target
        )
        sparsity_bonus_matrix = np.zeros_like(phase_reward_matrix)
        coupling_cost_bonus_matrix = np.zeros_like(phase_reward_matrix)
        successful = np.zeros_like(phase_reward_matrix, dtype=bool)
        phase_or_sparse_reward = phase_reward.copy()
        rank_advantages = None
        if config.enable_learned_backbone_density:
            active_fraction_matrix = prepared["realized_rho"].reshape(
                prepared["target_count"], config.candidates_per_target
            )
            rank_advantages, successful, _ = (
                lexicographic_resource_rank_advantages(
                    phase_reward_matrix,
                    active_fraction_matrix,
                    coupling_cost_matrix,
                    phase_reference=np.max(phase_reward_matrix, axis=1),
                    phase_retention_tolerance=config.phase_retention_tolerance,
                )
            )
            phase_or_sparse_reward = phase_reward.copy()
        elif config.enable_learned_connected_sparsity:
            active_fraction_matrix = prepared[
                "flat_gates"
            ].reshape(
                prepared["target_count"],
                config.candidates_per_target,
                -1,
            ).mean(axis=2)
            phase_reference = np.max(phase_reward_matrix, axis=1)
            if config.use_lexicographic_rank_advantages:
                rank_advantages, successful, _ = (
                    lexicographic_resource_rank_advantages(
                        phase_reward_matrix,
                        active_fraction_matrix,
                        coupling_cost_matrix,
                        phase_reference=phase_reference,
                        phase_retention_tolerance=(
                            config.phase_retention_tolerance
                        ),
                    )
                )
            else:
                (
                    phase_or_sparse_reward_matrix,
                    sparsity_bonus_matrix,
                    coupling_cost_bonus_matrix,
                    successful,
                ) = successful_sparse_reward_components(
                    phase_reward_matrix,
                    active_fraction_matrix,
                    coupling_cost_matrix,
                    n_lasers=prepared["n_lasers"],
                    phase_threshold=config.sparsity_phase_reward_threshold,
                    sparsity_weight=config.successful_sparsity_reward_weight,
                    coupling_cost_tiebreak_fraction=(
                        config.coupling_cost_tiebreak_fraction
                    ),
                    phase_reference=(
                        phase_reference
                        if config.use_relative_phase_retention
                        else None
                    ),
                    phase_retention_tolerance=(
                        config.phase_retention_tolerance
                    ),
                )
                phase_or_sparse_reward = (
                    phase_or_sparse_reward_matrix.reshape(-1)
                )
        else:
            successful = (
                phase_reward_matrix
                >= config.sparsity_phase_reward_threshold
            )
        training_reward = (
            phase_or_sparse_reward
            - config.magnitude_symmetry_weight * symmetry_penalty
        )
        auxiliary_loss = None
        error_prediction_mae_deg = float("nan")
        selector_budget_mae = float("nan")
        if config.enable_budget_error_selector:
            (
                auxiliary_loss,
                error_prediction_mae_deg,
                selector_budget_mae,
            ) = budget_error_selector_loss(
                policy,
                prepared["node_features"],
                prepared["targets_rad"],
                prepared["active_link_counts"],
                aligned_phase_matrix,
                prepared["n_lasers"],
                config,
                rng,
                deterministic_candidate_mask=prepared[
                    "deterministic_candidate_mask"
                ],
            )
        if rank_advantages is None:
            reward_matrix = training_reward.reshape(
                prepared["target_count"], config.candidates_per_target
            )
            if config.enable_budget_error_selector:
                advantages = budget_group_standardized_advantages(
                    reward_matrix,
                    prepared["active_link_counts"],
                    candidate_mask=prepared["reinforce_candidate_mask"],
                )
            else:
                advantages = reward_matrix - reward_matrix.mean(
                    axis=1, keepdims=True
                )
                advantages /= (
                    reward_matrix.std(axis=1, keepdims=True) + 1.0e-8
                )
            advantages_tensor = torch.as_tensor(
                advantages,
                dtype=torch.float32,
                device=prepared["log_probabilities"].device,
            )
            reinforce_candidate_mask = prepared[
                "reinforce_candidate_mask"
            ]
            if reinforce_candidate_mask is None:
                size_loss = -(
                    prepared["log_probabilities"] * advantages_tensor
                ).mean()
            else:
                reinforce_mask_tensor = torch.as_tensor(
                    reinforce_candidate_mask,
                    dtype=torch.bool,
                    device=prepared["log_probabilities"].device,
                )
                size_loss = -(
                    prepared["log_probabilities"][reinforce_mask_tensor]
                    * advantages_tensor[reinforce_mask_tensor]
                ).mean()
        else:
            advantages_tensor = torch.as_tensor(
                rank_advantages,
                dtype=torch.float32,
                device=prepared["log_probabilities"].device,
            )
            size_loss = -(
                prepared["log_probabilities"] * advantages_tensor
            ).mean()
        if auxiliary_loss is not None:
            size_loss = size_loss + auxiliary_loss
        losses.append(size_loss)
        size_metrics[int(prepared["n_lasers"])] = {
            "phase_reward": float(np.mean(phase_reward)),
            "training_reward": float(np.mean(training_reward)),
            "sparsity_bonus": float(np.mean(sparsity_bonus_matrix)),
            "normalized_coupling_cost": float(
                np.mean(coupling_cost_matrix)
            ),
            "coupling_cost_bonus": float(
                np.mean(coupling_cost_bonus_matrix)
            ),
            "phase_reference_reward": float(
                np.mean(np.max(phase_reward_matrix, axis=1))
            ),
            "phase_success_fraction": float(
                np.mean(successful)
            ),
            "magnitude_symmetry_penalty": float(
                np.mean(symmetry_penalty)
            ),
            "loss": float(size_loss.detach().cpu()),
            "mean_action_std": prepared["mean_action_std"],
            "active_connection_fraction": prepared[
                "active_connection_fraction"
            ],
            "error_prediction_mae_deg": error_prediction_mae_deg,
            "selector_budget_mae": selector_budget_mae,
        }
        if prepared["realized_rho"] is not None:
            size_metrics[int(prepared["n_lasers"])][
                "active_connection_fraction"
            ] = float(np.mean(prepared["realized_rho"]))

    # Every array size contributes one equally weighted scalar loss. The
    # optional PCGrad path changes only the gradient-combination rule, not
    # the per-size REINFORCE objectives above.
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    gradient_diagnostics: dict[str, torch.Tensor | float] = {
        "negative_gradient_pair_fraction": float("nan"),
        "mean_gradient_cosine": float("nan"),
    }
    if config.gradient_combination_mode == "pcgrad":
        parameters = [
            parameter
            for parameter in policy.parameters()
            if parameter.requires_grad
        ]
        flattened_task_gradients = []
        for size_loss in losses:
            task_gradient = torch.autograd.grad(
                size_loss,
                parameters,
                allow_unused=True,
            )
            flattened_task_gradients.append(
                torch.cat(
                    [
                        torch.zeros_like(parameter).reshape(-1)
                        if gradient is None
                        else gradient.detach().reshape(-1)
                        for parameter, gradient in zip(
                            parameters, task_gradient
                        )
                    ]
                )
            )
        combined_gradient, gradient_diagnostics = (
            combine_task_gradients_pcgrad(
                torch.stack(flattened_task_gradients),
                config.per_size_gradient_clip,
                rng,
            )
        )
        offset = 0
        for parameter in parameters:
            parameter_size = parameter.numel()
            parameter.grad = combined_gradient[
                offset : offset + parameter_size
            ].reshape_as(parameter).clone()
            offset += parameter_size
        raw_task_norms = gradient_diagnostics["raw_task_norms"]
        clipped_task_norms = gradient_diagnostics["clipped_task_norms"]
        if not isinstance(raw_task_norms, torch.Tensor):
            raise RuntimeError("PCGrad did not return task gradient norms")
        if not isinstance(clipped_task_norms, torch.Tensor):
            raise RuntimeError("PCGrad did not return clipped gradient norms")
        for index, prepared in enumerate(prepared_batches):
            metrics = size_metrics[int(prepared["n_lasers"])]
            metrics["raw_gradient_norm"] = float(
                raw_task_norms[index].cpu()
            )
            metrics["clipped_gradient_norm"] = float(
                clipped_task_norms[index].cpu()
            )
    else:
        loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config.gradient_clip
    )
    optimizer.step()
    with torch.no_grad():
        policy.log_std_kappa.clamp_(
            current_minimum_log_std, current_maximum_log_std
        )
        policy.log_std_phi.clamp_(
            current_minimum_log_std, current_maximum_log_std
        )
        if config.enable_learned_backbone_density:
            policy.backbone_density_log_std.clamp_(
                current_minimum_log_std, current_maximum_log_std
            )

    def mean_size_metric(name: str) -> float:
        return float(np.mean([values[name] for values in size_metrics.values()]))

    return {
        "loss": float(loss.detach().cpu()),
        "phase_reward": mean_size_metric("phase_reward"),
        "training_reward": mean_size_metric("training_reward"),
        "sparsity_bonus": mean_size_metric("sparsity_bonus"),
        "normalized_coupling_cost": mean_size_metric(
            "normalized_coupling_cost"
        ),
        "coupling_cost_bonus": mean_size_metric("coupling_cost_bonus"),
        "phase_reference_reward": mean_size_metric(
            "phase_reference_reward"
        ),
        "phase_success_fraction": mean_size_metric(
            "phase_success_fraction"
        ),
        "magnitude_symmetry_penalty": mean_size_metric(
            "magnitude_symmetry_penalty"
        ),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "minimum_log_std": current_minimum_log_std,
        "maximum_log_std": current_maximum_log_std,
        "mean_action_std": mean_size_metric("mean_action_std"),
        "active_connection_fraction": mean_size_metric(
            "active_connection_fraction"
        ),
        "size_metrics": size_metrics,
        "negative_gradient_pair_fraction": float(
            gradient_diagnostics["negative_gradient_pair_fraction"]
        ),
        "mean_gradient_cosine": float(
            gradient_diagnostics["mean_gradient_cosine"]
        ),
    }


# ---------------------------------------------------------------------------
# 7. Deterministic evaluation, training loop, and small plots
# ---------------------------------------------------------------------------


def evaluate_policy(
    policy: PolicyNetwork,
    target_phases_rad: np.ndarray,
    config: DesignerConfig,
    *,
    detuning_distributions_ghz: np.ndarray | None = None,
    active_link_counts: np.ndarray | int | None = None,
    simulation_pool: Any | None = None,
) -> dict[str, np.ndarray]:
    """Simulate the deterministic policy mean for explicit conditions.

    Connected-budget policies default to the maximally sparse connected
    budget ``M-1``. Pass an explicit scalar or per-target vector to evaluate
    another exact link count.
    """
    targets = np.asarray(target_phases_rad, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    targets = make_relative_to_laser_1(targets)
    if targets.shape[1] != config.n_lasers:
        raise ValueError(
            f"target arrays must have shape (batch, {config.n_lasers})"
        )
    if detuning_distributions_ghz is None:
        detuning_distributions_ghz = sample_detuning_distributions(
            len(targets),
            np.random.default_rng(config.random_seed + 20_000),
            config,
            iteration=config.detuning_curriculum_iterations,
        )
    else:
        detuning_distributions_ghz = np.asarray(
            detuning_distributions_ghz, dtype=float
        )
        if detuning_distributions_ghz.ndim == 1:
            detuning_distributions_ghz = (
                detuning_distributions_ghz[None, :]
            )
        if detuning_distributions_ghz.shape != (
            len(targets),
            config.n_lasers,
        ):
            raise ValueError(
                "detuning_distributions_ghz must have shape "
                f"({len(targets)}, {config.n_lasers})"
            )
    targets, detuning_distributions_ghz = sort_by_detuning(
        targets, detuning_distributions_ghz
    )
    if (
        config.enable_connected_edge_budgets
        or config.enable_conditional_backbone_budget
    ):
        if active_link_counts is None:
            default_count = (
                config.n_lasers - 1
                if config.enable_connected_edge_budgets
                else config.n_lasers * (config.n_lasers - 1)
            )
            active_link_counts_array = np.full(
                len(targets), default_count, dtype=np.int64
            )
        else:
            active_link_counts_array = np.asarray(
                active_link_counts, dtype=np.int64
            )
            if active_link_counts_array.ndim == 0:
                active_link_counts_array = np.full(
                    len(targets), int(active_link_counts_array), dtype=np.int64
                )
            if active_link_counts_array.shape != (len(targets),):
                raise ValueError(
                    "active_link_counts must be scalar or have one value "
                    "per target"
                )
    else:
        active_link_counts_array = None

    policy.eval()
    with torch.no_grad():
        encoded = encode_for_policy(
            targets,
            policy,
            config,
            detuning_distributions_ghz,
            active_link_counts=active_link_counts_array,
        )
        actions = policy(encoded).detach().cpu().numpy()
        predicted_rho = None
        if config.enable_learned_backbone_density:
            predicted_rho = torch.sigmoid(
                policy.backbone_density_distribution(encoded).mean
            ).detach().cpu().numpy().reshape(-1)
        if getattr(policy, "enable_sparse_gates", False):
            gate_distribution = policy.gate_distribution(encoded)
            gate_probabilities = (
                gate_distribution.probs.detach().cpu().numpy()
            )
            if config.enable_connected_edge_budgets:
                magnitude_gates = deterministic_connected_gates(
                    gate_distribution.logits.detach().cpu().numpy(),
                    active_link_counts_array,
                    config.n_lasers,
                )
            elif config.enable_learned_connected_sparsity:
                magnitude_gates = deterministic_learned_connected_gates(
                    gate_distribution.logits.detach().cpu().numpy(),
                    config.n_lasers,
                    config.gate_probability_threshold,
                )
            else:
                magnitude_gates = (
                    policy.deterministic_gates(encoded).detach().cpu().numpy()
                )
        else:
            _, n_magnitude_links, _ = policy.link_counts(config.n_lasers)
            gate_probabilities = np.ones(
                (len(targets), n_magnitude_links), dtype=float
            )
            magnitude_gates = gate_probabilities.copy()
    kappa_per_ns, phi_p_rad = decode_action(
        actions,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
        config.balanced_coupling,
        magnitude_gates=magnitude_gates,
        normalize_incoming_coupling_by_degree=(
            config.normalize_incoming_coupling_by_degree
        ),
    )
    if config.enable_conditional_backbone_budget:
        kappa_per_ns, phi_p_rad, conditional_rho = (
            prune_coupling_batch_to_link_budget(
                kappa_per_ns,
                phi_p_rad,
                active_link_counts_array,
            )
        )
        predicted_rho = conditional_rho
        receivers, sources = directed_link_indices(config.n_lasers)
        magnitude_gates = (
            kappa_per_ns[:, receivers, sources] > 0.0
        ).astype(float)
    elif predicted_rho is not None:
        kappa_per_ns, phi_p_rad, predicted_rho = (
            prune_coupling_batch_to_backbone(
                kappa_per_ns, phi_p_rad, predicted_rho
            )
        )
        receivers, sources = directed_link_indices(config.n_lasers)
        magnitude_gates = (kappa_per_ns[:, receivers, sources] > 0.0).astype(float)
    simulation = simulate_candidates_parallel(
        targets,
        kappa_per_ns[:, None, :, :],
        phi_p_rad[:, None, :, :],
        config,
        detuning_distributions_ghz=detuning_distributions_ghz,
        simulation_pool=simulation_pool,
    )
    return {
        "targets_rad": targets,
        "detuning_distributions_ghz": detuning_distributions_ghz,
        "actions": actions,
        "kappa_per_ns": kappa_per_ns,
        "phi_p_rad": phi_p_rad,
        "gate_probabilities": gate_probabilities,
        "magnitude_gates": magnitude_gates,
        "predicted_rho": predicted_rho,
        "active_link_counts": np.sum(magnitude_gates, axis=1).astype(int),
        "normalized_coupling_budget": np.sum(
            kappa_per_ns, axis=(1, 2)
        )
        / (
            config.n_lasers
            * (config.n_lasers - 1)
            * effective_maximum_kappa_per_link(
                config.maximum_kappa_per_ns,
                config.n_lasers,
                config.normalize_incoming_coupling_by_degree,
            )
        ),
        "normalized_squared_coupling_cost": (
            normalized_squared_coupling_cost(
                kappa_per_ns,
                effective_maximum_kappa_per_link(
                    config.maximum_kappa_per_ns,
                    config.n_lasers,
                    config.normalize_incoming_coupling_by_degree,
                ),
            )
        ),
        "active_connection_fraction": np.mean(magnitude_gates, axis=1),
        "phase_reward": simulation["phase_reward"][:, 0],
        "mean_phase_score": simulation["mean_phase_score"][:, 0],
        "worst_phase_score": simulation["worst_phase_score"][:, 0],
        "achieved_phases_rad": simulation[
            "achieved_phases_rad"
        ][:, 0],
        "aligned_achieved_phases_rad": simulation[
            "aligned_achieved_phases_rad"
        ][:, 0],
        "orientation": simulation["orientation"][:, 0],
    }


def make_training_figure(
    config: DesignerConfig,
) -> tuple[plt.Figure, np.ndarray]:
    """Create one small persistent figure; no new figure is made per update."""
    plt.ion()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    if config.jupyter_mode:
        plt.close(figure)
    return figure, axes


def update_training_figure(
    figure: plt.Figure,
    axes: np.ndarray,
    history: dict[str, Any],
) -> None:
    """Update the two essential learning curves."""
    axes[0].clear()
    rank_mode = history.get("uses_lexicographic_rank_advantages", False)
    rho_mode = history.get("uses_learned_backbone_density", False)
    selector_mode = history.get("uses_budget_error_selector", False)
    if selector_mode:
        axes[0].plot(history["training_reward"], label="mean phase reward")
        axes[0].plot(
            history.get("normalized_sparsity_score", []),
            color="tab:green",
            linewidth=2.0,
            linestyle="--",
            label="normalized sparsity score",
        )
    elif rho_mode:
        axes[0].plot(history["phase_reward"], label="mean phase reward")
        axes[0].plot(
            history.get("active_connection_fraction", []),
            color="tab:green",
            linewidth=2.0,
            linestyle="--",
            label=r"mean realized $\rho$",
        )
    elif rank_mode:
        axes[0].plot(history["phase_reward"], label="mean sampled phase")
        axes[0].plot(
            history["phase_reference_reward"],
            label="best sampled phase per target",
        )
    else:
        axes[0].plot(history["training_reward"], label="total reward")
    axes[0].set(
        title=(
            "REINFORCE phase + sparsity training"
            if selector_mode
            else "REINFORCE phase + sparsity training"
            if rho_mode
            else "REINFORCE lexicographic-rank training"
            if rank_mode
            else "REINFORCE training"
        ),
        xlabel="iteration",
        ylabel=(
            "mean phase reward / normalized sparsity score"
            if selector_mode
            else "mean phase reward / retained-link fraction"
            if (rank_mode or rho_mode)
            else "mean total reward"
        ),
        ylim=(0.0, 1.0),
    )
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.25)

    axes[1].clear()
    for n_lasers, rewards in history["held_out_reward_by_m"].items():
        axes[1].plot(rewards, label=f"M={n_lasers}")
    axes[1].plot(
        history["held_out_reward"],
        color="black",
        linewidth=2.0,
        label="overall",
    )
    validation_title = "Fixed held-out targets by array size"
    validation_spans = history.get("validation_detuning_span_ghz", [])
    if (
        history.get("validation_follows_detuning_curriculum", False)
        and validation_spans
    ):
        validation_title = (
            "Held-out targets at current detuning span "
            f"($D_{{\\max}}={validation_spans[-1]:.2f}$ GHz)"
        )
    axes[1].set(
        title=validation_title,
        xlabel="iteration",
        ylabel="mean phase reward",
        ylim=(0.0, 1.0),
    )
    axes[1].legend(loc="lower left", ncol=2)
    axes[1].grid(alpha=0.25)
    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    plt.pause(0.001)


def train_reinforce(
    config: DesignerConfig,
    policy: PolicyNetwork | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    live_plot: bool = True,
) -> tuple[PolicyNetwork, torch.optim.Optimizer, dict[str, Any]]:
    """Train one policy using one randomly selected array size per update."""
    rng = set_random_seed(config.random_seed)
    device = torch.device(config.device)
    policy = make_policy(config).to(device) if policy is None else policy
    optimizer = (
        torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        if optimizer is None
        else optimizer
    )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = config.learning_rate
    # Generate held-out targets and full-span detuning templates once. When
    # curriculum-following validation is selected, only the physical scale of
    # each template changes; the phase targets, span quantiles, and relative
    # positions of the lasers remain fixed.
    held_out_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n_lasers in config.training_n_lasers:
        size_config = replace(config, n_lasers=n_lasers)
        held_out_targets = sample_target_phases(
            config.held_out_target_count,
            np.random.default_rng(config.random_seed + 10_000 + n_lasers),
            n_lasers,
            config.structured_target_fraction,
            config.structured_target_jitter_rad,
        )
        held_out_detunings = sample_detuning_distributions(
            config.held_out_target_count,
            np.random.default_rng(config.random_seed + 20_000 + n_lasers),
            size_config,
            iteration=config.detuning_curriculum_iterations,
        )
        held_out_sets[n_lasers] = sort_by_detuning(
            held_out_targets, held_out_detunings
        )

    history: dict[str, Any] = {
        "n_lasers": [],
        "phase_reward": [],
        "training_reward": [],
        "sparsity_bonus": [],
        "normalized_coupling_cost": [],
        "coupling_cost_bonus": [],
        "phase_reference_reward": [],
        "phase_success_fraction": [],
        "magnitude_symmetry_penalty": [],
        "loss": [],
        "gradient_norm": [],
        "minimum_log_std": [],
        "maximum_log_std": [],
        "mean_action_std": [],
        "active_connection_fraction": [],
        "negative_gradient_pair_fraction": [],
        "mean_gradient_cosine": [],
        "uses_lexicographic_rank_advantages": (
            config.use_lexicographic_rank_advantages
        ),
        "uses_learned_backbone_density": (
            config.enable_learned_backbone_density
        ),
        "uses_budget_error_selector": config.enable_budget_error_selector,
        "normalized_sparsity_score": [],
        "validation_follows_detuning_curriculum": (
            config.validation_follows_detuning_curriculum
        ),
        "validation_detuning_span_ghz": [],
        "held_out_reward": [],
        "held_out_reward_by_m": {
            n_lasers: [] for n_lasers in config.training_n_lasers
        },
        "held_out_rho": [],
        "held_out_rho_by_m": {
            n_lasers: [] for n_lasers in config.training_n_lasers
        },
    }
    latest_held_out_reward = np.nan
    latest_held_out_by_m = {
        n_lasers: np.nan for n_lasers in config.training_n_lasers
    }
    latest_held_out_rho = np.nan
    latest_normalized_sparsity_score = np.nan
    latest_sparsity_score_by_m = {
        n_lasers: np.nan for n_lasers in config.training_n_lasers
    }
    latest_held_out_rho_by_m = {
        n_lasers: np.nan for n_lasers in config.training_n_lasers
    }
    latest_validation_span_ghz = np.nan
    best_held_out_reward = -np.inf
    best_validation_span_ghz = -np.inf
    figure_and_axes = make_training_figure(config) if live_plot else None
    progress_figure_directory = None
    if live_plot:
        progress_figure_directory = RESULTS_DIR
        progress_figure_directory.mkdir(parents=True, exist_ok=True)
    simulation_pool = None
    if config.n_jobs == 1:
        target_description = (
            "targets spread across every M"
            if config.train_all_sizes_each_iteration
            else "same-size targets"
        )
        print(
            "VCSEL simulation: serial vectorized path for "
            f"{config.targets_per_batch} {target_description} per update"
        )
    else:
        active_training_jobs = min(config.n_jobs, config.targets_per_batch)
        print(
            f"VCSEL simulation: {active_training_jobs} worker processes "
            f"for {config.targets_per_batch} same-size targets per update"
        )
        spawn_context = mp.get_context("spawn")
        simulation_pool = spawn_context.Pool(
            processes=active_training_jobs,
            initializer=_initialize_simulation_worker,
        )

    try:
        for iteration in range(config.training_iterations):
            if iteration == config.learning_rate_switch_iteration:
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = config.learning_rate_after_switch

            active_training_sizes = active_training_n_lasers_at(
                iteration, config
            )
            if config.train_all_sizes_each_iteration:
                metrics = run_multisize_training_batch(
                    policy,
                    optimizer,
                    config,
                    rng,
                    iteration,
                    simulation_pool=simulation_pool,
                )
                selected_n_lasers: int | tuple[int, ...] = (
                    active_training_sizes
                )
                selected_n_lasers_text = "all"
            else:
                selection_probabilities = None
                if config.training_n_laser_weights is not None:
                    active_count = len(active_training_sizes)
                    selection_probabilities = np.asarray(
                        config.training_n_laser_weights[:active_count],
                        dtype=float,
                    )
                    selection_probabilities /= selection_probabilities.sum()
                selected_n_lasers = int(
                    rng.choice(
                        active_training_sizes,
                        p=selection_probabilities,
                    )
                )
                selected_n_lasers_text = str(selected_n_lasers)
                iteration_config = replace(
                    config, n_lasers=selected_n_lasers
                )
                metrics = run_training_batch(
                    policy,
                    optimizer,
                    iteration_config,
                    rng,
                    iteration,
                    simulation_pool=simulation_pool,
                )
            history["n_lasers"].append(selected_n_lasers)
            for key in (
                "phase_reward",
                "training_reward",
                "sparsity_bonus",
                "normalized_coupling_cost",
                "coupling_cost_bonus",
                "phase_reference_reward",
                "phase_success_fraction",
                "magnitude_symmetry_penalty",
                "loss",
                "gradient_norm",
                "minimum_log_std",
                "maximum_log_std",
                "mean_action_std",
            ):
                history[key].append(metrics[key])
            history["active_connection_fraction"].append(
                metrics.get("active_connection_fraction", 1.0)
            )
            history["negative_gradient_pair_fraction"].append(
                metrics.get("negative_gradient_pair_fraction", np.nan)
            )
            history["mean_gradient_cosine"].append(
                metrics.get("mean_gradient_cosine", np.nan)
            )

            should_validate = (
                iteration == 0
                or (iteration + 1) % config.validation_interval == 0
                or iteration + 1 == config.training_iterations
            )
            validation_half_span = (
                detuning_curriculum_half_span_at(iteration, config)
                if config.validation_follows_detuning_curriculum
                else 0.5 * config.detuning_span_ghz
            )
            validation_span_ghz = 2.0 * validation_half_span
            if should_validate:
                latest_validation_span_ghz = validation_span_ghz
                for n_lasers in config.training_n_lasers:
                    (
                        held_out_targets,
                        full_span_held_out_detunings,
                    ) = held_out_sets[n_lasers]
                    held_out_detunings = full_span_held_out_detunings
                    if config.validation_follows_detuning_curriculum:
                        held_out_detunings = (
                            full_span_held_out_detunings
                            * validation_half_span
                            / (0.5 * config.detuning_span_ghz)
                        )
                    validation = evaluate_policy(
                        policy,
                        held_out_targets,
                        replace(config, n_lasers=n_lasers),
                        detuning_distributions_ghz=held_out_detunings,
                        active_link_counts=(
                            np.rint(
                                np.linspace(
                                    n_lasers - 1,
                                    n_lasers * (n_lasers - 1),
                                    len(held_out_targets),
                                )
                            ).astype(np.int64)
                            if config.enable_conditional_backbone_budget
                            else None
                        ),
                        simulation_pool=simulation_pool,
                    )
                    latest_held_out_by_m[n_lasers] = float(
                        np.mean(validation["phase_reward"])
                    )
                    if config.enable_budget_error_selector:
                        with torch.no_grad():
                            selector_features = encode_for_policy(
                                held_out_targets,
                                policy,
                                replace(config, n_lasers=n_lasers),
                                held_out_detunings,
                                active_link_counts=np.full(
                                    len(held_out_targets),
                                    n_lasers * (n_lasers - 1),
                                    dtype=np.int64,
                                ),
                            )
                            tolerance = torch.full(
                                (len(held_out_targets),),
                                config.selector_validation_error_deg / 180.0,
                                dtype=torch.float32,
                                device=selector_features.device,
                            )
                            predicted_q = policy.predict_normalized_budget(
                                selector_features, tolerance
                            )
                            latest_sparsity_score_by_m[n_lasers] = float(
                                torch.mean(1.0 - predicted_q).cpu()
                            )
                    if validation["predicted_rho"] is not None:
                        latest_held_out_rho_by_m[n_lasers] = float(
                            np.mean(validation["predicted_rho"])
                        )
                latest_held_out_reward = float(
                    np.mean(list(latest_held_out_by_m.values()))
                )
                if config.enable_learned_backbone_density:
                    latest_held_out_rho = float(
                        np.mean(list(latest_held_out_rho_by_m.values()))
                    )
                if config.enable_budget_error_selector:
                    latest_normalized_sparsity_score = float(
                        np.mean(list(latest_sparsity_score_by_m.values()))
                    )
                # When validation difficulty follows the curriculum, prefer
                # any checkpoint evaluated at a wider span over checkpoints
                # from an easier stage. Once the final span is reached, select
                # the highest reward normally within that fixed difficulty.
                if validation_span_ghz > best_validation_span_ghz + 1.0e-12:
                    best_validation_span_ghz = validation_span_ghz
                    best_held_out_reward = -np.inf
                if latest_held_out_reward > best_held_out_reward:
                    best_held_out_reward = latest_held_out_reward
                    save_checkpoint(
                        config.best_checkpoint_file,
                        policy,
                        optimizer,
                        config,
                        iteration=iteration + 1,
                        held_out_reward=best_held_out_reward,
                    )
            history["held_out_reward"].append(latest_held_out_reward)
            history["held_out_rho"].append(latest_held_out_rho)
            history["normalized_sparsity_score"].append(
                latest_normalized_sparsity_score
            )
            history["validation_detuning_span_ghz"].append(
                latest_validation_span_ghz
            )
            for n_lasers in config.training_n_lasers:
                history["held_out_reward_by_m"][n_lasers].append(
                    latest_held_out_by_m[n_lasers]
                )
                history["held_out_rho_by_m"][n_lasers].append(
                    latest_held_out_rho_by_m[n_lasers]
                )

            # Overwrite one atomic snapshot every iteration so an external
            # analysis can safely inspect the live policy while training.
            save_checkpoint(
                config.current_checkpoint_file,
                policy,
                optimizer,
                config,
                iteration=iteration + 1,
                held_out_reward=latest_held_out_reward,
            )

            if config.jupyter_mode:
                clear_output(wait=True)
            objective_text = (
                f"reference={metrics['phase_reference_reward']:.3f}"
                if (
                    config.use_lexicographic_rank_advantages
                    or config.enable_learned_backbone_density
                )
                else f"training={metrics['training_reward']:.3f}"
            )
            print(
                f"Iteration {iteration + 1:4d}/"
                f"{config.training_iterations} | "
                f"M={selected_n_lasers_text} | "
                f"active-M={active_training_sizes[0]}-"
                f"{active_training_sizes[-1]} | "
                f"phase={metrics['phase_reward']:.3f} | "
                f"{objective_text} | "
                f"held-out={latest_held_out_reward:.3f} | "
                f"K-sym={metrics['magnitude_symmetry_penalty']:.3f} | "
                f"K-cost={metrics['normalized_coupling_cost']:.3f} | "
                f"action-std={metrics['mean_action_std']:.3f} | "
                f"active={metrics.get('active_connection_fraction', 1.0):.3f} | "
                f"{'eligible' if config.use_relative_phase_retention else 'success'}="
                f"{metrics.get('phase_success_fraction', 0.0):.3f} | "
                f"std-floor={np.exp(metrics['minimum_log_std']):.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e} | "
                f"loss={metrics['loss']:.3f}"
            )
            if config.gradient_combination_mode == "pcgrad":
                print(
                    "PCGrad | conflicting pairs="
                    f"{metrics['negative_gradient_pair_fraction']:.3f} | "
                    "mean cosine="
                    f"{metrics['mean_gradient_cosine']:+.3f}"
                )
            if should_validate:
                validation_text = " | ".join(
                    f"M={n_lasers}: {latest_held_out_by_m[n_lasers]:.3f}"
                    for n_lasers in config.training_n_lasers
                )
                print(
                    f"Validation | D_max={validation_span_ghz:.3f} GHz | "
                    f"{validation_text} | overall: "
                    f"{latest_held_out_reward:.3f}"
                    + (
                        f" | mean-rho={latest_held_out_rho:.3f}"
                        if config.enable_learned_backbone_density
                        else ""
                    )
                )

            should_plot = (
                live_plot
                and (
                    iteration == 0
                    or (
                        (iteration + 1)
                        % config.plot_update_interval
                        == 0
                    )
                    or iteration + 1 == config.training_iterations
                )
            )
            if figure_and_axes is not None:
                update_training_figure(
                    figure_and_axes[0], figure_and_axes[1], history
                )
                figure_and_axes[0].savefig(
                    progress_figure_directory
                    / (
                        f"{Path(config.current_checkpoint_file).stem}_"
                        "training_progress.png"
                    ),
                    dpi=150,
                )
                if should_plot and config.jupyter_mode:
                    display(figure_and_axes[0])

        save_checkpoint(
            config.final_checkpoint_file,
            policy,
            optimizer,
            config,
            iteration=config.training_iterations,
            held_out_reward=latest_held_out_reward,
        )
    finally:
        if simulation_pool is not None:
            simulation_pool.close()
            simulation_pool.join()
    return policy, optimizer, history


def plot_training_history(
    history: dict[str, Any],
) -> plt.Figure:
    """Plot final training and validation curves outside the core loop."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    update_training_figure(figure, axes, history)
    return figure


# ---------------------------------------------------------------------------
# 8. Candidate design, validation, saving, and loading
# ---------------------------------------------------------------------------


def predict_conditional_link_budget(
    policy: GNNPolicyNetwork,
    target_phases_rad: np.ndarray,
    detuning_distribution_ghz: np.ndarray,
    config: DesignerConfig,
    allowable_rms_phase_error_deg: float,
) -> dict[str, float | int]:
    """Predict one connected-link budget from the configured error bound.

    The legacy argument name is retained for checkpoint and notebook
    compatibility. For deterministic maximum-error selector checkpoints, the
    value is interpreted as a maximum per-laser circular phase error.
    """
    if not config.enable_budget_error_selector:
        raise ValueError("checkpoint does not contain a budget selector")
    if not 0.0 < allowable_rms_phase_error_deg <= 180.0:
        metric_name = (
            "maximum"
            if config.selector_use_deterministic_max_error
            else "RMS"
        )
        raise ValueError(
            f"allowable {metric_name} phase error must lie in (0, 180]"
        )
    target = make_relative_to_laser_1(target_phases_rad)
    detuning = np.asarray(detuning_distribution_ghz, dtype=float)
    target, detuning = sort_by_detuning(target, detuning)
    maximum_links = config.n_lasers * (config.n_lasers - 1)
    minimum_links = config.n_lasers - 1
    encoded = encode_for_policy(
        target,
        policy,
        config,
        detuning,
        active_link_counts=np.asarray([maximum_links], dtype=np.int64),
    )
    policy.eval()
    with torch.no_grad():
        predicted_q = float(
            policy.predict_normalized_budget(
                encoded,
                torch.as_tensor(
                    [allowable_rms_phase_error_deg / 180.0],
                    dtype=torch.float32,
                    device=encoded.device,
                ),
            )[0].cpu()
        )
    active_link_count = minimum_links + int(
        round(predicted_q * (maximum_links - minimum_links))
    )
    active_link_count = int(
        np.clip(active_link_count, minimum_links, maximum_links)
    )
    return {
        "normalized_budget_q": predicted_q,
        "active_link_count": active_link_count,
        "rho": active_link_count / maximum_links,
        "normalized_sparsity_score": 1.0 - predicted_q,
    }


def design_for_target(
    policy: PolicyNetwork,
    target_phases_rad: np.ndarray,
    config: DesignerConfig,
    *,
    number_of_candidates: int = 128,
    top_k: int = 1,
    detuning_distribution_ghz: np.ndarray | None = None,
    active_link_count: int | None = None,
    include_deterministic_candidate: bool = True,
    mirror_if_lower_triangle_disabled: bool = False,
    simulation_pool: Any | None = None,
) -> list[dict[str, np.ndarray | float | int | bool | None]]:
    """Sample designs for one target condition and return the best."""
    target = make_relative_to_laser_1(target_phases_rad)
    if target.shape != (config.n_lasers,):
        raise ValueError(
            "target_phases_rad must have shape "
            f"({config.n_lasers},)"
        )
    if number_of_candidates < 1 or top_k < 1:
        raise ValueError("candidate counts must be positive")
    if detuning_distribution_ghz is None:
        detuning_distribution_ghz = sample_detuning_distributions(
            1,
            np.random.default_rng(config.random_seed + 30_000),
            config,
            iteration=config.detuning_curriculum_iterations,
        )[0]
    else:
        detuning_distribution_ghz = np.asarray(
            detuning_distribution_ghz, dtype=float
        )
        if detuning_distribution_ghz.shape != (config.n_lasers,):
            raise ValueError(
                "detuning_distribution_ghz must have shape "
                f"({config.n_lasers},)"
            )
    target, detuning_distribution_ghz = sort_by_detuning(
        target, detuning_distribution_ghz
    )
    if (
        config.enable_connected_edge_budgets
        or config.enable_conditional_backbone_budget
    ):
        if active_link_count is None:
            active_link_count = config.n_lasers * (config.n_lasers - 1)
        active_link_counts = np.asarray([active_link_count], dtype=np.int64)
    else:
        active_link_counts = None

    policy.eval()
    with torch.no_grad():
        encoded = encode_for_policy(
            target,
            policy,
            config,
            detuning_distribution_ghz,
            active_link_counts=active_link_counts,
        )
        distribution = policy.distribution(encoded)
        if config.enable_connected_edge_budgets:
            sampled_actions, sampled_gates, _ = (
                sample_connected_coupling_designs(
                    policy,
                    encoded,
                    number_of_candidates,
                    active_link_counts,
                )
            )
            actions = sampled_actions[0]
            magnitude_gates = sampled_gates[0]
            if include_deterministic_candidate:
                actions[0] = distribution.mean[0]
                gate_logits = policy.gate_distribution(encoded).logits
                deterministic_gate = deterministic_connected_gates(
                    gate_logits.detach().cpu().numpy(),
                    active_link_counts,
                    config.n_lasers,
                )[0]
                magnitude_gates[0] = torch.as_tensor(
                    deterministic_gate,
                    dtype=magnitude_gates.dtype,
                    device=magnitude_gates.device,
                )
        elif config.enable_learned_connected_sparsity:
            sampled_actions, sampled_gates, _ = (
                sample_learned_connected_coupling_designs(
                    policy,
                    encoded,
                    number_of_candidates,
                )
            )
            actions = sampled_actions[0]
            magnitude_gates = sampled_gates[0]
            if include_deterministic_candidate:
                actions[0] = distribution.mean[0]
                deterministic_gate = deterministic_learned_connected_gates(
                    policy.gate_distribution(encoded)
                    .logits.detach().cpu().numpy(),
                    config.n_lasers,
                    config.gate_probability_threshold,
                )[0]
                magnitude_gates[0] = torch.as_tensor(
                    deterministic_gate,
                    dtype=magnitude_gates.dtype,
                    device=magnitude_gates.device,
                )
        else:
            actions = distribution.sample((number_of_candidates,))[:, 0, :]
            if include_deterministic_candidate:
                actions[0] = distribution.mean[0]
        predicted_rho = None
        if config.enable_learned_backbone_density:
            rho_distribution = policy.backbone_density_distribution(encoded)
            rho_logits = rho_distribution.sample(
                (number_of_candidates,)
            )[:, 0, 0]
            if include_deterministic_candidate:
                rho_logits[0] = rho_distribution.mean[0, 0]
            predicted_rho = torch.sigmoid(rho_logits).detach().cpu().numpy()
        if (
            getattr(policy, "enable_sparse_gates", False)
            and not config.enable_connected_edge_budgets
            and not config.enable_learned_connected_sparsity
        ):
            gate_distribution = policy.gate_distribution(encoded)
            magnitude_gates = gate_distribution.sample(
                (number_of_candidates,)
            )[:, 0, :]
            if include_deterministic_candidate:
                magnitude_gates[0] = policy.deterministic_gates(encoded)[0]
        elif (
            not config.enable_connected_edge_budgets
            and not config.enable_learned_connected_sparsity
        ):
            magnitude_gates = None
    actions_numpy = actions.detach().cpu().numpy()
    magnitude_gates_numpy = (
        None
        if magnitude_gates is None
        else magnitude_gates.detach().cpu().numpy()
    )
    kappa_per_ns, phi_p_rad = decode_action(
        actions_numpy,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
        config.balanced_coupling,
        magnitude_gates=magnitude_gates_numpy,
        normalize_incoming_coupling_by_degree=(
            config.normalize_incoming_coupling_by_degree
        ),
    )
    realized_rho = None
    if predicted_rho is not None:
        kappa_per_ns, phi_p_rad, realized_rho = (
            prune_coupling_batch_to_backbone(
                kappa_per_ns, phi_p_rad, predicted_rho
            )
        )
        receivers, sources = directed_link_indices(config.n_lasers)
        magnitude_gates_numpy = (
            kappa_per_ns[:, receivers, sources] > 0.0
        ).astype(float)
    elif config.enable_conditional_backbone_budget:
        candidate_counts = np.full(
            number_of_candidates, active_link_count, dtype=np.int64
        )
        kappa_per_ns, phi_p_rad, realized_rho = (
            prune_coupling_batch_to_link_budget(
                kappa_per_ns, phi_p_rad, candidate_counts
            )
        )
        predicted_rho = realized_rho.copy()
        receivers, sources = directed_link_indices(config.n_lasers)
        magnitude_gates_numpy = (
            kappa_per_ns[:, receivers, sources] > 0.0
        ).astype(float)
    mirrored_about_diagonal = np.zeros(number_of_candidates, dtype=bool)
    if mirror_if_lower_triangle_disabled:
        for candidate_index in range(number_of_candidates):
            upper_kappa = np.triu(kappa_per_ns[candidate_index], k=1)
            lower_kappa = np.tril(kappa_per_ns[candidate_index], k=-1)
            if np.sum(upper_kappa) >= np.sum(lower_kappa):
                upper_kappa = np.triu(kappa_per_ns[candidate_index], k=1)
                upper_phi = np.triu(phi_p_rad[candidate_index], k=1)
                kappa_per_ns[candidate_index] = upper_kappa + upper_kappa.T
                phi_p_rad[candidate_index] = upper_phi + upper_phi.T
            else:
                lower_kappa = np.tril(kappa_per_ns[candidate_index], k=-1)
                lower_phi = np.tril(phi_p_rad[candidate_index], k=-1)
                kappa_per_ns[candidate_index] = lower_kappa + lower_kappa.T
                phi_p_rad[candidate_index] = lower_phi + lower_phi.T
            mirrored_about_diagonal[candidate_index] = True
        if np.any(mirrored_about_diagonal):
            receivers, sources = directed_link_indices(config.n_lasers)
            magnitude_gates_numpy = (
                kappa_per_ns[:, receivers, sources] > 0.0
            ).astype(float)
            realized_rho = np.mean(magnitude_gates_numpy, axis=1)
    simulation = simulate_candidates_parallel(
        target[None, :],
        kappa_per_ns[None, :, :, :],
        phi_p_rad[None, :, :, :],
        config,
        detuning_distributions_ghz=detuning_distribution_ghz[None, :],
        simulation_pool=simulation_pool,
    )
    magnitude_symmetry_penalty = normalized_magnitude_symmetry_penalty(
        kappa_per_ns,
        effective_maximum_kappa_per_link(
            config.maximum_kappa_per_ns,
            config.n_lasers,
            config.normalize_incoming_coupling_by_degree,
        ),
    )
    phase_reward = simulation["phase_reward"][0]
    coupling_cost = normalized_squared_coupling_cost(
        kappa_per_ns,
        effective_maximum_kappa_per_link(
            config.maximum_kappa_per_ns,
            config.n_lasers,
            config.normalize_incoming_coupling_by_degree,
        ),
    )
    ranking_reward = (
        phase_reward
        - config.magnitude_symmetry_weight
        * magnitude_symmetry_penalty
    )
    if config.enable_learned_backbone_density:
        active_fractions = realized_rho
        ranking_reward, _, _ = lexicographic_resource_rank_advantages(
            phase_reward[None, :],
            active_fractions[None, :],
            coupling_cost[None, :],
            phase_reference=np.array([np.max(phase_reward)]),
            phase_retention_tolerance=config.phase_retention_tolerance,
        )
        ranking_reward = ranking_reward[0]
    elif config.enable_learned_connected_sparsity:
        active_fractions = np.mean(magnitude_gates_numpy, axis=1)
        phase_reference = np.array([np.max(phase_reward)])
        if config.use_lexicographic_rank_advantages:
            ranking_reward, _, _ = (
                lexicographic_resource_rank_advantages(
                    phase_reward[None, :],
                    active_fractions[None, :],
                    coupling_cost[None, :],
                    phase_reference=phase_reference,
                    phase_retention_tolerance=(
                        config.phase_retention_tolerance
                    ),
                )
            )
            ranking_reward = ranking_reward[0]
        else:
            relative_reference = (
                phase_reference
                if config.use_relative_phase_retention
                else None
            )
            phase_or_sparse_reward, _, _, _ = (
                successful_sparse_reward_components(
                    phase_reward[None, :],
                    active_fractions[None, :],
                    coupling_cost[None, :],
                    n_lasers=config.n_lasers,
                    phase_threshold=config.sparsity_phase_reward_threshold,
                    sparsity_weight=config.successful_sparsity_reward_weight,
                    coupling_cost_tiebreak_fraction=(
                        config.coupling_cost_tiebreak_fraction
                    ),
                    phase_reference=relative_reference,
                    phase_retention_tolerance=(
                        config.phase_retention_tolerance
                    ),
                )
            )
            ranking_reward = phase_or_sparse_reward[0]
        ranking_reward = ranking_reward - (
            config.magnitude_symmetry_weight
            * magnitude_symmetry_penalty
        )
    order = np.argsort(ranking_reward)[::-1][: min(top_k, number_of_candidates)]

    results: list[dict[str, np.ndarray | float | int]] = []
    for rank, index in enumerate(order, start=1):
        results.append(
            {
                "rank": rank,
                "target_phases_rad": target.copy(),
                "detuning_distribution_ghz": (
                    detuning_distribution_ghz.copy()
                ),
                "kappa_per_ns": kappa_per_ns[index].copy(),
                "phi_p_rad": phi_p_rad[index].copy(),
                "magnitude_gates": (
                    np.ones(
                        policy.link_counts(config.n_lasers)[1], dtype=float
                    )
                    if magnitude_gates_numpy is None
                    else magnitude_gates_numpy[index].copy()
                ),
                "normalized_coupling_budget": float(
                    np.sum(kappa_per_ns[index])
                    / (
                        config.n_lasers
                        * (config.n_lasers - 1)
                        * effective_maximum_kappa_per_link(
                            config.maximum_kappa_per_ns,
                            config.n_lasers,
                            config.normalize_incoming_coupling_by_degree,
                        )
                    )
                ),
                "active_connection_fraction": float(
                    realized_rho[index]
                    if realized_rho is not None
                    else 1.0
                    if magnitude_gates_numpy is None
                    else np.mean(magnitude_gates_numpy[index])
                ),
                "active_link_count": int(
                    policy.link_counts(config.n_lasers)[1]
                    if magnitude_gates_numpy is None
                    else np.sum(magnitude_gates_numpy[index])
                ),
                "predicted_rho": (
                    None
                    if predicted_rho is None
                    else float(predicted_rho[index])
                ),
                "realized_rho": (
                    None
                    if realized_rho is None
                    else float(realized_rho[index])
                ),
                "mirrored_about_diagonal": bool(
                    mirrored_about_diagonal[index]
                ),
                "achieved_phases_rad": simulation[
                    "achieved_phases_rad"
                ][0, index].copy(),
                "aligned_achieved_phases_rad": simulation[
                    "aligned_achieved_phases_rad"
                ][0, index].copy(),
                "orientation": float(
                    simulation["orientation"][0, index]
                ),
                "phase_reward": float(phase_reward[index]),
                "normalized_squared_coupling_cost": float(
                    coupling_cost[index]
                ),
                "magnitude_symmetry_penalty": float(
                    magnitude_symmetry_penalty[index]
                ),
                "ranking_reward": float(ranking_reward[index]),
            }
        )
    return results


def design_sparsest_for_target(
    policy: PolicyNetwork,
    target_phases_rad: np.ndarray,
    config: DesignerConfig,
    *,
    minimum_phase_reward: float = 0.98,
    number_of_candidates: int = 128,
    detuning_distribution_ghz: np.ndarray | None = None,
    simulation_pool: Any | None = None,
) -> dict[str, np.ndarray | float | int | bool]:
    """Return the first exact edge budget that meets the phase threshold."""
    if not config.enable_connected_edge_budgets:
        raise ValueError(
            "sparsest-design search requires connected edge budgets"
        )
    if not 0.0 <= minimum_phase_reward <= 1.0:
        raise ValueError("minimum_phase_reward must lie in [0, 1]")
    maximum_links = config.n_lasers * (config.n_lasers - 1)
    last_result: dict[str, np.ndarray | float | int | bool] | None = None
    for active_link_count in range(config.n_lasers - 1, maximum_links + 1):
        result = dict(
            design_for_target(
                policy,
                target_phases_rad,
                config,
                number_of_candidates=number_of_candidates,
                top_k=1,
                detuning_distribution_ghz=detuning_distribution_ghz,
                active_link_count=active_link_count,
                simulation_pool=simulation_pool,
            )[0]
        )
        result["active_link_count"] = active_link_count
        satisfied = float(result["phase_reward"]) >= minimum_phase_reward
        result["phase_target_satisfied"] = satisfied
        last_result = result
        if satisfied:
            return result
    if last_result is None:
        raise RuntimeError("sparsest-design search did not evaluate a budget")
    return last_result


def validate_policy(
    policy: PolicyNetwork,
    config: DesignerConfig,
    *,
    candidates_per_target: int = 64,
    simulation_pool: Any | None = None,
) -> list[dict[str, np.ndarray | float | int | str]]:
    """Design and print one result for each named target."""
    rows: list[dict[str, np.ndarray | float | int | str]] = []
    print("target          phase reward  K-symmetry")
    print("-----------------------------------------")
    owned_pool = None
    if simulation_pool is None and config.n_jobs > 1:
        # Validation evaluates one named target at a time, so one worker keeps
        # all of that target's candidates vectorized in a single simulation.
        owned_pool = mp.get_context("spawn").Pool(
            processes=1,
            initializer=_initialize_simulation_worker,
        )
        simulation_pool = owned_pool
    try:
        validation_rng = np.random.default_rng(config.random_seed + 30_001)
        for name, target in named_phase_targets(config.n_lasers).items():
            detuning_distribution_ghz = sample_detuning_distributions(
                1, validation_rng, config
            )[0]
            result = design_for_target(
                policy,
                target,
                config,
                number_of_candidates=candidates_per_target,
                top_k=1,
                detuning_distribution_ghz=detuning_distribution_ghz,
                simulation_pool=simulation_pool,
            )[0]
            row = dict(result)
            row["name"] = name
            rows.append(row)
            print(
                f"{name:<15}{result['phase_reward']:>12.3f}"
                f"{result['magnitude_symmetry_penalty']:>12.3f}"
            )
    finally:
        if owned_pool is not None:
            owned_pool.close()
            owned_pool.join()
    return rows


def validate_policy_sizes(
    policy: PolicyNetwork,
    config: DesignerConfig,
    n_lasers_values: tuple[int, ...],
    *,
    candidates_per_target: int = 64,
) -> dict[int, list[dict[str, np.ndarray | float | int | str]]]:
    """Evaluate one unchanged policy on each requested array size."""
    results = {}
    for n_lasers in n_lasers_values:
        print(f"\nM={n_lasers}")
        results[n_lasers] = validate_policy(
            policy,
            replace(config, n_lasers=n_lasers),
            candidates_per_target=candidates_per_target,
        )
    return results


def save_checkpoint(
    filename: str | Path,
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: DesignerConfig,
    *,
    iteration: int,
    held_out_reward: float,
) -> Path:
    """Save a variable-size pooled, BiGRU, or GNN policy checkpoint."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "format": (
                GNN_CHECKPOINT_FORMAT
                if isinstance(policy, GNNPolicyNetwork)
                else (
                    (
                        BIGRU_MODULATED_CHECKPOINT_FORMAT
                        if policy.modulate_edge_decoder_by_n_lasers
                        else (
                            BIGRU_SPARSE_CHECKPOINT_FORMAT
                            if policy.enable_sparse_gates
                            else (
                                BIGRU_EDGE_STD_CHECKPOINT_FORMAT
                                if policy.edge_conditioned_log_std
                                else (
                                    BIGRU_M_CONDITIONED_CHECKPOINT_FORMAT
                                    if policy.condition_on_n_lasers
                                    or policy.condition_log_std_on_n_lasers
                                    else BIGRU_CHECKPOINT_FORMAT
                                )
                            )
                        )
                    )
                    if isinstance(policy, BiGRUPolicyNetwork)
                    else CHECKPOINT_FORMAT
                )
            ),
            "variable_m_policy": True,
            "encoder_architecture": (
                "gnn"
                if isinstance(policy, GNNPolicyNetwork)
                else (
                    "bigru"
                    if isinstance(policy, BiGRUPolicyNetwork)
                    else "pooled"
                )
            ),
            "training_n_lasers": tuple(config.training_n_lasers),
            "node_embedding_dim": int(config.node_embedding_dim),
            "global_embedding_dim": (
                int(policy.global_embedding_dim)
                if isinstance(policy, PolicyNetwork)
                else None
            ),
            "gru_hidden_size": int(config.gru_hidden_size),
            "config": asdict(config),
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": int(iteration),
            "held_out_reward": float(held_out_reward),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)
    return path


def load_checkpoint(
    filename: str | Path,
    *,
    device: str | None = None,
    architecture: str = "auto",
) -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    DesignerConfig,
    dict[str, float | int],
]:
    """Load a pooled, BiGRU, or graph-policy checkpoint.

    ``architecture`` may be ``"auto"`` (detect from checkpoint metadata),
    ``"pooled"``, ``"bigru"``, or ``"gnn"``.
    """
    if architecture not in {"auto", "pooled", "bigru", "gnn"}:
        raise ValueError(
            "architecture must be 'auto', 'pooled', 'bigru', or 'gnn'"
        )
    checkpoint = torch.load(
        filename,
        map_location=device or "cpu",
        weights_only=False,
    )
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in {
        CHECKPOINT_FORMAT,
        BIGRU_CHECKPOINT_FORMAT,
        BIGRU_M_CONDITIONED_CHECKPOINT_FORMAT,
        BIGRU_EDGE_STD_CHECKPOINT_FORMAT,
        BIGRU_SPARSE_CHECKPOINT_FORMAT,
        BIGRU_MODULATED_CHECKPOINT_FORMAT,
        GNN_CHECKPOINT_FORMAT,
    }:
        raise ValueError(
            "This checkpoint is not a variable-M pooled, BiGRU, or GNN "
            "policy. "
            "Fixed-size conditional-coupling checkpoints have incompatible "
            "encoder and edge-decoder shapes."
        )
    if not checkpoint.get("variable_m_policy", False):
        raise ValueError("Checkpoint is missing variable_m_policy=True")
    if checkpoint_format == GNN_CHECKPOINT_FORMAT:
        checkpoint_architecture = "gnn"
    elif checkpoint_format in {
            BIGRU_CHECKPOINT_FORMAT,
            BIGRU_M_CONDITIONED_CHECKPOINT_FORMAT,
            BIGRU_EDGE_STD_CHECKPOINT_FORMAT,
            BIGRU_SPARSE_CHECKPOINT_FORMAT,
            BIGRU_MODULATED_CHECKPOINT_FORMAT,
    }:
        checkpoint_architecture = "bigru"
    else:
        checkpoint_architecture = "pooled"
    if architecture != "auto" and architecture != checkpoint_architecture:
        raise ValueError(
            f"Requested {architecture} architecture, but checkpoint contains "
            f"the {checkpoint_architecture} architecture."
        )
    saved_config = dict(checkpoint["config"])
    saved_config["encoder_architecture"] = checkpoint_architecture
    if checkpoint_format == BIGRU_CHECKPOINT_FORMAT:
        # Archived BiGRU checkpoints predate explicit M conditioning and must
        # retain their original decoder and exploration parameter shapes.
        saved_config["condition_on_n_lasers"] = False
        saved_config["condition_log_std_on_n_lasers"] = False
    if "force_symmetric_coupling" in saved_config:
        old_forced_symmetry = bool(
            saved_config.pop("force_symmetric_coupling")
        )
        if old_forced_symmetry:
            raise ValueError(
                "This checkpoint used the older policy that forced both "
                "kappa and phi_p to be symmetric. Its action dimensions are "
                "incompatible with the new symmetric-kappa/directed-phi_p "
                "policy, so train a new model."
            )
        saved_config["force_symmetric_kappa"] = False
    # Checkpoints written before the constant-rate simplification contain
    # these now-unused schedule fields. Ignore them when loading old models.
    for legacy_field in (
        "learning_rate_decay_iteration",
        "learning_rate_after_decay",
        "final_learning_rate_iteration",
        "final_learning_rate",
        "frobenius_weight",
        "regularization_start_iteration",
        "regularization_ramp_iterations",
    ):
        saved_config.pop(legacy_field, None)
    config = DesignerConfig(**saved_config)
    if device is not None:
        config = replace(config, device=device)
    policy = make_policy(config).to(torch.device(config.device))
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.learning_rate
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = config.learning_rate
    metadata = {
        "iteration": int(checkpoint["iteration"]),
        "held_out_reward": float(checkpoint["held_out_reward"]),
    }
    return policy, optimizer, config, metadata


def add_sparse_gates_to_policy(
    policy: nn.Module,
    config: DesignerConfig,
    *,
    initial_gate_logit: float = 5.0,
    gate_probability_threshold: float = 0.5,
) -> tuple[nn.Module, DesignerConfig]:
    """Copy a dense BiGRU or GNN into a gate-enabled refinement policy."""
    if not isinstance(policy, (BiGRUPolicyNetwork, GNNPolicyNetwork)):
        raise ValueError(
            "sparse refinement requires a BiGRU or GNN policy"
        )
    if policy.enable_sparse_gates:
        return policy, config
    sparse_config = replace(
        config,
        enable_sparse_gates=True,
        initial_gate_logit=float(initial_gate_logit),
        gate_probability_threshold=float(gate_probability_threshold),
    )
    policy_type = type(policy)
    sparse_policy = policy_type(sparse_config).to(next(policy.parameters()).device)
    incompatible = sparse_policy.load_state_dict(
        policy.state_dict(), strict=False
    )
    expected_missing = {"edge_gate_head.weight", "edge_gate_head.bias"}
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "unexpected parameters were missing while adding sparse gates: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "unexpected dense-policy parameters while adding sparse gates: "
            f"{incompatible.unexpected_keys}"
        )
    sparse_policy.train(policy.training)
    return sparse_policy, sparse_config


__all__ = [
    "BIGRU_CHECKPOINT_FORMAT",
    "BIGRU_EDGE_STD_CHECKPOINT_FORMAT",
    "BIGRU_M_CONDITIONED_CHECKPOINT_FORMAT",
    "BIGRU_MODULATED_CHECKPOINT_FORMAT",
    "BIGRU_SPARSE_CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT",
    "GNN_CHECKPOINT_FORMAT",
    "DEFAULT_N_LASERS",
    "DesignerConfig",
    "PHASE_TICK_LABELS",
    "PHASE_TICKS",
    "PolicyNetwork",
    "BiGRUPolicyNetwork",
    "GNNPolicyNetwork",
    "GraphMessagePassingLayer",
    "active_training_n_lasers_at",
    "add_sparse_gates_to_policy",
    "action_size_for",
    "budget_error_selector_loss",
    "budget_group_standardized_advantages",
    "calculate_phase_reward",
    "canonicalize_global_phase_conjugate",
    "combine_task_gradients_pcgrad",
    "decode_action",
    "deterministic_budget_candidate_mask",
    "deterministic_connected_gates",
    "deterministic_learned_connected_gates",
    "design_for_target",
    "design_sparsest_for_target",
    "detuning_curriculum_half_span_at",
    "directed_link_indices",
    "encode_detuning_distributions",
    "encode_for_policy",
    "encode_target_phases",
    "effective_maximum_kappa_per_link",
    "evaluate_policy",
    "load_checkpoint",
    "lexicographic_resource_rank_advantages",
    "make_relative_to_laser_1",
    "make_policy",
    "make_vcsel_physical_parameters",
    "minimum_log_std_at",
    "minimum_active_links_at",
    "maximum_log_std_at",
    "named_phase_targets",
    "nonincreasing_isotonic_fit",
    "normalized_magnitude_symmetry_penalty",
    "normalized_squared_coupling_cost",
    "plot_training_history",
    "predict_conditional_link_budget",
    "run_training_batch",
    "run_multisize_training_batch",
    "run_architecture_sanity_checks",
    "sample_coupling_designs",
    "sample_connected_coupling_designs",
    "sample_learned_connected_coupling_designs",
    "sample_active_link_counts",
    "sample_conditional_link_counts",
    "sample_multibudget_candidate_counts",
    "sample_sparse_coupling_designs",
    "sample_detuning_distributions",
    "sample_target_phases",
    "sort_by_detuning",
    "successful_sparse_reward_components",
    "save_checkpoint",
    "set_random_seed",
    "simulate_candidates_parallel",
    "simulate_target_chunk",
    "simulate_indexed_target_chunk",
    "simulate_with_vcsel",
    "split_simulation_batch",
    "train_reinforce",
    "validate_policy",
    "validate_policy_sizes",
]
