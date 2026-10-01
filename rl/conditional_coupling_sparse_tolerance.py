"""Tolerance-conditioned connected sparse coupling for variable-size arrays.

This module is intentionally separate from the existing deterministic budget
selector experiment.  It reuses the same physical simulator, phase reward,
variable-M message passing, exact connected pruning, and PCGrad machinery, but
makes the normalized excess link budget ``q`` part of the Gaussian policy.

The external context is ``(target phases, detunings, allowable maximum phase
error)``.  A policy sample first chooses ``q`` and then chooses the continuous
coupling magnitudes and phases conditioned on that sampled budget.  Every
sample is converted to an exact directed-link count and pruned through the
existing maximum-weight-spanning-tree backbone before simulation.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from torch import nn

from rl import conditional_coupling_designer_variable_m as base
from rl.paths import RESULTS_DIR


SPARSE_TOLERANCE_CHECKPOINT_FORMAT = (
    "conditional_coupling_variable_m_gnn_sparse_tolerance_v1"
)


@dataclass
class SparseToleranceConfig(base.DesignerConfig):
    """Configuration for the isolated direct sparse-policy experiment."""

    # Stage boundaries.  Sparse training cannot begin before the detuning
    # curriculum has reached its configured final span.
    sparse_stage_start_iteration: int = 1000
    full_sparse_stage_start_iteration: int = 1300
    refinement_stage_start_iteration: int = 2200

    # Externally supplied maximum circular phase-error curriculum.
    dense_stage_tolerance_deg: float = 3.0
    tolerance_minimum_deg: float = 1.0
    mild_tolerance_maximum_deg: float = 3.0
    strict_tolerance_maximum_deg: float = 5.0
    tolerance_maximum_deg: float = 60.0
    strict_tolerance_fraction: float = 0.5
    tolerance_normalization_deg: float = 180.0

    # When enabled, separate each detuning vector into a bounded relative
    # pattern plus its physical peak-to-peak span.  This lets the GNN reuse
    # the same detuning-shape representation beyond the absolute span seen
    # during training while retaining the information needed to judge task
    # difficulty.  The default keeps existing checkpoints compatible.
    scale_aware_detuning_features: bool = False

    # Structured density exploration.  q=0 is a spanning tree and q=1 is
    # all-to-all.  The lower exploration bound falls from mild pruning to the
    # full range during the third stage.
    mild_minimum_q: float = 0.8
    initial_q_mean: float = 0.95
    initial_q_log_std: float = 0.0
    minimum_q_log_std: float = -2.0
    maximum_q_log_std: float = 0.75
    final_maximum_q_log_std: float = -0.5

    # Optional exponential learning-rate schedule.  The defaults retain the
    # existing constant/single-switch learning-rate behavior.
    learning_rate_final: float | None = None
    learning_rate_decay_start_iteration: int = 0
    learning_rate_decay_iterations: int = 0

    # Of 72 simulations per target, 64 are joint q/kappa/phi policy samples
    # and eight preserve dense phase-control competence.
    dense_replay_candidates: int = 8

    # The tiered reward is lexicographic by construction.  Tau shapes only
    # the recovery signal for infeasible candidates; it does not alter the
    # ordering among feasible candidates.
    infeasible_reward_tau_deg: float = 5.0

    # Fixed diagnostics.
    held_out_strict_tolerance_deg: float = 3.0
    tolerance_bin_edges_deg: tuple[float, ...] = (
        0.0,
        5.0,
        15.0,
        30.0,
        60.0,
        180.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.encoder_architecture != "gnn":
            raise ValueError("sparse tolerance training requires a GNN")
        if self.force_symmetric_kappa or self.force_symmetric_phi_p:
            raise ValueError(
                "sparse tolerance training currently requires directed "
                "kappa and phi_p actions"
            )
        if any(
            (
                self.enable_sparse_gates,
                self.enable_connected_edge_budgets,
                self.enable_learned_connected_sparsity,
                self.enable_learned_backbone_density,
                self.enable_conditional_backbone_budget,
                self.enable_budget_error_selector,
            )
        ):
            raise ValueError(
                "the direct sparse policy cannot be combined with the older "
                "gate, learned-rho, conditional-budget, or selector modes"
            )
        if self.sparse_stage_start_iteration < self.detuning_curriculum_iterations:
            raise ValueError(
                "sparse training must start after the detuning curriculum"
            )
        if not (
            self.sparse_stage_start_iteration
            <= self.full_sparse_stage_start_iteration
            <= self.refinement_stage_start_iteration
            <= self.training_iterations
        ):
            raise ValueError("sparse stage boundaries must be ordered")
        if not (
            0.0
            < self.tolerance_minimum_deg
            <= self.mild_tolerance_maximum_deg
            <= self.strict_tolerance_maximum_deg
            <= self.tolerance_maximum_deg
            <= 180.0
        ):
            raise ValueError("tolerance limits must be ordered within (0, 180]")
        if not 0.0 < self.dense_stage_tolerance_deg <= 180.0:
            raise ValueError("dense_stage_tolerance_deg must lie in (0, 180]")
        if not 0.0 < self.held_out_strict_tolerance_deg <= 180.0:
            raise ValueError(
                "held_out_strict_tolerance_deg must lie in (0, 180]"
            )
        if self.tolerance_normalization_deg <= 0.0:
            raise ValueError("tolerance_normalization_deg must be positive")
        if not 0.0 < self.strict_tolerance_fraction <= 1.0:
            raise ValueError("strict_tolerance_fraction must lie in (0, 1]")
        if not 0.0 <= self.mild_minimum_q < 1.0:
            raise ValueError("mild_minimum_q must lie in [0, 1)")
        if not 0.0 < self.initial_q_mean < 1.0:
            raise ValueError("initial_q_mean must lie in (0, 1)")
        if self.initial_q_mean <= self.mild_minimum_q:
            raise ValueError(
                "initial_q_mean must exceed mild_minimum_q"
            )
        if not (
            self.minimum_q_log_std
            <= self.initial_q_log_std
            <= self.maximum_q_log_std
        ):
            raise ValueError("initial_q_log_std must lie within its bounds")
        if not (
            self.minimum_q_log_std
            <= self.final_maximum_q_log_std
            <= self.maximum_q_log_std
        ):
            raise ValueError(
                "final_maximum_q_log_std must lie within its bounds"
            )
        if self.learning_rate_final is not None:
            if self.learning_rate_final <= 0.0:
                raise ValueError("learning_rate_final must be positive")
            if self.learning_rate_decay_start_iteration < 0:
                raise ValueError(
                    "learning_rate_decay_start_iteration must be nonnegative"
                )
            if self.learning_rate_decay_iterations < 1:
                raise ValueError(
                    "learning_rate_decay_iterations must be positive when "
                    "learning_rate_final is set"
                )
            if (
                self.learning_rate_decay_start_iteration
                + self.learning_rate_decay_iterations
                > self.training_iterations
            ):
                raise ValueError(
                    "learning-rate decay must finish within training_iterations"
                )
        if not 2 <= self.dense_replay_candidates <= self.candidates_per_target - 2:
            raise ValueError(
                "dense_replay_candidates must leave at least two joint "
                "candidates and contain at least two replay candidates"
            )
        if self.infeasible_reward_tau_deg <= 0.0:
            raise ValueError("infeasible_reward_tau_deg must be positive")
        edges = np.asarray(self.tolerance_bin_edges_deg, dtype=float)
        if len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("tolerance_bin_edges_deg must strictly increase")
        if edges[0] > self.tolerance_minimum_deg or edges[-1] < self.tolerance_maximum_deg:
            raise ValueError("tolerance bins must cover the sampled range")


class SparseToleranceGNNPolicy(nn.Module):
    """Hierarchical GNN policy for q followed by kappa/phi actions."""

    def __init__(self, config: SparseToleranceConfig):
        super().__init__()
        self.node_embedding_dim = config.node_embedding_dim
        self.minimum_training_n_lasers = min(config.training_n_lasers)
        self.maximum_training_n_lasers = max(config.training_n_lasers)
        self.condition_on_n_lasers = config.condition_on_n_lasers
        self.edge_conditioned_log_std = config.edge_conditioned_log_std
        self.smooth_edge_log_std_bounds = config.smooth_edge_log_std_bounds
        self.minimum_log_std = config.minimum_log_std
        self.maximum_log_std = config.maximum_log_std
        self.minimum_q_log_std = config.minimum_q_log_std
        self.maximum_q_log_std = config.maximum_q_log_std
        self.scale_aware_detuning_features = (
            config.scale_aware_detuning_features
        )

        node_layers: list[nn.Module] = []
        input_width = 5 if self.scale_aware_detuning_features else 4
        for output_width in config.node_hidden_sizes:
            layer = nn.Linear(input_width, output_width)
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            node_layers.extend((layer, nn.SiLU()))
            input_width = output_width
        output = nn.Linear(input_width, config.node_embedding_dim)
        nn.init.orthogonal_(output.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(output.bias)
        node_layers.extend((output, nn.SiLU()))
        self.node_encoder = nn.Sequential(*node_layers)

        self.message_layers = nn.ModuleList(
            base.GraphMessagePassingLayer(
                config.node_embedding_dim,
                config.gnn_message_hidden_size,
                edge_feature_width=4,
            )
            for _ in range(config.gnn_message_passing_steps)
        )

        edge_input_width = 2 * config.node_embedding_dim + 4 + 1
        if config.condition_on_n_lasers:
            edge_input_width += 2
        edge_layers: list[nn.Module] = []
        hidden_input_width = edge_input_width
        for output_width in config.edge_hidden_sizes:
            layer = nn.Linear(hidden_input_width, output_width)
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            edge_layers.extend((layer, nn.SiLU()))
            hidden_input_width = output_width
        edge_output_width = 4 if config.edge_conditioned_log_std else 2
        edge_output = nn.Linear(hidden_input_width, edge_output_width)
        nn.init.orthogonal_(edge_output.weight, gain=0.01)
        nn.init.zeros_(edge_output.bias)
        initial_fraction = config.initial_kappa_fraction
        initial_kappa_logit = np.log(initial_fraction / (1.0 - initial_fraction))
        with torch.no_grad():
            edge_output.bias[:2].copy_(
                torch.tensor([initial_kappa_logit, 0.0], dtype=edge_output.bias.dtype)
            )
            if config.edge_conditioned_log_std:
                edge_output.weight[2:].zero_()
                initial_raw = float(config.initial_log_std)
                if config.smooth_edge_log_std_bounds:
                    initial_raw = base.raw_log_std_for_smooth_bound(
                        config.initial_log_std,
                        config.minimum_log_std,
                        config.maximum_log_std,
                    )
                edge_output.bias[2:].fill_(initial_raw)
        edge_layers.append(edge_output)
        self.edge_decoder = nn.Sequential(*edge_layers)

        density_context_width = 2 * config.node_embedding_dim + 2
        self.density_head = nn.Sequential(
            nn.Linear(density_context_width, config.node_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.node_embedding_dim, 1),
        )
        for layer in self.density_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.density_head[-1].weight)
        # ``initial_q_mean`` names the realized mean at the beginning of the
        # mild-pruning stage.  Undo that stage's lower-bound transform before
        # initializing the unconstrained Gaussian-logit mean.
        initial_latent_fraction = (
            config.initial_q_mean - config.mild_minimum_q
        ) / (1.0 - config.mild_minimum_q)
        initial_latent_fraction = float(
            np.clip(initial_latent_fraction, 1.0e-4, 1.0 - 1.0e-4)
        )
        initial_q_logit = np.log(
            initial_latent_fraction / (1.0 - initial_latent_fraction)
        )
        nn.init.constant_(self.density_head[-1].bias, float(initial_q_logit))
        self.q_log_std = nn.Parameter(torch.tensor(float(config.initial_q_log_std)))

        self.log_std_kappa = nn.Parameter(torch.tensor(float(config.initial_log_std)))
        self.log_std_phi = nn.Parameter(torch.tensor(float(config.initial_log_std)))

    @staticmethod
    def bounded_degree_features(
        n_lasers: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        inverse = 1.0 / float(n_lasers - 1)
        return torch.tensor([inverse, inverse**2], device=device, dtype=dtype)

    def array_size_features(
        self, n_lasers: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        span = self.maximum_training_n_lasers - self.minimum_training_n_lasers
        normalized = (
            0.0
            if span == 0
            else 2.0 * (n_lasers - self.minimum_training_n_lasers) / span - 1.0
        )
        return torch.tensor(
            [normalized, 1.0 / float(n_lasers - 1)],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def link_counts(n_lasers: int) -> tuple[int, int, int]:
        directed = n_lasers * (n_lasers - 1)
        return directed, directed, directed

    @staticmethod
    def action_size_for(n_lasers: int) -> int:
        return 2 * n_lasers * (n_lasers - 1)

    def _graph_states(
        self, node_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_count = 5 if self.scale_aware_detuning_features else 4
        if node_features.ndim != 3 or node_features.shape[2] != feature_count:
            raise ValueError(
                "node_features must have shape "
                f"(batch, M, {feature_count})"
            )
        batch_size, n_lasers, _ = node_features.shape
        if n_lasers < 2:
            raise ValueError("node_features must contain at least two lasers")
        receiver_np, source_np = base.directed_link_indices(n_lasers)
        receivers = torch.as_tensor(
            receiver_np, dtype=torch.long, device=node_features.device
        )
        sources = torch.as_tensor(
            source_np, dtype=torch.long, device=node_features.device
        )
        cosine = node_features[:, :, 0]
        sine = node_features[:, :, 1]
        detuning = node_features[:, :, 2]
        local_cosine = (
            cosine[:, sources] * cosine[:, receivers]
            + sine[:, sources] * sine[:, receivers]
        )
        local_sine = (
            sine[:, sources] * cosine[:, receivers]
            - cosine[:, sources] * sine[:, receivers]
        )
        detuning_difference = 0.5 * (
            detuning[:, sources] - detuning[:, receivers]
        )
        edge_features = torch.stack(
            (
                local_cosine,
                local_sine,
                detuning_difference,
                torch.abs(detuning_difference),
            ),
            dim=2,
        )
        states = self.node_encoder(node_features)
        for layer in self.message_layers:
            states = layer(states, edge_features, receivers, sources)
        if states.shape[0] != batch_size:
            raise RuntimeError("GNN changed the batch dimension")
        return states, edge_features, receivers, sources

    def density_distribution(
        self, node_features: torch.Tensor
    ) -> torch.distributions.Normal:
        states, _, _, _ = self._graph_states(node_features)
        pooled = torch.cat((states.mean(dim=1), states.amax(dim=1)), dim=1)
        degree = self.bounded_degree_features(
            node_features.shape[1], device=pooled.device, dtype=pooled.dtype
        ).view(1, 2).expand(pooled.shape[0], 2)
        mean = self.density_head(torch.cat((pooled, degree), dim=1))
        log_std = torch.clamp(
            self.q_log_std, self.minimum_q_log_std, self.maximum_q_log_std
        )
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def _action_parameters(
        self, node_features: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        states, edge_features, receivers, sources = self._graph_states(node_features)
        batch_size = node_features.shape[0]
        n_lasers = node_features.shape[1]
        directed_links = n_lasers * (n_lasers - 1)
        q_column = q.to(states).reshape(-1, 1, 1)
        if q_column.shape[0] != batch_size:
            raise ValueError("q must contain one value per context")
        q_column = q_column.expand(batch_size, directed_links, 1)
        decoder_features = torch.cat((edge_features, q_column), dim=2)
        if self.condition_on_n_lasers:
            size = self.array_size_features(
                n_lasers, device=states.device, dtype=states.dtype
            ).view(1, 1, 2).expand(batch_size, directed_links, 2)
            decoder_features = torch.cat((decoder_features, size), dim=2)
        edge_inputs = torch.cat(
            (
                states.index_select(1, sources),
                states.index_select(1, receivers),
                decoder_features,
            ),
            dim=2,
        )
        flat = edge_inputs.reshape(batch_size * directed_links, -1)
        latent = self.edge_decoder[:-1](flat)
        outputs = self.edge_decoder[-1](latent).reshape(
            batch_size, directed_links, -1
        )
        kappa_mean = 6.0 * torch.tanh(outputs[:, :, 0] / 6.0)
        phase_mean = outputs[:, :, 1]
        means = torch.cat((kappa_mean, phase_mean), dim=1)
        if not self.edge_conditioned_log_std:
            return means, None
        return means, torch.cat((outputs[:, :, 2], outputs[:, :, 3]), dim=1)

    def action_distribution(
        self, node_features: torch.Tensor, q: torch.Tensor
    ) -> torch.distributions.Normal:
        mean, edge_log_std = self._action_parameters(node_features, q)
        if edge_log_std is not None:
            if self.smooth_edge_log_std_bounds:
                log_std = base.smoothly_bound_log_std(
                    edge_log_std, self.minimum_log_std, self.maximum_log_std
                )
            else:
                log_std = torch.clamp(
                    edge_log_std, self.minimum_log_std, self.maximum_log_std
                )
            return torch.distributions.Normal(mean, torch.exp(log_std))
        directed_links = node_features.shape[1] * (node_features.shape[1] - 1)
        log_std = torch.cat(
            (
                torch.clamp(
                    self.log_std_kappa,
                    self.minimum_log_std,
                    self.maximum_log_std,
                ).expand(directed_links),
                torch.clamp(
                    self.log_std_phi,
                    self.minimum_log_std,
                    self.maximum_log_std,
                ).expand(directed_links),
            )
        )
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def action_mean(self, node_features: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self._action_parameters(node_features, q)[0]


def encode_sparse_tolerance_context(
    targets_rad: np.ndarray,
    detuning_distributions_ghz: np.ndarray,
    allowable_error_deg: np.ndarray | float,
    policy: SparseToleranceGNNPolicy,
    config: SparseToleranceConfig,
) -> torch.Tensor:
    """Encode phase, detuning, and external tolerance as node features."""
    targets = np.asarray(targets_rad, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    if targets.shape[1] != config.n_lasers:
        raise ValueError(f"targets must have shape (batch, {config.n_lasers})")
    relative = (
        base.canonicalize_global_phase_conjugate(targets)
        if config.allow_global_phase_conjugate
        else base.make_relative_to_laser_1(targets)
    )
    raw_detuning = np.asarray(detuning_distributions_ghz, dtype=float)
    if raw_detuning.ndim == 1:
        raw_detuning = raw_detuning[None, :]
    if raw_detuning.shape != relative.shape:
        raise ValueError("targets and detunings must have matching shapes")
    if not np.all(np.isfinite(raw_detuning)):
        raise ValueError("detuning distributions must be finite")
    # All public training/evaluation/design paths sort targets alongside
    # detunings before calling this encoder.  Preserve that convention here.
    raw_detuning = np.sort(raw_detuning, axis=1)
    if config.scale_aware_detuning_features:
        detuning_minimum = raw_detuning.min(axis=1, keepdims=True)
        detuning_maximum = raw_detuning.max(axis=1, keepdims=True)
        detuning_span = detuning_maximum - detuning_minimum
        safe_span = np.maximum(detuning_span, 1.0e-12)
        # A normalized pattern has fixed endpoints -1 and 1 for every
        # nonzero-span array.  The separate span feature carries the physical
        # disorder scale in GHz.
        detuning = 2.0 * (raw_detuning - detuning_minimum) / safe_span - 1.0
        detuning = np.where(detuning_span > 1.0e-12, detuning, 0.0)
    else:
        detuning = base.encode_detuning_distributions(raw_detuning, config)
        if detuning.ndim == 1:
            detuning = detuning[None, :]
    if detuning.shape != relative.shape:
        raise ValueError("targets and detunings must have matching shapes")
    tolerance = np.asarray(allowable_error_deg, dtype=float)
    if tolerance.ndim == 0:
        tolerance = np.full(len(relative), float(tolerance))
    if tolerance.shape != (len(relative),):
        raise ValueError("allowable_error_deg must be scalar or one per target")
    if np.any(tolerance <= 0.0) or np.any(tolerance > 180.0):
        raise ValueError("allowable_error_deg must lie in (0, 180]")
    normalized_tolerance = tolerance / config.tolerance_normalization_deg
    feature_columns = [
        np.cos(relative),
        np.sin(relative),
        detuning,
        np.broadcast_to(normalized_tolerance[:, None], relative.shape),
    ]
    if config.scale_aware_detuning_features:
        feature_columns.append(
            np.broadcast_to(detuning_span, relative.shape)
        )
    features = np.stack(feature_columns, axis=2).astype(np.float32)
    return torch.as_tensor(
        features,
        dtype=torch.float32,
        device=next(policy.parameters()).device,
    )


def q_to_active_link_counts(q: np.ndarray | float, n_lasers: int) -> np.ndarray:
    """Map normalized excess budget q to exact connected directed counts."""
    q_array = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q_array)) or np.any(q_array < 0.0) or np.any(q_array > 1.0):
        raise ValueError("q must contain finite values in [0, 1]")
    minimum = n_lasers - 1
    maximum = n_lasers * (n_lasers - 1)
    counts = minimum + np.floor(q_array * (maximum - minimum) + 0.5).astype(int)
    return np.clip(counts, minimum, maximum)


def maximum_circular_phase_error_deg(
    target_phases_rad: np.ndarray,
    aligned_achieved_phases_rad: np.ndarray,
) -> np.ndarray:
    """Return settled maximum absolute circular error over nonreference lasers."""
    targets = base.make_relative_to_laser_1(target_phases_rad)
    achieved = np.asarray(aligned_achieved_phases_rad, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    if achieved.ndim == 2:
        achieved = achieved[:, None, :]
    if achieved.ndim != 3 or achieved.shape[0] != targets.shape[0] or achieved.shape[2] != targets.shape[1]:
        raise ValueError(
            "aligned_achieved_phases_rad must have shape (targets, candidates, M)"
        )
    error = np.angle(np.exp(1.0j * (achieved - targets[:, None, :])))
    return np.rad2deg(np.max(np.abs(error[:, :, 1:]), axis=2))


def tiered_sparse_reward(
    phase_reward: np.ndarray,
    maximum_error_deg: np.ndarray,
    allowable_error_deg: np.ndarray | float,
    q: np.ndarray,
    tau_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return constraint-first reward and feasibility mask.

    Infeasible designs always score below 0.5 and receive a shaped phase
    recovery signal.  Feasible designs score at least 0.5 and are ordered
    only by q, so every feasible reduction in link budget is preferred.
    """
    phase = np.asarray(phase_reward, dtype=float)
    error = np.asarray(maximum_error_deg, dtype=float)
    q_array = np.asarray(q, dtype=float)
    tolerance = np.asarray(allowable_error_deg, dtype=float)
    if tolerance.ndim == 0:
        tolerance = np.full(phase.shape[0], float(tolerance))
    if phase.shape != error.shape or phase.shape != q_array.shape:
        raise ValueError("phase_reward, maximum_error_deg, and q must match")
    if tolerance.shape != (phase.shape[0],):
        raise ValueError("allowable_error_deg must contain one value per target")
    if tau_deg <= 0.0:
        raise ValueError("tau_deg must be positive")
    finite_error = np.isfinite(error)
    feasible = finite_error & (error <= tolerance[:, None])
    excess = np.where(
        finite_error,
        np.maximum(error - tolerance[:, None], 0.0),
        np.inf,
    )
    recovery = 0.5 * np.clip(phase, 0.0, 1.0) * np.exp(-excess / tau_deg)
    feasible_reward = 0.5 + 0.5 * (1.0 - q_array)
    return np.where(feasible, feasible_reward, recovery), feasible


def training_stage_at(iteration: int, config: SparseToleranceConfig) -> str:
    if iteration < config.sparse_stage_start_iteration:
        return "dense"
    if iteration < config.full_sparse_stage_start_iteration:
        return "mild_sparse"
    if iteration < config.refinement_stage_start_iteration:
        return "full_sparse"
    return "refinement"


def minimum_q_at(iteration: int, config: SparseToleranceConfig) -> float:
    stage = training_stage_at(iteration, config)
    if stage == "dense":
        return 1.0
    if stage == "mild_sparse":
        return config.mild_minimum_q
    if stage == "refinement":
        return 0.0
    duration = max(
        config.refinement_stage_start_iteration
        - config.full_sparse_stage_start_iteration,
        1,
    )
    progress = np.clip(
        (iteration - config.full_sparse_stage_start_iteration) / duration,
        0.0,
        1.0,
    )
    return float((1.0 - progress) * config.mild_minimum_q)


def maximum_tolerance_at(iteration: int, config: SparseToleranceConfig) -> float:
    stage = training_stage_at(iteration, config)
    if stage == "dense":
        return config.dense_stage_tolerance_deg
    if stage == "mild_sparse":
        return config.mild_tolerance_maximum_deg
    if stage == "refinement":
        return config.tolerance_maximum_deg
    duration = max(
        config.refinement_stage_start_iteration
        - config.full_sparse_stage_start_iteration,
        1,
    )
    progress = np.clip(
        (iteration - config.full_sparse_stage_start_iteration) / duration,
        0.0,
        1.0,
    )
    return float(
        config.mild_tolerance_maximum_deg
        + progress
        * (config.tolerance_maximum_deg - config.mild_tolerance_maximum_deg)
    )


def sample_allowable_tolerances(
    count: int,
    iteration: int,
    config: SparseToleranceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one external tolerance per target with strict cases retained."""
    if count < 1:
        raise ValueError("count must be positive")
    stage = training_stage_at(iteration, config)
    if stage == "dense":
        return np.full(count, config.dense_stage_tolerance_deg)
    if stage == "mild_sparse":
        return rng.uniform(
            config.tolerance_minimum_deg,
            config.mild_tolerance_maximum_deg,
            size=count,
        )
    maximum = maximum_tolerance_at(iteration, config)
    strict_count = max(1, int(round(config.strict_tolerance_fraction * count)))
    tolerances = np.empty(count, dtype=float)
    tolerances[:strict_count] = rng.uniform(
        config.tolerance_minimum_deg,
        min(config.strict_tolerance_maximum_deg, maximum),
        size=strict_count,
    )
    if strict_count < count:
        lower = min(config.strict_tolerance_maximum_deg, maximum)
        tolerances[strict_count:] = rng.uniform(lower, maximum, size=count - strict_count)
    rng.shuffle(tolerances)
    return tolerances


def q_log_std_ceiling_at(iteration: int, config: SparseToleranceConfig) -> float:
    if iteration < config.refinement_stage_start_iteration:
        return config.maximum_q_log_std
    duration = max(config.training_iterations - config.refinement_stage_start_iteration, 1)
    progress = np.clip(
        (iteration - config.refinement_stage_start_iteration) / duration,
        0.0,
        1.0,
    )
    return float(
        (1.0 - progress) * config.maximum_q_log_std
        + progress * config.final_maximum_q_log_std
    )


def learning_rate_at(
    iteration: int, config: SparseToleranceConfig
) -> float | None:
    """Return the scheduled learning rate, or ``None`` without a schedule."""
    if config.learning_rate_final is None:
        return None
    duration = config.learning_rate_decay_iterations
    start = config.learning_rate_decay_start_iteration
    progress = np.clip(
        (iteration - start) / max(duration - 1, 1), 0.0, 1.0
    )
    return float(
        np.exp(
            np.log(config.learning_rate)
            + progress
            * (np.log(config.learning_rate_final) - np.log(config.learning_rate))
        )
    )


def _repeat_contexts(
    node_features: torch.Tensor, candidates: int
) -> torch.Tensor:
    targets, n_lasers, feature_count = node_features.shape
    return (
        node_features[:, None]
        .expand(targets, candidates, n_lasers, feature_count)
        .reshape(targets * candidates, n_lasers, feature_count)
    )


def sample_joint_sparse_candidates(
    policy: SparseToleranceGNNPolicy,
    node_features: torch.Tensor,
    candidates: int,
    minimum_q: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratify Gaussian q samples, then sample edge actions conditional on q."""
    if candidates < 2:
        raise ValueError("at least two joint candidates are required")
    density_distribution = policy.density_distribution(node_features)
    target_count = node_features.shape[0]
    offsets = torch.rand(
        (target_count, 1),
        device=node_features.device,
        dtype=node_features.dtype,
    )
    quantiles = (
        torch.arange(candidates, device=node_features.device, dtype=node_features.dtype)[None, :]
        + offsets
    ) / float(candidates)
    quantiles = quantiles.clamp(1.0e-5, 1.0 - 1.0e-5)
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, device=node_features.device),
        torch.tensor(1.0, device=node_features.device),
    ).icdf(quantiles)
    logits = (
        density_distribution.mean[:, 0, None]
        + density_distribution.stddev[:, 0, None] * standard_normal
    )
    q = minimum_q + (1.0 - minimum_q) * torch.sigmoid(logits.detach())
    repeated = _repeat_contexts(node_features, candidates)
    flat_q = q.reshape(-1)
    action_distribution = policy.action_distribution(repeated, flat_q)
    actions = action_distribution.sample().detach()
    action_log_probability = action_distribution.log_prob(actions).mean(dim=1)
    q_score_distribution = torch.distributions.Normal(
        density_distribution.mean[:, 0, None],
        density_distribution.stddev[:, 0, None],
    )
    q_log_probability = q_score_distribution.log_prob(logits.detach())
    joint_log_probability = (
        action_log_probability.reshape(target_count, candidates)
        + q_log_probability
    )
    return (
        actions.reshape(target_count, candidates, -1),
        q,
        joint_log_probability,
    )


def sample_edge_candidates_at_q(
    policy: SparseToleranceGNNPolicy,
    node_features: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample edge actions for an explicit target-by-candidate q matrix."""
    if q.ndim != 2 or q.shape[0] != node_features.shape[0]:
        raise ValueError("q must have shape (targets, candidates)")
    repeated = _repeat_contexts(node_features, q.shape[1])
    distribution = policy.action_distribution(repeated, q.reshape(-1))
    actions = distribution.sample().detach()
    log_probability = distribution.log_prob(actions).mean(dim=1)
    return (
        actions.reshape(q.shape[0], q.shape[1], -1),
        log_probability.reshape(q.shape),
    )


def _standardized_advantages(reward: np.ndarray) -> np.ndarray:
    centered = reward - reward.mean(axis=1, keepdims=True)
    return centered / (reward.std(axis=1, keepdims=True) + 1.0e-8)


def _decode_and_prune(
    actions: torch.Tensor,
    q: np.ndarray,
    config: SparseToleranceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_count, candidates, _ = actions.shape
    flat_actions = actions.detach().cpu().numpy().reshape(
        target_count * candidates, -1
    )
    kappa, phi = base.decode_action(
        flat_actions,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        False,
        False,
        config.balanced_coupling,
        normalize_incoming_coupling_by_degree=(
            config.normalize_incoming_coupling_by_degree
        ),
    )
    counts = q_to_active_link_counts(q.reshape(-1), config.n_lasers)
    kappa, phi, rho = base.prune_coupling_batch_to_link_budget(
        kappa, phi, counts
    )
    matrix_shape = (
        target_count,
        candidates,
        config.n_lasers,
        config.n_lasers,
    )
    return (
        kappa.reshape(matrix_shape),
        phi.reshape(matrix_shape),
        counts.reshape(target_count, candidates),
        rho.reshape(target_count, candidates),
    )


def _bin_metrics(
    tolerances: np.ndarray,
    reward: np.ndarray,
    feasible: np.ndarray,
    rho: np.ndarray,
    edges: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for lower, upper in zip(edges[:-1], edges[1:]):
        label = f"{lower:g}-{upper:g} deg"
        target_mask = (tolerances >= lower) & (
            tolerances <= upper if upper == edges[-1] else tolerances < upper
        )
        if not np.any(target_mask):
            result[label] = {"reward": np.nan, "satisfaction": np.nan, "rho": np.nan}
            continue
        result[label] = {
            "reward": float(np.mean(reward[target_mask])),
            "satisfaction": float(np.mean(feasible[target_mask])),
            "rho": float(np.mean(rho[target_mask])),
        }
    return result


def run_sparse_multisize_training_batch(
    policy: SparseToleranceGNNPolicy,
    optimizer: torch.optim.Optimizer,
    config: SparseToleranceConfig,
    rng: np.random.Generator,
    iteration: int,
    *,
    simulation_pool: Any | None = None,
) -> dict[str, Any]:
    """Run one all-M update and combine the per-M gradients with PCGrad.

    Every size is prepared before any simulation starts.  The resulting
    target chunks are submitted together, preserving the existing runner's
    ability to keep all workers busy across M=3,...,10.
    """
    policy.train()
    policy.minimum_log_std = base.minimum_log_std_at(iteration, config)
    policy.maximum_log_std = base.maximum_log_std_at(iteration, config)
    policy.maximum_q_log_std = q_log_std_ceiling_at(iteration, config)
    stage = training_stage_at(iteration, config)
    minimum_q = minimum_q_at(iteration, config)

    training_sizes = tuple(config.training_n_lasers)
    schedule_offset = (iteration * config.targets_per_batch) % len(training_sizes)
    target_sizes = [
        training_sizes[(schedule_offset + index) % len(training_sizes)]
        for index in range(config.targets_per_batch)
    ]
    target_counts = {
        size: target_sizes.count(size)
        for size in training_sizes
        if size in target_sizes
    }
    all_tolerances = sample_allowable_tolerances(
        config.targets_per_batch, iteration, config, rng
    )
    tolerance_offset = 0
    prepared_batches: list[dict[str, Any]] = []
    indexed_work_items: list[tuple[int, int, int, tuple[Any, ...]]] = []

    for size_batch_index, (n_lasers, target_count) in enumerate(
        target_counts.items()
    ):
        size_config = replace(config, n_lasers=n_lasers)
        tolerances = all_tolerances[
            tolerance_offset : tolerance_offset + target_count
        ]
        tolerance_offset += target_count
        targets = base.sample_target_phases(
            target_count,
            rng,
            n_lasers,
            config.structured_target_fraction,
            config.structured_target_jitter_rad,
        )
        detunings = base.sample_detuning_distributions(
            target_count, rng, size_config, iteration=iteration
        )
        targets, detunings = base.sort_by_detuning(targets, detunings)
        features = encode_sparse_tolerance_context(
            targets, detunings, tolerances, policy, size_config
        )

        if stage == "dense":
            q = torch.ones(
                (target_count, config.candidates_per_target),
                device=features.device,
                dtype=features.dtype,
            )
            actions, log_probabilities = sample_edge_candidates_at_q(
                policy, features, q
            )
            sparse_candidates = config.candidates_per_target
        else:
            sparse_candidates = (
                config.candidates_per_target - config.dense_replay_candidates
            )
            sparse_actions, sparse_q, sparse_log_probability = (
                sample_joint_sparse_candidates(
                    policy, features, sparse_candidates, minimum_q
                )
            )
            replay_q = torch.ones(
                (target_count, config.dense_replay_candidates),
                device=features.device,
                dtype=features.dtype,
            )
            replay_actions, replay_log_probability = sample_edge_candidates_at_q(
                policy, features, replay_q
            )
            actions = torch.cat((sparse_actions, replay_actions), dim=1)
            q = torch.cat((sparse_q, replay_q), dim=1)
            log_probabilities = torch.cat(
                (sparse_log_probability, replay_log_probability), dim=1
            )

        q_numpy = q.detach().cpu().numpy()
        kappa, phi, link_counts, rho = _decode_and_prune(
            actions, q_numpy, size_config
        )
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
            work_items = base.split_simulation_batch(
                targets,
                kappa[:, candidate_start:candidate_stop],
                phi[:, candidate_start:candidate_stop],
                size_config,
                detuning_distributions_ghz=detunings,
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
                "targets": targets,
                "tolerances": tolerances,
                "features": features,
                "q": q_numpy,
                "rho": rho,
                "link_counts": link_counts,
                "log_probabilities": log_probabilities,
                "sparse_candidates": sparse_candidates,
                "chunk_results": [],
                "mean_action_std": float(
                    policy.action_distribution(
                        features,
                        torch.ones(target_count, device=features.device),
                    ).stddev.mean().detach().cpu()
                ),
            }
        )

    indexed_work_items.sort(
        key=lambda item: prepared_batches[item[0]]["n_lasers"], reverse=True
    )
    if config.n_jobs == 1:
        chunk_results = [
            base.simulate_indexed_target_chunk(item)
            for item in indexed_work_items
        ]
    elif simulation_pool is not None:
        chunk_results = list(
            simulation_pool.imap_unordered(
                base.simulate_indexed_target_chunk, indexed_work_items
            )
        )
    else:
        context = mp.get_context("spawn")
        temporary_pool = context.Pool(
            processes=min(config.n_jobs, len(indexed_work_items)),
            initializer=base._initialize_simulation_worker,
        )
        try:
            chunk_results = list(
                temporary_pool.imap_unordered(
                    base.simulate_indexed_target_chunk, indexed_work_items
                )
            )
        finally:
            temporary_pool.close()
            temporary_pool.join()
    for result in chunk_results:
        prepared_batches[int(result["size_batch_index"])][
            "chunk_results"
        ].append(result)

    losses: list[torch.Tensor] = []
    size_metrics: dict[int, dict[str, float]] = {}
    bin_accumulator: dict[str, list[dict[str, float]]] = {}
    for prepared in prepared_batches:
        target_count = prepared["target_count"]
        phase_reward = np.empty(
            (target_count, config.candidates_per_target), dtype=float
        )
        aligned_phases = np.empty(
            (
                target_count,
                config.candidates_per_target,
                prepared["n_lasers"],
            ),
            dtype=float,
        )
        coverage = np.zeros_like(phase_reward, dtype=int)
        for result in prepared["chunk_results"]:
            target_start = int(result["start_index"])
            target_stop = int(result["stop_index"])
            candidate_start = int(result["candidate_start_index"])
            candidate_stop = int(result["candidate_stop_index"])
            phase_reward[
                target_start:target_stop, candidate_start:candidate_stop
            ] = result["phase_reward"]
            aligned_phases[
                target_start:target_stop, candidate_start:candidate_stop
            ] = result["aligned_achieved_phases_rad"]
            coverage[
                target_start:target_stop, candidate_start:candidate_stop
            ] += 1
        if not np.all(coverage == 1):
            raise RuntimeError(
                "sparse multi-size simulation did not return every candidate "
                f"exactly once for M={prepared['n_lasers']}"
            )
        maximum_error = maximum_circular_phase_error_deg(
            prepared["targets"], aligned_phases
        )
        features = prepared["features"]
        log_probabilities = prepared["log_probabilities"]
        tolerances = prepared["tolerances"]
        q_numpy = prepared["q"]
        sparse_candidates = prepared["sparse_candidates"]

        if stage == "dense":
            training_reward = phase_reward
            feasible = maximum_error <= tolerances[:, None]
            advantages = _standardized_advantages(training_reward)
            size_loss = -(
                log_probabilities
                * torch.as_tensor(
                    advantages, dtype=torch.float32, device=features.device
                )
            ).mean()
            metric_slice = slice(0, config.candidates_per_target)
        else:
            sparse_reward, sparse_feasible = tiered_sparse_reward(
                phase_reward[:, :sparse_candidates],
                maximum_error[:, :sparse_candidates],
                tolerances,
                q_numpy[:, :sparse_candidates],
                config.infeasible_reward_tau_deg,
            )
            sparse_advantage = torch.as_tensor(
                _standardized_advantages(sparse_reward),
                dtype=torch.float32,
                device=features.device,
            )
            sparse_loss = -(
                log_probabilities[:, :sparse_candidates] * sparse_advantage
            ).mean()
            replay_reward = phase_reward[:, sparse_candidates:]
            replay_advantage = torch.as_tensor(
                _standardized_advantages(replay_reward),
                dtype=torch.float32,
                device=features.device,
            )
            replay_loss = -(
                log_probabilities[:, sparse_candidates:] * replay_advantage
            ).mean()
            sparse_weight = sparse_candidates / config.candidates_per_target
            replay_weight = (
                config.dense_replay_candidates / config.candidates_per_target
            )
            size_loss = sparse_weight * sparse_loss + replay_weight * replay_loss
            training_reward = sparse_reward
            feasible = sparse_feasible
            metric_slice = slice(0, sparse_candidates)

        losses.append(size_loss)
        metric_phase = phase_reward[:, metric_slice]
        metric_error = maximum_error[:, metric_slice]
        metric_q = q_numpy[:, metric_slice]
        metric_rho = prepared["rho"][:, metric_slice]
        size_metrics[prepared["n_lasers"]] = {
            "loss": float(size_loss.detach().cpu()),
            "phase_reward": float(np.mean(metric_phase)),
            "training_reward": float(np.mean(training_reward)),
            "maximum_phase_error_deg": float(np.mean(metric_error)),
            "constraint_satisfaction_rate": float(np.mean(feasible)),
            "mean_q": float(np.mean(metric_q)),
            "retained_link_fraction": float(np.mean(metric_rho)),
            "mean_active_links": float(
                np.mean(prepared["link_counts"][:, metric_slice])
            ),
            "mean_action_std": prepared["mean_action_std"],
        }
        per_bin = _bin_metrics(
            tolerances,
            training_reward,
            feasible,
            metric_rho,
            config.tolerance_bin_edges_deg,
        )
        for label, values in per_bin.items():
            if np.isfinite(values["reward"]):
                bin_accumulator.setdefault(label, []).append(values)

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    gradient_diagnostics = {
        "negative_gradient_pair_fraction": np.nan,
        "mean_gradient_cosine": np.nan,
    }
    if config.gradient_combination_mode == "pcgrad":
        parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        task_gradients = []
        for size_loss in losses:
            gradients = torch.autograd.grad(
                size_loss, parameters, allow_unused=True
            )
            task_gradients.append(
                torch.cat(
                    [
                        torch.zeros_like(parameter).reshape(-1)
                        if gradient is None
                        else gradient.detach().reshape(-1)
                        for parameter, gradient in zip(parameters, gradients)
                    ]
                )
            )
        combined, gradient_diagnostics = base.combine_task_gradients_pcgrad(
            torch.stack(task_gradients), config.per_size_gradient_clip, rng
        )
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad = combined[offset : offset + count].reshape_as(parameter).clone()
            offset += count
    else:
        loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config.gradient_clip
    )
    optimizer.step()
    with torch.no_grad():
        policy.log_std_kappa.clamp_(policy.minimum_log_std, policy.maximum_log_std)
        policy.log_std_phi.clamp_(policy.minimum_log_std, policy.maximum_log_std)
        policy.q_log_std.clamp_(
            config.minimum_q_log_std, policy.maximum_q_log_std
        )

    def mean_metric(name: str) -> float:
        return float(np.mean([values[name] for values in size_metrics.values()]))

    tolerance_bin_metrics: dict[str, dict[str, float]] = {}
    for lower, upper in zip(
        config.tolerance_bin_edges_deg[:-1],
        config.tolerance_bin_edges_deg[1:],
    ):
        label = f"{lower:g}-{upper:g} deg"
        values = bin_accumulator.get(label, [])
        tolerance_bin_metrics[label] = {
            key: float(np.mean([item[key] for item in values])) if values else np.nan
            for key in ("reward", "satisfaction", "rho")
        }
    return {
        "stage": stage,
        "loss": float(loss.detach().cpu()),
        "phase_reward": mean_metric("phase_reward"),
        "training_reward": mean_metric("training_reward"),
        "maximum_phase_error_deg": mean_metric("maximum_phase_error_deg"),
        "constraint_satisfaction_rate": mean_metric(
            "constraint_satisfaction_rate"
        ),
        "mean_q": mean_metric("mean_q"),
        "retained_link_fraction": mean_metric("retained_link_fraction"),
        "mean_action_std": mean_metric("mean_action_std"),
        "mean_q_std": float(torch.exp(policy.q_log_std).detach().cpu()),
        "minimum_q": minimum_q,
        "maximum_tolerance_deg": maximum_tolerance_at(iteration, config),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "negative_gradient_pair_fraction": float(
            gradient_diagnostics["negative_gradient_pair_fraction"]
        ),
        "mean_gradient_cosine": float(
            gradient_diagnostics["mean_gradient_cosine"]
        ),
        "size_metrics": size_metrics,
        "tolerance_bin_metrics": tolerance_bin_metrics,
    }


def evaluate_sparse_policy(
    policy: SparseToleranceGNNPolicy,
    targets_rad: np.ndarray,
    detuning_distributions_ghz: np.ndarray,
    allowable_error_deg: np.ndarray | float,
    config: SparseToleranceConfig,
    *,
    q_override: np.ndarray | float | None = None,
    simulation_pool: Any | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate the deterministic hierarchical policy with one simulation each."""
    targets = np.asarray(targets_rad, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    detunings = np.asarray(detuning_distributions_ghz, dtype=float)
    if detunings.ndim == 1:
        detunings = detunings[None, :]
    targets, detunings = base.sort_by_detuning(targets, detunings)
    tolerances = np.asarray(allowable_error_deg, dtype=float)
    if tolerances.ndim == 0:
        tolerances = np.full(len(targets), float(tolerances))
    features = encode_sparse_tolerance_context(
        targets, detunings, tolerances, policy, config
    )
    policy.eval()
    with torch.no_grad():
        if q_override is None:
            q = torch.sigmoid(policy.density_distribution(features).mean[:, 0])
        else:
            q_array = np.asarray(q_override, dtype=float)
            if q_array.ndim == 0:
                q_array = np.full(len(targets), float(q_array))
            if q_array.shape != (len(targets),):
                raise ValueError("q_override must be scalar or one per target")
            q = torch.as_tensor(q_array, dtype=features.dtype, device=features.device)
        actions = policy.action_mean(features, q)[:, None, :]
    q_matrix = q.detach().cpu().numpy()[:, None]
    kappa, phi, counts, rho = _decode_and_prune(actions, q_matrix, config)
    simulation = base.simulate_candidates_parallel(
        targets,
        kappa,
        phi,
        config,
        detuning_distributions_ghz=detunings,
        simulation_pool=simulation_pool,
    )
    maximum_error = maximum_circular_phase_error_deg(
        targets, simulation["aligned_achieved_phases_rad"]
    )[:, 0]
    reward, feasible = tiered_sparse_reward(
        simulation["phase_reward"],
        maximum_error[:, None],
        tolerances,
        q_matrix,
        config.infeasible_reward_tau_deg,
    )
    return {
        "targets_rad": targets,
        "detuning_distributions_ghz": detunings,
        "allowable_error_deg": tolerances,
        "q": q_matrix[:, 0],
        "active_link_counts": counts[:, 0],
        "retained_link_fraction": rho[:, 0],
        "kappa_per_ns": kappa[:, 0],
        "phi_p_rad": phi[:, 0],
        "phase_reward": simulation["phase_reward"][:, 0],
        "maximum_phase_error_deg": maximum_error,
        "constraint_satisfied": feasible[:, 0],
        "training_reward": reward[:, 0],
        "achieved_phases_rad": simulation["achieved_phases_rad"][:, 0],
        "aligned_achieved_phases_rad": simulation[
            "aligned_achieved_phases_rad"
        ][:, 0],
        "orientation": simulation["orientation"][:, 0],
    }


def design_sparse_for_target(
    policy: SparseToleranceGNNPolicy,
    target_phases_rad: np.ndarray,
    detuning_distribution_ghz: np.ndarray,
    allowable_error_deg: float,
    config: SparseToleranceConfig,
    *,
    number_of_candidates: int = 1,
    q_override: float | None = None,
    include_deterministic_candidate: bool = True,
    simulation_pool: Any | None = None,
) -> list[dict[str, Any]]:
    """Sample or deterministically evaluate sparse designs for one context."""
    if number_of_candidates < 1:
        raise ValueError("number_of_candidates must be positive")
    target = np.asarray(target_phases_rad, dtype=float)
    detuning = np.asarray(detuning_distribution_ghz, dtype=float)
    target, detuning = base.sort_by_detuning(target, detuning)
    features = encode_sparse_tolerance_context(
        target, detuning, allowable_error_deg, policy, config
    )
    policy.eval()
    with torch.no_grad():
        if q_override is not None:
            q = torch.full(
                (1, number_of_candidates),
                float(q_override),
                device=features.device,
                dtype=features.dtype,
            )
            # A user-specified q is an inspection control, not an
            # exploration request.  Use the same deterministic edge mean as
            # automatic one-candidate inference so matching q values produce
            # matching coupling matrices and pruning decisions.
            repeated_features = _repeat_contexts(features, number_of_candidates)
            actions = policy.action_mean(
                repeated_features, q.reshape(-1)
            ).reshape(1, number_of_candidates, -1)
        elif number_of_candidates == 1:
            q_mean = torch.sigmoid(policy.density_distribution(features).mean[:, 0])
            q = q_mean[:, None]
            actions = policy.action_mean(features, q_mean)[:, None, :]
        else:
            actions, q, _ = sample_joint_sparse_candidates(
                policy, features, number_of_candidates, 0.0
            )
        if include_deterministic_candidate and number_of_candidates > 1:
            q_mean = torch.sigmoid(policy.density_distribution(features).mean[:, 0])
            q[0, 0] = q_mean[0]
            actions[0, 0] = policy.action_mean(features, q_mean)[0]
    q_numpy = q.detach().cpu().numpy()
    kappa, phi, counts, rho = _decode_and_prune(actions, q_numpy, config)
    simulation = base.simulate_candidates_parallel(
        target[None, :],
        kappa,
        phi,
        config,
        detuning_distributions_ghz=detuning[None, :],
        simulation_pool=simulation_pool,
    )
    maximum_error = maximum_circular_phase_error_deg(
        target[None, :], simulation["aligned_achieved_phases_rad"]
    )
    reward, feasible = tiered_sparse_reward(
        simulation["phase_reward"],
        maximum_error,
        np.asarray([allowable_error_deg]),
        q_numpy,
        config.infeasible_reward_tau_deg,
    )
    order = np.argsort(reward[0])[::-1]
    results: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        mask = kappa[0, index] > 0.0
        results.append(
            {
                "rank": rank,
                "target_phases_rad": target.copy(),
                "detuning_distribution_ghz": detuning.copy(),
                "allowable_error_deg": float(allowable_error_deg),
                "q": float(q_numpy[0, index]),
                "predicted_q": float(q_numpy[0, index]),
                "active_link_count": int(counts[0, index]),
                "active_connection_fraction": float(rho[0, index]),
                "realized_rho": float(rho[0, index]),
                "predicted_rho": float(rho[0, index]),
                "kappa_per_ns": kappa[0, index].copy(),
                "phi_p_rad": phi[0, index].copy(),
                "magnitude_gates": mask[base.directed_link_indices(config.n_lasers)].astype(float),
                "phase_reward": float(simulation["phase_reward"][0, index]),
                "maximum_phase_error_deg": float(maximum_error[0, index]),
                "constraint_satisfied": bool(feasible[0, index]),
                "ranking_reward": float(reward[0, index]),
                "training_reward": float(reward[0, index]),
                "achieved_phases_rad": simulation["achieved_phases_rad"][0, index].copy(),
                "aligned_achieved_phases_rad": simulation[
                    "aligned_achieved_phases_rad"
                ][0, index].copy(),
                "orientation": float(simulation["orientation"][0, index]),
                "normalized_coupling_budget": float(
                    np.sum(kappa[0, index])
                    / max(
                        config.n_lasers
                        * (config.n_lasers - 1)
                        * base.effective_maximum_kappa_per_link(
                            config.maximum_kappa_per_ns,
                            config.n_lasers,
                            config.normalize_incoming_coupling_by_degree,
                        ),
                        1.0e-12,
                    )
                ),
                "magnitude_symmetry_penalty": float(
                    base.normalized_magnitude_symmetry_penalty(
                        kappa[0, index][None],
                        base.effective_maximum_kappa_per_link(
                            config.maximum_kappa_per_ns,
                            config.n_lasers,
                            config.normalize_incoming_coupling_by_degree,
                        ),
                    )[0]
                ),
            }
        )
    return results


def save_sparse_checkpoint(
    filename: str | Path,
    policy: SparseToleranceGNNPolicy,
    optimizer: torch.optim.Optimizer,
    config: SparseToleranceConfig,
    *,
    iteration: int,
    held_out_reward: float,
) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": SPARSE_TOLERANCE_CHECKPOINT_FORMAT,
            "variable_m_policy": True,
            "encoder_architecture": "gnn_sparse_tolerance",
            "config": asdict(config),
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": int(iteration),
            "held_out_reward": float(held_out_reward),
        },
        temporary,
    )
    os.replace(temporary, path)
    return path


def load_sparse_checkpoint(
    filename: str | Path,
    *,
    device: str | None = None,
) -> tuple[
    SparseToleranceGNNPolicy,
    torch.optim.Optimizer,
    SparseToleranceConfig,
    dict[str, float | int],
]:
    checkpoint = torch.load(
        filename, map_location=device or "cpu", weights_only=False
    )
    if checkpoint.get("format") != SPARSE_TOLERANCE_CHECKPOINT_FORMAT:
        raise ValueError(
            "Checkpoint does not contain the selector-free sparse-tolerance "
            "policy. Use its dedicated inspect script, or train this new "
            "experiment from scratch."
        )
    config = SparseToleranceConfig(**dict(checkpoint["config"]))
    if device is not None:
        config = replace(config, device=device)
    policy = SparseToleranceGNNPolicy(config).to(torch.device(config.device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return policy, optimizer, config, {
        "iteration": int(checkpoint["iteration"]),
        "held_out_reward": float(checkpoint["held_out_reward"]),
    }


def _make_history(config: SparseToleranceConfig) -> dict[str, Any]:
    size_keys = tuple(config.training_n_lasers)
    bin_labels = [
        f"{lower:g}-{upper:g} deg"
        for lower, upper in zip(
            config.tolerance_bin_edges_deg[:-1],
            config.tolerance_bin_edges_deg[1:],
        )
    ]
    return {
        "stage": [],
        "phase_reward": [],
        "training_reward": [],
        "maximum_phase_error_deg": [],
        "constraint_satisfaction_rate": [],
        "mean_q": [],
        "retained_link_fraction": [],
        "mean_q_std": [],
        "minimum_q": [],
        "maximum_tolerance_deg": [],
        "loss": [],
        "gradient_norm": [],
        "negative_gradient_pair_fraction": [],
        "mean_gradient_cosine": [],
        "reward_by_m": {size: [] for size in size_keys},
        "rho_by_m": {size: [] for size in size_keys},
        "held_out_dense_reward": [],
        "held_out_dense_reward_by_m": {size: [] for size in size_keys},
        "tolerance_bins": {
            label: {key: [] for key in ("reward", "satisfaction", "rho")}
            for label in bin_labels
        },
    }


def update_sparse_training_figure(
    history: dict[str, Any],
    config: SparseToleranceConfig,
    *,
    start_iteration: int = 0,
) -> plt.Figure:
    """Match the existing runners' simple two-panel progress figure."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    iterations = np.arange(start_iteration, start_iteration + len(history["training_reward"]))
    axes[0].plot(iterations, history["training_reward"], label="total reward")
    axes[0].plot(
        iterations,
        history["retained_link_fraction"],
        color="tab:green",
        linestyle="--",
        label=r"mean retained-link fraction $\rho$",
    )
    axes[0].set(
        title="REINFORCE training",
        xlabel="iteration",
        ylabel="mean reward / retained-link fraction",
        ylim=(0.0, 1.0),
    )
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.25)

    for n_lasers, values in history["held_out_dense_reward_by_m"].items():
        axes[1].plot(iterations, values, label=f"M={n_lasers}")
    axes[1].plot(
        iterations,
        history["held_out_dense_reward"],
        color="black",
        linewidth=2.0,
        label="overall",
    )
    axes[1].set(
        title=(
            "Dense held-out targets "
            f"(T={config.held_out_strict_tolerance_deg:g} deg)"
        ),
        xlabel="iteration",
        ylabel="mean phase reward",
        ylim=(0.0, 1.0),
    )
    axes[1].legend(loc="lower left", ncol=2)
    axes[1].grid(alpha=0.25)
    return figure


def train_sparse_tolerance_policy(
    config: SparseToleranceConfig,
    policy: SparseToleranceGNNPolicy | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    live_plot: bool = True,
    start_iteration: int = 0,
) -> tuple[SparseToleranceGNNPolicy, torch.optim.Optimizer, dict[str, Any]]:
    """Train or resume the staged selector-free sparse policy."""
    if not 0 <= start_iteration < config.training_iterations:
        raise ValueError("start_iteration must lie within the training range")
    rng = base.set_random_seed(config.random_seed)
    policy = (
        SparseToleranceGNNPolicy(config).to(torch.device(config.device))
        if policy is None
        else policy
    )
    optimizer = (
        torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        if optimizer is None
        else optimizer
    )
    history = _make_history(config)
    held_out_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n_lasers in config.training_n_lasers:
        size_config = replace(config, n_lasers=n_lasers)
        targets = base.sample_target_phases(
            config.held_out_target_count,
            np.random.default_rng(config.random_seed + 10_000 + n_lasers),
            n_lasers,
            config.structured_target_fraction,
            config.structured_target_jitter_rad,
        )
        detunings = base.sample_detuning_distributions(
            config.held_out_target_count,
            np.random.default_rng(config.random_seed + 20_000 + n_lasers),
            size_config,
            iteration=config.detuning_curriculum_iterations,
        )
        held_out_sets[n_lasers] = base.sort_by_detuning(targets, detunings)

    latest_held_out_by_m = {size: np.nan for size in config.training_n_lasers}
    latest_held_out = np.nan
    best_held_out = -np.inf
    simulation_pool = None
    if config.n_jobs > 1:
        context = mp.get_context("spawn")
        simulation_pool = context.Pool(
            processes=min(config.n_jobs, config.targets_per_batch),
            initializer=base._initialize_simulation_worker,
        )
    progress_directory = RESULTS_DIR
    progress_directory.mkdir(parents=True, exist_ok=True)

    try:
        for iteration in range(start_iteration, config.training_iterations):
            scheduled_learning_rate = learning_rate_at(iteration, config)
            if scheduled_learning_rate is not None:
                for group in optimizer.param_groups:
                    group["lr"] = scheduled_learning_rate
            elif iteration == config.learning_rate_switch_iteration:
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate_after_switch
            metrics = run_sparse_multisize_training_batch(
                policy,
                optimizer,
                config,
                rng,
                iteration,
                simulation_pool=simulation_pool,
            )
            for key in (
                "stage",
                "phase_reward",
                "training_reward",
                "maximum_phase_error_deg",
                "constraint_satisfaction_rate",
                "mean_q",
                "retained_link_fraction",
                "mean_q_std",
                "minimum_q",
                "maximum_tolerance_deg",
                "loss",
                "gradient_norm",
                "negative_gradient_pair_fraction",
                "mean_gradient_cosine",
            ):
                history[key].append(metrics[key])
            for size in config.training_n_lasers:
                history["reward_by_m"][size].append(
                    metrics["size_metrics"][size]["training_reward"]
                )
                history["rho_by_m"][size].append(
                    metrics["size_metrics"][size]["retained_link_fraction"]
                )
            for label, values in history["tolerance_bins"].items():
                current = metrics["tolerance_bin_metrics"][label]
                for key in values:
                    values[key].append(current[key])

            should_validate = (
                iteration == 0
                or (iteration + 1) % config.validation_interval == 0
                or iteration + 1 == config.training_iterations
            )
            if should_validate:
                half_span = base.detuning_curriculum_half_span_at(iteration, config)
                for n_lasers in config.training_n_lasers:
                    targets, full_detunings = held_out_sets[n_lasers]
                    detunings = full_detunings
                    if config.validation_follows_detuning_curriculum:
                        detunings = (
                            full_detunings
                            * half_span
                            / (0.5 * config.detuning_span_ghz)
                        )
                    validation = evaluate_sparse_policy(
                        policy,
                        targets,
                        detunings,
                        config.held_out_strict_tolerance_deg,
                        replace(config, n_lasers=n_lasers),
                        q_override=1.0,
                        simulation_pool=simulation_pool,
                    )
                    latest_held_out_by_m[n_lasers] = float(
                        np.mean(validation["phase_reward"])
                    )
                latest_held_out = float(np.mean(list(latest_held_out_by_m.values())))
                if latest_held_out > best_held_out:
                    best_held_out = latest_held_out
                    save_sparse_checkpoint(
                        config.best_checkpoint_file,
                        policy,
                        optimizer,
                        config,
                        iteration=iteration + 1,
                        held_out_reward=latest_held_out,
                    )
            history["held_out_dense_reward"].append(latest_held_out)
            for size in config.training_n_lasers:
                history["held_out_dense_reward_by_m"][size].append(
                    latest_held_out_by_m[size]
                )

            # Match the existing runner: keep an atomic current checkpoint at
            # every iteration so the inspect script can follow a live run.
            save_sparse_checkpoint(
                config.current_checkpoint_file,
                policy,
                optimizer,
                config,
                iteration=iteration + 1,
                held_out_reward=latest_held_out,
            )
            if live_plot and (
                iteration == 0
                or (iteration + 1) % config.plot_update_interval == 0
                or iteration + 1 == config.training_iterations
            ):
                figure = update_sparse_training_figure(
                    history, config, start_iteration=start_iteration
                )
                stem = Path(config.current_checkpoint_file).stem
                figure.savefig(
                    progress_directory / f"{stem}_training_progress.png",
                    dpi=150,
                )
                if config.jupyter_mode:
                    clear_output(wait=True)
                    display(figure)
                plt.close(figure)

            print(
                f"Iteration {iteration + 1:4d}/{config.training_iterations} | "
                f"stage={metrics['stage']} | "
                f"phase={metrics['phase_reward']:.3f} | "
                f"reward={metrics['training_reward']:.3f} | "
                f"Emax={metrics['maximum_phase_error_deg']:.1f} deg | "
                f"satisfied={metrics['constraint_satisfaction_rate']:.3f} | "
                f"q={metrics['mean_q']:.3f} | "
                f"rho={metrics['retained_link_fraction']:.3f} | "
                f"q-std={metrics['mean_q_std']:.3f} | "
                f"held-out-dense={latest_held_out:.3f}"
            )
            if config.gradient_combination_mode == "pcgrad":
                print(
                    "PCGrad | conflicting pairs="
                    f"{metrics['negative_gradient_pair_fraction']:.3f} | "
                    f"mean cosine={metrics['mean_gradient_cosine']:+.3f}"
                )

        save_sparse_checkpoint(
            config.final_checkpoint_file,
            policy,
            optimizer,
            config,
            iteration=config.training_iterations,
            held_out_reward=latest_held_out,
        )
    finally:
        if simulation_pool is not None:
            simulation_pool.close()
            simulation_pool.join()
    return policy, optimizer, history


__all__ = [
    "SPARSE_TOLERANCE_CHECKPOINT_FORMAT",
    "SparseToleranceConfig",
    "SparseToleranceGNNPolicy",
    "design_sparse_for_target",
    "encode_sparse_tolerance_context",
    "evaluate_sparse_policy",
    "load_sparse_checkpoint",
    "learning_rate_at",
    "maximum_circular_phase_error_deg",
    "maximum_tolerance_at",
    "minimum_q_at",
    "q_to_active_link_counts",
    "sample_allowable_tolerances",
    "save_sparse_checkpoint",
    "tiered_sparse_reward",
    "train_sparse_tolerance_policy",
    "training_stage_at",
    "update_sparse_training_figure",
]
