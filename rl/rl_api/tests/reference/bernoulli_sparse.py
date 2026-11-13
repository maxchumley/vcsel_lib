"""Sparse coupling policies with sampled gates or supplied rooted topologies.

Each directed edge has Gaussian strength/phase actions and a Bernoulli on/off
action. The existing dense VCSEL simulator receives the resulting matrices.
Enabled edges absorb the candidate's removed strength. The original capped
decoder remains available for older checkpoints; the softplus decoder has no
strength ceiling and preserves the raw total under rooted masks.
Training-only logit offsets expose the policy to a broad range of densities;
they are annealed toward zero, which is the default inference distribution.
Optionally, non-rooted directed masks receive a graded negative reward without
running the VCSEL simulator.
Alternatively, a learned root and outward directed tree are sampled first;
this guarantees rooted masks while retaining Bernoulli optional links.
Topology-input mode instead supplies a density-stratified rooted mask before
the GNN, and optimizes only the conditional Gaussian strength/phase/budget
policy against phase reward.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from scipy.optimize import differential_evolution
from torch import nn

from rl import conditional_coupling_designer_variable_m as base
from rl.paths import RESULTS_DIR


CHECKPOINT_FORMAT = "conditional_coupling_variable_m_bernoulli_sparse_v1"
DENSITY_BANDS = ("sparse", "middle", "dense")


@dataclass
class BernoulliSparseConfig(base.DesignerConfig):
    dense_reference_feasibility: bool = False
    feasible_sparsity_bonus_weight: float = 0.01
    uncapped_redistribution: bool = False
    strength_action_transform: str = "bounded_sigmoid"
    softplus_strength_scale_per_ns: float = 25.0
    require_rooted_connectivity: bool = False
    construct_rooted_backbone: bool = False
    sparsity_reward_mode: str = "total_sparsity_fraction"
    link_cost_weight: float = 0.25
    coupling_cost_weight: float = 0.15
    coupling_cost_start_iteration: int = 0
    coupling_cost_ramp_iterations: int = 0
    plot_coupling_reference: bool = True
    exploration_logit_offset_start: float = 4.0
    exploration_logit_offset_end: float = 0.25
    exploration_offset_anneal_iterations: int = 4000
    gate_head_hold_iterations: int = 0
    uniform_random_rooted_tree: bool = False
    topology_input_mode: bool = False
    actions_per_topology: int = 1
    training_topology_min_fraction: float = 0.0
    training_topology_density_bias: float = 1.0
    topology_star_backbone: bool = False
    topology_curriculum_start_iteration: int | None = None
    topology_curriculum_end_iteration: int | None = None
    topology_advantage_scale: str = "per_topology"  # or "global_running"
    # Legacy checkpoints retain one task per M. The new option splits each M
    # by excess-link density, using the same three bands as validation.
    pcgrad_task_grouping: str = "array_size"
    global_reward_std_ema_decay: float = 0.95
    learning_rate_final: float = 1.0e-5
    learning_rate_decay_start_iteration: int = 3500
    budget_reference_calibration_targets: int = 64
    targets_per_worker_task: int = 1
    # Frozen before the first optimizer update and saved with the checkpoint.
    # Each entry is (M, initial all-on median total kappa in ns^-1).
    coupling_budget_reference_by_m: tuple[tuple[int, float], ...] = ()
    objective_mode: str = "weighted"  # "weighted", "lexicographic", or "phase_only"
    lexicographic_order: str = "sparsity_budget"  # or "budget_sparsity"
    lexicographic_maximum_phase_error_deg: float = 3.0
    lexicographic_initial_phase_error_deg: float = 3.0
    lexicographic_phase_error_anneal_iterations: int = 0
    lexicographic_minimum_phase_reward: float = 0.98
    lexicographic_initial_minimum_phase_reward: float = 0.98
    budget_action_transform: str = "scaled_softplus"  # legacy, or "log_budget"
    budget_exploration_mode: str = "stratified_offsets"  # "local" or "mixed_std"
    budget_wide_exploration_fraction: float = 0.0
    budget_exploration_start_iteration: int = 1000
    budget_exploration_end_iteration: int = 4500
    budget_exploration_start_factor: float = 100.0
    budget_exploration_end_factor: float = 1.2
    budget_exploration_start_std: float = 0.3
    budget_exploration_end_std: float = 0.1
    # Training-only wide strength samples; the remaining actions per topology
    # retain the decoder's learned local standard deviations.
    strength_exploration_fraction: float = 0.0
    strength_exploration_start_iteration: int = 500
    strength_exploration_end_iteration: int = 4000
    strength_exploration_start_std: float = 5.0
    # Training-only wide phase samples; phases are wrapped after sampling.
    phase_exploration_fraction: float = 0.0
    phase_exploration_start_iteration: int = 500
    phase_exploration_end_iteration: int = 4000
    phase_exploration_start_std: float = 0.0
    de_teacher_enabled: bool = False
    de_teacher_start_iteration: int = 0
    de_teacher_interval: int = 1
    de_teacher_population: int = 5
    de_teacher_generations: int = 3
    de_teacher_mutation: float = 0.8
    de_teacher_recombination: float = 0.7
    de_teacher_loss_weight: float = 0.1
    de_teacher_min_improvement: float = 1.0e-4
    # Retains the absolute offset of older log-budget checkpoints at inference.
    log_budget_initial_per_laser: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.encoder_architecture != "gnn":
            raise ValueError("Bernoulli sparse policy requires the GNN encoder")
        if not self.enable_sparse_gates and not self.topology_input_mode:
            raise ValueError("Bernoulli sparse policy requires sparse gates")
        if self.force_symmetric_kappa or self.force_symmetric_phi_p:
            raise ValueError("Bernoulli sparse policy uses directed edges")
        if self.construct_rooted_backbone and self.require_rooted_connectivity:
            raise ValueError("A constructed rooted backbone does not need the non-rooted reward")
        if self.uniform_random_rooted_tree and not self.construct_rooted_backbone:
            raise ValueError("Uniform random rooted trees require a constructed backbone")
        if self.topology_input_mode and (self.construct_rooted_backbone or self.uniform_random_rooted_tree):
            raise ValueError("Input topologies replace sampled policy backbones")
        if self.topology_input_mode and self.enable_sparse_gates:
            raise ValueError("Topology-input mode has no Bernoulli gate head")
        if self.topology_input_mode and not self.uncapped_redistribution:
            raise ValueError("Topology-input budgets require uncapped redistribution")
        if self.topology_input_mode and self.objective_mode != "phase_only":
            raise ValueError("Topology-input mode currently uses the phase-only objective")
        if self.objective_mode == "phase_only" and not self.topology_input_mode:
            raise ValueError("Phase-only mode requires topology_input_mode")
        if self.topology_input_mode and self.dense_reference_feasibility:
            raise ValueError("Topology-input mode does not use dense-reference feasibility")
        if self.actions_per_topology < 1 or self.candidates_per_target % self.actions_per_topology:
            raise ValueError("actions_per_topology must divide candidates_per_target")
        if self.topology_input_mode and self.actions_per_topology < 2:
            raise ValueError("Topology-input training needs multiple actions per topology")
        if (not np.isfinite(self.training_topology_min_fraction)
                or not 0 <= self.training_topology_min_fraction <= 1):
            raise ValueError("Training topology minimum fraction must lie in [0, 1]")
        if (not np.isfinite(self.training_topology_density_bias)
                or self.training_topology_density_bias <= 0):
            raise ValueError("Training topology density bias must be positive and finite")
        if self.topology_star_backbone and not self.topology_input_mode:
            raise ValueError("A fixed star backbone requires topology-input mode")
        start, end = (self.topology_curriculum_start_iteration,
                      self.topology_curriculum_end_iteration)
        if (start is None) != (end is None):
            raise ValueError("Topology curriculum needs both start and end iterations")
        if start is not None and (not self.topology_input_mode or start < 0 or end <= start):
            raise ValueError("Topology curriculum needs topology-input mode and increasing iterations")
        if self.topology_advantage_scale not in {"per_topology", "global_running"}:
            raise ValueError("topology_advantage_scale must be per_topology or global_running")
        if self.pcgrad_task_grouping not in {"array_size", "array_size_topology_density"}:
            raise ValueError("Unknown PCGrad task grouping")
        if self.pcgrad_task_grouping != "array_size" and not self.topology_input_mode:
            raise ValueError("Topology-density PCGrad requires topology-input mode")
        if not 0 <= self.global_reward_std_ema_decay < 1:
            raise ValueError("global_reward_std_ema_decay must lie in [0, 1)")
        if self.strength_action_transform not in {"bounded_sigmoid", "uncapped_softplus"}:
            raise ValueError("Unknown strength action transform")
        if not np.isfinite(self.softplus_strength_scale_per_ns) or self.softplus_strength_scale_per_ns <= 0:
            raise ValueError("Softplus strength scale must be positive and finite")
        if self.strength_action_transform == "uncapped_softplus" and not self.uncapped_redistribution:
            raise ValueError("Uncapped softplus strengths require uncapped redistribution")
        if self.sparsity_reward_mode not in {"links_per_laser", "total_sparsity_fraction"}:
            raise ValueError("sparsity_reward_mode must be links_per_laser or total_sparsity_fraction")
        if self.objective_mode not in {"weighted", "lexicographic", "phase_only"}:
            raise ValueError("objective_mode must be weighted, lexicographic, or phase_only")
        if self.lexicographic_order not in {"sparsity_budget", "budget_sparsity"}:
            raise ValueError("lexicographic_order must be sparsity_budget or budget_sparsity")
        if self.budget_action_transform not in {"scaled_softplus", "log_budget"}:
            raise ValueError("budget_action_transform must be scaled_softplus or log_budget")
        if self.budget_exploration_mode not in {"stratified_offsets", "mixed_std", "local"}:
            raise ValueError("budget_exploration_mode must be stratified_offsets, mixed_std, or local")
        if not 0 <= self.budget_wide_exploration_fraction <= 1:
            raise ValueError("budget_wide_exploration_fraction must lie in [0, 1]")
        if not np.isfinite(self.log_budget_initial_per_laser) or self.log_budget_initial_per_laser <= 0:
            raise ValueError("Log-budget initial scale must be positive and finite")
        if not 0 <= self.budget_exploration_start_iteration < self.budget_exploration_end_iteration:
            raise ValueError("Budget exploration schedule iterations must increase")
        if (not np.isfinite(self.budget_exploration_start_factor)
                or not np.isfinite(self.budget_exploration_end_factor)
                or self.budget_exploration_start_factor < self.budget_exploration_end_factor
                or self.budget_exploration_end_factor < 1):
            raise ValueError("Budget exploration factors must decrease and stay at least one")
        if (not np.isfinite(self.budget_exploration_start_std)
                or not np.isfinite(self.budget_exploration_end_std)
                or self.budget_exploration_start_std < self.budget_exploration_end_std
                or self.budget_exploration_end_std <= 0):
            raise ValueError("Budget exploration standard deviations must decrease and stay positive")
        if not 0 <= self.strength_exploration_fraction <= 1:
            raise ValueError("strength_exploration_fraction must lie in [0, 1]")
        if (self.strength_exploration_start_iteration < 0
                or self.strength_exploration_end_iteration
                <= self.strength_exploration_start_iteration):
            raise ValueError("Strength exploration schedule iterations must increase")
        if (not np.isfinite(self.strength_exploration_start_std)
                or self.strength_exploration_start_std < 0):
            raise ValueError("strength_exploration_start_std must be nonnegative and finite")
        if not 0 <= self.phase_exploration_fraction <= 1:
            raise ValueError("phase_exploration_fraction must lie in [0, 1]")
        if (self.phase_exploration_start_iteration < 0
                or self.phase_exploration_end_iteration
                <= self.phase_exploration_start_iteration):
            raise ValueError("Phase exploration schedule iterations must increase")
        if (not np.isfinite(self.phase_exploration_start_std)
                or self.phase_exploration_start_std < 0):
            raise ValueError("phase_exploration_start_std must be nonnegative and finite")
        if self.de_teacher_enabled and not self.topology_input_mode:
            raise ValueError("The DE teacher requires topology-input mode")
        if self.de_teacher_start_iteration < 0 or self.de_teacher_interval < 1:
            raise ValueError("DE teacher schedule must start nonnegative with a positive interval")
        if self.de_teacher_population < 5 or self.de_teacher_generations < 1:
            raise ValueError("DE teacher needs at least five members and one generation")
        if (not np.isfinite(self.de_teacher_mutation)
                or not 0 < self.de_teacher_mutation < 2):
            raise ValueError("DE teacher mutation must lie in (0, 2)")
        if (not np.isfinite(self.de_teacher_recombination)
                or not 0 <= self.de_teacher_recombination <= 1):
            raise ValueError("DE teacher recombination must lie in [0, 1]")
        if (not np.isfinite(self.de_teacher_loss_weight)
                or self.de_teacher_loss_weight < 0):
            raise ValueError("DE teacher loss weight must be nonnegative and finite")
        if (not np.isfinite(self.de_teacher_min_improvement)
                or self.de_teacher_min_improvement < 0):
            raise ValueError("DE teacher minimum improvement must be nonnegative and finite")
        if self.lexicographic_maximum_phase_error_deg <= 0 or not np.isfinite(self.lexicographic_maximum_phase_error_deg):
            raise ValueError("lexicographic maximum phase error must be positive and finite")
        if (not np.isfinite(self.lexicographic_initial_phase_error_deg)
                or self.lexicographic_initial_phase_error_deg < self.lexicographic_maximum_phase_error_deg):
            raise ValueError("initial phase error must be finite and at least the final maximum")
        if self.lexicographic_phase_error_anneal_iterations < 0:
            raise ValueError("phase error anneal iterations must be nonnegative")
        if not 0 <= self.lexicographic_minimum_phase_reward <= 1:
            raise ValueError("lexicographic minimum phase reward must lie in [0, 1]")
        if (not np.isfinite(self.lexicographic_initial_minimum_phase_reward)
                or not 0 <= self.lexicographic_initial_minimum_phase_reward
                <= self.lexicographic_minimum_phase_reward):
            raise ValueError("initial phase reward must lie between zero and the final minimum")
        if self.objective_mode == "lexicographic" and self.dense_reference_feasibility:
            raise ValueError("Lexicographic mode does not use a dense reference")
        if any((self.enable_connected_edge_budgets,
                self.enable_learned_connected_sparsity,
                self.enable_learned_backbone_density,
                self.enable_conditional_backbone_budget,
                self.enable_budget_error_selector,
                self.condition_on_edge_budget)):
            raise ValueError("This policy cannot use an older budget/tree mode")
        if self.link_cost_weight < 0 or self.coupling_cost_weight < 0:
            raise ValueError("Resource weights must be nonnegative")
        if self.feasible_sparsity_bonus_weight < 0:
            raise ValueError("feasible_sparsity_bonus_weight must be nonnegative")
        if self.coupling_cost_start_iteration < 0 or self.coupling_cost_ramp_iterations < 0:
            raise ValueError("coupling cost schedule iterations must be nonnegative")
        if self.exploration_offset_anneal_iterations < 1:
            raise ValueError("exploration_offset_anneal_iterations must be positive")
        if self.gate_head_hold_iterations < 0:
            raise ValueError("gate_head_hold_iterations must be nonnegative")
        if self.learning_rate_final <= 0:
            raise ValueError("learning_rate_final must be positive")
        if self.budget_reference_calibration_targets < 1:
            raise ValueError("budget_reference_calibration_targets must be positive")
        if self.targets_per_worker_task < 1:
            raise ValueError("targets_per_worker_task must be positive")
        if any(size < 2 or not np.isfinite(value) or value <= 0
               for size, value in self.coupling_budget_reference_by_m):
            raise ValueError("coupling budget references must be positive and finite")
        reference_sizes = [size for size, _ in self.coupling_budget_reference_by_m]
        if len(reference_sizes) != len(set(reference_sizes)):
            raise ValueError("coupling budget reference sizes must be unique")


class BernoulliSparseGNNPolicy(nn.Module):
    """Shared GNN/edge decoder with optional Bernoulli or topology inputs."""

    def __init__(self, config: BernoulliSparseConfig):
        super().__init__()
        self.config = config
        self.minimum_log_std = config.minimum_log_std
        self.maximum_log_std = config.maximum_log_std
        self.edge_conditioned_log_std = config.edge_conditioned_log_std
        self.smooth_edge_log_std_bounds = config.smooth_edge_log_std_bounds
        width = 4  # cos(target), sin(target), normalized detuning, physical span
        layers: list[nn.Module] = []
        for hidden in config.node_hidden_sizes:
            linear = nn.Linear(width, hidden)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.SiLU()))
            width = hidden
        linear = nn.Linear(width, config.node_embedding_dim)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.SiLU()))
        self.node_encoder = nn.Sequential(*layers)
        self.message_layers = nn.ModuleList(
            base.GraphMessagePassingLayer(
                config.node_embedding_dim, config.gnn_message_hidden_size,
                edge_feature_width=6 if config.topology_input_mode else 4,
            ) for _ in range(config.gnn_message_passing_steps)
        )
        edge_width = ((3 if config.topology_input_mode else 2) * config.node_embedding_dim
                      + (6 if config.topology_input_mode else 4))
        if config.condition_on_n_lasers:
            edge_width += 2
        edge_layers: list[nn.Module] = []
        for hidden in config.edge_hidden_sizes:
            linear = nn.Linear(edge_width, hidden)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            edge_layers.extend((linear, nn.SiLU()))
            edge_width = hidden
        self.edge_latent = nn.Sequential(*edge_layers)
        self.edge_output = nn.Linear(edge_width, 4 if config.edge_conditioned_log_std else 2)
        nn.init.orthogonal_(self.edge_output.weight, gain=0.01)
        nn.init.zeros_(self.edge_output.bias)
        with torch.no_grad():
            if config.strength_action_transform == "uncapped_softplus":
                # softplus(inverse_softplus(1))=1: the initial raw strength is
                # scale/(M-1), with no maximum on subsequent learned actions.
                self.edge_output.bias[0] = float(np.log(np.expm1(1.0)))
            else:
                fraction = config.initial_kappa_fraction
                self.edge_output.bias[0] = float(np.log(fraction / (1 - fraction)))
            if config.edge_conditioned_log_std:
                raw = config.initial_log_std
                if config.smooth_edge_log_std_bounds:
                    raw = base.raw_log_std_for_smooth_bound(
                        raw, config.minimum_log_std, config.maximum_log_std
                    )
                self.edge_output.bias[2:].fill_(float(raw))
        if not config.topology_input_mode:
            self.edge_gate_head = nn.Linear(edge_width, 1)
            nn.init.zeros_(self.edge_gate_head.weight)
            nn.init.constant_(self.edge_gate_head.bias, config.initial_gate_logit)
        if config.construct_rooted_backbone:
            self.root_head = nn.Linear(config.node_embedding_dim, 1)
            self.tree_edge_head = nn.Linear(edge_width, 1)
            nn.init.orthogonal_(self.root_head.weight, gain=0.01)
            nn.init.zeros_(self.root_head.bias)
            nn.init.orthogonal_(self.tree_edge_head.weight, gain=0.01)
            nn.init.zeros_(self.tree_edge_head.bias)
        self.log_std_kappa = nn.Parameter(torch.tensor(float(config.initial_log_std)))
        self.log_std_phi = nn.Parameter(torch.tensor(float(config.initial_log_std)))
        if config.objective_mode in {"lexicographic", "phase_only"}:
            # One target-dependent total budget, independent of the edge mask.
            self.budget_head = nn.Sequential(
                nn.Linear(config.node_embedding_dim + 1, config.node_embedding_dim),
                nn.SiLU(), nn.Linear(config.node_embedding_dim, 1),
            )
            nn.init.orthogonal_(self.budget_head[-1].weight, gain=0.01)
            initial_bias = (0.0 if config.budget_action_transform == "log_budget"
                            else float(np.log(np.expm1(1.0))))
            nn.init.constant_(self.budget_head[-1].bias, initial_bias)
            # Softplus budgets need broad initial policy exploration without
            # an external search ceiling; the existing variance curriculum
            # lowers this bound during refinement.
            initial_log_std = (0.0 if config.budget_action_transform == "log_budget"
                               else float(config.maximum_log_std))
            self.budget_log_std = nn.Parameter(torch.tensor(initial_log_std))

    def budget_parameters(self, features: torch.Tensor, *,
                          topology_masks: torch.Tensor | None = None,
                          pooled_states: torch.Tensor | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.objective_mode not in {"lexicographic", "phase_only"}:
            raise ValueError("The graph budget head is unavailable in weighted mode")
        if pooled_states is None:
            pooled_states = self._graph_context(features, topology_masks)[0].mean(dim=1)
        pooled = pooled_states
        batch, n_lasers, _ = features.shape
        size = features.new_full((batch, 1), float(n_lasers) / 10.0)
        mean = self.budget_head(torch.cat((pooled, size), dim=-1)).squeeze(-1)
        if self.config.budget_action_transform == "log_budget":
            mean = mean + float(np.log(n_lasers * self.config.log_budget_initial_per_laser))
            log_std = (float(np.log(self.config.budget_exploration_end_std))
                       + 0.25 * torch.tanh(self.budget_log_std)).expand_as(mean)
        else:
            log_std = self.budget_log_std.clamp(self.minimum_log_std, self.maximum_log_std).expand_as(mean)
        return mean, log_std

    def _graph_context(self, features: torch.Tensor,
                       topology_masks: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != 4:
            raise ValueError("features must have shape (batch, M, 4)")
        batch, n_lasers, _ = features.shape
        receivers_np, sources_np = base.directed_link_indices(n_lasers)
        receivers = torch.as_tensor(receivers_np, device=features.device)
        sources = torch.as_tensor(sources_np, device=features.device)
        cosine, sine, detuning = features[:, :, 0], features[:, :, 1], features[:, :, 2]
        local_cosine = cosine[:, sources] * cosine[:, receivers] + sine[:, sources] * sine[:, receivers]
        local_sine = sine[:, sources] * cosine[:, receivers] - cosine[:, sources] * sine[:, receivers]
        difference = 0.5 * (detuning[:, sources] - detuning[:, receivers])
        local = torch.stack((local_cosine, local_sine, difference, difference.abs()), dim=-1)
        if self.config.topology_input_mode:
            if topology_masks is None or topology_masks.shape != (batch, n_lasers * (n_lasers - 1)):
                raise ValueError("topology_masks must have shape (batch, M*(M-1))")
            topology_matrix = features.new_zeros((batch, n_lasers, n_lasers))
            topology_matrix[:, receivers, sources] = topology_masks
            reverse = topology_matrix[:, sources, receivers]
            local = torch.cat((local, topology_masks[..., None], reverse[..., None]), dim=-1)
        elif topology_masks is not None:
            raise ValueError("This policy does not accept an input topology")
        states = self.node_encoder(features)
        for layer in self.message_layers:
            states = layer(states, local, receivers, sources)
        return states, local, receivers, sources

    def parameters_for_edges(self, features: torch.Tensor, *, return_topology: bool = False,
                             topology_masks: torch.Tensor | None = None,
                             return_pooled: bool = False):
        states, local, receivers, sources = self._graph_context(features, topology_masks)
        batch, n_lasers, _ = features.shape
        edge_features = local
        n_edges = n_lasers * (n_lasers - 1)
        if self.config.condition_on_n_lasers:
            span = max(self.config.training_n_lasers) - min(self.config.training_n_lasers)
            scaled_size = 0.0 if span == 0 else 2 * (n_lasers - min(self.config.training_n_lasers)) / span - 1
            size = features.new_tensor((scaled_size, 1.0 / (n_lasers - 1))).view(1, 1, 2)
            edge_features = torch.cat((edge_features, size.expand(batch, n_edges, 2)), dim=-1)
        edge_inputs = torch.cat((states[:, sources], states[:, receivers], edge_features), dim=-1)
        if self.config.topology_input_mode:
            pooled = states.mean(dim=1)
            edge_inputs = torch.cat((edge_inputs, pooled[:, None, :].expand(-1, n_edges, -1)), dim=-1)
        latent = self.edge_latent(edge_inputs.reshape(batch * n_edges, -1))
        outputs = self.edge_output(latent).reshape(batch, n_edges, -1)
        strength_mean = (outputs[:, :, 0]
                         if self.config.strength_action_transform == "uncapped_softplus" else
                         6 * torch.tanh(outputs[:, :, 0] / 6))
        mean = torch.cat((strength_mean, outputs[:, :, 1]), dim=-1)
        if self.edge_conditioned_log_std:
            raw = torch.cat((outputs[:, :, 2], outputs[:, :, 3]), dim=-1)
            if self.smooth_edge_log_std_bounds:
                log_std = base.smoothly_bound_log_std(raw, self.minimum_log_std, self.maximum_log_std)
            else:
                log_std = raw.clamp(self.minimum_log_std, self.maximum_log_std)
        else:
            log_std = torch.cat((
                self.log_std_kappa.clamp(self.minimum_log_std, self.maximum_log_std).expand(n_edges),
                self.log_std_phi.clamp(self.minimum_log_std, self.maximum_log_std).expand(n_edges),
            )).expand(batch, 2 * n_edges)
        logits = (latent.new_zeros((batch, n_edges)) if self.config.topology_input_mode
                  else self.edge_gate_head(latent).reshape(batch, n_edges))
        if return_pooled:
            if not self.config.topology_input_mode or return_topology:
                raise ValueError("Pooled edge output is only used by topology-input mode")
            return mean, log_std, logits, pooled
        if return_topology:
            if not self.config.construct_rooted_backbone:
                raise ValueError("Tree scores require construct_rooted_backbone")
            root_logits = self.root_head(states).squeeze(-1)
            tree_scores = self.tree_edge_head(latent).reshape(batch, n_edges)
            return mean, log_std, logits, root_logits, tree_scores
        return mean, log_std, logits


def encode_context(targets_rad: np.ndarray, detunings_ghz: np.ndarray,
                   policy: BernoulliSparseGNNPolicy) -> torch.Tensor:
    targets = np.atleast_2d(np.asarray(targets_rad, dtype=float))
    detunings = np.atleast_2d(np.asarray(detunings_ghz, dtype=float))
    if targets.shape != detunings.shape:
        raise ValueError("target and detuning shapes must match")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(detunings)):
        raise ValueError("targets and detunings must be finite")
    relative = (base.canonicalize_global_phase_conjugate(targets)
                if policy.config.allow_global_phase_conjugate
                else base.make_relative_to_laser_1(targets))
    detunings = np.sort(detunings, axis=1)
    minimum = detunings.min(axis=1, keepdims=True)
    span = detunings.max(axis=1, keepdims=True) - minimum
    normalized = np.where(span > 1e-12, 2 * (detunings - minimum) / np.maximum(span, 1e-12) - 1, 0)
    columns = (np.cos(relative), np.sin(relative), normalized, np.broadcast_to(span, targets.shape))
    return torch.as_tensor(np.stack(columns, axis=-1).astype(np.float32),
                           device=next(policy.parameters()).device)


def coupling_cost_weight_at(iteration: int | None,
                            config: BernoulliSparseConfig) -> float:
    """Training uses a one-based iteration; evaluation uses the final weight."""
    if iteration is None:
        return config.coupling_cost_weight
    if iteration < config.coupling_cost_start_iteration:
        return 0.0
    if config.coupling_cost_ramp_iterations == 0:
        return config.coupling_cost_weight
    fraction = np.clip(
        (iteration - config.coupling_cost_start_iteration)
        / config.coupling_cost_ramp_iterations, 0.0, 1.0,
    )
    return float(config.coupling_cost_weight * fraction)


def root_reach_fractions(gates: np.ndarray, n_lasers: int) -> np.ndarray:
    """Largest directed reachable fraction for masks in receiver/source order."""
    masks = np.asarray(gates)
    n_edges = n_lasers * (n_lasers - 1)
    if n_lasers < 2 or masks.ndim < 1 or masks.shape[-1] != n_edges:
        raise ValueError("gates must end in M * (M - 1) directed edges")
    if np.any((masks != 0) & (masks != 1)):
        raise ValueError("gates must be binary")
    batch_shape = masks.shape[:-1]
    flat = masks.reshape(-1, n_edges).astype(bool)
    receivers, sources = base.directed_link_indices(n_lasers)
    reach = np.broadcast_to(np.eye(n_lasers, dtype=bool),
                            (len(flat), n_lasers, n_lasers)).copy()
    reach[:, sources, receivers] = flat
    for via in range(n_lasers):
        reach |= reach[:, :, via, None] & reach[:, None, via, :]
    largest = reach.sum(axis=-1).max(axis=-1)
    return ((largest - 1) / (n_lasers - 1)).reshape(batch_shape)


def reward_function(phase: np.ndarray, rho: np.ndarray, budget: np.ndarray,
                    config: BernoulliSparseConfig, *,
                    iteration: int | None = None,
                    root_reach_fraction: np.ndarray | None = None) -> np.ndarray:
    """Phase/resource objective; non-rooted masks rank below every rooted mask."""
    phase = np.asarray(phase, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if config.sparsity_reward_mode == "links_per_laser":
        n_lasers = config.n_lasers
        n_edges = n_lasers * (n_lasers - 1)
        sparsity = np.maximum(rho * n_edges - (n_lasers - 1), 0.0) / n_lasers
    else:
        sparsity = rho
    cost = (config.link_cost_weight * sparsity
            + coupling_cost_weight_at(iteration, config) * budget)
    score = phase / (1.0 + cost)
    if not config.require_rooted_connectivity:
        return score
    if root_reach_fraction is None:
        raise ValueError("root_reach_fraction is required for rooted connectivity")
    reach = np.asarray(root_reach_fraction, dtype=float)
    if reach.shape != score.shape or np.any(~np.isfinite(reach)) or np.any((reach < 0) | (reach > 1)):
        raise ValueError("root_reach_fraction must match the reward shape and lie in [0, 1]")
    return np.where(reach == 1.0, score, reach - 1.0)


def dense_reference_reward(phase: np.ndarray, dense_phase: np.ndarray,
                           rho: np.ndarray, *,
                           sparsity_bonus_weight: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Keep absolute phase primary; unlock a small link bonus only if feasible."""
    phase = np.asarray(phase, dtype=float)
    dense_phase = np.asarray(dense_phase, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if phase.shape != rho.shape or dense_phase.shape != phase.shape[:-1] + (1,):
        raise ValueError("dense phase must have one value per target")
    if np.any(~np.isfinite(phase)) or np.any(~np.isfinite(dense_phase)):
        raise ValueError("phase rewards must be finite")
    if sparsity_bonus_weight < 0:
        raise ValueError("sparsity_bonus_weight must be nonnegative")
    feasible = phase >= dense_phase
    # Unlike a feasibility-first discontinuity, a weak initial dense policy
    # cannot grant a large sparsity reward before phase accuracy is learned.
    reward = phase + feasible * sparsity_bonus_weight * phase * (1.0 - rho)
    return reward, feasible


def calibrated_logit_bias(logits: torch.Tensor, active_fraction: float) -> torch.Tensor:
    """Find one scalar offset so mean Bernoulli probability equals the request."""
    if not 0.0 < active_fraction < 1.0:
        raise ValueError("active_fraction must be strictly between 0 and 1")
    detached = logits.detach()
    lower = float(-40.0 - detached.max().item())
    upper = float(40.0 - detached.min().item())
    for _ in range(45):
        middle = 0.5 * (lower + upper)
        if float(torch.sigmoid(detached + middle).mean()) < active_fraction:
            lower = middle
        else:
            upper = middle
    return logits.new_tensor(0.5 * (lower + upper))


def decode_continuous_actions(actions: np.ndarray, config: BernoulliSparseConfig
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Decode Bernoulli-policy strengths/phases without changing old policies.

    The softplus mode has no magnitude clipping or upper bound. Its fixed
    scale sets the Gaussian action's physical units and initial conditioning;
    it is not a maximum strength. Degree normalization preserves the original
    M scaling but imposes no bound on the learned action.
    """
    actions = np.asarray(actions, dtype=float)
    n_edges = config.n_lasers * (config.n_lasers - 1)
    if actions.ndim != 2 or actions.shape[1] != 2 * n_edges or not np.all(np.isfinite(actions)):
        raise ValueError("Continuous actions must be finite with two values per directed edge")
    if config.strength_action_transform == "bounded_sigmoid":
        return base.decode_action(
            actions, config.maximum_kappa_per_ns, config.n_lasers,
            normalize_incoming_coupling_by_degree=config.normalize_incoming_coupling_by_degree,
        )
    divisor = config.n_lasers - 1 if config.normalize_incoming_coupling_by_degree else 1
    strengths = (config.softplus_strength_scale_per_ns / divisor
                 * np.logaddexp(0.0, actions[:, :n_edges]))
    phases = np.angle(np.exp(1j * actions[:, n_edges:]))
    receivers, sources = base.directed_link_indices(config.n_lasers)
    kappa = np.zeros((len(actions), config.n_lasers, config.n_lasers), dtype=float)
    phi = np.zeros_like(kappa)
    kappa[:, receivers, sources] = strengths
    phi[:, receivers, sources] = phases
    return kappa, phi


def decode_samples(actions: torch.Tensor, gates: torch.Tensor,
                   config: BernoulliSparseConfig, *, budget_actions: torch.Tensor | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask edges, then redistribute each candidate's raw total."""
    if actions.ndim != 3 or gates.ndim != 3:
        raise ValueError("actions and gates must have (target, candidate, edge) axes")
    targets, candidates, n_edges = gates.shape
    if actions.shape != (targets, candidates, 2 * n_edges):
        raise ValueError("actions and gates have incompatible shapes")
    if n_edges != config.n_lasers * (config.n_lasers - 1):
        raise ValueError("gate count does not match n_lasers")
    action_np = actions.detach().cpu().numpy().reshape(targets * candidates, -1)
    gate_np = gates.detach().cpu().numpy().reshape(targets * candidates, -1)
    raw_kappa, phi = decode_continuous_actions(action_np, config)
    receivers, sources = base.directed_link_indices(config.n_lasers)
    raw_edges = raw_kappa[:, receivers, sources]
    if budget_actions is not None:
        if budget_actions.shape != (targets, candidates):
            raise ValueError("budget_actions must have shape (targets, candidates)")
        raw_budget_actions = budget_actions.detach().cpu().numpy().reshape(-1)
        if config.budget_action_transform == "log_budget":
            budget = np.exp(raw_budget_actions)
        else:
            budget = (config.softplus_strength_scale_per_ns * config.n_lasers
                      * np.logaddexp(0.0, raw_budget_actions))
        if not np.all(np.isfinite(budget)):
            raise ValueError("Decoded coupling budgets must be finite")
        totals = raw_edges.sum(axis=1, keepdims=True)
        raw_edges = raw_edges * (budget[:, None] / totals)
    active = gate_np.astype(bool)
    if not np.array_equal(gate_np, active.astype(gate_np.dtype)):
        raise ValueError("Bernoulli gates must be exactly zero or one")
    cap = (None if config.uncapped_redistribution else
           base.effective_maximum_kappa_per_link(
               config.maximum_kappa_per_ns, config.n_lasers,
               config.normalize_incoming_coupling_by_degree,
           ))
    used_edges, shortfall = redistribute_coupling(
        raw_edges, active, cap,
    )
    kappa = np.zeros_like(raw_kappa)
    kappa[:, receivers, sources] = used_edges
    counts = gate_np.sum(axis=1).reshape(targets, candidates).astype(int)
    return (kappa.reshape(targets, candidates, config.n_lasers, config.n_lasers),
            phi.reshape(targets, candidates, config.n_lasers, config.n_lasers),
            counts, counts / n_edges, shortfall.reshape(targets, candidates))


def budget_exploration_at(iteration: int | None, config: BernoulliSparseConfig) -> tuple[float, float]:
    """Training offset width (legacy mode) and Gaussian standard deviation."""
    if iteration is None:
        return 0.0, config.budget_exploration_end_std
    fraction = np.clip((iteration - config.budget_exploration_start_iteration)
                       / (config.budget_exploration_end_iteration
                          - config.budget_exploration_start_iteration), 0.0, 1.0)
    blend = 0.5 * (1.0 + np.cos(np.pi * fraction))
    width = (np.log(config.budget_exploration_end_factor)
             + blend * (np.log(config.budget_exploration_start_factor)
                        - np.log(config.budget_exploration_end_factor)))
    std = (config.budget_exploration_end_std
           + blend * (config.budget_exploration_start_std
                      - config.budget_exploration_end_std))
    return float(width), float(std)


def strength_exploration_at(iteration: int | None, config: BernoulliSparseConfig) -> float:
    """Extra raw-strength standard deviation for wide training samples."""
    if iteration is None or config.strength_exploration_fraction == 0:
        return 0.0
    fraction = np.clip((iteration - config.strength_exploration_start_iteration)
                       / (config.strength_exploration_end_iteration
                          - config.strength_exploration_start_iteration), 0.0, 1.0)
    return float(config.strength_exploration_start_std
                 * 0.5 * (1.0 + np.cos(np.pi * fraction)))


def phase_exploration_at(iteration: int | None, config: BernoulliSparseConfig) -> float:
    """Extra raw-phase standard deviation for wide training samples."""
    if iteration is None or config.phase_exploration_fraction == 0:
        return 0.0
    fraction = np.clip((iteration - config.phase_exploration_start_iteration)
                       / (config.phase_exploration_end_iteration
                          - config.phase_exploration_start_iteration), 0.0, 1.0)
    return float(config.phase_exploration_start_std
                 * 0.5 * (1.0 + np.cos(np.pi * fraction)))


def _topology_action_std(log_std: torch.Tensor, config: BernoulliSparseConfig,
                         iteration: int | None, continuous_mean: bool) -> torch.Tensor:
    """Keep local samples while widening selected strength or phase actions."""
    action_std = log_std.exp()
    strength_wide_std = strength_exploration_at(iteration, config)
    phase_wide_std = phase_exploration_at(iteration, config)
    strength_wide_count = round(config.actions_per_topology * config.strength_exploration_fraction)
    phase_wide_count = round(config.actions_per_topology * config.phase_exploration_fraction)
    if continuous_mean or ((strength_wide_std <= 0 or strength_wide_count == 0)
                           and (phase_wide_std <= 0 or phase_wide_count == 0)):
        return action_std
    _, candidates, n_actions = log_std.shape
    if candidates % config.actions_per_topology:
        raise ValueError("Training candidates must contain complete topology groups")
    n_edges = n_actions // 2
    slots = torch.arange(candidates, device=log_std.device) % config.actions_per_topology
    strength_extra = (slots < strength_wide_count).to(action_std.dtype)[None, :, None] * strength_wide_std
    phase_extra = (slots < phase_wide_count).to(action_std.dtype)[None, :, None] * phase_wide_std
    strength_std = torch.sqrt(action_std[..., :n_edges].square() + strength_extra.square())
    phase_std = torch.sqrt(action_std[..., n_edges:].square() + phase_extra.square())
    return torch.cat((strength_std, phase_std), dim=-1)


def _budget_offsets(targets: int, candidates: int, width: float,
                    rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    rows = np.empty((targets, candidates), dtype=np.float32)
    for index in range(targets):
        if candidates == 1:
            row = np.zeros(1, dtype=np.float32)
        elif candidates == 2:
            row = np.array([-width, width], dtype=np.float32)
        else:
            row = np.r_[np.linspace(-width, width, candidates - 1,
                                    dtype=np.float32), np.float32(0.0)]
        rng.shuffle(row)
        rows[index] = row
    return torch.as_tensor(rows, device=device)


def _budget_action_parameters(
    mean: torch.Tensor, log_std: torch.Tensor, config: BernoulliSparseConfig,
    iteration: int | None, rng: np.random.Generator | None, group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the actual training Normal parameters for each budget action."""
    base_std = log_std.exp()
    if (config.budget_action_transform != "log_budget" or iteration is None
            or config.budget_exploration_mode == "local"):
        return mean, base_std
    batch, candidates = mean.shape
    if group_size < 1 or candidates % group_size:
        raise ValueError("Training candidates must contain complete budget groups")
    _, scheduled_std = budget_exploration_at(iteration, config)
    if config.budget_exploration_mode == "mixed_std":
        wide_count = round(group_size * config.budget_wide_exploration_fraction)
        wide_slots = (torch.arange(candidates, device=mean.device) % group_size) < wide_count
        wide_std = base_std * (scheduled_std / config.budget_exploration_end_std)
        return mean, torch.where(wide_slots[None, :], wide_std, base_std)
    width, _ = budget_exploration_at(iteration, config)
    offsets = _budget_offsets(
        batch * (candidates // group_size), group_size, width,
        rng if rng is not None else np.random.default_rng(), mean.device,
    ).reshape(batch, candidates)
    return mean + offsets, base_std * (scheduled_std / config.budget_exploration_end_std)


def sample_budget_actions(policy: BernoulliSparseGNNPolicy, features: torch.Tensor,
                          count: int, *, continuous_mean: bool = False,
                          iteration: int | None = None,
                          rng: np.random.Generator | None = None,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_std = policy.budget_parameters(features)
    action_mean, action_std = _budget_action_parameters(
        mean[:, None].expand(-1, count), log_std[:, None].expand(-1, count),
        policy.config, iteration, rng, count)
    normal = torch.distributions.Normal(action_mean, action_std)
    actions = (mean[:, None].expand(-1, count) if continuous_mean else normal.sample())
    return actions, normal.log_prob(actions)


def maximum_phase_error_deg(aligned: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Maximum settled circular mean error, excluding the reference laser."""
    aligned = np.asarray(aligned, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if aligned.shape[-1] != targets.shape[-1]:
        raise ValueError("aligned phases and targets have different laser counts")
    delta = np.angle(np.exp(1j * (aligned - targets)))
    return np.rad2deg(np.max(np.abs(delta[..., 1:]), axis=-1))


def lexicographic_feasibility_progress_at(iteration: int, config: BernoulliSparseConfig) -> float:
    """Shared training curriculum progress; validation uses final thresholds."""
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    duration = config.lexicographic_phase_error_anneal_iterations
    return min(iteration / duration, 1.0) if duration else 1.0


def lexicographic_phase_error_limit_at(iteration: int, config: BernoulliSparseConfig) -> float:
    """Training-only phase-error allowance; validation uses the fixed final limit."""
    progress = lexicographic_feasibility_progress_at(iteration, config)
    return (config.lexicographic_initial_phase_error_deg
            + progress * (config.lexicographic_maximum_phase_error_deg
                          - config.lexicographic_initial_phase_error_deg))


def lexicographic_phase_reward_limit_at(iteration: int, config: BernoulliSparseConfig) -> float:
    """Training-only tail phase-reward floor on the same curriculum."""
    progress = lexicographic_feasibility_progress_at(iteration, config)
    return (config.lexicographic_initial_minimum_phase_reward
            + progress * (config.lexicographic_minimum_phase_reward
                          - config.lexicographic_initial_minimum_phase_reward))


def lexicographic_candidate_scores(
    phase: np.ndarray, error_deg: np.ndarray, link_counts: np.ndarray,
    budgets: np.ndarray, config: BernoulliSparseConfig,
    *, rooted: np.ndarray | None = None, maximum_phase_error_deg: float | None = None,
    minimum_phase_reward: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Within-target ordinal score, best=1. Phase always precedes resources."""
    phase = np.asarray(phase, dtype=float)
    error_deg = np.asarray(error_deg, dtype=float)
    link_counts = np.asarray(link_counts)
    budgets = np.asarray(budgets, dtype=float)
    if not (phase.shape == error_deg.shape == link_counts.shape == budgets.shape) or phase.ndim != 2:
        raise ValueError("candidate arrays must have matching (target, candidate) shapes")
    rooted = np.ones_like(phase, dtype=bool) if rooted is None else np.asarray(rooted, dtype=bool)
    if rooted.shape != phase.shape:
        raise ValueError("rooted shape must match candidates")
    phase_error_limit = (config.lexicographic_maximum_phase_error_deg
                         if maximum_phase_error_deg is None else maximum_phase_error_deg)
    if not np.isfinite(phase_error_limit) or phase_error_limit <= 0:
        raise ValueError("maximum phase error must be positive and finite")
    phase_reward_limit = (config.lexicographic_minimum_phase_reward
                          if minimum_phase_reward is None else minimum_phase_reward)
    if not np.isfinite(phase_reward_limit) or not 0 <= phase_reward_limit <= 1:
        raise ValueError("minimum phase reward must lie in [0, 1]")
    feasible = (rooted & np.isfinite(phase) & np.isfinite(error_deg)
                & (phase >= phase_reward_limit)
                & (error_deg <= phase_error_limit))
    scores = np.empty_like(phase)
    count = phase.shape[1]
    for row in range(len(phase)):
        primary = link_counts[row] if config.lexicographic_order == "sparsity_budget" else budgets[row]
        secondary = budgets[row] if config.lexicographic_order == "sparsity_budget" else link_counts[row]
        # lexsort is ascending, with the final key highest priority.
        order = np.lexsort((np.arange(count), phase[row],
                            -np.where(feasible[row], secondary, 0.0),
                            -np.where(feasible[row], primary, 0.0),
                            feasible[row].astype(int), rooted[row].astype(int)))
        scores[row, order] = (np.arange(count) + 0.5) / count
    return scores, feasible


def redistribute_coupling(raw_edges: np.ndarray, active: np.ndarray,
                          cap: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Redistribute raw coupling over active links, optionally respecting a cap.

    With ``cap=None``, each rooted mask preserves the complete raw total and
    active strengths keep their relative proportions. A finite cap preserves
    the legacy saturation/shortfall behavior. An all-off mask remains exactly
    uncoupled and reports its entire raw total as shortfall.
    """
    raw_edges = np.asarray(raw_edges, dtype=float)
    active = np.asarray(active, dtype=bool)
    if raw_edges.ndim != 2 or raw_edges.shape != active.shape:
        raise ValueError("raw_edges and active must have matching 2D shapes")
    if np.any(~np.isfinite(raw_edges)) or (cap is not None and (cap <= 0 or not np.isfinite(cap))):
        raise ValueError("strengths and cap must be finite, with a positive cap")
    if np.any(raw_edges < 0) or (cap is not None and np.any((raw_edges == 0) | (raw_edges > cap * (1 + 1e-12)))):
        raise ValueError("raw strengths must be nonnegative and within any link cap")
    retained = np.where(active, raw_edges, 0.0)
    raw_total = raw_edges.sum(axis=1)
    active_count = active.sum(axis=1)
    if cap is None:
        retained_total = retained.sum(axis=1)
        scale = np.divide(raw_total, retained_total, out=np.zeros_like(raw_total),
                          where=retained_total > 0)
        used = retained * scale[:, None]
        no_retained_strength = (active_count > 0) & (retained_total == 0) & (raw_total > 0)
        if np.any(no_retained_strength):
            used[no_retained_strength] = (active[no_retained_strength]
                                          * (raw_total[no_retained_strength]
                                             / active_count[no_retained_strength])[:, None])
        return used, np.where(active_count > 0, 0.0, raw_total)
    capacity = active_count * cap
    target = np.minimum(raw_total, capacity)
    used = retained.copy()
    saturated = (active_count > 0) & (raw_total >= capacity)
    used[saturated] = active[saturated] * cap
    needs_redistribution = (active_count > 0) & ~saturated & (
        retained.sum(axis=1) < target - 1e-12
    )
    if np.any(needs_redistribution):
        rows = retained[needs_redistribution]
        desired = target[needs_redistribution]
        minimum = np.where(rows > 0, rows, np.inf).min(axis=1)
        lower = np.ones(len(rows))
        upper = cap / minimum
        for _ in range(48):
            middle = (lower + upper) / 2
            total = np.minimum(cap, rows * middle[:, None]).sum(axis=1)
            lower = np.where(total < desired, middle, lower)
            upper = np.where(total >= desired, middle, upper)
        used[needs_redistribution] = np.minimum(cap, rows * upper[:, None])
    shortfall = np.maximum(raw_total - used.sum(axis=1), 0.0)
    return used, shortfall


def select_rooted_tree(root_logits: torch.Tensor, tree_scores: torch.Tensor,
                       n_candidates: int, *, deterministic: bool = False,
                       uniform_random: bool = False
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grow a directed arborescence, optionally sampling uniformly at each choice."""
    batch, n_lasers = root_logits.shape
    n_edges = n_lasers * (n_lasers - 1)
    if tree_scores.shape != (batch, n_edges) or n_candidates < 1:
        raise ValueError("Root and edge scores have incompatible shapes")
    if deterministic and uniform_random:
        raise ValueError("A uniformly random tree cannot be selected deterministically")
    receivers_np, sources_np = base.directed_link_indices(n_lasers)
    receivers = torch.as_tensor(receivers_np, device=root_logits.device)
    sources = torch.as_tensor(sources_np, device=root_logits.device)
    flat = batch * n_candidates
    root_rows = root_logits[:, None, :].expand(batch, n_candidates, n_lasers).reshape(flat, n_lasers)
    edge_rows = tree_scores[:, None, :].expand(batch, n_candidates, n_edges).reshape(flat, n_edges)
    if uniform_random:
        # Neither the root nor required-edge choices depend on learned scores.
        root_rows = torch.zeros_like(root_rows)
        edge_rows = torch.zeros_like(edge_rows)
    if deterministic:
        roots = root_rows.argmax(dim=-1)
        log_prob = root_rows.new_zeros(flat)
    else:
        root_dist = torch.distributions.Categorical(logits=root_rows)
        roots = root_dist.sample()
        log_prob = (root_rows.new_zeros(flat) if uniform_random
                    else root_dist.log_prob(roots))
    reached = torch.zeros((flat, n_lasers), dtype=torch.bool, device=root_logits.device)
    reached.scatter_(1, roots[:, None], True)
    mask = torch.zeros((flat, n_edges), dtype=torch.bool, device=root_logits.device)
    for _ in range(n_lasers - 1):
        eligible = reached[:, sources] & ~reached[:, receivers]
        candidates = edge_rows.masked_fill(~eligible, -torch.inf)
        if deterministic:
            chosen = candidates.argmax(dim=-1)
        else:
            edge_dist = torch.distributions.Categorical(logits=candidates)
            chosen = edge_dist.sample()
            if not uniform_random:
                log_prob = log_prob + edge_dist.log_prob(chosen)
        mask.scatter_(1, chosen[:, None], True)
        reached.scatter_(1, receivers[chosen, None], True)
    if not bool(reached.all()):
        raise RuntimeError("Constructed tree did not reach every laser")
    return (mask.reshape(batch, n_candidates, n_edges),
            log_prob.reshape(batch, n_candidates), roots.reshape(batch, n_candidates))


def sample_candidates(policy: BernoulliSparseGNNPolicy, features: torch.Tensor,
                      n_candidates: int, offsets: torch.Tensor | None = None,
                      *, continuous_mean: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Gaussian actions and Bernoulli gates, returning joint log-prob."""
    batch = features.shape[0]
    if policy.config.construct_rooted_backbone:
        mean, log_std, logits, root_logits, tree_scores = policy.parameters_for_edges(
            features, return_topology=True)
    else:
        mean, log_std, logits = policy.parameters_for_edges(features)
    n_edges = logits.shape[-1]
    if offsets is None:
        offsets = features.new_zeros((batch, n_candidates))
    if offsets.shape != (batch, n_candidates):
        raise ValueError("offsets must have shape (targets, candidates)")
    normal = torch.distributions.Normal(mean[:, None], log_std[:, None].exp())
    actions = (mean[:, None].expand(batch, n_candidates, 2 * n_edges).clone()
               if continuous_mean else
               torch.distributions.Normal(mean, log_std.exp()).sample((n_candidates,)).permute(1, 0, 2))
    gates_dist = torch.distributions.Bernoulli(logits=logits[:, None, :] + offsets[:, :, None])
    optional_gates = gates_dist.sample()
    if policy.config.construct_rooted_backbone:
        tree, tree_log_prob, _ = select_rooted_tree(
            root_logits, tree_scores, n_candidates,
            uniform_random=policy.config.uniform_random_rooted_tree)
        gates = torch.maximum(optional_gates, tree.to(optional_gates.dtype))
        gate_log_prob = (gates_dist.log_prob(optional_gates) * ~tree).sum(dim=-1) + tree_log_prob
    else:
        gates = optional_gates
        gate_log_prob = gates_dist.log_prob(gates).sum(dim=-1)
    # Every predicted strength affects redistribution if any edge is on.
    # With an all-off mask, no strength or phase affects the simulator.
    scored_strength = (gates.sum(dim=-1, keepdim=True) > 0).to(gates.dtype).expand_as(gates)
    scored = torch.cat((scored_strength, gates), dim=-1)
    continuous_log_prob = (normal.log_prob(actions) * scored).sum(dim=-1)
    return actions, gates, continuous_log_prob + gate_log_prob, logits


def topology_min_fraction_at(iteration: int, config: BernoulliSparseConfig) -> float:
    """Lower the minimum link count from all-to-all to the size-specific floor."""
    start = config.topology_curriculum_start_iteration
    end = config.topology_curriculum_end_iteration
    if start is None:
        return config.training_topology_min_fraction
    progress = np.clip((iteration - start) / (end - start), 0.0, 1.0)
    final_fraction = max(config.training_topology_min_fraction, 1.0 / config.n_lasers)
    return float(1.0 + progress * (final_fraction - 1.0))


def sample_stratified_rooted_topologies(
    n_targets: int, n_topologies: int, n_lasers: int,
    rng: np.random.Generator, device: torch.device, *,
    min_fraction: float = 0.0, density_bias: float = 1.0,
    fixed_star: bool = False,
) -> torch.Tensor:
    """Sample rooted masks from a random tree or an outward star to all-to-all."""
    if n_targets < 1 or n_topologies < 1 or n_lasers < 2:
        raise ValueError("Topology sampler needs positive target/topology counts and M >= 2")
    if not np.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
        raise ValueError("Topology minimum fraction must lie in [0, 1]")
    if not np.isfinite(density_bias) or density_bias <= 0:
        raise ValueError("Topology density bias must be positive and finite")
    n_edges = n_lasers * (n_lasers - 1)
    # A fraction such as 1/M can round just above the exact integer M-1.
    min_links = max(n_lasers - 1, int(np.ceil(min_fraction * n_edges - 1e-10)))
    if fixed_star:
        _, sources = base.directed_link_indices(n_lasers)
        star_edges = torch.as_tensor(sources == 0, device=device)
        tree = star_edges[None, None, :].expand(n_targets, n_topologies, -1)
    else:
        roots = torch.zeros((n_targets, n_lasers), device=device)
        edge_scores = torch.zeros((n_targets, n_edges), device=device)
        tree, _, _ = select_rooted_tree(
            roots, edge_scores, n_topologies, uniform_random=True)
    levels = np.arange(min_links, n_edges + 1)
    if n_topologies == 1:
        if density_bias == 1.0:
            counts = rng.choice(levels, size=(n_targets, 1))
        else:
            quantiles = rng.random((n_targets, 1))
            counts = np.rint(min_links + (n_edges - min_links)
                            * (1 - (1 - quantiles) ** density_bias)).astype(int)
    elif density_bias != 1.0:
        quantiles = np.linspace(0.0, 1.0, n_topologies)
        spaced = np.rint(min_links + (n_edges - min_links)
                         * (1 - (1 - quantiles) ** density_bias)).astype(int)
        counts = np.stack([rng.permutation(spaced) for _ in range(n_targets)])
    elif n_topologies >= len(levels):
        repeats, remainder = divmod(n_topologies, len(levels))
        repeated = np.tile(levels, repeats)
        counts = np.stack([
            rng.permutation(np.r_[repeated, rng.choice(levels, remainder, replace=False)])
            for _ in range(n_targets)
        ])
    else:
        spaced = np.rint(np.linspace(min_links, n_edges, n_topologies)).astype(int)
        counts = np.stack([rng.permutation(spaced) for _ in range(n_targets)])
    extra = torch.as_tensor(counts - (n_lasers - 1), device=device)
    scores = torch.as_tensor(rng.random(tree.shape), device=device, dtype=torch.float32)
    ranks = scores.masked_fill(tree, torch.inf).argsort(dim=-1).argsort(dim=-1)
    masks = tree | (ranks < extra[..., None])
    if not torch.all(masks.sum(dim=-1) == torch.as_tensor(counts, device=device)):
        raise RuntimeError("Topology sampler did not produce the requested link counts")
    return masks.to(torch.float32)


def sample_topology_conditioned_candidates(
    policy: BernoulliSparseGNNPolicy, features: torch.Tensor,
    masks: torch.Tensor, *, continuous_mean: bool = False,
    iteration: int | None = None, rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Gaussian edge and budget actions conditioned on each supplied mask."""
    if not policy.config.topology_input_mode:
        raise ValueError("Topology-conditioned actions require topology_input_mode")
    batch, candidates, n_edges = masks.shape
    n_lasers = features.shape[1]
    if features.shape != (batch, n_lasers, 4) or n_edges != n_lasers * (n_lasers - 1):
        raise ValueError("Features and topology masks have incompatible shapes")
    repeated_features = features[:, None].expand(-1, candidates, -1, -1).reshape(
        batch * candidates, n_lasers, 4)
    flat_masks = masks.reshape(batch * candidates, n_edges)
    mean, log_std, _, pooled = policy.parameters_for_edges(
        repeated_features, topology_masks=flat_masks, return_pooled=True)
    mean = mean.reshape(batch, candidates, 2 * n_edges)
    log_std = log_std.reshape(batch, candidates, 2 * n_edges)
    normal = torch.distributions.Normal(
        mean, _topology_action_std(log_std, policy.config, iteration, continuous_mean))
    actions = mean.clone() if continuous_mean else normal.sample()
    active_actions = torch.cat((masks, masks), dim=-1)
    log_prob = (normal.log_prob(actions) * active_actions).sum(dim=-1)

    budget_mean, budget_log_std = policy.budget_parameters(
        repeated_features, pooled_states=pooled)
    budget_mean = budget_mean.reshape(batch, candidates)
    budget_log_std = budget_log_std.reshape(batch, candidates)
    action_mean, action_std = _budget_action_parameters(
        budget_mean, budget_log_std, policy.config, iteration, rng,
        policy.config.actions_per_topology)
    budget_normal = torch.distributions.Normal(action_mean, action_std)
    budget_actions = budget_mean.clone() if continuous_mean else budget_normal.sample()
    return actions, budget_actions, log_prob + budget_normal.log_prob(budget_actions)


def exploration_offset_at(iteration: int, config: BernoulliSparseConfig) -> float:
    fraction = min(max(iteration / config.exploration_offset_anneal_iterations, 0.0), 1.0)
    return (1 - fraction) * config.exploration_logit_offset_start + fraction * config.exploration_logit_offset_end


def _training_offsets(targets: int, candidates: int, amplitude: float,
                      rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    # Stratification spans both dense and sparse designs even at M=3; a
    # shared per-candidate offset avoids narrow binomial density coverage.
    rows = np.empty((targets, candidates), dtype=np.float32)
    for index in range(targets):
        row = np.linspace(-amplitude, amplitude, candidates, dtype=np.float32)
        row += rng.uniform(-amplitude / candidates, amplitude / candidates, candidates)
        row[0] = 0.0  # Always train the unbiased deployment distribution.
        rng.shuffle(row)
        rows[index] = row
    return torch.as_tensor(rows, device=device)


def calibrate_budget_references(
    policy: BernoulliSparseGNNPolicy, config: BernoulliSparseConfig,
) -> BernoulliSparseConfig:
    """Freeze a median initial all-on budget for every trained M, without simulation."""
    if config.coupling_budget_reference_by_m:
        return config
    references: list[tuple[int, float]] = []
    was_training = policy.training
    policy.eval()
    with torch.inference_mode():
        for n_lasers in config.training_n_lasers:
            local = replace(config, n_lasers=n_lasers)
            count = config.budget_reference_calibration_targets
            targets = base.sample_target_phases(
                count, np.random.default_rng(config.random_seed + 40000 + n_lasers),
                n_lasers, config.structured_target_fraction,
                config.structured_target_jitter_rad,
            )
            detunings = base.sample_detuning_distributions(
                count, np.random.default_rng(config.random_seed + 50000 + n_lasers),
                local, iteration=config.detuning_curriculum_iterations,
            )
            targets, detunings = base.sort_by_detuning(targets, detunings)
            features = encode_context(targets, detunings, policy)
            if config.topology_input_mode:
                all_on = features.new_ones((count, n_lasers * (n_lasers - 1)))
                mean, _, _ = policy.parameters_for_edges(features, topology_masks=all_on)
            else:
                mean, _, _ = policy.parameters_for_edges(features)
            kappa, _ = decode_continuous_actions(mean.detach().cpu().numpy(), local)
            references.append((n_lasers, float(np.median(kappa.sum(axis=(1, 2))))))
    policy.train(was_training)
    return replace(config, coupling_budget_reference_by_m=tuple(references))


def reference_budget_for_m(config: BernoulliSparseConfig, n_lasers: int) -> float:
    """Use frozen per-M values and preserve the decoder's size scaling."""
    references = dict(config.coupling_budget_reference_by_m)
    if not references:
        raise ValueError("Coupling budget references have not been calibrated")
    if n_lasers in references:
        return references[n_lasers]
    # With incoming-degree normalization, each raw edge's output scale
    # falls as 1/(M-1), so an all-on total scales approximately as M.
    # Otherwise it scales with the M(M-1) possible directed edges.
    def size_factor(size: int) -> int:
        return size if config.normalize_incoming_coupling_by_degree else size * (size - 1)

    unit_reference = np.median([
        budget / size_factor(size) for size, budget in references.items()
    ])
    return float(size_factor(n_lasers) * unit_reference)


def relative_coupling_budget(kappa: np.ndarray,
                             config: BernoulliSparseConfig) -> np.ndarray:
    """Actual total strength divided by the frozen, M-dependent reference."""
    return np.asarray(kappa).sum(axis=(-2, -1)) / reference_budget_for_m(
        config, config.n_lasers
    )


def dense_reference_designs(
    policy: BernoulliSparseGNNPolicy, targets: np.ndarray,
    detunings: np.ndarray, config: BernoulliSparseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the current policy's continuous mean with every gate on."""
    count = len(targets)
    with torch.inference_mode():
        current_mean = policy.parameters_for_edges(
            encode_context(targets, detunings, policy))[0]
    current_kappa, current_phi = decode_continuous_actions(current_mean.cpu().numpy(), config)
    shape = (count, 1, config.n_lasers, config.n_lasers)
    return current_kappa.reshape(shape), current_phi.reshape(shape)


def _standardized_advantages(reward: np.ndarray) -> np.ndarray:
    centered = reward - reward.mean(axis=1, keepdims=True)
    return centered / np.maximum(reward.std(axis=1, keepdims=True), 1e-6)


def _topology_advantages(rewards: list[np.ndarray], config: BernoulliSparseConfig,
                         reward_scale_state: dict[str, float] | None = None) -> list[np.ndarray]:
    """Center within each supplied mask; optionally share one running reward scale."""
    groups = [reward.reshape(len(reward), -1, config.actions_per_topology)
              for reward in rewards]
    if config.topology_advantage_scale == "global_running":
        # Across all M, targets, masks, and sampled actions in this iteration.
        batch_variance = float(np.var(np.concatenate([reward.ravel() for reward in rewards])))
        if reward_scale_state is None:
            running_variance = batch_variance
        else:
            previous = reward_scale_state.get("variance")
            running_variance = (batch_variance if previous is None else
                                config.global_reward_std_ema_decay * previous
                                + (1 - config.global_reward_std_ema_decay) * batch_variance)
            reward_scale_state["variance"] = running_variance
        scale = max(np.sqrt(running_variance), 1e-6)
        return [((group - group.mean(axis=-1, keepdims=True)) / scale
                 ).reshape(reward.shape) for group, reward in zip(groups, rewards)]
    return [((group - group.mean(axis=-1, keepdims=True)) /
             np.maximum(group.std(axis=-1, keepdims=True), 1e-6)
             ).reshape(reward.shape) for group, reward in zip(groups, rewards)]


def topology_density_band_ids(active_link_counts: np.ndarray, n_lasers: int) -> np.ndarray:
    """Bin excess links above a rooted tree into sparse/middle/dense thirds.

    Density progress is (K - (M-1)) / (M*(M-1) - (M-1)); a tree is zero
    and all-to-all is one. These are density groups, not individual graphs.
    """
    counts = np.asarray(active_link_counts)
    minimum, maximum = n_lasers - 1, n_lasers * (n_lasers - 1)
    if (n_lasers < 2 or np.any(~np.isfinite(counts))
            or np.any(counts != np.floor(counts))
            or np.any(counts < minimum) or np.any(counts > maximum)):
        raise ValueError("Link counts must be integers between M-1 and M*(M-1)")
    # Integer arithmetic makes exact third-boundaries unambiguous.
    return np.minimum(3 * (counts.astype(np.int64) - minimum)
                      // (maximum - minimum), 2).astype(int)


def simulate_rooted_indexed_chunk(
    indexed_work_item: tuple[int, int, int, tuple[Any, ...], np.ndarray],
) -> dict[str, int | np.ndarray]:
    """Simulate only rooted cases in a target/candidate worker chunk."""
    group, candidate_start, candidate_stop, work_item, rooted = indexed_work_item
    chunk_index, target_start, target_stop, targets, kappa, phi, detunings, config = work_item
    expected_shape = (target_stop - target_start, candidate_stop - candidate_start)
    rooted = np.asarray(rooted, dtype=bool)
    if rooted.shape != expected_shape or kappa.shape[:2] != expected_shape:
        raise ValueError("rooted mask and coupling chunk must match target/candidate indices")
    phase = np.zeros(expected_shape, dtype=float)
    aligned = np.full(expected_shape + (config.n_lasers,), np.nan)
    if np.any(rooted):
        flat = rooted.ravel()
        repeated_targets = np.repeat(targets, expected_shape[1], axis=0)[flat]
        repeated_detunings = (
            None if detunings is None else
            np.repeat(detunings, expected_shape[1], axis=0)[flat]
        )
        simulation_kwargs = {}
        if repeated_detunings is not None:
            simulation_kwargs["detuning_distribution_ghz"] = repeated_detunings
        time_seconds, states, _, _ = base.simulate_with_vcsel(
            kappa.reshape(-1, config.n_lasers, config.n_lasers)[flat],
            phi.reshape(-1, config.n_lasers, config.n_lasers)[flat],
            config, **simulation_kwargs,
        )
        result = base.calculate_phase_reward(time_seconds, states, repeated_targets, config)
        phase.ravel()[flat] = result[0]
        aligned.reshape(-1, config.n_lasers)[flat] = result[4]
    return dict(size_batch_index=group, candidate_start_index=candidate_start,
                candidate_stop_index=candidate_stop, start_index=target_start,
                stop_index=target_stop, phase_reward=phase,
                aligned_achieved_phases_rad=aligned)


def worst_topology_policy_candidate(
    rewards: np.ndarray, masks: np.ndarray, actions_per_topology: int,
) -> int:
    """Choose the best policy action from the least successful distinct mask."""
    rewards = np.asarray(rewards, dtype=float)
    masks = np.asarray(masks)
    if (rewards.ndim != 1 or masks.ndim != 2 or len(rewards) != len(masks)
            or len(rewards) % actions_per_topology):
        raise ValueError("Rewards and masks must contain complete topology groups")
    grouped_masks = masks.reshape(-1, actions_per_topology, masks.shape[-1])
    if not np.all(grouped_masks == grouped_masks[:, :1]):
        raise ValueError("Actions in one topology group must share a mask")
    _, inverse = np.unique(grouped_masks[:, 0], axis=0, return_inverse=True)
    scores = np.where(np.isfinite(rewards), rewards, -np.inf)
    best_by_mask = []
    for mask_id in range(inverse.max() + 1):
        groups = np.flatnonzero(inverse == mask_id)
        candidates = (groups[:, None] * actions_per_topology
                      + np.arange(actions_per_topology)).ravel()
        best_by_mask.append(int(candidates[np.argmax(scores[candidates])]))
    return best_by_mask[int(np.argmin(scores[best_by_mask]))]


def phase_de_teacher_search(work: dict[str, Any]) -> dict[str, Any]:
    """Run one vectorized phase-only DE search at a fixed policy kappa."""
    config = work["config"]
    kappa = np.asarray(work["kappa_per_ns"], dtype=float)
    policy_phi = np.asarray(work["phi_p_rad"], dtype=float)
    target = np.asarray(work["target_phases_rad"], dtype=float)
    detunings = np.asarray(work["detunings_ghz"], dtype=float)
    receivers, sources = np.nonzero(kappa > 0)
    if len(receivers) == 0:
        raise ValueError("DE teacher needs at least one active link")
    rng = np.random.default_rng(work["seed"])
    initial = rng.uniform(-np.pi, np.pi,
                          (config.de_teacher_population, len(receivers)))
    initial[0] = policy_phi[receivers, sources]
    evaluations = 0

    def loss(values: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        cases = values.shape[1]
        phases = np.zeros((cases, config.n_lasers, config.n_lasers))
        phases[:, receivers, sources] = values.T
        simulation = base.simulate_with_vcsel(
            np.broadcast_to(kappa, phases.shape), phases, config,
            detuning_distribution_ghz=np.broadcast_to(
                detunings, (cases, config.n_lasers)),
        )
        reward = base.calculate_phase_reward(
            simulation[0], simulation[1],
            np.broadcast_to(target, (cases, config.n_lasers)), config,
        )[0]
        evaluations += cases
        values = np.where(np.isfinite(reward), 1.0 - reward, 2.0)
        return values

    search = differential_evolution(
        loss, [(-np.pi, np.pi)] * len(receivers), init=initial,
        maxiter=config.de_teacher_generations,
        mutation=config.de_teacher_mutation,
        recombination=config.de_teacher_recombination,
        seed=work["seed"], vectorized=True, updating="deferred", polish=False,
    )
    best_phi = np.zeros_like(kappa)
    best_phi[receivers, sources] = search.x
    return dict(group=work["group"], target_index=work["target_index"],
                candidate_index=work["candidate_index"],
                baseline_reward=work["baseline_reward"],
                best_reward=max(0.0, 1.0 - float(search.fun)),
                phi_p_rad=best_phi, evaluations=evaluations)


def set_gate_head_trainable_at(policy: BernoulliSparseGNNPolicy,
                               iteration: int, config: BernoulliSparseConfig) -> None:
    """Hold optional-gate logits at initialization while other heads train."""
    if config.topology_input_mode:
        return
    policy.edge_gate_head.requires_grad_(iteration >= config.gate_head_hold_iterations)
    if config.construct_rooted_backbone and config.uniform_random_rooted_tree:
        policy.root_head.requires_grad_(False)
        policy.tree_edge_head.requires_grad_(False)


def run_training_batch(policy: BernoulliSparseGNNPolicy, optimizer: torch.optim.Optimizer,
                       config: BernoulliSparseConfig, rng: np.random.Generator,
                       iteration: int, *, simulation_pool: Any = None,
                       reward_scale_state: dict[str, float] | None = None) -> dict[str, Any]:
    policy.train()
    set_gate_head_trainable_at(policy, iteration, config)
    policy.minimum_log_std = base.minimum_log_std_at(iteration, config)
    policy.maximum_log_std = base.maximum_log_std_at(iteration, config)
    sizes = tuple(config.training_n_lasers)
    offset = (iteration * config.targets_per_batch) % len(sizes)
    size_list = [sizes[(offset + i) % len(sizes)] for i in range(config.targets_per_batch)]
    prepared: list[dict[str, Any]] = []
    work: list[tuple[int, int, int, tuple[Any, ...]]] = []
    for group, n_lasers in enumerate(size for size in sizes if size in size_list):
        n_targets = size_list.count(n_lasers)
        local_config = replace(config, n_lasers=n_lasers)
        targets = base.sample_target_phases(
            n_targets, rng, n_lasers, config.structured_target_fraction,
            config.structured_target_jitter_rad,
        )
        detunings = base.sample_detuning_distributions(n_targets, rng, local_config, iteration=iteration)
        targets, detunings = base.sort_by_detuning(targets, detunings)
        features = encode_context(targets, detunings, policy)
        if config.topology_input_mode:
            n_topologies = config.candidates_per_target // config.actions_per_topology
            gates = sample_stratified_rooted_topologies(
                n_targets, n_topologies, n_lasers, rng, features.device,
                min_fraction=topology_min_fraction_at(iteration, local_config),
                density_bias=config.training_topology_density_bias,
                fixed_star=config.topology_star_backbone,
            ).repeat_interleave(config.actions_per_topology, dim=1)
            actions, budget_actions, log_prob = sample_topology_conditioned_candidates(
                policy, features, gates, iteration=iteration, rng=rng)
            logits = None
        else:
            offsets = _training_offsets(n_targets, config.candidates_per_target,
                                        exploration_offset_at(iteration, config), rng, features.device)
            actions, gates, log_prob, logits = sample_candidates(
                policy, features, config.candidates_per_target, offsets
            )
            budget_actions = None
            if config.objective_mode == "lexicographic":
                budget_actions, budget_log_prob = sample_budget_actions(
                    policy, features, config.candidates_per_target,
                    iteration=iteration, rng=rng)
                log_prob = log_prob + budget_log_prob
        reach = (root_reach_fractions(gates.detach().cpu().numpy(), n_lasers)
                 if config.require_rooted_connectivity or config.construct_rooted_backbone
                 or config.topology_input_mode else
                 np.ones((n_targets, config.candidates_per_target)))
        if (config.construct_rooted_backbone or config.topology_input_mode) and not np.all(reach == 1.0):
            raise RuntimeError("Constructed training mask is not rooted")
        kappa, phi, counts, rho, shortfall = decode_samples(
            actions, gates, local_config, budget_actions=budget_actions)
        simulated_kappa, simulated_phi = kappa, phi
        if config.dense_reference_feasibility:
            dense_kappa, dense_phi = dense_reference_designs(
                policy, targets, detunings, local_config,
            )
            simulated_kappa = np.concatenate((kappa, dense_kappa), axis=1)
            simulated_phi = np.concatenate((phi, dense_phi), axis=1)
        simulated_candidates = simulated_kappa.shape[1]
        chunk_size = min(config.candidates_per_worker_task, config.candidates_per_target)
        target_jobs = max(1, (n_targets + config.targets_per_worker_task - 1)
                          // config.targets_per_worker_task)
        for start in range(0, simulated_candidates, chunk_size):
            stop = min(start + chunk_size, simulated_candidates)
            chunks = base.split_simulation_batch(
                targets, simulated_kappa[:, start:stop], simulated_phi[:, start:stop], local_config,
                detuning_distributions_ghz=detunings, n_jobs=target_jobs,
            )
            if config.require_rooted_connectivity:
                work.extend((group, start, stop, item,
                             (np.concatenate((reach, np.ones((n_targets, 1))), axis=1)
                              if config.dense_reference_feasibility else reach)[
                                  item[1]:item[2], start:stop] == 1.0)
                            for item in chunks)
            else:
                work.extend((group, start, stop, item) for item in chunks)
        prepared.append(dict(n_lasers=n_lasers, config=local_config, targets=targets,
                             detunings=detunings, features=features, log_prob=log_prob,
                             gates=gates, kappa=kappa, phi=phi,
                             rho=rho, counts=counts, shortfall=shortfall,
                             reach=reach,
                             gate_probability=(float("nan") if logits is None else
                                               float(torch.sigmoid(logits).mean().detach().cpu())),
                             results=[]))
    work.sort(key=lambda item: prepared[item[0]]["n_lasers"], reverse=True)
    worker = (simulate_rooted_indexed_chunk if config.require_rooted_connectivity
              else base.simulate_indexed_target_chunk)
    if simulation_pool is not None:
        results = list(simulation_pool.imap_unordered(worker, work))
    else:
        results = [worker(item) for item in work]
    for result in results:
        prepared[int(result["size_batch_index"])]["results"].append(result)
    losses = []
    rewards_for_advantage: list[np.ndarray] = []
    size_metrics = {}
    sampled_budget_per_laser: list[np.ndarray] = []
    selected_feasible_budget_per_laser: list[np.ndarray] = []
    de_work: list[dict[str, Any]] = []
    use_de_teacher = (config.de_teacher_enabled
                      and iteration >= config.de_teacher_start_iteration
                      and (iteration - config.de_teacher_start_iteration)
                      % config.de_teacher_interval == 0)
    for group, item in enumerate(prepared):
        simulated_candidates = config.candidates_per_target + int(config.dense_reference_feasibility)
        all_phase = np.empty((len(item["targets"]), simulated_candidates), dtype=float)
        all_aligned = (np.empty((len(item["targets"]), simulated_candidates, item["n_lasers"]), dtype=float)
                       if config.objective_mode == "lexicographic" else None)
        coverage = np.zeros_like(all_phase, dtype=int)
        for result in item["results"]:
            ts, te = int(result["start_index"]), int(result["stop_index"])
            cs, ce = int(result["candidate_start_index"]), int(result["candidate_stop_index"])
            all_phase[ts:te, cs:ce] = result["phase_reward"]
            if all_aligned is not None:
                all_aligned[ts:te, cs:ce] = result["aligned_achieved_phases_rad"]
            coverage[ts:te, cs:ce] += 1
        if not np.all(coverage == 1):
            raise RuntimeError("Simulation did not return every candidate exactly once")
        phase = all_phase[:, :config.candidates_per_target]
        budget = relative_coupling_budget(item["kappa"], item["config"])
        physical_budget_per_laser = (item["kappa"].sum(axis=(-2, -1))
                                     / item["n_lasers"])
        if config.objective_mode in {"lexicographic", "phase_only"}:
            sampled_budget_per_laser.append(physical_budget_per_laser.ravel())
        if config.objective_mode == "lexicographic":
            target_relative = base.make_relative_to_laser_1(item["targets"])
            errors = maximum_phase_error_deg(all_aligned, target_relative[:, None, :])
            reward, feasible = lexicographic_candidate_scores(
                phase, errors, item["counts"], item["kappa"].sum(axis=(-2, -1)),
                item["config"], rooted=item["reach"] == 1.0,
                maximum_phase_error_deg=lexicographic_phase_error_limit_at(iteration, config),
                minimum_phase_reward=lexicographic_phase_reward_limit_at(iteration, config))
            winning_candidate = reward.argmax(axis=1)
            winning_rows = np.arange(len(phase))
            winning_feasible = feasible[winning_rows, winning_candidate]
            selected_feasible_budget_per_laser.append(
                physical_budget_per_laser[winning_rows[winning_feasible],
                                          winning_candidate[winning_feasible]])
            dense_phase = np.full((len(phase), 1), np.nan)
        elif config.objective_mode == "phase_only":
            dense_phase = np.full((len(phase), 1), np.nan)
            feasible = np.zeros_like(phase, dtype=bool)
            reward = phase
        elif config.dense_reference_feasibility:
            dense_phase = all_phase[:, config.candidates_per_target:].max(axis=1, keepdims=True)
            reward, feasible = dense_reference_reward(
                phase, dense_phase, item["rho"],
                sparsity_bonus_weight=config.feasible_sparsity_bonus_weight,
            )
            if config.require_rooted_connectivity:
                reward = np.where(item["reach"] == 1.0, reward, item["reach"] - 2.0)
                feasible &= item["reach"] == 1.0
        else:
            dense_phase = np.full((len(phase), 1), np.nan)
            feasible = np.zeros_like(phase, dtype=bool)
            reward = reward_function(phase, item["rho"], budget, item["config"],
                                     iteration=iteration + 1,
                                     root_reach_fraction=item["reach"])
        rewards_for_advantage.append(reward)
        if use_de_teacher:
            masks = item["gates"].detach().cpu().numpy()
            for target_index in range(len(item["targets"])):
                candidate_index = worst_topology_policy_candidate(
                    phase[target_index], masks[target_index],
                    config.actions_per_topology,
                )
                de_work.append(dict(
                    group=group, target_index=target_index,
                    candidate_index=candidate_index,
                    baseline_reward=float(phase[target_index, candidate_index]),
                    target_phases_rad=item["targets"][target_index],
                    detunings_ghz=item["detunings"][target_index],
                    kappa_per_ns=item["kappa"][target_index, candidate_index],
                    phi_p_rad=item["phi"][target_index, candidate_index],
                    config=item["config"],
                    seed=int(rng.integers(0, 2**32 - 1)),
                ))
        size_metrics[item["n_lasers"]] = dict(
            phase_reward=float(phase.mean()), training_reward=float(reward.mean()),
            retained_link_fraction=float(item["rho"].mean()),
            normalized_coupling_budget=float(budget.mean()),
            mean_redistribution_shortfall_per_ns=float(item["shortfall"].mean()),
            mean_active_links=float(item["counts"].mean()),
            rooted_fraction=float(np.mean(item["reach"] == 1.0)),
            mean_gate_probability=item["gate_probability"],
            dense_reference_phase_reward=float(dense_phase.mean()),
            feasible_candidate_fraction=float(feasible.mean()),
            targets_with_feasible_candidate=float(feasible.any(axis=1).mean()),
        )
    if simulation_pool is not None and de_work:
        de_results = list(simulation_pool.imap_unordered(phase_de_teacher_search, de_work))
    else:
        de_results = [phase_de_teacher_search(work) for work in de_work]
    teacher_by_group: list[list[dict[str, Any]]] = [[] for _ in prepared]
    for result in de_results:
        if result["best_reward"] > (result["baseline_reward"]
                                    + config.de_teacher_min_improvement):
            teacher_by_group[result["group"]].append(result)
    advantages = (_topology_advantages(rewards_for_advantage, config, reward_scale_state)
                  if config.topology_input_mode else
                  [_standardized_advantages(reward) for reward in rewards_for_advantage])
    teacher_losses = []
    task_weights = []
    task_labels = []
    split_topology_tasks = (config.gradient_combination_mode == "pcgrad"
                           and config.pcgrad_task_grouping == "array_size_topology_density")
    for group, (item, advantage_np) in enumerate(zip(prepared, advantages)):
        advantage = torch.as_tensor(advantage_np,
                                    dtype=item["features"].dtype, device=item["features"].device)
        candidate_losses = -(item["log_prob"] * advantage)
        auxiliary_loss = candidate_losses.new_zeros(())
        size_loss = candidate_losses.mean()
        if teacher_by_group[group]:
            receivers, sources = base.directed_link_indices(item["n_lasers"])
            n_edges = len(receivers)
            local_teacher_losses = []
            for teacher in teacher_by_group[group]:
                target_index = teacher["target_index"]
                mask = item["gates"][target_index, teacher["candidate_index"]]
                mean, _, _ = policy.parameters_for_edges(
                    item["features"][target_index:target_index + 1],
                    topology_masks=mask[None],
                )
                target_phases = torch.as_tensor(
                    teacher["phi_p_rad"][receivers, sources],
                    dtype=mean.dtype, device=mean.device,
                )
                circular_loss = (1.0 - torch.cos(mean[0, n_edges:] - target_phases))
                local_teacher_losses.append(
                    (circular_loss * mask).sum() / mask.sum().clamp_min(1.0))
            teacher_loss = torch.stack(local_teacher_losses).mean()
            auxiliary_loss = config.de_teacher_loss_weight * teacher_loss
            size_loss = size_loss + auxiliary_loss
            teacher_losses.append(float(teacher_loss.detach().cpu()))
        if split_topology_tasks:
            band_ids = topology_density_band_ids(item["counts"], item["n_lasers"])
            for band_id, band in enumerate(DENSITY_BANDS):
                selected = band_ids == band_id
                if not np.any(selected):
                    continue
                selected_tensor = torch.as_tensor(selected, device=candidate_losses.device)
                losses.append(candidate_losses[selected_tensor].mean() + auxiliary_loss)
                # Averaging these weighted losses reconstructs the original
                # equal-M objective before clipping and conflict projection.
                task_weights.append(float(selected.mean()) / len(prepared))
                task_labels.append(f"M{item['n_lasers']}:{band}")
        else:
            losses.append(size_loss)
            task_weights.append(1.0 / len(prepared))
            task_labels.append(f"M{item['n_lasers']}")
    optimizer.zero_grad(set_to_none=True)
    gradient_diagnostics = dict(negative_gradient_pair_fraction=np.nan, mean_gradient_cosine=np.nan)
    if config.gradient_combination_mode == "pcgrad" and len(losses) > 1:
        parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        task_gradients = []
        for task_index, loss in enumerate(losses):
            # Density losses within one M share a forward graph.
            gradients = torch.autograd.grad(
                loss, parameters, allow_unused=True,
                retain_graph=split_topology_tasks and task_index < len(losses) - 1)
            task_gradients.append(torch.cat([
                torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
                for parameter, gradient in zip(parameters, gradients)
            ]))
        combination_options = (dict(task_weights=task_weights) if split_topology_tasks else {})
        combined, gradient_diagnostics = base.combine_task_gradients_pcgrad(
            torch.stack(task_gradients), config.per_size_gradient_clip, rng,
            **combination_options)
        cursor = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad = combined[cursor:cursor + count].reshape_as(parameter).clone()
            cursor += count
    else:
        torch.stack(losses).mean().backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
    optimizer.step()
    with torch.no_grad():
        policy.log_std_kappa.clamp_(policy.minimum_log_std, policy.maximum_log_std)
        policy.log_std_phi.clamp_(policy.minimum_log_std, policy.maximum_log_std)
    average = lambda key: float(np.mean([metric[key] for metric in size_metrics.values()]))
    sampled_budgets = (np.concatenate(sampled_budget_per_laser)
                       if sampled_budget_per_laser else np.array([]))
    selected_budgets = (np.concatenate(selected_feasible_budget_per_laser)
                        if selected_feasible_budget_per_laser else np.array([]))
    budget_percentiles = (np.percentile(sampled_budgets, [10, 50, 90])
                          if len(sampled_budgets) else np.full(3, np.nan))
    return dict(phase_reward=average("phase_reward"), training_reward=average("training_reward"),
                retained_link_fraction=average("retained_link_fraction"),
                rooted_fraction=average("rooted_fraction"),
                normalized_coupling_budget=average("normalized_coupling_budget"),
                coupling_cost_weight=coupling_cost_weight_at(iteration + 1, config),
                mean_redistribution_shortfall_per_ns=average("mean_redistribution_shortfall_per_ns"),
                mean_gate_probability=average("mean_gate_probability"),
                dense_reference_phase_reward=average("dense_reference_phase_reward"),
                feasible_candidate_fraction=average("feasible_candidate_fraction"),
                targets_with_feasible_candidate=average("targets_with_feasible_candidate"),
                sampled_budget_p10_per_laser=float(budget_percentiles[0]),
                sampled_budget_median_per_laser=float(budget_percentiles[1]),
                sampled_budget_p90_per_laser=float(budget_percentiles[2]),
                selected_feasible_budget_median_per_laser=(
                    float(np.median(selected_budgets)) if len(selected_budgets) else np.nan),
                exploration_offset=(float("nan") if config.topology_input_mode else
                                    exploration_offset_at(iteration, config)),
                gradient_norm=float(gradient_norm.detach().cpu()),
                de_teacher_searches=len(de_results),
                de_teacher_simulations=sum(result["evaluations"] for result in de_results),
                de_teacher_improved=sum(map(len, teacher_by_group)),
                de_teacher_mean_improvement=(
                    float(np.mean([max(0.0, result["best_reward"] - result["baseline_reward"])
                                   for result in de_results])) if de_results else 0.0),
                de_teacher_loss=(float(np.mean(teacher_losses)) if teacher_losses else 0.0),
                pcgrad_task_count=(len(losses) if config.gradient_combination_mode == "pcgrad" else 0),
                pcgrad_task_labels=task_labels,
                **gradient_diagnostics, size_metrics=size_metrics)


def infer_designs(policy: BernoulliSparseGNNPolicy, target: np.ndarray,
                  detunings: np.ndarray, config: BernoulliSparseConfig, *,
                  count: int = 1, active_fraction: float | None = None,
                  seed: int | None = None, continuous_mean: bool = True,
                  sample_optional_gates: bool = False,
                  topology: np.ndarray | None = None) -> dict[str, Any]:
    """Decode masks, optionally sampling gates beyond the deterministic tree."""
    if config.topology_input_mode:
        return infer_topology_conditioned_designs(
            policy, target, detunings, config, count=count,
            active_fraction=active_fraction, seed=seed,
            continuous_mean=continuous_mean, topology=topology)
    if topology is not None:
        raise ValueError("An explicit topology requires topology_input_mode")
    if count < 1:
        raise ValueError("count must be positive")
    target, detunings = base.sort_by_detuning(np.asarray(target), np.asarray(detunings))
    features = encode_context(target, detunings, policy)
    policy.eval()
    if seed is not None:
        # fork_rng restores the caller's CPU RNG state after inference.
        rng_context = torch.random.fork_rng(devices=[])
    else:
        from contextlib import nullcontext
        rng_context = nullcontext()
    with rng_context, torch.inference_mode():
        if seed is not None:
            torch.manual_seed(seed)
        if config.construct_rooted_backbone:
            mean, log_std, logits, root_logits, tree_scores = policy.parameters_for_edges(
                features, return_topology=True)
            tree, _, roots = select_rooted_tree(
                root_logits, tree_scores, count,
                deterministic=not config.uniform_random_rooted_tree,
                uniform_random=config.uniform_random_rooted_tree)
        else:
            mean, log_std, logits = policy.parameters_for_edges(features)
            tree, roots = None, None
        bias = (calibrated_logit_bias(logits, active_fraction)
                if active_fraction is not None else logits.new_zeros(()))
        gate_distribution = torch.distributions.Bernoulli(logits=logits + bias)
        if config.construct_rooted_backbone:
            if sample_optional_gates:
                optional = gate_distribution.sample((count,)).permute(1, 0, 2)
            else:
                optional = (torch.sigmoid(logits + bias) >= 0.5).to(logits.dtype)
                optional = optional[:, None, :].expand(1, count, -1)
            gates = torch.maximum(optional, tree.to(logits.dtype))
        else:
            gates = gate_distribution.sample((count,)).permute(1, 0, 2)
        if continuous_mean:
            actions = mean[:, None].expand(1, count, -1).clone()
        else:
            normal = torch.distributions.Normal(mean, log_std.exp())
            actions = normal.sample((count,)).permute(1, 0, 2)
        budget_actions = None
        if config.objective_mode == "lexicographic":
            budget_actions, _ = sample_budget_actions(
                policy, features, count, continuous_mean=continuous_mean)
        raw_gate_logits = logits.detach().cpu().numpy()[0]
        natural_gate_probability = torch.sigmoid(logits).detach().cpu().numpy()[0]
        optional_gate_probability = torch.sigmoid(logits + bias).detach().cpu().numpy()[0]
        probability = optional_gate_probability.copy()
        if tree is not None:
            probability = np.where(tree.detach().cpu().numpy()[0, 0], 1.0, probability)
            if not np.all(root_reach_fractions(gates.detach().cpu().numpy(), config.n_lasers) == 1.0):
                raise RuntimeError("Constructed inference mask is not rooted")
    kappa, phi, counts, rho, shortfall = decode_samples(
        actions, gates, config, budget_actions=budget_actions)
    return dict(target_phases_rad=target, detuning_distribution_ghz=detunings,
                kappa_per_ns=kappa[0], phi_p_rad=phi[0],
                gates=gates.detach().cpu().numpy()[0],
                root_indices=None if roots is None else roots.detach().cpu().numpy()[0],
                tree_gates=None if tree is None else tree.detach().cpu().numpy()[0],
                gate_logits=raw_gate_logits,
                natural_gate_probabilities=natural_gate_probability,
                optional_gate_probabilities=optional_gate_probability,
                gate_logit_bias=float(bias.detach().cpu()),
                gate_probabilities=probability, expected_active_fraction=float(probability.mean()),
                active_link_counts=counts[0], retained_link_fractions=rho[0],
                raw_total_coupling_per_ns=kappa[0].sum(axis=(-2, -1)) + shortfall[0],
                redistribution_shortfall_per_ns=shortfall[0],
                requested_active_fraction=active_fraction)


def infer_topology_conditioned_designs(
    policy: BernoulliSparseGNNPolicy, target: np.ndarray,
    detunings: np.ndarray, config: BernoulliSparseConfig, *,
    count: int = 1, active_fraction: float | None = None,
    seed: int | None = None, continuous_mean: bool = True,
    topology: np.ndarray | None = None,
) -> dict[str, Any]:
    """Design for supplied source-row/receiver-column masks or sampled rooted masks."""
    if count < 1 or active_fraction is not None:
        raise ValueError("Topology-input inference needs count >= 1 and no gate-density override")
    target = np.asarray(target, dtype=float)
    detunings = np.asarray(detunings, dtype=float)
    order = np.argsort(detunings, kind="stable")
    target, detunings = base.sort_by_detuning(target, detunings)
    features = encode_context(target, detunings, policy)
    policy.eval()
    n_lasers = config.n_lasers
    n_edges = n_lasers * (n_lasers - 1)
    with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext(), torch.inference_mode():
        if seed is not None:
            torch.manual_seed(seed)
        if topology is None:
            masks = sample_stratified_rooted_topologies(
                1, count, n_lasers, np.random.default_rng(seed), features.device,
                min_fraction=config.training_topology_min_fraction,
                fixed_star=config.topology_star_backbone)
        else:
            matrices = np.asarray(topology)
            if matrices.shape == (n_lasers, n_lasers):
                matrices = np.broadcast_to(matrices, (count, n_lasers, n_lasers))
            if matrices.shape != (count, n_lasers, n_lasers):
                raise ValueError("topology must have shape (M,M) or (count,M,M)")
            if (not np.all(np.isfinite(matrices)) or not np.all((matrices == 0) | (matrices == 1))
                    or np.any(np.diagonal(matrices, axis1=-2, axis2=-1) != 0)):
                raise ValueError("topology must be binary with a zero diagonal")
            matrices = matrices[:, order, :][:, :, order]
            receivers, sources = base.directed_link_indices(n_lasers)
            masks_np = matrices[:, sources, receivers]  # user matrix: source row, receiver column
            if not np.all(root_reach_fractions(masks_np, n_lasers) == 1.0):
                raise ValueError("Every supplied topology must be rooted")
            masks = torch.as_tensor(masks_np[None], device=features.device, dtype=features.dtype)
        actions, budget_actions, _ = sample_topology_conditioned_candidates(
            policy, features, masks, continuous_mean=continuous_mean)
    kappa, phi, counts, rho, shortfall = decode_samples(
        actions, masks, config, budget_actions=budget_actions)
    gates = masks.detach().cpu().numpy()[0]
    return dict(target_phases_rad=target, detuning_distribution_ghz=detunings,
                kappa_per_ns=kappa[0], phi_p_rad=phi[0], gates=gates,
                root_indices=None, tree_gates=None, gate_logits=None,
                natural_gate_probabilities=None, optional_gate_probabilities=None,
                gate_logit_bias=0.0, gate_probabilities=gates[0],
                expected_active_fraction=float(rho.mean()),
                active_link_counts=counts[0], retained_link_fractions=rho[0],
                raw_total_coupling_per_ns=kappa[0].sum(axis=(-2, -1)) + shortfall[0],
                redistribution_shortfall_per_ns=shortfall[0],
                requested_active_fraction=None)


def evaluate_policy(policy: BernoulliSparseGNNPolicy, targets: np.ndarray,
                    detunings: np.ndarray, config: BernoulliSparseConfig,
                    *, seed: int = 0, simulation_pool: Any = None) -> dict[str, float]:
    if config.topology_input_mode:
        return evaluate_topology_conditioned_policy(
            policy, targets, detunings, config, seed=seed,
            simulation_pool=simulation_pool)
    targets, detunings = base.sort_by_detuning(targets, detunings)
    features = encode_context(targets, detunings, policy)
    policy.eval()
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if config.construct_rooted_backbone:
            mean, _, logits, root_logits, tree_scores = policy.parameters_for_edges(
                features, return_topology=True)
            tree, _, _ = select_rooted_tree(
                root_logits, tree_scores, 1,
                deterministic=not config.uniform_random_rooted_tree,
                uniform_random=config.uniform_random_rooted_tree)
            optional = (torch.distributions.Bernoulli(logits=logits).sample()
                        if config.uniform_random_rooted_tree else
                        (torch.sigmoid(logits) >= 0.5).to(logits.dtype))
            gates = torch.maximum(tree.to(logits.dtype), optional[:, None, :])
        else:
            mean, _, logits = policy.parameters_for_edges(features)
            gates = torch.distributions.Bernoulli(logits=logits).sample()[:, None]
        actions = mean[:, None]
        budget_actions = None
        if config.objective_mode == "lexicographic":
            budget_actions, _ = sample_budget_actions(policy, features, 1, continuous_mean=True)
    reach = (root_reach_fractions(gates.detach().cpu().numpy(), config.n_lasers)
             if config.require_rooted_connectivity or config.construct_rooted_backbone else
             np.ones((len(targets), 1)))
    if config.construct_rooted_backbone and not np.all(reach == 1.0):
        raise RuntimeError("Constructed validation mask is not rooted")
    kappa, phi, counts, rho, shortfall = decode_samples(
        actions, gates, config, budget_actions=budget_actions)
    phase = np.zeros((len(targets), 1), dtype=float)
    aligned = np.full((len(targets), 1, config.n_lasers), np.nan)
    rooted = reach[:, 0] == 1.0
    if np.any(rooted):
        simulation = base.simulate_candidates_parallel(
            targets[rooted], kappa[rooted], phi[rooted], config,
            detuning_distributions_ghz=detunings[rooted],
            simulation_pool=simulation_pool,
        )
        phase[rooted] = simulation["phase_reward"]
        if config.objective_mode == "lexicographic":
            aligned[rooted] = simulation["aligned_achieved_phases_rad"]
    budget = relative_coupling_budget(kappa, config)
    if config.objective_mode == "lexicographic":
        errors = maximum_phase_error_deg(
            aligned, base.make_relative_to_laser_1(targets)[:, None, :])
        _, feasible = lexicographic_candidate_scores(
            phase, errors, counts, kappa.sum(axis=(-2, -1)), config,
            rooted=reach == 1.0)
        dense_phase = np.full_like(phase, np.nan)
        reward = phase  # Held-out curve remains an interpretable phase reward.
    elif config.dense_reference_feasibility:
        dense_kappa, dense_phi = dense_reference_designs(
            policy, targets, detunings, config,
        )
        dense_result = base.simulate_candidates_parallel(
            targets, dense_kappa, dense_phi, config,
            detuning_distributions_ghz=detunings, simulation_pool=simulation_pool,
        )
        dense_phase = dense_result["phase_reward"].max(axis=1, keepdims=True)
        reward, feasible = dense_reference_reward(
            phase, dense_phase, rho,
            sparsity_bonus_weight=config.feasible_sparsity_bonus_weight,
        )
        if config.require_rooted_connectivity:
            reward = np.where(reach == 1.0, reward, reach - 2.0)
            feasible &= reach == 1.0
    else:
        dense_phase = np.full_like(phase, np.nan)
        feasible = np.zeros_like(phase, dtype=bool)
        reward = reward_function(phase, rho, budget, config,
                                 root_reach_fraction=reach)
    return dict(phase_reward=float(phase.mean()), training_reward=float(reward.mean()),
                dense_reference_phase_reward=float(dense_phase.mean()),
                feasible_fraction=float(feasible.mean()),
                rooted_fraction=float(rooted.mean()),
                retained_link_fraction=float(rho.mean()), normalized_coupling_budget=float(budget.mean()),
                coupling_budget_per_laser=float(kappa.sum(axis=(-2, -1)).mean() / config.n_lasers),
                mean_redistribution_shortfall_per_ns=float(shortfall.mean()))


def evaluate_topology_conditioned_policy(
    policy: BernoulliSparseGNNPolicy, targets: np.ndarray,
    detunings: np.ndarray, config: BernoulliSparseConfig, *,
    seed: int = 0, simulation_pool: Any = None,
) -> dict[str, float]:
    """Use fixed-seed held-out topologies over the configured density range."""
    targets, detunings = base.sort_by_detuning(targets, detunings)
    features = encode_context(targets, detunings, policy)
    policy.eval()
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        masks = sample_stratified_rooted_topologies(
            1, len(targets), config.n_lasers,
            np.random.default_rng(seed), features.device,
            min_fraction=config.training_topology_min_fraction,
            fixed_star=config.topology_star_backbone,
        ).reshape(len(targets), 1, config.n_lasers * (config.n_lasers - 1))
        actions, budget_actions, _ = sample_topology_conditioned_candidates(
            policy, features, masks, continuous_mean=True)
    kappa, phi, counts, rho, shortfall = decode_samples(
        actions, masks, config, budget_actions=budget_actions)
    simulation = base.simulate_candidates_parallel(
        targets, kappa, phi, config,
        detuning_distributions_ghz=detunings, simulation_pool=simulation_pool)
    phase = simulation["phase_reward"]
    budget = relative_coupling_budget(kappa, config)
    band_ids = topology_density_band_ids(counts[:, 0], config.n_lasers)
    phase_by_density = {
        label: float(phase[band_ids == index].mean()) if np.any(band_ids == index) else float("nan")
        for index, label in enumerate(DENSITY_BANDS)
    }
    return dict(phase_reward=float(phase.mean()), training_reward=float(phase.mean()),
                dense_reference_phase_reward=float("nan"), feasible_fraction=0.0,
                rooted_fraction=1.0, retained_link_fraction=float(rho.mean()),
                normalized_coupling_budget=float(budget.mean()),
                coupling_budget_per_laser=float(kappa.sum(axis=(-2, -1)).mean() / config.n_lasers),
                mean_redistribution_shortfall_per_ns=float(shortfall.mean()),
                phase_reward_by_density=phase_by_density)


def save_checkpoint(path: str | Path, policy: BernoulliSparseGNNPolicy,
                    optimizer: torch.optim.Optimizer, config: BernoulliSparseConfig,
                    iteration: int, held_out_reward: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(format=CHECKPOINT_FORMAT, config=asdict(config),
                    policy_state_dict=policy.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    iteration=iteration, held_out_reward=held_out_reward), temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[BernoulliSparseGNNPolicy, BernoulliSparseConfig, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Not a Bernoulli sparse checkpoint")
    saved_config = dict(checkpoint["config"])
    # Older checkpoints may contain the removed P^4 reward hyperparameter.
    saved_config.pop("phase_cost_exponent", None)
    # Preserve the inference transform of earlier log-budget checkpoints,
    # without bringing their training-only adaptive controller back.
    old_budget_mode = saved_config.get("budget_exploration_mode")
    if old_budget_mode not in {"stratified_offsets", "mixed_std", "local"}:
        saved_config.pop("budget_exploration_mode", None)
    old_floor = saved_config.pop("budget_search_floor_per_laser", None)
    old_ceiling = saved_config.pop("budget_search_initial_ceiling_per_laser", None)
    for key in ("budget_search_growth_factor", "budget_search_update_targets",
                "budget_search_span_bands", "budget_search_chaos_margin",
                "budget_search_target_feasible_fraction"):
        saved_config.pop(key, None)
    if old_budget_mode == "adaptive_frontier" and old_floor is not None and old_ceiling is not None:
        saved_config["log_budget_initial_per_laser"] = float(np.sqrt(old_floor * old_ceiling))
    config = replace(BernoulliSparseConfig(**saved_config), device=device)
    policy = BernoulliSparseGNNPolicy(config).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    return policy, config, dict(iteration=checkpoint["iteration"],
                                held_out_reward=checkpoint["held_out_reward"])


def training_figure(history: dict[str, Any], config: BernoulliSparseConfig) -> plt.Figure:
    lexicographic = config.objective_mode == "lexicographic"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(1, len(history["training_reward"]) + 1)
    if lexicographic:
        axes[0].plot(x, history["feasible_candidate_fraction"], label="feasible sample fraction")
        axes[0].plot(x, history["targets_with_feasible_candidate"], "-.",
                     label="targets with feasible sample")
    elif not config.topology_input_mode:
        axes[0].plot(x, history["training_reward"], label="total reward")
    phase_label = ("phase reward" if config.construct_rooted_backbone or config.topology_input_mode
                   else "phase reward (invalid=0)")
    axes[0].plot(x, history["phase_reward"], label=phase_label)
    rho_label = (r"mean supplied fraction $\rho$" if config.topology_input_mode else
                 r"mean active fraction $\rho$")
    axes[0].plot(x, history["retained_link_fraction"], "--", label=rho_label)
    if config.dense_reference_feasibility:
        axes[0].plot(x, history["dense_reference_phase_reward"], ":",
                     label="dense reference phase")
        axes[0].plot(x, history["targets_with_feasible_candidate"], "-.",
                     label="targets with feasible sample")
    axes[0].set(xlabel="iteration", ylabel="reward / fraction", title="REINFORCE training")
    if lexicographic:
        axes[0].set_ylim(-0.05, 1.05)
    else:
        axes[0].set_ylim(min(-0.05, float(np.nanmin(history["training_reward"])) * 1.05),
                         max(1.05, float(np.nanmax(history["training_reward"])) * 1.05))
    axes[0].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if config.plot_coupling_reference:
        # The frozen initial dense reference is not a maximum. Give its ratio an
        # independent scale so increases above one do not flatten the other lines.
        budget_axis = axes[0].twinx()
        budget_axis.plot(x, history["normalized_coupling_budget"], ":",
                         color="tab:red", label="coupling / initial dense reference")
        budget_axis.set_ylabel("coupling / initial dense reference", color="tab:red")
        budget_axis.tick_params(axis="y", labelcolor="tab:red")
        finite_budget = np.asarray(history["normalized_coupling_budget"], dtype=float)
        finite_budget = finite_budget[np.isfinite(finite_budget)]
        upper = max(1.05, 1.05 * float(finite_budget.max())) if len(finite_budget) else 1.05
        budget_axis.set_ylim(0, upper)
        budget_handles, budget_labels = budget_axis.get_legend_handles_labels()
        handles += budget_handles
        labels += budget_labels
    axes[0].legend(handles, labels, loc="lower right", fontsize=8)
    show_density = config.topology_input_mode and "held_out_by_density" in history
    for size, values in history["held_out_by_m"].items():
        axes[1].plot(x, values, alpha=0.45 if show_density else 1.0, label=f"M={size}")
    if show_density:
        for label, color in zip(DENSITY_BANDS, ("tab:purple", "tab:orange", "tab:green")):
            values = np.asarray(history["held_out_by_density"][label], dtype=float)
            if np.any(np.isfinite(values)):
                axes[1].plot(x, values, "--", color=color, linewidth=1.8,
                             label=f"{label} topology")
    axes[1].plot(x, history["held_out_reward"], color="black", linewidth=2, label="overall")
    if config.dense_reference_feasibility or lexicographic:
        axes[1].plot(x, history["held_out_feasible_fraction"], "--", color="tab:green",
                     label="feasible fraction")
    axes[1].set(xlabel="iteration",
                ylabel="phase reward / fraction" if lexicographic else "mean reward",
                title=("Star-to-dense held-out targets" if config.topology_star_backbone else
                       "Topology-input held-out targets" if config.topology_input_mode
                       else "Bernoulli-z held-out targets"))
    finite_heldout = np.asarray(history["held_out_reward"], dtype=float)
    finite_heldout = finite_heldout[np.isfinite(finite_heldout)]
    axes[1].set_ylim(min(-0.05, float(finite_heldout.min()) * 1.05) if len(finite_heldout) else -0.05,
                     max(1.0, float(finite_heldout.max()) * 1.05) if len(finite_heldout) else 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right", ncol=2)
    return figure


def train_policy(config: BernoulliSparseConfig, *, live_plot: bool = True) -> tuple[BernoulliSparseGNNPolicy, dict[str, Any]]:
    rng = base.set_random_seed(config.random_seed)
    policy = BernoulliSparseGNNPolicy(config).to(config.device)
    config = calibrate_budget_references(policy, config)
    policy.config = config
    reference_label = ("Frozen initial all-on coupling references, diagnostic only (ns^-1): "
                       if config.objective_mode in {"lexicographic", "phase_only"} else
                       "Frozen initial all-on coupling references (ns^-1): ")
    print(reference_label + ", ".join(
        f"M={size}: {budget:.4g}" for size, budget in config.coupling_budget_reference_by_m
    ))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    if config.gradient_combination_mode == "pcgrad":
        print(f"PCGrad grouping: {config.pcgrad_task_grouping}; "
              f"per-task gradient clip: {config.per_size_gradient_clip:g}; "
              f"DE teacher: {'enabled' if config.de_teacher_enabled else 'disabled'}")
    reward_scale_state: dict[str, float] = {}
    history: dict[str, Any] = {key: [] for key in (
        "phase_reward", "training_reward", "retained_link_fraction", "rooted_fraction",
        "normalized_coupling_budget", "mean_gate_probability",
        "mean_redistribution_shortfall_per_ns", "held_out_reward",
        "held_out_rooted_fraction", "dense_reference_phase_reward",
        "feasible_candidate_fraction", "targets_with_feasible_candidate",
        "held_out_feasible_fraction", "held_out_phase_reward",
        "sampled_budget_p10_per_laser", "sampled_budget_median_per_laser",
        "sampled_budget_p90_per_laser", "selected_feasible_budget_median_per_laser",
        "held_out_budget_per_laser",
        "de_teacher_searches", "de_teacher_simulations", "de_teacher_improved",
        "de_teacher_mean_improvement", "de_teacher_loss",
        "pcgrad_task_count", "negative_gradient_pair_fraction", "mean_gradient_cosine",
    )}
    history["held_out_by_m"] = {size: [] for size in config.training_n_lasers}
    if config.topology_input_mode:
        history["held_out_by_density"] = {band: [] for band in DENSITY_BANDS}
    held_out = {}
    for size in config.training_n_lasers:
        local = replace(config, n_lasers=size)
        targets = base.sample_target_phases(
            config.held_out_target_count,
            np.random.default_rng(config.random_seed + 10000 + size), size,
            config.structured_target_fraction, config.structured_target_jitter_rad,
        )
        detunings = base.sample_detuning_distributions(
            config.held_out_target_count,
            np.random.default_rng(config.random_seed + 20000 + size), local,
            iteration=config.detuning_curriculum_iterations,
        )
        held_out[size] = base.sort_by_detuning(targets, detunings)
    latest = {size: np.nan for size in config.training_n_lasers}
    latest_rooted = {size: np.nan for size in config.training_n_lasers}
    latest_feasible = {size: np.nan for size in config.training_n_lasers}
    latest_phase = {size: np.nan for size in config.training_n_lasers}
    latest_rho = {size: np.nan for size in config.training_n_lasers}
    latest_budget_per_laser = {size: np.nan for size in config.training_n_lasers}
    latest_density_by_m = {
        size: {band: np.nan for band in DENSITY_BANDS}
        for size in config.training_n_lasers
    }
    best = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf)
            if config.objective_mode == "lexicographic" else
            (-np.inf, -np.inf, -np.inf) if config.dense_reference_feasibility else -np.inf)
    pool = None
    if config.n_jobs > 1:
        context = mp.get_context("spawn")
        pool = context.Pool(processes=min(config.n_jobs, config.targets_per_batch),
                            initializer=base._initialize_simulation_worker)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for iteration in range(config.training_iterations):
            if iteration >= config.learning_rate_decay_start_iteration:
                progress = min((iteration - config.learning_rate_decay_start_iteration) /
                               max(config.training_iterations - config.learning_rate_decay_start_iteration - 1, 1), 1)
                lr = config.learning_rate * (config.learning_rate_final / config.learning_rate) ** progress
                for group in optimizer.param_groups:
                    group["lr"] = lr
            metrics = run_training_batch(policy, optimizer, config, rng, iteration,
                                         simulation_pool=pool,
                                         reward_scale_state=reward_scale_state)
            for key in ("phase_reward", "training_reward", "retained_link_fraction",
                        "rooted_fraction",
                        "normalized_coupling_budget", "mean_gate_probability",
                        "mean_redistribution_shortfall_per_ns", "dense_reference_phase_reward",
                        "feasible_candidate_fraction", "targets_with_feasible_candidate"):
                history[key].append(metrics[key])
            for key in ("sampled_budget_p10_per_laser", "sampled_budget_median_per_laser",
                        "sampled_budget_p90_per_laser",
                        "selected_feasible_budget_median_per_laser",
                        "de_teacher_searches", "de_teacher_simulations",
                        "de_teacher_improved", "de_teacher_mean_improvement",
                        "de_teacher_loss", "pcgrad_task_count",
                        "negative_gradient_pair_fraction", "mean_gradient_cosine"):
                history[key].append(metrics[key])
            if (iteration == 0 or (iteration + 1) % config.validation_interval == 0
                    or iteration + 1 == config.training_iterations):
                span = base.detuning_curriculum_half_span_at(iteration, config)
                for size in config.training_n_lasers:
                    targets, full_detunings = held_out[size]
                    detunings = full_detunings
                    if config.validation_follows_detuning_curriculum:
                        detunings = full_detunings * span / (0.5 * config.detuning_span_ghz)
                    validation = evaluate_policy(
                        policy, targets, detunings, replace(config, n_lasers=size),
                        seed=config.random_seed + 30000 + size, simulation_pool=pool,
                    )
                    latest[size] = validation["training_reward"]
                    latest_rooted[size] = validation["rooted_fraction"]
                    latest_feasible[size] = validation["feasible_fraction"]
                    latest_phase[size] = validation["phase_reward"]
                    latest_rho[size] = validation["retained_link_fraction"]
                    latest_budget_per_laser[size] = validation["coupling_budget_per_laser"]
                    if config.topology_input_mode:
                        latest_density_by_m[size] = validation["phase_reward_by_density"]
                held_out_reward = float(np.mean(list(latest.values())))
                resource_order = ((-float(np.mean(list(latest_rho.values()))),
                                   -float(np.mean(list(latest_budget_per_laser.values()))))
                                  if config.lexicographic_order == "sparsity_budget" else
                                  (-float(np.mean(list(latest_budget_per_laser.values()))),
                                   -float(np.mean(list(latest_rho.values())))))
                held_out_feasibility = float(np.mean(list(latest_feasible.values())))
                quality = ((held_out_feasibility,
                            held_out_reward if held_out_feasibility == 0.0 else 0.0,
                            *resource_order, held_out_reward)
                           if config.objective_mode == "lexicographic" else
                           (float(np.mean(list(latest_feasible.values()))),
                            held_out_reward, float(np.mean(list(latest_phase.values()))))
                           if config.dense_reference_feasibility else held_out_reward)
                if quality > best:
                    best = quality
                    save_checkpoint(config.best_checkpoint_file, policy, optimizer,
                                    config, iteration + 1, held_out_reward)
            held_out_reward = float(np.mean(list(latest.values())))
            history["held_out_reward"].append(held_out_reward)
            history["held_out_rooted_fraction"].append(float(np.mean(list(latest_rooted.values()))))
            history["held_out_feasible_fraction"].append(float(np.mean(list(latest_feasible.values()))))
            history["held_out_phase_reward"].append(float(np.mean(list(latest_phase.values()))))
            history["held_out_budget_per_laser"].append(
                float(np.mean(list(latest_budget_per_laser.values()))))
            for size in config.training_n_lasers:
                history["held_out_by_m"][size].append(latest[size])
            if config.topology_input_mode:
                for band in DENSITY_BANDS:
                    values = [latest_density_by_m[size][band]
                              for size in config.training_n_lasers]
                    finite = [value for value in values if np.isfinite(value)]
                    history["held_out_by_density"][band].append(
                        float(np.mean(finite)) if finite else float("nan"))
            save_checkpoint(config.current_checkpoint_file, policy, optimizer,
                            config, iteration + 1, held_out_reward)
            if live_plot and (iteration == 0 or (iteration + 1) % config.plot_update_interval == 0
                              or iteration + 1 == config.training_iterations):
                figure = training_figure(history, config)
                figure.savefig(RESULTS_DIR / f"{Path(config.current_checkpoint_file).stem}_training_progress.png", dpi=150)
                if config.jupyter_mode:
                    clear_output(wait=True)
                    display(figure)
                plt.close(figure)
            budget_label = (f"budget/M={metrics['sampled_budget_median_per_laser']:.3g} ns^-1"
                            if config.objective_mode in {"lexicographic", "phase_only"} else
                            f"budget/reference={metrics['normalized_coupling_budget']:.3f}")
            reward_label = ("" if config.objective_mode in {"lexicographic", "phase_only"} else
                            f"reward={metrics['training_reward']:.3f} ")
            print(f"{iteration + 1}/{config.training_iterations} "
                  f"phase={metrics['phase_reward']:.3f} {reward_label}"
                  f"rho={metrics['retained_link_fraction']:.3f} "
                  f"rooted={metrics['rooted_fraction']:.3f} "
                  f"{budget_label} "
                  f"unallocated={metrics['mean_redistribution_shortfall_per_ns']:.3g} ns^-1 "
                  + ("" if config.topology_input_mode else
                     f"offset={metrics['exploration_offset']:.2f} ")
                  + f"held-out={held_out_reward:.3f} "
                  f"held-out-rooted={history['held_out_rooted_fraction'][-1]:.3f}"
                  + (f" PCGrad-tasks={metrics['pcgrad_task_count']} "
                     f"conflicts={metrics['negative_gradient_pair_fraction']:.3f}"
                     if config.gradient_combination_mode == "pcgrad" else "")
                  + (f" DE={metrics['de_teacher_improved']}/{metrics['de_teacher_searches']} "
                     f"gain={metrics['de_teacher_mean_improvement']:.3f} "
                     f"sims={metrics['de_teacher_simulations']}"
                     if config.de_teacher_enabled else "")
                  + (f" dense-phase={metrics['dense_reference_phase_reward']:.3f} "
                     f"train-feasible={metrics['feasible_candidate_fraction']:.3f} "
                     f"held-out-feasible={history['held_out_feasible_fraction'][-1]:.3f}"
                     if config.dense_reference_feasibility or config.objective_mode == "lexicographic" else ""))
        save_checkpoint(config.final_checkpoint_file, policy, optimizer,
                        config, config.training_iterations, held_out_reward)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return policy, history
