"""Python configuration sections, explicitly populated by the run files.

All schedule boundaries are printed, one-based iteration numbers. Physical
parameters use vcsel_lib's native dictionary keys and SI units.
See docs/configuration.md for the exact mapping into the numerical backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import numpy as np

from ._compat import (VCSEL_FIELDS, compatibility_settings, make_backend_config,
                      vcsel_parameters_to_settings)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture, phase output convention, and optional root encoder input."""
    node_hidden_sizes: tuple[int, ...] = (64,)
    node_embedding_dim: int = 64
    message_hidden_size: int = 64
    message_passing_steps: int = 3
    edge_hidden_sizes: tuple[int, ...] = (128, 128)
    normalize_incoming_by_degree: bool = True
    softplus_strength_scale_per_ns: float = 1.0
    initial_budget_per_laser: float = .5
    phase_parameterization: str = "absolute"
    encode_root: bool = False

    def __post_init__(self):
        if not isinstance(self.encode_root, bool):
            raise ValueError("encode_root must be True or False")
        if self.encode_root and self.phase_parameterization != 'target_root_delay_residual':
            raise ValueError("Root encoding currently requires target_root_delay_residual")
        if self.phase_parameterization not in {"absolute", "target_residual", "target_delay_residual",
                                              "target_root_delay_residual"}:
            raise ValueError("phase_parameterization must be absolute, target_residual, "
                             "target_delay_residual or target_root_delay_residual")


def _default_phys():
    """Native defaults for short API calls; the run file defines its own phys."""
    return dict(tau_p=5.4e-12, tau_n=2.5e-10, g0=875000., N0=286000.,
        s=4e-6, q=1.602e-19, beta=.001, I=.0008609971885714285, alpha=2.,
        tau=1e-9, dt=5.4e-12, Tmax=500.*1e-9, save_every=10, noise_amplitude=0.)


@dataclass(frozen=True)
class TopologyConfig:
    """Provided masks now; learned Bernoulli gates are a reserved extension."""
    mode: str = "provided"
    backbone: str = "rooted_star"
    minimum_extra_link_fraction: float = 0.0
    density_bias: float = 1.0

    def __post_init__(self):
        if self.mode != "provided":
            raise NotImplementedError(
                "This API version reproduces supplied-topology training. "
                "Learned Bernoulli gates are planned but are not enabled.")
        if self.backbone not in {"rooted_star", "random_reachable"}:
            raise ValueError("backbone must be rooted_star or random_reachable")


@dataclass(frozen=True)
class SamplingConfig:
    """Span bins and topology draws; per_target reproduces BACKUP sampling."""
    pairs_per_size: int = 5
    topologies_per_pair: int = 4
    actions_per_topology: int = 10
    detuning_span_ghz: float = 4.0
    pairs_per_job: int = 5
    actions_per_job: int = 10
    structured_target_fraction: float = 0.0
    structured_target_jitter_rad: float = .1
    reuse_masks_across_pairs: bool = False
    topology_sampling_mode: str = "paired_contexts"
    action_sampling: str = "independent"  # mirrored pairs opposite latent perturbations.
    # paired_contexts: spread one pool across pairs; per_target: span the full
    # density range separately for every pair, with independent optional edges.

    @property
    def detuning_bins(self) -> int:
        """One bin per pair, matching the current stratified-span sampler."""
        return self.pairs_per_size

    def __post_init__(self):
        for name in ("pairs_per_size", "topologies_per_pair", "actions_per_topology",
                     "pairs_per_job", "actions_per_job"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.actions_per_topology < 2:
            raise ValueError("Within-topology advantages require at least two actions")
        if not math.isfinite(self.detuning_span_ghz) or self.detuning_span_ghz <= 0:
            raise ValueError("detuning_span_ghz must be positive and finite")
        if not isinstance(self.reuse_masks_across_pairs, bool):
            raise ValueError("reuse_masks_across_pairs must be True or False")
        if self.topology_sampling_mode not in {"paired_contexts", "per_target", "band_random", "band_fixed"}:
            raise ValueError("topology_sampling_mode must be paired_contexts, per_target, band_random or band_fixed")
        if self.action_sampling not in {"independent", "mirrored"}:
            raise ValueError("action_sampling must be independent or mirrored")


@dataclass(frozen=True)
class OptimizationConfig:
    """One shared Adam update after per-task clipping and PCGrad projection."""
    iterations: int = 3000
    workers: int = 14
    learning_rate: float = 1e-4
    final_learning_rate: float = 1e-5
    per_task_clip: float = 1.0
    global_clip: float = 1.0
    reward_std_ema_decay: float = .95
    budget_calibration_targets: int = 64
    pcgrad_grouping: str = "topology_strata"
    reward_baseline: str = "topology_mean"  # pair_excluded uses the other action pairs.
    # density_bands uses BACKUP's physical sparse/middle/dense boundaries.

    def __post_init__(self):
        if self.pcgrad_grouping not in {"topology_strata", "density_bands"}:
            raise ValueError("pcgrad_grouping must be topology_strata or density_bands")
        if self.reward_baseline not in {"topology_mean", "pair_excluded"}:
            raise ValueError("reward_baseline must be topology_mean or pair_excluded")
        for name in ("iterations", "workers"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "final_learning_rate", "per_task_clip", "global_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class ScheduleConfig:
    """Printed iteration numbers; shortened runs simply stop before later stages."""
    ordinary_start: int = 1000
    ordinary_end: int = 1500
    wide_start: int = 1000
    wide_end: int = 1500
    learning_rate_decay_start: int = 2000
    wide_phase_actions: int = 2
    wide_phase_extra_sd_deg: float = 180.0
    initial_log_std: float = -.25
    initial_minimum_log_std: float = -1.5
    initial_maximum_log_std: float = .75
    final_minimum_log_std: float = -3.0
    final_maximum_log_std: float = -1.5
    learning_rate_decay_end: int | None = None  # None decays through the final iteration.
    precision_start: int | None = None         # None disables the extra refinement stage.
    precision_end: int | None = None
    precision_minimum_log_std: float = -4.5
    precision_maximum_log_std: float = -3.0
    precision_budget_std: float = 0.05
    anneal_to_mean: bool = False  # Scale all action SDs linearly to zero; final batch is evaluation only.


    def __post_init__(self):
        minimum_start = 1 if self.anneal_to_mean else 2
        if not minimum_start <= self.ordinary_start < self.ordinary_end:
            raise ValueError(f"Ordinary annealing needs increasing printed boundaries >={minimum_start}")
        if not minimum_start <= self.wide_start < self.wide_end:
            raise ValueError(f"Wide annealing needs increasing printed boundaries >={minimum_start}")
        if self.anneal_to_mean:
            if ((self.ordinary_start, self.ordinary_end) != (self.wide_start, self.wide_end)
                    or self.precision_start is not None or self.precision_end is not None):
                raise ValueError("Anneal-to-mean uses one shared ordinary/wide interval and no precision stage")
        if self.learning_rate_decay_start < 1:
            raise ValueError("learning_rate_decay_start must be a printed iteration >=1")
        if (self.learning_rate_decay_end is not None
                and self.learning_rate_decay_end <= self.learning_rate_decay_start):
            raise ValueError("learning_rate_decay_end must follow its start")
        if (self.precision_start is None) != (self.precision_end is None):
            raise ValueError("Precision refinement needs both start and end")
        if self.precision_start is not None:
            if not self.ordinary_end <= self.precision_start < self.precision_end:
                raise ValueError("Precision refinement must follow ordinary annealing")
            if not (math.isfinite(self.precision_minimum_log_std)
                    and math.isfinite(self.precision_maximum_log_std)):
                raise ValueError("Precision log SD bounds must be finite")
            if not self.precision_minimum_log_std <= self.precision_maximum_log_std:
                raise ValueError("Precision minimum log SD must not exceed its maximum")
            if (self.precision_minimum_log_std > self.final_minimum_log_std
                    or self.precision_maximum_log_std > self.final_maximum_log_std):
                raise ValueError("Precision bounds must narrow ordinary exploration")
            if not math.isfinite(self.precision_budget_std) or not 0 < self.precision_budget_std <= .35:
                raise ValueError("precision_budget_std must be positive and no greater than .35")
        if not isinstance(self.wide_phase_actions, int) or self.wide_phase_actions < 0:
            raise ValueError("wide_phase_actions must be a nonnegative integer")
        if not math.isfinite(self.wide_phase_extra_sd_deg) or self.wide_phase_extra_sd_deg < 0:
            raise ValueError("wide_phase_extra_sd_deg must be finite and nonnegative")


@dataclass(frozen=True)
class ValidationConfig:
    """Fixed held-out bank and both metrics; best checkpoints use RMS reward."""
    every: int = 50
    pairs_per_size: int = 64
    plot_every: int = 1
    notebook_display: bool = True
    plot_dpi: int = 300
    plot_use_tex: bool = False


@dataclass(frozen=True)
class RewardConfig:
    """Wrapped RMS learning reward; cosine remains an inspection metric."""
    tail_fraction: float = .35
    worst_laser_weight: float = .5
    allow_global_phase_conjugate: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Current recipe with separate, documented user-facing sections."""
    sizes: tuple[int, ...] = tuple(range(3, 10))
    seed: int = 7
    device: str = "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    phys: dict = field(default_factory=_default_phys)
    ramp_start_delays: float = 5.0
    ramp_rise_delays: float = 100.0
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    schedules: ScheduleConfig = field(default_factory=ScheduleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        object.__setattr__(self, "sizes", tuple(self.sizes))
        physical_settings = vcsel_parameters_to_settings(self.phys)
        # Snapshot shared scalars; per-batch arrays are constructed by VCSEL.
        physical_snapshot = {name: physical_settings[old] for name, old in VCSEL_FIELDS.items()}
        if "phase_model" in physical_settings:
            physical_snapshot["phase_model"] = physical_settings["phase_model"]
        object.__setattr__(self, "phys", physical_snapshot)
        if (not self.sizes or len(set(self.sizes)) != len(self.sizes)
                or any(not isinstance(m, int) or m < 2 for m in self.sizes)):
            raise ValueError("sizes must contain distinct integers >=2")
        if self.schedules.wide_phase_actions > self.sampling.actions_per_topology:
            raise ValueError("wide_phase_actions cannot exceed actions_per_topology")
        if self.sampling.action_sampling == "mirrored":
            if not self.schedules.anneal_to_mean:
                raise ValueError("Mirrored sampling currently requires anneal_to_mean=True")
            if self.sampling.actions_per_topology < 4 or self.sampling.actions_per_topology % 2:
                raise ValueError("Mirrored sampling needs an even actions_per_topology >=4")
            if self.schedules.wide_phase_actions % 2:
                raise ValueError("Mirrored sampling needs an even wide_phase_actions")
            if self.optimization.reward_baseline != "pair_excluded":
                raise ValueError("Mirrored sampling requires reward_baseline='pair_excluded'")
        elif self.optimization.reward_baseline == "pair_excluded":
            raise ValueError("pair_excluded requires action_sampling='mirrored'")
        if self.schedules.anneal_to_mean and self.schedules.ordinary_end != self.optimization.iterations:
            raise ValueError("Anneal-to-mean must end at the final iteration")
        if self.optimization.iterations < 1 or self.optimization.workers < 1:
            raise ValueError("iterations and workers must be positive")
        if self.validation.every < 1 or self.validation.pairs_per_size < 1:
            raise ValueError("Validation interval and held-out count must be positive")
        if self.validation.plot_dpi < 1:
            raise ValueError("plot_dpi must be positive")
        if self.sampling.topology_sampling_mode in {"band_random", "band_fixed"}:
            if (self.sampling.topologies_per_pair != 5
                    or self.optimization.pcgrad_grouping != "density_bands"
                    or self.sampling.reuse_masks_across_pairs
                    or self.topology.minimum_extra_link_fraction != 0.
                    or self.topology.density_bias != 1.
                    or min(self.sizes) < 3):
                raise ValueError("Band sampling needs five independent rooted masks, physical density bands, "
                                 "uniform counts, no minimum-density restriction, and M>=3")
        if self.topology.backbone == 'random_reachable':
            if (self.sampling.topology_sampling_mode != 'band_random'
                    or self.sampling.reuse_masks_across_pairs):
                raise ValueError("Random reachable training currently uses independent band_random masks")
        if (self.model.phase_parameterization == 'target_root_delay_residual'
                and self.topology.backbone != 'random_reachable'):
            raise ValueError("Root delay references require random_reachable topology sampling")
        if (self.schedules.precision_start is not None
                and self.sampling.topology_sampling_mode not in {"band_random", "band_fixed", "per_target"}):
            raise ValueError("The precision stage requires band_random, band_fixed or per_target training")
        if (self.optimization.pcgrad_grouping == "topology_strata"
                and self.sampling.topology_sampling_mode != "paired_contexts"):
            raise ValueError("Topology-stratum PCGrad requires paired_contexts sampling")
        if self.sampling.reuse_masks_across_pairs:
            if (self.sampling.topology_sampling_mode != "paired_contexts"
                    or self.optimization.pcgrad_grouping != "topology_strata"):
                raise ValueError("Shared-mask training uses paired_contexts and topology_strata")
            if (self.topology.minimum_extra_link_fraction != 0.
                    or self.topology.density_bias != 1.):
                raise ValueError("Shared masks use an exact star and uniform density bands")
            if self.sampling.topologies_per_pair < 2:
                raise ValueError("Shared masks need a star and at least one density band")
            groups = self.sampling.topologies_per_pair
            if groups > 2 and any((m-1)**2 <= groups-1 for m in self.sizes):
                raise ValueError("An array size has too few link counts for the shared density groups")

    def to_backend(self, run_directory: str | Path):
        """Build the current algorithm from Python settings; no JSON is read."""
        settings = compatibility_settings()
        # The recipes use the same policy and numerical kernels. The explicit
        # run-file choices below select sampling and PCGrad grouping.
        settings.update(encoder_architecture="gnn", condition_on_n_lasers=False,
            condition_log_std_on_n_lasers=False, edge_conditioned_log_std=True,
            enable_sparse_gates=False, train_all_sizes_each_iteration=True,
            gradient_combination_mode="pcgrad", enable_minimum_log_std_annealing=True,
            enable_exploration_annealing=True, detuning_curriculum_warmup_iterations=0,
            detuning_curriculum_iterations=0, detuning_span_sampling_mode="stratified_span",
            uncapped_redistribution=True, strength_action_transform="uncapped_softplus",
            link_cost_weight=0., coupling_cost_weight=0., plot_coupling_reference=False,
            topology_input_mode=True, topology_sampling_mode="paired_contexts",
            topology_advantage_scale="global_running", phase_reward_mode="wrapped_rms",
            pcgrad_task_grouping="array_size_topology_strata", objective_mode="phase_only",
            budget_action_transform="log_budget", budget_exploration_mode="local")
        m, s, o, a, v = (self.model, self.sampling,
                             self.optimization, self.schedules, self.validation)
        # The new independent band sampler lives in the API batch; retain a
        # compatible backend schema for simulation, validation and checkpoints.
        settings.update(topology_sampling_mode=("per_target" if s.topology_sampling_mode in {"band_random", "band_fixed"}
                                                else s.topology_sampling_mode),
            pcgrad_task_grouping=("array_size_topology_density" if o.pcgrad_grouping == "density_bands"
                                 else "array_size_topology_strata"))
        settings.update(n_lasers=self.sizes[0], training_n_lasers=self.sizes,
            random_seed=self.seed, device=self.device,
            node_hidden_sizes=m.node_hidden_sizes, node_embedding_dim=m.node_embedding_dim,
            gnn_message_hidden_size=m.message_hidden_size,
            gnn_message_passing_steps=m.message_passing_steps, edge_hidden_sizes=m.edge_hidden_sizes,
            normalize_incoming_coupling_by_degree=m.normalize_incoming_by_degree,
            softplus_strength_scale_per_ns=m.softplus_strength_scale_per_ns,
            log_budget_initial_per_laser=m.initial_budget_per_laser,
            coupling_ramp_start_delays=self.ramp_start_delays,
            coupling_ramp_rise_delays=self.ramp_rise_delays,
            targets_per_batch=len(self.sizes)*s.pairs_per_size,
            candidates_per_target=s.topologies_per_pair*s.actions_per_topology,
            actions_per_topology=s.actions_per_topology,
            targets_per_worker_task=s.pairs_per_job,
            candidates_per_worker_task=s.actions_per_job,
            pcgrad_topology_groups=s.topologies_per_pair, detuning_span_ghz=s.detuning_span_ghz,
            structured_target_fraction=s.structured_target_fraction,
            structured_target_jitter_rad=s.structured_target_jitter_rad,
            topology_star_backbone=self.topology.backbone == "rooted_star",
            training_topology_min_fraction=self.topology.minimum_extra_link_fraction,
            training_topology_density_bias=self.topology.density_bias,
            training_iterations=o.iterations, n_jobs=o.workers,
            learning_rate=o.learning_rate, learning_rate_final=o.final_learning_rate,
            per_size_gradient_clip=o.per_task_clip, gradient_clip=o.global_clip,
            global_reward_std_ema_decay=o.reward_std_ema_decay,
            budget_reference_calibration_targets=o.budget_calibration_targets,
            initial_log_std=a.initial_log_std,
            initial_minimum_log_std=a.initial_minimum_log_std,
            maximum_log_std=a.initial_maximum_log_std,
            minimum_log_std=a.final_minimum_log_std,
            final_maximum_log_std=a.final_maximum_log_std,
            minimum_log_std_anneal_start_iteration=max(a.ordinary_start-2, 0),
            minimum_log_std_anneal_iterations=a.ordinary_end-a.ordinary_start+1,
            exploration_anneal_start_iteration=a.ordinary_start-1,
            exploration_anneal_iterations=a.ordinary_end-a.ordinary_start+1,
            phase_exploration_start_iteration=max(a.wide_start-2, 0),
            phase_exploration_end_iteration=a.wide_end-1,
            phase_exploration_fraction=a.wide_phase_actions/s.actions_per_topology,
            phase_exploration_start_std=math.radians(a.wide_phase_extra_sd_deg),
            learning_rate_decay_start_iteration=a.learning_rate_decay_start-1,
            held_out_target_count=v.pairs_per_size, validation_interval=v.every,
            plot_update_interval=v.plot_every, jupyter_mode=v.notebook_display,
            reward_tail_fraction=self.reward.tail_fraction,
            worst_laser_reward_weight=self.reward.worst_laser_weight,
            allow_global_phase_conjugate=self.reward.allow_global_phase_conjugate)
        settings.update(vcsel_parameters_to_settings(self.phys))
        if m.phase_parameterization != "absolute":
            settings['phase_parameterization'] = m.phase_parameterization
        if m.encode_root:
            settings['encode_root'] = True
        if self.topology.backbone == 'random_reachable':
            settings['topology_random_reachable'] = True
        if a.anneal_to_mean:
            # The API batch applies one SD multiplier instead of the legacy bound schedules.
            settings.update(phase_parameterization=m.phase_parameterization,
                            exploration_std_scale=1., enable_minimum_log_std_annealing=False,
                            enable_exploration_annealing=False)
        if s.action_sampling == "mirrored":
            settings.update(action_sampling=s.action_sampling,
                            reward_baseline=o.reward_baseline)
        directory = Path(run_directory).resolve()
        for kind in ("best", "current", "final"):
            settings[f"{kind}_checkpoint_file"] = str(directory / f"{kind}.pt")
        return make_backend_config(settings)

    def summary(self) -> dict:
        """Return counts without initializing a policy or creating directories."""
        s = self.sampling
        candidates = s.topologies_per_pair*s.actions_per_topology
        target_jobs = math.ceil(s.pairs_per_size/s.pairs_per_job)
        candidate_jobs = math.ceil(candidates/s.actions_per_job)
        task_count = len(self.sizes)*s.topologies_per_pair
        if s.topology_sampling_mode in {"band_random", "band_fixed"}:
            task_count = 3*len(self.sizes)
        elif self.optimization.pcgrad_grouping == "density_bands":
            # Physical bands can be empty when only a few density levels are
            # drawn. Count the occupied bands using the sampler's link counts.
            task_count = 0
            for size in self.sizes:
                total = size*(size-1)
                minimum = max(size-1, int(np.ceil(
                    self.topology.minimum_extra_link_fraction*total-1e-10)))
                count = (s.topologies_per_pair if s.topology_sampling_mode == "per_target"
                         else s.pairs_per_size*s.topologies_per_pair)
                if count == 1:
                    task_count += 1
                    continue
                if count >= total-minimum+1 and self.topology.density_bias == 1.:
                    counts = np.arange(minimum, total+1)
                else:
                    q = np.linspace(0., 1., count)
                    counts = np.rint(minimum+(total-minimum)*
                                     (1.-(1.-q)**self.topology.density_bias))
                bands = np.minimum(3*(counts.astype(int)-(size-1))//((size-1)**2), 2)
                task_count += len(np.unique(bands))
        summary = dict(sizes=list(self.sizes), pairs_per_size=s.pairs_per_size,
            detuning_bins=s.detuning_bins, detuning_bin_width_ghz=s.detuning_span_ghz/s.detuning_bins,
            topologies_per_pair=s.topologies_per_pair, actions_per_topology=s.actions_per_topology,
            contexts=len(self.sizes)*s.pairs_per_size*s.topologies_per_pair,
            simulations_per_iteration=len(self.sizes)*s.pairs_per_size*candidates,
            pcgrad_tasks=task_count,
            worker_jobs=len(self.sizes)*target_jobs*candidate_jobs,
            workers=self.optimization.workers, iterations=self.optimization.iterations,
            learning_reward="wrapped_rms", phase_parameterization=self.model.phase_parameterization,
            action_sampling=s.action_sampling, reward_baseline=self.optimization.reward_baseline,
            topology_mode=self.topology.mode,
            topology_sampling=s.topology_sampling_mode,
            pcgrad_grouping=self.optimization.pcgrad_grouping,
            mask_reuse="shared_across_pairs" if s.reuse_masks_across_pairs else "per_pair",
            sampled_masks_per_size=(s.topologies_per_pair if s.reuse_masks_across_pairs
                                    else s.pairs_per_size*s.topologies_per_pair))
        if self.topology.backbone == 'random_reachable':
            summary.update(topology_backbone='random_reachable', root_selection='uniform per topology',
                           pcgrad_tasks_are_upper_bound=True)
        if self.model.encode_root:
            summary.update(root_encoder_input='one binary root flag per laser', node_input_features=5)
        return summary
