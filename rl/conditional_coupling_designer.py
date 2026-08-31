"""Readable conditional reinforcement learning for VCSEL coupling design.

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
targets -> encoding -> policy -> action -> matrices -> VCSEL -> reward -> loss.
It depends only on NumPy, PyTorch, Matplotlib, IPython, and ``vcsel_lib``.
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
CHECKPOINT_FORMAT = "conditional_coupling_edge_v3_local_detuning"


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

    n_lasers: int = 5

    # Neural network and REINFORCE training
    training_iterations: int = 700
    targets_per_batch: int = 16
    candidates_per_target: int = 128
    hidden_sizes: tuple[int, ...] = (256, 256, 128)
    detuning_hidden_sizes: tuple[int, ...] = (64, 64)
    edge_hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 1.0e-3
    learning_rate_switch_iteration: int = 1500
    learning_rate_after_switch: float = 1.0e-4
    force_symmetric_kappa: bool = False
    force_symmetric_phi_p: bool = False
    # Keep individual links asymmetric while balancing total upper/lower
    # triangular coupling strength after action decoding.
    balanced_coupling: bool = False
    initial_log_std: float = -0.25
    minimum_log_std: float = -3.0
    maximum_log_std: float = 0.75
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
    time_step_seconds: float = 5.4e-12
    simulation_time_seconds: float = 500.0e-9
    save_every: int = 10
    coupling_ramp_start_delays: float = 2.0
    coupling_ramp_rise_delays: float = 5.0
    reward_tail_fraction: float = 0.35

    # Validation, plotting, and saving
    held_out_target_count: int = 64
    validation_interval: int = 10
    plot_update_interval: int = 1
    best_checkpoint_file: str | None = None
    current_checkpoint_file: str | None = None
    final_checkpoint_file: str | None = None
    jupyter_mode: bool = True

    def __post_init__(self) -> None:
        if self.best_checkpoint_file is None:
            self.best_checkpoint_file = str(
                MODEL_DIR
                / f"conditional_coupling_edge_{self.n_lasers}_lasers_best.pt"
            )
        if self.current_checkpoint_file is None:
            self.current_checkpoint_file = str(
                MODEL_DIR
                / f"conditional_coupling_edge_{self.n_lasers}_lasers_current.pt"
            )
        if self.final_checkpoint_file is None:
            self.final_checkpoint_file = str(
                MODEL_DIR
                / f"conditional_coupling_edge_{self.n_lasers}_lasers_final.pt"
            )

        if self.n_lasers < 2:
            raise ValueError("n_lasers must be at least 2")
        if self.targets_per_batch < 1:
            raise ValueError("targets_per_batch must be positive")
        if self.candidates_per_target < 2:
            raise ValueError("candidates_per_target must be at least 2")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if len(self.hidden_sizes) < 2:
            raise ValueError("hidden_sizes must contain at least two layers")
        if len(self.detuning_hidden_sizes) < 1:
            raise ValueError(
                "detuning_hidden_sizes must contain at least one layer"
            )
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
        if not (
            self.minimum_log_std
            <= self.initial_log_std
            <= self.maximum_log_std
        ):
            raise ValueError(
                "initial_log_std must be within the log-std bounds"
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


def sample_detuning_distributions(
    count: int,
    rng: np.random.Generator,
    config: DesignerConfig,
    iteration: int | None = None,
) -> np.ndarray:
    """Sample one ordered detuning vector per target, in GHz.

    The returned values are detunings relative to the mean optical
    frequency, whose reference is 0 GHz. Training holds a narrow random span
    for the configured warm-up, then grows that span to the full configured
    range. Rows are centered and sorted so matrix indices always run from the
    lowest to the highest detuning.
    """
    if count < 1:
        raise ValueError("count must be positive")
    half_span = 0.5 * config.detuning_span_ghz
    if iteration is None or config.detuning_curriculum_iterations == 0:
        span_fraction = 1.0
    elif iteration < config.detuning_curriculum_warmup_iterations:
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
    initial_half_span = (
        config.detuning_curriculum_initial_half_span_ghz
    )
    current_half_span = (
        initial_half_span
        + span_fraction * (half_span - initial_half_span)
    )
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
    """Gaussian policy with one shared decoder for every learned link."""

    def __init__(self, config: DesignerConfig):
        super().__init__()
        self.n_lasers = config.n_lasers
        self.force_symmetric_kappa = config.force_symmetric_kappa
        self.force_symmetric_phi_p = config.force_symmetric_phi_p
        self.n_directed_links = self.n_lasers * (self.n_lasers - 1)
        self.n_magnitude_links = (
            self.n_lasers * (self.n_lasers - 1) // 2
            if self.force_symmetric_kappa
            else self.n_directed_links
        )
        self.n_phase_links = (
            self.n_lasers * (self.n_lasers - 1) // 2
            if self.force_symmetric_phi_p
            else self.n_directed_links
        )
        self.action_size = self.n_magnitude_links + self.n_phase_links
        # Phase features are [cos(relative phase), sin(relative phase)] for
        # lasers 2..N. Detuning features contain all N detunings relative to
        # the mean-frequency reference (0 GHz).
        self.phase_feature_width = 2 * (self.n_lasers - 1)
        self.detuning_feature_width = self.n_lasers
        self.input_width = (
            self.phase_feature_width + self.detuning_feature_width
        )

        # Preserve the successful phase-encoder widths while giving the
        # physically different detuning inputs their own small encoder.
        phase_layers: list[nn.Module] = []
        input_width = self.phase_feature_width
        for output_width in config.hidden_sizes:
            linear = nn.Linear(input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            phase_layers.extend((linear, nn.SiLU()))
            input_width = output_width
        self.phase_encoder = nn.Sequential(*phase_layers)
        phase_embedding_width = input_width

        detuning_layers: list[nn.Module] = []
        input_width = self.detuning_feature_width
        for output_width in config.detuning_hidden_sizes:
            linear = nn.Linear(input_width, output_width)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            detuning_layers.extend((linear, nn.SiLU()))
            input_width = output_width
        self.detuning_encoder = nn.Sequential(*detuning_layers)
        detuning_embedding_width = input_width

        # Fuse both summaries back to the same global width used previously.
        self.global_embedding_width = phase_embedding_width
        fusion = nn.Linear(
            phase_embedding_width + detuning_embedding_width,
            self.global_embedding_width,
        )
        nn.init.orthogonal_(fusion.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(fusion.bias)
        self.global_fusion = nn.Sequential(fusion, nn.SiLU())

        # The shared decoder evaluates every directed edge. For either matrix
        # configured as symmetric, only its upper-triangle outputs become
        # actions and decoding mirrors them into the opposite direction.
        link_receivers, link_sources = directed_link_indices(self.n_lasers)
        self.register_buffer(
            "link_receivers",
            torch.as_tensor(link_receivers, dtype=torch.long),
        )
        self.register_buffer(
            "link_sources",
            torch.as_tensor(link_sources, dtype=torch.long),
        )
        magnitude_edge_indices = (
            np.flatnonzero(link_receivers < link_sources)
            if self.force_symmetric_kappa
            else np.arange(self.n_directed_links)
        )
        self.register_buffer(
            "magnitude_edge_indices",
            torch.as_tensor(magnitude_edge_indices, dtype=torch.long),
            persistent=False,
        )
        phase_edge_indices = (
            np.flatnonzero(link_receivers < link_sources)
            if self.force_symmetric_phi_p
            else np.arange(self.n_directed_links)
        )
        self.register_buffer(
            "phase_edge_indices",
            torch.as_tensor(phase_edge_indices, dtype=torch.long),
            persistent=False,
        )

        # Each edge sees the global embedding plus four local source/receiver
        # features: cosine/sine phase difference and signed/absolute detuning
        # difference. The same decoder is reused for all directed links.
        edge_layers: list[nn.Module] = []
        edge_input_width = self.global_embedding_width + 4
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

        self.log_std = nn.Parameter(
            torch.full((self.action_size,), config.initial_log_std)
        )
        self.minimum_log_std = config.minimum_log_std
        self.maximum_log_std = config.maximum_log_std

    def action_mean(self, encoded_targets: torch.Tensor) -> torch.Tensor:
        """Return unique-kappa and directed-phase action means."""
        if (
            encoded_targets.ndim != 2
            or encoded_targets.shape[1] != self.input_width
        ):
            raise ValueError(
                "encoded_targets must have shape "
                f"(batch, {self.input_width})"
            )

        batch_size = encoded_targets.shape[0]
        relative_count = self.n_lasers - 1
        phase_features = encoded_targets[:, : self.phase_feature_width]
        detuning_features = encoded_targets[:, self.phase_feature_width :]
        phase_embedding = self.phase_encoder(phase_features)
        detuning_embedding = self.detuning_encoder(detuning_features)
        global_embedding = self.global_fusion(
            torch.cat((phase_embedding, detuning_embedding), dim=1)
        )
        cosine = torch.cat(
            (
                torch.ones(
                    (batch_size, 1),
                    dtype=encoded_targets.dtype,
                    device=encoded_targets.device,
                ),
                phase_features[:, :relative_count],
            ),
            dim=1,
        )
        sine = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1),
                    dtype=encoded_targets.dtype,
                    device=encoded_targets.device,
                ),
                phase_features[:, relative_count:],
            ),
            dim=1,
        )

        # target(source) - target(receiver), represented without a wrap
        # discontinuity. These features are derived exactly from the original
        # relative-phase encoding, so the public target interface is unchanged.
        local_cosine = (
            cosine[:, self.link_sources] * cosine[:, self.link_receivers]
            + sine[:, self.link_sources] * sine[:, self.link_receivers]
        )
        local_sine = (
            sine[:, self.link_sources] * cosine[:, self.link_receivers]
            - cosine[:, self.link_sources] * sine[:, self.link_receivers]
        )
        # Individual detunings are normalized by half the configured span.
        # Dividing their difference by two keeps the signed edge separation
        # near [-1, 1]; its magnitude is a useful direction-independent cue.
        local_detuning_difference = 0.5 * (
            detuning_features[:, self.link_sources]
            - detuning_features[:, self.link_receivers]
        )
        local_detuning_magnitude = torch.abs(local_detuning_difference)
        local_detuning_features = torch.stack(
            (local_detuning_difference, local_detuning_magnitude), dim=2
        )
        local_phase_features = torch.stack(
            (local_cosine, local_sine), dim=2
        )
        local_features = torch.cat(
            (local_phase_features, local_detuning_features), dim=2
        )
        repeated_global = global_embedding[:, None, :].expand(
            batch_size,
            self.n_directed_links,
            global_embedding.shape[1],
        )
        edge_inputs = torch.cat(
            (repeated_global, local_features), dim=2
        )
        edge_outputs = self.edge_decoder(
            edge_inputs.reshape(
                batch_size * self.n_directed_links, -1
            )
        ).reshape(batch_size, self.n_directed_links, 2)

        # Smoothly bound the Gaussian magnitude means to [-6, 6].
        raw_magnitude_means = edge_outputs[:, :, 0]
        directed_magnitude_means = 6.0 * torch.tanh(
            raw_magnitude_means / 6.0
        )
        magnitude_means = directed_magnitude_means.index_select(
            1, self.magnitude_edge_indices
        )

        phase_means = edge_outputs[:, :, 1].index_select(
            1, self.phase_edge_indices
        )

        # Preserve the action convention: all magnitude logits first,
        # followed by all coupling phases.
        return torch.cat(
            (magnitude_means, phase_means),
            dim=1,
        )

    def forward(self, encoded_targets: torch.Tensor) -> torch.Tensor:
        """Return means shaped ``(batch, action_size)``."""
        return self.action_mean(encoded_targets)

    def distribution(
        self, encoded_targets: torch.Tensor
    ) -> torch.distributions.Normal:
        mean = self(encoded_targets)
        log_std = torch.clamp(
            self.log_std, self.minimum_log_std, self.maximum_log_std
        )
        return torch.distributions.Normal(mean, torch.exp(log_std))


def encode_for_policy(
    targets_rad: np.ndarray,
    policy: PolicyNetwork,
    config: DesignerConfig,
    detuning_distribution_ghz: np.ndarray | None = None,
) -> torch.Tensor:
    """Prepare phase and detuning features on the policy's device.

    Detunings are supplied in GHz relative to the mean-frequency reference
    (0 GHz).  A zero vector is used only for backwards-compatible callers
    that do not provide detunings explicitly.
    """
    if policy.n_lasers != config.n_lasers:
        raise ValueError("policy and config use different n_lasers")
    if np.asarray(targets_rad).shape[-1] != config.n_lasers:
        raise ValueError(
            f"targets must contain {config.n_lasers} phases"
        )
    policy_targets = (
        canonicalize_global_phase_conjugate(targets_rad)
        if config.allow_global_phase_conjugate
        else make_relative_to_laser_1(targets_rad)
    )
    encoded = encode_target_phases(policy_targets)
    if encoded.ndim == 1:
        encoded = encoded[None, :]
    if detuning_distribution_ghz is None:
        detuning_features = np.zeros(
            (encoded.shape[0], config.n_lasers), dtype=np.float32
        )
    else:
        detuning_features = encode_detuning_distributions(
            detuning_distribution_ghz, config
        )
        if detuning_features.ndim == 1:
            detuning_features = detuning_features[None, :]
        if detuning_features.shape[0] != encoded.shape[0]:
            raise ValueError(
                "targets and detuning distributions must have the same "
                "batch size"
            )
    encoded = np.concatenate((encoded, detuning_features), axis=1)
    return torch.as_tensor(
        encoded,
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
    log_probabilities = log_probabilities.sum(dim=2).permute(1, 0)
    return actions, log_probabilities


# ---------------------------------------------------------------------------
# 3. Action vector -> complete coupling matrices
# ---------------------------------------------------------------------------


def decode_action(
    actions: np.ndarray,
    maximum_kappa_per_ns: float,
    n_lasers: int = DEFAULT_N_LASERS,
    force_symmetric_kappa: bool = False,
    force_symmetric_phi_p: bool = False,
    balanced_coupling: bool = False,
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
    link_kappa_per_ns = maximum_kappa_per_ns / (
        1.0 + np.exp(-magnitude_logits)
    )
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

    # targets: (T, N); detunings: (T, N); encoded inputs:
    # (T, 2*(N-1) + N)
    encoded_targets = encode_for_policy(
        targets_rad,
        policy,
        config,
        detuning_distributions_ghz,
    )
    actions, log_probabilities = sample_coupling_designs(
        policy, encoded_targets, config.candidates_per_target
    )

    # actions: (T, C, 2*N*(N-1)) -> flattened candidate actions
    actions_numpy = actions.detach().cpu().numpy()
    flat_actions = actions_numpy.reshape(
        -1, policy.action_size
    )
    flat_kappa_per_ns, flat_phi_p_rad = decode_action(
        flat_actions,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
        config.balanced_coupling,
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
        flat_kappa_per_ns, config.maximum_kappa_per_ns
    )

    # Keep the physical objective and engineering preference visibly separate.
    training_reward = (
        phase_reward
        - config.magnitude_symmetry_weight
        * magnitude_symmetry_penalty
    )

    # Compare candidates only with candidates generated for the same target.
    reward_matrix = training_reward.reshape(
        config.targets_per_batch, config.candidates_per_target
    )
    advantages = reward_matrix - reward_matrix.mean(axis=1, keepdims=True)
    advantages /= reward_matrix.std(axis=1, keepdims=True) + 1.0e-8
    advantages_tensor = torch.as_tensor(
        advantages,
        dtype=torch.float32,
        device=log_probabilities.device,
    )

    # Minimizing this raises the probability of above-average actions and
    # lowers the probability of below-average actions.
    loss = -(log_probabilities * advantages_tensor).mean()

    optimizer.zero_grad()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config.gradient_clip
    )
    optimizer.step()
    current_maximum_log_std = maximum_log_std_at(iteration, config)
    with torch.no_grad():
        policy.log_std.clamp_(
            config.minimum_log_std,
            current_maximum_log_std,
        )
    mean_action_std = float(
        torch.exp(policy.log_std.detach()).mean().cpu()
    )

    return {
        "loss": float(loss.detach().cpu()),
        "phase_reward": float(np.mean(phase_reward)),
        "magnitude_symmetry_penalty": float(
            np.mean(magnitude_symmetry_penalty)
        ),
        "training_reward": float(np.mean(training_reward)),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "maximum_log_std": current_maximum_log_std,
        "mean_action_std": mean_action_std,
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
    simulation_pool: Any | None = None,
) -> dict[str, np.ndarray]:
    """Simulate the deterministic policy mean for explicit conditions."""
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

    policy.eval()
    with torch.no_grad():
        encoded = encode_for_policy(
            targets,
            policy,
            config,
            detuning_distributions_ghz,
        )
        actions = policy(encoded).detach().cpu().numpy()
    kappa_per_ns, phi_p_rad = decode_action(
        actions,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
        config.balanced_coupling,
    )
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
    history: dict[str, list[float]],
) -> None:
    """Update the two essential learning curves."""
    axes[0].clear()
    axes[0].plot(history["training_reward"], label="total reward")
    axes[0].set(
        title="REINFORCE training",
        xlabel="iteration",
        ylabel="mean total reward",
        ylim=(0.0, 1.0),
    )
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.25)

    axes[1].clear()
    axes[1].plot(history["held_out_reward"], label="held-out phase reward")
    axes[1].set(
        title="Fixed held-out targets",
        xlabel="iteration",
        ylabel="mean phase reward",
        ylim=(0.0, 1.0),
    )
    axes[1].legend(loc="lower left")
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
) -> tuple[PolicyNetwork, torch.optim.Optimizer, dict[str, list[float]]]:
    """Train the conditional Gaussian policy using simple REINFORCE."""
    rng = set_random_seed(config.random_seed)
    device = torch.device(config.device)
    policy = PolicyNetwork(config).to(device) if policy is None else policy
    optimizer = (
        torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        if optimizer is None
        else optimizer
    )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = config.learning_rate
    held_out_targets = sample_target_phases(
        config.held_out_target_count,
        np.random.default_rng(config.random_seed + 10_000),
        config.n_lasers,
        config.structured_target_fraction,
        config.structured_target_jitter_rad,
    )
    held_out_detunings = sample_detuning_distributions(
        config.held_out_target_count,
        np.random.default_rng(config.random_seed + 20_001),
        config,
        iteration=config.detuning_curriculum_iterations,
    )
    held_out_targets, held_out_detunings = sort_by_detuning(
        held_out_targets, held_out_detunings
    )
    history: dict[str, list[float]] = {
        "phase_reward": [],
        "training_reward": [],
        "magnitude_symmetry_penalty": [],
        "loss": [],
        "gradient_norm": [],
        "maximum_log_std": [],
        "mean_action_std": [],
        "held_out_reward": [],
    }
    latest_held_out_reward = np.nan
    best_held_out_reward = -np.inf
    figure_and_axes = make_training_figure(config) if live_plot else None
    progress_figure_directory = None
    if live_plot:
        progress_figure_directory = RESULTS_DIR
        progress_figure_directory.mkdir(parents=True, exist_ok=True)
    simulation_pool = None
    if config.n_jobs == 1:
        print(
            "VCSEL simulation: serial vectorized path for "
            f"{config.targets_per_batch} targets"
        )
    else:
        active_training_jobs = min(
            config.n_jobs, config.targets_per_batch
        )
        split_sizes = [
            len(indices)
            for indices in np.array_split(
                np.arange(config.targets_per_batch),
                active_training_jobs,
            )
        ]
        print(
            f"VCSEL simulation: {active_training_jobs} worker processes "
            f"for {config.targets_per_batch} targets"
        )
        print(f"Targets per worker: {split_sizes}")
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

            metrics = run_training_batch(
                policy,
                optimizer,
                config,
                rng,
                iteration,
                simulation_pool=simulation_pool,
            )
            for key in (
                "phase_reward",
                "training_reward",
                "magnitude_symmetry_penalty",
                "loss",
                "gradient_norm",
                "maximum_log_std",
                "mean_action_std",
            ):
                history[key].append(metrics[key])

            should_validate = (
                iteration == 0
                or (iteration + 1) % config.validation_interval == 0
                or iteration + 1 == config.training_iterations
            )
            if should_validate:
                validation = evaluate_policy(
                    policy,
                    held_out_targets,
                    config,
                    detuning_distributions_ghz=held_out_detunings,
                    simulation_pool=simulation_pool,
                )
                latest_held_out_reward = float(
                    np.mean(validation["phase_reward"])
                )
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
            print(
                f"Iteration {iteration + 1:4d}/"
                f"{config.training_iterations} | "
                f"phase={metrics['phase_reward']:.3f} | "
                f"training={metrics['training_reward']:.3f} | "
                f"held-out={latest_held_out_reward:.3f} | "
                f"K-sym={metrics['magnitude_symmetry_penalty']:.3f} | "
                f"action-std={metrics['mean_action_std']:.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e} | "
                f"loss={metrics['loss']:.3f}"
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
    history: dict[str, list[float]],
) -> plt.Figure:
    """Plot final training and validation curves outside the core loop."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    update_training_figure(figure, axes, history)
    return figure


# ---------------------------------------------------------------------------
# 8. Candidate design, validation, saving, and loading
# ---------------------------------------------------------------------------


def design_for_target(
    policy: PolicyNetwork,
    target_phases_rad: np.ndarray,
    config: DesignerConfig,
    *,
    number_of_candidates: int = 128,
    top_k: int = 1,
    detuning_distribution_ghz: np.ndarray | None = None,
    simulation_pool: Any | None = None,
) -> list[dict[str, np.ndarray | float | int]]:
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

    policy.eval()
    with torch.no_grad():
        encoded = encode_for_policy(
            target,
            policy,
            config,
            detuning_distribution_ghz,
        )
        distribution = policy.distribution(encoded)
        actions = distribution.sample((number_of_candidates,))[:, 0, :]
        actions[0] = distribution.mean[0]  # always include deterministic design
    actions_numpy = actions.detach().cpu().numpy()
    kappa_per_ns, phi_p_rad = decode_action(
        actions_numpy,
        config.maximum_kappa_per_ns,
        config.n_lasers,
        config.force_symmetric_kappa,
        config.force_symmetric_phi_p,
        config.balanced_coupling,
    )
    simulation = simulate_candidates_parallel(
        target[None, :],
        kappa_per_ns[None, :, :, :],
        phi_p_rad[None, :, :, :],
        config,
        detuning_distributions_ghz=detuning_distribution_ghz[None, :],
        simulation_pool=simulation_pool,
    )
    magnitude_symmetry_penalty = normalized_magnitude_symmetry_penalty(
        kappa_per_ns, config.maximum_kappa_per_ns
    )
    phase_reward = simulation["phase_reward"][0]
    ranking_reward = (
        phase_reward
        - config.magnitude_symmetry_weight
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
                "magnitude_symmetry_penalty": float(
                    magnitude_symmetry_penalty[index]
                ),
                "ranking_reward": float(ranking_reward[index]),
            }
        )
    return results


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


def save_checkpoint(
    filename: str | Path,
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: DesignerConfig,
    *,
    iteration: int,
    held_out_reward: float,
) -> Path:
    """Save the standalone edge-conditioned policy checkpoint."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
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
) -> tuple[
    PolicyNetwork,
    torch.optim.Optimizer,
    DesignerConfig,
    dict[str, float | int],
]:
    """Load an edge-conditioned checkpoint and reject incompatible heads."""
    checkpoint = torch.load(
        filename,
        map_location=device or "cpu",
        weights_only=False,
    )
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format != CHECKPOINT_FORMAT:
        if checkpoint_format == (
            "conditional_coupling_edge_v2_detuning_conditioned"
        ):
            raise ValueError(
                "This checkpoint uses the previous combined global encoder "
                "and has no local detuning edge features. Train a new "
                f"{CHECKPOINT_FORMAT} policy."
            )
        if checkpoint_format == "conditional_coupling_simple_v1":
            raise ValueError(
                "This checkpoint uses the older plain 40-output policy "
                "without detuning conditioning. Set TRAIN_NEW_MODEL=True "
                "to train the new policy."
            )
        raise ValueError(
            "This checkpoint belongs to another policy architecture. "
            "Use working_model_backup/conditional_coupling_designer.py to "
            f"load the archived model, or train a new {CHECKPOINT_FORMAT} "
            "model."
        )
    saved_config = dict(checkpoint["config"])
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
    policy = PolicyNetwork(config).to(torch.device(config.device))
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


__all__ = [
    "CHECKPOINT_FORMAT",
    "DEFAULT_N_LASERS",
    "DesignerConfig",
    "PHASE_TICK_LABELS",
    "PHASE_TICKS",
    "PolicyNetwork",
    "action_size_for",
    "calculate_phase_reward",
    "canonicalize_global_phase_conjugate",
    "decode_action",
    "design_for_target",
    "directed_link_indices",
    "encode_detuning_distributions",
    "encode_for_policy",
    "encode_target_phases",
    "evaluate_policy",
    "load_checkpoint",
    "make_relative_to_laser_1",
    "make_vcsel_physical_parameters",
    "maximum_log_std_at",
    "named_phase_targets",
    "normalized_magnitude_symmetry_penalty",
    "plot_training_history",
    "run_training_batch",
    "sample_coupling_designs",
    "sample_detuning_distributions",
    "sample_target_phases",
    "sort_by_detuning",
    "save_checkpoint",
    "set_random_seed",
    "simulate_candidates_parallel",
    "simulate_target_chunk",
    "simulate_with_vcsel",
    "split_simulation_batch",
    "train_reinforce",
    "validate_policy",
]
