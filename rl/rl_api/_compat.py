"""Single boundary to the existing, unchanged numerical implementation.

Version one deliberately reuses the verified policy, simulator, sampler and
gradient routines. Nothing here imports an experiment runner or starts work.
The readable API/configuration and artifact ownership live in this package.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rl import conditional_coupling_bernoulli_sparse as engine
from rl import conditional_coupling_designer_variable_m as core

PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PACKAGE_DIR.parents[1]


@dataclass
class PhaseModelConfig(engine.BernoulliSparseConfig):
    """Carry VCSEL's phase convention through worker jobs and checkpoints."""
    phase_model: str = "chumley_2026"


@dataclass
class ResidualPhaseConfig(PhaseModelConfig):
    """Record the opt-in phase-mean transform without changing legacy configs."""
    phase_parameterization: str = "target_residual"


@dataclass
class AnnealedExplorationConfig(ResidualPhaseConfig):
    """Record the actual action-SD multiplier for opt-in anneal-to-mean runs."""
    exploration_std_scale: float = 1.0


@dataclass
class MirroredExplorationConfig(AnnealedExplorationConfig):
    """Record the opt-in mirrored sampler and its pair-independent baseline."""
    action_sampling: str = "mirrored"
    reward_baseline: str = "pair_excluded"


@dataclass
class RandomReachableTopology:
    """Opt-in sampler metadata; older checkpoint schemas remain unchanged."""
    topology_random_reachable: bool = True


@dataclass
class ReachableBackendConfig(engine.BernoulliSparseConfig, RandomReachableTopology):
    pass


@dataclass
class ReachablePhaseModelConfig(PhaseModelConfig, RandomReachableTopology):
    pass


@dataclass
class ReachableResidualPhaseConfig(ResidualPhaseConfig, RandomReachableTopology):
    pass


@dataclass
class ReachableAnnealedConfig(AnnealedExplorationConfig, RandomReachableTopology):
    pass


@dataclass
class ReachableMirroredConfig(MirroredExplorationConfig, RandomReachableTopology):
    pass


@dataclass
class RootEncoderInput:
    """Saved only for new experiments that explicitly encode the chosen root."""
    encode_root: bool = True


@dataclass
class RootEncodedResidualConfig(ReachableResidualPhaseConfig, RootEncoderInput):
    pass


@dataclass
class RootEncodedAnnealedConfig(ReachableAnnealedConfig, RootEncoderInput):
    pass


@dataclass
class RootEncodedMirroredConfig(ReachableMirroredConfig, RootEncoderInput):
    pass

# Native vcsel_lib SI names -> names in the unchanged training dataclass.
VCSEL_FIELDS = {
    "tau_p": "photon_lifetime_seconds", "tau_n": "carrier_lifetime_seconds",
    "g0": "gain_per_second", "N0": "transparency_carriers", "s": "gain_saturation",
    "q": "electron_charge_coulomb", "beta": "spontaneous_emission_factor",
    "I": "pump_current_amp", "alpha": "alpha", "tau": "delay_seconds",
    "dt": "time_step_seconds", "Tmax": "simulation_time_seconds",
    "save_every": "save_every", "noise_amplitude": "noise_amplitude",
}


def vcsel_parameters_to_settings(phys):
    """Adapt native VCSEL parameters without changing units or scaling them.

    M, detunings, strengths and phases belong to each RL context/action, so
    those template entries are replaced for every simulation batch. Additional
    physics modes need worker support; never silently accept an ignored mode.
    """
    missing = set(VCSEL_FIELDS)-set(phys)
    if missing:
        raise ValueError("Missing VCSEL physical parameters: " + ", ".join(sorted(missing)))
    dynamic = {"N_lasers", "delta", "kappa_c_mat", "phi_p_mat", "N_bar"}
    fixed = {"coupling": 1.0, "self_feedback": 0.0}
    unknown = set(phys)-set(VCSEL_FIELDS)-dynamic-set(fixed)-{"phase_model"}
    if unknown:
        raise ValueError("Physical options not supported by the current training worker: "
                         + ", ".join(sorted(unknown)))
    for key, expected in fixed.items():
        if key in phys and phys[key] != expected:
            raise NotImplementedError(f"Current training uses {key}={expected!r}")
    settings = {old_name: float(phys[name]) for name, old_name in VCSEL_FIELDS.items()}
    if settings["save_every"] < 1 or not settings["save_every"].is_integer():
        raise ValueError("save_every must be a positive integer")
    settings["save_every"] = int(settings["save_every"])
    phase_model = phys.get("phase_model", "chumley_2026")
    if phase_model not in {"ma_2019", "standard_lk", "chumley_2026"}:
        raise ValueError(f"Unknown phase_model {phase_model!r}")
    if phase_model != "chumley_2026":
        settings["phase_model"] = phase_model
    return settings


def reference_settings() -> dict:
    """Read the historical snapshot for diagnostics only, never runtime settings."""
    return json.loads((PACKAGE_DIR / "presets/current_training.json").read_text())


def compatibility_settings():
    """Inactive metadata needed to match the old dataclass exactly.

    These values affect neither the active policy nor its loss. Experiment
    parameters are Python configuration objects in the run file. Keeping this
    small compatibility block preserves old config comparisons without using
    the historical JSON as a runtime preset.
    """
    return dict(initial_gate_logit=0.0, initial_kappa_fraction=.05,
        successful_sparsity_reward_weight=0.0,
        detuning_curriculum_initial_half_span_ghz=.005,
        validation_follows_detuning_curriculum=True,
        sparsity_reward_mode="links_per_laser",
        lexicographic_initial_phase_error_deg=30.,
        lexicographic_initial_minimum_phase_reward=.9330127018922194,
        budget_exploration_start_std=.35, budget_exploration_end_std=.35)


def reference_source_status() -> dict[str, bool]:
    """Accept exact baseline hashes or explicitly reviewed source corrections."""
    manifest = json.loads((PACKAGE_DIR / "presets/reference_manifest.json").read_text())
    revisions = manifest.get("reviewed_source_revisions", {})
    status = {}
    for name, digest in manifest["source_files"].items():
        accepted = {digest, *(item["sha256"] for item in revisions.get(name, []))}
        path = REPOSITORY_DIR / name
        status[name] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() in accepted
    return status


def make_backend_config(settings: dict):
    """Restore tuple-valued fields and use the existing validated schema."""
    converted = dict(settings)
    for name in ("training_n_lasers", "training_n_laser_weights", "node_hidden_sizes",
                 "edge_hidden_sizes"):
        if converted.get(name) is not None:
            converted[name] = tuple(converted[name])
    converted["coupling_budget_reference_by_m"] = tuple(
        tuple(item) for item in converted.get("coupling_budget_reference_by_m", ()))
    if "action_sampling" in converted:
        config_class = MirroredExplorationConfig
    elif "exploration_std_scale" in converted:
        config_class = AnnealedExplorationConfig
    elif "phase_parameterization" in converted:
        config_class = ResidualPhaseConfig
    else:
        config_class = PhaseModelConfig if "phase_model" in converted else engine.BernoulliSparseConfig
    if 'topology_random_reachable' in converted:
        config_class = {
            engine.BernoulliSparseConfig: ReachableBackendConfig,
            PhaseModelConfig: ReachablePhaseModelConfig,
            ResidualPhaseConfig: ReachableResidualPhaseConfig,
            AnnealedExplorationConfig: ReachableAnnealedConfig,
            MirroredExplorationConfig: ReachableMirroredConfig,
        }[config_class]
    if 'encode_root' in converted:
        config_class = {
            ReachableResidualPhaseConfig: RootEncodedResidualConfig,
            ReachableAnnealedConfig: RootEncodedAnnealedConfig,
            ReachableMirroredConfig: RootEncodedMirroredConfig,
        }[config_class]
    return config_class(**converted)


def exported_settings(config):
    """Omit obsolete teacher fields from API metadata and model payloads.

    The unchanged backend dataclass still defines these attributes. Its loader
    supplies their disabled defaults when reading a checkpoint without them.
    This does not affect model weights or any current training calculation.
    """
    return {key: value for key, value in asdict(config).items()
            if not key.startswith("de_teacher_")}
