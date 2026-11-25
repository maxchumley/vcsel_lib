"""Compatible model files and API-owned, atomically written run metadata."""
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import torch

from ._compat import engine, exported_settings, reference_source_status, make_backend_config
from .models import build_policy


def write_json(path, data):
    """Atomic JSON replacement using the same NaN handling as current histories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def save(path, policy, optimizer, config, iteration, held_out_reward, *, sampling_recipe=None,
         exploration_state=None):
    """Keep the loadable model format without obsolete teacher metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = dict(format=engine.CHECKPOINT_FORMAT, config=exported_settings(config),
        policy_state_dict=policy.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        iteration=iteration, held_out_reward=held_out_reward)
    if sampling_recipe is not None:
        payload['sampling_recipe'] = sampling_recipe
    if exploration_state is not None:
        payload['exploration_state'] = exploration_state
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path, *, device="cpu"):
    """Read existing and API-generated topology-input checkpoints."""
    saved = torch.load(path, map_location=device, weights_only=False)
    if ("exploration_state" in saved or "phase_model" in saved.get("config", {})
            or "phase_parameterization" in saved.get("config", {})):
        if saved.get("format") != engine.CHECKPOINT_FORMAT:
            raise ValueError("Not a Bernoulli sparse checkpoint")
        config = replace(make_backend_config(saved["config"]), device=device)
        policy = build_policy(config)
        policy.load_state_dict(saved["policy_state_dict"])
        state = saved.get("exploration_state")
        if state is not None:
            policy.minimum_log_std = state["minimum_log_std"]
            policy.maximum_log_std = state["maximum_log_std"]
        info = dict(iteration=saved["iteration"], held_out_reward=saved["held_out_reward"],
                    exploration_state=state, sampling_recipe=saved.get("sampling_recipe"))
    else:
        # Retain the original loader's upgrades and legacy SD behavior.
        policy, config, info = engine.load_checkpoint(path, device=device)
    if not config.topology_input_mode:
        raise NotImplementedError("Learned-gate checkpoint support is reserved for the Bernoulli extension")
    return policy, config, info


def initialize_from_checkpoint(policy, config, path):
    """Copy weights and physical budget calibration, never optimizer/exploration state."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    saved = torch.load(io.BytesIO(raw), map_location=config.device, weights_only=False)
    if saved.get('format') != engine.CHECKPOINT_FORMAT:
        raise ValueError('Initialization requires a compatible coupling-model checkpoint')
    source = make_backend_config(saved['config'])
    for key, default in (('phase_parameterization', 'absolute'), ('phase_model', 'chumley_2026'),
                         ('budget_action_transform', 'log_budget'),
                         ('strength_action_transform', 'uncapped_softplus'),
                         ('normalize_incoming_coupling_by_degree', False),
                         ('softplus_strength_scale_per_ns', 1.)):
        if getattr(source, key, default) != getattr(config, key, default):
            raise ValueError(f'Initialization checkpoint has a different {key}')
    references = source.coupling_budget_reference_by_m
    if not set(config.training_n_lasers).issubset(dict(references)):
        raise ValueError('Initialization checkpoint must include budget calibration for every trained size')
    policy.load_state_dict(saved['policy_state_dict'], strict=True)
    config = replace(config, coupling_budget_reference_by_m=references)
    policy.config = config
    metadata = dict(checkpoint=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        source_iteration=saved['iteration'], source_held_out_reward=saved['held_out_reward'],
        copied='policy weights and budget calibration',
        reset='optimizer, sampler, reward scale, exploration schedule and history')
    if getattr(config, 'density_control', False):
        metadata['architecture_change'] = 'none; reuse adjacency-feature slots for uniform density'
        metadata['new_parameters'] = 0
    return config, metadata


def write_run_manifest(path, *, name, public_config, backend_config, initialization=None):
    """Describe the exact recipe and output-only differences from its reference."""
    manifest = dict(api_version=1, name=name, public_config=asdict(public_config),
        backend_config=exported_settings(backend_config), reference_source_status=reference_source_status(),
        checkpoint_format=engine.CHECKPOINT_FORMAT,
        numerical_backend=("all connected symmetric masks and reciprocal physical actions; existing GNN and simulator"
            if public_config.topology.backbone == 'all_connected_symmetric' else
            "density input and strength-ranked symmetric masks; existing simulation/REINFORCE/PCGrad kernels"
            if public_config.topology.mode == 'density' else
            "independent banded-topology batch and precision refinement; existing numerical kernels"
            if (public_config.sampling.topology_sampling_mode in {"band_random", "band_fixed"}
                or public_config.schedules.precision_start is not None) else
            "shared-mask batch; existing policy/simulation/PCGrad kernels"
            if public_config.sampling.reuse_masks_across_pairs else
            "existing topology-input implementation, unmodified"),
        resume_supported=False)
    if initialization is not None:
        manifest['initialization'] = initialization
    if public_config.ppo.enabled:
        manifest['numerical_backend'] = 'fresh-batch PPO/PCGrad; existing density GNN and VCSEL simulator'
    write_json(path, manifest)
