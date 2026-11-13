"""Compatible model files and API-owned, atomically written run metadata."""
from dataclasses import asdict, replace
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


def write_run_manifest(path, *, name, public_config, backend_config):
    """Describe the exact recipe and output-only differences from its reference."""
    write_json(path, dict(api_version=1, name=name, public_config=asdict(public_config),
        backend_config=exported_settings(backend_config), reference_source_status=reference_source_status(),
        checkpoint_format=engine.CHECKPOINT_FORMAT,
        numerical_backend=("independent banded-topology batch and precision refinement; existing numerical kernels"
            if (public_config.sampling.topology_sampling_mode in {"band_random", "band_fixed"}
                or public_config.schedules.precision_start is not None) else
            "shared-mask batch; existing policy/simulation/PCGrad kernels"
            if public_config.sampling.reuse_masks_across_pairs else
            "existing topology-input implementation, unmodified"),
        resume_supported=False))
