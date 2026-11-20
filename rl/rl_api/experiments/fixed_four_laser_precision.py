"""Narrow-exploration continuation of a verified fixed four-laser checkpoint."""
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from .. import CouplingModel

START_SCALE = .03
END_SCALE = .003


def load_precision_policy(path, directory, *, iterations, workers, actions, topology, seed=7):
    """Keep weights, physics and budget calibration; the caller creates new Adam."""
    path = Path(path).expanduser().resolve()
    model = CouplingModel.load(path, device='cpu', output_directory=directory)
    source = model.backend_config
    recipe = model.metadata.get('sampling_recipe') or {}
    if (recipe.get('mode') not in {'fixed_context_fresh_joint', 'fixed_context_precision_joint'}
            or recipe.get('span_ghz') != 4
            or not np.array_equal(recipe.get('target_phases_rad'), np.zeros(4))
            or not np.allclose(recipe.get('detunings_ghz', []), np.linspace(-2.,2.,4))
            or not np.array_equal(recipe.get('topology_receiver_source'), topology)
            or not getattr(source, 'symmetric_coupling_values', False)
            or source.symmetric_action_score != 'shared_pair'
            or source.phase_parameterization != 'target_delay_residual'
            or source.objective_mode != 'phase_only'):
        raise ValueError('Precision continuation requires the symmetric fixed-topology 4 GHz in-phase checkpoint')
    config = replace(source, training_iterations=iterations, n_jobs=workers, random_seed=seed,
        candidates_per_target=actions, actions_per_topology=actions,
        learning_rate=1e-5, learning_rate_final=1e-6, learning_rate_decay_start_iteration=0,
        exploration_std_scale=START_SCALE, phase_exploration_fraction=0., phase_exploration_start_std=0.,
        enable_exploration_annealing=False, enable_minimum_log_std_annealing=False,
        minimum_log_std=model.policy.minimum_log_std, maximum_log_std=model.policy.maximum_log_std,
        jupyter_mode=False,
        best_checkpoint_file=str(directory/'best.pt'), current_checkpoint_file=str(directory/'current.pt'),
        final_checkpoint_file=str(directory/'final.pt'))
    model.policy.config = config
    initialization = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_iteration=model.metadata['iteration'], source_reward=model.metadata['held_out_reward'],
        copied='all policy weights, physical settings and budget calibration',
        reset='Adam optimizer, sampler seed, reward scale, schedules and history')
    return model.policy, config, initialization


def iteration_config(config, iteration):
    """Narrow exploration for every update, then one exact-mean evaluation."""
    if iteration == config.training_iterations-1:
        return replace(config, exploration_std_scale=0.)
    progress = iteration/max(config.training_iterations-2, 1)
    scale = START_SCALE*(END_SCALE/START_SCALE)**progress
    return replace(config, exploration_std_scale=scale)
