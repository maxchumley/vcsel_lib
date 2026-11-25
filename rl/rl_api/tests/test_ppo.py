"""Joint conditional likelihoods, controlled PPO updates and matched run wiring."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from rl.rl_api import CouplingModel, Experiment, PPOConfig
from rl.rl_api._compat import core, engine
from rl.rl_api.density_control import prepare_density_size
from rl.rl_api.experiments import train_density_ppo_1000 as base
from rl.rl_api.experiments import train_density_ppo_target_offset_1000 as offset
from rl.rl_api.experiments import train_density_actor_critic_replay_asymphase_1000 as previous
from rl.rl_api.models import build_policy
from rl.rl_api.models.policy import target_phase_offsets
from rl.rl_api.ppo import action_distribution, action_log_prob, clipped_loss, ppo_update
from rl.rl_api.training_batch import pcgrad_update


def prepared(tmp_path, size=5, recipe=base):
    config = replace(recipe.CONFIG.to_backend(tmp_path), n_lasers=size, exploration_std_scale=.5)
    torch.manual_seed(71)
    policy = build_policy(config)
    # Exercise conditional means that differ between mirrored strength draws.
    with torch.no_grad():
        policy.coupling_phase_head.weight.normal_(0., .07)
        item = prepare_density_size(policy, config, np.random.default_rng(9), size, 2)
    return config, policy, item


@pytest.mark.parametrize('size', range(3, 10))
@pytest.mark.parametrize('recipe', [base, offset], ids=['base', 'offset'])
def test_saved_action_rescoring_matches_sampler_and_keeps_strengths(tmp_path, size, recipe):
    config, policy, item = prepared(tmp_path, size, recipe)
    rng = torch.get_rng_state()
    distribution, columns = action_distribution(policy, item)
    torch.testing.assert_close(action_log_prob(distribution, columns, item), item['log_prob'],
                               rtol=2e-5, atol=2e-4)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    assert distribution.loc.shape[-1] == 3*size*(size-1)//2+1
    # Recompute under changed parameters, conditioning on the stored design.
    seen = []
    original = policy.phase_residual_for_coupling
    def track(features, strengths):
        seen.append(np.array(strengths, copy=True))
        return original(features, strengths)
    policy.phase_residual_for_coupling = track
    with torch.no_grad():
        policy.edge_output.bias.add_(.05)
    changed, columns = action_distribution(policy, item)
    np.testing.assert_array_equal(seen[0], item['kappa'].reshape(-1, size, size))
    assert not torch.equal(distribution.loc, changed.loc)
    (-action_log_prob(changed, columns, item).mean()).backward()
    assert policy.coupling_phase_head.weight.grad.abs().sum() > 0
    assert policy.edge_output.weight.grad.abs().sum() > 0
    assert not item['actions'].requires_grad


def test_clipping_for_positive_and_negative_advantages():
    ratio = torch.tensor([1.5, .5, 1.5, .5], requires_grad=True)
    losses = clipped_loss(ratio.log(), torch.zeros(4), torch.tensor([1., 1., -1., -1.]), .1)
    torch.testing.assert_close(losses, torch.tensor([-1.1, -.5, 1.5, .9]))
    losses.sum().backward()
    torch.testing.assert_close(ratio.grad, torch.tensor([0., -1., 1., 0.]))


def test_first_update_matches_real_reward_policy_gradient_and_reuses_batch(tmp_path):
    config, policy, item = prepared(tmp_path)
    reference = build_policy(config)
    reference.load_state_dict(policy.state_dict())
    distribution, columns = action_distribution(reference, item)
    reference_item = dict(item, log_prob=action_log_prob(distribution, columns, item))
    advantages = [np.sin(np.arange(80)).reshape(2, 40)]
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-5)
    pcgrad_update(reference, reference_optimizer, config, np.random.default_rng(2),
                  [reference_item], advantages)
    result = ppo_update(policy, optimizer, config, np.random.default_rng(2), [item], advantages,
                        PPOConfig(enabled=True, updates=1))
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=2e-6, rtol=1e-5)
    assert result['ppo_updates'] == 1
    assert result['ppo_kl'] >= 0


def test_four_updates_kl_stopping_and_deterministic_skip(tmp_path):
    config, policy, item = prepared(tmp_path)
    advantages = [np.sin(np.arange(80)).reshape(2, 40)]
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    result = ppo_update(policy, optimizer, config, np.random.default_rng(4), [item], advantages,
                        PPOConfig(enabled=True, target_kl=100.))
    assert result['ppo_updates'] == 4 and not result['ppo_early_stop']
    config, policy, item = prepared(tmp_path)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    result = ppo_update(policy, optimizer, config, np.random.default_rng(4), [item], advantages,
                        PPOConfig(enabled=True, target_kl=1e-15))
    assert result['ppo_updates'] == 1 and result['ppo_early_stop']
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    step = [state['step'].clone() for state in optimizer.state.values()]
    result = ppo_update(policy, optimizer, replace(config, exploration_std_scale=0.),
                        np.random.default_rng(4), [item], advantages, base.CONFIG.ppo)
    assert result['ppo_updates'] == 0 and result['mean_action_only']
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for a, state in zip(step, optimizer.state.values()):
        torch.testing.assert_close(a, state['step'], rtol=0, atol=0)


def test_extra_updates_do_not_shift_future_context_rng(tmp_path):
    states = []
    for updates in (1, 4):
        config, policy, item = prepared(tmp_path)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
        rng = np.random.default_rng(4)
        ppo_update(policy, optimizer, config, rng, [item],
            [np.sin(np.arange(80)).reshape(2, 40)],
            PPOConfig(enabled=True, updates=updates, target_kl=100.))
        states.append(rng.random(10))
    np.testing.assert_array_equal(*states)


def test_runs_differ_only_by_target_offset_and_names():
    a, b, old = asdict(base.CONFIG), asdict(offset.CONFIG), asdict(previous.CONFIG)
    assert not a['model'].pop('add_target_phase_offset') and b['model'].pop('add_target_phase_offset')
    old['model'].pop('add_target_phase_offset')
    for value in (a, b, old): value['validation'].pop('progress_filename')
    assert a == b
    a.pop('ppo'); old.pop('ppo'); a.pop('actor_critic'); old.pop('actor_critic')
    assert a == old
    assert not base.CONFIG.actor_critic.enabled and not base.CONFIG.actor_critic.use_replay
    assert base.INITIAL_CHECKPOINT is offset.INITIAL_CHECKPOINT is None
    assert base.CONFIG.summary()['simulations_per_iteration'] == 1400
    assert base.CONFIG.summary()['validation_simulations_per_check'] == 448
    script = Path(__file__).resolve().parents[3]/'scripts/submit_density_ppo_tests.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    output = subprocess.run(['bash', str(script), '--dry-run'], check=True,
                            capture_output=True, text=True).stdout
    assert output.count('sbatch ') == 2 and output.count('--time=7-00:00:00') == 2
    assert '--job-name=density_ppo ' in output and '--job-name=density_ppo_offset ' in output


def test_offset_exact_sign_and_checkpoint_inference(tmp_path):
    a = base.CONFIG.to_backend(tmp_path/'a')
    b = offset.CONFIG.to_backend(tmp_path/'b')
    torch.manual_seed(1)
    policy, shifted = build_policy(a), build_policy(b)
    shifted.load_state_dict(policy.state_dict())
    features = engine.encode_context(np.array([[0., .7, -1.2]]), np.array([[-.4, -.1, .5]]), policy)
    masks = torch.ones((1, 6))
    mean = policy.parameters_for_edges(features, topology_masks=masks)[0]
    shifted_mean = shifted.parameters_for_edges(features, topology_masks=masks)[0]
    torch.testing.assert_close(mean[:, :6], shifted_mean[:, :6], rtol=0, atol=0)
    torch.testing.assert_close(shifted_mean[:, 6:]-mean[:, 6:], target_phase_offsets(features))
    receivers, sources = core.directed_link_indices(3)
    theta = torch.atan2(features[..., 1], features[..., 0])
    expected = torch.atan2(torch.sin(theta[:, sources]-theta[:, receivers]),
                           torch.cos(theta[:, sources]-theta[:, receivers]))
    torch.testing.assert_close(target_phase_offsets(features), expected)
    from rl.rl_api.checkpoints import save
    path = tmp_path/'shifted.pt'
    save(path, shifted, torch.optim.Adam(shifted.parameters()), b, 1, .5)
    model = CouplingModel.load(path)
    assert model.backend_config.add_target_phase_offset
    torch.testing.assert_close(model.policy.parameters_for_edges(features, topology_masks=masks)[0], shifted_mean)


@pytest.mark.parametrize('recipe', [base, offset], ids=['base', 'offset'])
def test_training_pipeline_no_extra_simulations_no_critic_artifacts(tmp_path, fake_physics, recipe):
    c = recipe.CONFIG
    c = replace(c, optimization=replace(c.optimization, workers=1, iterations=2),
        sampling=replace(c.sampling, pairs_per_size=2, actions_per_topology=4),
        schedules=replace(c.schedules, ordinary_end=2, wide_end=2),
        validation=replace(c.validation, pairs_per_size=5, plot_use_tex=False,
                           notebook_display=False, plot_dpi=40))
    result = Experiment('ppo-smoke', config=c, output_directory=tmp_path/'run').train(
        live_plot=True, verify_reference=False)
    assert 1 <= result.history['ppo_updates'][0] <= 4 and result.history['ppo_updates'][1] == 0
    assert 'critic_loss' not in result.history
    assert sum(len(k) for k, _, _ in fake_physics) == 630
    assert not list(result.directory.glob('critic*')) and not (result.directory/'experiences').exists()
    manifest = json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['ppo']['enabled'] and 'PPO' in manifest['numerical_backend']
    model = CouplingModel.load(result.checkpoint('final'))
    assert model.backend_config.add_target_phase_offset == recipe.CONFIG.model.add_target_phase_offset
    from rl.rl_api.plotting.training import training_figure
    with training_figure(result.history, model.backend_config, c.validation) as figure:
        assert len(figure.axes) == 2 and figure.axes[0].get_title().startswith('PPO training')
    assert (result.directory.parent/c.validation.progress_filename).exists()


@pytest.mark.parametrize('values', [dict(clip_ratio=0.), dict(target_kl=float('nan')), dict(updates=0)])
def test_invalid_ppo_settings(values):
    with pytest.raises(ValueError): PPOConfig(**values)
