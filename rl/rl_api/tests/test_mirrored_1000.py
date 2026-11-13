"""Validate the paired estimator, old-run compatibility and new run wiring."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle

import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, CouplingModel, ValidationConfig, Topology
from rl.rl_api._compat import core, engine
from rl.rl_api.advantages import exclude_pair_baseline, pair_excluded_advantages
from rl.rl_api.experiments import train_mirrored_1000 as run, train_anneal_1000 as baseline
from rl.rl_api.exploration import refinement_config, sample_annealed_candidates
from rl.rl_api.models import build_policy
from rl.rl_api.sampling import sample_band_random_topologies
from rl.rl_api.tests.test_parity import assert_equal
from rl.rl_api import training, training_batch


def ready_policy(directory, printed=598):
    config = run.CONFIG.to_backend(directory)
    torch.manual_seed(7)
    policy = build_policy(config)
    current = refinement_config(config, run.CONFIG.schedules, printed-1)
    policy.config = current
    policy.minimum_log_std = core.minimum_log_std_at(printed-1, current)
    policy.maximum_log_std = core.maximum_log_std_at(printed-1, current)
    return policy, current


def features_and_masks(policy):
    features = engine.encode_context([[0., -.7, 1.3]], [[-2., 0., 2.]], policy)
    masks = sample_band_random_topologies(1, 5, 3, np.random.default_rng(11), 'cpu')
    return features, masks.repeat_interleave(8, dim=1)


def gaussian_parameters(policy, features, masks, *, mirrored=False):
    batch, candidates, edges = masks.shape
    repeated = features[:, None].expand(-1, candidates, -1, -1).reshape(-1, 3, 4)
    mean, log_std, _, pooled = policy.parameters_for_edges(repeated,
        topology_masks=masks.reshape(-1, edges), return_pooled=True)
    mean = mean.reshape(batch, candidates, 2*edges)
    std = log_std.reshape_as(mean).exp()
    extra = torch.zeros_like(std[..., edges:])
    extra[:, torch.arange(candidates) % 8 < 2] = np.pi*policy.config.exploration_std_scale
    std = torch.cat((std[..., :edges], (std[..., edges:]**2+extra**2).sqrt()), dim=-1)
    bm, bs = policy.budget_parameters(repeated, pooled_states=pooled)
    parameters = mean, std, bm.reshape(batch, candidates), bs.reshape(batch, candidates).exp()
    if mirrored and policy.config.exploration_std_scale != 0.:
        # Both scores belong to the positive member's Gaussian, even if raw
        # partner predictions differ by an ULP. The zero-SD endpoint uses raw means.
        parameters = tuple(value[:, ::2].repeat_interleave(2, dim=1)
                           for value in parameters)
    return parameters


def test_only_the_two_estimator_options_change_and_no_run_is_started(tmp_path):
    old, new = asdict(baseline.CONFIG), asdict(run.CONFIG)
    new['sampling']['action_sampling'] = 'independent'
    new['optimization']['reward_baseline'] = 'topology_mean'
    assert_equal(old, new)
    assert run.RUN_NAME != baseline.RUN_NAME
    assert run.START_TRAINING is True
    assert run.RESTART_EXISTING_RUN is True
    source = open(run.__file__).read()
    assert source.count('#%%') == 1 and 'def make_experiment' not in source
    assert 'current_process().name == "MainProcess"' in source
    experiment = Experiment('preview', config=run.CONFIG, output_directory=tmp_path/'preview')
    assert not experiment.directory.exists()
    summary = run.CONFIG.summary()
    assert summary['simulations_per_iteration'] == 1400
    assert summary['worker_jobs'] == 35
    assert summary['pcgrad_tasks'] == 21
    assert summary['action_sampling'] == 'mirrored'
    assert summary['reward_baseline'] == 'pair_excluded'
    old_config = baseline.CONFIG.to_backend(tmp_path/'old')
    assert 'action_sampling' not in asdict(old_config)
    assert 'reward_baseline' not in asdict(old_config)
    states = []
    for public in (baseline.CONFIG, run.CONFIG):
        torch.manual_seed(7)
        policy = build_policy(public.to_backend(tmp_path))
        assert sum(p.numel() for p in policy.parameters()) == 127304
        states.append((policy.state_dict(), torch.get_rng_state()))
    assert_equal(*states)


@pytest.mark.parametrize('printed', [1, 598, 999])
@pytest.mark.parametrize('partner_roundoff', [False, True])
def test_three_ordinary_pairs_one_wide_pair_scores_and_rng(
    tmp_path, monkeypatch, printed, partner_roundoff,
):
    policy, config = ready_policy(tmp_path, printed)
    if partner_roundoff:
        edge_parameters = policy.parameters_for_edges

        def rounded_edges(*args, **options):
            mean, log_std, *other = edge_parameters(*args, **options)
            mean = mean.clone()
            # Force a one-ULP row difference even on BLAS backends with equal rows.
            with torch.no_grad():
                mean[1::2] = torch.nextafter(
                    mean[1::2], torch.full_like(mean[1::2], torch.inf))
            return mean, log_std, *other

        monkeypatch.setattr(policy, 'parameters_for_edges', rounded_edges)
    features, masks = features_and_masks(policy)
    torch.manual_seed(42)
    independent = sample_annealed_candidates(policy, features, masks)
    independent_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    actions, budget, score = sample_annealed_candidates(policy, features, masks, mirrored=True)
    assert torch.equal(torch.get_rng_state(), independent_rng)
    assert not actions.requires_grad and not budget.requires_grad
    # Shared positive samples are kept bit for bit, including at small SDs.
    torch.testing.assert_close(actions[:, ::2], independent[0][:, ::2], rtol=0, atol=0)
    torch.testing.assert_close(budget[:, ::2], independent[1][:, ::2], rtol=0, atol=0)
    mean, std, bm, bs = gaussian_parameters(policy, features, masks, mirrored=True)
    torch.testing.assert_close(actions[:, ::2]+actions[:, 1::2], 2*mean[:, ::2])
    torch.testing.assert_close(budget[:, ::2]+budget[:, 1::2], 2*bm[:, ::2])
    torch.testing.assert_close(std[:, ::2], std[:, 1::2], rtol=0, atol=0)
    assert torch.all(std[:, 0, 6:] > std[:, 2, 6:])
    expected = (torch.distributions.Normal(mean, std).log_prob(actions)
        * torch.cat((masks, masks), dim=-1)).sum(-1)
    expected += torch.distributions.Normal(bm, bs).log_prob(budget)
    torch.testing.assert_close(score, expected, rtol=0, atol=0)
    weights = torch.linspace(-1., 1., score.numel()).reshape_as(score)
    (score*weights).mean().backward()
    assert torch.isfinite(policy.edge_output.weight.grad).all()
    assert policy.edge_output.weight.grad[1].abs().max() > 0
    assert torch.isfinite(policy.budget_log_std.grad)


def test_zero_endpoint_returns_means_and_consumes_no_sampling_rng(tmp_path):
    policy, _ = ready_policy(tmp_path, 1000)
    features, masks = features_and_masks(policy)
    before = torch.get_rng_state().clone()
    actions, budget, score = sample_annealed_candidates(policy, features, masks, mirrored=True)
    assert torch.equal(before, torch.get_rng_state())
    mean, _, bm, _ = gaussian_parameters(policy, features, masks)
    torch.testing.assert_close(actions, mean, rtol=0, atol=0)
    torch.testing.assert_close(budget, bm, rtol=0, atol=0)
    assert torch.count_nonzero(score) == 0


def test_roundoff_between_partner_predictions_uses_one_scored_gaussian(tmp_path, monkeypatch):
    """Repeated rows may differ by an ULP, especially after an update/with BLAS threads."""
    policy, _ = ready_policy(tmp_path)
    features, masks = features_and_masks(policy)
    edge_parameters = policy.parameters_for_edges
    budget_parameters = policy.budget_parameters
    def perturbed_edges(*args, **options):
        mean, log_std, *other = edge_parameters(*args, **options)
        mean = mean.clone(); log_std = log_std.clone()
        mean[1::2] += 2e-7
        log_std[1::2] += 6e-8
        return mean, log_std, *other
    def perturbed_budget(*args, **options):
        mean, log_std = budget_parameters(*args, **options)
        mean = mean.clone(); log_std = log_std.clone()
        mean[1::2] += 2e-7
        log_std[1::2] += 6e-8
        return mean, log_std
    monkeypatch.setattr(policy, 'parameters_for_edges', perturbed_edges)
    monkeypatch.setattr(policy, 'budget_parameters', perturbed_budget)
    mean, std, bm, bs = gaussian_parameters(policy, features, masks)
    assert not torch.equal(mean[:, ::2], mean[:, 1::2])
    mean = mean[:, ::2].repeat_interleave(2, dim=1)
    std = std[:, ::2].repeat_interleave(2, dim=1)
    bm = bm[:, ::2].repeat_interleave(2, dim=1)
    bs = bs[:, ::2].repeat_interleave(2, dim=1)
    actions, budget, score = sample_annealed_candidates(policy, features, masks, mirrored=True)
    torch.testing.assert_close(actions[:, ::2]+actions[:, 1::2], 2*mean[:, ::2])
    torch.testing.assert_close(budget[:, ::2]+budget[:, 1::2], 2*bm[:, ::2])
    expected = (torch.distributions.Normal(mean, std).log_prob(actions)
        * torch.cat((masks, masks), dim=-1)).sum(-1)
    expected += torch.distributions.Normal(bm, bs).log_prob(budget)
    torch.testing.assert_close(score, expected, rtol=0, atol=0)
    score.mean().backward()
    assert torch.isfinite(policy.edge_output.weight.grad).all()


def test_mirroring_still_rejects_different_partner_topologies(tmp_path):
    policy, _ = ready_policy(tmp_path)
    features, masks = features_and_masks(policy)
    masks[0, 1, 0] = 1-masks[0, 1, 0]
    with pytest.raises(ValueError, match='same supplied topology'):
        sample_annealed_candidates(policy, features, masks, mirrored=True)


def test_pair_baseline_uses_only_other_six_and_updates_reward_scale_once(tmp_path):
    config = run.CONFIG.to_backend(tmp_path)
    reward = np.array([[.9, .6, .8, .7, .75, .75, .75, .75,
                        .1, .2, .3, .4, .5, .6, .7, .8],
                       [.2, .4, .3, .5, .6, .8, .7, .9,
                        .8, .6, .4, .2, .1, .3, .5, .7]])
    state, legacy_state = {'variance': .04}, {'variance': .04}
    actual = pair_excluded_advantages([reward], config, state)[0]
    engine._topology_advantages([reward], config, legacy_state)
    assert state == legacy_state
    scale = np.sqrt(state['variance'])
    expected = np.empty_like(reward)
    for row in range(2):
        for start in (0, 8):
            for slot in range(8):
                pair = slot//2
                others = [i for i in range(8) if i//2 != pair]
                expected[row, start+slot] = (reward[row, start+slot]
                    - reward[row, start+np.array(others)].mean())/scale
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert actual[0, 0] == pytest.approx(.15/scale)
    assert actual[0, 1] == pytest.approx(-.15/scale)
    # Altering a pair cannot change its own unscaled baseline.
    altered = reward.copy(); altered[:, :2] += .2
    a = reward-exclude_pair_baseline(reward, 8)
    b = altered-exclude_pair_baseline(altered, 8)
    np.testing.assert_allclose(a[:, :2], b[:, :2])


def test_pair_excluded_scores_learn_both_mean_and_sd_on_known_quadratic():
    rng = np.random.default_rng(991)
    mu, sigma, target = .4, .7, 1.
    z = rng.standard_normal((100000, 8))
    z[:, 1::2] = -z[:, ::2]
    reward = -(mu+sigma*z-target)**2
    advantage = exclude_pair_baseline(reward, 8)
    mean_gradient = np.mean(advantage*z/sigma)
    log_sd_gradient = np.mean(advantage*(z*z-1))
    np.testing.assert_allclose([mean_gradient, log_sd_gradient],
        [2*(target-mu), -2*sigma**2], rtol=0, atol=.025)


def test_full_pipeline_records_pairing_21_tasks_and_mean_only_endpoint(tmp_path, monkeypatch, fake_physics):
    schedules = replace(run.CONFIG.schedules, ordinary_end=3, wide_end=3,
                        learning_rate_decay_start=2, learning_rate_decay_end=3)
    public = replace(run.CONFIG, schedules=schedules,
        optimization=replace(run.CONFIG.optimization, iterations=3),
        validation=ValidationConfig(pairs_per_size=4))
    from rl.rl_api.simulation import simulate_indexed_chunk
    calls = []
    class Pool:
        def __init__(self, processes, initializer): assert processes == 14
        def imap_unordered(self, worker, jobs):
            jobs = list(jobs)
            if worker is simulate_indexed_chunk:
                assert len(jobs) == 35
                assert all(j[3][4].shape[:2] == (5, 8) for j in jobs)
                calls.append(jobs)
            return reversed([worker(j) for j in jobs])
        def close(self): pass
        def join(self): pass
    class Context:
        def Pool(self, **options): return Pool(**options)
    monkeypatch.setattr(training.mp, 'get_context', lambda method: Context())
    combine = core.combine_task_gradients_pcgrad
    def check_weights(gradients, clip, rng, **options):
        np.testing.assert_allclose(options['task_weights'], np.tile(np.array([.4, .2, .4])/7, 7))
        return combine(gradients, clip, rng, **options)
    monkeypatch.setattr(core, 'combine_task_gradients_pcgrad', check_weights)
    result = Experiment('paired-test', config=public, output_directory=tmp_path/'run').train(live_plot=False)
    assert len(calls) == 3
    assert result.history['pcgrad_task_count'] == [21]*3
    assert result.history['rooted_fraction'] == [1.]*3
    data = torch.load(result.checkpoint('final'), weights_only=False)
    assert data['config']['action_sampling'] == 'mirrored'
    assert data['config']['reward_baseline'] == 'pair_excluded'
    assert data['config']['exploration_std_scale'] == 0.
    assert data['sampling_recipe']['ordinary_action_pairs'] == 3
    assert data['sampling_recipe']['wide_action_pairs'] == 1
    assert data['sampling_recipe']['sampler'] == 'band_random'
    assert all(int(v['step']) == 2 for v in data['optimizer_state_dict']['state'].values())
    history = json.loads((result.directory/'current_history.json').read_text())
    assert history['config']['reward_baseline'] == 'pair_excluded'
    loaded = CouplingModel.load(result.checkpoint('final'), device='cpu', output_directory=tmp_path)
    assert loaded.backend_config.action_sampling == 'mirrored'
    assert loaded.backend_config.reward_baseline == 'pair_excluded'
    assert type(pickle.loads(pickle.dumps(loaded.backend_config))) is type(loaded.backend_config)
    topology = Topology.from_receiver_source(np.ones((3, 3))-np.eye(3))
    options = dict(target_phases_rad=[0., -.7, 1.3], detunings_ghz=[-1., 0., 1.], topology=topology, count=3)
    mean = loaded.propose(**options, continuous_actions='mean', seed=7)
    sample = loaded.propose(**options, continuous_actions='sample', seed=91)
    np.testing.assert_array_equal(mean.coupling_strengths_per_ns, sample.coupling_strengths_per_ns)
    np.testing.assert_array_equal(mean.coupling_phases_rad, sample.coupling_phases_rad)


def test_mirrored_endpoint_keeps_adam_momentum_and_weights_fixed(tmp_path, fake_physics):
    policy, _ = ready_policy(tmp_path, 999)
    config = replace(run.CONFIG.to_backend(tmp_path), training_n_lasers=(3,), targets_per_batch=5)
    config = engine.calibrate_budget_references(policy, config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(9)
    options = dict(reward_scale_state={}, mask_sampler=sample_band_random_topologies,
                   schedules=run.CONFIG.schedules)
    training_batch.run_training_batch(policy, optimizer, config, rng, 998, **options)
    before = deepcopy((policy.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
    result = training_batch.run_training_batch(policy, optimizer, config, rng, 999, **options)
    assert result['mean_action_only'] is True
    assert_equal(before, (policy.state_dict(), optimizer.state_dict(), torch.get_rng_state()))


def test_invalid_pair_settings_are_rejected():
    with pytest.raises(ValueError, match='even actions_per_topology'):
        replace(run.CONFIG, sampling=replace(run.CONFIG.sampling, actions_per_topology=7))
    with pytest.raises(ValueError, match='even wide_phase_actions'):
        replace(run.CONFIG, schedules=replace(run.CONFIG.schedules, wide_phase_actions=1))
    with pytest.raises(ValueError, match="requires reward_baseline"):
        replace(run.CONFIG, optimization=replace(run.CONFIG.optimization, reward_baseline='topology_mean'))
    with pytest.raises(ValueError, match='requires action_sampling'):
        replace(run.CONFIG, sampling=replace(run.CONFIG.sampling, action_sampling='independent'))
    with pytest.raises(ValueError, match='anneal_to_mean'):
        public = replace(baseline.CONFIG.schedules, anneal_to_mean=False, ordinary_start=1000,
                         ordinary_end=1500, wide_start=1000, wide_end=1500)
        replace(run.CONFIG, schedules=public)
