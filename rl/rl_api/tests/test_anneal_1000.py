"""Check the 1k random-count run, actual sampling widths and mean-only endpoint."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import pickle

import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, CouplingModel, ValidationConfig, Topology
from rl.rl_api._compat import core, engine
from rl.rl_api.checkpoints import save
from rl.rl_api.experiments import train_anneal_1000 as run, train_phase_residual as previous
from rl.rl_api.exploration import refinement_config, exploration_state, sample_annealed_candidates
from rl.rl_api.models import build_policy
from rl.rl_api.sampling import sample_band_random_topologies
from rl.rl_api import training, training_batch
from rl.rl_api.tests.test_parity import assert_equal


def ready_policy(directory, iteration):
    config = replace(run.CONFIG.to_backend(directory), training_n_lasers=(3,), targets_per_batch=5)
    torch.manual_seed(7)
    policy = build_policy(config)
    current = refinement_config(config, run.CONFIG.schedules, iteration)
    policy.config = current
    policy.minimum_log_std = core.minimum_log_std_at(iteration, current)
    policy.maximum_log_std = core.maximum_log_std_at(iteration, current)
    return policy, current


def features_and_masks(policy):
    features = engine.encode_context([[0., -.7, 1.3]], [[-2., 0., 2.]], policy)
    masks = sample_band_random_topologies(1, 5, 3, np.random.default_rng(11), 'cpu')
    return features, masks.repeat_interleave(8, dim=1)


def test_recipe_preserves_physics_model_rms_and_exposes_one_cell(tmp_path):
    assert run.CONFIG.phys == previous.CONFIG.phys
    assert run.CONFIG.model == previous.CONFIG.model
    assert run.CONFIG.topology == previous.CONFIG.topology
    assert run.CONFIG.reward == previous.CONFIG.reward
    assert run.CONFIG.validation == previous.CONFIG.validation
    assert run.RUN_NAME != previous.RUN_NAME
    assert run.START_TRAINING is True
    assert run.RESTART_EXISTING_RUN is True
    preview = Experiment('preview',config=run.CONFIG,output_directory=tmp_path/'preview')
    assert not preview.directory.exists()
    assert run.CONFIG.optimization.iterations == 1000
    assert run.CONFIG.optimization.learning_rate == 1e-4
    assert run.CONFIG.optimization.final_learning_rate == 1e-5
    summary = run.CONFIG.summary()
    assert summary['simulations_per_iteration'] == 1400
    assert summary['pcgrad_tasks'] == 21
    assert summary['worker_jobs'] == 35
    assert run.CONFIG.sampling.topology_sampling_mode == 'band_random'
    assert run.CONFIG.sampling.reuse_masks_across_pairs is False
    source = open(run.__file__).read()
    assert source.count('#%%') == 1
    assert 'def make_experiment' not in source
    assert 'current_process().name == "MainProcess"' in source
    states = []
    for public in (previous.CONFIG, run.CONFIG):
        torch.manual_seed(7)
        policy = build_policy(public.to_backend(tmp_path))
        states.append((policy.state_dict(), torch.get_rng_state()))
        assert sum(p.numel() for p in policy.parameters()) == 127304
    assert_equal(states[0], states[1])


@pytest.mark.parametrize('printed,scale', [(1,1.), (500,500/999),
    (900,100/999), (999,1/999), (1000,0.)])
def test_linear_multiplier_applies_to_edges_budget_and_wide_actions(printed, scale, tmp_path):
    policy, config = ready_policy(tmp_path, printed-1)
    assert config.exploration_std_scale == pytest.approx(scale, abs=1e-15)
    assert policy.minimum_log_std == -1.5
    assert policy.maximum_log_std == .75
    features, masks = features_and_masks(policy)
    scaled = policy.parameters_for_edges(features, topology_masks=masks[:,0], return_pooled=True)
    budget_scaled = policy.budget_parameters(features, pooled_states=scaled[3])
    policy.config = replace(config, exploration_std_scale=1.)
    original = policy.parameters_for_edges(features, topology_masks=masks[:,0], return_pooled=True)
    budget_original = policy.budget_parameters(features, pooled_states=original[3])
    torch.testing.assert_close(scaled[0], original[0], rtol=0, atol=0)
    torch.testing.assert_close(scaled[1].exp(), original[1].exp()*max(scale,1e-12))
    torch.testing.assert_close(budget_scaled[1].exp(), budget_original[1].exp()*max(scale,1e-12))
    policy.config = config
    state = exploration_state(policy)
    assert state['action_std_scale'] == pytest.approx(scale)
    assert state['extra_wide_phase_sd_deg'] == pytest.approx(180*scale)
    assert state['effective_nominal_budget_std'] == pytest.approx(.35*scale)


@pytest.mark.parametrize('printed', [1,500,900,999])
def test_sampler_scores_actual_widths_with_finite_phase_gradients(printed, tmp_path):
    policy, config = ready_policy(tmp_path, printed-1)
    features, masks = features_and_masks(policy)
    actions, budget, score = sample_annealed_candidates(policy, features, masks)
    batch, candidates, edges = masks.shape
    repeated = features[:,None].expand(-1,candidates,-1,-1).reshape(-1,3,4)
    mean, log_std, _, pooled = policy.parameters_for_edges(repeated,
        topology_masks=masks.reshape(-1,edges), return_pooled=True)
    mean = mean.reshape_as(actions)
    std = log_std.reshape_as(actions).exp()
    extra = torch.zeros_like(std[...,edges:])
    extra[:, torch.arange(candidates)%8 < 2] = math.pi*config.exploration_std_scale
    std = torch.cat((std[...,:edges], (std[...,edges:]**2+extra**2).sqrt()), dim=-1)
    expected = (torch.distributions.Normal(mean,std).log_prob(actions)
                * torch.cat((masks,masks),dim=-1)).sum(-1)
    bm, bs = policy.budget_parameters(repeated, pooled_states=pooled)
    expected += torch.distributions.Normal(bm.reshape(batch,candidates),
        bs.reshape(batch,candidates).exp()).log_prob(budget)
    torch.testing.assert_close(score,expected,rtol=0,atol=0)
    (score*torch.linspace(-1.,1.,score.numel()).reshape_as(score)).mean().backward()
    assert torch.isfinite(policy.edge_output.weight.grad).all()
    assert policy.edge_output.weight.grad[1].abs().max() > 0


def test_endpoint_returns_exact_means_without_sampling_rng(tmp_path):
    policy, _ = ready_policy(tmp_path,999)
    features, masks = features_and_masks(policy)
    before = torch.get_rng_state().clone()
    actions, budget, score = sample_annealed_candidates(policy,features,masks)
    assert torch.equal(before,torch.get_rng_state())
    repeated = features[:,None].expand(-1,40,-1,-1).reshape(40,3,4)
    mean,_,_,pooled = policy.parameters_for_edges(repeated,
        topology_masks=masks.reshape(40,6),return_pooled=True)
    bm,_ = policy.budget_parameters(repeated,pooled_states=pooled)
    torch.testing.assert_close(actions,mean.reshape_as(actions),rtol=0,atol=0)
    torch.testing.assert_close(budget,bm.reshape_as(budget),rtol=0,atol=0)
    assert torch.count_nonzero(score) == 0


@pytest.mark.parametrize('printed,rate',[(1,1e-4),(899,1e-4),(900,1e-4),
    (950,math.sqrt(1e-4*1e-5)),(1000,1e-5)])
def test_learning_rate_decay_900_through_1000(printed,rate,tmp_path):
    config = run.CONFIG.to_backend(tmp_path)
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))],lr=config.learning_rate)
    training.set_learning_rate(optimizer,config,printed-1,decay_end=1000)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(rate)


def test_final_batch_preserves_weights_and_adam_state_and_reload(tmp_path,fake_physics):
    policy, _ = ready_policy(tmp_path,998)
    config = replace(run.CONFIG.to_backend(tmp_path),training_n_lasers=(3,),targets_per_batch=5)
    config = engine.calibrate_budget_references(policy,config)
    optimizer = torch.optim.Adam(policy.parameters(),lr=config.learning_rate)
    rng = np.random.default_rng(9)
    options = dict(reward_scale_state={},mask_sampler=sample_band_random_topologies,
                   schedules=run.CONFIG.schedules)
    training_batch.run_training_batch(policy,optimizer,config,rng,998,**options)
    before = deepcopy((policy.state_dict(),optimizer.state_dict(),torch.get_rng_state()))
    result = training_batch.run_training_batch(policy,optimizer,config,rng,999,**options)
    assert result['mean_action_only'] is True
    assert result['pcgrad_task_count'] == 3
    assert_equal(before,(policy.state_dict(),optimizer.state_dict(),torch.get_rng_state()))
    path=tmp_path/'final.pt'
    state=exploration_state(policy)
    save(path,policy,optimizer,policy.config,1000,.8,exploration_state=state)
    loaded=CouplingModel.load(path,device='cpu',output_directory=tmp_path)
    assert loaded.backend_config.exploration_std_scale == 0.
    assert loaded.metadata['exploration_state'] == state
    assert type(pickle.loads(pickle.dumps(loaded.backend_config))) is type(loaded.backend_config)
    topology=Topology.from_receiver_source(np.ones((3,3))-np.eye(3))
    options=dict(target_phases_rad=[0.,-.7,1.3],detunings_ghz=[-1.,0.,1.],topology=topology,count=3)
    mean=loaded.propose(**options,continuous_actions='mean',seed=7)
    sample=loaded.propose(**options,continuous_actions='sample',seed=91)
    np.testing.assert_array_equal(mean.coupling_strengths_per_ns,sample.coupling_strengths_per_ns)
    np.testing.assert_array_equal(mean.coupling_phases_rad,sample.coupling_phases_rad)


def test_full_pipeline_35_jobs_21_tasks_and_mean_only_final_checkpoint(tmp_path,monkeypatch,fake_physics):
    schedules=replace(run.CONFIG.schedules,ordinary_end=3,wide_end=3,
                      learning_rate_decay_start=2,learning_rate_decay_end=3)
    config=replace(run.CONFIG,schedules=schedules,
        optimization=replace(run.CONFIG.optimization,iterations=3),
        validation=ValidationConfig(pairs_per_size=4))
    from rl.rl_api.simulation import simulate_indexed_chunk
    batches=[]
    class Pool:
        def __init__(self,processes,initializer): assert processes == 14
        def imap_unordered(self,worker,jobs):
            jobs=list(jobs)
            if worker is simulate_indexed_chunk:
                assert len(jobs) == 35
                assert sum(j[3][4].shape[0]*j[3][4].shape[1] for j in jobs) == 1400
                assert all(j[3][4].shape[:2] == (5,8) for j in jobs)
                batches.append(jobs)
            return reversed([worker(j) for j in jobs])
        def close(self): pass
        def join(self): pass
    class Context:
        def Pool(self,**options): return Pool(**options)
    monkeypatch.setattr(training.mp,'get_context',lambda method:Context())
    combine=core.combine_task_gradients_pcgrad
    def check_weights(gradients,clip,rng,**options):
        np.testing.assert_allclose(options['task_weights'],np.tile(np.array([.4,.2,.4])/7,7))
        return combine(gradients,clip,rng,**options)
    monkeypatch.setattr(core,'combine_task_gradients_pcgrad',check_weights)
    result=Experiment('test-mean-anneal',config=config,output_directory=tmp_path/'run').train(live_plot=False)
    assert len(batches) == 3
    assert result.history['pcgrad_task_count'] == [21]*3
    assert result.history['rooted_fraction'] == [1.]*3
    data=torch.load(result.checkpoint('final'),weights_only=False)
    assert data['iteration'] == 3
    assert data['config']['exploration_std_scale'] == 0.
    assert data['config']['phase_parameterization'] == 'target_residual'
    assert data['config']['phase_model'] == 'standard_lk'
    assert data['exploration_state']['extra_wide_phase_sd_deg'] == 0.
    assert data['sampling_recipe']['sampler'] == 'band_random'
    assert data['sampling_recipe']['precision_schedule']['anneal_to_mean'] is True
    assert all(int(v['step']) == 2 for v in data['optimizer_state_dict']['state'].values())
    history=json.loads((result.directory/'current_history.json').read_text())
    assert history['exploration_state']['action_std_scale'] == 0.


def test_schedule_rejects_mismatched_zero_endpoint():
    with pytest.raises(ValueError,match='final iteration'):
        replace(run.CONFIG,optimization=replace(run.CONFIG.optimization,iterations=999))
    with pytest.raises(ValueError,match='one shared'):
        replace(run.CONFIG.schedules,wide_end=900)
