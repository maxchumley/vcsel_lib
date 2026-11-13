"""The opt-in residual phase mean is the only numerical change in this experiment."""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, CouplingModel, ValidationConfig
from rl.rl_api._compat import core, engine, make_backend_config
from rl.rl_api.experiments import train_phase_residual as run, train_four_counts as baseline
from rl.rl_api.models import build_policy
from rl.rl_api.models.policy import TargetResidualPhasePolicy, target_phase_offsets
from rl.rl_api.exploration import refinement_config
from rl.rl_api import training, training_batch


def paired_policies(directory, size=3, conjugate=True):
    old_config = replace(baseline.CONFIG.to_backend(directory),n_lasers=size,
                         allow_global_phase_conjugate=conjugate)
    new_config = make_backend_config(dict(asdict(old_config),phase_parameterization='target_residual'))
    torch.manual_seed(7)
    old = build_policy(old_config)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    new = build_policy(new_config)
    assert torch.equal(old_rng,torch.get_rng_state())
    return old, new


def context(size):
    first = np.r_[0.,np.linspace(-2.8,2.8,size-1)]
    targets = np.stack((first,-first))
    detunings = np.tile(np.linspace(-2.,2.,size),(2,1))
    return targets, detunings


def test_configuration_diff_is_only_phase_parameterization(tmp_path):
    before, after = asdict(baseline.CONFIG), asdict(run.CONFIG)
    assert before['model'].pop('phase_parameterization') == 'absolute'
    assert after['model'].pop('phase_parameterization') == 'target_residual'
    assert before == after
    old_backend, new_backend = asdict(baseline.CONFIG.to_backend(tmp_path)), asdict(run.CONFIG.to_backend(tmp_path))
    assert new_backend.pop('phase_parameterization') == 'target_residual'
    assert old_backend == new_backend
    assert run.RUN_NAME != baseline.RUN_NAME
    for setting in ('START_TRAINING','RESTART_EXISTING_RUN','LIVE_PLOT','VERIFY_REFERENCE','OUTPUT_DIRECTORY'):
        assert getattr(run,setting) == getattr(baseline,setting)
    assert run.START_TRAINING is True
    old_summary, new_summary = baseline.CONFIG.summary(), run.CONFIG.summary()
    assert old_summary.pop('phase_parameterization') == 'absolute'
    assert new_summary.pop('phase_parameterization') == 'target_residual'
    assert old_summary == new_summary


def test_same_parameters_initialization_and_zero_target_behavior(tmp_path):
    old, new = paired_policies(tmp_path)
    assert type(old) is engine.BernoulliSparseGNNPolicy
    assert isinstance(new,TargetResidualPhasePolicy)
    assert old.state_dict().keys() == new.state_dict().keys()
    for key in old.state_dict():
        torch.testing.assert_close(old.state_dict()[key],new.state_dict()[key],rtol=0,atol=0)
    assert sum(p.numel() for p in new.parameters()) == 127304
    features = engine.encode_context(np.zeros((1,3)),np.array([[-1.,0.,1.]]),new)
    masks = torch.ones((1,6))
    for first,second in zip(old.parameters_for_edges(features,topology_masks=masks),
                            new.parameters_for_edges(features,topology_masks=masks)):
        torch.testing.assert_close(first,second,rtol=0,atol=0)


@pytest.mark.parametrize('size',[3,6,9])
@pytest.mark.parametrize('conjugate',[False,True])
def test_only_phase_mean_changes_with_correct_source_receiver_sign(size,conjugate,tmp_path):
    old, new = paired_policies(tmp_path,size,conjugate)
    targets, detuning = context(size)
    features = engine.encode_context(targets,detuning,new)
    masks = engine.sample_stratified_rooted_topologies(2,4,size,np.random.default_rng(11),'cpu',fixed_star=True)
    repeated = features.repeat_interleave(4,dim=0)
    masks = masks.reshape(8,-1)
    first = old.parameters_for_edges(repeated,topology_masks=masks,return_pooled=True)
    second = new.parameters_for_edges(repeated,topology_masks=masks,return_pooled=True)
    edges = size*(size-1)
    torch.testing.assert_close(first[0][:,:edges],second[0][:,:edges],rtol=0,atol=0)
    for a,b in zip(first[1:],second[1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    oriented = core.canonicalize_global_phase_conjugate(targets) if conjugate else core.make_relative_to_laser_1(targets)
    receivers,sources = core.directed_link_indices(size)
    expected = np.angle(np.exp(1j*(oriented[:,sources]-oriented[:,receivers])))
    expected = np.repeat(expected,4,axis=0)
    np.testing.assert_allclose(target_phase_offsets(repeated).numpy(),expected,rtol=0,atol=5e-7)
    actual = (second[0][:,edges:]-first[0][:,edges:]).detach().numpy()
    np.testing.assert_allclose(actual,expected,rtol=0,atol=5e-7)
    a = old.budget_parameters(repeated,pooled_states=first[3])
    b = new.budget_parameters(repeated,pooled_states=second[3])
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_zero_residual_means_exact_target_difference(tmp_path):
    _, policy = paired_policies(tmp_path,conjugate=False)
    with torch.no_grad():
        policy.edge_output.weight[1].zero_()
        policy.edge_output.bias[1].zero_()
    features = engine.encode_context([[0.,np.pi/2,-np.pi/4]],[[-1.,0.,1.]],policy)
    mean,_,_ = policy.parameters_for_edges(features,topology_masks=torch.ones((1,6)))
    receivers,sources = core.directed_link_indices(3)
    edge = int(np.flatnonzero((receivers==1)&(sources==0))[0])
    assert mean[0,6+edge].item() == pytest.approx(-np.pi/2,abs=1e-6)


@pytest.mark.parametrize('iteration',[0,1999])
def test_sampling_and_log_probability_use_corrected_mean(iteration,tmp_path):
    old, new = paired_policies(tmp_path)
    for policy in (old,new):
        policy.config = refinement_config(policy.config,run.CONFIG.schedules,iteration)
        policy.minimum_log_std = core.minimum_log_std_at(iteration,policy.config)
        policy.maximum_log_std = core.maximum_log_std_at(iteration,policy.config)
    targets,detuning = context(3)
    features = engine.encode_context(targets,detuning,new)
    masks = engine.sample_stratified_rooted_topologies(2,4,3,np.random.default_rng(8),'cpu',fixed_star=True)
    masks = masks.repeat_interleave(8,dim=1)
    torch.manual_seed(91)
    old_actions,old_budget,_ = engine.sample_topology_conditioned_candidates(
        old,features,masks,iteration=iteration,rng=np.random.default_rng(9))
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    actions,budget,log_prob = engine.sample_topology_conditioned_candidates(
        new,features,masks,iteration=iteration,rng=np.random.default_rng(9))
    assert torch.equal(old_rng,torch.get_rng_state())
    torch.testing.assert_close(actions[:,:,:6],old_actions[:,:,:6],rtol=0,atol=0)
    torch.testing.assert_close(budget,old_budget,rtol=0,atol=0)
    offsets = target_phase_offsets(features)[:,None,:]
    torch.testing.assert_close(actions[:,:,6:],old_actions[:,:,6:]+offsets,rtol=0,atol=1e-6)
    repeated = features[:,None].expand(-1,32,-1,-1).reshape(-1,3,4)
    outputs = old.parameters_for_edges(repeated,topology_masks=masks.reshape(-1,6),return_pooled=True)
    expected_mean = outputs[0].reshape(2,32,12)
    expected_mean = torch.cat((expected_mean[:,:,:6],expected_mean[:,:,6:]+offsets),dim=-1)
    std = engine._topology_action_std(outputs[1].reshape(2,32,12),new.config,iteration,False)
    normal = torch.distributions.Normal(expected_mean,std)
    expected = (normal.log_prob(actions)*torch.cat((masks,masks),dim=-1)).sum(-1)
    budget_mean,budget_log_std = old.budget_parameters(repeated,pooled_states=outputs[3])
    budget_normal = torch.distributions.Normal(budget_mean.reshape(2,32),budget_log_std.reshape(2,32).exp())
    expected = expected+budget_normal.log_prob(budget)
    torch.testing.assert_close(log_prob,expected,rtol=0,atol=0)
    weights = torch.linspace(-1.,1.,log_prob.numel()).reshape_as(log_prob)
    (weights*log_prob).mean().backward()
    assert torch.isfinite(new.edge_output.bias.grad).all()
    assert abs(new.edge_output.bias.grad[1].item())>1e-6


def test_first_batch_keeps_strengths_budgets_contexts_masks_and_rng(tmp_path):
    batches = []
    for public in (baseline.CONFIG,run.CONFIG):
        config = public.to_backend(tmp_path)
        rng = core.set_random_seed(7)
        policy = build_policy(config)
        config = engine.calibrate_budget_references(policy,config)
        policy.config = config
        policy.minimum_log_std = core.minimum_log_std_at(0,config)
        policy.maximum_log_std = core.maximum_log_std_at(0,config)
        prepared = [training_batch.prepare_size(policy,config,rng,m,5,0,
            engine.sample_stratified_rooted_topologies) for m in config.training_n_lasers]
        batches.append((prepared,rng.bit_generator.state,torch.get_rng_state(),config.coupling_budget_reference_by_m))
    assert batches[0][1] == batches[1][1]
    assert torch.equal(batches[0][2],batches[1][2])
    assert batches[0][3] == batches[1][3]
    for old,new in zip(batches[0][0],batches[1][0]):
        for key in ('targets','detunings','kappa','counts','rho','shortfall'):
            np.testing.assert_array_equal(old[key],new[key])
        torch.testing.assert_close(old['gates'],new['gates'],rtol=0,atol=0)
        offsets = target_phase_offsets(new['features']).detach().numpy()[:,None,:]
        receivers,sources = core.directed_link_indices(new['n_lasers'])
        expected = np.angle(np.exp(1j*(old['phi'][:,:,receivers,sources]+offsets)))
        delta = np.angle(np.exp(1j*(new['phi'][:,:,receivers,sources]-expected)))
        np.testing.assert_allclose(delta,0.,rtol=0,atol=2e-6)


def test_training_validation_checkpoint_and_inference_keep_residual_mode(tmp_path,monkeypatch,fake_physics):
    from rl.rl_api.simulation import simulate_indexed_chunk
    settings = replace(run.CONFIG,
        optimization=replace(run.CONFIG.optimization,iterations=1),
        validation=ValidationConfig(pairs_per_size=4))
    jobs_seen = []
    class Pool:
        def __init__(self,processes,initializer):assert processes==14
        def imap_unordered(self,worker,jobs):
            jobs=list(jobs)
            if worker is simulate_indexed_chunk:jobs_seen.extend(jobs)
            assert all(getattr(job[3][-1] if worker is simulate_indexed_chunk else job[-1],
                               'phase_parameterization') == 'target_residual' for job in jobs)
            return reversed([worker(job) for job in jobs])
        def close(self):pass
        def join(self):pass
    class Context:
        def Pool(self,**options):return Pool(**options)
    monkeypatch.setattr(training.mp,'get_context',lambda method:Context())
    result = Experiment('residual-test',config=settings,output_directory=tmp_path/'run').train(live_plot=False)
    assert len(jobs_seen)==28
    assert sum(job[3][4].shape[0]*job[3][4].shape[1] for job in jobs_seen)==1120
    assert result.history['pcgrad_task_count']==[16]
    assert isinstance(result.model.policy,TargetResidualPhasePolicy)
    saved = torch.load(result.checkpoint('final'),weights_only=False)
    assert saved['config']['phase_parameterization']=='target_residual'
    loaded = CouplingModel.load(result.checkpoint('final'),device='cpu')
    assert isinstance(loaded.policy,TargetResidualPhasePolicy)
    assert loaded.backend_config.phase_model=='standard_lk'
    assert loaded.policy.minimum_log_std==result.model.policy.minimum_log_std
    manifest = json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['model']['phase_parameterization']=='target_residual'
    for actions in ('mean','sample'):
        options=dict(target_phases_rad=[0.,-1.,2.],detunings_ghz=[-.5,0.,.5],count=2,
                     continuous_actions=actions,seed=23)
        first,second=result.model.propose(**options),loaded.propose(**options)
        np.testing.assert_array_equal(first.coupling_phases_rad,second.coupling_phases_rad)
        np.testing.assert_array_equal(first.coupling_strengths_per_ns,second.coupling_strengths_per_ns)
        assert np.isfinite(second.evaluate().rms_reward).all()


def test_run_starts_from_one_cell_and_worker_import_is_safe(tmp_path,monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls = []
    monkeypatch.setattr(Experiment,'train',lambda self,**options:calls.append((self,options)))
    cells = read_cells('train_phase_residual.py')
    assert len(cells)==1
    cell=cells[0].replace('OUTPUT_DIRECTORY = None',f'OUTPUT_DIRECTORY = {str(tmp_path/"cell")!r}')
    exec(compile(cell,'<residual-cell>','exec'),{'__name__':'__main__'})
    assert len(calls)==1
    assert calls[0][0].config.model.phase_parameterization=='target_residual'
    assert calls[0][1]['restart'] is True
    exec(compile(cell,'<worker-import>','exec'),{'__name__':'__mp_main__'})
    assert len(calls)==1
    assert not (tmp_path/'cell').exists()
