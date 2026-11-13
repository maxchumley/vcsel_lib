"""The sampled root reaches the encoder consistently through the public workflow."""
from dataclasses import asdict, replace
import json
import pickle

import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel, Experiment, ModelConfig, Topology, ValidationConfig
from rl.rl_api._compat import core, engine
from rl.rl_api import evaluation, training
from rl.rl_api.experiments import train_mirrored_root_delay_1000 as previous
from rl.rl_api.experiments import train_mirrored_root_encoder_1000 as run
from rl.rl_api.models import build_policy
from rl.rl_api.root_context import condition_on_roots, root_node_features, calibrate_budget_references
from rl.rl_api.sampling import sample_band_random_reachable_topologies, sample_random_reachable_topologies
from rl.rl_api.training_batch import prepare_size


def policy_for(tmp_path, public=run.CONFIG, size=5):
    cfg=replace(public.to_backend(tmp_path),n_lasers=size)
    torch.manual_seed(7)
    return build_policy(cfg)


def test_only_encoder_flag_changes_and_shared_initial_weights_and_rng_match(tmp_path):
    old,new=asdict(previous.CONFIG),asdict(run.CONFIG)
    assert old['model'].pop('encode_root') is False
    assert new['model'].pop('encode_root') is True
    assert old==new
    old,new=asdict(previous.CONFIG.to_backend(tmp_path)),asdict(run.CONFIG.to_backend(tmp_path))
    assert 'encode_root' not in old
    assert new.pop('encode_root') is True
    assert old==new
    states=[]
    for public in (previous.CONFIG,run.CONFIG):
        policy=policy_for(tmp_path,public)
        states.append((policy.state_dict(),torch.get_rng_state().clone()))
    for key,value in states[0][0].items():
        other=states[1][0][key]
        if key=='node_encoder.0.weight':
            torch.testing.assert_close(value,other[:,:4],rtol=0,atol=0)
            assert torch.count_nonzero(other[:,4])==0
        else:
            torch.testing.assert_close(value,other,rtol=0,atol=0)
    assert torch.equal(states[0][1],states[1][1])
    assert sum(p.numel() for p in policy.parameters())==127368
    for key in ('START_TRAINING','RESTART_EXISTING_RUN','LIVE_PLOT','VERIFY_REFERENCE','OUTPUT_DIRECTORY'):
        assert getattr(run,key)==getattr(previous,key)
    assert run.RUN_NAME!=previous.RUN_NAME
    assert run.CONFIG.summary()['node_input_features']==5
    assert run.CONFIG.summary()['simulations_per_iteration']==1400
    with pytest.raises(ValueError,match='requires target_root_delay_residual'):
        ModelConfig(encode_root=True)


def test_root_feature_is_binary_one_per_node_and_preserves_all_other_features():
    features=torch.randn((4,5,4))
    roots=np.array([0,4,1,3])
    extended=root_node_features(features,roots)
    torch.testing.assert_close(extended[:,:,:4],features,rtol=0,atol=0)
    torch.testing.assert_close(extended[:,:,4].sum(dim=1),torch.ones(4),rtol=0,atol=0)
    np.testing.assert_array_equal(extended[:,:,4].argmax(dim=1).numpy(),roots)
    with pytest.raises(ValueError,match='valid integer root'):
        root_node_features(features,[0.,1.,2.,3.])
    with pytest.raises(ValueError,match='valid integer root'):
        root_node_features(features,[0,1,2,5])


def test_training_reuses_exact_contexts_masks_and_roots_and_root_column_has_gradient(tmp_path):
    prepared=[];captured=[]
    for public in (previous.CONFIG,run.CONFIG):
        policy=policy_for(tmp_path,public)
        if public.model.encode_root:
            policy.node_encoder.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        torch.manual_seed(117)
        item=prepare_size(policy,policy.config,np.random.default_rng(571),5,5,0,
                          sample_band_random_reachable_topologies)
        prepared.append((policy,item))
    old,new=prepared[0][1],prepared[1][1]
    for key in ('targets','detunings','roots','counts'):
        np.testing.assert_array_equal(old[key],new[key])
    torch.testing.assert_close(old['gates'],new['gates'],rtol=0,atol=0)
    expected=new['roots'].ravel()
    assert len(captured)==1 and captured[0].shape==(200,5,5)
    np.testing.assert_array_equal(captured[0][:,:,4].argmax(dim=1).numpy(),expected)
    assert np.all(new['roots'].reshape(5,5,8)==new['roots'].reshape(5,5,8)[:,:,:1])
    # The feature starts at zero weight, preserving the original initial policy.
    np.testing.assert_allclose(old['kappa'],new['kappa'],rtol=2e-6,atol=1e-7)
    np.testing.assert_allclose(old['phi'],new['phi'],rtol=0,atol=3e-6)
    policy,item=prepared[1]
    weights=torch.linspace(-1.,1.,item['log_prob'].numel()).reshape_as(item['log_prob'])
    loss=(item['log_prob']*weights).mean()
    loss.backward()
    grad=policy.node_encoder[0].weight.grad[:,4]
    assert torch.isfinite(grad).all() and grad.abs().sum()>1e-10
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4);optimizer.step()
    assert policy.node_encoder[0].weight[:,4].abs().sum()>0.
    with pytest.raises(ValueError,match='explicit roots'):
        condition_on_roots(policy,None)


def test_identical_graph_with_different_roots_can_now_produce_different_predictions(tmp_path):
    policy=policy_for(tmp_path,size=3)
    with torch.no_grad():
        policy.node_encoder[0].weight[:,4].copy_(torch.linspace(-.5,.5,64))
    features=engine.encode_context(np.array([[0.,.6,-1.3]]),np.array([[-1.1,.2,.9]]),policy)
    masks=torch.ones((1,3,6));roots=np.array([[0,1,2]])
    with torch.inference_mode():
        action,budget,_=engine.sample_topology_conditioned_candidates(
            condition_on_roots(policy,roots),features,masks,continuous_mean=True)
    assert torch.isfinite(action).all() and torch.isfinite(budget).all()
    assert not torch.allclose(action[:,0],action[:,1],atol=1e-8,rtol=0)


def test_validation_passes_the_same_sampled_roots_into_the_encoder(tmp_path,monkeypatch):
    policy=policy_for(tmp_path,size=5)
    policy.config=calibrate_budget_references(policy,policy.config)
    captured=[]
    policy.node_encoder.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    rng=np.random.default_rng(9)
    target,detuning=core.sort_by_detuning(rng.uniform(-3.,3.,(8,5)),rng.uniform(-2.,2.,(8,5)))
    def simulate(target,kappa,phi,config,**options):
        z=np.zeros(kappa.shape[:2])
        return dict(phase_reward=z,phase_rms_reward=z,mean_phase_rms_error_deg=z,
                    worst_phase_rms_error_deg=z,phase_rms_success=z)
    monkeypatch.setattr(evaluation,'simulate_phase_model_candidates',simulate)
    evaluation.evaluate_phase_model_policy(policy,target,detuning,policy.config,seed=127)
    _,roots=sample_random_reachable_topologies(1,8,5,np.random.default_rng(127),'cpu',return_roots=True)
    assert len(captured)==1
    np.testing.assert_array_equal(captured[0][:,:,4].argmax(dim=1).numpy(),roots.ravel())


@pytest.mark.parametrize('actions',['mean','sample'])
def test_automatic_and_manual_inference_use_root_flags_in_sorted_laser_order(tmp_path,actions):
    policy=policy_for(tmp_path,size=4)
    cfg=calibrate_budget_references(policy,policy.config);policy.config=cfg
    model=CouplingModel(policy,cfg,output_directory=tmp_path)
    captured=[]
    policy.node_encoder.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    context=Context([0.,.7,-1.2,.3],[2.,-1.,1.,.5])
    result=model.propose(context,count=12,seed=17,continuous_actions=actions)
    np.testing.assert_array_equal(captured[-1][:,:,4].argmax(dim=1).numpy(),result.raw['root_indices'])
    order=np.argsort(context.detunings_ghz)
    second=model.propose(Context(context.target_phases_rad[order],context.detunings_ghz[order]),
                         count=12,seed=17,continuous_actions=actions)
    np.testing.assert_array_equal(result.coupling_strengths_per_ns,second.coupling_strengths_per_ns)
    np.testing.assert_array_equal(result.coupling_phases_rad,second.coupling_phases_rad)
    # Original laser 0 is the only graph root, and becomes sorted laser 3.
    manual=np.zeros((4,4),dtype=bool);manual[0,1:]=True
    supplied=model.propose(context,topology=Topology.from_source_receiver(manual),count=2,seed=17)
    np.testing.assert_array_equal(supplied.raw['root_indices'],[3,3])
    np.testing.assert_array_equal(captured[-1][:,:,4].argmax(dim=1).numpy(),[3,3])
    np.testing.assert_array_equal(supplied.topologies[0],manual.T[order][:,order])


def test_checkpoint_reload_training_and_loaded_inspection_preserve_root_encoder(tmp_path,monkeypatch,fake_physics):
    schedules=replace(run.CONFIG.schedules,ordinary_end=3,wide_end=3,
                      learning_rate_decay_start=2,learning_rate_decay_end=3)
    public=replace(run.CONFIG,schedules=schedules,
        optimization=replace(run.CONFIG.optimization,iterations=3),
        validation=ValidationConfig(pairs_per_size=4))
    from rl.rl_api.simulation import simulate_indexed_chunk
    batches=[]
    class Pool:
        def __init__(self,processes,initializer): assert processes==14
        def imap_unordered(self,worker,jobs):
            jobs=list(jobs)
            if worker is simulate_indexed_chunk:
                assert len(jobs)==35
                assert all(job[3][4].shape[:2]==(5,8) for job in jobs)
                batches.append(jobs)
            return reversed([worker(job) for job in jobs])
        def close(self):pass
        def join(self):pass
    class ProcessContext:
        def Pool(self,**kwargs):return Pool(**kwargs)
    monkeypatch.setattr(training.mp,'get_context',lambda method:ProcessContext())
    result=Experiment('root-encoder-test',config=public,output_directory=tmp_path/'run').train(
        live_plot=False,verify_reference=False)
    assert len(batches)==3
    saved=torch.load(result.checkpoint('final'),weights_only=False)
    assert saved['config']['encode_root'] is True
    assert saved['sampling_recipe']['encoder_root_feature']=='1 at sampled root, 0 elsewhere'
    assert saved['config']['phase_parameterization']=='target_root_delay_residual'
    assert all(int(s['step'])==2 for s in saved['optimizer_state_dict']['state'].values())
    assert torch.count_nonzero(saved['policy_state_dict']['node_encoder.0.weight'][:,4])>0
    loaded=CouplingModel.load(result.checkpoint('final'),device='cpu')
    assert pickle.loads(pickle.dumps(loaded.backend_config)).encode_root
    assert loaded.policy.node_encoder[0].in_features==5
    context=Context([0.,-.4,1.2,.8],[1.,-2.,.5,-.5])
    a=result.model.propose(context,count=12,seed=17)
    b=loaded.propose(context,count=12,seed=17)
    np.testing.assert_array_equal(a.coupling_strengths_per_ns,b.coupling_strengths_per_ns)
    np.testing.assert_array_equal(a.coupling_phases_rad,b.coupling_phases_rad)
    np.testing.assert_array_equal(a.raw['root_indices'],b.raw['root_indices'])
    assert np.isfinite(b.evaluate().rms_reward).all()
    manifest=json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['model']['encode_root'] is True


def test_single_cell_previews_on_import_and_starts_only_as_main(tmp_path,monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls=[]
    monkeypatch.setattr(Experiment,'train',lambda self,**options:calls.append((self,options)))
    cells=read_cells('train_mirrored_root_encoder_1000.py');assert len(cells)==1
    cell=cells[0].replace('OUTPUT_DIRECTORY = None',f'OUTPUT_DIRECTORY = {str(tmp_path/"cell")!r}')
    exec(compile(cell,'<root-encoder-cell>','exec'),{'__name__':'__main__'})
    assert len(calls)==1 and calls[0][0].name==run.RUN_NAME
    assert calls[0][0].config.model.encode_root
    exec(compile(cell,'<root-encoder-worker>','exec'),{'__name__':'__mp_main__'})
    assert len(calls)==1 and not (tmp_path/'cell').exists()
