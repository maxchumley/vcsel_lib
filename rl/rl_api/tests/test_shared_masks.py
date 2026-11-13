"""Shared matrices, independent actions, unchanged kernels and isolated outputs."""
from dataclasses import asdict, replace
import json
import sys
import numpy as np
import pytest
import torch

from rl.rl_api import Experiment, TrainingConfig, SamplingConfig, OptimizationConfig, ValidationConfig
from rl.rl_api._compat import core, engine
from rl.rl_api.models import build_policy
from rl.rl_api.sampling import sample_shared_rooted_topologies
from rl.rl_api import training_batch
from rl.rl_api.tests.test_parity import assert_equal


@pytest.mark.parametrize('size',range(3,10))
def test_four_identical_matrices_reach_all_five_pairs(size):
    masks=sample_shared_rooted_topologies(5,4,size,np.random.default_rng(8),torch.device('cpu'))
    assert masks.shape==(5,4,size*(size-1))
    for pair in range(1,5):
        torch.testing.assert_close(masks[pair],masks[0],rtol=0,atol=0)
    _,sources=core.directed_link_indices(size)
    np.testing.assert_array_equal(masks[0,0].numpy(),sources==0)
    counts=masks[0].sum(-1).numpy()
    np.testing.assert_array_equal(engine.topology_density_band_ids(counts[1:],size),[0,1,2])
    assert len(np.unique(masks[0].numpy(),axis=0))==4
    assert np.all(engine.root_reach_fractions(masks.numpy(),size)==1.)
    assert int((masks.sum(-1)==size-1).sum())==5
    assert 5*4*10==200  # Per M; seven sizes give 1,400 cases.


def test_masks_resample_between_iterations():
    rng=np.random.default_rng(7)
    first=sample_shared_rooted_topologies(5,4,6,rng,'cpu')
    second=sample_shared_rooted_topologies(5,4,6,rng,'cpu')
    assert not torch.equal(first[:,1:],second[:,1:])
    torch.testing.assert_close(first[:,0],second[:,0],rtol=0,atol=0)


def test_shared_masks_do_not_share_action_noise(tmp_path,monkeypatch):
    config=Experiment('noise',sizes=(3,),output_directory=tmp_path).backend_config
    torch.manual_seed(11)
    policy=build_policy(config)
    policy.minimum_log_std=core.minimum_log_std_at(0,config)
    target=np.array([[0.,-.4,.8]])
    delta=np.array([[-.5,0.,.5]])
    monkeypatch.setattr(core,'sample_target_phases',lambda count,*args:np.repeat(target,count,axis=0))
    monkeypatch.setattr(core,'sample_detuning_distributions',lambda count,*args,**kwargs:np.repeat(delta,count,axis=0))
    item=training_batch.prepare_size(policy,config,np.random.default_rng(9),3,5,0,
                                    sample_shared_rooted_topologies)
    torch.testing.assert_close(item['gates'][0],item['gates'][1],rtol=0,atol=0)
    assert not np.array_equal(item['phi'][0],item['phi'][1])
    assert not np.array_equal(item['kappa'][0],item['kappa'][1])


def test_active_batch_matches_reference_when_given_original_sampler(tmp_path,fake_physics):
    config=Experiment('parity',sizes=(3,5,9),output_directory=tmp_path).backend_config
    def run(reference):
        rng=core.set_random_seed(7)
        policy=build_policy(config)
        local=engine.calibrate_budget_references(policy,config)
        policy.config=local
        optimizer=torch.optim.Adam(policy.parameters(),lr=local.learning_rate)
        scale={}
        metrics=[]
        for iteration in (0,999,1499):
            if reference:
                result=engine.run_training_batch(policy,optimizer,local,rng,iteration,reward_scale_state=scale)
            else:
                result=training_batch.run_training_batch(policy,optimizer,local,rng,iteration,
                    reward_scale_state=scale,mask_sampler=engine.sample_paired_rooted_topologies)
            metrics.append(result)
        return policy.state_dict(),optimizer.state_dict(),scale,rng.bit_generator.state,torch.get_rng_state(),metrics
    expected=run(True)
    actual=run(False)
    for first,second in zip(expected[:5],actual[:5]):assert_equal(first,second)
    for old,new in zip(expected[5],actual[5]):
        for key,value in new.items():
            if key=='mean_active_links':continue
            if key=='size_metrics':
                for size,row in value.items():assert_equal(row,{k:old[key][size][k] for k in row})
            else:assert_equal(old[key],value)


def test_full_size_shared_batch_has_28_jobs_and_28_five_pair_tasks(tmp_path,fake_physics,monkeypatch):
    config=Experiment('counts',output_directory=tmp_path).backend_config
    rng=core.set_random_seed(7)
    policy=build_policy(config)
    config=engine.calibrate_budget_references(policy,config);policy.config=config
    optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate)
    original=training_batch.simulation_jobs
    def check_jobs(prepared,config):
        for item in prepared:
            for pair in range(1,5):
                torch.testing.assert_close(item['gates'][pair],item['gates'][0],rtol=0,atol=0)
        work=original(prepared,config)
        assert len(work)==28
        assert all(job[3][4].shape[:2]==(5,10) for job in work)
        assert sum(job[3][4].shape[0]*job[3][4].shape[1] for job in work)==1400
        return work
    monkeypatch.setattr(training_batch,'simulation_jobs',check_jobs)
    # Model arbitrary worker completion order without launching processes.
    class Pool:
        def imap_unordered(self,worker,jobs):return reversed([worker(job) for job in jobs])
    metrics=training_batch.run_training_batch(policy,optimizer,config,rng,0,
        simulation_pool=Pool(),reward_scale_state={})
    assert metrics['pcgrad_task_count']==28
    assert metrics['pcgrad_task_context_counts']==[5]*28
    assert metrics['rooted_fraction']==1.


def test_shared_run_configuration_and_isolated_output_name(tmp_path, monkeypatch):
    from rl.rl_api.experiments import train_current,train_shared_masks
    previous=asdict(train_current.CONFIG)
    proposed=asdict(train_shared_masks.CONFIG)
    previous['sampling']['reuse_masks_across_pairs']=True
    assert proposed==previous
    assert train_shared_masks.RUN_NAME!=train_current.RUN_NAME
    # This older run may now be enabled or active in the user's workspace.
    # Test its recipe using a throwaway output rather than its live folder.
    monkeypatch.setattr(train_shared_masks, 'OUTPUT_DIRECTORY', tmp_path/'shared-preview')
    experiment=Experiment(train_shared_masks.RUN_NAME, vcsel=train_shared_masks.vcsel,
                          config=train_shared_masks.CONFIG,
                          output_directory=train_shared_masks.OUTPUT_DIRECTORY)
    assert experiment.config.summary()['sampled_masks_per_size']==4
    assert experiment.config.summary()['simulations_per_iteration']==1400
    assert not experiment.directory.exists()


def test_shared_run_records_its_sampler_in_saved_outputs(tmp_path,fake_physics):
    config=TrainingConfig(sizes=(3,),sampling=SamplingConfig(reuse_masks_across_pairs=True),
        optimization=OptimizationConfig(iterations=1,workers=1),
        validation=ValidationConfig(pairs_per_size=4))
    result=Experiment('shared',config=config,output_directory=tmp_path/'shared').train(live_plot=False)
    assert result.history['pcgrad_task_count']==[4]
    for kind in ('best','current','final'):
        saved=torch.load(result.checkpoint(kind),weights_only=False)
        assert saved['sampling_recipe']['mode']=='shared_across_pairs'
        assert saved['sampling_recipe']['pairs_per_size']==5
        policy,loaded,info=engine.load_checkpoint(result.checkpoint(kind))
        assert_equal(policy.state_dict(),saved['policy_state_dict'])
    history=json.loads((result.directory/'current_history.json').read_text())
    assert history['sampling_recipe']['mode']=='shared_across_pairs'
    manifest=json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['sampling']['reuse_masks_across_pairs'] is True


def test_next_run_cells_preview_and_wait_for_explicit_start(tmp_path,monkeypatch):
    from rl.rl_api.tests.test_cells import read_cells
    calls=[]
    token=object()
    def train(self,**options):
        calls.append(self)
        return token
    monkeypatch.setattr(Experiment,'train',train)
    monkeypatch.setattr(sys,'argv',['ipykernel_launcher.py','-f','connection.json'])
    scope={'__name__':'__main__','__package__':None}
    cells=read_cells('train_shared_masks.py')
    for cell in cells:
        # Exercise preview mode regardless of the user's editable start flag.
        preview_cell=cell.replace('START_TRAINING = True', 'START_TRAINING = False')
        exec(compile(preview_cell,'<shared-mask-cell>','exec'),scope)
        if 'CONFIG' in scope:scope['OUTPUT_DIRECTORY']=tmp_path/'cell-run'
    assert calls==[] and '__file__' not in scope
    assert scope['experiment'].config.sampling.reuse_masks_across_pairs is True
    assert not (tmp_path/'cell-run').exists()
    scope['START_TRAINING']=True
    start_cell=next(cell for cell in cells if 'result = experiment.train(' in cell)
    exec(compile(start_cell,'<train-cell>','exec'),scope)
    assert len(calls)==1 and scope['result'] is token
