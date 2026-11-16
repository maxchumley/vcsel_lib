"""Complete graph coverage, physical reciprocity, and independent worker use."""
from collections import Counter
from dataclasses import replace
import itertools
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel, Experiment
from rl.rl_api._compat import core
from rl.rl_api.experiments import train_four_laser_inphase_all_symmetric_1000 as run
from rl.rl_api.models import build_policy
from rl.rl_api.phase_reference import decode_samples
from rl.rl_api.symmetric_topologies import connected_symmetric_masks
from rl.rl_api.training_batch import prepare_size, simulation_jobs
from rl.rl_api.symmetric_topologies import sample_all_symmetric_topologies


def test_complete_bank_matches_independent_graph_enumeration():
    bank=connected_symmetric_masks()
    assert bank.shape==(38,12) and len({row.tobytes() for row in bank})==38
    assert Counter(bank.sum(1)//2)=={3:16,4:15,5:6,6:1}
    receiver,source=core.directed_link_indices(4)
    matrices=np.zeros((38,4,4),bool)
    matrices[:,receiver,source]=bank
    np.testing.assert_array_equal(matrices,matrices.swapaxes(-1,-2))
    expected=set()
    pairs=list(itertools.combinations(range(4),2))
    for values in itertools.product((False,True),repeat=6):
        graph=np.zeros((4,4),bool)
        for (a,b),active in zip(pairs,values):
            graph[a,b]=graph[b,a]=active
        # Matrix powers count paths of length 0..3; every connected 4-node
        # graph has a path from node 0 to every other node within three edges.
        paths=np.linalg.matrix_power(graph.astype(int)+np.eye(4,dtype=int),3)
        if np.all(paths[0]>0):
            expected.add(graph[receiver,source].tobytes())
    assert {row.tobytes() for row in bank}==expected


def test_prepared_batch_covers_every_graph_and_keeps_physical_symmetry(tmp_path):
    torch.manual_seed(7)
    config=run.CONFIG.to_backend(tmp_path)
    policy=build_policy(config)
    item=prepare_size(policy,config,np.random.default_rng(7),4,5,0,sample_all_symmetric_topologies)
    assert np.all(item['targets']==0.)
    spans=np.ptp(item['detunings'],axis=1)
    assert np.all(spans<=4.) and len(np.unique((spans/.8).astype(int)))==5
    expected=torch.tensor(connected_symmetric_masks(),dtype=torch.float32)
    for masks in item['gates'][:,::8]:
        torch.testing.assert_close(masks,expected)
    for matrix in (item['kappa'],item['phi']):
        np.testing.assert_array_equal(matrix,matrix.swapaxes(-1,-2))
    assert np.all(item['reach']==1.)
    jobs=simulation_jobs([item],config)
    assert len(jobs)==31
    assert run.CONFIG.summary()['simulations_per_iteration']==1520
    assert run.CONFIG.summary()['validation_simulations_per_check']==380
    # The original full directed samples, not their averages, receive the score.
    gradient=torch.autograd.grad(item['log_prob'][0,0],policy.edge_output.bias)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


def test_projection_preserves_budget_wraps_phases_and_rejects_directed_masks(tmp_path):
    config=run.CONFIG.to_backend(tmp_path)
    actions=torch.linspace(-4.,5.,48).reshape(1,2,24)
    masks=torch.ones((1,2,12))
    budget=torch.tensor([[1.2,2.4]])
    kappa,phase,_,_,shortfall=decode_samples(actions,masks,config,budget_actions=budget)
    np.testing.assert_array_equal(kappa,kappa.swapaxes(-1,-2))
    np.testing.assert_array_equal(phase,phase.swapaxes(-1,-2))
    np.testing.assert_allclose(kappa.sum((-2,-1)),budget.exp().numpy(),rtol=2e-6)
    assert not shortfall.any() and np.all(np.abs(phase)<=np.pi)
    masks[0,0,0]=0.
    with pytest.raises(ValueError,match='symmetric physical topology'):
        decode_samples(actions,masks,config,budget_actions=budget)


def test_exhaustive_training_validation_checkpoint_and_worker_cap(tmp_path,fake_physics,monkeypatch):
    from rl.rl_api import training
    pools=[]
    class Pool:
        def __init__(self,processes,initializer):
            pools.append(processes)
        def imap_unordered(self,function,work):
            return map(function,work)
        def close(self):
            pass
        def join(self):
            pass
    monkeypatch.setattr(training.mp,'get_context',lambda method:SimpleNamespace(Pool=Pool))
    public=replace(run.CONFIG,
        sampling=replace(run.CONFIG.sampling,pairs_per_size=1),
        optimization=replace(run.CONFIG.optimization,iterations=2,workers=6),
        schedules=replace(run.CONFIG.schedules,ordinary_end=2,wide_end=2),
        validation=replace(run.CONFIG.validation,pairs_per_size=2,plot_use_tex=False))
    result=Experiment('all-symmetric-test',config=public,output_directory=tmp_path/'run').train(
        live_plot=False,verify_reference=False)
    assert pools==[6]  # One context must not cap the six configured workers.
    assert len(result.history['training_reward'])==2
    for key in ('simulation_seconds','update_seconds','validation_seconds'):
        assert len(result.history[key])==2 and np.isfinite(result.history[key]).all()
    assert result.history['pcgrad_task_count']==[3,3]
    # Training sends 38*8 cases/iteration; validation sends 2*38 each check.
    assert sum(len(k) for k,_,_ in fake_physics)==2*(38*8+2*38)
    for kappa,phase,_ in fake_physics:
        np.testing.assert_array_equal(kappa,kappa.swapaxes(-1,-2))
        np.testing.assert_array_equal(phase,phase.swapaxes(-1,-2))
    metadata=json.loads((result.directory/'symmetric_topology_bank.json').read_text())
    assert metadata['count']==38
    loaded=CouplingModel.load(result.checkpoint('final'),device='cpu')
    assert loaded.backend_config.exploration_std_scale==0.
    assert loaded.backend_config.symmetric_coupling_values
    designs=loaded.propose(Context(np.zeros(4),[.7,-.9,.3,1.]),count=38,seed=71)
    assert len({mask.tobytes() for mask in designs.topologies})==38
    np.testing.assert_array_equal(designs.coupling_strengths_per_ns,designs.coupling_strengths_per_ns.swapaxes(-1,-2))
    np.testing.assert_array_equal(designs.coupling_phases_rad,designs.coupling_phases_rad.swapaxes(-1,-2))
    # A source-labeled manual topology must follow the same detuning sort.
    graph=np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
    manual=loaded.propose(Context(np.zeros(4),[.7,-.9,.3,1.]),topology=graph,count=1)
    order=np.argsort([.7,-.9,.3,1.])
    np.testing.assert_array_equal(manual.topologies[0],graph[order][:,order])
    with pytest.raises(ValueError,match='in-phase'):
        loaded.propose(Context([0.,.2,0.,0.],[-1.,0.,.5,1.]),count=1)


def test_cluster_batch_overrides_change_only_simulation_chunking(tmp_path):
    from rl.rl_api.experiments.train_batch import main
    experiment=main(['train_four_laser_inphase_all_symmetric_1000','--preview',
        '--output-directory',str(tmp_path/'preview'),'--workers','6',
        '--pairs-per-job','5','--actions-per-job','16'])
    assert experiment.config.optimization.workers==6
    assert experiment.config.sampling.pairs_per_size==5
    assert experiment.config.sampling.topologies_per_pair==38
    assert experiment.config.sampling.actions_per_topology==8
    assert experiment.config.summary()['simulations_per_iteration']==1520
    assert experiment.config.summary()['worker_jobs']==19
    assert not experiment.directory.exists()
