from dataclasses import replace,asdict
import numpy as np
import pytest
import torch
from rl.rl_api import Experiment,CouplingModel,Context
from rl.rl_api._compat import engine
from rl.rl_api.models import build_policy
from rl.rl_api.conditional_phases import sample_conditional_density
from rl.rl_api.density_control import DensityConditionedPolicy
from rl.rl_api.reciprocal_pairs import pair_layout
from rl.rl_api.experiments import train_density_actor_critic_replay_asymphase_1000 as run
from rl.rl_api.experiments import train_density_actor_critic_replay_1000 as source


def test_only_phase_tying_and_output_names_change():
    a,b=asdict(run.CONFIG),asdict(source.CONFIG)
    assert a['model'].pop('asymmetric_coupling_phases') and not b['model'].pop('asymmetric_coupling_phases')
    a['validation'].pop('progress_filename');b['validation'].pop('progress_filename')
    assert a==b and run.INITIAL_CHECKPOINT is None
    assert run.CONFIG.summary()['symmetric_strengths'] and not run.CONFIG.summary()['symmetric_phases']


@pytest.mark.parametrize('size',range(3,10))
def test_mixed_scores_symmetry_gradients_and_mean_path(tmp_path,size):
    config=replace(run.CONFIG.to_backend(tmp_path),n_lasers=size,exploration_std_scale=.5)
    torch.manual_seed(77);policy=build_policy(config)
    features=engine.encode_context(np.linspace(0,2,size)[None],np.linspace(-1,1,size)[None],policy)
    fractions=np.repeat(np.array([[0.,.1,.5,.8,1.]]),8,axis=1)
    edges=size*(size-1);canonical,expansion,reverse=pair_layout(size)
    for mean_only in (False,True):
        actions,budget,score,mask,physical=sample_conditional_density(policy,features,config,fractions,
            mirrored=True,continuous_mean=mean_only)
        torch.testing.assert_close(actions[...,:edges],actions[...,:edges][...,reverse],atol=0,rtol=0)
        np.testing.assert_array_equal(physical[0],physical[0].swapaxes(-1,-2))
        assert np.max(np.abs(np.angle(np.exp(1j*(physical[1]-physical[1].swapaxes(-1,-2))))))>1e-3
        if not mean_only:
            repeated=features[:,None].expand(-1,40,-1,-1).reshape(40,size,4)
            density=DensityConditionedPolicy(policy,fractions)
            mu,ls,_,pooled=density.parameters_for_edges(repeated,topology_masks=features.new_ones((40,edges)),return_pooled=True)
            mu,sd=mu.reshape(1,40,2*edges),ls.exp().reshape(1,40,2*edges)
            # Reconstruct the joint density from P strengths and 2P phases.
            columns=torch.cat((canonical,torch.arange(edges,2*edges)))
            mu,sd=mu[...,columns],sd[...,columns]
            mu=mu[:,::2].repeat_interleave(2,1);sd=sd[:,::2].repeat_interleave(2,1)
            wide=(torch.arange(40)%8<2)[None,:,None]*config.phase_exploration_start_std*.5/np.sqrt(2)
            sd=torch.cat((sd[...,:len(canonical)],torch.sqrt(sd[...,len(canonical):]**2+wide**2)),dim=-1)
            bm,bl=density.budget_parameters(repeated,pooled_states=pooled)
            bm=bm.reshape(1,40)[:,::2].repeat_interleave(2,1)
            bs=bl.exp().reshape(1,40)[:,::2].repeat_interleave(2,1)
            expected=torch.distributions.Normal(mu,sd).log_prob(actions[...,columns]).sum(-1)
            expected+=torch.distributions.Normal(bm,bs).log_prob(budget)
            torch.testing.assert_close(score,expected,rtol=2e-5,atol=2e-5)
            (score*torch.arange(40)).sum().backward()
            assert policy.coupling_phase_head.weight.grad.abs().sum()>0
            assert policy.edge_output.weight.grad[:2].abs().sum()>0


def test_fresh_pipeline_and_checkpoint_preserve_mixed_constraints(tmp_path,fake_physics):
    c=run.CONFIG
    c=replace(c,optimization=replace(c.optimization,workers=1,iterations=2),
        sampling=replace(c.sampling,pairs_per_size=2,actions_per_topology=4),
        schedules=replace(c.schedules,ordinary_end=2,wide_end=2),
        actor_critic=replace(c.actor_critic,critic_updates=2,critic_batch_size=32),
        validation=replace(c.validation,pairs_per_size=5,plot_use_tex=False,notebook_display=False,plot_dpi=50))
    result=Experiment('mixed-smoke',config=c,output_directory=tmp_path/'run').train(verify_reference=False)
    assert result.history['critic_buffer_size']==[280,560]
    assert sum(len(k) for k,_,_ in fake_physics)==630
    for k,_,_ in fake_physics:np.testing.assert_array_equal(k,k.swapaxes(-1,-2))
    model=CouplingModel.load(result.checkpoint('final'),device='cpu')
    designs=model.propose(Context([0.,.3,-.6],[-1.,0.,1.]),density_fraction=1.,count=2)
    np.testing.assert_array_equal(designs.coupling_strengths_per_ns,designs.coupling_strengths_per_ns.swapaxes(-1,-2))
    assert model.backend_config.asymmetric_coupling_phases
