"""Replay separation, critic action ranking and simulator-free actor updates."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from rl.rl_api import ActorCriticConfig, Context, CouplingModel, Experiment
from rl.rl_api.actor_critic import ActorCriticState, DesignReplay, RewardCritic, experience, ranking_metrics
from rl.rl_api.density_control import prepare_density_size
from rl.rl_api.experiments import train_density_actor_critic_1000 as current
from rl.rl_api.experiments import train_density_actor_critic_replay_1000 as replay
from rl.rl_api.experiments import train_density_conditioned_fresh_1000 as baseline
from rl.rl_api.experiments import train_density_actor_critic_replay_asymphase_1000 as mixed
from rl.rl_api.exploration import refinement_config
from rl.rl_api.models import build_policy


def small_config(recipe=current):
    return replace(recipe.CONFIG,
        sampling=replace(recipe.CONFIG.sampling, pairs_per_size=2, actions_per_topology=4),
        optimization=replace(recipe.CONFIG.optimization, iterations=2, workers=1),
        schedules=replace(recipe.CONFIG.schedules, ordinary_end=2, wide_end=2),
        actor_critic=replace(recipe.CONFIG.actor_critic, critic_updates=2, actor_updates=1, critic_batch_size=32),
        validation=replace(recipe.CONFIG.validation, pairs_per_size=5, plot_use_tex=False,
                           notebook_display=False, plot_dpi=60))


def prepared(tmp_path,recipe=current):
    public = small_config(recipe)
    config = refinement_config(public.to_backend(tmp_path), public.schedules, 0)
    torch.manual_seed(config.random_seed)
    policy = build_policy(config)
    items = [prepare_density_size(policy, config, np.random.default_rng(72), size, 2) for size in (3, 9)]
    return public, config, policy, items


def test_matched_fresh_recipes_and_resources():
    a, b, original = asdict(current.CONFIG), asdict(replay.CONFIG), asdict(baseline.CONFIG)
    assert not a['actor_critic'].pop('use_replay') and b['actor_critic'].pop('use_replay')
    assert a['validation'].pop('progress_filename')!=b['validation'].pop('progress_filename')
    assert a==b and current.INITIAL_CHECKPOINT is replay.INITIAL_CHECKPOINT is None
    original.pop('actor_critic'); a.pop('actor_critic')
    original['validation'].pop('progress_filename')
    assert a==original
    assert current.CONFIG.summary()['simulations_per_iteration']==1400
    assert current.CONFIG.summary()['validation_simulations_per_check']==448
    script=Path(__file__).resolve().parents[3]/'scripts/submit_density_actor_critic_tests.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    output=subprocess.run(['bash',str(script),'--dry-run'],check=True,capture_output=True,text=True).stdout
    assert output.count('sbatch ')==2 and '--initial-checkpoint' not in output and 'scancel' not in output
    for token in ('--cpus-per-task=32','--time=7-00:00:00','--mem=64G','--workers 31'):
        assert output.count(token)==2
    assert '--job-name=density_ac_current' in output and '--job-name=density_ac_replay' in output


def test_replay_capacity_wrap_sampling_and_clear(tmp_path):
    _, _, _, items = prepared(tmp_path)
    batches = {item['n_lasers']:experience(item, np.linspace(0,1,40).reshape(2,20), 0) for item in items}
    buffer = DesignReplay((3,9), 100)
    for size, batch in batches.items():
        buffer.add(size,batch)
    assert len(buffer)==80
    for size,batch in batches.items():
        buffer.add(size,dict(batch,reward=batch['reward']*.1))
    assert len(buffer)==100 and all(count==50 for count in buffer.counts.values())
    # Last full batch and ten surviving older records of each size.
    for data in buffer.data.values():
        assert np.count_nonzero(data['reward']>.1)==10
    sample=buffer.sample(3000,np.random.default_rng(8),buffer.strata())
    assert set(sample)=={3,9}
    for data in sample.values():
        assert data['trainable'].all()
    assert 1200<len(sample[3]['reward'])<1800
    buffer.clear()
    for size,batch in batches.items():buffer.add(size,batch)
    assert len(buffer)==80 and all(count==40 for count in buffer.counts.values())
    assert all(value.shape[0]==40 for value in buffer.snapshot().values())


@pytest.mark.parametrize('size',[3,9])
def test_critic_inputs_circularity_and_permutation(size):
    torch.manual_seed(8)
    critic=RewardCritic()
    features=torch.randn(2,size,4)
    kappa=torch.rand(2,size,size)
    phi=torch.rand(2,size,size)*6
    mask=torch.ones_like(kappa)-torch.eye(size)
    mask[:,0,1]=mask[:,1,0]=0
    fraction=torch.tensor([.3,.7])
    def predict(k=kappa,p=phi,m=mask,f=features):return critic(f,k,p,m,fraction)
    original=predict()
    assert torch.all((original>0)&(original<1))
    torch.testing.assert_close(original,predict(p=phi+2*np.pi),atol=2e-7,rtol=1e-6)
    ignored=phi.clone();ignored[:,0,1]+=1.7
    torch.testing.assert_close(original,predict(p=ignored),atol=0,rtol=0)
    assert not torch.equal(original,predict(k=kappa*3))
    assert not torch.equal(original,predict(p=phi+.6))
    assert not torch.equal(original,predict(m=1-torch.eye(size).expand_as(mask)))
    permutation=torch.randperm(size)
    torch.testing.assert_close(original,predict(f=features[:,permutation],
        k=kappa[:,permutation][:,:,permutation],p=phi[:,permutation][:,:,permutation],
        m=mask[:,permutation][:,:,permutation]),atol=2e-7,rtol=1e-6)


def test_initialization_holdouts_and_no_actor_change_during_critic_fit(tmp_path):
    public, config, policy, items=prepared(tmp_path)
    before={key:value.clone() for key,value in policy.state_dict().items()}
    rng=torch.get_rng_state()
    state=ActorCriticState(public.actor_critic,config,tmp_path)
    torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
    for item in items:
        data=experience(item,np.linspace(0,1,40).reshape(2,20),0)
        withheld=~data['trainable'].reshape(2,20)
        assert withheld.sum()==20 and np.all(withheld.any(1)==withheld.all(1))
        state.buffer.add(item['n_lasers'],data)
    state.fit()
    for key,value in before.items():torch.testing.assert_close(value,policy.state_dict()[key],atol=0,rtol=0)
    assert all(parameter.grad is None for parameter in state.critic.parameters())
    rewards=np.tile(np.linspace(.1,.9,8),(2,1))
    trainable=np.repeat([True,False],8)
    assert ranking_metrics(rewards,rewards,trainable,8)[1:3]==(28.,28)
    assert ranking_metrics(-rewards,rewards,trainable,8)[1]==0
    assert ranking_metrics(np.zeros_like(rewards),rewards,trainable,8)[1]==14


@pytest.mark.parametrize('recipe',[current,mixed],ids=['symmetric-phases','directed-phases'])
def test_virtual_actor_uses_fresh_actions_no_simulations_or_critic_gradients(tmp_path,monkeypatch,fake_physics,recipe):
    public,config,policy,items=prepared(tmp_path,recipe)
    settings=replace(public.actor_critic,warmup_iterations=1)
    state=ActorCriticState(settings,config,tmp_path)
    # Analytic test oracle: action-dependent, independently measurable reward.
    def oracle(data):
        angle=torch.as_tensor(data['phi'][:,0,1],dtype=torch.float32)
        return .5+.4*torch.sin(angle)
    monkeypatch.setattr(state,'fit',lambda:0.)
    monkeypatch.setattr(state,'predict',oracle)
    rewards=[oracle(experience(item,np.zeros((2,20)),0)).numpy().reshape(2,20) for item in items]
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    before=[p.clone() for p in policy.parameters()]
    rng=torch.get_rng_state()
    from rl.rl_api import conditional_phases
    sample=conditional_phases.sample_conditional_density
    draws=[]
    def track(*args,**kwargs):
        result=sample(*args,**kwargs)
        draws.append(result)
        return result
    monkeypatch.setattr(conditional_phases,'sample_conditional_density',track)
    metrics=state.update(policy,optimizer,config,items,rewards,0)
    assert metrics['critic_mae']<1e-6 and metrics['critic_rank_accuracy']==1.
    assert metrics['critic_actor_updates']==1 and len(draws)==2 and len(fake_physics)==0
    assert any(not torch.equal(a,b) for a,b in zip(before,policy.parameters()))
    assert all(p.grad is None for p in state.critic.parameters())
    torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
    for item,result in zip(items,draws):
        assert not np.array_equal(item['kappa'],result[-1][0])
        np.testing.assert_array_equal(result[-1][0],result[-1][0].swapaxes(-1,-2))
        if recipe is current:
            np.testing.assert_array_equal(result[-1][1],result[-1][1].swapaxes(-1,-2))
    # A bad local ranking blocks virtual updates despite small absolute error.
    monkeypatch.setattr(state,'predict',lambda data:torch.full((len(data['reward']),),.5))
    metrics=state.update(policy,optimizer,config,items,rewards,1)
    assert metrics['critic_actor_updates']==0 and metrics['critic_rank_accuracy']==.5
    monkeypatch.setattr(state,'predict',oracle)
    deterministic=replace(config,exploration_std_scale=0.)
    before=[p.clone() for p in policy.parameters()]
    metrics=state.update(policy,optimizer,deterministic,items,rewards,2)
    assert metrics['critic_actor_updates']==0
    assert all(torch.equal(a,b) for a,b in zip(before,policy.parameters()))


@pytest.mark.parametrize('recipe',[current,replay],ids=['current','replay'])
def test_full_training_pipeline_archives_reload_and_plot(tmp_path,fake_physics,recipe):
    config=small_config(recipe)
    result=Experiment('critic-smoke',config=config,output_directory=tmp_path/'run').train(
        live_plot=True,verify_reference=False)
    assert result.history['critic_buffer_size']==([280,280] if recipe is current else [280,560])
    assert result.history['critic_actor_updates']==[0,0]  # Warmup and exact-mean final check.
    assert all(np.isfinite(result.history['critic_loss']))
    assert sum(len(k) for k,_,_ in fake_physics)==630  # 2*280 training + 2*35 validation, no extra rollouts.
    manifest=json.loads((result.directory/'run.json').read_text())
    assert manifest['public_config']['actor_critic']['enabled']
    archive=np.load(result.directory/'experiences/batch_000002.npz')
    assert sum(len(archive[f'M{m}_reward']) for m in range(3,10))==280
    snapshot=np.load(result.directory/'critic_buffer_latest.npz')
    assert sum(len(snapshot[f'M{m}_reward']) for m in range(3,10))==result.history['critic_buffer_size'][-1]
    saved=torch.load(result.directory/'critic_current.pt',weights_only=False)
    assert saved['iteration']==2 and saved['settings']['use_replay']==(recipe is replay)
    model=CouplingModel.load(result.checkpoint('final'),device='cpu')
    designs=model.propose(Context([0.,.2,-.7],[-1.,0.,1.]),density_fraction=.5,count=2)
    for values in (designs.coupling_strengths_per_ns,designs.coupling_phases_rad):
        np.testing.assert_array_equal(values,values.swapaxes(-1,-2))
    from rl.rl_api.plotting.training import training_figure
    import matplotlib.pyplot as plt
    with training_figure(result.history,model.backend_config,config.validation) as figure:
        assert len(figure.axes)==2 and figure.axes[0].get_title().startswith('Actor–critic')
        plt.close(figure)
    assert (result.directory.parent/config.validation.progress_filename).is_file()


@pytest.mark.parametrize('kwargs',[dict(critic_updates=0),dict(use_replay=1),
    dict(minimum_rank_accuracy=.5),dict(maximum_reward_mae=float('nan'))])
def test_critic_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):ActorCriticConfig(**kwargs)
