"""Fresh symmetric pair conditioning across all sizes and standard plotting."""
from dataclasses import asdict,replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from rl.rl_api import Context,CouplingModel,Experiment,TrainingConfig,ModelConfig
from rl.rl_api._compat import engine
from rl.rl_api.conditional_phases import sample_conditional_density
from rl.rl_api.density_control import DensityConditionedPolicy,decode_density_actions
from rl.rl_api.exploration import sample_annealed_candidates
from rl.rl_api.models import build_policy
from rl.rl_api.reciprocal_pairs import pair_layout,check_shared_actions
from rl.rl_api.experiments import train_mirrored_symmetric_density_pairs_1000 as base
from rl.rl_api.experiments import train_density_conditioned_fresh_1000 as run
from rl.rl_api.experiments import train_density_conditioned_fresh_zeroref_1000 as zero


def test_recipe_only_adds_conditioning_and_unique_outputs():
    a,b=asdict(base.CONFIG),asdict(run.CONFIG)
    assert not a['model'].pop('condition_phases_on_coupling')
    assert b['model'].pop('condition_phases_on_coupling')
    a['validation'].pop('progress_filename');b['validation'].pop('progress_filename')
    assert a==b and run.INITIAL_CHECKPOINT is None
    summary=run.CONFIG.summary()
    assert summary['simulations_per_iteration']==1400
    assert summary['validation_simulations_per_check']==448
    assert summary['new_parameters']==129 and summary['extra_simulations']==0
    with pytest.raises(ValueError,match='density training'):
        TrainingConfig(model=ModelConfig(condition_phases_on_coupling=True))


@pytest.mark.parametrize('size',range(3,10))
def test_pair_conditioning_preserves_start_and_scores_unique_coordinates(tmp_path,size,assert_mirrored_phase_noise):
    config=replace(run.CONFIG.to_backend(tmp_path),n_lasers=size)
    torch.manual_seed(7);original=build_policy(base.CONFIG.to_backend(tmp_path))
    original.config=replace(original.config,n_lasers=size)
    rng=torch.get_rng_state()
    torch.manual_seed(7);policy=build_policy(config)
    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
    for name,p in original.state_dict().items():
        torch.testing.assert_close(p,policy.state_dict()[name],rtol=0,atol=0)
    features=engine.encode_context(np.linspace(0.,2.,size)[None],np.linspace(-2.,2.,size)[None],policy)
    fractions=np.array([[0.,0.,.3,.3,.5,.5,1.,1.]])
    edges=size*(size-1);dense=features.new_ones((1,8,edges))
    torch.manual_seed(29)
    a,b,s=sample_annealed_candidates(DensityConditionedPolicy(original,fractions),features,dense,mirrored=True)
    masks,physical=decode_density_actions(a,b,fractions,original.config)
    torch.manual_seed(29)
    aa,bb,ss,mm,pp=sample_conditional_density(policy,features,config,fractions,mirrored=True)
    for old,new in ((a,aa),(b,bb),(s,ss),(masks,mm)):
        torch.testing.assert_close(old,new,rtol=0,atol=0)
    for old,new in zip(physical,pp):np.testing.assert_array_equal(old,new)
    with torch.no_grad():policy.coupling_phase_head.weight.fill_(.07)
    torch.manual_seed(29)
    a,b,s,mean,std=sample_annealed_candidates(DensityConditionedPolicy(policy,fractions),
        features,dense,mirrored=True,return_phase_parameters=True)
    canonical=pair_layout(size)[0]
    assert mean.shape==std.shape==(1,8,edges//2)
    residual=policy.phase_residual_for_coupling(features.expand(8,-1,-1),
        physical[0].reshape(8,size,size)).reshape(1,8,edges)[...,canonical]
    torch.manual_seed(29)
    aa,bb,ss,mm,pp=sample_conditional_density(policy,features,config,fractions,mirrored=True)
    phases=aa[...,edges:][...,canonical]
    expected=s+(torch.distributions.Normal(mean+residual,std).log_prob(phases)-
                torch.distributions.Normal(mean,std).log_prob(a[...,edges:][...,canonical])).sum(-1)
    torch.testing.assert_close(ss,expected)
    assert_mirrored_phase_noise(phases,mean+residual)
    check_shared_actions(aa,mm,size)
    for values in pp[:2]:np.testing.assert_array_equal(values,values.swapaxes(-1,-2))
    np.testing.assert_array_equal(pp[0],physical[0])
    (ss*torch.arange(8)).sum().backward()
    assert policy.coupling_phase_head.weight.grad.abs().sum()>0
    assert policy.edge_output.weight.grad[:2].abs().sum()>0


@pytest.mark.parametrize('recipe',[run,zero],ids=['laser1','zero'])
def test_full_size_pipeline_reload_and_two_panel_plot(tmp_path,fake_physics,recipe):
    config=replace(recipe.CONFIG,
        sampling=replace(recipe.CONFIG.sampling,pairs_per_size=1,pairs_per_job=1),
        optimization=replace(recipe.CONFIG.optimization,iterations=2,workers=1),
        schedules=replace(recipe.CONFIG.schedules,ordinary_end=2,wide_end=2),
        validation=replace(recipe.CONFIG.validation,pairs_per_size=5,plot_use_tex=False,notebook_display=False))
    result=Experiment('fresh-conditional',config=config,output_directory=tmp_path/'run').train(
        live_plot=True,verify_reference=False)
    manifest=json.loads((result.directory/'run.json').read_text())
    assert manifest.get('initialization') is None
    payload=torch.load(result.checkpoint('final'),weights_only=False)
    recipe=payload['sampling_recipe']
    assert recipe['condition_phases_on_coupling'] and recipe['extra_simulations']==0
    assert 'initialization' not in recipe and 'no averaging' in recipe['symmetric_values']
    assert {k.shape[1] for k,_,_ in fake_physics}==set(range(3,10))
    for k,p,_ in fake_physics:
        for values in (k,p):np.testing.assert_array_equal(values,values.swapaxes(-1,-2))
    model=CouplingModel.load(result.checkpoint('best'),device='cpu')
    assert not hasattr(model.backend_config,'coupling_symmetry_fraction')
    assert model.backend_config.exploration_std_scale>0.
    context=Context([0.,.2,-.8,.7],[-2.,-.3,.5,2.])
    for mode in ('mean','sample'):
        a=model.propose(context,density_fraction=None,count=11,continuous_actions=mode,seed=71)
        with torch.no_grad():model.policy.coupling_phase_head.bias.add_(.2)
        b=model.propose(context,density_fraction=None,count=11,continuous_actions=mode,seed=71)
        np.testing.assert_array_equal(a.coupling_strengths_per_ns,b.coupling_strengths_per_ns)
        for values in (b.coupling_strengths_per_ns,b.coupling_phases_rad):
            np.testing.assert_array_equal(values,values.swapaxes(-1,-2))
        active=a.coupling_strengths_per_ns>0
        np.testing.assert_allclose(np.angle(np.exp(1j*(b.coupling_phases_rad-a.coupling_phases_rad)))[active],.2,atol=4e-6)
    from rl.rl_api.plotting.training import training_figure
    with training_figure(result.history,model.backend_config,config.validation) as figure:
        assert len(figure.axes)==2
        assert figure.axes[1].get_title()=='Density-conditioned held-out targets'
        import matplotlib.pyplot as plt
        plt.close(figure)
    assert (result.directory/config.validation.progress_filename).is_file()


def test_zero_reference_only_changes_phase_offset_and_output_names():
    a,b=asdict(run.CONFIG),asdict(zero.CONFIG)
    assert a['model'].pop('phase_parameterization')=='target_delay_residual'
    assert b['model'].pop('phase_parameterization')=='target_residual'
    assert a['validation'].pop('progress_filename')!=b['validation'].pop('progress_filename')
    assert a==b and zero.INITIAL_CHECKPOINT is None
    assert run.RUN_NAME!=zero.RUN_NAME


def test_fresh_job_resources():
    script=Path(__file__).resolve().parents[3]/'scripts/submit_density_conditioned_fresh.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    output=subprocess.run(['bash',str(script),'--dry-run'],check=True,capture_output=True,text=True).stdout
    assert output.count('sbatch ')==1 and '--initial-checkpoint' not in output
    for token in ('--job-name=density_cond_fresh','--cpus-per-task=32','--time=7-00:00:00','--workers 31'):
        assert token in output
    output=subprocess.run(['bash',str(script),'--dry-run','--zero-reference'],
        check=True,capture_output=True,text=True).stdout
    assert output.count('sbatch ')==1 and '--job-name=density_cond_zero' in output
    assert 'train_density_conditioned_fresh_zeroref_1000' in output
    assert '--initial-checkpoint' not in output
