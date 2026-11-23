"""Sequential Gaussian scoring, warm-start parity and checkpoint inference."""
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel
from rl.rl_api._compat import engine, exported_settings, make_backend_config
from rl.rl_api.checkpoints import save
from rl.rl_api.conditional_phases import sample_conditional_density
from rl.rl_api.density_control import DensityConditionedPolicy, decode_density_actions
from rl.rl_api.exploration import sample_annealed_candidates
from rl.rl_api.models import build_policy
from rl.rl_api.root_context import calibrate_budget_references
from rl.rl_api.experiments import train_mirrored_density_asymmetric_values_1000 as source
from rl.rl_api.experiments import train_density_conditional_phase as run


def setup(tmp_path,size=4):
    base=source.CONFIG.to_backend(tmp_path)
    settings=exported_settings(base)
    settings.update(coupling_symmetry_fraction=.3,coupling_symmetry_method='physical_bounds')
    plain=replace(make_backend_config(settings),n_lasers=size,exploration_std_scale=.03,
                  phase_exploration_fraction=0.,phase_exploration_start_std=0.)
    original=build_policy(plain)
    settings=exported_settings(plain);settings['condition_phases_on_coupling']=True
    config=make_backend_config(settings);policy=build_policy(config)
    missing,unexpected=policy.load_state_dict(original.state_dict(),strict=False)
    assert set(missing)=={'coupling_phase_head.weight','coupling_phase_head.bias'} and not unexpected
    features=engine.encode_context(np.linspace(0.,2.,size)[None],np.linspace(-2.,2.,size)[None],policy)
    fractions=np.array([[0.,0.,.3,.3,.5,.5,1.,1.]])
    return original,policy,config,features,fractions


@pytest.mark.parametrize('size',range(3,10))
def test_zero_head_preserves_actions_scores_rng_and_physical_designs(tmp_path,size):
    original,policy,config,features,fractions=setup(tmp_path,size)
    dense=features.new_ones((1,8,size*(size-1)))
    torch.manual_seed(72)
    a,b,s=sample_annealed_candidates(DensityConditionedPolicy(original,fractions),features,dense,mirrored=True)
    m,physical=decode_density_actions(a,b,fractions,original.config);rng=torch.get_rng_state()
    torch.manual_seed(72)
    aa,bb,ss,mm,pp=sample_conditional_density(policy,features,config,fractions,mirrored=True)
    for old,new in ((a,aa),(b,bb),(s,ss),(m,mm),(rng,torch.get_rng_state())):
        torch.testing.assert_close(old,new,rtol=0,atol=0)
    for old,new in zip(physical,pp):np.testing.assert_array_equal(old,new)
    assert sum(p.numel() for p in policy.coupling_phase_head.parameters())==129


def test_conditional_scores_own_mirror_means_and_detached_strengths(tmp_path,assert_mirrored_phase_noise):
    _,policy,config,features,fractions=setup(tmp_path)
    with torch.no_grad():policy.coupling_phase_head.weight.fill_(.07)
    dense=features.new_ones((1,8,12))
    torch.manual_seed(29)
    a,b,s,mean,std=sample_annealed_candidates(DensityConditionedPolicy(policy,fractions),
        features,dense,mirrored=True,return_phase_parameters=True)
    masks,physical=decode_density_actions(a,b,fractions,config)
    repeated=features.expand(8,-1,-1)
    k=torch.tensor(physical[0].reshape(8,4,4),dtype=features.dtype,requires_grad=True)
    residual=policy.phase_residual_for_coupling(repeated,k).reshape(1,8,12)
    assert not torch.allclose(residual[:,::2],residual[:,1::2],rtol=0,atol=1e-8)
    torch.manual_seed(29)
    aa,bb,ss,mm,pp=sample_conditional_density(policy,features,config,fractions,mirrored=True)
    torch.testing.assert_close(aa[...,:12],a[...,:12]);torch.testing.assert_close(bb,b)
    torch.testing.assert_close(mm,masks);np.testing.assert_array_equal(pp[0],physical[0])
    assert_mirrored_phase_noise(aa[...,12:],mean+residual)
    expected=s+(torch.distributions.Normal(mean+residual,std).log_prob(aa[...,12:])-
                torch.distributions.Normal(mean,std).log_prob(a[...,12:])).sum(-1)
    torch.testing.assert_close(ss,expected)
    residual.sum().backward();assert k.grad is None
    policy.zero_grad()
    (ss*torch.arange(8)).sum().backward()
    assert policy.coupling_phase_head.weight.grad.abs().sum()>0
    assert policy.edge_output.weight.grad[0].abs().sum()>0
    assert policy.edge_output.weight.grad[1].abs().sum()>0
    assert next(policy.budget_head.parameters()).grad.abs().sum()>0


def test_warm_runner_reload_and_inference_share_conditional_path(tmp_path,fake_physics):
    config=source.CONFIG.to_backend(tmp_path/'source');policy=build_policy(config)
    config=calibrate_budget_references(policy,config);policy.config=config
    checkpoint=tmp_path/'source.pt';save(checkpoint,policy,torch.optim.Adam(policy.parameters()),config,950,.991)
    directory=tmp_path/'conditional'
    result=run.main(['--initial-checkpoint',str(checkpoint),'--output-directory',str(directory),
        '--workers','1','--smoke-test','--reward-drop','1','--success-drop','1'])
    assert result['reached_fraction']==1.
    model=CouplingModel.load(directory/'current.pt',device='cpu')
    assert model.backend_config.condition_phases_on_coupling
    with torch.no_grad():model.policy.coupling_phase_head.bias.fill_(.25)
    for continuous_mean in (True,False):
        designs=model.propose(Context([0.,.2,-.7],[-1.,0.,1.]),density_fraction=.5,
                              count=2,continuous_actions='mean' if continuous_mean else 'sample',seed=7)
        np.testing.assert_array_equal(designs.coupling_strengths_per_ns,
                                      designs.coupling_strengths_per_ns.swapaxes(-1,-2))
        np.testing.assert_array_equal(designs.coupling_phases_rad,
                                      designs.coupling_phases_rad.swapaxes(-1,-2))
    assert (directory/'density_conditional_phase_progress.png').is_file()
    assert sum(len(k) for k,_,_ in fake_physics)==200  # No additional simulated designs.


def test_additional_launcher_is_isolated():
    root=Path(__file__).resolve().parents[3]
    launcher=root/'scripts/submit_density_conditional_phase.sh'
    subprocess.run(['bash','-n',str(launcher)],check=True)
    text=subprocess.run(['bash',str(launcher),'--dry-run'],capture_output=True,text=True,check=True).stdout
    assert text.count('sbatch ')==1 and '--job-name=density_cond_phase' in text
    slurm=(root/'scripts/train_density_conditional_phase.slurm').read_text()
    assert '--time=7-00:00:00' in slurm and '--cpus-per-task=32' in slurm
