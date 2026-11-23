"""Hard physical limits, protected legacy behavior and isolated test variants."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import subprocess,shlex

import numpy as np
import pytest
import torch

from rl.rl_api import Context,CouplingModel
from rl.rl_api._compat import core,exported_settings,make_backend_config
from rl.rl_api.checkpoints import save
from rl.rl_api.models import build_policy
from rl.rl_api.physical_symmetry import constrain_physical_values
from rl.rl_api.root_context import calibrate_budget_references
from rl.rl_api.density_control import decode_density_actions
from rl.rl_api.exploration import sample_annealed_candidates
from rl.rl_api.experiments import train_mirrored_density_asymmetric_values_1000 as source
from rl.rl_api.experiments import train_density_physical_bounds as run


def cfg(fraction,strengths=True,phases=True):
    return SimpleNamespace(coupling_symmetry_fraction=fraction,
        coupling_symmetry_strengths=strengths,coupling_symmetry_phases=phases)


@pytest.mark.parametrize('bound,expected',[(1.,(8.,2.)),(.5,(7.5,2.5)),(.2,(6.,4.)),(0.,(5.,5.))])
def test_physical_strength_example(bound,expected):
    k=np.array([[[[0.,8.],[2.,0.]]]]);p=np.zeros_like(k);mask=np.ones((1,1,2))
    result,_=constrain_physical_values(k,p,mask,cfg(1.-bound))
    np.testing.assert_allclose(result[0,0,0,1],expected[0])
    np.testing.assert_allclose(result[0,0,1,0],expected[1])
    assert result.sum()==k.sum()


def test_pairs_inside_limits_are_bit_identical_and_wrapping_is_short():
    k=np.array([[[[0.,6.],[4.,0.]]]]);p=np.deg2rad([[[[0.,179.],[-179.,0.]]]])
    mask=np.ones((1,1,2))
    a,b=constrain_physical_values(k,p,mask,cfg(.5))
    np.testing.assert_array_equal(a,k);np.testing.assert_array_equal(b,p)
    a,b=constrain_physical_values(k,p,mask,cfg(1.-1/180,False,True))
    np.testing.assert_array_equal(a,k)
    delta=np.angle(np.exp(1j*(b[0,0,1,0]-b[0,0,0,1])))
    np.testing.assert_allclose(abs(np.rad2deg(delta)),1.,atol=1e-10)
    _,b=constrain_physical_values(k,p,mask,cfg(1.,False,True))
    np.testing.assert_array_equal(b,b.swapaxes(-1,-2))


@pytest.mark.parametrize('components',['both','strengths','phases'])
def test_bounds_cannot_be_bypassed_and_only_selected_values_change(components):
    strengths=components!='phases';phases=components!='strengths'
    k=np.array([[[[0.,1e12],[1e-12,0.]]]]);p=np.array([[[[0.,0.],[np.pi,0.]]]])
    a,b=constrain_physical_values(k,p,np.ones((1,1,2)),cfg(.9,strengths,phases))
    if strengths:
        ratio=abs(a[0,0,0,1]-a[0,0,1,0])/a.sum()
        assert ratio<=.1+1e-12
        np.testing.assert_allclose(a.sum(),k.sum(),rtol=1e-15)
    else:np.testing.assert_array_equal(a,k)
    if phases:
        assert abs(np.angle(np.exp(1j*(b[0,0,1,0]-b[0,0,0,1]))))<=.1*np.pi+1e-12
    else:np.testing.assert_array_equal(b,p)


def test_asymmetric_topology_rejected():
    k=np.array([[[[0.,1.],[0.,0.]]]])
    with pytest.raises(ValueError,match='symmetric topology'):
        constrain_physical_values(k,k,np.array([[[1.,0.]]]),cfg(.5))


@pytest.mark.parametrize('size',range(3,10))
def test_density_ranking_budget_and_physical_bounds_for_all_sizes(tmp_path,size):
    base=source.CONFIG.to_backend(tmp_path)
    settings=exported_settings(base);settings.update(coupling_symmetry_fraction=0.,coupling_symmetry_method='physical_bounds')
    config=replace(make_backend_config(settings),n_lasers=size)
    policy=build_policy(config)
    from rl.rl_api._compat import engine
    features=engine.encode_context(np.linspace(0.,2.,size)[None],np.linspace(-2.,2.,size)[None],policy)
    masks=torch.ones((1,8,size*(size-1)));fractions=np.array([[0.,0.,.3,.3,.5,.5,1.,1.]])
    torch.manual_seed(71)
    actions,budget,score=sample_annealed_candidates(policy,features,masks,mirrored=True)
    original_masks,original=decode_density_actions(actions,budget,fractions,config)
    i,j=np.triu_indices(size,1)
    for fraction in (.05,.5,1.):
        selected,(k,p,*_)=decode_density_actions(actions,budget,fractions,replace(config,coupling_symmetry_fraction=fraction))
        torch.testing.assert_close(selected,original_masks,rtol=0,atol=0)
        np.testing.assert_allclose(k.sum((-2,-1)),original[0].sum((-2,-1)),rtol=2e-6)
        a,b=k[...,i,j],k[...,j,i];ratio=np.abs(a-b)/np.maximum(a+b,1e-30)
        assert np.max(ratio)<=1.-fraction+1e-10
        delta=np.angle(np.exp(1j*(p[...,i,j]-p[...,j,i])))
        assert np.max(np.abs(delta))<=(1.-fraction)*np.pi+1e-10
        if fraction==1.:
            np.testing.assert_array_equal(k,k.swapaxes(-1,-2));np.testing.assert_array_equal(p,p.swapaxes(-1,-2))


@pytest.mark.parametrize('components',['both','strengths','phases'])
def test_warm_runner_endpoint_reload_and_distinct_artifacts(tmp_path,fake_physics,components):
    torch.manual_seed(7);config=source.CONFIG.to_backend(tmp_path/'source');policy=build_policy(config)
    config=calibrate_budget_references(policy,config);policy.config=config
    checkpoint=tmp_path/'source.pt';save(checkpoint,policy,torch.optim.Adam(policy.parameters()),config,950,.991)
    directory=tmp_path/components
    result=run.main(['--components',components,'--initial-checkpoint',str(checkpoint),
        '--output-directory',str(directory),'--workers','1','--smoke-test','--reward-drop','1','--success-drop','1'])
    assert result['reached_fraction']==1.
    assert result['components']==components and result['projection_method']=='physical_bounds'
    name='fully_symmetric.pt' if components=='both' else 'constraint_endpoint.pt'
    assert result['endpoint_checkpoint']==name
    model=CouplingModel.load(directory/name,device='cpu')
    assert model.backend_config.coupling_symmetry_method=='physical_bounds'
    designs=model.propose(Context([0.,.2,-.7],[-1.,0.,1.]),density_fraction=1.,count=1)
    for selected,values in ((components!='phases',designs.coupling_strengths_per_ns),(components!='strengths',designs.coupling_phases_rad)):
        if selected:np.testing.assert_array_equal(values,values.swapaxes(-1,-2))
        else:assert not np.allclose(values,values.swapaxes(-1,-2))
    label={'both':'both','strengths':'kappa','phases':'phase'}[components]
    assert (directory/f'density_bound_{label}_progress.png').is_file()
    if components!='both':assert not (directory/'fully_symmetric.pt').exists()


def test_three_job_launcher_and_old_launcher_unchanged():
    root=Path(__file__).resolve().parents[3]
    script=root/'scripts/submit_density_physical_bounds_tests.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    text=subprocess.run(['bash',str(script),'--dry-run','--smoke-test'],check=True,capture_output=True,text=True).stdout
    cmds=[shlex.split(line) for line in text.splitlines() if line.startswith('sbatch ')]
    assert len(cmds)==3
    for c,name,part in zip(cmds,('density_bound_both','density_bound_kappa','density_bound_phase'),('both','strengths','phases')):
        assert '--job-name='+name in c and c[c.index('--components')+1]==part and '--smoke-test' in c
    assert 'cancel' not in text.lower().replace('unchanged','')
    slurm=(root/'scripts/train_density_physical_bounds.slurm').read_text()
    assert '--time=7-00:00:00' in slurm and '--cpus-per-task=32' in slurm
    old=(root/'scripts/train_density_symmetry_refinement.slurm').read_text()
    assert 'rl.rl_api.experiments.train_density_symmetry_refinement' in old
