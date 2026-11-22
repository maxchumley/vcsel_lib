"""Continuous physical projection, quality guard and warm checkpoint replay."""
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel
from rl.rl_api._compat import exported_settings, make_backend_config, core
from rl.rl_api.checkpoints import save
from rl.rl_api.exploration import sample_annealed_candidates
from rl.rl_api.models import build_policy
from rl.rl_api.phase_reference import decode_samples
from rl.rl_api.reciprocal_pairs import pair_layout
from rl.rl_api.root_context import calibrate_budget_references
from rl.rl_api.experiments import train_mirrored_density_asymmetric_values_1000 as source
from rl.rl_api.experiments import train_density_symmetry_refinement as run


def config(tmp_path,fraction):
    d=exported_settings(source.CONFIG.to_backend(tmp_path));d['coupling_symmetry_fraction']=fraction
    return make_backend_config(d)


@pytest.mark.parametrize('fraction',[-.1,1.1,np.nan])
def test_invalid_fraction_rejected(tmp_path,fraction):
    with pytest.raises(ValueError,match='coupling_symmetry_fraction'):config(tmp_path,fraction)


def test_zero_is_exact_original_and_endpoint_is_symmetric(tmp_path):
    base=source.CONFIG.to_backend(tmp_path);torch.manual_seed(7)
    actions=torch.randn((1,8,12));masks=torch.ones((1,8,6));budgets=torch.full((1,8),float(np.log(12.34)))
    original=decode_samples(actions,masks,base,budget_actions=budgets)
    zero=decode_samples(actions,masks,config(tmp_path,0.),budget_actions=budgets)
    for a,b in zip(original,zero):np.testing.assert_array_equal(a,b)
    for fraction in (.25,.5,1.):
        k,p,*_=decode_samples(actions,masks,config(tmp_path,fraction),budget_actions=budgets)
        np.testing.assert_allclose(k.sum((-1,-2)),12.34,rtol=2e-6)
        assert np.isfinite(k).all() and np.isfinite(p).all()
        if fraction==1.:
            np.testing.assert_array_equal(k,k.swapaxes(-1,-2))
            np.testing.assert_array_equal(p,p.swapaxes(-1,-2))
        else:assert not np.allclose(k,k.swapaxes(-1,-2))


def test_phases_follow_short_arc_across_wrap(tmp_path):
    canonical,_,reverse=pair_layout(3)
    actions=torch.ones((1,1,12));actions[...,6:]=0.
    i=int(canonical[0]);j=int(reverse[i])
    actions[...,6+i]=np.deg2rad(179.);actions[...,6+j]=np.deg2rad(-179.)
    masks=torch.ones((1,1,6));r,s=core.directed_link_indices(3)
    for fraction,expected in ((.5,179.5),(1.,180.)):
        _,p,*_=decode_samples(actions,masks,config(tmp_path,fraction))
        distance=np.angle(np.exp(1j*(p[0,0,r[i],s[i]]-np.deg2rad(expected))))
        assert abs(distance)<1e-6
        if fraction==1.:np.testing.assert_array_equal(p,p.swapaxes(-1,-2))


def test_full_latent_score_preserved_even_for_circular_endpoint(tmp_path):
    cfg=config(tmp_path,0.);policy=build_policy(cfg)
    from rl.rl_api._compat import engine
    features=engine.encode_context([[0.,.2,-.7]],[[-1.,0.,1.]],policy)
    masks=torch.ones((1,8,6));original=None
    for fraction in (0.,.5,1.):
        policy.config=replace(cfg,coupling_symmetry_fraction=fraction)
        torch.manual_seed(71)
        values=sample_annealed_candidates(policy,features,masks,mirrored=True)
        if original is None:original=values
        else:
            for a,b in zip(values,original):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_retention_guard_is_absolute_to_original_baseline():
    baseline=dict(reward=.99,success=.93)
    assert run.retained(dict(reward=.985,success=.90),baseline,.01,.05)
    assert not run.retained(dict(reward=.97,success=.90),baseline,.01,.05)
    assert not run.retained(dict(reward=.985,success=.87),baseline,.01,.05)
    assert not run.retained(dict(reward=np.nan,success=.93),baseline,.01,.05)


@pytest.mark.parametrize('hold',[False,True],ids=['advance','hold'])
def test_warm_training_guard_and_checkpoint_inference(tmp_path,fake_physics,monkeypatch,hold):
    torch.manual_seed(7);cfg=source.CONFIG.to_backend(tmp_path/'source');policy=build_policy(cfg)
    cfg=calibrate_budget_references(policy,cfg);policy.config=cfg
    checkpoint=tmp_path/'source.pt';save(checkpoint,policy,torch.optim.Adam(policy.parameters()),cfg,950,.991)
    original={k:v.clone() for k,v in policy.state_dict().items()}
    if hold:
        calls=[]
        def quality(_):
            calls.append(1)
            return dict(reward=.99 if len(calls)==1 else .1,success=.99 if len(calls)==1 else 0.)
        monkeypatch.setattr(run,'quality',quality)
    directory=tmp_path/'refinement'
    args=['--initial-checkpoint',str(checkpoint),'--output-directory',str(directory),'--workers','1','--smoke-test']
    if not hold:args+=['--reward-drop','1','--success-drop','1']
    result=run.main(args)
    assert result['reached_fraction']==(.5 if hold else 1.)
    assert result['best_accepted_fraction']==(0. if hold else 1.)
    initial=torch.load(directory/'initial.pt',weights_only=False)
    for k,v in original.items():torch.testing.assert_close(initial['policy_state_dict'][k],v,rtol=0,atol=0)
    final=CouplingModel.load(directory/'final.pt',device='cpu')
    assert final.backend_config.coupling_symmetry_fraction==result['reached_fraction']
    fully=CouplingModel.load(directory/'fully_symmetric.pt',device='cpu')
    assert fully.backend_config.coupling_symmetry_fraction==1.
    designs=fully.propose(Context([0.,.2,-.7],[-1.,0.,1.]),density_fraction=[0.,1.],count=2)
    for a in (designs.coupling_strengths_per_ns,designs.coupling_phases_rad):np.testing.assert_array_equal(a,a.swapaxes(-1,-2))
    assert (directory/run.PROGRESS_FILENAME).is_file()
    assert fully.metadata['sampling_recipe']['forced_endpoint_evaluation']==hold


def test_launcher_and_allocation():
    root=Path(__file__).resolve().parents[3]
    script=root/'scripts/submit_density_symmetry_refinement.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    output=subprocess.run(['bash',str(script),'--dry-run','--smoke-test'],capture_output=True,text=True,check=True).stdout
    assert '--initial-checkpoint' in output and '--smoke-test' in output
    slurm=(root/'scripts/train_density_symmetry_refinement.slurm').read_text()
    for expected in ('--cpus-per-task=32','--time=7-00:00:00','--mem=64G','--account=arrays0','--job-name=density_sym_refine'):
        assert expected in slurm
