"""Matched density recipes: symmetric masks with independent directed values."""
from dataclasses import asdict, replace
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel, Experiment
from rl.rl_api.density_control import pair_counts
from rl.rl_api.models import build_policy
from rl.rl_api.experiments import train_mirrored_symmetric_density_pairs_1000 as old_ref
from rl.rl_api.experiments import train_mirrored_symmetric_density_pairs_zeroref_1000 as old_zero
from rl.rl_api.experiments import train_mirrored_density_asymmetric_values_1000 as ref
from rl.rl_api.experiments import train_mirrored_density_asymmetric_values_zeroref_1000 as zero


@pytest.mark.parametrize('before,after', [(old_ref,ref),(old_zero,zero)], ids=['laser1','zero'])
def test_only_value_symmetry_and_artifact_names_change(before,after):
    a,b=asdict(before.CONFIG),asdict(after.CONFIG)
    for flag in ('symmetric_coupling_values','reciprocal_pair_actions'):
        assert a['model'].pop(flag) is True
        assert b['model'].pop(flag) is False
    assert a['validation'].pop('progress_filename') != b['validation'].pop('progress_filename')
    assert a == b
    assert after.INITIAL_CHECKPOINT is None
    assert before.RUN_NAME != after.RUN_NAME
    summary=after.CONFIG.summary()
    assert summary['simulations_per_iteration']==1400
    assert summary['validation_simulations_per_check']==448
    assert summary['symmetric_masks'] and not summary['symmetric_strengths'] and not summary['symmetric_phases']


def connected(mask):
    seen={0}
    while True:
        updated=seen|{int(j) for i in seen for j in np.flatnonzero(mask[i])}
        if updated==seen:return len(seen)==len(mask)
        seen=updated


@pytest.mark.parametrize('recipe', [ref,zero], ids=['laser1','zero'])
def test_masks_remain_symmetric_and_values_are_independent_for_every_size(tmp_path,recipe):
    torch.manual_seed(7)
    config=recipe.CONFIG.to_backend(tmp_path)
    policy=build_policy(config)
    old=build_policy(old_ref.CONFIG.to_backend(tmp_path))
    assert sum(p.numel() for p in policy.parameters())==sum(p.numel() for p in old.parameters())
    model=CouplingModel(policy,config)
    for size in range(3,10):
        context=Context(np.linspace(0.,2.,size),np.linspace(-2.,2.,size))
        for actions in ('mean','sample'):
            designs=model.propose(context,density_fraction=[0.,.5,1.],count=3,
                                  continuous_actions=actions,seed=71)
            masks=designs.topologies
            np.testing.assert_array_equal(masks,masks.swapaxes(-1,-2))
            np.testing.assert_array_equal(designs.active_link_counts,2*pair_counts(size,np.array([0.,.5,1.])))
            assert all(connected(mask) for mask in masks)
            for values in (designs.coupling_strengths_per_ns,designs.coupling_phases_rad):
                assert np.isfinite(values).all()
                assert not np.allclose(values,values.swapaxes(-1,-2),atol=1e-7)
            replay=model.propose(context,density_fraction=[0.,.5,1.],count=3,
                                  continuous_actions=actions,seed=71)
            np.testing.assert_array_equal(replay.coupling_strengths_per_ns,designs.coupling_strengths_per_ns)


@pytest.mark.parametrize('recipe', [ref,zero], ids=['laser1','zero'])
def test_training_and_saved_model_use_asymmetric_values(tmp_path,fake_physics,recipe):
    config=replace(recipe.CONFIG,sizes=(3,4),
        sampling=replace(recipe.CONFIG.sampling,pairs_per_size=1,pairs_per_job=1),
        optimization=replace(recipe.CONFIG.optimization,iterations=2,workers=1),
        schedules=replace(recipe.CONFIG.schedules,ordinary_end=2,wide_end=2),
        validation=replace(recipe.CONFIG.validation,pairs_per_size=5,plot_use_tex=False,notebook_display=False))
    result=Experiment('asymmetric-density',config=config,output_directory=tmp_path/'run').train(
        live_plot=True,verify_reference=False,initial_checkpoint=recipe.INITIAL_CHECKPOINT)
    payload=torch.load(result.checkpoint('final'),weights_only=False)
    assert payload['iteration']==2
    assert 'initialization' not in payload['sampling_recipe']
    assert 'symmetric_values' not in payload['sampling_recipe']
    assert payload['exploration_state']['action_std_scale']==0.
    model=CouplingModel.load(result.checkpoint('final'),device='cpu')
    assert not getattr(model.backend_config,'symmetric_coupling_values',False)
    assert not getattr(model.backend_config,'reciprocal_pair_actions',False)
    assert model.backend_config.phase_parameterization==recipe.CONFIG.model.phase_parameterization
    designs=model.propose(Context([0.,.3,-.7,.8],[-2.,-.3,.5,2.]),density_fraction=None,count=7,seed=72)
    np.testing.assert_array_equal(designs.topologies,designs.topologies.swapaxes(-1,-2))
    assert (result.directory/recipe.CONFIG.validation.progress_filename).is_file()


def test_superseded_launcher_routes_to_warm_refinement():
    script=Path(__file__).resolve().parents[3]/'scripts/submit_asymmetric_value_density_tests.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    output=subprocess.run(['bash',str(script),'--dry-run','--smoke-test'],check=True,capture_output=True,text=True).stdout
    commands=[shlex.split(line) for line in output.splitlines() if line.startswith('sbatch ')]
    assert len(commands)==1
    assert 'scripts/train_density_symmetry_refinement.slurm' in commands[0]
    assert '--initial-checkpoint' in commands[0] and '--smoke-test' in commands[0]
