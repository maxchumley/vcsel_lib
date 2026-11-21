"""Pair sampling, executed symmetry, score correctness and saved-model replay."""
from dataclasses import asdict, replace
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.rl_api import Context, CouplingModel, Experiment
from rl.rl_api._compat import engine
from rl.rl_api.density_control import DensityConditionedPolicy, decode_density_actions, pair_counts
from rl.rl_api.exploration import sample_annealed_candidates, shared_pair_parameters
from rl.rl_api.models import build_policy
from rl.rl_api.models.policy import delay_phase_offsets
from rl.rl_api.reciprocal_pairs import pair_layout, check_shared_actions
from rl.rl_api.experiments import train_mirrored_symmetric_density_pairs_1000 as ref
from rl.rl_api.experiments import train_mirrored_symmetric_density_pairs_zeroref_1000 as zero
from rl.rl_api.experiments import train_mirrored_symmetric_density_values_1000 as old


def test_recipes_only_change_pair_parameterization_and_output_names():
    before, after = asdict(old.CONFIG), asdict(ref.CONFIG)
    assert not before['model'].pop('reciprocal_pair_actions')
    assert after['model'].pop('reciprocal_pair_actions')
    before['validation'].pop('progress_filename')
    after['validation'].pop('progress_filename')
    assert before == after
    a, b = asdict(ref.CONFIG), asdict(zero.CONFIG)
    assert a['model'].pop('phase_parameterization') == 'target_delay_residual'
    assert b['model'].pop('phase_parameterization') == 'target_residual'
    a['validation'].pop('progress_filename'); b['validation'].pop('progress_filename')
    assert a == b
    assert ref.INITIAL_CHECKPOINT is zero.INITIAL_CHECKPOINT is None
    assert len({old.RUN_NAME, ref.RUN_NAME, zero.RUN_NAME}) == 3
    assert ref.CONFIG.summary()['simulations_per_iteration'] == 1400
    assert ref.CONFIG.summary()['validation_simulations_per_check'] == 448


@pytest.mark.parametrize('size', range(3, 10))
def test_one_pair_draw_exact_score_mirroring_and_physical_symmetry(tmp_path, monkeypatch, size):
    config = replace(ref.CONFIG.to_backend(tmp_path), n_lasers=size)
    torch.manual_seed(7)
    policy = build_policy(config)
    baseline = build_policy(old.CONFIG.to_backend(tmp_path))
    assert sum(p.numel() for p in policy.parameters()) == sum(p.numel() for p in baseline.parameters())
    features = engine.encode_context(np.linspace(0., 2., size)[None],
                                     np.linspace(-2., 2., size)[None], policy)
    edges, pairs = size*(size-1), size*(size-1)//2
    dense = features.new_ones((1, 8, edges))
    fractions = np.full((1, 8), .5)
    view = DensityConditionedPolicy(policy, fractions)
    shapes = []
    original = torch.distributions.Normal.sample
    def capture(normal, *args, **kwargs):
        shapes.append(tuple(normal.loc.shape))
        return original(normal, *args, **kwargs)
    monkeypatch.setattr(torch.distributions.Normal, 'sample', capture)
    actions, budget, score = sample_annealed_candidates(view, features, dense, mirrored=True)
    assert shapes == [(1, 8, 2*pairs), (1, 8)]
    check_shared_actions(actions, dense, size)
    canonical, _, _ = pair_layout(size)
    columns = torch.cat((canonical, canonical + edges))
    repeated = features.repeat_interleave(8, dim=0)
    mean, log_std, _, pooled = view.parameters_for_edges(repeated,
        topology_masks=dense.reshape(8, edges), return_pooled=True)
    mean = shared_pair_parameters(mean.reshape(1, 8, 2*edges)[..., columns])
    std = log_std.reshape(1, 8, 2*edges)[..., columns].exp()
    extra = features.new_zeros((1, 8, 1))
    extra[:, :2] = config.phase_exploration_start_std/math.sqrt(2.)
    std = shared_pair_parameters(torch.cat((std[..., :pairs],
                                          (std[..., pairs:]**2+extra**2).sqrt()), dim=-1))
    bm, bs = view.budget_parameters(repeated, pooled_states=pooled)
    bm = shared_pair_parameters(bm.reshape(1, 8))
    bs = shared_pair_parameters(bs.reshape(1, 8).exp())
    reduced = actions[..., columns]
    expected = torch.distributions.Normal(mean, std).log_prob(reduced).sum(-1)
    expected += torch.distributions.Normal(bm, bs).log_prob(budget)
    torch.testing.assert_close(score, expected)
    torch.testing.assert_close(reduced[:, 1::2], 2*mean[:, 1::2]-reduced[:, ::2])
    gradients = torch.autograd.grad(score[0, 0], policy.edge_output.bias)[0]
    assert torch.isfinite(gradients).all() and gradients[:2].abs().sum() > 0
    assert not actions.requires_grad
    # An accidental projection would hide independent-direction sampling errors.
    from rl.rl_api import symmetric_topologies
    def no_projection(*args, **kwargs):
        raise AssertionError('Pair mode must not average directed actions')
    monkeypatch.setattr(symmetric_topologies, 'reciprocal_actions', no_projection)
    masks, (kappa, phase, counts, _, shortfall) = decode_density_actions(actions, budget, fractions, config)
    for values in (kappa, phase):
        np.testing.assert_array_equal(values, values.swapaxes(-1, -2))
    np.testing.assert_array_equal(counts, 2*pair_counts(size, fractions))
    np.testing.assert_allclose(kappa.sum((-1, -2)), budget.detach().exp(), rtol=2e-6)
    assert not shortfall.any()


def test_reference_is_common_pair_offset_and_no_target_difference_is_added(tmp_path):
    c = ref.CONFIG.to_backend(tmp_path)
    policy = build_policy(c)
    zero_policy = build_policy(zero.CONFIG.to_backend(tmp_path))
    zero_policy.load_state_dict(policy.state_dict())
    features = engine.encode_context([[0., .3, 1.7]], [[-2., -.7, 2.]], policy)
    masks = features.new_ones((1, 6))
    a = policy.parameters_for_edges(features, topology_masks=masks)[0]
    b = zero_policy.parameters_for_edges(features, topology_masks=masks)[0]
    torch.testing.assert_close(a[:, :6], b[:, :6])
    torch.testing.assert_close(a[:, 6:]-b[:, 6:], delay_phase_offsets(features, c).expand(-1, 6))
    with torch.no_grad():
        zero_policy.edge_output.weight.zero_()
        zero_policy.edge_output.bias.zero_()
    mean = zero_policy.parameters_for_edges(features, topology_masks=masks)[0]
    assert torch.equal(mean, torch.zeros_like(mean))


@pytest.mark.parametrize('recipe', [ref, zero], ids=['laser1', 'zero'])
def test_training_reload_and_stochastic_inspection(tmp_path, fake_physics, recipe):
    config = replace(recipe.CONFIG, sizes=(3, 4),
        sampling=replace(recipe.CONFIG.sampling, pairs_per_size=1, pairs_per_job=1),
        optimization=replace(recipe.CONFIG.optimization, iterations=2, workers=1),
        schedules=replace(recipe.CONFIG.schedules, ordinary_end=2, wide_end=2),
        validation=replace(recipe.CONFIG.validation, pairs_per_size=5,
                           plot_use_tex=False, notebook_display=False))
    result = Experiment('pair-test', config=config, output_directory=tmp_path/'run').train(
        live_plot=True, verify_reference=False)
    payload = torch.load(result.checkpoint('final'), weights_only=False)
    assert payload['iteration'] == 2
    assert 'initialization' not in payload['sampling_recipe']
    assert 'no averaging' in payload['sampling_recipe']['symmetric_values']
    for kappa, phase, _ in fake_physics:
        for values in (kappa, phase):
            np.testing.assert_array_equal(values, values.swapaxes(-1, -2))
    model = CouplingModel.load(result.checkpoint('final'), device='cpu')
    assert model.backend_config.reciprocal_pair_actions
    # First validation saves the best checkpoint while exploration is active.
    model = CouplingModel.load(result.checkpoint('best'), device='cpu')
    assert model.backend_config.exploration_std_scale > 0.
    context = Context([0., .2, -.8, .7], [-2., -.3, .5, 2.])
    for continuous_mean in (False, True):
        designs = model.propose(context, density_fraction=None, count=11,
                                continuous_actions="mean" if continuous_mean else "sample", seed=71)
        for values in (designs.coupling_strengths_per_ns, designs.coupling_phases_rad):
            np.testing.assert_array_equal(values, values.swapaxes(-1, -2))
    assert (result.directory/recipe.CONFIG.validation.progress_filename).is_file()


def test_launcher_resources_and_distinct_jobs():
    script = Path(__file__).resolve().parents[3]/'scripts/submit_reciprocal_pair_density_tests.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    result = subprocess.run(['bash', str(script), '--dry-run'], check=True, capture_output=True, text=True)
    commands = [line for line in result.stdout.splitlines() if line.startswith('sbatch ')]
    assert len(commands) == 2
    for line, name in zip(commands, ('density_pair_ref', 'density_pair_zero')):
        assert '--job-name='+name in line
        for token in ('--cpus-per-task=32', '--time=7-00:00:00', '--workers 31',
                      '--pairs-per-job 5', '--actions-per-job 8'):
            assert token in line
