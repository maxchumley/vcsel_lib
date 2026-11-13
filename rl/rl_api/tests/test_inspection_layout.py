"""The inspector uses the original layout and active settings in one cell."""
import numpy as np
import matplotlib.pyplot as plt
import pytest

from rl.rl_api import PlotOptions, EvaluationBatch
from rl.rl_api.experiments import inspect_current as run
from rl.rl_api.tests.test_workflow import make_model


def test_selected_and_ensemble_figures_in_order_without_resimulation_or_changed_globals(tmp_path, fake_physics):
    from rl import conditional_coupling_run_variable_m as original
    model = make_model(tmp_path)
    results = model.propose(target_phases_rad=[0., -.4, 1.2],
        detunings_ghz=[-.5, 0., .5], count=3, seed=4).evaluate()
    original_directory = original.figure_directory
    original_show, original_save = plt.show, plt.savefig
    original_style = dict(plt.rcParams)
    before = len(fake_physics)
    figures = results.plot(show=False, options=PlotOptions(layout='inspection',
        dpi=45, n_theta=51, max_time_points=16))
    assert list(figures) == ['design', 'dynamics', 'ensemble_couplings', 'ensemble']
    assert len(plt.get_fignums()) == 4
    assert len(fake_physics) == before
    assert original.figure_directory == original_directory
    assert plt.show is original_show and plt.savefig is original_save
    assert dict(plt.rcParams) == original_style
    assert len(list((tmp_path/'figures').glob('*.png'))) == 4
    # The old dynamics figure contains the far field and its final slice.
    assert any('far-field' in axis.get_title().lower() for axis in figures['dynamics'].axes)
    for figure in figures.values():
        plt.close(figure)


def test_ensemble_averages_use_zeros_for_absent_strengths_and_active_circular_phases(tmp_path, fake_physics):
    from rl.rl_api.plotting.inspection import average_coupling_matrices, plot_ensemble_couplings
    model = make_model(tmp_path)
    results = model.propose(target_phases_rad=[0., -.4, 1.2],
        detunings_ghz=[-.5, 0., .5], count=3, seed=4).evaluate(noise_realizations=2)
    kappa = np.zeros((3, 3, 3))
    phase = np.zeros_like(kappa)
    kappa[:2, 1, 0] = [2., 4.]
    phase[:2, 1, 0] = np.deg2rad([179., -179.])
    kappa[0, 2, 0] = 6.
    phase[:2, 2, 0] = [np.pi/2, -np.pi]  # Second link is absent: ignore its phase.
    kappa[:2, 2, 1] = 1.
    phase[:2, 2, 1] = [np.pi/2, -np.pi/2]  # Undefined circular mean.
    kappa[2] = 100.  # Invalid candidate must not enter either mean.
    results.designs.raw['kappa_per_ns'] = kappa
    results.designs.raw['phi_p_rad'] = phase
    valid = np.array([True, True, False])
    mean, angle = average_coupling_matrices(results, valid)
    assert mean[1, 0] == 3.
    assert mean[2, 0] == 3.
    assert abs(angle[1, 0]) == pytest.approx(np.pi)
    assert angle[2, 0] == pytest.approx(np.pi/2)
    assert angle.mask[0, 1] and angle.mask[2, 1]
    assert mean[2, 1] == 1.
    np.testing.assert_array_equal(np.diag(mean), 0.)
    figure = plot_ensemble_couplings(results, valid)
    # Internal [receiver,source] arrays become [source,receiver] in the plots.
    assert figure.axes[0].images[0].get_array()[0, 1] == 3.
    assert figure.axes[1].images[0].get_array()[0, 2] == pytest.approx(np.pi/2)
    np.testing.assert_array_equal(results.designs.coupling_strengths_per_ns, kappa)
    np.testing.assert_array_equal(results.designs.coupling_phases_rad, phase)
    plt.close(figure)
    with pytest.raises(ValueError, match='valid topology'):
        average_coupling_matrices(results, np.zeros(3, bool))


def test_phase_ring_keeps_noise_distribution_wraps_and_uses_single_case_markers(tmp_path, fake_physics):
    from rl.rl_api.plotting.inspection import plot_ensemble_phase_distribution
    model = make_model(tmp_path)
    target = np.array([0., np.pi/2, -np.pi/2])
    results = model.propose(target_phases_rad=target,
        detunings_ghz=[-.5, 0., .5], count=3, seed=4).evaluate(noise_realizations=2)
    pair = np.deg2rad([[0., 179., -179.], [0., -179., 179.]])
    # The invalid topology has phases at 90 degrees; it must not enter the ring.
    results.aligned_achieved_phases = np.vstack((pair, pair, np.full((2, 3), np.pi/2)))
    figure, axis = plt.subplots()
    ring = plot_ensemble_phase_distribution(axis, results, np.array([True, True, False]), 5.)
    assert ring.get_array().sum() == pytest.approx(100.)
    assert np.count_nonzero(ring.get_array()) == 3  # 0, +179 and -179 degrees.
    np.testing.assert_allclose(np.sort(ring.get_array()[ring.get_array() > 0]), [100/3]*3)
    markers = [artist for artist in axis.collections if artist is not ring]
    np.testing.assert_allclose(markers[0].get_offsets(), np.c_[np.cos(target), np.sin(target)], atol=1e-12)
    np.testing.assert_allclose(markers[1].get_offsets(), [[1., 0.], [-1., 0.], [-1., 0.]], atol=1e-12)
    assert {t.get_text() for t in axis.get_legend().get_texts()} == {
        r'max $\pm 5^\circ$', 'target', 'ensemble mean'}
    plt.close(figure)


def test_large_ensemble_matrix_ticks_match_single_case_convention(tmp_path, fake_physics):
    from rl.rl_api.plotting.inspection import plot_ensemble_couplings
    model = make_model(tmp_path)
    results = model.propose(target_phases_rad=np.zeros(30),
        detunings_ghz=np.linspace(-1., 1., 30), count=2, seed=4).evaluate()
    figure = plot_ensemble_couplings(results, results.valid_topologies)
    for axis in figure.axes[:2]:
        assert len(axis.get_xticks()) == len(axis.get_yticks()) == 6
        assert axis.get_xticklabels()[0].get_text() == '1'
        assert axis.get_xticklabels()[-1].get_text() == '30'
        np.testing.assert_allclose(axis.get_xlim(), [-.5, 29.5])
        np.testing.assert_allclose(axis.get_ylim(), [29.5, -.5])
    assert 'Settled phase comparison' in figure.axes[2].get_title()
    plt.close(figure)


@pytest.mark.parametrize('layout', ['inspection', 'separate'])
def test_select_none_skips_selection_and_selected_renderers(tmp_path, monkeypatch, fake_physics, layout):
    from rl.rl_api.plotting import inspection
    model = make_model(tmp_path)
    results = model.propose(target_phases_rad=[0., -.4, 1.2],
        detunings_ghz=[-.5, 0., .5], count=3, seed=4).evaluate(noise_realizations=2)
    before = len(fake_physics)
    def forbidden(*args, **kwargs):
        raise AssertionError('Ensemble-only mode must not select or render one candidate')
    monkeypatch.setattr(results, 'select', forbidden)
    monkeypatch.setattr(inspection, 'selected_inspection_figures', forbidden)
    figures = results.plot(select=None, show=False,
        options=PlotOptions(layout=layout, dpi=40, n_theta=51, max_time_points=16))
    assert list(figures) == ['ensemble_couplings', 'ensemble']
    assert len(fake_physics) == before
    assert len(list((tmp_path/'figures').glob('*.png'))) == 2
    for figure in figures.values():
        plt.close(figure)


@pytest.mark.parametrize('explicit_override', [False, True])
def test_inspect_none_setting_and_override_return_ensemble_only(tmp_path, monkeypatch, fake_physics, explicit_override):
    model = make_model(tmp_path)
    monkeypatch.setattr(run, 'load_model', lambda *args, **kwargs:model)
    settings = dict(N_LASERS=3, N_CLUSTERS=1, MANUAL_TARGET_PHASES_PI=None,
        DETUNING_MODE='equal', DETUNING_SPAN_GHZ=1., TOPOLOGY_MATRIX_SOURCE_RECEIVER=None,
        RUN_VALIDATION=True, PLOT_TOPOLOGY_ENSEMBLE=True, FIGURE_DIRECTORY=tmp_path/'figures',
        ENSEMBLE_SELECTED_TOPOLOGY='best' if explicit_override else None,
        PLOT_STYLE={**run.PLOT_STYLE, 'text.usetex':False, 'figure.dpi':40})
    for key, value in settings.items():
        monkeypatch.setattr(run, key, value)
    def forbidden(*args, **kwargs):
        raise AssertionError('No selected candidate is requested')
    monkeypatch.setattr(EvaluationBatch, 'select', forbidden)
    options = {'select':None} if explicit_override else {}
    result = run.inspect(count=3, show=False, **options)
    assert result['chosen'] is None
    assert list(result['figures']) == ['ensemble_couplings', 'ensemble']
    assert len(fake_physics) == 1
    for figure in result['figures'].values():
        plt.close(figure)


@pytest.mark.parametrize('validation,ensemble', [(False, True), (True, False)])
def test_ensemble_only_requires_ensemble_validation(monkeypatch, validation, ensemble):
    monkeypatch.setattr(run, 'ENSEMBLE_SELECTED_TOPOLOGY', None)
    monkeypatch.setattr(run, 'RUN_VALIDATION', validation)
    monkeypatch.setattr(run, 'PLOT_TOPOLOGY_ENSEMBLE', ensemble)
    with pytest.raises(ValueError, match='Ensemble-only inspection requires'):
        run.inspect(show=False)


def test_disabling_ensemble_keeps_only_selected_figures(tmp_path, fake_physics):
    model = make_model(tmp_path)
    results = model.propose(target_phases_rad=[0., -.4, 1.2],
        detunings_ghz=[-.5, 0., .5], count=3, seed=4).evaluate()
    figures = results.plot(show=False,
        options=PlotOptions(layout='inspection', show_ensemble=False, show_far_field=False, dpi=40))
    assert list(figures) == ['design', 'dynamics']
    for figure in figures.values():
        plt.close(figure)
    with pytest.raises(ValueError, match='show_ensemble=True'):
        results.plot(select=None, show=False, options=PlotOptions(layout='inspection', show_ensemble=False))


def test_old_phase_cluster_and_detuning_conventions(monkeypatch):
    monkeypatch.setattr(run, 'N_LASERS', 4)
    monkeypatch.setattr(run, 'N_CLUSTERS', 2)
    monkeypatch.setattr(run, 'MANUAL_TARGET_PHASES_PI', None)
    monkeypatch.setattr(run, 'DETUNING_MODE', 'equal')
    monkeypatch.setattr(run, 'DETUNING_SPAN_GHZ', 3.)
    context = run.make_context()
    np.testing.assert_array_equal(context.target_phases_rad, [0., 0., np.pi, np.pi])
    np.testing.assert_allclose(context.detunings_ghz, [-1.5, -.5, .5, 1.5])
    monkeypatch.setattr(run, 'N_LASERS', 3)
    monkeypatch.setattr(run, 'MANUAL_TARGET_PHASES_PI', [0., -1/np.pi, 2/np.pi])
    monkeypatch.setattr(run, 'DETUNING_MODE', 'manual')
    monkeypatch.setattr(run, 'MANUAL_DETUNINGS_GHZ', [0., 1., 3.])
    context = run.make_context()
    np.testing.assert_array_equal(context.target_phases_rad, [0., -1., 2.])
    np.testing.assert_allclose(context.detunings_ghz, np.array([0., 1., 3.])-4/3)


def test_original_topology_convention_and_ensemble_mean_action_override(tmp_path, monkeypatch, fake_physics):
    model = make_model(tmp_path)
    monkeypatch.setattr(run.CouplingModel, 'load', lambda *args, **kwargs: model)
    monkeypatch.setattr(run, 'N_LASERS', 3)
    monkeypatch.setattr(run, 'MANUAL_TARGET_PHASES_PI', [0., 0., 0.])
    monkeypatch.setattr(run, 'MANUAL_DETUNINGS_GHZ', [-1., 0., 1.])
    source_receiver = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    monkeypatch.setattr(run, 'TOPOLOGY_MATRIX_SOURCE_RECEIVER', source_receiver)
    monkeypatch.setattr(run, 'SAMPLE_CONTINUOUS_ACTIONS', 1)
    monkeypatch.setattr(run, 'ENSEMBLE_TOPOLOGY_ONLY', True)
    monkeypatch.setattr(run, 'PLOT_TOPOLOGY_ENSEMBLE', True)
    monkeypatch.setattr(run, 'OVERRIDE_COUPLING_RAMP', True)
    monkeypatch.setattr(run, 'COUPLING_RAMP_RISE_DELAYS', 123.)
    proposal = model.propose
    calls = []
    def propose(*args, **kwargs):
        calls.append(kwargs)
        return proposal(*args, **kwargs)
    monkeypatch.setattr(model, 'propose', propose)
    monkeypatch.setattr(EvaluationBatch, 'plot', lambda *args, **kwargs: {})
    inspection = run.inspect(count=2, show=False)
    assert calls[0]['continuous_actions'] == 'mean'
    np.testing.assert_array_equal(calls[0]['topology'].as_source_receiver(), source_receiver)
    assert inspection['results'].config.coupling_ramp_rise_delays == 123.
    assert len(fake_physics) == 1


def test_validation_off_skips_all_simulations(tmp_path, monkeypatch, fake_physics):
    model = make_model(tmp_path)
    monkeypatch.setattr(run.CouplingModel, 'load', lambda *args, **kwargs: model)
    monkeypatch.setattr(run, 'RUN_VALIDATION', False)
    monkeypatch.setattr(run, 'ENSEMBLE_SELECTED_TOPOLOGY', 'best')
    monkeypatch.setattr(run, 'PLOT_STYLE', {**run.PLOT_STYLE, 'text.usetex': False, 'figure.dpi': 45})
    inspection = run.inspect(count=2, show=False)
    assert inspection['results'] is None
    assert set(inspection['figures']) == {'design'}
    assert fake_physics == []
    for figure in inspection['figures'].values():
        plt.close(figure)


@pytest.mark.parametrize('budget', [0., 30.])
def test_manual_budget_preserves_relative_designs_and_original_proposals(tmp_path, budget):
    model = make_model(tmp_path)
    designs = model.propose(target_phases_rad=[0., -.4, 1.2],
        detunings_ghz=[-.5, 0., .5], count=3, seed=4)
    original = designs.coupling_strengths_per_ns.copy()
    assert run.apply_manual_budget(designs, None) is designs
    adjusted = run.apply_manual_budget(designs, budget)
    np.testing.assert_allclose(adjusted.budgets_per_ns, budget)
    expected = original * (budget/original.sum(axis=(-2, -1)))[:, None, None]
    np.testing.assert_allclose(adjusted.coupling_strengths_per_ns, expected)
    np.testing.assert_array_equal(designs.coupling_strengths_per_ns, original)
    np.testing.assert_array_equal(adjusted.coupling_phases_rad, designs.coupling_phases_rad)
    np.testing.assert_array_equal(adjusted.topologies, designs.topologies)
    np.testing.assert_array_equal(adjusted.active_link_counts, designs.active_link_counts)
    np.testing.assert_allclose(adjusted.raw['raw_total_coupling_per_ns'], budget)
    np.testing.assert_array_equal(adjusted.raw['redistribution_shortfall_per_ns'], 0.)
    adjusted.raw['kappa_per_ns'][:] = 0.
    with pytest.raises(ValueError, match='positive budget'):
        run.apply_manual_budget(adjusted, 30.)


@pytest.mark.parametrize('budget', [-1., np.nan, np.inf, [1., 2.]])
def test_manual_budget_rejects_invalid_totals(budget):
    with pytest.raises(ValueError, match='MANUAL_BUDGET_PER_NS'):
        run.apply_manual_budget(None, budget)


@pytest.mark.parametrize('validate', [False, True])
def test_inspect_manual_budget_reaches_plots_and_physical_evaluation(
        tmp_path, monkeypatch, fake_physics, validate):
    from rl.rl_api.data import DesignBatch
    model = make_model(tmp_path)
    monkeypatch.setattr(run, 'load_model', lambda *args, **kwargs: model)
    settings = dict(N_LASERS=3, N_CLUSTERS=1, MANUAL_TARGET_PHASES_PI=None,
        DETUNING_MODE='equal', DETUNING_SPAN_GHZ=1., TOPOLOGY_MATRIX_SOURCE_RECEIVER=None,
        RUN_VALIDATION=validate, PLOT_TOPOLOGY_ENSEMBLE=validate,
        ENSEMBLE_SELECTED_TOPOLOGY=None if validate else 'best',
        MANUAL_BUDGET_PER_NS=42.)
    for key, value in settings.items():
        monkeypatch.setattr(run, key, value)
    plotted = []
    def plot_designs(self, **options):
        plotted.append(self.budgets_per_ns.copy())
        return {}
    def plot_results(self, **options):
        plotted.append(self.designs.budgets_per_ns.copy())
        return {}
    monkeypatch.setattr(DesignBatch, 'plot', plot_designs)
    monkeypatch.setattr(EvaluationBatch, 'plot', plot_results)
    inspection = run.inspect(count=3, show=False)
    np.testing.assert_allclose(inspection['designs'].budgets_per_ns, 42.)
    assert len(plotted) == 1
    np.testing.assert_allclose(plotted[0], 42.)
    assert len(fake_physics) == int(validate)
    if validate:
        np.testing.assert_allclose(fake_physics[0][0].sum(axis=(-2, -1)), 42.)
        np.testing.assert_array_equal(fake_physics[0][1], inspection['designs'].coupling_phases_rad)
