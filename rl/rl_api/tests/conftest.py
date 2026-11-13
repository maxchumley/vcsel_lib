"""Fixtures only use throwaway models and files under this new package."""
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def fake_physics(monkeypatch):
    """Action-dependent finite trajectories; no real VCSEL simulation."""
    from rl.rl_api._compat import core
    calls = []

    def simulate(kappa, phi, config, **options):
        calls.append((kappa.copy(), phi.copy(), options))
        n, m = kappa.shape[:2]
        times = np.linspace(0, config.simulation_time_seconds, 64)
        states = np.zeros((n, 3*m, len(times)))
        states[:, 1::3] = (1 + .001*kappa.sum(axis=2))[:, :, None]
        phases = -phi[:, :, 0] + .01*kappa.sum(axis=2)
        states[:, 2::3] = phases[:, :, None] + .003*np.sin(np.arange(64))[None, None]
        frequencies = np.zeros((n, m, len(times)))
        history = np.zeros((n, m, 20))
        return times, states, frequencies, history

    monkeypatch.setattr(core, "simulate_with_vcsel", simulate)
    # API evaluation now calls VCSEL directly; keep this fixture's fake physics
    # at that boundary too. Physical equivalence is checked separately.
    from rl.rl_api import evaluation, simulation
    def evaluate(kappa, phi, config, **options):
        options.pop('seed',None)
        options['detuning_distribution_ghz']=options.pop('detunings_ghz')
        return simulate(kappa,phi,config,**options)
    monkeypatch.setattr(evaluation, 'simulate', evaluate)
    monkeypatch.setattr(simulation, 'simulate', evaluate)
    return calls
