"""Physical evaluation, both rewards, and current inspector selection rules."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import torch

from ._compat import core, engine
from .simulation import simulate, simulate_phase_model_candidates
from .phase_reference import decode_samples, uses_root_reference
from .root_context import condition_on_roots


@dataclass
class EvaluatedDesign:
    """One selected design with access to its existing evaluated trajectory."""
    results: "EvaluationBatch"
    index: int

    @property
    def coupling_strengths_per_ns(self):
        return self.results.designs.coupling_strengths_per_ns[self.index]

    @property
    def coupling_phases_rad(self):
        return self.results.designs.coupling_phases_rad[self.index]

    @property
    def metrics(self):
        """Explicitly distinguish tail RMS from circular-mean maximum error."""
        r, i = self.results, self.index
        return dict(cosine_reward=float(r.cosine_reward[i]), rms_reward=float(r.rms_reward[i]),
            maximum_phase_rms_deg=float(r.maximum_phase_rms_deg[i]),
            maximum_mean_phase_error_deg=float(r.maximum_mean_phase_error_deg[i]),
            phase_rms_success=bool(r.phase_rms_success[i]),
            budget_per_ns=float(r.designs.budgets_per_ns[i]),
            active_links=int(r.designs.active_link_counts[i]))

    def plot(self, **options):
        """Plot this selection without running another simulation."""
        return self.results.plot(select=self.index, **options)

    def connectivity(self, minimum_strength_per_ns=0.0):
        """Return the original connectivity analysis for this physical matrix."""
        from rl.inspect_saved_coupling_model_variable_m_bernoulli_sparse import inspect_coupling_connectivity
        return inspect_coupling_connectivity(self.coupling_strengths_per_ns, minimum_strength_per_ns)


@dataclass
class EvaluationBatch:
    """All candidate/noise realizations; selection and plotting reuse this data."""
    designs: object
    config: object
    simulation: tuple
    noise_realizations: int
    cosine_reward_by_realization: np.ndarray
    rms_reward_by_realization: np.ndarray
    worst_rms_by_realization_deg: np.ndarray
    rms_success_by_realization: np.ndarray
    raw_achieved_phases: np.ndarray
    aligned_achieved_phases: np.ndarray
    orientations: np.ndarray
    valid_topologies: np.ndarray

    @property
    def cosine_reward(self):
        return self.cosine_reward_by_realization.mean(axis=1)

    @property
    def rms_reward(self):
        return self.rms_reward_by_realization.mean(axis=1)

    @property
    def maximum_phase_rms_deg(self):
        """Worst nonreference laser, then worst noise realization per design."""
        return self.worst_rms_by_realization_deg.max(axis=1)

    @property
    def phase_rms_success(self):
        """All noise realizations must satisfy the backend's 3-degree RMS check."""
        return self.rms_success_by_realization.all(axis=1)

    @property
    def maximum_mean_phase_error_deg(self):
        """Match the old inspector's circular-mean error, not its RMS metric."""
        n, r, m = len(self.designs), self.noise_realizations, self.config.n_lasers
        phases = self.aligned_achieved_phases.reshape(n, r, m)
        aligned = np.angle(np.mean(np.exp(1j*phases), axis=1))
        target = core.make_relative_to_laser_1(self.designs.context.target_phases_rad)
        error = np.angle(np.exp(1j*(aligned-target)))
        return np.rad2deg(np.max(np.abs(error[:, 1:]), axis=1))

    def select(self, mode="best"):
        """best uses cosine, matching current inspection; best_rms is explicit.

        sparsest/min_budget have no hidden accuracy cutoff. Their tie-breaking
        is delegated to the existing validated inspector selection helper.
        """
        if isinstance(mode, (int, np.integer)):
            if not 0 <= mode < len(self.designs):
                raise IndexError("Candidate index is outside this batch")
            index = int(mode)
        elif mode == "best_rms":
            valid = self.valid_topologies & np.isfinite(self.rms_reward)
            if not valid.any():
                raise ValueError("No finite candidate is available")
            index = int(np.argmax(np.where(valid, self.rms_reward, -np.inf)))
        else:
            from rl.inspect_saved_coupling_model_variable_m_bernoulli_sparse import select_ensemble_topology
            index = select_ensemble_topology(self.cosine_reward, self.designs.active_link_counts,
                mode, valid_topologies=self.valid_topologies,
                coupling_budgets=self.designs.budgets_per_ns)
        return EvaluatedDesign(self, index)

    def plot(self, *, select="best", output_directory=None, show=True, options=None):
        """Plot existing data; select=None shows only ensemble matrices/trajectories."""
        from .plotting import plot_evaluation
        return plot_evaluation(self, select=select, output_directory=output_directory,
                               show=show, options=options)


def evaluate_designs(designs, *, duration_ns=None, noise_amplitude=None,
                     noise_realizations=1, seed=None, progress=False,
                     smooth_frequencies=True):
    """Simulate each candidate/noise realization once; never select implicitly."""
    if not isinstance(noise_realizations, int) or noise_realizations < 1:
        raise ValueError("noise_realizations must be a positive integer")
    settings = {}
    if duration_ns is not None:
        if not np.isfinite(duration_ns) or duration_ns <= 0:
            raise ValueError("duration_ns must be positive and finite")
        settings["simulation_time_seconds"] = duration_ns*1e-9
    if noise_amplitude is not None:
        if not np.isfinite(noise_amplitude) or noise_amplitude < 0:
            raise ValueError("noise_amplitude must be finite and nonnegative")
        settings["noise_amplitude"] = noise_amplitude
    config = replace(designs.config, **settings)
    r, n = noise_realizations, len(designs)
    kappa = np.repeat(designs.coupling_strengths_per_ns, r, axis=0)
    phi = np.repeat(designs.coupling_phases_rad, r, axis=0)
    target = np.repeat(designs.context.target_phases_rad[None], n*r, axis=0)
    detuning = np.repeat(designs.context.detunings_ghz[None], n*r, axis=0)
    result = simulate(kappa, phi, config, detunings_ghz=detuning, seed=seed,
                      progress=progress, smooth_frequencies=smooth_frequencies)
    cosine = core.calculate_phase_reward(result[0], result[1], target, config)
    rms = core.calculate_phase_rms_metrics(result[0], result[1], target, config)
    finite = np.all(np.isfinite(result[1]), axis=(1, 2)) & np.all(np.isfinite(result[2]), axis=(1, 2))
    return EvaluationBatch(designs, config, result, r, cosine[0].reshape(n,r),
        rms["phase_rms_reward"].reshape(n,r), rms["worst_phase_rms_error_deg"].reshape(n,r),
        rms["phase_rms_success"].reshape(n,r), cosine[3], cosine[4], cosine[5],
        finite.reshape(n,r).all(axis=1))


def evaluate_phase_model_policy(policy, targets, detunings, config, *,
                                seed=0, simulation_pool=None):
    """Keep the held-out procedure while passing the native phase convention.

    The historical evaluator's workers omit phase_model. Sampling, mean
    actions, decoding and metric definitions here match that evaluator.
    """
    targets, detunings = core.sort_by_detuning(targets, detunings)
    features = engine.encode_context(targets, detunings, policy)
    policy.eval()
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        mask_sampler = engine.sample_stratified_rooted_topologies
        if getattr(config, 'topology_random_reachable', False):
            from .sampling import sample_random_reachable_topologies
            mask_sampler = sample_random_reachable_topologies
        root_options = dict(return_roots=True) if uses_root_reference(config) else {}
        masks = mask_sampler(
            1, len(targets), config.n_lasers, np.random.default_rng(seed),
            features.device, min_fraction=config.training_topology_min_fraction,
            fixed_star=config.topology_star_backbone, **root_options,
        )
        masks, roots = masks if root_options else (masks, None)
        masks = masks.reshape(len(targets), 1, config.n_lasers * (config.n_lasers - 1))
        roots = None if roots is None else roots.reshape(len(targets), 1)
        actions, budget_actions, _ = engine.sample_topology_conditioned_candidates(
            condition_on_roots(policy, roots), features, masks, continuous_mean=True)
    kappa, phi, counts, rho, shortfall = decode_samples(
        actions, masks, config, budget_actions=budget_actions,
        detunings_ghz=detunings, root_indices=roots)
    simulation = simulate_phase_model_candidates(targets, kappa, phi, config,
        detunings=detunings, simulation_pool=simulation_pool)
    phase = simulation["phase_reward"]
    band_ids = engine.topology_density_band_ids(counts[:, 0], config.n_lasers)

    def by_density(values):
        return {label: float(values[band_ids == index].mean())
                if np.any(band_ids == index) else float("nan")
                for index, label in enumerate(engine.DENSITY_BANDS)}

    rms_metrics = {}
    if config.phase_reward_mode == "wrapped_rms":
        rms_metrics = dict(
            mean_phase_rms_error_deg=float(simulation["mean_phase_rms_error_deg"].mean()),
            worst_phase_rms_error_deg=float(simulation["worst_phase_rms_error_deg"].mean()),
            phase_rms_success_fraction=float(simulation["phase_rms_success"].mean()),
            rms_reward_by_density=by_density(simulation["phase_rms_reward"]))
    return dict(phase_reward=float(phase.mean()),
        training_reward=float(simulation["phase_rms_reward"].mean())
            if rms_metrics else float(phase.mean()),
        dense_reference_phase_reward=float("nan"), feasible_fraction=0.0,
        rooted_fraction=1.0, retained_link_fraction=float(rho.mean()),
        normalized_coupling_budget=float(engine.relative_coupling_budget(kappa, config).mean()),
        coupling_budget_per_laser=float(kappa.sum(axis=(-2, -1)).mean() / config.n_lasers),
        mean_redistribution_shortfall_per_ns=float(shortfall.mean()),
        phase_reward_by_density=by_density(phase), **rms_metrics)
