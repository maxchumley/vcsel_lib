"""VCSEL boundary and scoped noise seeding; no alternative simulator."""
from contextlib import contextmanager
import numpy as np

from ._compat import core


@contextmanager
def numpy_seed(seed):
    """Match an explicit NumPy seed while restoring the caller's RNG afterward.

    The simulator currently uses NumPy's process-global noise RNG. This scope
    is for synchronous evaluation, not concurrent calls from multiple threads.
    With seed=None, preserve normal advancing-RNG behavior.
    """
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def simulate(kappa, phi, config, *, detunings_ghz, seed=None,
             progress=False, smooth_frequencies=True):
    """Integrate with vcsel_lib's API, retaining the original numerical settings."""
    with numpy_seed(seed):
        vcsel, history, initial_frequency_ghz = prepare_vcsel(
            kappa, phi, config, detunings_ghz=detunings_ghz)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            time, states, frequencies = vcsel.integrate(history, nd=vcsel.nd,
                progress=progress, max_iter=1, smooth_freqs=smooth_frequencies)
        return time, states, frequencies, initial_frequency_ghz


def prepare_vcsel(kappa_per_ns, phi_p_rad, config, *, detunings_ghz):
    """Return (VCSEL instance, history, initial frequency) without integrating.

    Arrays use (cases,M,M), with [receiver,source] indexing. The returned
    object's .phys and .nd expose the physical and scaled parameter arrays.
    VCSEL supplies scaling, the shared cosine ramp, and free-running history.
    There is no separate RL rate equation or integrator.
    """
    kappa = np.asarray(kappa_per_ns, dtype=float)
    phi = np.asarray(phi_p_rad, dtype=float)
    if kappa.shape != phi.shape or kappa.ndim != 3 or kappa.shape[1:] != (config.n_lasers, config.n_lasers):
        raise ValueError("Coupling strengths/phases must have matching (cases,M,M) shapes")
    # Reuse the existing mapping from named RL settings to VCSEL's SI keys.
    physical = core.make_vcsel_physical_parameters(config, detuning_distribution_ghz=detunings_ghz)
    physical["phase_model"] = getattr(config, "phase_model", "chumley_2026")
    kappa_per_second = kappa*1e9
    physical["kappa_c_mat"] = kappa_per_second
    physical["phi_p_mat"] = phi
    VCSEL = core._load_vcsel_class()
    vcsel = VCSEL(physical)
    nd = vcsel.scale_params()
    # Match the original multiplication order exactly; scale_params uses /gamma.
    nd["kappa"] = kappa_per_second*config.photon_lifetime_seconds
    nd["kappa_case_dependent"] = True
    times = np.arange(nd["steps"])*config.time_step_seconds
    nd["kappa_ramp"] = VCSEL.cosine_ramp(times,
        t_start=config.coupling_ramp_start_delays*config.delay_seconds,
        rise_10_90=config.coupling_ramp_rise_delays*config.delay_seconds,
        kappa_initial=0., kappa_final=1.)
    history, initial_frequency_ghz, _, _ = vcsel.generate_history(nd, shape="FR", n_cases=len(kappa))
    return vcsel, history, initial_frequency_ghz


split_batch = core.split_simulation_batch
initialize_worker = core._initialize_simulation_worker


def simulate_phase_model_chunk(work_item):
    """Vectorized worker using the API's native VCSEL parameter pass-through."""
    chunk, start, stop, targets, kappa, phi, detuning, config = work_item
    shape = kappa.shape[:2]
    repeated_targets = np.repeat(targets, shape[1], axis=0)
    repeated_detuning = np.repeat(detuning, shape[1], axis=0) if detuning is not None else None
    output = simulate(kappa.reshape(-1,config.n_lasers,config.n_lasers),
        phi.reshape(-1,config.n_lasers,config.n_lasers), config,
        detunings_ghz=repeated_detuning, smooth_frequencies=False)
    cosine = core.calculate_phase_reward(output[0],output[1],repeated_targets,config)
    result = dict(chunk_index=chunk,start_index=start,stop_index=stop,
        phase_reward=cosine[0].reshape(shape),
        mean_phase_score=cosine[1].reshape(shape),
        worst_phase_score=cosine[2].reshape(shape),
        achieved_phases_rad=cosine[3].reshape(*shape,config.n_lasers),
        aligned_achieved_phases_rad=cosine[4].reshape(*shape,config.n_lasers),
        orientation=cosine[5].reshape(shape))
    if config.phase_reward_mode == "wrapped_rms":
        rms = core.calculate_phase_rms_metrics(output[0],output[1],repeated_targets,config)
        result.update({key:value.reshape(shape) for key,value in rms.items()})
    return result


def simulate_indexed_chunk(indexed_work):
    """Retain the training job's size and candidate indices after simulation."""
    size, start, stop, work = indexed_work
    result = simulate_phase_model_chunk(work)
    result.update(size_batch_index=size,candidate_start_index=start,candidate_stop_index=stop)
    return result


def simulate_phase_model_candidates(targets,kappa,phi,config,*,detunings,simulation_pool=None):
    """Validation worker dispatch with the same ordered target/candidate axes."""
    jobs = core.split_simulation_batch(targets,kappa,phi,config,
        detuning_distributions_ghz=detunings,n_jobs=config.n_jobs)
    results = (simulation_pool.imap_unordered(simulate_phase_model_chunk,jobs)
               if simulation_pool is not None else map(simulate_phase_model_chunk,jobs))
    combined, coverage = {}, np.zeros(len(targets),dtype=int)
    for result in results:
        start, stop = result['start_index'],result['stop_index']
        if np.any(coverage[start:stop]):
            raise RuntimeError("Duplicate target range returned by simulation")
        coverage[start:stop] += 1
        for key,value in result.items():
            if key in {'chunk_index','start_index','stop_index'}:
                continue
            if key not in combined:
                combined[key] = np.empty((len(targets),*value.shape[1:]),dtype=value.dtype)
            combined[key][start:stop] = value
    if not np.all(coverage == 1):
        raise RuntimeError("Simulation did not return every target")
    return combined
