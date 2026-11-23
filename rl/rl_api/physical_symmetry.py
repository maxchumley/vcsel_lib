"""Hard physical reciprocity limits, independent of unbounded policy outputs."""
import numpy as np
from ._compat import core


def constrain_physical_values(kappa, phases, masks, config):
    """Limit relative strength mismatch and shortest circular phase separation.

    At tightening fraction s, |ki-kj|/(ki+kj) <= 1-s and phase distance
    <= pi*(1-s). Pairs inside the requested bounds are left bit-identical.
    Strength pair totals, budget and masks are preserved. At s=1 selected
    components are copied exactly in both directions. The function acts only
    on decoded physical values; latent Gaussian REINFORCE scores are unchanged.
    """
    fraction = config.coupling_symmetry_fraction
    if not 0. <= fraction <= 1.:
        raise ValueError('coupling_symmetry_fraction must be in [0,1]')
    if fraction == 0.:
        return kappa, phases
    size = kappa.shape[-1]
    receiver, source = core.directed_link_indices(size)
    if hasattr(masks, 'detach'):
        masks = masks.detach().cpu().numpy()
    graph = np.zeros(kappa.shape, dtype=bool)
    graph[..., receiver, source] = np.asarray(masks, dtype=bool)
    if not np.array_equal(graph, graph.swapaxes(-1, -2)):
        raise ValueError('Physical symmetry bounds require symmetric topology masks')
    i, j = np.triu_indices(size, 1)
    enabled = graph[..., i, j]
    bound = 1.-fraction
    if config.coupling_symmetry_strengths:
        ka, kb = kappa[..., i, j], kappa[..., j, i]
        total, difference = ka+kb, ka-kb
        changed = enabled & ((np.abs(difference) > bound*total) | (fraction == 1.))
        clipped = np.clip(difference, -bound*total, bound*total)
        a = .5*(total+clipped)
        b = a if fraction == 1. else total-a
        kappa = kappa.copy()
        kappa[..., i, j] = np.where(changed, a, ka)
        kappa[..., j, i] = np.where(changed, b, kb)
    # Cap inactive phase entries too; their values do not affect dynamics,
    # but this keeps the full returned phase matrix reciprocal at the endpoint.
    if config.coupling_symmetry_phases:
        pa, pb = phases[..., i, j], phases[..., j, i]
        delta = np.arctan2(np.sin(pb-pa), np.cos(pb-pa))
        changed = (np.abs(delta) > np.pi*bound) | (fraction == 1.)
        clipped = np.clip(delta, -np.pi*bound, np.pi*bound)
        midpoint = pa+.5*delta
        a = np.angle(np.exp(1j*(midpoint-.5*clipped)))
        b = a if fraction == 1. else np.angle(np.exp(1j*(midpoint+.5*clipped)))
        phases = phases.copy()
        phases[..., i, j] = np.where(changed, a, pa)
        phases[..., j, i] = np.where(changed, b, pb)
    return kappa, phases
