"""The opt-in sampled-root delay reference, shared by training and inference."""
import numpy as np

from ._compat import core, engine


ROOT_PHASE_MODE = "target_root_delay_residual"


def uses_root_reference(config):
    return getattr(config, "phase_parameterization", "absolute") == ROOT_PHASE_MODE


def apply_root_reference(phases, masks, detunings_ghz, root_indices, config):
    """Translate the existing minimum-reference outputs to each sampled root.

    The policy retains the existing minimum-reference Gaussian coordinates.
    This fixed translation at physical decoding gives
    phi_p = target_source - target_receiver - 2*pi*f_root*tau + residual.
    A context-dependent translation has unit Jacobian, so the original
    Gaussian REINFORCE scores and mirrored perturbations remain valid.
    No policy inputs, parameters, random draws or simulator equations change.

    Shapes: phases (pairs,candidates,M,M), masks (pairs,candidates,E),
    detunings (pairs,M), and zero-based roots (pairs,candidates). Laser labels
    and detunings must already share the same sorted order.
    """
    if not uses_root_reference(config):
        return phases
    detunings = np.asarray(detunings_ghz)
    roots = np.asarray(root_indices)
    if (phases.ndim != 4 or detunings.shape != (phases.shape[0], phases.shape[-1])
            or roots.shape != phases.shape[:2]
            or not np.issubdtype(roots.dtype, np.integer)
            or np.any(roots < 0) or np.any(roots >= phases.shape[-1])):
        raise ValueError("Root-reference decoding needs one valid root per topology/action")
    reference = np.take_along_axis(detunings, roots, axis=1)
    minimum = detunings.min(axis=1, keepdims=True)
    tau = int(config.delay_seconds/config.time_step_seconds)*config.time_step_seconds
    rotation = -2*np.pi*1e9*tau*(reference-minimum)
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    receivers, sources = core.directed_link_indices(phases.shape[-1])
    enabled = np.zeros(phases.shape, dtype=bool)
    enabled[..., receivers, sources] = np.asarray(masks, dtype=bool)
    changed = enabled & (rotation[..., None, None] != 0.)
    wrapped = np.angle(np.exp(1j*(phases+rotation[..., None, None])))
    return np.where(changed, wrapped, phases)


def decode_samples(actions, masks, config, *, budget_actions=None,
                   detunings_ghz=None, root_indices=None):
    """Existing physical decoder, followed only by the opt-in reference shift."""
    kappa, phases, counts, rho, shortfall = engine.decode_samples(
        actions, masks, config, budget_actions=budget_actions)
    phases = apply_root_reference(phases, masks, detunings_ghz, root_indices, config)
    return kappa, phases, counts, rho, shortfall


def roots_for_supplied_masks(masks, n_lasers):
    """For manual graphs, choose the lowest-detuned laser that reaches all.

    Automatic graphs always retain their actual sampled construction root.
    A manually supplied graph has no construction-root metadata, so this
    deterministic rule also handles feedback graphs with multiple roots.
    """
    receivers, sources = core.directed_link_indices(n_lasers)
    roots = []
    for mask in np.asarray(masks, dtype=bool):
        for root in range(n_lasers):
            seen = np.zeros(n_lasers, dtype=bool)
            seen[root] = True
            while True:
                updated = seen.copy()
                updated[receivers[mask & seen[sources]]] = True
                if np.array_equal(updated, seen):
                    break
                seen = updated
            if seen.all():
                roots.append(root)
                break
        else:
            raise ValueError("Every supplied topology must have a root reaching all lasers")
    return np.asarray(roots, dtype=int)
