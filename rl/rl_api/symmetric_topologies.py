"""Complete four-laser connected bank and reciprocal physical action values."""
import numpy as np
import torch

from ._compat import core, engine


def connected_symmetric_masks(size=4):
    """Enumerate labeled masks once each, ordered by pair count then edge bits."""
    if size != 4:
        raise ValueError('This exhaustive experiment supports exactly four lasers')
    first, second = np.triu_indices(size, 1)
    matrices = []
    for bits in range(1 << len(first)):
        active = (bits >> np.arange(len(first))) & 1
        graph = np.zeros((size, size), dtype=bool)
        graph[first, second] = active
        graph[second, first] = active
        reached = np.zeros(size, dtype=bool)
        reached[0] = True
        for _ in range(size-1):
            reached |= graph[reached].any(axis=0)
        if reached.all():
            matrices.append(graph)
    matrices.sort(key=lambda matrix: (int(matrix.sum()), tuple(matrix[first, second])))
    receiver, source = core.directed_link_indices(size)
    return np.stack(matrices)[:, receiver, source]


def sample_all_symmetric_topologies(n_pairs, n_groups, n_lasers, rng, device, **options):
    masks = connected_symmetric_masks(n_lasers)
    if n_groups != len(masks):
        raise ValueError('All 38 connected symmetric topologies must be included')
    result = torch.as_tensor(masks, dtype=torch.float32, device=device)[None].repeat(n_pairs, 1, 1)
    return (result, np.zeros((n_pairs, n_groups), dtype=int)) if options.get('return_roots') else result


def topology_metadata():
    masks = connected_symmetric_masks()
    receiver, source = core.directed_link_indices(4)
    matrices = np.zeros((len(masks), 4, 4), dtype=int)
    matrices[:, receiver, source] = masks
    return dict(size=4, count=len(masks), pair_counts=(masks.sum(1)//2).tolist(),
        matrices_receiver_source=matrices.tolist(),
        convention='receiver rows, source columns; labels follow sorted detuning order',
        coverage='all connected labeled symmetric masks, exactly once per detuning context')


def reciprocal_actions(actions, masks, size):
    """Average unwrapped reciprocal latent actions before physical decoding.

    The Gaussian samples and REINFORCE scores remain the original independent
    directed actions. This deterministic projection adds no policy parameter.
    In-phase target offsets and the shared laser-1 delay offset are symmetric.
    """
    receiver, source = core.directed_link_indices(size)
    indices = np.full((size, size), -1, dtype=int)
    indices[receiver, source] = np.arange(len(receiver))
    reverse = torch.as_tensor(indices[source, receiver], device=actions.device)
    if not torch.equal(masks, masks[..., reverse]):
        raise ValueError('Symmetric coupling values require a symmetric physical topology')
    both = torch.cat((reverse, reverse+len(receiver)))
    return .5*(actions+actions[..., both])


def reciprocal_normal_log_prob(actions, mean, std, masks, size):
    """Score the executed pair averages, once per active reciprocal pair.

    If independent a,b are normal, (a+b)/2 is normal with mean (mu_a+mu_b)/2
    and variance (sigma_a**2+sigma_b**2)/4. Marginalizing the discarded
    difference preserves the expected REINFORCE gradient and removes its
    nuisance noise. Sampling/RNG draws and the physical decoder are unchanged.
    """
    receiver, source = core.directed_link_indices(size)
    edge_count = len(receiver)
    if (actions.shape != mean.shape or actions.shape != std.shape
            or actions.shape[:-1] != masks.shape[:-1]
            or actions.shape[-1] != 2*edge_count or masks.shape[-1] != edge_count):
        raise ValueError('Shared-pair scoring needs matching directed action parameters and masks')
    indices = np.full((size, size), -1, dtype=int)
    indices[receiver, source] = np.arange(edge_count)
    reverse = torch.as_tensor(indices[source, receiver], device=actions.device)
    if not torch.equal(masks, masks[..., reverse]):
        raise ValueError('Shared-pair scoring requires symmetric topology masks')
    first = torch.as_tensor(np.flatnonzero(receiver < source), device=actions.device)
    second = reverse[first]
    first_actions = torch.cat((first, first+edge_count))
    second_actions = torch.cat((second, second+edge_count))
    paired_mean = .5*(mean[..., first_actions]+mean[..., second_actions])
    paired_std = .5*torch.sqrt(std[..., first_actions].square()+std[..., second_actions].square())
    values = .5*(actions[..., first_actions]+actions[..., second_actions])
    normal = torch.distributions.Normal(paired_mean, paired_std)
    active = torch.cat((masks[..., first], masks[..., first]), dim=-1)
    return (normal.log_prob(values.detach())*active).sum(dim=-1)


def infer_symmetric_designs(policy, context, config, topology, count, continuous_mean, seed):
    """Predict on the complete bank or explicit symmetric masks, using the training decoder."""
    from .phase_reference import decode_samples
    if context.n_lasers != 4:
        raise ValueError('This exhaustive model was configured for four lasers')
    if np.max(np.abs(np.angle(np.exp(1j*(context.target_phases_rad-context.target_phases_rad[0]))))) > 1e-10:
        raise ValueError('This experiment uses in-phase targets only')
    target, detunings = core.sort_by_detuning(context.target_phases_rad, context.detunings_ghz)
    receiver, source = core.directed_link_indices(4)
    if topology is None:
        bank = connected_symmetric_masks()
        order = np.random.default_rng(seed).permutation(len(bank))
        masks = bank[order[np.arange(count) % len(bank)]]
    else:
        matrices = np.asarray(topology)
        if matrices.ndim == 2:
            matrices = np.broadcast_to(matrices, (count, 4, 4))
        if matrices.shape != (count, 4, 4) or not np.array_equal(matrices, matrices.swapaxes(-1,-2)):
            raise ValueError('Each supplied topology must be a symmetric four-laser mask')
        order = np.argsort(context.detunings_ghz, kind='stable')
        matrices = matrices[:, order, :][:, :, order]
        masks = matrices[:, receiver, source]
        if not np.all(engine.root_reach_fractions(masks, 4) == 1.):
            raise ValueError('Every supplied symmetric topology must be connected')
    features = engine.encode_context(target, detunings, policy)
    gates = torch.as_tensor(masks[None], dtype=features.dtype, device=features.device)
    from contextlib import nullcontext
    with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext(), torch.inference_mode():
        if seed is not None:
            torch.manual_seed(seed)
        actions, budget, _ = engine.sample_topology_conditioned_candidates(
            policy, features, gates,
            continuous_mean=continuous_mean or config.exploration_std_scale == 0.)
        kappa, phi, counts, rho, shortfall = decode_samples(actions, gates, config, budget_actions=budget)
    return dict(target_phases_rad=target, detuning_distribution_ghz=detunings,
        kappa_per_ns=kappa[0], phi_p_rad=phi[0], gates=gates.cpu().numpy()[0],
        root_indices=None, tree_gates=None, gate_logits=None,
        natural_gate_probabilities=None, optional_gate_probabilities=None,
        expected_active_fraction=float(rho.mean()), active_link_counts=counts[0],
        retained_link_fractions=rho[0], raw_total_coupling_per_ns=kappa[0].sum((-2,-1))+shortfall[0],
        redistribution_shortfall_per_ns=shortfall[0])
