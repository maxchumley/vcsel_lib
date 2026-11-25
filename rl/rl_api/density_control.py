"""Density-conditioned connectivity using the existing continuous actions.

The physical mask is a deterministic function of sampled strengths and the
requested pair count. Score the original all-to-all Gaussian action, including
removed strengths: those strengths can change which pairs get selected.
"""
from dataclasses import replace
import numpy as np
import torch

from ._compat import core, engine


def density_requests(rows, rng):
    """Tree, one fraction in each third of [0,1], and all-to-all."""
    values = np.empty((rows, 5))
    values[:, 0], values[:, -1] = 0., 1.
    values[:, 1:4] = (np.arange(3)[None]+rng.random((rows, 3)))/3
    return values


def pair_counts(size, fractions):
    values = np.asarray(fractions, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('density_fraction must be finite and in [0,1]')
    return (size-1 + np.rint(values*((size-1)*(size-2)//2)).astype(int))


def density_band_ids(directed_counts, size):
    """Physical excess-pair bands measured from a symmetric tree to ATA."""
    counts = np.asarray(directed_counts)
    return np.minimum(3*(counts//2-(size-1))//max((size-1)*(size-2)//2, 1), 2)


class DensityConditionedPolicy:
    """Reuse both adjacency-feature slots for one uniform retained-pair fraction.

    All computational edges remain present. The sampler's all-one scoring mask
    is separate from these features, so fractional conditioning never weights
    the Gaussian score or acts as a physical gate.
    """
    def __init__(self, policy, fractions):
        self.policy = policy
        self.fractions = np.asarray(fractions, dtype=float).reshape(-1)
        pair_counts(3, self.fractions)  # Validate without rounding the actual encoder input.

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def _density_features(self, features):
        if features.shape[-1] != 4 or len(features) != len(self.fractions):
            raise ValueError('Supply one density fraction per repeated four-column context')
        size = features.shape[1]
        density = pair_counts(size, self.fractions)/(size*(size-1)/2)
        values = torch.as_tensor(density, dtype=features.dtype, device=features.device)
        return values[:, None].expand(-1, size*(size-1))

    def parameters_for_edges(self, features, **options):
        options['topology_masks'] = self._density_features(features)
        return self.policy.parameters_for_edges(features, **options)

    def budget_parameters(self, features, **options):
        if options.get('pooled_states') is None:
            options['topology_masks'] = self._density_features(features)
        return self.policy.budget_parameters(features, **options)


def select_masks(kappa, fractions):
    """Kruskal maximum-score tree, then highest-score remaining reciprocal pairs.

    Ties follow lexicographic pair order. The construction root is unrestricted;
    a selected connected symmetric graph makes every laser a possible root.
    Only the mask is symmetric: directed strengths/phases remain independent.
    """
    kappa = np.asarray(kappa)
    size = kappa.shape[-1]
    requested = pair_counts(size, fractions)
    if (kappa.ndim != 4 or kappa.shape[-2] != size or requested.shape != kappa.shape[:2]
            or not np.isfinite(kappa).all() or np.any(kappa < 0)):
        raise ValueError('Selection needs finite nonnegative (contexts,actions,M,M) strengths and fractions')
    first, second = np.triu_indices(size, 1)
    masks = np.zeros_like(kappa, dtype=bool)
    for strengths, count, output in zip(kappa.reshape(-1, size, size), requested.flat,
                                         masks.reshape(-1, size, size)):
        order = np.argsort(-(strengths[first, second]+strengths[second, first]), kind='stable')
        selected = np.zeros(len(first), dtype=bool)
        parents = list(range(size))

        def root(node):
            while parents[node] != node:
                parents[node] = parents[parents[node]]
                node = parents[node]
            return node

        for edge in order:
            a, b = root(int(first[edge])), root(int(second[edge]))
            if a != b:
                parents[a] = b
                selected[edge] = True
                if selected.sum() == size-1:
                    break
        extra = int(count)-(size-1)
        selected[order[~selected[order]][:extra]] = True
        output[first[selected], second[selected]] = True
        output[second[selected], first[selected]] = True
    receiver, source = core.directed_link_indices(size)
    return masks[..., receiver, source]


def decode_density_actions(actions, budget_actions, fractions, config):
    """Select from the sampled dense matrix, then reuse the physical decoder."""
    from .phase_reference import decode_samples
    dense = torch.ones((*actions.shape[:2], actions.shape[-1]//2),
                       dtype=actions.dtype, device=actions.device)
    # Physical bounds preserve pair totals. Rank the original strengths explicitly
    # so floating-point ties cannot change a graph merely as the bound tightens.
    ranking_config = (replace(config, coupling_symmetry_fraction=0.)
        if getattr(config, 'coupling_symmetry_method', '') == 'physical_bounds' else config)
    kappa = decode_samples(actions, dense, ranking_config, budget_actions=budget_actions)[0]
    masks = torch.as_tensor(select_masks(kappa, fractions), dtype=actions.dtype, device=actions.device)
    decoded = decode_samples(actions, masks, config, budget_actions=budget_actions)
    return masks, decoded


def prepare_density_size(policy, config, rng, size, pair_count):
    """Same contexts, action groups and exploration; replace mask input by s."""
    from .exploration import sample_annealed_candidates
    local = replace(config, n_lasers=size)
    targets = core.sample_target_phases(pair_count, rng, size,
        config.structured_target_fraction, config.structured_target_jitter_rad)
    detunings = core.sample_detuning_distributions(pair_count, rng, local)
    targets, detunings = core.sort_by_detuning(targets, detunings)
    features = engine.encode_context(targets, detunings, policy)
    fractions = np.repeat(density_requests(pair_count, rng), config.actions_per_topology, axis=1)
    dense = features.new_ones((pair_count, config.candidates_per_target, size*(size-1)))
    if getattr(config, 'condition_phases_on_coupling', False):
        from .conditional_phases import sample_conditional_density
        actions, budget_actions, log_prob, masks, decoded = sample_conditional_density(
            policy, features, local, fractions, mirrored=True)
    else:
        actions, budget_actions, log_prob = sample_annealed_candidates(
            DensityConditionedPolicy(policy, fractions), features, dense, mirrored=True)
        masks, decoded = decode_density_actions(actions, budget_actions, fractions, local)
    kappa, phi, counts, rho, shortfall = decoded
    return dict(n_lasers=size, config=local, targets=targets, detunings=detunings,
        features=features, gates=masks, log_prob=log_prob, kappa=kappa, phi=phi,
        counts=counts, rho=rho, shortfall=shortfall, reach=np.ones_like(counts, dtype=float),
        roots=None, density_fractions=fractions, actions=actions.detach(),
        budget_actions=budget_actions.detach(), results=[])


def density_candidates(policy, features, masks, *, continuous_mean=False):
    if getattr(policy.config, 'reciprocal_pair_actions', False):
        from .exploration import sample_annealed_candidates
        return sample_annealed_candidates(policy, features, masks,
                                           continuous_mean=continuous_mean)
    return engine.sample_topology_conditioned_candidates(
        policy, features, masks, continuous_mean=continuous_mean)


def validation_actions(policy, features, config, seed):
    """One fixed density request per held-out context, balanced across five slots."""
    count = len(features)
    groups = density_requests((count+4)//5, np.random.default_rng(seed)).reshape(-1)[:count, None]
    if getattr(config, 'condition_phases_on_coupling', False):
        from .conditional_phases import sample_conditional_density
        return sample_conditional_density(policy, features, config, groups, continuous_mean=True)[-1]
    dense = features.new_ones((count, 1, config.n_lasers*(config.n_lasers-1)))
    actions, budget, _ = density_candidates(
        DensityConditionedPolicy(policy, groups), features, dense, continuous_mean=True)
    _, decoded = decode_density_actions(actions, budget, groups, config)
    return decoded


def infer_density_designs(policy, context, config, fractions, count, continuous_mean, seed):
    """Inference shares training's density encoding, ranking and redistribution."""
    target, detunings = core.sort_by_detuning(context.target_phases_rad, context.detunings_ghz)
    if fractions is None:
        rng = np.random.default_rng(seed)
        values = (np.arange(count)+rng.random(count))/count
        if count > 1:
            values[0], values[-1] = 0., 1.
        rng.shuffle(values)
    else:
        values = np.asarray(fractions, dtype=float)
        if values.ndim == 0:
            values = np.full(count, float(values))
    if values.shape != (count,):
        raise ValueError('density_fraction must be a scalar or one value per candidate')
    pair_counts(config.n_lasers, values)
    features = engine.encode_context(target, detunings, policy)
    dense = features.new_ones((1, count, config.n_lasers*(config.n_lasers-1)))
    from contextlib import nullcontext
    with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext(), torch.inference_mode():
        if seed is not None:
            torch.manual_seed(seed)
        if getattr(config, 'condition_phases_on_coupling', False):
            from .conditional_phases import sample_conditional_density
            _, _, _, masks, decoded = sample_conditional_density(
                policy, features, config, values[None], continuous_mean=continuous_mean)
        else:
            actions, budget, _ = density_candidates(
                DensityConditionedPolicy(policy, values), features, dense,
                continuous_mean=continuous_mean or config.exploration_std_scale == 0.)
            masks, decoded = decode_density_actions(actions, budget, values[None], config)
    kappa, phi, counts, rho, shortfall = decoded
    return dict(target_phases_rad=target, detuning_distribution_ghz=detunings,
        kappa_per_ns=kappa[0], phi_p_rad=phi[0], gates=masks.cpu().numpy()[0],
        root_indices=None, tree_gates=None, gate_logits=None,
        natural_gate_probabilities=None, optional_gate_probabilities=None,
        gate_logit_bias=0., expected_active_fraction=float(rho.mean()),
        active_link_counts=counts[0], retained_link_fractions=rho[0],
        raw_total_coupling_per_ns=kappa[0].sum(axis=(-2,-1))+shortfall[0],
        redistribution_shortfall_per_ns=shortfall[0], density_fractions=values,
        requested_pair_counts=pair_counts(config.n_lasers, values))
