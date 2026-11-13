"""Reference sampling helpers, preserving RNG consumption and mask ordering."""
from dataclasses import dataclass, replace
import numpy as np
import torch

from ._compat import core, engine


@dataclass(frozen=True)
class PairedContexts:
    """Five target/detuning rows and four masks per row in the default recipe."""
    target_phases_rad: np.ndarray
    detunings_ghz: np.ndarray
    edge_masks: torch.Tensor
    root_indices: np.ndarray | None = None


def random_reachable_mask(n_lasers, rng, minimum_links=0):
    """Choose a root, add arbitrary random directed edges until it reaches all nodes.

    Continue along the same random edge order for any requested extra links.
    Edges between unreached nodes, cycles, and feedback into the root are all
    eligible. No spanning tree or star is inserted, and no self-links occur.
    """
    receivers, sources = core.directed_link_indices(n_lasers)
    if n_lasers < 2 or not 0 <= minimum_links <= len(receivers):
        raise ValueError("Reachable sampling needs M>=2 and a valid minimum link count")
    root = int(rng.integers(n_lasers))
    order = rng.permutation(len(receivers))
    mask = np.zeros(len(receivers), dtype=bool)
    reachable = np.zeros(n_lasers, dtype=bool)
    reachable[root] = True
    for count, edge in enumerate(order, start=1):
        mask[edge] = True
        while True:
            updated = reachable.copy()
            updated[receivers[mask & reachable[sources]]] = True
            if np.array_equal(updated, reachable):
                break
            reachable = updated
        if reachable.all() and count >= minimum_links:
            return mask, root
    raise RuntimeError("Complete directed graph did not reach every laser")


def sample_random_reachable_topologies(n_pairs, n_groups, n_lasers, rng, device, *,
                                       min_fraction=0., density_bias=1., fixed_star=False,
                                       return_roots=False):
    """Reachability-grown graphs with stratified requested counts for inspection/validation.

    Reachability takes priority over the requested count: a random prefix may
    need more edges to connect the chosen root. Every graph is grown independently.
    """
    if (n_pairs < 1 or n_groups < 1 or n_lasers < 2 or fixed_star
            or not 0. <= min_fraction <= 1. or density_bias != 1.):
        raise ValueError("Random reachable sampling needs positive counts and no fixed star")
    edges = n_lasers*(n_lasers-1)
    minimum = max(n_lasers-1, int(np.ceil(min_fraction*edges-1e-10)))
    levels = np.arange(minimum, edges+1)
    if n_groups == 1:
        counts = rng.choice(levels, size=(n_pairs, 1))
    elif n_groups >= len(levels):
        repeats, remainder = divmod(n_groups, len(levels))
        counts = np.stack([rng.permutation(np.r_[np.tile(levels, repeats),
            rng.choice(levels, remainder, replace=False)]) for _ in range(n_pairs)])
    else:
        counts = np.stack([rng.permutation(np.rint(np.linspace(minimum, edges, n_groups)).astype(int))
                           for _ in range(n_pairs)])
    masks = np.empty((n_pairs, n_groups, edges), dtype=bool)
    roots = np.empty((n_pairs, n_groups), dtype=int)
    for pair in range(n_pairs):
        for group in range(n_groups):
            masks[pair, group], roots[pair, group] = random_reachable_mask(n_lasers, rng, int(counts[pair, group]))
    result = torch.as_tensor(masks, device=device, dtype=torch.float32)
    return (result, roots) if return_roots else result


def sample_band_random_reachable_topologies(n_pairs, n_groups, n_lasers, rng, device, *,
                                            min_fraction=0., density_bias=1., fixed_star=False,
                                            return_roots=False):
    """Five density slots: first reachable graph, random band extras, all-to-all.

    Use the previous run's requested sparse/middle/dense counts, but keep
    adding edges when required for reachability. Each slot chooses its own root
    and random edge order; optional edges can occur anywhere in the graph.
    """
    if (n_pairs < 1 or n_groups != 5 or n_lasers < 3 or min_fraction != 0.
            or density_bias != 1. or fixed_star):
        raise ValueError("Reachable band sampling needs five masks, M>=3 and no fixed star")
    edges = n_lasers*(n_lasers-1)
    extras = (n_lasers-1)**2
    levels = np.arange(1, extras)
    bands = np.minimum(3*levels//extras, 2)
    masks = np.empty((n_pairs, n_groups, edges), dtype=bool)
    roots = np.empty((n_pairs, n_groups), dtype=int)
    for pair in range(n_pairs):
        for group in range(n_groups):
            requested = (n_lasers-1 if group == 0 else edges if group == 4 else
                         n_lasers-1+int(rng.choice(levels[bands == group-1])))
            masks[pair, group], roots[pair, group] = random_reachable_mask(n_lasers, rng, requested)
    result = torch.as_tensor(masks, device=device, dtype=torch.float32)
    return (result, roots) if return_roots else result


def sample_band_random_topologies(n_pairs, n_groups, n_lasers, rng, device, *,
                                  min_fraction=0., density_bias=1., fixed_star=True):
    """Per pair: star, random sparse/middle/dense masks, then all-to-all.

    Endpoints are excluded from random counts. Every optional edge subset is
    drawn independently for its pair and mask. No graphs are broadcast between
    pairs, and every call redraws all intermediate counts and wiring.
    """
    if (n_pairs < 1 or n_groups != 5 or n_lasers < 3 or min_fraction != 0.
            or density_bias != 1. or not fixed_star):
        raise ValueError("Band sampling requires five rooted masks per pair, M>=3 and uniform counts")
    _, sources = core.directed_link_indices(n_lasers)
    star = sources == 0
    optional = np.flatnonzero(~star)
    levels = np.arange(1, len(optional))  # Extra-link counts, excluding both endpoints.
    bands = np.minimum(3*levels//len(optional), 2)
    masks = np.broadcast_to(star, (n_pairs, 5, len(star))).copy()
    masks[:, -1] = True
    for pair in range(n_pairs):
        for band in range(3):
            count = int(rng.choice(levels[bands == band]))
            edges = rng.choice(optional, size=count, replace=False)
            masks[pair, band+1, edges] = True
    return torch.as_tensor(masks, device=device, dtype=torch.float32)


def sample_fixed_count_topologies(n_pairs, n_groups, n_lasers, rng, device, *,
                                  min_fraction=0., density_bias=1., fixed_star=True):
    """Five evenly spaced counts; redraw optional wiring for every pair.

    Use the same star/sparse/middle/dense/all-to-all order and independent
    optional-edge draws as band_random. Only the intermediate counts are fixed.
    Integer rounding uses NumPy's rint, as in the original fixed-count sampler.
    """
    if (n_pairs < 1 or n_groups != 5 or n_lasers < 3 or min_fraction != 0.
            or density_bias != 1. or not fixed_star):
        raise ValueError("Fixed-count sampling requires five rooted masks per pair, M>=3 and uniform counts")
    _, sources = core.directed_link_indices(n_lasers)
    star = sources == 0
    optional = np.flatnonzero(~star)
    counts = np.rint(np.linspace(n_lasers-1, len(star), 5)).astype(int)
    masks = np.broadcast_to(star, (n_pairs, 5, len(star))).copy()
    masks[:, -1] = True
    for pair in range(n_pairs):
        for column, count in enumerate(counts[1:-1], start=1):
            edges = rng.choice(optional, size=int(count-(n_lasers-1)), replace=False)
            masks[pair, column, edges] = True
    return torch.as_tensor(masks, device=device, dtype=torch.float32)


def sample_shared_rooted_topologies(n_pairs, n_groups, n_lasers, rng, device, *,
                                   min_fraction=0., density_bias=1., fixed_star=True):
    """Sample once per M, then reuse identical matrices across every pair.

    Column zero is the exact outward star. Other columns sample one link
    count from each equal excess-density band, then choose optional edges at
    random. Four groups therefore mean star + sparse + middle + dense.
    A new call samples new masks; action noise is sampled separately per pair.
    """
    if n_pairs < 1 or n_groups < 2 or n_lasers < 2:
        raise ValueError("Shared sampling needs positive pairs, >=2 groups and >=2 lasers")
    if min_fraction != 0. or density_bias != 1. or not fixed_star:
        raise ValueError("Shared sampling uses an exact star and uniform density bands")
    _, sources = core.directed_link_indices(n_lasers)
    star = sources == 0
    optional = np.flatnonzero(~star)
    extra_levels = np.arange(1, len(optional)+1)
    bands = np.minimum(extra_levels*(n_groups-1)//len(optional), n_groups-2)
    masks = np.broadcast_to(star, (n_groups, len(star))).copy()
    for column in range(1, n_groups):
        levels = extra_levels[bands == column-1]
        if not len(levels):
            raise ValueError(f"M={n_lasers} has too few link counts for {n_groups} shared groups")
        extra_count = int(rng.choice(levels))
        chosen = rng.choice(optional, size=extra_count, replace=False)
        masks[column, chosen] = True
    shared = torch.as_tensor(masks, device=device, dtype=torch.float32)
    return shared[None].expand(n_pairs, -1, -1).clone()


def sample_pairs(config, rng, *, n_lasers, iteration=0, device="cpu",
                 reuse_masks_across_pairs=False, topology_sampling_mode=None):
    """Inspect the same pair/mask sampler used by current training.

    ``iteration`` is zero-based, as in the backend. Do not call this helper
    on the training RNG merely for logging: that would consume future draws.
    Set reuse_masks_across_pairs=True to inspect the next run's shared sampler.
    The reference recipe delegates to the original batch; shared training uses
    training_batch.py with these same context/mask operations.
    """
    local = replace(config, n_lasers=n_lasers)
    count = config.targets_per_batch // len(config.training_n_lasers)
    target = core.sample_target_phases(count, rng, n_lasers,
        config.structured_target_fraction, config.structured_target_jitter_rad)
    detuning = core.sample_detuning_distributions(count, rng, local, iteration=iteration)
    target, detuning = core.sort_by_detuning(target, detuning)
    mode = topology_sampling_mode or config.topology_sampling_mode
    if getattr(config, 'topology_random_reachable', False):
        # This opt-in recipe trains with band_random; its legacy-compatible
        # backend field is serialized as per_target, as for the old band run.
        sampler = (sample_band_random_reachable_topologies
                   if topology_sampling_mode is None or mode == 'band_random'
                   else sample_random_reachable_topologies)
    elif mode == "band_random":
        sampler = sample_band_random_topologies
    elif mode == "band_fixed":
        sampler = sample_fixed_count_topologies
    elif reuse_masks_across_pairs:
        sampler = sample_shared_rooted_topologies
    elif config.topology_sampling_mode == "per_target":
        sampler = engine.sample_stratified_rooted_topologies
    else:
        sampler = engine.sample_paired_rooted_topologies
    from .phase_reference import uses_root_reference
    root_options = dict(return_roots=True) if uses_root_reference(config) else {}
    masks = sampler(count, config.pcgrad_topology_groups,
        n_lasers, rng, torch.device(device),
        min_fraction=engine.topology_min_fraction_at(iteration, local),
        density_bias=config.training_topology_density_bias,
        fixed_star=config.topology_star_backbone, **root_options)
    masks, roots = masks if root_options else (masks, None)
    return PairedContexts(target, detuning, masks, roots)
