"""A fixed graph bank for isolating topology generalization, with no action bias."""
import numpy as np
import torch

from ._compat import core, engine
from .sampling import (sample_band_random_reachable_topologies,
                       sample_random_reachable_topologies, reachable_root_options)


class FixedTopologyBank:
    """Five distinct masks per size, chosen without policy predictions or rewards."""
    def __init__(self, config, seed):
        self.seed = seed
        self.root_options = reachable_root_options(config)
        self.masks, self.roots = {}, {}
        for size in config.training_n_lasers:
            rng = np.random.default_rng(seed+size)
            for _ in range(10000):
                masks, roots = sample_band_random_reachable_topologies(
                    1, 5, size, rng, 'cpu', return_roots=True, **self.root_options)
                array = masks[0].numpy().copy()
                counts = array.sum(1)
                bands = engine.topology_density_band_ids(counts, size)
                # Avoid a smaller bank from duplicate draws and retain the
                # intended sparse/middle/dense task coverage. No physics is used.
                if len({row.tobytes() for row in array}) == 5 and len(np.unique(bands)) == 3:
                    self.masks[size], self.roots[size] = array, roots[0].copy()
                    break
            else:
                raise RuntimeError(f'Could not construct a distinct five-graph bank for M={size}')

    def __call__(self, n_pairs, n_groups, n_lasers, rng, device, **options):
        if n_groups != 5 or n_lasers not in self.masks:
            raise ValueError('The fixed topology bank contains five masks per configured size')
        # Advance the original topology RNG as before. Target draws are not
        # accidentally shifted merely because we stopped using fresh masks.
        # This is a cheap mask draw, not an additional physical simulation.
        sample_band_random_reachable_topologies(
            n_pairs, n_groups, n_lasers, rng, device, **options)
        masks = torch.as_tensor(self.masks[n_lasers], device=device).unsqueeze(0).repeat(n_pairs, 1, 1)
        if options.get('return_roots', False):
            return masks, np.broadcast_to(self.roots[n_lasers], (n_pairs, 5)).copy()
        return masks

    def validation_masks(self, size, count, *, seen, seed, device='cpu'):
        """One graph per held-out context; unseen graphs exclude the entire bank."""
        if seen:
            indices = np.arange(count) % 5
            return (torch.as_tensor(self.masks[size][indices].copy(), device=device),
                    self.roots[size][indices].copy())
        rng = np.random.default_rng(seed)
        masks, roots = sample_random_reachable_topologies(
            1, count, size, rng, 'cpu', return_roots=True, **self.root_options)
        masks, roots = masks[0].numpy().copy(), roots[0].copy()
        excluded = {row.tobytes() for row in self.masks[size]}
        for index in range(count):
            for _ in range(10000):
                if masks[index].tobytes() not in excluded:
                    break
                replacement, root = sample_random_reachable_topologies(
                    1, 1, size, rng, 'cpu', return_roots=True, **self.root_options)
                masks[index], roots[index] = replacement[0, 0].numpy(), root[0, 0]
            else:
                raise RuntimeError('Could not draw a topology outside the training bank')
        return torch.as_tensor(masks, device=device), roots

    def metadata(self):
        sizes = {}
        for size, masks in self.masks.items():
            receiver, source = core.directed_link_indices(size)
            matrices = np.zeros((5, size, size), dtype=int)
            matrices[:, receiver, source] = masks.astype(int)
            sizes[str(size)] = dict(masks=masks.astype(int).tolist(),
                root_indices=self.roots[size].tolist(),
                link_counts=masks.sum(1).astype(int).tolist(),
                matrices_receiver_source=matrices.tolist())
        return dict(seed=self.seed, masks_per_size=5,
            convention='receiver rows, source columns; laser labels follow sorted detuning order',
            selection='same band_random growth sampler; distinct masks and all three actual density bands',
            reward_used_in_selection=False, sizes=sizes)
