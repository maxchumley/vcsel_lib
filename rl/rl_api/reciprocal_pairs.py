"""Canonical unordered pairs, expanded to the existing directed decoder layout."""
import torch
from ._compat import core


def pair_layout(size, device=None):
    receiver, source = core.directed_link_indices(size)
    lookup = {(int(r), int(s)): i for i, (r, s) in enumerate(zip(receiver, source))}
    canonical = [i for i, (r, s) in enumerate(zip(receiver, source)) if r < s]
    pair_number = {(int(receiver[i]), int(source[i])): j for j, i in enumerate(canonical)}
    expansion = [pair_number[tuple(sorted((int(r), int(s))))] for r, s in zip(receiver, source)]
    reverse = [lookup[(int(s), int(r))] for r, s in zip(receiver, source)]
    return tuple(torch.tensor(x, dtype=torch.long, device=device)
                 for x in (canonical, expansion, reverse))


def expand_pair_actions(actions, expansion):
    pairs = actions.shape[-1] // 2
    return torch.cat((actions[..., :pairs][..., expansion],
                      actions[..., pairs:][..., expansion]), dim=-1)


def check_shared_actions(actions, masks, size):
    _, _, reverse = pair_layout(size, actions.device)
    edges = len(reverse)
    if not torch.equal(masks, masks[..., reverse]):
        raise ValueError('Reciprocal pair actions require symmetric masks')
    if (not torch.equal(actions[..., :edges], actions[..., :edges][..., reverse])
            or not torch.equal(actions[..., edges:], actions[..., edges:][..., reverse])):
        raise ValueError('Reciprocal pairs must share exactly one strength and phase action')
