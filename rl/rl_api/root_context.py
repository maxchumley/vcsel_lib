"""Attach one explicit root flag to node features at the encoder boundary."""
import numpy as np
import torch

from ._compat import engine


def root_node_features(features, root_indices):
    """Append one binary column; roots follow flattened context/action order."""
    roots = torch.as_tensor(root_indices, device=features.device)
    if (features.ndim != 3 or features.shape[-1] != 4
            or roots.shape != (len(features),)
            or roots.dtype not in (torch.int32, torch.int64)
            or torch.any(roots < 0) or torch.any(roots >= features.shape[1])):
        raise ValueError("Root encoding needs one valid integer root per feature row")
    flag = features.new_zeros((*features.shape[:2], 1))
    flag[torch.arange(len(features), device=features.device), roots.long(), 0] = 1.
    return torch.cat((features, flag), dim=-1)


class RootConditionedPolicy:
    """A per-call view of the policy with explicit roots, not persistent state.

    Existing samplers repeat four-column context rows in context/action order.
    This adapter supplies the matching fifth column to the actual encoder.
    Gaussian sampling, scores, budgets, parameters and checkpoint ownership
    still belong to the original policy and existing numerical routines.
    """
    def __init__(self, policy, roots):
        self.policy = policy
        self.roots = np.asarray(roots).reshape(-1)

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def parameters_for_edges(self, features, **options):
        return self.policy.parameters_for_edges(root_node_features(features, self.roots), **options)

    def budget_parameters(self, features, **options):
        return self.policy.budget_parameters(root_node_features(features, self.roots), **options)


def condition_on_roots(policy, root_indices):
    if not getattr(policy.config, 'encode_root', False):
        return policy
    if root_indices is None:
        raise ValueError("Root-aware policies require explicit roots before prediction")
    return RootConditionedPolicy(policy, root_indices)


def calibrate_budget_references(policy, config):
    """Keep the original calibration using root 0 for its complete graphs.

    The new root column starts at zero, preserving the predecessor's initial
    raw strengths and calibration values exactly without any extra RNG draws.
    """
    conditioned = condition_on_roots(policy, np.zeros(config.budget_reference_calibration_targets, dtype=int))
    return engine.calibrate_budget_references(conditioned, config)
