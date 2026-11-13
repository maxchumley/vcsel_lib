"""Existing policy with opt-in phase offsets and action-SD scaling."""
import math
import torch

from .._compat import core, engine


def target_phase_offsets(features):
    """Source minus receiver target phase, using the encoded target orientation.

    Features already contain cos/sin of the relative, optionally conjugated
    targets. Read these instead of independently canonicalizing the targets.
    The output uses the same directed-edge ordering as the strength/phase head.
    """
    receivers, sources = core.directed_link_indices(features.shape[1])
    receivers = torch.as_tensor(receivers,device=features.device)
    sources = torch.as_tensor(sources,device=features.device)
    cosine, sine = features[...,0], features[...,1]
    difference_cosine = cosine[:,sources]*cosine[:,receivers] + sine[:,sources]*sine[:,receivers]
    difference_sine = sine[:,sources]*cosine[:,receivers] - cosine[:,sources]*sine[:,receivers]
    return torch.atan2(difference_sine,difference_cosine)


class TargetResidualPhasePolicy(engine.BernoulliSparseGNNPolicy):
    """Same weights and inputs; the existing phase output is a residual mean.

    Add the known target difference before sampling and log-probability scoring.
    The original decoder still wraps sampled phases to physical angles. Strength,
    budget, SD and pooled-feature outputs come directly from the original policy.
    """
    def parameters_for_edges(self, features, **options):
        outputs = super().parameters_for_edges(features, **options)
        mean = outputs[0]
        edges = mean.shape[-1]//2
        phase_mean = mean[:,edges:] + target_phase_offsets(features)
        mean = torch.cat((mean[:,:edges],phase_mean),dim=-1)
        return (mean,*outputs[1:])


def delay_phase_offsets(features, config):
    """Known -2*pi*f_reference*tau rotation, one value per context.

    The mirrored run uses mean-centered detunings. Recover their minimum
    from the existing normalized positions and GHz span; no input or weight
    is added. Subtracting the position mean matters for asymmetric spacings.
    Use the simulator's discrete delay, int(tau/dt)*dt (no interpolation).
    """
    positions = features[..., 2]
    span_ghz = features[:, 0, 3]
    reference_ghz = .5 * span_ghz * (positions.amin(dim=1) - positions.mean(dim=1))
    delay_seconds = int(config.delay_seconds / config.time_step_seconds) * config.time_step_seconds
    return (-2 * math.pi * 1e9 * delay_seconds) * reference_ghz[:, None]


class TargetDelayResidualPhasePolicy(TargetResidualPhasePolicy):
    """Target difference + explicit reference-delay rotation + learned residual.

    Apply the offset before Gaussian sampling and scoring, so training,
    validation, and checkpoint inference share the same phase convention.
    """
    def parameters_for_edges(self, features, **options):
        outputs = super().parameters_for_edges(features, **options)
        mean = outputs[0]
        edges = mean.shape[-1] // 2
        phase_mean = mean[:, edges:] + delay_phase_offsets(features, self.config)
        return (torch.cat((mean[:, :edges], phase_mean), dim=-1), *outputs[1:])


class ScaledExploration:
    """Scale edge and budget SDs without changing means, weights or RNG draws."""
    def parameters_for_edges(self, features, **options):
        outputs = super().parameters_for_edges(features, **options)
        scale = self.config.exploration_std_scale
        if scale == 1.:
            return outputs
        # At zero the sampler returns exact means; this positive placeholder
        # only lets legacy inference construct its unused Normal safely.
        offset = math.log(max(scale, 1e-12))
        return (outputs[0], outputs[1] + offset, *outputs[2:])

    def budget_parameters(self, features, **options):
        mean, log_std = super().budget_parameters(features, **options)
        scale = self.config.exploration_std_scale
        if scale != 1.:
            log_std = log_std + math.log(max(scale, 1e-12))
        return mean, log_std


class AnnealedAbsolutePhasePolicy(ScaledExploration, engine.BernoulliSparseGNNPolicy):
    """Absolute phase means with checkpointed action-SD scaling."""


class AnnealedResidualPhasePolicy(ScaledExploration, TargetResidualPhasePolicy):
    """Target-plus-residual means with checkpointed action-SD scaling."""


class AnnealedDelayResidualPhasePolicy(ScaledExploration, TargetDelayResidualPhasePolicy):
    """The same mirrored-run annealing with explicit reference-delay phases."""


class RootAwareDelayResidualPhasePolicy(TargetDelayResidualPhasePolicy):
    """One extra node feature; the message layers and output heads are unchanged."""
    def __init__(self, config):
        super().__init__(config)
        first = self.node_encoder[0]
        # Preserve every old initialization and RNG draw. This new input
        # column learns from the first update; all shared weights start equal.
        first.weight = torch.nn.Parameter(torch.cat((first.weight.detach(),
            first.weight.new_zeros((first.out_features, 1))), dim=1))
        first.in_features += 1

    def _graph_context(self, features, topology_masks=None):
        if features.ndim != 3 or features.shape[-1] != 5:
            raise ValueError("Root-aware features must have shape (batch,M,5)")
        flag = features[..., 4]
        if torch.any((flag != 0.) & (flag != 1.)) or torch.any(flag.sum(dim=1) != 1.):
            raise ValueError("Each root-aware context must flag exactly one laser")
        batch, size, _ = features.shape
        receiver, source = core.directed_link_indices(size)
        receiver = torch.as_tensor(receiver, device=features.device)
        source = torch.as_tensor(source, device=features.device)
        cosine, sine, detuning = features[..., 0], features[..., 1], features[..., 2]
        difference = .5*(detuning[:, source]-detuning[:, receiver])
        local = torch.stack((cosine[:, source]*cosine[:, receiver]+sine[:, source]*sine[:, receiver],
            sine[:, source]*cosine[:, receiver]-cosine[:, source]*sine[:, receiver],
            difference, difference.abs()), dim=-1)
        if topology_masks is None or topology_masks.shape != (batch, size*(size-1)):
            raise ValueError("topology_masks must have shape (batch,M*(M-1))")
        matrix = features.new_zeros((batch, size, size))
        matrix[:, receiver, source] = topology_masks
        local = torch.cat((local, topology_masks[..., None], matrix[:, source, receiver, None]), dim=-1)
        states = self.node_encoder(features)
        for layer in self.message_layers:
            states = layer(states, local, receiver, source)
        return states, local, receiver, source


class AnnealedRootAwarePhasePolicy(ScaledExploration, RootAwareDelayResidualPhasePolicy):
    """The existing SD schedule with a root-aware node encoder."""


def build_policy(config):
    """Select the recorded phase convention, preserving all parameter keys/order.

    In particular, retain the two compatibility SD scalars: removing them now
    would alter model/optimizer state layout during a reproducibility refactor.
    """
    mode = getattr(config,'phase_parameterization','absolute')
    classes = {
        'absolute': engine.BernoulliSparseGNNPolicy,
        'target_residual': TargetResidualPhasePolicy,
        'target_delay_residual': TargetDelayResidualPhasePolicy,
        # Retain identical latent Gaussians. phase_reference.decode_samples
        # translates physical outputs from the minimum to the sampled root.
        'target_root_delay_residual': TargetDelayResidualPhasePolicy,
    }
    if mode not in classes:
        raise ValueError(f"Unknown phase_parameterization {mode!r}")
    policy_class = classes[mode]
    if hasattr(config, 'exploration_std_scale'):
        policy_class = {
            'absolute': AnnealedAbsolutePhasePolicy,
            'target_residual': AnnealedResidualPhasePolicy,
            'target_delay_residual': AnnealedDelayResidualPhasePolicy,
            'target_root_delay_residual': AnnealedDelayResidualPhasePolicy,
        }[mode]
    if getattr(config, 'encode_root', False):
        if mode != 'target_root_delay_residual' or not config.topology_input_mode:
            raise ValueError("Root encoding requires the supplied-topology root phase mode")
        policy_class = (AnnealedRootAwarePhasePolicy if hasattr(config, 'exploration_std_scale')
                        else RootAwareDelayResidualPhasePolicy)
    return policy_class(config).to(config.device)
