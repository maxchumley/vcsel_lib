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


class CouplingConditionedPhase:
    """Reuse graph layers on executed strengths, then add a zero-start phase head."""
    def __init__(self, config):
        super().__init__(config)
        # Adding a zero-start head must not shift the fresh run's sampling RNG.
        with torch.random.fork_rng(devices=[]):
            self.coupling_phase_head = torch.nn.Linear(self.edge_output.in_features, 1)
        torch.nn.init.zeros_(self.coupling_phase_head.weight)
        torch.nn.init.zeros_(self.coupling_phase_head.bias)

    def phase_residual_for_coupling(self, features, kappa_per_ns):
        # The sampled strengths are a conditioning variable, not a pathwise
        # gradient. Their own REINFORCE scores still train strength selection.
        kappa = torch.as_tensor(kappa_per_ns, dtype=features.dtype,
                                device=features.device).detach()
        batch, size, _ = features.shape
        if kappa.shape != (batch, size, size):
            raise ValueError('Conditional strengths must have shape (batch,M,M)')
        receivers, sources = core.directed_link_indices(size)
        encoded = torch.log1p(kappa[:, receivers, sources] /
                             self.config.softplus_strength_scale_per_ns)
        states, local, receivers, sources = self._graph_context(features, encoded)
        edges = size*(size-1)
        if self.config.condition_on_n_lasers:
            low, high = min(self.config.training_n_lasers), max(self.config.training_n_lasers)
            position = 0. if low == high else 2*(size-low)/(high-low)-1
            sizes = features.new_tensor((position, 1./(size-1))).view(1,1,2)
            local = torch.cat((local, sizes.expand(batch,edges,2)), dim=-1)
        inputs = torch.cat((states[:,sources], states[:,receivers], local,
                            states.mean(1)[:,None].expand(-1,edges,-1)), dim=-1)
        latent = self.edge_latent(inputs.reshape(batch*edges,-1))
        residual = self.coupling_phase_head(latent).reshape(batch,edges)
        if getattr(self.config, 'reciprocal_pair_actions', False):
            from ..reciprocal_pairs import pair_layout
            canonical, expansion, _ = pair_layout(size,features.device)
            residual = residual[:,canonical][:,expansion]
        return residual


class ConditionalDelayResidualPhasePolicy(CouplingConditionedPhase, AnnealedDelayResidualPhasePolicy):
    """Density base phases plus a correction conditioned on the executed graph."""


class ConditionalResidualPhasePolicy(CouplingConditionedPhase, AnnealedResidualPhasePolicy):
    """The same conditional correction with a zero delay-frequency reference."""


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


class ReciprocalPairPolicy(ScaledExploration, engine.BernoulliSparseGNNPolicy):
    """Use the canonical orientation's edge head once for each unordered pair.

    The lower/higher detuning indices choose a consistent orientation. Both
    directions use its distribution; the opposite head prediction is unused.
    Keep the existing head weights/layers and all target inputs. Antisymmetric
    target offsets are absent, as they canceled in the previous projection.
    """
    def parameters_for_edges(self, features, **options):
        from ..reciprocal_pairs import pair_layout, expand_pair_actions
        outputs = super().parameters_for_edges(features, **options)
        canonical, expansion, _ = pair_layout(features.shape[1], features.device)
        edges = outputs[0].shape[-1] // 2
        columns = torch.cat((canonical, canonical + edges))
        mean = outputs[0][:, columns]
        pairs = len(canonical)
        if self.config.phase_parameterization == 'target_delay_residual':
            mean = torch.cat((mean[:, :pairs], mean[:, pairs:] +
                              delay_phase_offsets(features, self.config)), dim=-1)
        # Match the executed SD of averaging two equally wide directed draws.
        log_std = outputs[1][:, columns] - .5 * math.log(2.)
        return (expand_pair_actions(mean, expansion),
                expand_pair_actions(log_std, expansion), *outputs[2:])


class ConditionalReciprocalPairPhasePolicy(CouplingConditionedPhase, ReciprocalPairPolicy):
    """One conditional phase per reciprocal pair, copied to both directions."""


def build_policy(config):
    """Select the recorded phase convention, preserving all parameter keys/order.

    In particular, retain the two compatibility SD scalars: removing them now
    would alter model/optimizer state layout during a reproducibility refactor.
    """
    mode = getattr(config,'phase_parameterization','absolute')
    if getattr(config, 'condition_phases_on_coupling', False):
        if not getattr(config, 'density_control', False) or getattr(config, 'encode_root', False):
            raise ValueError('Conditional phases require a density policy without root encoding')
        classes = {'target_delay_residual': ConditionalDelayResidualPhasePolicy,
                   'target_residual': ConditionalResidualPhasePolicy}
        if mode not in classes:
            raise ValueError('Conditional phases require a zero or laser-1 reference')
        if getattr(config, 'reciprocal_pair_actions', False):
            return ConditionalReciprocalPairPhasePolicy(config).to(config.device)
        return classes[mode](config).to(config.device)
    if getattr(config, 'reciprocal_pair_actions', False):
        if mode not in {'target_residual', 'target_delay_residual'}:
            raise ValueError('Reciprocal pair policy requires a zero or laser-1 reference')
        return ReciprocalPairPolicy(config).to(config.device)
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
