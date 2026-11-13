"""Small public workflow: configure, train/load, propose, evaluate, plot."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import numpy as np
import torch

from ._compat import PACKAGE_DIR, engine, reference_source_status
from .config import TrainingConfig
from .data import Context, DesignBatch
from .models import Topology
from .phase_reference import apply_root_reference, roots_for_supplied_masks, uses_root_reference
from .root_context import condition_on_roots


def _device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda or mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if requested == "mps" and not (getattr(torch.backends, "mps", None)
                                   and torch.backends.mps.is_available()):
        raise RuntimeError("MPS is not available")
    return requested


class Experiment:
    """A fresh training recipe; construction/describe never starts training.

    Quick controls override the corresponding fields in a TrainingConfig.
    Outputs default to rl/rl_api/runs/<name>/, separate from the live RL run.
    The older current_training and paired_topologies names are aliases for
    Python defaults, not JSON files. Pass config to use the run file's settings.
    Pass vcsel=VCSEL(phys) to take shared SI parameters from a native model.
    Context-specific laser counts, detunings and coupling arrays are supplied
    by RL; the passed template's corresponding entries are replaced per batch.
    bernoulli_gates is reserved and fails explicitly for now.
    """
    def __init__(self, name, preset="current_training", *, config=None, vcsel=None, sizes=None,
                 pairs_per_size=None, topologies_per_pair=None,
                 actions_per_topology=None, output_directory=None):
        if preset == "bernoulli_gates":
            raise NotImplementedError("Bernoulli gate training is reserved; use current_training for parity first")
        if preset not in {"current_training", "paired_topologies"}:
            raise ValueError("Unknown preset; available: current_training, paired_topologies")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(name)):
            raise ValueError("name must be a filename-safe label starting with a letter or number")
        config = config or TrainingConfig()
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig")
        if vcsel is not None:
            config = replace(config, phys=vcsel.phys)
        changes = {key: value for key, value in dict(pairs_per_size=pairs_per_size,
            topologies_per_pair=topologies_per_pair, actions_per_topology=actions_per_topology).items()
            if value is not None}
        if changes:
            config = replace(config, sampling=replace(config.sampling, **changes))
        if sizes is not None:
            config = replace(config, sizes=tuple(sizes))
        self.name = str(name)
        self.config = config
        self.vcsel = vcsel
        self.directory = (Path(output_directory) if output_directory is not None
                          else PACKAGE_DIR / "runs" / self.name).resolve()

    @property
    def backend_config(self):
        """The complete validated numerical config, with isolated output paths."""
        return self.config.to_backend(self.directory)

    def describe(self):
        """Print and return the recipe's counts; does not create output files."""
        summary = self.config.summary()
        print(f"Experiment: {self.name}\nOutput: {self.directory}")
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ')}: {value}")
        return summary

    def verify_reference(self):
        """Report source fingerprints without changing any existing files."""
        return reference_source_status()

    def train(self, *, iterations=None, workers=None, live_plot=True, verify_reference=True,
              restart=False):
        """Start fresh; restart=True archives existing results before starting over.

        This initializes new weights and optimizer state. It does not resume
        a checkpoint. Optional iteration/worker overrides leave schedules fixed.
        """
        from .training import train
        updates = {k: v for k, v in dict(iterations=iterations, workers=workers).items() if v is not None}
        chosen = self
        if updates:
            config = replace(self.config, optimization=replace(self.config.optimization, **updates))
            chosen = Experiment(self.name, config=config, output_directory=self.directory)
        return train(chosen, live_plot=live_plot, verify_reference=verify_reference,
                     restart=restart)


class CouplingModel:
    """One variable-size policy; proposing designs does not simulate them."""
    def __init__(self, policy, config, *, metadata=None, output_directory=None):
        if not config.topology_input_mode:
            raise NotImplementedError("This release supports topology-input models only")
        self.policy = policy
        self.backend_config = config
        self.metadata = metadata or {}
        self.output_directory = Path(output_directory or PACKAGE_DIR / "runs/inspection").resolve()

    @classmethod
    def load(cls, path, *, device="auto", output_directory=None):
        """Load an existing or API checkpoint without resuming its optimizer.

        Preserve the existing loader's exploration-bound behavior. Deterministic
        means are unaffected; see docs/reproducibility.md before sampling actions
        from a checkpoint whose training SD bounds were annealed.
        """
        from .checkpoints import load
        path = Path(path).resolve()
        with torch.random.fork_rng(devices=[]):
            policy, config, metadata = load(path, device=_device(device))
        directory = output_directory or PACKAGE_DIR / "runs/inspection" / path.stem
        return cls(policy.eval(), config, metadata=metadata, output_directory=directory)

    def propose(self, context=None, *, target_phases_rad=None, detunings_ghz=None,
                topology=None, count=100, continuous_actions="mean", seed=None):
        """Sample rooted masks or use a supplied Topology, then predict couplings.

        Plain topology arrays use [receiver,source]. For the old inspector's
        opposite convention, pass Topology.from_source_receiver(matrix).
        Inferred outputs are sorted by detuning with targets/masks permuted
        together. Topology-input inference matches the existing inspector.
        Root-reference models retain each automatic graph's sampled root.
        For manual masks, use the lowest-detuned laser that reaches all nodes.
        """
        if context is not None and (target_phases_rad is not None or detunings_ghz is not None):
            raise ValueError("Pass a Context or target/detuning vectors, not both")
        if context is None:
            context = Context(target_phases_rad, detunings_ghz)
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        if continuous_actions not in {"mean", "sample"}:
            raise ValueError("continuous_actions must be mean or sample")
        config = replace(self.backend_config, n_lasers=context.n_lasers, n_jobs=1,
                         jupyter_mode=False, noise_amplitude=0.0)
        supplied = None
        roots = None
        if topology is not None:
            mask = topology if isinstance(topology, Topology) else Topology.from_receiver_source(topology)
            if mask.receiver_source.ndim == 3:
                count = len(mask.receiver_source)
            supplied = mask.as_source_receiver()
        elif getattr(config, 'topology_random_reachable', False):
            from .sampling import sample_random_reachable_topologies
            from ._compat import core
            root_options = dict(return_roots=True) if uses_root_reference(config) else {}
            masks = sample_random_reachable_topologies(1, count, context.n_lasers,
                np.random.default_rng(seed), 'cpu',
                min_fraction=config.training_topology_min_fraction,
                density_bias=config.training_topology_density_bias, **root_options)
            masks, roots = masks if root_options else (masks, None)
            receivers, sources = core.directed_link_indices(context.n_lasers)
            matrices = np.zeros((count, context.n_lasers, context.n_lasers), dtype=bool)
            matrices[:, sources, receivers] = masks[0].numpy()
            # Sampler labels refer to detuning order. The existing inference
            # path expects the supplied mask in the caller's original order.
            inverse = np.argsort(np.argsort(context.detunings_ghz, kind='stable'))
            supplied = matrices[:, inverse, :][:, :, inverse]
        if getattr(config, 'encode_root', False) and roots is None:
            from ._compat import core
            order = np.argsort(context.detunings_ghz, kind='stable')
            matrices = np.asarray(supplied)
            if matrices.ndim == 2:
                matrices = np.broadcast_to(matrices, (count, context.n_lasers, context.n_lasers))
            matrices = matrices[:, order, :][:, :, order]
            receiver, source = core.directed_link_indices(context.n_lasers)
            roots = roots_for_supplied_masks(matrices[:, source, receiver], context.n_lasers)[None]
        conditioned = condition_on_roots(self.policy, roots)
        raw = engine.infer_designs(conditioned, context.target_phases_rad, context.detunings_ghz,
            config, count=count, topology=supplied,
            continuous_mean=(continuous_actions == "mean" or getattr(config, 'exploration_std_scale', 1.) == 0.),
            seed=seed, active_fraction=None, sample_optional_gates=True)
        ordered = Context(raw["target_phases_rad"], raw["detuning_distribution_ghz"])
        if uses_root_reference(config):
            if roots is None:
                roots = roots_for_supplied_masks(raw['gates'], context.n_lasers)[None]
            raw['phi_p_rad'] = apply_root_reference(raw['phi_p_rad'][None], raw['gates'][None],
                ordered.detunings_ghz[None], roots, config)[0]
            raw['root_indices'] = roots[0].copy()
            centered = ordered.detunings_ghz-ordered.detunings_ghz.mean()
            raw['phase_reference_detunings_ghz'] = centered[roots[0]]
        return DesignBatch(ordered, raw, config, self.output_directory)
