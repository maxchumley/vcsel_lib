"""User-facing inputs and proposed designs, with explicit units and shapes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from .models.topology import Topology


@dataclass(frozen=True)
class Context:
    """One target/detuning problem; M is inferred from the vector lengths."""
    target_phases_rad: np.ndarray
    detunings_ghz: np.ndarray

    def __post_init__(self):
        target = np.array(self.target_phases_rad, dtype=float, copy=True)
        detuning = np.array(self.detunings_ghz, dtype=float, copy=True)
        if target.ndim != 1 or target.shape != detuning.shape or len(target) < 2:
            raise ValueError("Targets and detunings must be equal-length vectors for at least two lasers")
        if not np.isfinite(target).all() or not np.isfinite(detuning).all():
            raise ValueError("Targets and detunings must be finite")
        target.setflags(write=False); detuning.setflags(write=False)
        object.__setattr__(self, "target_phases_rad", target)
        object.__setattr__(self, "detunings_ghz", detuning)

    @property
    def n_lasers(self):
        """Laser count for this problem, independent of training array sizes."""
        return len(self.target_phases_rad)


@dataclass
class DesignBatch:
    """Proposals only: constructing this object does not run the simulator.

    Arrays have shape (count,M,M), with [receiver,source] indexing. Budgets
    sum all directed strengths once, matching the RL code (not reciprocal DE).
    """
    context: Context
    raw: dict[str, Any]
    config: Any
    output_directory: Path

    def __len__(self):
        return len(self.raw["kappa_per_ns"])

    @property
    def coupling_strengths_per_ns(self):
        return self.raw["kappa_per_ns"]

    @property
    def coupling_phases_rad(self):
        return self.raw["phi_p_rad"]

    @property
    def budgets_per_ns(self):
        return self.coupling_strengths_per_ns.sum(axis=(-2, -1))

    @property
    def active_link_counts(self):
        return self.raw["active_link_counts"]

    @property
    def topologies(self):
        """Actual binary masks, including enabled edges with tiny strengths."""
        from ._compat import core
        receiver, source = core.directed_link_indices(self.context.n_lasers)
        matrices = np.zeros_like(self.coupling_strengths_per_ns, dtype=bool)
        matrices[:, receiver, source] = self.raw["gates"].astype(bool)
        return matrices

    @property
    def gate_logits(self):
        """None for supplied-topology models; reserved for future learned gates."""
        return self.raw.get("gate_logits")

    @property
    def gate_probabilities(self):
        """Learned gate probabilities, not a misleading copy of supplied masks."""
        return self.raw.get("natural_gate_probabilities")

    def evaluate(self, **options):
        """Simulate these proposals; see evaluation.evaluate_designs for options."""
        from .evaluation import evaluate_designs
        return evaluate_designs(self, **options)

    def plot(self, index=0, *, output_directory=None, show=True, options=None):
        """Plot one proposal's matrices/topology without simulating dynamics."""
        from .plotting import plot_proposal
        return plot_proposal(self, index, output_directory=output_directory, show=show, options=options)

    def topology(self, index=0):
        """Return a validated reusable Topology object for one proposal."""
        return Topology.from_receiver_source(self.topologies[index])
