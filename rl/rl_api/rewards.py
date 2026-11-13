"""The two existing metrics, exposed without changing their definitions.

RMS is the learning reward. Cosine is retained for monitoring/inspection.
Both exclude the reference laser and use one global conjugate orientation.
"""
from ._compat import core

cosine_metrics = core.calculate_phase_reward
wrapped_rms_metrics = core.calculate_phase_rms_metrics

__all__ = ["cosine_metrics", "wrapped_rms_metrics"]
