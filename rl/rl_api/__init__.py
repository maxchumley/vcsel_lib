"""User-friendly, parity-first API for the current VCSEL RL experiment.

Nothing trains, loads a checkpoint, creates a run directory, or simulates on
import. Begin with Experiment(...).describe(); training requires .train().
"""
from .api import CouplingModel, Experiment
from .config import (ActorCriticConfig, ModelConfig, OptimizationConfig, PhaseCorrectionConfig, PPOConfig, RewardConfig, SamplingConfig,
                     ScheduleConfig, TopologyConfig, TrainingConfig, ValidationConfig)
from .data import Context, DesignBatch
from .evaluation import EvaluatedDesign, EvaluationBatch
from .models.topology import Topology
from .plotting.options import PlotOptions
from .simulation import prepare_vcsel

__all__ = ["Experiment", "CouplingModel", "Context", "Topology", "DesignBatch",
    "EvaluationBatch", "EvaluatedDesign", "prepare_vcsel", "TrainingConfig", "ModelConfig", "RewardConfig",
    "SamplingConfig", "OptimizationConfig", "PhaseCorrectionConfig", "ActorCriticConfig", "PPOConfig", "ScheduleConfig", "TopologyConfig", "ValidationConfig", "PlotOptions"]
