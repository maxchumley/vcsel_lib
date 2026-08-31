"""Stable filesystem locations for RL source and generated artifacts."""

from pathlib import Path


RL_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = RL_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
DATA_DIR = ARTIFACTS_DIR / "data"


def ensure_artifact_dirs() -> None:
    """Create the ignored directories used by RL programs."""
    for directory in (MODEL_DIR, RESULTS_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)

