"""Stable paths shared by examples and notebook-style executions."""

from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent

BASICS_DIR = EXAMPLES_DIR / "basics"
BASICS_RESULTS_DIR = BASICS_DIR / "results"

COUPLING_DIR = EXAMPLES_DIR / "coupling"
COUPLING_DATA_DIR = COUPLING_DIR / "data"
COUPLING_RESULTS_DIR = COUPLING_DIR / "results"

INJECTION_DIR = EXAMPLES_DIR / "injection"
INJECTION_DATA_DIR = INJECTION_DIR / "data"
INJECTION_RESULTS_DIR = INJECTION_DIR / "results"

LINEWIDTH_DIR = EXAMPLES_DIR / "linewidth"
LINEWIDTH_RESULTS_DIR = LINEWIDTH_DIR / "results" / "linewidth_estimation"

STABILITY_DIR = EXAMPLES_DIR / "stability"
STABILITY_DATA_DIR = STABILITY_DIR / "data"
STABILITY_RESULTS_DIR = STABILITY_DIR / "results"

PAPER_DIR = EXAMPLES_DIR / "paper"
PAPER_DATA_DIR = PAPER_DIR / "data"
PAPER_RESULTS_DIR = PAPER_DIR / "results"

VISUALIZATION_DIR = EXAMPLES_DIR / "visualization"
VISUALIZATION_RESULTS_DIR = VISUALIZATION_DIR / "results"
