#%%
"""Plot M=3 phase reward versus detuning span for several phase targets.

Run this cell in VS Code Interactive. Every point uses one natural Bernoulli
gate draw, Gaussian mean strengths/phases, and the same relative detuning
pattern as the manual three-laser inspection case. All cases are simulated
together with the saved current checkpoint and the usual 500 ns horizon.
"""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_bernoulli_sparse import infer_designs, load_checkpoint
from rl.conditional_coupling_designer_variable_m import (
    calculate_phase_reward, simulate_with_vcsel,
)
from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import (
    config as training_config,
)
from rl.paths import RESULTS_DIR


CHECKPOINT_KIND = "current"  # "current", "best", or "final"
SIMULATION_TIME_NS = 500.0
INFERENCE_RANDOM_SEED = 123
SPAN_VALUES_GHZ = np.array([
    0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75,
    2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0,
])
# Preserve the ordering and relative spacing of the earlier manual example.
MANUAL_DETUNING_PATTERN_GHZ = np.array([1.5, -0.5, -1.5])
TARGETS_RAD = {
    "In-phase": np.array([0.0, 0.0, 0.0]),
    "Binary": np.array([0.0, np.pi, 0.0]),
    "Splay (120°)": np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]),
    "Arbitrary": np.array([0.0, 0.35 * np.pi, -0.8 * np.pi]),
}
OUTPUT_PATH = (
    RESULTS_DIR / "inference" / "bernoulli_sparse"
    / "phase_reward_vs_detuning_span_m3.png"
)


def main() -> dict:
    if CHECKPOINT_KIND not in {"current", "best", "final"}:
        raise ValueError("CHECKPOINT_KIND must be current, best, or final")
    checkpoint = getattr(training_config, f"{CHECKPOINT_KIND}_checkpoint_file")
    policy, saved_config, metadata = load_checkpoint(checkpoint, device="cpu")
    config = replace(
        saved_config, n_lasers=3, n_jobs=1, jupyter_mode=False,
        noise_amplitude=0.0,
        simulation_time_seconds=SIMULATION_TIME_NS * 1e-9,
    )
    pattern = np.asarray(MANUAL_DETUNING_PATTERN_GHZ, dtype=float)
    pattern -= pattern.mean()
    pattern /= np.ptp(pattern)

    target_rows, detuning_rows, kappa_rows, phi_rows = [], [], [], []
    for target in TARGETS_RAD.values():
        for span in SPAN_VALUES_GHZ:
            design = infer_designs(
                policy, target, span * pattern, config,
                count=1, active_fraction=None,
                seed=INFERENCE_RANDOM_SEED, continuous_mean=True,
            )
            target_rows.append(design["target_phases_rad"])
            detuning_rows.append(design["detuning_distribution_ghz"])
            kappa_rows.append(design["kappa_per_ns"][0])
            phi_rows.append(design["phi_p_rad"][0])

    time_seconds, states, _, _ = simulate_with_vcsel(
        np.stack(kappa_rows), np.stack(phi_rows), config,
        detuning_distribution_ghz=np.stack(detuning_rows), progress=True,
    )
    phase_reward, *_ = calculate_phase_reward(
        time_seconds, states, np.stack(target_rows), config,
    )
    rewards = phase_reward.reshape(len(TARGETS_RAD), len(SPAN_VALUES_GHZ))

    figure, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for label, values in zip(TARGETS_RAD, rewards):
        ax.plot(SPAN_VALUES_GHZ, values, marker="o", markersize=3.5,
                linewidth=1.8, label=label)
    ax.axhline(0.98, color="0.35", linestyle="--", linewidth=1,
               label="0.98 reference")
    ax.set(xlim=(SPAN_VALUES_GHZ.min(), SPAN_VALUES_GHZ.max()),
           ylim=(0, 1.02), xlabel="peak-to-peak detuning span (GHz)",
           ylabel="settled phase reward",
           title=("Three-laser phase reward versus detuning span\n"
                  "Fixed manual detuning order; one natural gate draw per point"))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, loc="lower left")
    print(f"Loaded {CHECKPOINT_KIND} checkpoint at iteration {metadata['iteration']}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    return dict(spans_ghz=SPAN_VALUES_GHZ, targets_rad=TARGETS_RAD,
                phase_rewards=rewards, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    phase_reward_span_results = main()
