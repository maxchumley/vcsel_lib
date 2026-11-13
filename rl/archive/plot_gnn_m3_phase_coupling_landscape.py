#%%
"""Plot the M=3 GNN coupling budget over requested target-phase space.

The policy is evaluated directly on a dense grid of requested three-laser
phase patterns. The plot coordinates are target-phase differences analogous
to the coupling-phase coordinates in Fig. 3 of arXiv:2609.15752:

    x = (theta_12* - theta_23*) / pi
    y = (theta_13* - theta_23*) / pi

where the star denotes the requested target phase. The color is the GNN's
predicted coupling budget, normalized using the paper's K0:

    b_tot = (1 / M) * sum_nm |kappa_nm|,
    color = b_tot / K0,  K0 = 62.18 ns^-1.

The predicted directed coupling phases remain in the returned results but do
not set the plot coordinates. This shows predicted feasible budgets rather
than a brute-force synchronization-threshold sweep.
"""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import get_ipython
from IPython.display import clear_output, display

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_designer_variable_m import (
    _load_vcsel_class,
    decode_action,
    effective_maximum_kappa_per_link,
    encode_for_policy,
    load_checkpoint,
    make_vcsel_physical_parameters,
    simulate_with_vcsel,
)
from rl.paths import MODEL_DIR


# ---------------------------------------------------------------------------
# Settings to edit
# ---------------------------------------------------------------------------

MODEL_SUFFIX = (
    "gnn_M3-10_span5_edge_std_degree_scaled_stratified_span_"
    "no_M_features_lr1e-4_pcgrad_aggressive_min_coupling_lr1e-5_pcgrad"
)
CHECKPOINT_KIND = "current"  # "best", "current", or "final"
N_LASERS = 3

# Detunings used in Figs. 2 and 3 of arXiv:2609.15752, retained in the paper's
# laser-label order. The inference function handles the model's internal
# detuning ordering and restores the matrices to this original label order.
DETUNINGS_GHZ = np.array([1.66, -0.33, -1.33], dtype=float)

TARGET_GRID_POINTS = 500
INFERENCE_BATCH_SIZE = 4096

COLORMAP = "hot_r"
# Fig. 3 uses K0=62.18 GHz. Since 1 GHz = 1 ns^-1, the same value applies here.
PAPER_K0_PER_NS = 62.18
# True: plot b_tot/K0 from 0 to 1, matching the paper's normalized colorbar.
# False: plot b_tot from 0 to the grid's maximum in physical ns^-1 units.
NORMALIZE_BUDGET_TO_PAPER_K0 = False
# Fixed physical-unit range for the GNN budget map when normalization is off.
GNN_BUDGET_COLOR_MAX_PER_NS = 16.631


def checkpoint_path() -> Path:
    """Resolve the checkpoint without accidentally choosing a refinement."""
    checkpoint = MODEL_DIR / (
        "conditional_coupling_variable_m_3to10_lasers_"
        f"{CHECKPOINT_KIND}_{MODEL_SUFFIX}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def targets_from_phase_coordinates(
    x_coordinate: np.ndarray,
    y_coordinate: np.ndarray,
) -> np.ndarray:
    """Convert the plotted coordinates into three absolute target phases.

    With theta_ij = theta_i - theta_j and theta_3 as the global reference,

        theta_12 = pi*y,
        theta_23 = pi*(y - x),
        (theta_1, theta_2, theta_3) = (pi*(2*y - x), pi*(y - x), 0).

    Only relative phases enter the policy, so choosing theta_3=0 is arbitrary.
    """
    theta_1 = np.pi * (2.0 * y_coordinate - x_coordinate)
    theta_2 = np.pi * (y_coordinate - x_coordinate)
    theta_3 = np.zeros_like(theta_1)
    return np.column_stack((theta_1, theta_2, theta_3))


def coupling_budget_per_ns(kappa_per_ns: np.ndarray) -> np.ndarray:
    """Return the paper's mean coupling strength per laser."""
    off_diagonal = ~np.eye(N_LASERS, dtype=bool)
    return kappa_per_ns[:, off_diagonal].sum(axis=1) / float(N_LASERS)


def predict_coupling_matrices(
    policy: torch.nn.Module,
    config,
    target_phases_rad: np.ndarray,
    detunings_ghz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the deterministic policy mean in memory-bounded batches.

    The GNN operates on detuning-ordered lasers. Inputs are reordered for the
    policy, then decoded matrices are restored to the user's laser labels.
    """
    kappa_matrices: list[np.ndarray] = []
    phase_matrices: list[np.ndarray] = []
    policy.eval()
    detuning_order = np.argsort(detunings_ghz, kind="stable")
    inverse_order = np.argsort(detuning_order)
    ordered_detunings_ghz = detunings_ghz[detuning_order]

    with torch.inference_mode():
        for start in range(0, len(target_phases_rad), INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, len(target_phases_rad))
            targets = target_phases_rad[start:stop, detuning_order]
            detunings = np.broadcast_to(
                ordered_detunings_ghz, (stop - start, N_LASERS)
            )
            encoded = encode_for_policy(
                targets,
                policy,
                config,
                detunings,
            )
            actions = policy(encoded).cpu().numpy()

            magnitude_gates = None
            if getattr(policy, "enable_sparse_gates", False):
                magnitude_gates = (
                    policy.deterministic_gates(encoded).cpu().numpy()
                )

            kappa_per_ns, phi_p_rad = decode_action(
                actions,
                config.maximum_kappa_per_ns,
                N_LASERS,
                config.force_symmetric_kappa,
                config.force_symmetric_phi_p,
                config.balanced_coupling,
                magnitude_gates=magnitude_gates,
                normalize_incoming_coupling_by_degree=(
                    config.normalize_incoming_coupling_by_degree
                ),
            )
            kappa_matrices.append(
                kappa_per_ns[:, inverse_order, :][:, :, inverse_order]
            )
            phase_matrices.append(
                phi_p_rad[:, inverse_order, :][:, :, inverse_order]
            )

    return (
        np.concatenate(kappa_matrices, axis=0),
        np.concatenate(phase_matrices, axis=0),
    )


def main() -> dict[str, object]:
    """Load the GNN and display normalized budget over target-phase space."""
    if N_LASERS != 3:
        raise ValueError("This two-coordinate plot is defined for M=3")
    if TARGET_GRID_POINTS < 2:
        raise ValueError("TARGET_GRID_POINTS must be at least 2")

    detunings_ghz = np.asarray(DETUNINGS_GHZ, dtype=float)
    if detunings_ghz.shape != (N_LASERS,):
        raise ValueError(f"DETUNINGS_GHZ must have shape ({N_LASERS},)")
    detunings_ghz = detunings_ghz - detunings_ghz.mean()

    checkpoint = checkpoint_path()
    policy, _, saved_config, metadata = load_checkpoint(
        checkpoint,
        device="cpu",
        architecture="gnn",
    )
    config = replace(
        saved_config,
        n_lasers=N_LASERS,
        device="cpu",
        n_jobs=1,
        jupyter_mode=False,
    )

    target_coordinate = np.linspace(-1.0, 1.0, TARGET_GRID_POINTS)
    target_x_grid, target_y_grid = np.meshgrid(
        target_coordinate,
        target_coordinate,
        indexing="xy",
    )
    targets = targets_from_phase_coordinates(
        target_x_grid.ravel(), target_y_grid.ravel()
    )
    kappa_per_ns, phi_p_rad = predict_coupling_matrices(
        policy,
        config,
        targets,
        detunings_ghz,
    )
    budget_per_ns = coupling_budget_per_ns(kappa_per_ns)
    median_budget_per_ns = float(np.median(budget_per_ns))
    normalized_budget = budget_per_ns / PAPER_K0_PER_NS
    plotted_budget = (
        normalized_budget
        if NORMALIZE_BUDGET_TO_PAPER_K0
        else budget_per_ns
    )
    color_grid = plotted_budget.reshape(
        TARGET_GRID_POINTS,
        TARGET_GRID_POINTS,
    )
    color_maximum = (
        1.0
        if NORMALIZE_BUDGET_TO_PAPER_K0
        else GNN_BUDGET_COLOR_MAX_PER_NS
    )
    colorbar_label = (
        r"predicted $\kappa_{\mathrm{tot}}/K_0$, "
        r"$K_0=62.18\,\mathrm{ns}^{-1}$"
        if NORMALIZE_BUDGET_TO_PAPER_K0
        else r"predicted $\kappa_{\mathrm{tot}}$ ($\mathrm{ns}^{-1}$)"
    )
    detuning_text = ", ".join(f"{value:g}" for value in detunings_ghz)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
        }
    ):
        figure, axis = plt.subplots(
            figsize=(6.6, 5.5),
            dpi=200,
            constrained_layout=True,
        )
        image = axis.imshow(
            color_grid,
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            interpolation="nearest",
            aspect="equal",
            cmap=COLORMAP,
            vmin=0.0,
            vmax=color_maximum,
        )
        axis.set_xlabel(r"$(\theta_{12}^{*}-\theta_{23}^{*})/\pi$")
        axis.set_ylabel(r"$(\theta_{13}^{*}-\theta_{23}^{*})/\pi$")
        axis.set_title(
            "GNN coupling budget over target-phase space, $M=3$\n"
            + rf"$\delta/2\pi=[{detuning_text}]$ GHz; "
            + rf"median $\kappa_{{\mathrm{{tot}}}}={median_budget_per_ns:.3f}\,"
            + r"\mathrm{ns}^{-1}$"
        )
        colorbar = figure.colorbar(image, ax=axis, pad=0.025)
        colorbar.set_label(colorbar_label)

    plt.show()

    per_link_maximum = effective_maximum_kappa_per_link(
        config.maximum_kappa_per_ns,
        N_LASERS,
        config.normalize_incoming_coupling_by_degree,
    )
    print(f"Loaded: {checkpoint.name}")
    print(
        f"Checkpoint iteration {metadata['iteration']}; held-out reward "
        f"{metadata['held_out_reward']:.4f}"
    )
    print(f"Centered detunings (GHz): {detunings_ghz}")
    print(f"Physical per-link upper bound: {per_link_maximum:.3f} ns^-1")
    print(
        "Predicted coupling-budget range: "
        f"{budget_per_ns.min():.3f} to {budget_per_ns.max():.3f} ns^-1"
    )
    print(f"Median coupling budget: {median_budget_per_ns:.3f} ns^-1")
    print(
        "Normalized budget range: "
        f"{normalized_budget.min():.4f} to {normalized_budget.max():.4f}"
    )
    return {
        "checkpoint": checkpoint,
        "metadata": metadata,
        "config": config,
        "detunings_ghz": detunings_ghz,
        "target_x_grid": target_x_grid,
        "target_y_grid": target_y_grid,
        "target_phases_rad": targets,
        "kappa_per_ns": kappa_per_ns,
        "phi_p_rad": phi_p_rad,
        "coupling_budget_per_ns": budget_per_ns,
        "median_coupling_budget_per_ns": median_budget_per_ns,
        "normalized_coupling_budget": normalized_budget,
        "plotted_budget_grid": color_grid,
        "budget_is_normalized": NORMALIZE_BUDGET_TO_PAPER_K0,
        "figure": figure,
    }


if __name__ == "__main__":
    results = main()


#%% Legacy independent-restart scan (definitions only; not executed)
"""Legacy threshold scan retained only for comparison with continuation.

This cell scans reciprocal coupling phases for uniform and detuning-
proportional magnitude patterns. Unlike the target-conditioned GNN plot above,
the requested locked phase is unrestricted. The reported threshold is the
smallest tested coupling budget for which the lasers frequency-lock and their
relative phases become stationary.

Each budget is independently ramped from a free-running initial state, so this
cell is intentionally not executed by the script. The active continuation
implementation is in the following cell.
"""


# Forward continuation grid and budget schedule. The active scan uses every
# integer coupling budget from 0 through 60 ns^-1, as in the paper.
PAPER_SCAN_PHASE_POINTS = 50
PAPER_SCAN_MAX_BUDGET_PER_NS = 70.0
PAPER_SCAN_COARSE_BUDGET_STEP_PER_NS = 1.0
# Retained only for the inactive legacy independent-restart implementation.
PAPER_SCAN_BISECTION_STEPS = 1
# Simulate the complete phase grid in one vectorized integrator call at each
# continuation budget. Later calls automatically shrink as cases lock.
PAPER_SCAN_BATCH_SIZE = PAPER_SCAN_PHASE_POINTS**2

# Match the paper's numerical locking test as closely as the current simulator
# outputs allow: final mean frequency spread below 1% of the free-running
# spread, plus stationary relative phases over the final 50 delay periods.
PAPER_SCAN_TAIL_DELAYS = 50.0
PAPER_SCAN_FREQUENCY_SPREAD_FRACTION = 0.01
PAPER_SCAN_PHASE_STABILITY = 0.90


def reciprocal_topology_weights(
    detunings_ghz: np.ndarray,
    topology: str,
) -> np.ndarray:
    """Return normalized weights for pairs (12, 13, 23)."""
    if topology == "uniform":
        return np.full(3, 1.0 / 3.0)
    if topology == "proportional":
        pair_differences = np.array(
            [
                abs(detunings_ghz[0] - detunings_ghz[1]),
                abs(detunings_ghz[0] - detunings_ghz[2]),
                abs(detunings_ghz[1] - detunings_ghz[2]),
            ],
            dtype=float,
        )
        return pair_differences / pair_differences.sum()
    raise ValueError("topology must be 'uniform' or 'proportional'")


def reciprocal_coupling_matrices(
    budgets_per_ns: np.ndarray,
    phi_12_rad: np.ndarray,
    phi_13_rad: np.ndarray,
    topology_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build reciprocal matrices with the requested paper coupling budget.

    The pair weights sum to one. Setting each reciprocal pair magnitude to

        kappa_pair = (M * b_tot / 2) * pair_weight

    guarantees ``sum_nm |kappa_nm| / M == b_tot``.
    """
    budgets = np.asarray(budgets_per_ns, dtype=float)
    phi_12 = np.asarray(phi_12_rad, dtype=float)
    phi_13 = np.asarray(phi_13_rad, dtype=float)
    if not (budgets.shape == phi_12.shape == phi_13.shape):
        raise ValueError("budgets and coupling phases must have matching shapes")

    cases = len(budgets)
    kappa_per_ns = np.zeros((cases, N_LASERS, N_LASERS), dtype=float)
    phi_p_rad = np.zeros_like(kappa_per_ns)
    pair_magnitudes = (
        0.5 * N_LASERS * budgets[:, None] * topology_weights[None, :]
    )
    pair_indices = ((0, 1), (0, 2), (1, 2))
    pair_phases = np.column_stack(
        (phi_12, phi_13, np.zeros(cases, dtype=float))
    )
    for pair_index, (first, second) in enumerate(pair_indices):
        magnitude = pair_magnitudes[:, pair_index]
        phase = pair_phases[:, pair_index]
        kappa_per_ns[:, first, second] = magnitude
        kappa_per_ns[:, second, first] = magnitude
        phi_p_rad[:, first, second] = phase
        phi_p_rad[:, second, first] = phase
    return kappa_per_ns, phi_p_rad


def paper_style_locking_test(
    budgets_per_ns: np.ndarray,
    phi_12_rad: np.ndarray,
    phi_13_rad: np.ndarray,
    topology_weights: np.ndarray,
    detunings_ghz: np.ndarray,
    scan_config,
) -> np.ndarray:
    """Return whether each reciprocal design reaches a locked steady state."""
    locked_parts: list[np.ndarray] = []
    for start in range(0, len(budgets_per_ns), PAPER_SCAN_BATCH_SIZE):
        stop = min(start + PAPER_SCAN_BATCH_SIZE, len(budgets_per_ns))
        kappa_per_ns, phi_p_rad = reciprocal_coupling_matrices(
            budgets_per_ns[start:stop],
            phi_12_rad[start:stop],
            phi_13_rad[start:stop],
            topology_weights,
        )
        batch_detunings = np.broadcast_to(
            detunings_ghz,
            (stop - start, N_LASERS),
        )
        time_seconds, states, frequencies_nd, _ = simulate_with_vcsel(
            kappa_per_ns,
            phi_p_rad,
            scan_config,
            progress=False,
            smooth_frequencies=True,
            detuning_distribution_ghz=batch_detunings,
        )
        frequencies_ghz = frequencies_nd / (
            2.0
            * np.pi
            * scan_config.photon_lifetime_seconds
            * 1.0e9
        )
        tail_start_seconds = (
            time_seconds[-1]
            - PAPER_SCAN_TAIL_DELAYS * scan_config.delay_seconds
        )
        tail_mask = time_seconds >= tail_start_seconds
        if not np.any(tail_mask):
            raise RuntimeError("simulation output does not contain the tail")

        mean_frequencies = frequencies_ghz[:, :, tail_mask].mean(axis=2)
        final_frequency_spread = np.ptp(mean_frequencies, axis=1)
        initial_frequency_spread = max(float(np.ptp(detunings_ghz)), 1.0e-12)
        frequency_locked = final_frequency_spread <= (
            PAPER_SCAN_FREQUENCY_SPREAD_FRACTION
            * initial_frequency_spread
        )

        phases_rad = states[:, 2::3, :][:, :, tail_mask]
        relative_phases = np.angle(
            np.exp(1j * (phases_rad - phases_rad[:, :1, :]))
        )
        phase_resultant = np.abs(
            np.mean(np.exp(1j * relative_phases), axis=2)
        )
        phase_locked = np.min(phase_resultant[:, 1:], axis=1) >= (
            PAPER_SCAN_PHASE_STABILITY
        )
        finite = np.all(np.isfinite(states[:, :, tail_mask]), axis=(1, 2))
        locked_parts.append(frequency_locked & phase_locked & finite)
    return np.concatenate(locked_parts)


def threshold_map_for_topology(
    topology: str,
    phi_12_rad: np.ndarray,
    phi_13_rad: np.ndarray,
    detunings_ghz: np.ndarray,
    scan_config,
) -> np.ndarray:
    """Find the minimum locking budget for every coupling-phase pair."""
    weights = reciprocal_topology_weights(detunings_ghz, topology)
    cases = len(phi_12_rad)
    lower_budget = np.zeros(cases, dtype=float)
    upper_budget = np.full(cases, np.nan, dtype=float)

    coarse_budgets = np.arange(
        0.0,
        PAPER_SCAN_MAX_BUDGET_PER_NS
        + 0.5 * PAPER_SCAN_COARSE_BUDGET_STEP_PER_NS,
        PAPER_SCAN_COARSE_BUDGET_STEP_PER_NS,
    )
    for budget in coarse_budgets:
        unresolved = np.flatnonzero(~np.isfinite(upper_budget))
        if len(unresolved) == 0:
            break
        trial_budgets = np.full(len(unresolved), budget, dtype=float)
        locked = paper_style_locking_test(
            trial_budgets,
            phi_12_rad[unresolved],
            phi_13_rad[unresolved],
            weights,
            detunings_ghz,
            scan_config,
        )
        newly_locked = unresolved[locked]
        upper_budget[newly_locked] = budget
        still_unlocked = unresolved[~locked]
        lower_budget[still_unlocked] = budget
        print(
            f"{topology}: tested {budget:g} ns^-1; "
            f"locked {np.count_nonzero(np.isfinite(upper_budget))}/{cases}"
        )

    refinable = np.flatnonzero(
        np.isfinite(upper_budget) & (upper_budget > lower_budget)
    )
    for refinement in range(PAPER_SCAN_BISECTION_STEPS):
        if len(refinable) == 0:
            break
        midpoint = 0.5 * (
            lower_budget[refinable] + upper_budget[refinable]
        )
        locked = paper_style_locking_test(
            midpoint,
            phi_12_rad[refinable],
            phi_13_rad[refinable],
            weights,
            detunings_ghz,
            scan_config,
        )
        upper_budget[refinable[locked]] = midpoint[locked]
        lower_budget[refinable[~locked]] = midpoint[~locked]
        print(
            f"{topology}: completed threshold refinement "
            f"{refinement + 1}/{PAPER_SCAN_BISECTION_STEPS}"
        )

    return upper_budget


def run_independent_threshold_scan_legacy() -> dict[str, object]:
    """Legacy independent-restart scan retained for comparison."""
    _, _, saved_config, metadata = load_checkpoint(
        checkpoint_path(),
        device="cpu",
        architecture="gnn",
    )
    # Larger output strides reduce memory without changing integration steps.
    scan_config = replace(
        saved_config,
        n_lasers=N_LASERS,
        device="cpu",
        n_jobs=1,
        jupyter_mode=False,
        save_every=max(int(saved_config.save_every), 20),
    )
    detunings_ghz = np.asarray(DETUNINGS_GHZ, dtype=float)
    detunings_ghz -= detunings_ghz.mean()

    phase_coordinate = np.linspace(-1.0, 1.0, PAPER_SCAN_PHASE_POINTS)
    phase_x_grid, phase_y_grid = np.meshgrid(
        phase_coordinate,
        phase_coordinate,
        indexing="xy",
    )
    # phi_23 is the zero reference, so the paper's plotted coordinates give
    # phi_12=pi*x and phi_13=pi*y directly.
    phi_12_rad = np.pi * phase_x_grid.ravel()
    phi_13_rad = np.pi * phase_y_grid.ravel()

    threshold_maps: dict[str, np.ndarray] = {}
    for topology in ("uniform", "proportional"):
        thresholds = threshold_map_for_topology(
            topology,
            phi_12_rad,
            phi_13_rad,
            detunings_ghz,
            scan_config,
        )
        threshold_maps[topology] = thresholds.reshape(
            PAPER_SCAN_PHASE_POINTS,
            PAPER_SCAN_PHASE_POINTS,
        )

    if NORMALIZE_BUDGET_TO_PAPER_K0:
        plotted_maps = {
            name: values / PAPER_K0_PER_NS
            for name, values in threshold_maps.items()
        }
        color_maximum = 1.0
        colorbar_label = (
            r"synchronization threshold $b_{\mathrm{tot}}/K_0$, "
            r"$K_0=62.18\,\mathrm{ns}^{-1}$"
        )
    else:
        plotted_maps = threshold_maps
        color_maximum = PAPER_SCAN_MAX_BUDGET_PER_NS
        colorbar_label = (
            r"synchronization threshold $b_{\mathrm{tot}}$ "
            r"($\mathrm{ns}^{-1}$)"
        )

    colormap = plt.get_cmap(COLORMAP).copy()
    colormap.set_bad("black")
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
        }
    ):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(11.0, 4.8),
            dpi=200,
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        image = None
        for axis, topology in zip(
            axes,
            ("uniform", "proportional"),
            strict=True,
        ):
            image = axis.imshow(
                np.ma.masked_invalid(plotted_maps[topology]),
                origin="lower",
                extent=(-1.0, 1.0, -1.0, 1.0),
                interpolation="nearest",
                aspect="equal",
                cmap=colormap,
                vmin=0.0,
                vmax=color_maximum,
            )
            axis.set_title(f"{topology.capitalize()} coupling")
            axis.set_xlabel(r"$(\phi_{12}-\phi_{23})/\pi$")
        axes[0].set_ylabel(r"$(\phi_{13}-\phi_{23})/\pi$")
        colorbar = figure.colorbar(image, ax=axes, pad=0.025)
        colorbar.set_label(colorbar_label)
        detuning_text = ", ".join(f"{value:g}" for value in detunings_ghz)
        figure.suptitle(
            "Paper-style coupling-phase threshold scan with checkpoint "
            "physics\n"
            + rf"$\delta/2\pi=[{detuning_text}]$ GHz"
        )

    plt.show()
    return {
        "checkpoint_metadata": metadata,
        "config": scan_config,
        "detunings_ghz": detunings_ghz,
        "phase_x_grid": phase_x_grid,
        "phase_y_grid": phase_y_grid,
        "threshold_per_ns": threshold_maps,
        "plotted_threshold": plotted_maps,
        "figure": figure,
    }


if __name__ == "__main__" and False:
    paper_comparison_results = run_independent_threshold_scan_legacy()


# Uniform-coupling continuation scan
"""Reproduce the paper's forward coupling-budget continuation protocol.

Each phase-grid point starts from a free-running delay history. At every
budget step, the final full-resolution two-delay history is passed into the
next step. Only uniform reciprocal coupling is scanned in this cell, and the
threshold map is refreshed after every completed budget.
"""


# Full 0%-to-100% coupling transition used at every continuation step.
PAPER_SCAN_COUPLING_RAMP_DELAYS = 150.0


def free_running_continuation_histories(
    case_count: int,
    detunings_ghz: np.ndarray,
    scan_config,
) -> np.ndarray:
    """Create one free-running delay history for every phase-grid point."""
    batch_detunings = np.broadcast_to(
        detunings_ghz,
        (case_count, N_LASERS),
    )
    physical_parameters = make_vcsel_physical_parameters(
        scan_config,
        detuning_distribution_ghz=batch_detunings,
    )
    vcsel = _load_vcsel_class()(physical_parameters)
    nondimensional_parameters = vcsel.scale_params()
    histories, _, _, _ = vcsel.generate_history(
        nondimensional_parameters,
        shape="FR",
        n_cases=case_count,
    )
    return histories


def evaluate_continuation_step(
    budget_per_ns: float,
    previous_budget_per_ns: float,
    phi_12_rad: np.ndarray,
    phi_13_rad: np.ndarray,
    detunings_ghz: np.ndarray,
    scan_config,
    initial_histories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance uniform-coupling cases from their preceding-budget histories."""
    locked_parts: list[np.ndarray] = []
    history_parts: list[np.ndarray] = []
    case_count = len(phi_12_rad)
    if initial_histories.shape[0] != case_count:
        raise ValueError("one continuation history is required per case")
    uniform_weights = reciprocal_topology_weights(
        detunings_ghz,
        "uniform",
    )

    for start in range(0, case_count, PAPER_SCAN_BATCH_SIZE):
        stop = min(start + PAPER_SCAN_BATCH_SIZE, case_count)
        batch_size = stop - start
        budgets = np.full(batch_size, budget_per_ns, dtype=float)
        # vcsel_lib writes the mutual-coupling phase as
        # theta_source - theta_receiver - phi_p, whereas the paper defines
        # the complex coefficient with +phi_nm. Negate the plotted paper
        # phases when passing them to the simulator.
        kappa_per_ns, phi_p_rad = reciprocal_coupling_matrices(
            budgets,
            -phi_12_rad[start:stop],
            -phi_13_rad[start:stop],
            uniform_weights,
        )
        batch_detunings = np.broadcast_to(
            detunings_ghz,
            (batch_size, N_LASERS),
        )
        physical_parameters = make_vcsel_physical_parameters(
            scan_config,
            detuning_distribution_ghz=batch_detunings,
        )
        vcsel_class = _load_vcsel_class()
        vcsel = vcsel_class(physical_parameters)
        nondimensional_parameters = vcsel.scale_params()
        nondimensional_parameters["kappa"] = (
            kappa_per_ns
            * 1.0e9
            * scan_config.photon_lifetime_seconds
        )
        nondimensional_parameters["kappa_case_dependent"] = True
        nondimensional_parameters["phi_p"] = phi_p_rad

        # All uniform-coupling links scale linearly with b_tot. Starting the
        # scalar ramp at b_previous / b_current makes the new integration begin
        # at exactly the preceding step's coupling matrix.
        initial_fraction = (
            previous_budget_per_ns / budget_per_ns
            if budget_per_ns > 0.0
            else 0.0
        )
        full_time_grid_seconds = (
            np.arange(nondimensional_parameters["steps"])
            * scan_config.time_step_seconds
        )
        nondimensional_parameters["kappa_ramp"] = vcsel_class.cosine_ramp(
            full_time_grid_seconds,
            t_start=0.0,
            rise_10_90=(
                0.8 * PAPER_SCAN_COUPLING_RAMP_DELAYS
                * scan_config.delay_seconds
            ),
            kappa_initial=initial_fraction,
            kappa_final=1.0,
        )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            (
                time_seconds,
                states,
                frequencies_nd,
                final_histories,
            ) = vcsel.integrate(
                initial_histories[start:stop],
                nd=nondimensional_parameters,
                progress=False,
                max_iter=1,
                smooth_freqs=True,
                return_final_history=True,
            )
        history_parts.append(final_histories)

        frequencies_ghz = frequencies_nd / (
            2.0
            * np.pi
            * scan_config.photon_lifetime_seconds
            * 1.0e9
        )
        tail_start_seconds = (
            time_seconds[-1]
            - PAPER_SCAN_TAIL_DELAYS * scan_config.delay_seconds
        )
        tail_mask = time_seconds >= tail_start_seconds
        if not np.any(tail_mask):
            raise RuntimeError("simulation output does not contain the tail")

        mean_frequencies = frequencies_ghz[:, :, tail_mask].mean(axis=2)
        final_frequency_spread = np.ptp(mean_frequencies, axis=1)
        initial_frequency_spread = max(
            float(np.ptp(detunings_ghz)),
            1.0e-12,
        )
        frequency_locked = final_frequency_spread <= (
            PAPER_SCAN_FREQUENCY_SPREAD_FRACTION
            * initial_frequency_spread
        )

        phases_rad = states[:, 2::3, :][:, :, tail_mask]
        relative_phases = np.angle(
            np.exp(1j * (phases_rad - phases_rad[:, :1, :]))
        )
        phase_resultant = np.abs(
            np.mean(np.exp(1j * relative_phases), axis=2)
        )
        phase_locked = np.min(phase_resultant[:, 1:], axis=1) >= (
            PAPER_SCAN_PHASE_STABILITY
        )
        finite = np.all(np.isfinite(states[:, :, tail_mask]), axis=(1, 2))
        locked_parts.append(frequency_locked & phase_locked & finite)

    return np.concatenate(locked_parts), np.concatenate(history_parts, axis=0)


def uniform_threshold_map_with_continuation(
    phi_12_rad: np.ndarray,
    phi_13_rad: np.ndarray,
    detunings_ghz: np.ndarray,
    scan_config,
    *,
    progress_callback=None,
) -> np.ndarray:
    """Find the first locked budget along one forward continuation sweep."""
    case_count = len(phi_12_rad)
    thresholds = np.full(case_count, np.nan, dtype=float)
    histories = free_running_continuation_histories(
        case_count,
        detunings_ghz,
        scan_config,
    )
    budgets = np.arange(
        0.0,
        PAPER_SCAN_MAX_BUDGET_PER_NS
        + 0.5 * PAPER_SCAN_COARSE_BUDGET_STEP_PER_NS,
        PAPER_SCAN_COARSE_BUDGET_STEP_PER_NS,
    )
    previous_budget = 0.0
    for step_index, budget in enumerate(budgets):
        unresolved = np.flatnonzero(~np.isfinite(thresholds))
        if len(unresolved) == 0:
            break
        locked, final_histories = evaluate_continuation_step(
            float(budget),
            previous_budget,
            phi_12_rad[unresolved],
            phi_13_rad[unresolved],
            detunings_ghz,
            scan_config,
            histories[unresolved],
        )
        histories[unresolved] = final_histories
        thresholds[unresolved[locked]] = budget
        previous_budget = float(budget)
        locked_count = int(np.count_nonzero(np.isfinite(thresholds)))
        print(
            f"uniform continuation: step {step_index + 1}/{len(budgets)}, "
            f"b_tot={budget:g} ns^-1, locked {locked_count}/{case_count}"
        )
        if progress_callback is not None:
            progress_callback(thresholds, float(budget), locked_count)
    return thresholds


def running_in_ipython_kernel() -> bool:
    """Return whether live figures should use notebook display updates."""
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def run_paper_style_threshold_scan() -> dict[str, object]:
    """Generate a live uniform-coupling continuation threshold map."""
    _, _, saved_config, metadata = load_checkpoint(
        checkpoint_path(),
        device="cpu",
        architecture="gnn",
    )
    scan_config = replace(
        saved_config,
        n_lasers=N_LASERS,
        device="cpu",
        n_jobs=1,
        jupyter_mode=False,
        save_every=max(int(saved_config.save_every), 20),
    )
    detunings_ghz = np.asarray(DETUNINGS_GHZ, dtype=float)
    detunings_ghz -= detunings_ghz.mean()

    phase_coordinate = np.linspace(-1.0, 1.0, PAPER_SCAN_PHASE_POINTS)
    phase_x_grid, phase_y_grid = np.meshgrid(
        phase_coordinate,
        phase_coordinate,
        indexing="xy",
    )
    phi_12_rad = np.pi * phase_x_grid.ravel()
    phi_13_rad = np.pi * phase_y_grid.ravel()

    if NORMALIZE_BUDGET_TO_PAPER_K0:
        color_maximum = 1.0
        colorbar_label = (
            r"synchronization threshold $b_{\mathrm{tot}}/K_0$, "
            r"$K_0=62.18\,\mathrm{ns}^{-1}$"
        )
    else:
        color_maximum = PAPER_SCAN_MAX_BUDGET_PER_NS
        colorbar_label = (
            r"synchronization threshold $b_{\mathrm{tot}}$ "
            r"($\mathrm{ns}^{-1}$)"
        )

    colormap = plt.get_cmap(COLORMAP).copy()
    colormap.set_bad("black")
    notebook_display = running_in_ipython_kernel()
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
        }
    ):
        figure, axis = plt.subplots(
            figsize=(6.5, 5.5),
            dpi=200,
            constrained_layout=True,
        )
        if notebook_display:
            plt.close(figure)
        image = axis.imshow(
            np.ma.masked_invalid(
                np.full(
                    (PAPER_SCAN_PHASE_POINTS, PAPER_SCAN_PHASE_POINTS),
                    np.nan,
                )
            ),
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            interpolation="nearest",
            aspect="equal",
            cmap=colormap,
            vmin=0.0,
            vmax=color_maximum,
        )
        axis.set_xlabel(r"$(\phi_{12}-\phi_{23})/\pi$")
        axis.set_ylabel(r"$(\phi_{13}-\phi_{23})/\pi$")
        colorbar = figure.colorbar(image, ax=axis, pad=0.025)
        colorbar.set_label(colorbar_label)
        detuning_text = ", ".join(f"{value:g}" for value in detunings_ghz)
        figure.suptitle(
            "Uniform-coupling continuation scan with checkpoint physics\n"
            + rf"$\delta/2\pi=[{detuning_text}]$ GHz"
        )

        def refresh_plot(
            thresholds: np.ndarray,
            current_budget: float,
            locked_count: int,
        ) -> None:
            plotted = (
                thresholds / PAPER_K0_PER_NS
                if NORMALIZE_BUDGET_TO_PAPER_K0
                else thresholds
            )
            image.set_data(
                np.ma.masked_invalid(
                    plotted.reshape(
                        PAPER_SCAN_PHASE_POINTS,
                        PAPER_SCAN_PHASE_POINTS,
                    )
                )
            )
            axis.set_title(
                "Uniform coupling\n"
                + rf"current $b_{{\mathrm{{tot}}}}={current_budget:g}$ "
                + rf"ns$^{{-1}}$; locked {locked_count}/{len(thresholds)}"
            )
            if notebook_display:
                clear_output(wait=True)
                display(figure)
            else:
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
                plt.pause(0.05)

        thresholds = uniform_threshold_map_with_continuation(
            phi_12_rad,
            phi_13_rad,
            detunings_ghz,
            scan_config,
            progress_callback=refresh_plot,
        )
        final_budget = min(
            PAPER_SCAN_MAX_BUDGET_PER_NS,
            PAPER_SCAN_COARSE_BUDGET_STEP_PER_NS
            * np.floor(
                PAPER_SCAN_MAX_BUDGET_PER_NS
                / PAPER_SCAN_COARSE_BUDGET_STEP_PER_NS
            ),
        )
        refresh_plot(
            thresholds,
            final_budget,
            int(np.count_nonzero(np.isfinite(thresholds))),
        )

    if not notebook_display:
        plt.show()
    plotted_threshold = (
        thresholds / PAPER_K0_PER_NS
        if NORMALIZE_BUDGET_TO_PAPER_K0
        else thresholds.copy()
    )
    return {
        "checkpoint_metadata": metadata,
        "config": scan_config,
        "detunings_ghz": detunings_ghz,
        "phase_x_grid": phase_x_grid,
        "phase_y_grid": phase_y_grid,
        "threshold_per_ns": {
            "uniform": thresholds.reshape(
                PAPER_SCAN_PHASE_POINTS,
                PAPER_SCAN_PHASE_POINTS,
            )
        },
        "plotted_threshold": {
            "uniform": plotted_threshold.reshape(
                PAPER_SCAN_PHASE_POINTS,
                PAPER_SCAN_PHASE_POINTS,
            )
        },
        "figure": figure,
    }


if __name__ == "__main__":
    paper_comparison_results = run_paper_style_threshold_scan()
