#%%
"""Inspect one deterministic edge-list design. M>100 is inference/reporting only."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.archive.conditional_coupling_sparse_graph_tolerance import (
    DENSE_SIMULATION_LIMIT, dense_matrices, infer_sparse_graph,
    load_checkpoint, topology_summary,
)
from rl.archive.conditional_coupling_run_variable_m_gnn_pcgrad_sparse_graph_scale_aware import make_config

CHECKPOINT = Path(make_config().current_checkpoint_file)
N_LASERS = 10
N_CLUSTERS = 1
DETUNING_SPAN_GHZ = 0.1
MANUAL_DETUNINGS_GHZ = None
ALLOWABLE_MAX_PHASE_ERROR_DEG = 20.0
MANUAL_Q = None  # None uses the learned q mean; 0 is the fixed chain.
INFERENCE_DEVICE = "auto"
RUN_VALIDATION = True
VALIDATION_TIME_NS = 500.0
SHOW_FAR_FIELD = True
EMITTER_SPACING_UM = 10.0
FAR_FIELD_ELEMENT_FWHM_DEG = 20.0
FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG = None
TOPOLOGY_BINS = 10


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def plot_edge_list(design, bins=TOPOLOGY_BINS, *, curved_layout=False, ax=None):
    """Show every small-graph edge or aggregate large graphs in O(E+B^2)."""
    from matplotlib.patches import FancyArrowPatch
    import textwrap
    m = len(design["target_phases_rad"])
    bins = min(m, bins) if m > 20 else m
    mapping = np.minimum(np.arange(m)*bins//m, bins-1)
    r, s = mapping[design["receiver_indices"]], mapping[design["source_indices"]]
    weights = np.zeros((bins, bins))
    phases = np.zeros((bins, bins), dtype=complex)
    np.add.at(weights, (r, s), design["kappa_per_ns"])
    np.add.at(phases, (r, s), design["kappa_per_ns"]*np.exp(1j*design["phi_p_rad"]))
    detuning = np.array([design["detuning_distribution_ghz"][mapping == i].mean() for i in range(bins)])
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 7) if curved_layout else (10, 4),
                               constrained_layout=True)
    else:
        fig = ax.figure
    x = np.arange(bins, dtype=float)
    y = -0.7 * max(bins - 1, 1) * np.sin(np.linspace(0, np.pi, bins)) if curved_layout else np.zeros(bins)
    maximum = weights.max(initial=1e-12)
    for receiver, source in zip(*np.nonzero(weights)):
        if receiver == source:
            continue
        # Same signed local curvature gives reciprocal links opposite sides.
        arrow = FancyArrowPatch((x[source], y[source]), (x[receiver], y[receiver]),
            connectionstyle=f"arc3,rad={0.28 if curved_layout else 0.55}", arrowstyle="-|>",
            mutation_scale=10,
            shrinkA=5 if curved_layout and bins > 30 else 10,
            shrinkB=5 if curved_layout and bins > 30 else 10,
            linewidth=0.4+2.4*weights[receiver, source]/maximum,
            color=plt.cm.twilight((np.angle(phases[receiver, source])+np.pi)/(2*np.pi)), alpha=0.7)
        ax.add_patch(arrow)
    nodes = ax.scatter(x, y, c=detuning, cmap="bwr",
                       s=65 if curved_layout and bins > 30 else 180, zorder=3)
    label_stride = max(1, int(np.ceil(bins / 20))) if curved_layout else 1
    for i in range(bins):
        if i % label_stride == 0 or i == bins - 1:
            if curved_layout and bins > 30:
                ax.annotate(str(i + 1), (x[i], y[i]), xytext=(0, -11),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=10, zorder=4)
            else:
                ax.text(x[i], y[i], str(i + 1), ha="center", va="center",
                        fontsize=10, zorder=4)
    radius = max(1.0, 0.35*bins)
    ax.set(xlim=(-0.7, bins-0.3),
           ylim=(-1.15 * max(bins - 1, 1), 0.6 * max(bins - 1, 1)) if curved_layout else (-radius, radius),
           title="Sparse directed couplings" if bins == m else "Detuning-binned sparse couplings")
    ax.axis("off")
    detuning_bar = fig.colorbar(nodes, ax=ax, label="mean detuning (GHz)", fraction=0.035)
    detuning_bar.ax.tick_params(labelsize=11)
    caption = "color = coupling phase; width = coupling strength"
    if bins != m:
        caption += f"; width uses summed rates; {int(np.count_nonzero(r == s))} within-bin edges omitted"
    if standalone:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=10)
    else:
        caption = "\n".join(textwrap.wrap(caption, width=52))
        ax.text(0.5, -0.10, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
    return fig


def passivity_test(design, reflectivity, round_trip_time_fs):
    """Optional small-M test using the existing LK calibration convention."""
    if not 0 < reflectivity < 1 or round_trip_time_fs <= 0:
        raise ValueError("Require 0<R<1 and a positive round-trip time in fs")
    kappa, phi = dense_matrices(design, len(design["target_phases_rad"]))
    transfer = kappa*round_trip_time_fs*1e-6*np.sqrt(reflectivity)/(1-reflectivity)*np.exp(-1j*phi)
    maximum = float(np.linalg.svd(transfer, compute_uv=False)[0])
    return dict(maximum_singular_value=maximum, passive=maximum <= 1+1e-10)


def main(checkpoint=CHECKPOINT, n_lasers=N_LASERS, run_validation=RUN_VALIDATION):
    policy, _, metadata = load_checkpoint(checkpoint, device=resolve_device(INFERENCE_DEVICE))
    if not 1 <= N_CLUSTERS <= n_lasers:
        raise ValueError("Require 1 <= N_CLUSTERS <= M")
    counts = np.full(N_CLUSTERS, n_lasers//N_CLUSTERS)
    counts[:n_lasers % N_CLUSTERS] += 1
    target = np.repeat(2*np.pi*np.arange(N_CLUSTERS)/N_CLUSTERS, counts)
    detunings = (np.linspace(-DETUNING_SPAN_GHZ/2, DETUNING_SPAN_GHZ/2, n_lasers)
                 if MANUAL_DETUNINGS_GHZ is None else np.asarray(MANUAL_DETUNINGS_GHZ))
    design = infer_sparse_graph(policy, target, detunings, ALLOWABLE_MAX_PHASE_ERROR_DEG,
                                q_override=MANUAL_Q)
    print(f"Checkpoint iteration {metadata['iteration']}; device={next(policy.parameters()).device}")
    print(json.dumps(topology_summary(design), indent=2))
    print("Single deterministic inference. q=1 means the maximum configured sparse budget.")
    plot_edge_list(design)
    simulation = None
    if run_validation and n_lasers <= DENSE_SIMULATION_LIMIT:
        # Import only for small-M validation; its existing plotting behavior is reused.
        from rl.conditional_coupling_run_variable_m import simulate_and_plot_best_design
        config = replace(policy.config, n_lasers=n_lasers, n_jobs=1)
        kappa, phi = dense_matrices(design, n_lasers)
        simulation = simulate_and_plot_best_design(
            {**design, "kappa_per_ns": kappa, "phi_p_rad": phi}, config,
            validation_delay_count=VALIDATION_TIME_NS/(config.delay_seconds*1e9),
            desired_state="sparse_graph_prototype", detuning_distribution_ghz=design["detuning_distribution_ghz"],
            dual_axis_power_panel=True, far_field_panel=SHOW_FAR_FIELD,
            emitter_spacing_m=EMITTER_SPACING_UM*1e-6,
            far_field_element_fwhm_deg=FAR_FIELD_ELEMENT_FWHM_DEG,
            far_field_final_slice_panel=SHOW_FAR_FIELD,
            far_field_slice_convolution_fwhm_deg=FAR_FIELD_SLICE_CONVOLUTION_FWHM_DEG,
            allowable_phase_error_deg=ALLOWABLE_MAX_PHASE_ERROR_DEG,
        )
    elif n_lasers > DENSE_SIMULATION_LIMIT:
        print("M > 100: topology inference only. Dynamic validation and dense passivity tests are unsupported.")
    plt.show()
    return dict(policy=policy, design=design, metadata=metadata, simulation=simulation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--m", type=int, default=N_LASERS)
    parser.add_argument("--inference-only", action="store_true")
    args, _ = parser.parse_known_args()
    sparse_graph_inspection_results = main(args.checkpoint, args.m, RUN_VALIDATION and not args.inference_only)
