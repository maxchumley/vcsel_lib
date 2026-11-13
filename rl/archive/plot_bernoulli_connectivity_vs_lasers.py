"""Estimate connectivity of sampled Bernoulli-gate coupling matrices.

This is a topology-only study: it does not run the VCSEL dynamics. The current
checkpoint is copied once so a concurrent training save cannot change the
model partway through the sweep.
"""

from __future__ import annotations

import csv
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.conditional_coupling_bernoulli_sparse import infer_designs, load_checkpoint
from rl.conditional_coupling_run_variable_m_gnn_pcgrad_bernoulli_sparse import config as training_config
from rl.paths import RESULTS_DIR


LASER_COUNTS = (3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50)
DETUNING_SPANS_GHZ = (0.1, 1.0, 4.0)
DETUNING_PATTERNS = ("even", "random")
PHASE_PATTERNS = ("in_phase", "two_clusters", "random")
DRAWS_PER_CONDITION = 100
RANDOM_SEED = 20261010
OUTPUT_DIR = RESULTS_DIR / "analyses" / "bernoulli_connectivity"


def reachable_nodes(adjacency: np.ndarray, start: int) -> np.ndarray:
    """Return nodes reachable from one source in a directed adjacency matrix."""
    n_lasers = adjacency.shape[0]
    seen = np.zeros(n_lasers, dtype=bool)
    seen[start] = True
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in np.flatnonzero(adjacency[node] & ~seen):
            seen[neighbor] = True
            stack.append(int(neighbor))
    return seen


def connectivity(kappa: np.ndarray) -> tuple[bool, bool, bool, bool]:
    """Return weak, strong, 0-to-last, and last-to-0 connectivity."""
    # kappa[receiver, source] is coupling from source to receiver.
    directed = np.asarray(kappa).T > 0.0
    np.fill_diagonal(directed, False)
    weak = bool(reachable_nodes(directed | directed.T, 0).all())
    from_first = reachable_nodes(directed, 0)
    from_last = reachable_nodes(directed, len(directed) - 1)
    strong = weak and bool(from_first.all()) and bool(reachable_nodes(directed.T, 0).all())
    return weak, strong, bool(from_first[-1]), bool(from_last[0])


def phase_target(pattern: str, n_lasers: int, rng: np.random.Generator) -> np.ndarray:
    if pattern == "in_phase":
        return np.zeros(n_lasers)
    if pattern == "two_clusters":
        return np.r_[np.zeros((n_lasers + 1) // 2),
                     np.full(n_lasers // 2, np.pi)]
    if pattern == "random":
        return rng.uniform(-np.pi, np.pi, n_lasers)
    raise ValueError(pattern)


def detunings(pattern: str, span_ghz: float, n_lasers: int,
              rng: np.random.Generator) -> np.ndarray:
    if pattern == "even":
        values = np.linspace(-0.5, 0.5, n_lasers)
    elif pattern == "random":
        values = rng.uniform(-0.5, 0.5, n_lasers)
    else:
        raise ValueError(pattern)
    values = values - values.mean()
    return span_ghz * values / np.ptp(values)


def write_plot(rows: list[dict], path: Path, *, metric: str = "weak_connected") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                             constrained_layout=True)
    colors = {"in_phase": "#2471a3", "two_clusters": "#d35400",
              "random": "#239b56"}
    labels = {"in_phase": "In phase", "two_clusters": "Two clusters",
              "random": "Random phases"}
    for axis, span in zip(axes, DETUNING_SPANS_GHZ):
        for phase in PHASE_PATTERNS:
            selected = [row for row in rows
                        if row["detuning_span_ghz"] == span
                        and row["phase_pattern"] == phase]
            x = np.array(LASER_COUNTS)
            y = np.array([
                sum(row[metric] for row in selected if row["n_lasers"] == m)
                / sum(row["draws"] for row in selected if row["n_lasers"] == m)
                for m in x
            ])
            axis.plot(x, y, marker="o", markersize=4, linewidth=1.7,
                      color=colors[phase], label=labels[phase])
        axis.axvspan(10.5, max(LASER_COUNTS) + 1, color="#eeeeee", zorder=-1)
        axis.set_xscale("log")
        axis.set(title=f"Detuning span {span:g} GHz", xlabel="Number of lasers",
                 xlim=(2.8, max(LASER_COUNTS) + 5), ylim=(-0.03, 1.03),
                 xticks=LASER_COUNTS)
        axis.set_xticklabels([str(m) for m in LASER_COUNTS])
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(alpha=0.25)
    title = ("Bernoulli-gate coupling connectivity" if metric == "weak_connected"
             else "Directed path from laser M-1 to laser 0")
    axes[0].set_ylabel("Proportion weakly connected" if metric == "weak_connected"
                       else "Proportion with directed path M-1 to 0")
    axes[-1].legend(loc="lower right", frameon=True)
    fig.suptitle(f"{title}\n"
                 "Shaded region: array sizes beyond training (M > 10)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-snapshot", action="store_true",
                        help="Use the saved checkpoint snapshot for a comparable rerun")
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(training_config.current_checkpoint_file)
    if not source.is_file():
        raise FileNotFoundError(source)
    snapshot = OUTPUT_DIR / "current_checkpoint_snapshot.pt"
    if args.reuse_snapshot:
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
    else:
        shutil.copyfile(source, snapshot)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    policy, saved_config, checkpoint = load_checkpoint(snapshot, device="cpu")
    print(f"Checkpoint iteration {checkpoint['iteration']}; sha256 {digest}", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for m in LASER_COUNTS:
        config = replace(saved_config, n_lasers=m, n_jobs=1,
                         jupyter_mode=False, noise_amplitude=0.0)
        for span in DETUNING_SPANS_GHZ:
            for phase in PHASE_PATTERNS:
                for detuning_pattern in DETUNING_PATTERNS:
                    target = phase_target(phase, m, rng)
                    offsets = detunings(detuning_pattern, span, m, rng)
                    sample = infer_designs(
                        policy, target, offsets, config,
                        count=DRAWS_PER_CONDITION,
                        seed=int(rng.integers(0, 2**31)),
                        continuous_mean=True,
                    )
                    outcomes = [connectivity(matrix) for matrix in sample["kappa_per_ns"]]
                    rows.append(dict(n_lasers=m, detuning_span_ghz=span,
                                     phase_pattern=phase,
                                     detuning_pattern=detuning_pattern,
                                     draws=len(outcomes),
                                     weak_connected=sum(result[0] for result in outcomes),
                                     strong_connected=sum(result[1] for result in outcomes),
                                     first_to_last=sum(result[2] for result in outcomes),
                                     last_to_first=sum(result[3] for result in outcomes),
                                     mean_active_links=float(np.mean(sample["active_link_counts"])),
                                     expected_active_fraction=sample["expected_active_fraction"]))
        total = sum(row["weak_connected"] for row in rows if row["n_lasers"] == m)
        count = sum(row["draws"] for row in rows if row["n_lasers"] == m)
        print(f"M={m:>2}: {total}/{count} weakly connected ({total/count:.3f})", flush=True)

    csv_path = OUTPUT_DIR / "connectivity_by_condition.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = OUTPUT_DIR / "connected_proportion_vs_lasers.png"
    write_plot(rows, plot_path)
    reverse_plot_path = OUTPUT_DIR / "last_to_first_reachability_vs_lasers.png"
    write_plot(rows, reverse_plot_path, metric="last_to_first")
    metadata = dict(checkpoint_source=str(source), checkpoint_snapshot=str(snapshot),
                    checkpoint_sha256=digest, checkpoint_iteration=checkpoint["iteration"],
                    training_n_lasers=list(saved_config.training_n_lasers),
                    laser_counts=LASER_COUNTS, detuning_spans_ghz=DETUNING_SPANS_GHZ,
                    detuning_patterns=DETUNING_PATTERNS, phase_patterns=PHASE_PATTERNS,
                    draws_per_condition=DRAWS_PER_CONDITION, random_seed=RANDOM_SEED,
                    reused_snapshot=args.reuse_snapshot,
                    connectivity_definition="weak: positive coupling in either direction; "
                                            "strong: directed paths in both directions; "
                                            "last_to_first: directed path from M-1 to 0; "
                                            "kappa[receiver,source] gives source-to-receiver edge")
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {csv_path}, {plot_path}, and {reverse_plot_path}", flush=True)


if __name__ == "__main__":
    main()
