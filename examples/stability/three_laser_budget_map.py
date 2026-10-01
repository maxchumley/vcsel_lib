#%% Imports and scan functions (run this cell first in an interactive editor)
"""Find synchronization thresholds for reciprocal three-laser topologies.

Each grid point specifies the proportions of a12, a13, and a23. Their sum is
one; scaling by a trial budget gives a12+a13+a23 = B. For each connected
topology, scan B upward through 40 ns^-1 (configurable) and report the first
sampled value with a stable frequency-locked equilibrium. Black pixels are
disconnected or have no stable root found by the maximum budget. The estimate
depends on the budget step and equilibrium seeds. Noise is not included.

Coordinates cover the full three-link simplex without rejecting points:
    a12 = B * (1-u1)
    a13 = B * u1 * (1-u2)
    a23 = B * u1 * u2
where B is the sum of the three *unique reciprocal* link strengths in ns^-1.
The VCSEL coupling matrix has both directions for each link, so its six
off-diagonal entries sum to 2B. All coupling phases are zero.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from joblib import Parallel, delayed, parallel_config

if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = next(
        (p for p in (Path.cwd(), *Path.cwd().parents)
         if (p / "vcsel_lib.py").is_file()),
        None,
    )
    if REPO_ROOT is None:
        raise RuntimeError("Run the cell from the vcsel_lib repository")
sys.path.insert(0, str(REPO_ROOT))
from vcsel_lib import VCSEL


# Set the three laser frequencies here in GHz, before centering.
UNCENTERED_DETUNINGS_GHZ = np.array([-1.0, 0.0, 2.0])
TAU_P = 5.4e-12


def link_strengths(u1: np.ndarray | float, u2: np.ndarray | float,
                   budget_per_ns: np.ndarray | float) -> np.ndarray:
    """Return (..., 3) link strengths for scalar or broadcastable inputs."""
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)
    budget_per_ns = np.asarray(budget_per_ns)
    if (np.any((u1 < 0) | (u1 > 1))
            or np.any((u2 < 0) | (u2 > 1))
            or np.any(budget_per_ns <= 0)):
        raise ValueError("u1 and u2 must be in [0, 1]; budget must be positive")
    weights = np.stack(np.broadcast_arrays(
        1.0 - u1, u1 * (1.0 - u2), u1 * u2
    ), axis=-1)
    return np.asarray(budget_per_ns)[..., None] * weights


def coupling_matrix(links_per_ns: np.ndarray) -> np.ndarray:
    """Return (..., 3, 3) reciprocal coupling matrices in s^-1."""
    links_per_ns = np.asarray(links_per_ns)
    if links_per_ns.shape[-1] != 3:
        raise ValueError("last dimension must contain a12, a13, a23")
    matrices = np.zeros(links_per_ns.shape[:-1] + (3, 3), dtype=float)
    matrices[..., 0, 1] = matrices[..., 1, 0] = links_per_ns[..., 0]
    matrices[..., 0, 2] = matrices[..., 2, 0] = links_per_ns[..., 1]
    matrices[..., 1, 2] = matrices[..., 2, 1] = links_per_ns[..., 2]
    return matrices * 1e9


def physical_parameters() -> dict:
    detunings_ghz = np.asarray(UNCENTERED_DETUNINGS_GHZ, dtype=float)
    if detunings_ghz.shape != (3,) or not np.all(np.isfinite(detunings_ghz)):
        raise ValueError("UNCENTERED_DETUNINGS_GHZ must contain three finite values")
    centered_detunings_ghz = detunings_ghz - np.mean(detunings_ghz)
    q = 1.602e-19
    tau_n = 0.25e-9
    g0 = 8.75e5
    n0 = 2.86e5
    current = 0.9 * 3 * q / tau_n * (n0 + 1 / (g0 * TAU_P))
    return dict(
        tau_p=TAU_P, tau_n=tau_n, g0=g0, N0=n0, s=4e-6,
        beta=1e-3, kappa_c_mat=np.zeros((3, 3)),
        phi_p_mat=np.zeros((3, 3)), I=current, q=q, alpha=2,
        phase_model="standard_lk",
        delta=2 * np.pi * 1e9 * centered_detunings_ghz,
        coupling=1.0, self_feedback=0.0, noise_amplitude=0.0,
        dt=0.5 * TAU_P, Tmax=1e-8, tau=1e-9, N_lasers=3,
        show_output_size_message=False,
    )


def scan_point(vcsel: VCSEL, nd_base: dict, kappa_nd: np.ndarray,
               phase_count: int, freq_count: int, collocation: int,
               guesses: list[np.ndarray]) -> tuple[int, bool, list[np.ndarray]]:
    nd = dict(nd_base)
    nd["kappa"] = kappa_nd
    counts = dict(phase_count=phase_count, freq_count=freq_count,
                  adaptive_grid=False)
    _, roots, _ = vcsel.solve_equilibria(nd, counts=counts, guesses=guesses, n_jobs=1)
    if np.asarray(roots).size == 0:
        return 0, False, guesses
    next_guesses = [np.concatenate((root[1:6:2], root[6:8], root[-1:]))
                    for root in roots]
    for root in roots:
        _, eigenvalues = vcsel.compute_stability(
            root, nd, N=collocation, sparse=False, newton_maxit=20
        )
        # An empty spectrum cannot establish stability.
        finite = eigenvalues[np.isfinite(eigenvalues)]
        if finite.size == 0 or finite.size != eigenvalues.size:
            continue
        if np.max(finite.real) < 0:
            return len(roots), True, next_guesses
    return len(roots), False, next_guesses


def trial_budgets(budget_step: float, max_budget: float) -> np.ndarray:
    """Positive trial budgets, always including the exact maximum."""
    full_steps = int(np.floor(max_budget / budget_step))
    budgets = budget_step * np.arange(1, full_steps + 1, dtype=float)
    if budgets.size and np.isclose(budgets[-1], max_budget, rtol=1e-12):
        budgets[-1] = max_budget
    else:
        budgets = np.append(budgets, max_budget)
    return budgets


def scan_pixel(pixel_index: int, grid: int, u1: float, u2: float,
               fractions: np.ndarray, budgets: np.ndarray, nd_base: dict,
               phase_count: int, freq_count: int,
               collocation: int) -> tuple[int, dict]:
    """Search one topology independently; keep its budget sweep sequential."""
    row_index, column_index = divmod(pixel_index, grid)
    label = (f"row {row_index + 1}/{grid}, pixel {column_index + 1}/{grid} "
             f"(u1={u1:.3f}, u2={u2:.3f})")
    print(f"{label}: starting", flush=True)
    # Any two positive edges of a three-node triangle connect all nodes.
    connected = np.count_nonzero(fractions > 0) >= 2
    threshold = np.inf
    n_roots = 0
    n_trials = 0
    guesses = []
    if connected:
        vcsel = VCSEL(physical_parameters())
        trial_links = fractions[None, :] * budgets[:, None]
        trial_kappa_nd = coupling_matrix(trial_links) * TAU_P
        last_report = time.monotonic()
        for trial_index, budget in enumerate(budgets):
            n_roots, locked, guesses = scan_point(
                vcsel, nd_base, trial_kappa_nd[trial_index],
                phase_count, freq_count, collocation, guesses
            )
            n_trials = trial_index + 1
            if locked:
                threshold = budget
                break
            now = time.monotonic()
            if now - last_report >= 15:
                print(
                    f"{label}: searched through {budget:.2f}/{budgets[-1]:.2f} "
                    f"ns^-1 ({n_trials}/{len(budgets)} trials)",
                    flush=True,
                )
                last_report = now
    status = ("disconnected" if not connected else
              "locked" if np.isfinite(threshold) else "not_found_by_max")
    row = dict(u1=u1, u2=u2, a12_fraction=fractions[0],
               a13_fraction=fractions[1], a23_fraction=fractions[2],
               threshold_per_ns=threshold, n_roots_last=n_roots,
               n_trials=n_trials, status=status)
    outcome = (f"locked at {threshold:.2f} ns^-1" if status == "locked"
               else f"no stable root through {budgets[-1]:.2f} ns^-1"
               if status == "not_found_by_max" else "disconnected")
    trial_word = "trial" if n_trials == 1 else "trials"
    print(f"{label}: {outcome} ({n_trials} {trial_word})", flush=True)
    return pixel_index, row


def plot_map(rows: list[dict], grid: int, max_budget: float, output: Path) -> None:
    threshold = np.array([r["threshold_per_ns"] for r in rows]).reshape(grid, grid)
    masked = np.ma.masked_invalid(threshold)
    cmap = plt.colormaps["hot_r"].copy()
    cmap.set_bad("black")
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(11, 5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})
    im = ax.imshow(masked, origin="lower", extent=(0, 1, 0, 1),
                   interpolation="nearest", aspect="equal", cmap=cmap,
                   vmin=0)
    fig.colorbar(im, ax=ax, label=r"First stable locking budget (ns$^{-1}$)")
    ax.set(xlabel=r"$u_1$", ylabel=r"$u_2$",
           title="Three laser locking threshold map")
    valid = [r for r in rows if np.isfinite(r["threshold_per_ns"])]
    if valid:
        best = min(valid, key=lambda r: r["threshold_per_ns"])
        ax.plot(best["u1"], best["u2"], "co", markersize=8,
                markerfacecolor="none", markeredgewidth=2)
        values = [best["a12_fraction"], best["a13_fraction"], best["a23_fraction"]]
        bars.bar([r"$a_{12}$", r"$a_{13}$", r"$a_{23}$"], values)
        bars.set_title(f"Lowest threshold: {best['threshold_per_ns']:.2f} ns$^{{-1}}$")
    bars.set(ylabel="Pair coupling fraction", ylim=(0, 1))
    fig.suptitle(f"Detunings: 1, 2 GHz gaps; search through {max_budget:g} ns$^{{-1}}$; "
                 r"$\phi_{nm}=0$")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-step", type=float, default=0.5,
                        help="spacing of positive trial budgets in ns^-1 (default: 0.5)")
    parser.add_argument("--max-budget", type=float, default=40.0,
                        help="maximum total pair budget in ns^-1 (default: 40)")
    parser.add_argument("--grid", type=int, default=5,
                        help="grid points per coordinate (default: 5)")
    parser.add_argument("--phase-count", type=int, default=5)
    parser.add_argument("--freq-count", type=int, default=25)
    parser.add_argument("--collocation", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=4,
                        help="parallel heatmap pixels (default: 4; 1 disables parallelism)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "examples" / "stability" / "results" / "three_laser_budget_map")
    args = parser.parse_args(argv)
    if (not np.isfinite(args.budget_step) or args.budget_step <= 0
            or not np.isfinite(args.max_budget) or args.max_budget <= 0
            or min(args.grid, args.phase_count, args.freq_count, args.collocation) < 2):
        parser.error("budget-step and max-budget must be positive and finite; grid and solver counts must be >= 2")
    if args.jobs < 1:
        parser.error("jobs must be >= 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.linspace(0, 1, args.grid)
    budgets = trial_budgets(args.budget_step, args.max_budget)
    u1_grid, u2_grid = np.meshgrid(coordinates, coordinates, indexing="xy")
    fractions = link_strengths(u1_grid, u2_grid, 1.0).reshape(-1, 3)
    positions = np.column_stack((u1_grid.ravel(), u2_grid.ravel()))
    # SymPy-backed scaling is identical for every pixel. Compute it once in
    # the parent and pass the small nondimensional parameter dictionary on.
    nd_base = VCSEL(physical_parameters()).scale_params()
    pixel_count = args.grid * args.grid
    print(
        f"Scanning {args.grid}x{args.grid} pixels with {min(args.jobs, pixel_count)} "
        f"worker(s); {len(budgets)} budgets through {args.max_budget:g} ns^-1",
        flush=True,
    )
    rows = [None] * pixel_count
    completed_pixels = 0
    if args.jobs == 1:
        for pixel_index, (u1, u2) in enumerate(positions):
            result_index, row = scan_pixel(
                pixel_index, args.grid, float(u1), float(u2),
                fractions[pixel_index], budgets, nd_base,
                args.phase_count, args.freq_count, args.collocation
            )
            rows[result_index] = row
            completed_pixels += 1
            print(f"Completed pixels: {completed_pixels}/{pixel_count}", flush=True)
    else:
        # Independent pixels use separate processes. Restrict BLAS threads in
        # each process to avoid oversubscribing the machine during eigensolves.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            completed = Parallel(
                n_jobs=min(args.jobs, pixel_count),
                return_as="generator_unordered", batch_size=1,
            )(
                delayed(scan_pixel)(
                    pixel_index, args.grid, float(u1), float(u2),
                    fractions[pixel_index], budgets, nd_base,
                    args.phase_count, args.freq_count, args.collocation
                )
                for pixel_index, (u1, u2) in enumerate(positions)
            )
            for result_index, row in completed:
                rows[result_index] = row
                completed_pixels += 1
                print(f"Completed pixels: {completed_pixels}/{pixel_count}", flush=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    png_path = args.output.with_suffix(".png")
    plot_map(rows, args.grid, args.max_budget, png_path)
    print(f"Saved {csv_path} and {png_path}")
    return png_path


def saved_scan_csv() -> Path:
    """Use the current interactive scan output, or the default saved CSV."""
    if "figure_path" in globals():
        return Path(figure_path).with_suffix(".csv")
    return (REPO_ROOT / "examples" / "stability" / "results"
            / "three_laser_budget_map.csv")


def load_optimal_coupling(csv_path: Path) -> tuple[dict, np.ndarray]:
    """Select the same first minimum that the heatmap circles."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Run the scan first; threshold CSV not found: {csv_path}")
    with csv_path.open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    stable = [
        row for row in candidates
        if row.get("status", "locked") == "locked"
        and np.isfinite(float(row["threshold_per_ns"]))
    ]
    if not stable:
        raise RuntimeError(f"No stable topology was found in {csv_path}")
    best = min(stable, key=lambda row: float(row["threshold_per_ns"]))
    fractions = np.array([
        float(best["a12_fraction"]),
        float(best["a13_fraction"]),
        float(best["a23_fraction"]),
    ])
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Saved optimal coupling fractions are invalid")
    links_per_ns = float(best["threshold_per_ns"]) * fractions
    return best, links_per_ns


#%% Run the scan (run this cell after the imports/functions cell)
GRID = 5
BUDGET_STEP_PER_NS = 0.1
MAX_BUDGET_PER_NS = 50.0
PHASE_COUNT = 5
FREQ_COUNT = 25
COLLOCATION = 16
JOBS = 10

if __name__ == "__main__":
    if get_ipython() is None:
        # Terminal execution keeps support for the command-line flags.
        main()
    else:
        figure_path = main([
            "--grid", str(GRID),
            "--budget-step", str(BUDGET_STEP_PER_NS),
            "--max-budget", str(MAX_BUDGET_PER_NS),
            "--phase-count", str(PHASE_COUNT),
            "--freq-count", str(FREQ_COUNT),
            "--collocation", str(COLLOCATION),
            "--jobs", str(JOBS),
        ])
        from IPython.display import Image, display

        display(Image(filename=str(figure_path)))


# #%% Simulate the best topology and plot time traces (run after the scan cell)
# # This cell can also use an existing CSV from a previous scan. It is not run
# # when the script is launched from a terminal.
# SIMULATION_TMAX_SECONDS = 2.0e-7
# SIMULATION_DT_SECONDS = TAU_P
# SIMULATION_SAVE_EVERY = 1
# RAMP_START_DELAYS = 5.0
# RAMP_RISE_DELAYS = 10.0
# OPTICAL_WAVELENGTH_METERS = 910e-9


# def simulate_optimal_coupling(links_per_ns: np.ndarray):
#     """Ramp the exact optimal matrix from zero and integrate from FR history."""
#     phys = physical_parameters()
#     phys.update(
#         kappa_c_mat=coupling_matrix(links_per_ns),
#         dt=SIMULATION_DT_SECONDS,
#         Tmax=SIMULATION_TMAX_SECONDS,
#         save_every=SIMULATION_SAVE_EVERY,
#         noise_amplitude=0.0,
#     )
#     vcsel = VCSEL(phys)
#     nd = vcsel.scale_params()
#     simulation_time = np.arange(nd["steps"]) * phys["dt"]
#     nd["kappa_ramp"] = VCSEL.cosine_ramp(
#         simulation_time,
#         t_start=RAMP_START_DELAYS * phys["tau"],
#         rise_10_90=RAMP_RISE_DELAYS * phys["tau"],
#         kappa_initial=0.0,
#         kappa_final=1.0,
#     )
#     history, freq_hist, _, _ = vcsel.generate_history(
#         nd, shape="FR", n_cases=1
#     )
#     t, y, freqs = vcsel.integrate(
#         history, nd=nd, progress=True, max_iter=1, smooth_freqs=True
#     )
#     return phys, nd, t, y, freqs, freq_hist


# def plot_optimal_simulation(
#     best: dict, links_per_ns: np.ndarray, phys: dict, nd: dict,
#     t: np.ndarray, y: np.ndarray, freqs: np.ndarray,
#     freq_hist: np.ndarray, output: Path,
# ) -> None:
#     """Plot frequency, wrapped relative phase, and power as in simple_example."""
#     from scipy.constants import c, hbar

#     time_us = t * 1e6
#     photon_numbers = np.maximum(y[0, 1::3, :], 0.0)
#     phases = y[0, 2::3, :]
#     frequency_ghz = np.array(freqs[0], copy=True) * 1e-9 / (2 * np.pi * phys["tau_p"])
#     history_stride = max(1, int(nd.get("save_every", 1)))
#     frequency_history_ghz = freq_hist[0, :, ::history_stride]
#     history_length = min(frequency_history_ghz.shape[-1], frequency_ghz.shape[-1])
#     frequency_ghz[:, :history_length] = frequency_history_ghz[:, :history_length]

#     omega0 = 2 * np.pi * c / OPTICAL_WAVELENGTH_METERS
#     intensity_to_mw = (
#         1e3 * hbar * omega0
#         / (phys["g0"] * phys["tau_n"] * phys["tau_p"])
#     )
#     laser_power_mw = photon_numbers * intensity_to_mw
#     total_field = np.sum(
#         np.sqrt(photon_numbers) * np.exp(1j * phases), axis=0
#     )
#     total_power_mw = np.abs(total_field) ** 2 * intensity_to_mw

#     fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
#     for laser in range(3):
#         axes[0].plot(time_us, frequency_ghz[laser], label=f"Laser {laser + 1}")
#     axes[0].set(ylabel="Instantaneous frequency (GHz)")
#     axes[0].legend(loc="best")

#     for laser in (1, 2):
#         wrapped = np.angle(np.exp(1j * (phases[laser] - phases[0])))
#         axes[1].plot(time_us, wrapped,
#                      label=rf"$\phi_{{{laser + 1}}}-\phi_1$")
#     axes[1].set(ylabel="Wrapped relative phase (rad)", ylim=(-np.pi, np.pi))
#     axes[1].set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
#     axes[1].legend(loc="best")

#     for laser in range(3):
#         axes[2].plot(time_us, laser_power_mw[laser],
#                      label=rf"$P_{{{laser + 1}}}$")
#     axes[2].plot(time_us, total_power_mw, color="green", linewidth=2,
#                  label=r"$P_{\rm coherent,total}$")
#     axes[2].set(xlabel="Time (µs)", ylabel="Optical power (mW)")
#     axes[2].legend(loc="best")
#     budget_ramp = float(best["threshold_per_ns"]) * np.interp(
#         t, np.arange(nd["steps"]) * phys["dt"], nd["kappa_ramp"]
#     )
#     coupling_axis = axes[2].twinx()
#     coupling_axis.plot(time_us, budget_ramp, "k--", alpha=0.45,
#                        label="Pair coupling budget")
#     coupling_axis.set_ylabel(r"Coupling budget (ns$^{-1}$)")

#     for ax in axes:
#         ax.axvspan(0, 2 * phys["tau"] * 1e6, color="gray", alpha=0.12)
#         ax.grid(True, alpha=0.2)
#     fig.suptitle(
#         f"Best sampled topology: u1={float(best['u1']):.3f}, "
#         f"u2={float(best['u2']):.3f}, "
#         f"B={float(best['threshold_per_ns']):.2f} ns$^{{-1}}$\n"
#         f"a12={links_per_ns[0]:.2f}, a13={links_per_ns[1]:.2f}, "
#         f"a23={links_per_ns[2]:.2f} ns$^{{-1}}$"
#     )
#     fig.tight_layout()
#     output.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output, dpi=180, bbox_inches="tight")
#     plt.close(fig)


# if __name__ == "__main__" and get_ipython() is not None:
#     scan_csv = saved_scan_csv()
#     best_row, best_links_per_ns = load_optimal_coupling(scan_csv)
#     print("Optimal coupling matrix (ns^-1):")
#     print(coupling_matrix(best_links_per_ns) * 1e-9)
#     sim_phys, sim_nd, sim_t, sim_y, sim_freqs, sim_freq_hist = (
#         simulate_optimal_coupling(best_links_per_ns)
#     )
#     simulation_plot = scan_csv.with_name(scan_csv.stem + "_optimum_timeseries.png")
#     plot_optimal_simulation(
#         best_row, best_links_per_ns, sim_phys, sim_nd,
#         sim_t, sim_y, sim_freqs, sim_freq_hist, simulation_plot,
#     )
#     print(f"Saved {simulation_plot}")
#     from IPython.display import Image, display

#     display(Image(filename=str(simulation_plot)))


#%% Equilibrium branches for the optimal topology (run after the first cell)
# This cell reads the saved scan CSV directly; the time simulation cell need
# not be run. The solver sweeps the unique-link sum a12+a13+a23 while keeping
# a12:a13:a23 fixed; plots and CSV use the full symmetric-matrix sum, twice it.
BRANCH_BELOW_THRESHOLD_PER_NS = 6.0
BRANCH_ABOVE_THRESHOLD_PER_NS = 8.0
BRANCH_POINTS = 20
BRANCH_PHASE_COUNT = 5
BRANCH_FREQ_COUNT = 50
BRANCH_MAX_REFINE = 2
BRANCH_REFINE_FACTOR = 2
BRANCH_COLLOCATION = 30
BRANCH_NEWTON_MAXIT = 10000
BRANCH_STABILITY_THRESHOLD = 1e-10
BRANCH_SPECTRAL_SHIFT = 0.01 + 0.01j
BRANCH_ROOT_JOBS = -1


def branch_budgets(threshold_per_ns: float) -> np.ndarray:
    """Sweep around the saved optimum and include its exact sampled budget."""
    if threshold_per_ns <= 0 or BRANCH_POINTS < 2:
        raise ValueError("Threshold must be positive and BRANCH_POINTS >= 2")
    lower = max(1e-6, threshold_per_ns - BRANCH_BELOW_THRESHOLD_PER_NS)
    upper = threshold_per_ns + BRANCH_ABOVE_THRESHOLD_PER_NS
    return np.unique(np.append(np.linspace(lower, upper, BRANCH_POINTS),
                               threshold_per_ns))


def branch_observables(root: np.ndarray, tau: float) -> tuple[float, float]:
    """Common frequency and delayed coherent intensity, as in the tutorial."""
    intensity = np.maximum(root[1:6:2], 0.0)
    phases = np.r_[0.0, root[6:8]]
    omega = float(root[-1])
    total_field = np.sum(np.sqrt(intensity)
                         * np.exp(1j * (phases + omega * tau)))
    return omega / (2 * np.pi * TAU_P) * 1e-9, float(np.abs(total_field) ** 2)


def solve_optimal_branches(best: dict, links_per_ns: np.ndarray) -> list[dict]:
    """Continue roots across unique-link budgets; report full matrix sums."""
    threshold = float(best["threshold_per_ns"])
    fractions = links_per_ns / threshold
    budgets = branch_budgets(threshold)
    phys = physical_parameters()
    phys["sparse"] = True
    vcsel = VCSEL(phys)
    nd_base = vcsel.scale_params()
    counts = dict(phase_count=BRANCH_PHASE_COUNT,
                  freq_count=BRANCH_FREQ_COUNT, adaptive_grid=True,
                  max_refine=BRANCH_MAX_REFINE,
                  refine_factor=BRANCH_REFINE_FACTOR)
    guesses = []
    records = []
    for index, budget in enumerate(budgets, 1):
        nd = dict(nd_base)
        nd["kappa"] = coupling_matrix(fractions * budget) * TAU_P
        _, roots, _ = vcsel.solve_equilibria(
            nd, counts=counts, guesses=guesses, n_jobs=BRANCH_ROOT_JOBS
        )
        roots = np.asarray(roots)
        if roots.size:
            roots = np.atleast_2d(roots)
            guesses = [np.concatenate((root[1:6:2], root[6:8], root[-1:]))
                       for root in roots]
        else:
            roots = np.empty((0, 9))
        stable_count = 0
        for root in roots:
            stability_flag, eigenvalues = vcsel.compute_stability(
                root, nd, N=BRANCH_COLLOCATION,
                newton_maxit=BRANCH_NEWTON_MAXIT,
                threshold=BRANCH_STABILITY_THRESHOLD,
                sparse=phys["sparse"],
                spectral_shift=BRANCH_SPECTRAL_SHIFT,
                n_eigenvalues=BRANCH_COLLOCATION * 3 * 3 - 1,
            )
            eigenvalues = np.asarray(eigenvalues)
            finite_spectrum = (eigenvalues.size > 0
                               and np.all(np.isfinite(eigenvalues)))
            # Use the tutorial's returned flag. An empty or nonfinite spectrum
            # cannot establish stability even if the library returns 1.0.
            stable = int(stability_flag) if finite_spectrum else -1
            stable_count += stable == 1
            frequency_ghz, coherent_intensity = branch_observables(
                root, nd["tau"]
            )
            records.append(dict(
                matrix_sum_per_ns=float(2 * budget),
                frequency_ghz=frequency_ghz,
                coherent_intensity=coherent_intensity, stability=stable,
                max_real_eigenvalue=(float(np.max(eigenvalues.real))
                                     if finite_spectrum else np.nan),
            ))
        print(f"Branch {index}/{len(budgets)}: "
              f"matrix sum={2 * budget:.3f} ns^-1, "
              f"{len(roots)} roots, {stable_count} stable", flush=True)
    return records


def plot_optimal_branches(records: list[dict], best: dict,
                          output: Path) -> None:
    """Plot frequency and coherent intensity branches by DDE stability."""
    colors = {1: "blue", 0: "red", -1: "gray"}
    labels = {1: "stable", 0: "unstable", -1: "undetermined"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for stability in (0, 1, -1):
        group = [row for row in records if row["stability"] == stability]
        if not group:
            continue
        budgets = [row["matrix_sum_per_ns"] for row in group]
        for ax, key in zip(axes, ("frequency_ghz", "coherent_intensity")):
            ax.scatter(budgets, [row[key] for row in group], s=9,
                       c=colors[stability], label=labels[stability],
                       rasterized=True)
    threshold = 2 * float(best["threshold_per_ns"])
    for ax in axes:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
                   alpha=0.6)
        ax.set_xlabel(r"Sum of coupling matrix entries (ns$^{-1}$)")
        ax.grid(True, alpha=0.2)
    axes[0].set(title=r"Frequency branches ($\phi_{nm}=0$)",
                ylabel="Common frequency (GHz)")
    axes[1].set(title=r"Intensity branches ($\phi_{nm}=0$)",
                ylabel=r"Coherent intensity $|E_{\mathrm{total}}|^2$")
    axes[0].text(0.03, 0.94, "(a)", transform=axes[0].transAxes,
                 va="top", fontsize=20)
    axes[1].text(0.03, 0.94, "(b)", transform=axes[1].transAxes,
                 va="top", fontsize=20)
    axes[1].legend(loc="best")
    fig.suptitle(f"Optimal topology: u1={float(best['u1']):.3f}, "
                 f"u2={float(best['u2']):.3f}; "
                 f"sampled matrix-sum threshold={threshold:.2f} ns$^{{-1}}$")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__" and get_ipython() is not None:
    branch_csv = saved_scan_csv()
    branch_best, branch_links_per_ns = load_optimal_coupling(branch_csv)
    branch_records = solve_optimal_branches(branch_best, branch_links_per_ns)
    branch_data = branch_csv.with_name(branch_csv.stem + "_optimum_branches.csv")
    with branch_data.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "matrix_sum_per_ns", "frequency_ghz", "coherent_intensity",
            "stability", "max_real_eigenvalue",
        ))
        writer.writeheader()
        writer.writerows(branch_records)
    branch_plot = branch_csv.with_name(branch_csv.stem + "_optimum_branches.png")
    plot_optimal_branches(branch_records, branch_best, branch_plot)
    print(f"Saved {branch_data} and {branch_plot}")
    from IPython.display import Image, display

    display(Image(filename=str(branch_plot)))
