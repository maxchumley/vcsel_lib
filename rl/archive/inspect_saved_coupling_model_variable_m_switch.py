#%%
"""Switch one learned variable-M coupling design into another over time."""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from vcsel_lib import VCSEL
from rl import conditional_coupling_run_variable_m as coupling_plots
from rl.conditional_coupling_designer_variable_m import (
    design_for_target,
    load_checkpoint,
    make_vcsel_physical_parameters,
)
from rl.paths import MODEL_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Settings to edit
# ---------------------------------------------------------------------------

CHECKPOINT_KIND = "current"  # "best" or "current"
TRAINING_N_LASERS = tuple(range(2, 10))
CHECKPOINT_SIZE_LABEL = (
    f"{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}"
)
N_LASERS = 2
MODEL_SUFFIX = "_bigru_allM_Mconditioned"
MODEL_ARCHITECTURE = "auto"

FIGURE_DIRECTORY = RESULTS_DIR / "inference" / "bigru" / "general"
FIGURE_SUFFIX = f"{N_LASERS}_clusters_1_to_{N_LASERS}"

# The policy designs both matrices from the same centered detuning vector.
MANUAL_DETUNINGS_GHZ = 0.1 * 0.2 * np.linspace(-5.0, 5.0, N_LASERS)
NUMBER_OF_CANDIDATES = 1

# First transition: uncoupled -> one-cluster design.
INITIAL_RAMP_START_NS = 5.0
INITIAL_RAMP_RISE_10_90_NS = 100.0
# At 300 ns, remove the first design completely before introducing the second.
RAMP_DOWN_START_NS = 300.0
RAMP_DOWN_RISE_10_90_NS = 100.0
ZERO_COUPLING_HOLD_NS = 0.0
SECOND_RAMP_RISE_10_90_NS = 100.0
# cosine_ramp interprets its rise setting as the 10--90% interval, so its
# complete 0--100% duration is rise_10_90 / 0.8.
SECOND_RAMP_START_NS = (
    RAMP_DOWN_START_NS
    + RAMP_DOWN_RISE_10_90_NS / 0.8
    + ZERO_COUPLING_HOLD_NS
)
VALIDATION_TIME_NS = 600.0

NOISE_AMPLITUDE = 0.0
NOISE_ITERATIONS = 1
NOISE_RANDOM_SEED = None

# Time-resolved one-dimensional far-field array geometry.
SHOW_FAR_FIELD = True
EMITTER_SPACING_UM = 10.0
FAR_FIELD_THETA_RANGE_DEG = (-10.0, 10.0)
FAR_FIELD_N_THETA = 801
FAR_FIELD_MAX_TIME_POINTS = 2000
FAR_FIELD_ELEMENT_FWHM_DEG = 20.0

PLOT_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "lines.linewidth": 1.8,
    "figure.dpi": 200,
}


def clustered_target_phases_pi(
    n_lasers: int,
    n_clusters: int,
) -> np.ndarray:
    """Return evenly spaced phase clusters as multiples of pi."""
    if not 1 <= n_clusters <= n_lasers:
        raise ValueError("n_clusters must be between 1 and n_lasers")
    cluster_phases = 2.0 * np.arange(n_clusters) / n_clusters
    base_count, remainder = divmod(n_lasers, n_clusters)
    cluster_counts = [
        base_count + (cluster < remainder)
        for cluster in range(n_clusters)
    ]
    return np.repeat(cluster_phases, cluster_counts)


class SwitchingCouplingVCSEL(VCSEL):
    """VCSEL with compact two-stage kappa and coupling-phase schedules."""

    def configure_coupling_switch(
        self,
        *,
        first_kappa: np.ndarray,
        first_phi_p: np.ndarray,
        second_kappa: np.ndarray,
        second_phi_p: np.ndarray,
        initial_ramp: np.ndarray,
        ramp_down: np.ndarray,
        second_ramp: np.ndarray,
    ) -> None:
        self.first_kappa = np.asarray(first_kappa, dtype=float)
        self.first_phi_p = np.asarray(first_phi_p, dtype=float)
        self.second_kappa = np.asarray(second_kappa, dtype=float)
        self.second_phi_p = np.asarray(second_phi_p, dtype=float)
        self.initial_ramp = np.asarray(initial_ramp, dtype=float)
        self.ramp_down = np.asarray(ramp_down, dtype=float)
        self.second_ramp = np.asarray(second_ramp, dtype=float)

    def f_nd(self, x, x_tau, x_2tau, j, phi_p, nd=None):
        """Interpolate both coupling matrices before each derivative call."""
        if nd is None:
            nd = self.nd
        step = min(int(j), len(self.initial_ramp) - 1)
        initial_fraction = float(self.initial_ramp[step])
        ramp_down_fraction = float(self.ramp_down[step])
        second_fraction = float(self.second_ramp[step])

        kappa = (
            initial_fraction
            * (1.0 - ramp_down_fraction)
            * self.first_kappa
            + second_fraction * self.second_kappa
        )

        # Change to the second phase matrix only after the first coupling has
        # reached zero. Coupling phase has no dynamical effect at that instant.
        if ramp_down_fraction >= 1.0:
            current_phi_p = self.second_phi_p
        else:
            current_phi_p = initial_fraction * np.angle(
                np.exp(1j * self.first_phi_p)
            )

        original_kappa = nd["kappa"]
        nd["kappa"] = kappa
        try:
            return super().f_nd(
                x,
                x_tau,
                x_2tau,
                j,
                current_phi_p,
                nd,
            )
        finally:
            nd["kappa"] = original_kappa


def simulate_coupling_switch(
    first_design: dict,
    second_design: dict,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one continuous free-running -> first -> second design trajectory."""
    physical_parameters = make_vcsel_physical_parameters(
        config,
        detuning_distribution_ghz=first_design[
            "detuning_distribution_ghz"
        ],
    )
    vcsel = SwitchingCouplingVCSEL(physical_parameters)
    nd = vcsel.scale_params()

    first_kappa = (
        np.asarray(first_design["kappa_per_ns"], dtype=float)
        * 1.0e9
        * config.photon_lifetime_seconds
    )
    second_kappa = (
        np.asarray(second_design["kappa_per_ns"], dtype=float)
        * 1.0e9
        * config.photon_lifetime_seconds
    )
    first_phi_p = np.asarray(first_design["phi_p_rad"], dtype=float)
    second_phi_p = np.asarray(second_design["phi_p_rad"], dtype=float)

    full_time_seconds = np.arange(nd["steps"]) * config.time_step_seconds
    initial_ramp = VCSEL.cosine_ramp(
        full_time_seconds,
        t_start=INITIAL_RAMP_START_NS * 1.0e-9,
        rise_10_90=INITIAL_RAMP_RISE_10_90_NS * 1.0e-9,
        kappa_initial=0.0,
        kappa_final=1.0,
    )
    ramp_down = VCSEL.cosine_ramp(
        full_time_seconds,
        t_start=RAMP_DOWN_START_NS * 1.0e-9,
        rise_10_90=RAMP_DOWN_RISE_10_90_NS * 1.0e-9,
        kappa_initial=0.0,
        kappa_final=1.0,
    )
    second_ramp = VCSEL.cosine_ramp(
        full_time_seconds,
        t_start=SECOND_RAMP_START_NS * 1.0e-9,
        rise_10_90=SECOND_RAMP_RISE_10_90_NS * 1.0e-9,
        kappa_initial=0.0,
        kappa_final=1.0,
    )

    nd["kappa"] = first_kappa
    nd["kappa_case_dependent"] = False
    nd["phi_p"] = first_phi_p
    nd.pop("kappa_ramp", None)
    vcsel.configure_coupling_switch(
        first_kappa=first_kappa,
        first_phi_p=first_phi_p,
        second_kappa=second_kappa,
        second_phi_p=second_phi_p,
        initial_ramp=initial_ramp,
        ramp_down=ramp_down,
        second_ramp=second_ramp,
    )

    history, initial_frequency_ghz, _, _ = vcsel.generate_history(
        nd,
        shape="FR",
        n_cases=NOISE_ITERATIONS,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        time_seconds, states, frequencies_nd = vcsel.integrate(
            history,
            nd=nd,
            progress=True,
            max_iter=1,
            smooth_freqs=True,
        )
    return time_seconds, states, frequencies_nd, initial_frequency_ghz


def main() -> dict:
    """Design the two coupling states and simulate their continuous switch."""
    if CHECKPOINT_KIND not in {"best", "current"}:
        raise ValueError('CHECKPOINT_KIND must be "best" or "current"')
    second_ramp_finish_ns = (
        SECOND_RAMP_START_NS + SECOND_RAMP_RISE_10_90_NS / 0.8
    )
    if second_ramp_finish_ns >= VALIDATION_TIME_NS:
        raise ValueError(
            "VALIDATION_TIME_NS must extend beyond the complete second ramp"
        )

    suffix = MODEL_SUFFIX
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    checkpoint = MODEL_DIR / (
        f"conditional_coupling_variable_m_{CHECKPOINT_SIZE_LABEL}_lasers_"
        f"{CHECKPOINT_KIND}{suffix}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    policy, _, config, metadata = load_checkpoint(
        checkpoint,
        device="cpu",
        architecture=MODEL_ARCHITECTURE,
    )
    config = replace(
        config,
        n_lasers=N_LASERS,
        n_jobs=1,
        jupyter_mode=False,
        noise_amplitude=NOISE_AMPLITUDE,
        simulation_time_seconds=VALIDATION_TIME_NS * 1.0e-9,
    )

    detunings_ghz = np.asarray(MANUAL_DETUNINGS_GHZ, dtype=float)
    if detunings_ghz.shape != (N_LASERS,):
        raise ValueError(
            f"MANUAL_DETUNINGS_GHZ must have shape ({N_LASERS},)"
        )
    detunings_ghz -= np.mean(detunings_ghz)

    first_target = clustered_target_phases_pi(N_LASERS, 1) * np.pi
    second_target = (
        clustered_target_phases_pi(N_LASERS, N_LASERS) * np.pi
    )
    first_design = design_for_target(
        policy,
        first_target,
        config,
        number_of_candidates=NUMBER_OF_CANDIDATES,
        top_k=1,
        detuning_distribution_ghz=detunings_ghz,
    )[0]
    second_design = design_for_target(
        policy,
        second_target,
        config,
        number_of_candidates=NUMBER_OF_CANDIDATES,
        top_k=1,
        detuning_distribution_ghz=detunings_ghz,
    )[0]

    print(
        f"Loaded {CHECKPOINT_KIND} checkpoint at iteration "
        f"{metadata['iteration']} (held-out={metadata['held_out_reward']:.3f})"
    )
    print(f"One-cluster design reward: {first_design['phase_reward']:.4f}")
    print(
        f"{N_LASERS}-cluster design reward: "
        f"{second_design['phase_reward']:.4f}"
    )

    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    coupling_plots.figure_directory = FIGURE_DIRECTORY
    with plt.rc_context(PLOT_STYLE):
        for state_label, design in (
            ("initial_1_cluster", first_design),
            (f"final_{N_LASERS}_clusters", second_design),
        ):
            plotted_kappa_max = max(
                float(np.max(design["kappa_per_ns"])),
                1.0e-12,
            )
            coupling_plots.plot_selected_design(
                design,
                config,
                desired_state=f"{state_label}_{FIGURE_SUFFIX}",
                vertical_layout=False,
                kappa_vmax_per_ns=plotted_kappa_max,
            )
            plt.show()

    np.random.seed(NOISE_RANDOM_SEED)
    simulation_tuple = simulate_coupling_switch(
        first_design,
        second_design,
        config,
    )

    figure_suffix = FIGURE_SUFFIX
    if figure_suffix and not figure_suffix.startswith("_"):
        figure_suffix = f"_{figure_suffix}"
    figure_label = f"coupling_switch{figure_suffix}"
    with plt.rc_context(PLOT_STYLE):
        simulation = coupling_plots.simulate_and_plot_best_design(
            second_design,
            config,
            validation_delay_count=(
                VALIDATION_TIME_NS / (config.delay_seconds * 1.0e9)
            ),
            desired_state=figure_label,
            detuning_distribution_ghz=second_design[
                "detuning_distribution_ghz"
            ],
            legend_fontsize=PLOT_STYLE["legend.fontsize"],
            noise_iterations=NOISE_ITERATIONS,
            dual_axis_power_panel=True,
            far_field_panel=SHOW_FAR_FIELD,
            emitter_spacing_m=EMITTER_SPACING_UM * 1.0e-6,
            far_field_theta_range_deg=FAR_FIELD_THETA_RANGE_DEG,
            far_field_n_theta=FAR_FIELD_N_THETA,
            far_field_max_time_points=FAR_FIELD_MAX_TIME_POINTS,
            far_field_element_fwhm_deg=FAR_FIELD_ELEMENT_FWHM_DEG,
            precomputed_simulation=simulation_tuple,
            transition_time_ns=(RAMP_DOWN_START_NS, SECOND_RAMP_START_NS),
        )
        plt.show()

    print("Saved endpoint-design and switching-dynamics figures in:")
    print(f"  {FIGURE_DIRECTORY}")
    return {
        "policy": policy,
        "config": config,
        "metadata": metadata,
        "first_design": first_design,
        "second_design": second_design,
        "simulation": simulation,
    }


results = main()
