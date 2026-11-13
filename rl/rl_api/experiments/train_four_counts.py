#%%
"""Edit the settings, then run this single cell to start training.

From the repository root:
    python -m rl.rl_api.experiments.train_four_counts
    START_TRAINING=True starts training; set False for a settings preview.

Goal: use the working 5-bin/four-topology sampler with density PCGrad,
then refine with quieter ordinary and budget exploration. RMS, rooted
connectivity and model architecture are retained; phase_model is standard_lk.
This is a fresh 3000-iteration run; no curricula are enabled.

Use the vcsel environment. Cell execution uses no command-line arguments.
"""
from pathlib import Path
import sys
import numpy as np
from multiprocessing import current_process

# In cells __file__ may be absent. Start the kernel anywhere inside the repo.
_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _folder in (_start, *_start.parents):
    if (_folder / "rl/rl_api/__init__.py").is_file():
        if str(_folder) not in sys.path:
            sys.path.insert(0, str(_folder))
        break
else:
    raise ModuleNotFoundError("Start the kernel inside the vcsel_lib repository")

try:
    from vcsel_lib import VCSEL
except ImportError:
    from vcsel_lib.vcsel_lib import VCSEL

from rl.rl_api import (Experiment, TrainingConfig, ModelConfig,
    TopologyConfig, SamplingConfig, OptimizationConfig, ScheduleConfig,
    ValidationConfig, RewardConfig)


#  VCSEL parameters -- same variables and native phys keys as simple_example.py.
alpha = 2.0
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4e-6
q = 1.602e-19
beta = 1e-3
tau = 1e-9
eta = 0.9
current_threshold = 3
I = eta * current_threshold * q / tau_n * (N0 + 1 / (g0 * tau_p))

dt = tau_p
Tmax = 500.0 * 1e-9       # 500 ns; keep the current run's numerical value.
noise_amplitude = 0.0
N_lasers = 3             # Template size only; RL trains on CONFIG.sizes below.

phys = {
    "tau_p": tau_p,
    "tau_n": tau_n,
    "g0": g0,
    "N0": N0,
    "s": s,
    "beta": beta,
    "I": I,
    "q": q,
    "alpha": alpha,
    "phase_model": "standard_lk",
    "tau": tau,
    "dt": dt,
    "Tmax": Tmax,
    "noise_amplitude": noise_amplitude,
    "save_every": 10,
    "coupling": 1.0,
    "self_feedback": 0.0,
    # RL replaces these four entries for each context/action batch.
    "N_lasers": N_lasers,
    "delta": np.zeros(N_lasers),
    "kappa_c_mat": np.zeros((N_lasers, N_lasers)),
    "phi_p_mat": np.zeros((N_lasers, N_lasers)),
}

vcsel = VCSEL(phys)
# Workers call scale_params(), generate_history(), and integrate() on their
# own VCSEL instances after supplying the current M, detunings and couplings.


# Run settings -- edit these directly, or copy this file for another run.
RUN_NAME = "star_rms_bins5_fixedcounts4_precision_standard_lk_3k"
START_TRAINING = True           # Running this single cell starts training.
RESTART_EXISTING_RUN = True     # Archive the previous attempt and start from iteration 1.
LIVE_PLOT = True
VERIFY_REFERENCE = True
OUTPUT_DIRECTORY = None        # None -> rl/rl_api/runs/<RUN_NAME>/

CONFIG = TrainingConfig(
    phys=phys,
    sizes=(3, 4, 5, 6, 7, 8, 9),
    seed=7,
    device="cpu",
    ramp_start_delays=5.0,
    ramp_rise_delays=100.0,

    # GNN architecture and coupling initialization.
    model=ModelConfig(
        node_hidden_sizes=(64,),
        node_embedding_dim=64,
        message_hidden_size=64,
        message_passing_steps=3,
        edge_hidden_sizes=(128, 128),
        normalize_incoming_by_degree=True,
        softplus_strength_scale_per_ns=1.0,
        initial_budget_per_laser=0.5,
    ),

    # Supplied rooted topologies. No topology or detuning curriculum.
    topology=TopologyConfig(
        mode="provided",
        backbone="rooted_star",  # Lowest-detuned laser drives all others.
        minimum_extra_link_fraction=0.0,
        density_bias=1.0,
    ),

    # Each pair: four evenly spaced counts from star to all-to-all.
    # Counts are fixed; optional wiring is independent across pairs and iterations.
    # 7 sizes * 5 pairs * 4 topologies * 8 actions = 1,120 simulations.
    sampling=SamplingConfig(
        pairs_per_size=5,        # Also sets five detuning-span bins.
        topologies_per_pair=4,   # Working 5-bin sampler: star, two intermediate, all-to-all.
        actions_per_topology=8,   # Six ordinary + two wide actions, as in the working 5-bin run.
        detuning_span_ghz=4.0,
        pairs_per_job=5,
        actions_per_job=8,       # 28 jobs of 5 pairs x 8 actions.
        structured_target_fraction=0.0,
        structured_target_jitter_rad=0.1,
        reuse_masks_across_pairs=False,
        topology_sampling_mode="per_target",   # Working 5-bin sampler, with random wiring/order.
    ),

    # Original physical sparse/middle/dense PCGrad bands; empty bands are skipped.
    # Four fixed counts restore 16 populated tasks across seven sizes.
    optimization=OptimizationConfig(
        iterations=3000,
        workers=14,
        learning_rate=1e-4,
        final_learning_rate=1e-5,
        per_task_clip=1.0,
        global_clip=1.0,
        reward_std_ema_decay=0.95,
        budget_calibration_targets=64,
        pcgrad_grouping="density_bands",
    ),

    # Boundaries are the printed, one-based iteration numbers.
    schedules=ScheduleConfig(
        ordinary_start=1000,
        ordinary_end=1500,
        wide_start=1000,
        wide_end=1500,
        learning_rate_decay_start=1500,
        learning_rate_decay_end=2000,  # Hold 1e-5 for iterations 2001-3000.
        wide_phase_actions=2,            # Two of eight get extra phase variance.
        wide_phase_extra_sd_deg=180.0,   # Cosine-tapered to zero by 1500.
        initial_log_std=-0.25,
        initial_minimum_log_std=-1.5,
        initial_maximum_log_std=0.75,
        final_minimum_log_std=-3.0,
        final_maximum_log_std=-1.5,
        # After the original taper, cosine-refine both ordinary SD bounds and
        # nominal log-budget SD. Learned budget multiplier is still applied.
        precision_start=1500,
        precision_end=2000,
        precision_minimum_log_std=-4.5,  # 0.64 degrees for phase actions.
        precision_maximum_log_std=-3.0, # 2.85 degrees for phase actions.
        precision_budget_std=0.05,      # Actual SD about .039-.064.
    ),

    # Only wrapped RMS is optimized. Cosine is recorded for comparison.
    reward=RewardConfig(
        tail_fraction=0.35,
        worst_laser_weight=0.5,
        allow_global_phase_conjugate=True,
    ),

    validation=ValidationConfig(
        every=50,
        pairs_per_size=64,
        plot_every=1,
        notebook_display=True,
        plot_dpi=300,       # 3300 x 1200 pixels for the 11 x 4 inch figure.
        plot_use_tex=True,  # LaTeX serif font, as in simple_example.py.
    ),

)


#  Preview -- rerun after editing parameters or RL settings.
experiment = Experiment(RUN_NAME, vcsel=vcsel, config=CONFIG,output_directory=OUTPUT_DIRECTORY)

if current_process().name == "MainProcess":
    experiment.describe()  # Spawned workers do not repeat the preview.


# Train -- the guard prevents spawned workers from starting another run.
if __name__ == "__main__" and current_process().name == "MainProcess":
    if START_TRAINING:
        result = experiment.train(live_plot=LIVE_PLOT, verify_reference=VERIFY_REFERENCE,restart=RESTART_EXISTING_RUN)
    else:
        print("Preview only. Set START_TRAINING=True, then rerun this cell.")
