#%%
"""Fresh four-laser, in-phase, exhaustive symmetric-topology experiment.

All 38 connected labeled reciprocal masks are supplied for every detuning
context. Both physical coupling matrices are symmetric by averaging reciprocal
latent actions before decoding. All GNN layers, heads and parameter counts are
unchanged. Full 0-4 GHz spans, phase-only RMS, laser-1 delay reference and the
mirrored 1k schedules are retained. Import previews; execute to train.
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
N_lasers = 4             # Template size only; RL trains on CONFIG.sizes below.

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
RUN_NAME = "four_laser_inphase_all38_symmetric_values_rms_4ghz_standard_lk_mirrored_1k"
INITIAL_CHECKPOINT = None       # Seed-7 random weights and fresh budget calibration.
START_TRAINING = True           # Running this single cell starts training.
RESTART_EXISTING_RUN = True     # Archive the previous attempt and start from iteration 1.
LIVE_PLOT = True
VERIFY_REFERENCE = False  # The historical hash predates the reviewed trapezoid fix.
OUTPUT_DIRECTORY = None        # None -> rl/rl_api/runs/<RUN_NAME>/

CONFIG = TrainingConfig(
    phys=phys,
    sizes=(4,),
    seed=7,
    device="cpu",
    ramp_start_delays=5.0,
    ramp_rise_delays=100.0,

    # Identical GNN and output heads; reciprocal physical values are projected in decoding.
    model=ModelConfig(
        node_hidden_sizes=(64,),
        node_embedding_dim=64,
        message_hidden_size=64,
        message_passing_steps=3,
        edge_hidden_sizes=(128, 128),
        normalize_incoming_by_degree=True,
        softplus_strength_scale_per_ns=1.0,
        initial_budget_per_laser=0.5,
        symmetric_coupling_values=True,
        phase_parameterization="target_delay_residual",  # Target offset - 2*pi*f_reference*tau + residual.
    ),

    # Supply the complete labeled bank to the original topology-conditioned GNN.
    topology=TopologyConfig(
        mode="provided",
        backbone="all_connected_symmetric",
    ),

    # Five in-phase/random-detuning contexts; all 38 masks for every context.
    # 5 contexts * 38 masks * 8 actions = 1,520 simulations per iteration.
    # Five contexts x ten candidate slots per job gives 31 vectorized jobs.
    # Simulation chunks may cross topology boundaries; learning still groups eight actions per graph.
    sampling=SamplingConfig(
        pairs_per_size=5,
        topologies_per_pair=38,
        actions_per_topology=8,  # Three ordinary pairs plus one wide pair.
        action_sampling="mirrored",  # Opposite latent strength/phase/budget perturbations.
        detuning_span_ghz=4.0,
        pairs_per_job=5,
        actions_per_job=10,     # 30 jobs of 50 trajectories, plus one job of 20.
        structured_target_fraction=0.0,
        structured_target_jitter_rad=0.1,
        reuse_masks_across_pairs=False,
        topology_sampling_mode="per_target",
        target_mode="in_phase",
    ),

    # Three excess-pair density tasks; all 38 graphs receive equal rollout weight.
    optimization=OptimizationConfig(
        iterations=1000,
        workers=31,  # One trainer plus 31 workers in the 32-CPU Slurm allocation.
        learning_rate=1e-4,
        final_learning_rate=1e-5,
        per_task_clip=1.0,
        global_clip=1.0,
        reward_std_ema_decay=0.95,
        budget_calibration_targets=64,
        pcgrad_grouping="density_bands",
        reward_baseline="pair_excluded",  # Average the other six actions for each pair.
    ),

    # All Gaussian SDs are multiplied by (1000 - iteration) / 999.
    # This replaces the staged bound/precision schedules for this run.
    # Ordinary learned SDs use the initial bounds below; wide phase SD and
    # budget SD receive the same multiplier. At 1000 no random action is drawn.
    schedules=ScheduleConfig(
        anneal_to_mean=True,
        ordinary_start=1,
        ordinary_end=1000,
        wide_start=1,
        wide_end=1000,
        learning_rate_decay_start=900,
        learning_rate_decay_end=1000,
        wide_phase_actions=2,
        wide_phase_extra_sd_deg=180.0,
        initial_log_std=-0.25,
        initial_minimum_log_std=-1.5,
        initial_maximum_log_std=0.75,
        precision_start=None,
        precision_end=None,
    ),

    # Only wrapped RMS is optimized. Cosine is recorded for comparison.
    reward=RewardConfig(
        tail_fraction=0.35,
        worst_laser_weight=0.5,
        allow_global_phase_conjugate=True,
    ),

    validation=ValidationConfig(
        every=50,
        pairs_per_size=10,  # Every held-out detuning context uses all 38 masks: 380 simulations.
        plot_every=10,  # Save the progress PNG every ten iterations; checkpoints still update every iteration.
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
        result = experiment.train(live_plot=LIVE_PLOT, verify_reference=VERIFY_REFERENCE,restart=RESTART_EXISTING_RUN, initial_checkpoint=INITIAL_CHECKPOINT)
    else:
        print("Preview only. Set START_TRAINING=True, then rerun this cell.")
