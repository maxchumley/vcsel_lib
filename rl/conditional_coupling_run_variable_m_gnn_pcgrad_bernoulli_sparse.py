#%%
"""Train a fresh topology-conditioned GNN design model.

Run this cell in VS Code Interactive. This is a separate checkpoint family;
the q/tree sparse and dense training scripts are not changed.
"""

from dataclasses import asdict
from math import cos, degrees, radians
import multiprocessing as mp
from pathlib import Path
import sys

for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from rl.conditional_coupling_run_variable_m_gnn_pcgrad import config as previous_config
from rl.conditional_coupling_bernoulli_sparse import BernoulliSparseConfig, train_policy
from rl.paths import MODEL_DIR


TRAINING_N_LASERS = tuple(range(3, 10))
TRAINING_ITERATIONS = 3_000
LIVE_PLOT = True
# Set True for a learned root plus an outward directed M-1-link backbone.
# Keep False to continue the existing penalty-based experiment.
USE_CONSTRUCTIVE_ROOTED_BACKBONE = False
TOPOLOGY_INPUT_MODE = True
UNIFORM_RANDOM_ROOTED_TREE = False
SPARSITY_REWARD_MODE = "links_per_laser"  # or "total_sparsity_fraction"
OBJECTIVE_MODE = "phase_only"  # phase reward only; no sparsity/budget ranking
LEXICOGRAPHIC_ORDER = "sparsity_budget"  # "sparsity_budget" or "budget_sparsity"
MAXIMUM_PHASE_ERROR_DEG = 3.0
INITIAL_MAXIMUM_PHASE_ERROR_DEG = 30.0
PHASE_ERROR_ANNEAL_ITERATIONS = 0
MINIMUM_PHASE_REWARD = 0.98
INITIAL_MINIMUM_PHASE_REWARD = 0.5 * (1.0 + cos(radians(INITIAL_MAXIMUM_PHASE_ERROR_DEG)))
# Keep filenames short; the full training settings are saved in each checkpoint.
MODEL_SUFFIX = "_star_rms_pairs5_topo4_wide180_3k"
DEVICE = "cpu"
N_JOBS = 14
TARGETS_PER_M = 5  # Unique target/detuning pairs, each reused across four topologies.
TARGETS_PER_WORKER_TASK = 5
ACTIONS_PER_TOPOLOGY = 10  # One topology per context: eight local, two wider-phase actions.
TOPOLOGIES_PER_TARGET = 4
CANDIDATES_PER_TARGET = TOPOLOGIES_PER_TARGET * ACTIONS_PER_TOPOLOGY
CANDIDATES_PER_WORKER_TASK = ACTIONS_PER_TOPOLOGY  # Five pairs x ten actions = 50 simulations/job.
TRAINING_TOPOLOGY_MIN_FRACTION = 0.0  # final range reaches the M-1-link star
TRAINING_TOPOLOGY_DENSITY_BIAS = 1.0  # spread counts evenly over the available range
TOPOLOGY_STAR_BACKBONE = True  # lowest-detuned laser (index 0) drives every other laser
TOPOLOGY_CURRICULUM_START_ITERATION = None
TOPOLOGY_CURRICULUM_END_ITERATION = None
TOPOLOGY_ADVANTAGE_SCALE = "global_running"
# Four density groups each see the same five target/detuning pairs.
# Detuning is stratified for coverage, with no detuning PCGrad groups.
PCGRAD_TASK_GROUPING = "array_size_topology_strata"
PCGRAD_TOPOLOGY_GROUPS = TOPOLOGIES_PER_TARGET
TOPOLOGY_SAMPLING_MODE = "paired_contexts"
PHASE_REWARD_MODE = "wrapped_rms"
# Stage boundaries use the one-based iteration numbers printed during training.
ORDINARY_EXPLORATION_ANNEAL_START_ITERATION = 1000
ORDINARY_EXPLORATION_ANNEAL_END_ITERATION = 1500
LEARNING_RATE_DECAY_START_ITERATION = 2000
# 35 unique pairs x four topologies; 28 worker jobs of 50 vectorized simulations.
TARGETS_PER_BATCH = len(TRAINING_N_LASERS) * TARGETS_PER_M
STRUCTURED_TARGET_FRACTION = 0.0
HELD_OUT_TARGET_COUNT = 64
VALIDATION_INTERVAL = 50
SIMULATION_TIME_NS = 500.0
DETUNING_SPAN_GHZ = 4.0
DETUNING_CURRICULUM_INITIAL_SPAN_GHZ = 0.01  # full peak-to-peak span
DETUNING_CURRICULUM_WARMUP_ITERATIONS = 0
DETUNING_CURRICULUM_ITERATIONS = 0  # Full 0–4 GHz span coverage immediately.
GATE_HEAD_HOLD_ITERATIONS = 0  # no gate head in topology-input mode

# A rooted topology is generated outside the policy and supplied as an input.
# Every size sees 20 densities from the M-1-link outward star to all-to-all
# from the first iteration. Four neighboring density bands each receive
# one topology for each of five shared target/detuning pairs. Within each
# band, density positions and optional edges are randomized every iteration.
# Five detuning-span strata cover 0–4 GHz (0.8 GHz bins). No gate or tree heads are
# trained. All-to-all computational messages include forward and reverse
# adjacency bits; the edge and budget heads see topology-aware embeddings.
# Physical couplings are exactly zero on absent links. The sampled graph
# budget is redistributed only across enabled links without a cap.
# Eight actions use ordinary exploration; two initially add 180 degrees of
# phase SD in quadrature. That extra width anneals to zero over 1000–1500.
# Strength and budget sampling use their ordinary widths.
# REINFORCE centers rewards among actions for the SAME supplied mask and
# divides by one running reward standard deviation shared across all masks.
# The reward blends mean and worst settled wrapped phase RMS; neither
# sparsity nor coupling budget is ranked or penalized in this run.
REQUIRE_ROOTED_CONNECTIVITY = False
LINK_COST_WEIGHT = 0.0  # no sparsity penalty in this phase-only run
COUPLING_COST_WEIGHT = 0.0  # no budget penalty in this phase-only run
COUPLING_COST_START_ITERATION = 0
COUPLING_COST_RAMP_ITERATIONS = 0
INITIAL_GATE_LOGIT = 0.0
BUDGET_REFERENCE_CALIBRATION_TARGETS = 64
STRENGTH_ACTION_TRANSFORM = "uncapped_softplus"
SOFTPLUS_STRENGTH_SCALE_PER_NS = 1.0
INITIAL_BUDGET_PER_LASER_NS = 0.5  # initial mean B = 0.5 M ns^-1; learned afterward
LOCAL_BUDGET_LOG_STD = 0.35  # ordinary Gaussian budget sampling; no wide branch
WIDE_PHASE_ACTIONS = 2
WIDE_PHASE_FRACTION = WIDE_PHASE_ACTIONS / ACTIONS_PER_TOPOLOGY
WIDE_PHASE_STD = radians(180.0)  # Extra phase SD, combined as sqrt(local_SD**2 + extra_SD**2).
WIDE_PHASE_ANNEAL_START_ITERATION = ORDINARY_EXPLORATION_ANNEAL_START_ITERATION
WIDE_PHASE_ANNEAL_END_ITERATION = ORDINARY_EXPLORATION_ANNEAL_END_ITERATION

# The optional DE teacher stays disabled for this run.
DE_TEACHER_ENABLED = False
DE_TEACHER_START_ITERATION = 0
DE_TEACHER_INTERVAL = 1
DE_TEACHER_POPULATION = 5
DE_TEACHER_GENERATIONS = 3
DE_TEACHER_MUTATION = 0.8
DE_TEACHER_RECOMBINATION = 0.7
DE_TEACHER_LOSS_WEIGHT = 0.1
DE_TEACHER_MIN_IMPROVEMENT = 1.0e-4

# Retained for compatibility with earlier Bernoulli checkpoints; these gate
# exploration settings are unused when the topology is an explicit input.
EXPLORATION_LOGIT_OFFSET_START = 4.0
EXPLORATION_LOGIT_OFFSET_END = 0.25
EXPLORATION_OFFSET_ANNEAL_ITERATIONS = 4_000

_prefix = f"conditional_coupling_variable_m_{TRAINING_N_LASERS[0]}to{TRAINING_N_LASERS[-1]}_lasers"
_settings = asdict(previous_config)
_settings.update(
    n_lasers=TRAINING_N_LASERS[0],
    training_n_lasers=TRAINING_N_LASERS,
    training_iterations=TRAINING_ITERATIONS,
    encoder_architecture="gnn",
    condition_on_n_lasers=False,
    condition_log_std_on_n_lasers=False,
    edge_conditioned_log_std=True,
    condition_on_edge_budget=False,
    enable_sparse_gates=False,
    enable_connected_edge_budgets=False,
    enable_learned_connected_sparsity=False,
    enable_learned_backbone_density=False,
    enable_conditional_backbone_budget=False,
    enable_budget_error_selector=False,
    monotonic_budget_selector=False,
    selector_use_deterministic_max_error=False,
    use_relative_phase_retention=False,
    use_lexicographic_rank_advantages=False,
    include_dense_reference_candidate=False,
    force_symmetric_kappa=False,
    force_symmetric_phi_p=False,
    device=DEVICE,
    n_jobs=N_JOBS,
    detuning_span_ghz=DETUNING_SPAN_GHZ,
    detuning_curriculum_initial_half_span_ghz=0.5 * DETUNING_CURRICULUM_INITIAL_SPAN_GHZ,
    detuning_curriculum_warmup_iterations=DETUNING_CURRICULUM_WARMUP_ITERATIONS,
    detuning_curriculum_iterations=DETUNING_CURRICULUM_ITERATIONS,
    detuning_span_sampling_mode="stratified_span",
    simulation_time_seconds=SIMULATION_TIME_NS * 1e-9,
    targets_per_batch=TARGETS_PER_BATCH,
    targets_per_worker_task=TARGETS_PER_WORKER_TASK,
    candidates_per_target=CANDIDATES_PER_TARGET,
    candidates_per_worker_task=CANDIDATES_PER_WORKER_TASK,
    structured_target_fraction=STRUCTURED_TARGET_FRACTION,
    held_out_target_count=HELD_OUT_TARGET_COUNT,
    validation_interval=VALIDATION_INTERVAL,
    train_all_sizes_each_iteration=True,
    gradient_combination_mode="pcgrad",
    pcgrad_task_grouping=PCGRAD_TASK_GROUPING,
    pcgrad_topology_groups=PCGRAD_TOPOLOGY_GROUPS,
    topology_sampling_mode=TOPOLOGY_SAMPLING_MODE,
    phase_reward_mode=PHASE_REWARD_MODE,
    per_size_gradient_clip=1.0,
    learning_rate=1.0e-4,
    learning_rate_final=1.0e-5,
    # The LR is 1e-4 at printed iteration 2000 and reaches 1e-5 at 3000.
    learning_rate_decay_start_iteration=LEARNING_RATE_DECAY_START_ITERATION - 1,
    enable_minimum_log_std_annealing=True,
    # Align both legacy bound helpers to first change at printed 1000 and
    # finish at 1500. The maximum-bound helper adds one to its own progress.
    minimum_log_std_anneal_start_iteration=ORDINARY_EXPLORATION_ANNEAL_START_ITERATION - 2,
    minimum_log_std_anneal_iterations=(ORDINARY_EXPLORATION_ANNEAL_END_ITERATION
                                     - ORDINARY_EXPLORATION_ANNEAL_START_ITERATION + 1),
    enable_exploration_annealing=True,
    exploration_anneal_start_iteration=ORDINARY_EXPLORATION_ANNEAL_START_ITERATION - 1,
    exploration_anneal_iterations=(ORDINARY_EXPLORATION_ANNEAL_END_ITERATION
                                 - ORDINARY_EXPLORATION_ANNEAL_START_ITERATION + 1),
    allow_global_phase_conjugate=True,
    worst_laser_reward_weight=0.5,
    require_rooted_connectivity=REQUIRE_ROOTED_CONNECTIVITY,
    construct_rooted_backbone=USE_CONSTRUCTIVE_ROOTED_BACKBONE,
    topology_input_mode=TOPOLOGY_INPUT_MODE,
    actions_per_topology=ACTIONS_PER_TOPOLOGY,
    training_topology_min_fraction=TRAINING_TOPOLOGY_MIN_FRACTION,
    training_topology_density_bias=TRAINING_TOPOLOGY_DENSITY_BIAS,
    topology_star_backbone=TOPOLOGY_STAR_BACKBONE,
    topology_curriculum_start_iteration=TOPOLOGY_CURRICULUM_START_ITERATION,
    topology_curriculum_end_iteration=TOPOLOGY_CURRICULUM_END_ITERATION,
    topology_advantage_scale=TOPOLOGY_ADVANTAGE_SCALE,
    sparsity_reward_mode=SPARSITY_REWARD_MODE,
    objective_mode=OBJECTIVE_MODE,
    lexicographic_order=LEXICOGRAPHIC_ORDER,
    lexicographic_maximum_phase_error_deg=MAXIMUM_PHASE_ERROR_DEG,
    lexicographic_initial_phase_error_deg=INITIAL_MAXIMUM_PHASE_ERROR_DEG,
    lexicographic_phase_error_anneal_iterations=PHASE_ERROR_ANNEAL_ITERATIONS,
    lexicographic_minimum_phase_reward=MINIMUM_PHASE_REWARD,
    lexicographic_initial_minimum_phase_reward=INITIAL_MINIMUM_PHASE_REWARD,
    budget_action_transform="log_budget",
    budget_exploration_mode="local",
    budget_exploration_start_std=LOCAL_BUDGET_LOG_STD,
    budget_exploration_end_std=LOCAL_BUDGET_LOG_STD,
    log_budget_initial_per_laser=INITIAL_BUDGET_PER_LASER_NS,
    strength_exploration_fraction=0.0,
    phase_exploration_fraction=WIDE_PHASE_FRACTION,
    phase_exploration_start_std=WIDE_PHASE_STD,
    # Same printed first/last annealing iterations as the local SD bounds.
    phase_exploration_start_iteration=WIDE_PHASE_ANNEAL_START_ITERATION - 2,
    phase_exploration_end_iteration=WIDE_PHASE_ANNEAL_END_ITERATION - 1,
    de_teacher_enabled=DE_TEACHER_ENABLED,
    de_teacher_start_iteration=DE_TEACHER_START_ITERATION,
    de_teacher_interval=DE_TEACHER_INTERVAL,
    de_teacher_population=DE_TEACHER_POPULATION,
    de_teacher_generations=DE_TEACHER_GENERATIONS,
    de_teacher_mutation=DE_TEACHER_MUTATION,
    de_teacher_recombination=DE_TEACHER_RECOMBINATION,
    de_teacher_loss_weight=DE_TEACHER_LOSS_WEIGHT,
    de_teacher_min_improvement=DE_TEACHER_MIN_IMPROVEMENT,
    link_cost_weight=LINK_COST_WEIGHT,
    coupling_cost_weight=COUPLING_COST_WEIGHT,
    coupling_cost_start_iteration=COUPLING_COST_START_ITERATION,
    coupling_cost_ramp_iterations=COUPLING_COST_RAMP_ITERATIONS,
    dense_reference_feasibility=False,
    uncapped_redistribution=True,
    strength_action_transform=STRENGTH_ACTION_TRANSFORM,
    softplus_strength_scale_per_ns=SOFTPLUS_STRENGTH_SCALE_PER_NS,
    plot_coupling_reference=False,
    initial_gate_logit=INITIAL_GATE_LOGIT,
    budget_reference_calibration_targets=BUDGET_REFERENCE_CALIBRATION_TARGETS,
    exploration_logit_offset_start=EXPLORATION_LOGIT_OFFSET_START,
    exploration_logit_offset_end=EXPLORATION_LOGIT_OFFSET_END,
    exploration_offset_anneal_iterations=EXPLORATION_OFFSET_ANNEAL_ITERATIONS,
    gate_head_hold_iterations=GATE_HEAD_HOLD_ITERATIONS,
    uniform_random_rooted_tree=UNIFORM_RANDOM_ROOTED_TREE,
    plot_update_interval=1,
    best_checkpoint_file=str(MODEL_DIR / f"{_prefix}_best{MODEL_SUFFIX}.pt"),
    current_checkpoint_file=str(MODEL_DIR / f"{_prefix}_current{MODEL_SUFFIX}.pt"),
    final_checkpoint_file=str(MODEL_DIR / f"{_prefix}_final{MODEL_SUFFIX}.pt"),
)
config = BernoulliSparseConfig(**_settings)


def main() -> dict:
    print(f"Phase exploration per topology: {ACTIONS_PER_TOPOLOGY - WIDE_PHASE_ACTIONS} local + "
          f"{WIDE_PHASE_ACTIONS} wider; extra SD={degrees(WIDE_PHASE_STD):g} degrees, cosine-annealed "
          f"to zero over printed iterations {WIDE_PHASE_ANNEAL_START_ITERATION}–"
          f"{WIDE_PHASE_ANNEAL_END_ITERATION}")
    print(f"Local exploration anneals over {ORDINARY_EXPLORATION_ANNEAL_START_ITERATION}–"
          f"{ORDINARY_EXPLORATION_ANNEAL_END_ITERATION}; refinement through "
          f"{LEARNING_RATE_DECAY_START_ITERATION}; LR decay to {TRAINING_ITERATIONS}")
    policy, history = train_policy(config, live_plot=LIVE_PLOT)
    return dict(policy=policy, config=policy.config, history=history)


if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    bernoulli_sparse_training_result = main()
