"""A readable loop for the current experiment, with isolated output paths.

The reference recipe uses the original batch; the shared-mask recipe uses
training_batch.py with the same numerical kernels. This file owns
initialization, validation, logging and checkpoints.
Its numerical sequence is checked against the original training loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from dataclasses import replace
from datetime import datetime, timezone
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from ._compat import core, engine, reference_source_status
from .checkpoints import save, write_json, write_run_manifest
from .models import build_policy
from .config import ValidationConfig
from .plotting.training import training_figure, save_training_progress


# Keep the active reference metric names. Unsupported-mode placeholders are
# omitted; they do not contribute to the current loss, validation or plots.
HISTORY_KEYS = (
    "phase_reward", "training_reward", "retained_link_fraction", "rooted_fraction",
    "normalized_coupling_budget",
    "mean_redistribution_shortfall_per_ns", "held_out_reward",
    "held_out_rooted_fraction", "held_out_phase_reward",
    "sampled_budget_p10_per_laser", "sampled_budget_median_per_laser",
    "sampled_budget_p90_per_laser", "held_out_budget_per_laser",
    "pcgrad_task_count", "negative_gradient_pair_fraction", "mean_gradient_cosine",
    "mean_phase_rms_error_deg", "worst_phase_rms_error_deg", "phase_rms_success_fraction",
    "held_out_mean_phase_rms_error_deg", "held_out_worst_phase_rms_error_deg",
    "held_out_phase_rms_success_fraction",
)


def make_history(config, *, phase_correction=None, fixed_topology_bank=False, actor_critic=False):
    """Keep the current RMS run's active metrics, excluding unused placeholders."""
    history = {key: [] for key in HISTORY_KEYS}
    if actor_critic:
        from .actor_critic import HISTORY_KEYS as critic_keys
        history.update({key: [] for key in critic_keys})
    if phase_correction is not None and phase_correction.enabled:
        from .phase_correction import HISTORY_KEYS as correction_keys
        history.update({key: [] for key in correction_keys})
        if phase_correction.mode == 'mean_phase_only':
            from .mean_phase_correction import HISTORY_KEYS as mean_keys
            history.update({key: [] for key in mean_keys})
    for key in ("held_out_by_m", "held_out_rms_error_by_m", "held_out_rms_success_by_m"):
        history[key] = {size: [] for size in config.training_n_lasers}
    history["held_out_by_density"] = {band: [] for band in engine.DENSITY_BANDS}
    if getattr(config, 'density_control', False):
        for label in ('tree', 'ata'):
            for metric in ('reward', 'success_fraction'):
                history['held_out_'+label+'_'+metric] = []
    if getattr(config,'exhaustive_symmetric_topologies',False):
        history.update({name:[] for name in ('simulation_seconds','update_seconds','validation_seconds')})
    if fixed_topology_bank:
        for name in ('reward', 'phase_rms_success_fraction', 'worst_phase_rms_error_deg'):
            history['seen_topology_held_out_'+name] = []
        for name in ('rms_reward_by_m', 'rms_success_by_m'):
            history['seen_topology_held_out_'+name] = {size: [] for size in config.training_n_lasers}
    return history


def make_validation_bank(config):
    """Generate fixed validation pairs using the reference's separate RNG seeds."""
    bank = {}
    for size in config.training_n_lasers:
        local = replace(config, n_lasers=size)
        targets = core.sample_target_phases(config.held_out_target_count,
            np.random.default_rng(config.random_seed + 10000 + size), size,
            config.structured_target_fraction, config.structured_target_jitter_rad)
        if getattr(config, 'in_phase_targets', False):
            targets = np.zeros_like(targets)
        detunings = core.sample_detuning_distributions(config.held_out_target_count,
            np.random.default_rng(config.random_seed + 20000 + size), local,
            iteration=config.detuning_curriculum_iterations)
        bank[size] = core.sort_by_detuning(targets, detunings)
    return bank


def evaluate_bank(policy, config, bank, iteration, pool, *, topologies=None):
    """Evaluate each size, preserving the reference's ordering and fixed seeds."""
    results = {}
    span = core.detuning_curriculum_half_span_at(iteration, config)
    for size in config.training_n_lasers:
        targets, detunings = bank[size]
        exhaustive_masks = None
        if getattr(config, 'exhaustive_symmetric_topologies', False):
            from .symmetric_topologies import connected_symmetric_masks
            masks = connected_symmetric_masks(size)
            exhaustive_masks = np.tile(masks, (len(targets), 1))
            targets = np.repeat(targets, len(masks), axis=0)
            detunings = np.repeat(detunings, len(masks), axis=0)
        if config.validation_follows_detuning_curriculum:
            detunings = detunings * span / (0.5 * config.detuning_span_ghz)
        evaluator = engine.evaluate_policy
        if hasattr(config, "phase_model") or topologies is not None:
            from .evaluation import evaluate_phase_model_policy
            evaluator = evaluate_phase_model_policy
        options = (dict(topology_masks=topologies[size][0], root_indices=topologies[size][1])
                   if topologies is not None else {})
        if exhaustive_masks is not None:
            options = dict(topology_masks=exhaustive_masks)
        results[size] = evaluator(policy, targets, detunings,
            replace(config, n_lasers=size), seed=config.random_seed + 30000 + size,
            simulation_pool=pool, **options)
    return results


def set_learning_rate(optimizer, config, iteration, *, decay_end=None):
    """Apply the identical exponential schedule; iteration is zero-based."""
    if iteration >= config.learning_rate_decay_start_iteration:
        end = config.training_iterations if decay_end is None else decay_end
        progress = min((iteration - config.learning_rate_decay_start_iteration) /
            max(end - config.learning_rate_decay_start_iteration - 1, 1), 1)
        rate = config.learning_rate * (config.learning_rate_final / config.learning_rate) ** progress
        for group in optimizer.param_groups:
            group["lr"] = rate


def record_history(history, metrics, validation, config, *, seen_validation=None):
    """Append training metrics and the latest validation, including held values."""
    for key in HISTORY_KEYS:
        if not key.startswith("held_out_"):
            history[key].append(metrics[key])
    if getattr(config,'exhaustive_symmetric_topologies',False):
        for name in ('simulation_seconds','update_seconds','validation_seconds'):
            history[name].append(metrics[name])
    from .phase_correction import HISTORY_KEYS as correction_keys
    for key in correction_keys:
        if key in history:
            history[key].append(metrics[key])
    from .mean_phase_correction import HISTORY_KEYS as mean_keys
    for key in mean_keys:
        if key in history:
            history[key].append(metrics[key])
    if 'critic_loss' in history:
        from .actor_critic import HISTORY_KEYS as critic_keys
        for key in critic_keys:
            history[key].append(metrics[key])
    names = {
        "held_out_reward": "training_reward", "held_out_rooted_fraction": "rooted_fraction",
        "held_out_phase_reward": "phase_reward",
        "held_out_budget_per_laser": "coupling_budget_per_laser",
        "held_out_mean_phase_rms_error_deg": "mean_phase_rms_error_deg",
        "held_out_worst_phase_rms_error_deg": "worst_phase_rms_error_deg",
        "held_out_phase_rms_success_fraction": "phase_rms_success_fraction",
    }
    for key, measured in names.items():
        history[key].append(float(np.mean([value[measured] for value in validation.values()])))
    for size in config.training_n_lasers:
        result = validation[size]
        history["held_out_by_m"][size].append(result["phase_reward"])
        history["held_out_rms_error_by_m"][size].append(result["worst_phase_rms_error_deg"])
        history["held_out_rms_success_by_m"][size].append(result["phase_rms_success_fraction"])
    for band in engine.DENSITY_BANDS:
        values = [validation[size]["phase_reward_by_density"][band] for size in config.training_n_lasers]
        finite = [value for value in values if np.isfinite(value)]
        history["held_out_by_density"][band].append(float(np.mean(finite)) if finite else float("nan"))
    if 'seen_topology_held_out_reward' in history:
        for name, measured in [('reward', 'training_reward'),
                               ('phase_rms_success_fraction', 'phase_rms_success_fraction'),
                               ('worst_phase_rms_error_deg', 'worst_phase_rms_error_deg')]:
            history['seen_topology_held_out_'+name].append(
                float(np.mean([result[measured] for result in seen_validation.values()])))
        for size, result in seen_validation.items():
            history['seen_topology_held_out_rms_reward_by_m'][size].append(result['training_reward'])
            history['seen_topology_held_out_rms_success_by_m'][size].append(result['phase_rms_success_fraction'])
    if getattr(config, 'density_control', False):
        for label in ('tree', 'ata'):
            for metric in ('reward', 'success_fraction'):
                history['held_out_'+label+'_'+metric].append(
                    float(np.mean([result[label+'_'+metric] for result in validation.values()])))


def save_progress(history, config, directory, *, write_history, live_plot,
                  sampling_recipe=None, plot_settings=None, exploration_state=None):
    """Save run history and progress, plus a parent-folder live-viewer image."""
    from ._compat import exported_settings
    if write_history:
        data = dict(config=exported_settings(config), history=history)
        if sampling_recipe is not None:
            data['sampling_recipe'] = sampling_recipe
        if exploration_state is not None:
            data['exploration_state'] = exploration_state
        write_json(directory / "current_history.json", data)
    if live_plot:
        import matplotlib.pyplot as plt
        settings = plot_settings or ValidationConfig()
        with training_figure(history, config, settings) as figure:
            try:
                save_training_progress(figure, directory, settings.plot_dpi, settings.progress_filename)
                if config.jupyter_mode:
                    from IPython.display import clear_output, display
                    clear_output(wait=True)
                    display(figure)
            finally:
                plt.close(figure)


def print_progress(iteration, config, metrics, history):
    """Readable labels distinguish the learning reward from cosine monitoring."""
    correction = (f" phase-teachers={metrics['phase_correction_accepted']}/"
                  f"{metrics['phase_correction_attempted']} "
                  f"locked={metrics['phase_correction_locked_fraction']:.1%} "
                  f"correction-loss={metrics['phase_correction_loss']:.3g}"
                  if 'phase_correction_accepted' in metrics else '')
    if 'phase_correction_loss_before' in metrics:
        correction = (f" phase-teachers={metrics['phase_correction_accepted']}/"
            f"{metrics['phase_correction_attempted']} "
            f"mean-locked={metrics['phase_correction_locked_fraction']:.1%} "
            f"correction-loss={metrics['phase_correction_loss_before']:.6g}"
            f"->{metrics['phase_correction_loss']:.6g} "
            f"strength-change={metrics['phase_correction_strength_change_max']:.1g}")
    if 'seen_topology_held_out_reward' in history:
        correction += (f" seen-RMS={history['seen_topology_held_out_reward'][-1]:.3f}"
                       f" seen-3deg={history['seen_topology_held_out_phase_rms_success_fraction'][-1]:.1%}")
    if 'held_out_tree_success_fraction' in history:
        correction += (f" tree-3deg={history['held_out_tree_success_fraction'][-1]:.1%}"
                       f" ATA-3deg={history['held_out_ata_success_fraction'][-1]:.1%}")
    if 'simulation_seconds' in metrics:
        correction += (f" sim={metrics['simulation_seconds']:.1f}s"
                       f" update={metrics['update_seconds']:.2f}s"
                       f" validation={metrics['validation_seconds']:.1f}s")
    if 'critic_loss' in metrics:
        correction += (f" critic-MAE={metrics['critic_mae']:.3f}"
                       f" critic-rank={metrics['critic_rank_accuracy']:.1%}"
                       f" buffer={metrics['critic_buffer_size']}"
                       f" critic-actor-updates={metrics['critic_actor_updates']}")
    print(f"{iteration+1}/{config.training_iterations} "
          f"RMS={metrics['training_reward']:.3f} cosine={metrics['phase_reward']:.3f} "
          f"held-out-RMS={history['held_out_reward'][-1]:.3f} "
          f"held-out-cosine={history['held_out_phase_reward'][-1]:.3f} "
          f"held-out-3deg={history['held_out_phase_rms_success_fraction'][-1]:.1%} "
          f"rho={metrics['retained_link_fraction']:.3f} "
          f"budget/M={metrics['sampled_budget_median_per_laser']:.3g} ns^-1 "
          f"PCGrad-tasks={metrics['pcgrad_task_count']} "
          f"conflicts={metrics['negative_gradient_pair_fraction']:.3f}"+correction, flush=True)


@dataclass
class TrainingResult:
    """Completed fresh run; the model can immediately propose new designs."""
    model: object
    history: dict
    directory: Path
    plot_settings: ValidationConfig = field(default_factory=ValidationConfig)

    def checkpoint(self, kind="best"):
        """Return current/best/final checkpoint path without loading it."""
        if kind not in {"current", "best", "final"}:
            raise ValueError("Checkpoint kind must be current, best or final")
        return self.directory / f"{kind}.pt"

    def plot(self, *, show=True):
        """Return training progress with this run's resolution and font."""
        import matplotlib.pyplot as plt
        with training_figure(self.history, self.model.backend_config, self.plot_settings) as figure:
            if show:
                plt.show()
        return figure


def prepare_run_directory(directory, *, restart=False):
    """Keep old attempts in timestamped folders when a fresh restart is requested."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        if not restart:
            raise FileExistsError(f"Run directory is not empty: {directory}. "
                                  "Use restart=True to archive it and start fresh, "
                                  "or choose a new experiment name.")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = directory.parent / "restarts" / f"{directory.name}_{timestamp}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
        print(f"Previous attempt saved to: {archive}", flush=True)
    directory.mkdir(parents=True, exist_ok=True)


def train(experiment, *, live_plot=True, verify_reference=True, restart=False,
          initial_checkpoint=None):
    """Start fresh; an explicit restart archives the existing attempt first.

    No resume behavior is implied by loading a model. Exact restart/resume
    needs sampler and running-variance state beyond the existing .pt payload.
    """
    if verify_reference:
        status = reference_source_status()
        numerical_files = ("rl/conditional_coupling_designer_variable_m.py",
                           "rl/conditional_coupling_bernoulli_sparse.py", "vcsel_lib.py")
        changed = [name for name in numerical_files if not status.get(name, False)]
        if changed:
            raise RuntimeError("Reference numerical source changed: " + ", ".join(changed)
                + ". Recheck parity before reproducing this recipe.")
    directory = experiment.directory
    config = experiment.backend_config
    reuse_masks = experiment.config.sampling.reuse_masks_across_pairs
    sampling_mode = experiment.config.sampling.topology_sampling_mode
    band_sampling = sampling_mode in {"band_random", "band_fixed"}
    precision_enabled = experiment.config.schedules.precision_start is not None
    anneal_to_mean = experiment.config.schedules.anneal_to_mean
    phase_correction = experiment.config.phase_correction
    topology_bank = None
    batch_options = {}
    if getattr(config, 'exhaustive_symmetric_topologies', False):
        from .training_batch import run_training_batch
        from .symmetric_topologies import sample_all_symmetric_topologies
        batch_options = dict(mask_sampler=sample_all_symmetric_topologies,
                             schedules=experiment.config.schedules)
    elif reuse_masks:
        from .training_batch import run_training_batch
    elif band_sampling:
        from .training_batch import run_training_batch
        from .sampling import (sample_band_random_topologies, sample_fixed_count_topologies,
                               sample_band_random_reachable_topologies)
        sampler = (sample_band_random_reachable_topologies
                   if getattr(config, 'topology_random_reachable', False) else
                   sample_fixed_count_topologies if sampling_mode == "band_fixed" else
                   sample_band_random_topologies)
        if experiment.config.sampling.fixed_topology_bank_seed is not None:
            from .topology_bank import FixedTopologyBank
            topology_bank = FixedTopologyBank(config, experiment.config.sampling.fixed_topology_bank_seed)
            sampler = topology_bank
        batch_options = dict(mask_sampler=sampler,
                             schedules=experiment.config.schedules)
    elif (hasattr(config, "phase_model") or precision_enabled
          or experiment.config.schedules.anneal_to_mean or phase_correction.enabled):
        from .training_batch import run_training_batch
        sampler = (engine.sample_paired_rooted_topologies if config.topology_sampling_mode == "paired_contexts"
                   else engine.sample_stratified_rooted_topologies)
        batch_options = dict(mask_sampler=sampler)
        if precision_enabled or experiment.config.schedules.anneal_to_mean:
            batch_options['schedules'] = experiment.config.schedules
    else:
        run_training_batch = engine.run_training_batch
    if phase_correction.enabled:
        batch_options['phase_correction'] = phase_correction
    sampling_recipe = (dict(mode='shared_across_pairs', masks_per_size=config.pcgrad_topology_groups,
        pairs_per_size=config.targets_per_batch//len(config.training_n_lasers),
        exact_star=True, other_masks='one per excess-density band, sampled each iteration')
        if reuse_masks else None)
    if config.topology_sampling_mode == "per_target":
        sampling_recipe = dict(mode="independent_per_pair", sampler="backup_per_target",
            masks_per_pair=config.candidates_per_target//config.actions_per_topology,
            pairs_per_size=config.targets_per_batch//len(config.training_n_lasers),
            pcgrad_grouping=config.pcgrad_task_grouping)
        if precision_enabled:
            sampling_recipe['precision_schedule'] = asdict(experiment.config.schedules)
    if band_sampling:
        sampling_recipe = dict(mode="independent_per_pair", sampler=sampling_mode,
            masks_per_pair=5, pairs_per_size=experiment.config.sampling.pairs_per_size,
            endpoints="exact star and all-to-all",
            intermediate=("fixed rounded quarter, half and three-quarter excess-link counts"
                          if sampling_mode == "band_fixed" else "one uniform count per physical band"),
            pcgrad_grouping=config.pcgrad_task_grouping,
            precision_schedule=asdict(experiment.config.schedules))
        if getattr(config, 'topology_random_reachable', False):
            sampling_recipe.update(backbone='random_reachable', root_selection='uniform per topology',
                endpoints='first root-reachable random graph and all-to-all',
                intermediate='uniform requested count per physical band; add edges until chosen root reaches all lasers',
                optional_links='uniform random remaining directed pairs, including root feedback',
                density_note='Reachability can raise actual counts above requested bands; PCGrad groups by actual count')
            if getattr(config, 'topology_root_index', None) is not None:
                sampling_recipe['root_selection'] = f'fixed laser {config.topology_root_index+1} in detuning order'
    if experiment.config.sampling.action_sampling == "mirrored":
        sampling_recipe = dict(sampling_recipe or {},
            action_sampling="mirrored", reward_baseline="pair_excluded",
            ordinary_action_pairs=(config.actions_per_topology-experiment.config.schedules.wide_phase_actions)//2,
            wide_action_pairs=experiment.config.schedules.wide_phase_actions//2,
            mirrored_parameters="latent strengths, phases and budget",
            baseline_scope="other action pairs in the same context and topology")
    from .phase_reference import uses_root_reference
    if uses_root_reference(config):
        sampling_recipe = dict(sampling_recipe or {}, phase_delay_reference='sampled root detuning',
            root_reference_scope='one retained construction root per topology, shared by its actions')
    if getattr(config, 'encode_root', False):
        sampling_recipe = dict(sampling_recipe or {}, encoder_root_feature='1 at sampled root, 0 elsewhere')
    if topology_bank is not None:
        sampling_recipe.update(mode='fixed_graph_bank',
            fixed_topology_bank=topology_bank.metadata(),
            validation_groups='same held-out contexts on trained graphs and on graphs excluded from the bank',
            primary_held_out_metrics='unseen topology group; best.pt still selected by its RMS reward')
    if phase_correction.enabled:
        sampling_recipe = dict(sampling_recipe or {}, phase_correction=asdict(phase_correction),
            phase_correction_selection='one best locked action per context/topology',
            phase_correction_verification='fresh history, same strengths/mask, locked and improved RMS',
            phase_correction_scope='training only; validation and inference use decoder predictions')
        if phase_correction.mode == 'mean_phase_only':
            sampling_recipe.update(
                phase_correction_selection='post-RL deterministic mean design per context/topology',
                phase_correction_verification='same mean strengths, fresh history, locked, improved RMS and 3deg success',
                phase_correction_update='separate phase-output-row SGD after RL; frozen shared layers; same LR schedule',
                phase_correction_acceptance='retain the phase-only step only when fixed teacher loss decreases')
    if getattr(config, 'density_control', False):
        sampling_recipe = dict(sampling_recipe or {}, mode='density_controlled',
            sampler='density endpoints plus independent uniform fractions in each third',
            endpoints='connected symmetric tree and all-to-all',
            intermediate='one density fraction per third of [0,1]',
            topology_input='one density scalar; existing adjacency features uniformly K/(M*(M-1)/2)',
            selection='maximum sum-of-directed-strengths spanning tree, then highest remaining pairs',
            pair_count='M-1 + round(s * (M*(M-1)/2 - (M-1)))',
            phase_delay_reference=('zero frequency reference; target offsets plus learned residual'
                if config.phase_parameterization == 'target_residual' else
                'lowest mean-centered detuning (laser 1), unchanged from star run'),
            baseline_scope='other mirrored action pairs in the same target/detuning/density context',
            reinforce_score='original all-to-all Gaussian actions, including removed strengths',
            physical_budget='unchanged predicted total, redistributed proportionally over selected directions',
            validation='fixed new target/detuning contexts with balanced density requests',
            objective='unchanged wrapped-RMS phase accuracy; no sparsity or budget penalty')
        if getattr(config, 'symmetric_coupling_values', False):
            sampling_recipe['symmetric_values'] = 'average reciprocal latent strengths and unwrapped phases before decoding'
        if getattr(config, 'reciprocal_pair_actions', False):
            sampling_recipe.update(
                symmetric_values='one canonical pair strength and phase sample copied to both directions; no averaging',
                reinforce_score='one Gaussian strength/phase score per unordered pair, including pruned pairs',
                pair_orientation='receiver index below source index in sorted detuning order',
                pair_exploration='directed head SD and extra-wide phase SD divided by sqrt(2)',
                phase_delay_reference=('zero plus learned pair residual'
                    if config.phase_parameterization == 'target_residual' else
                    'lowest mean-centered detuning (laser 1) plus learned pair residual'))
        if getattr(config, 'condition_phases_on_coupling', False):
            sampling_recipe.update(condition_phases_on_coupling=True,
                phase_conditioning='second graph pass on detached, selected physical strengths and mask',
                phase_residual_initialization='zero linear head; preserve original starting policy',
                conditional_phase_scores='one Gaussian score per sampled phase coordinate, given its own strength draw',
                mirrored_parameters='opposite strength/budget noises; opposite phase noises around each conditional mean',
                extra_simulations=0)
        if getattr(config,'asymmetric_coupling_phases',False):
            sampling_recipe.update(symmetric_values='one shared strength per unordered pair; independent directed phases',
                reinforce_score='one Gaussian strength score per unordered pair plus one phase score per direction',
                symmetric_strengths=True,symmetric_phases=False,
                phase_delay_reference='lowest mean-centered detuning (laser 1) plus learned directed residuals',
                phase_conditioning='directed phase residuals conditioned on the selected symmetric strengths')
    if getattr(config, 'exhaustive_symmetric_topologies', False):
        from .symmetric_topologies import topology_metadata
        sampling_recipe = dict(sampling_recipe or {}, mode='all_connected_symmetric',
            topology_bank=topology_metadata(), target_mode='in_phase',
            sampler='complete fixed bank on every detuning context',
            symmetric_values='average reciprocal latent strengths and unwrapped phases before decoding',
            reinforce_score=('Gaussian marginal of each executed reciprocal pair average'
                if config.symmetric_action_score == 'shared_pair' else
                'original independent directed Gaussian samples before projection'),
            phase_delay_reference=('zero mean detuning; target offsets plus learned residual'
                if config.phase_parameterization == 'target_residual' else
                'lowest mean-centered detuning, unchanged laser-1 reference'),
            validation='every held-out detuning context on all 38 topologies')
    rng = core.set_random_seed(config.random_seed)
    policy = build_policy(config)
    initialization = None
    if initial_checkpoint is not None:
        if Path(initial_checkpoint).expanduser().resolve().is_relative_to(directory):
            raise ValueError('Initialization checkpoint must be outside the new output directory')
        from .checkpoints import initialize_from_checkpoint
        config, initialization = initialize_from_checkpoint(policy, config, initial_checkpoint)
        sampling_recipe = dict(sampling_recipe or {}, initialization=initialization)
        print(f"Initializing from: {initialization['checkpoint']} (iteration {initialization['source_iteration']})",
              flush=True)
    else:
        from .root_context import calibrate_budget_references
        config = calibrate_budget_references(policy, config)
    policy.config = config
    prepare_run_directory(directory, restart=restart)
    if topology_bank is not None:
        write_json(directory/'fixed_topology_bank.json', topology_bank.metadata())
    if getattr(config, 'exhaustive_symmetric_topologies', False):
        write_json(directory/'symmetric_topology_bank.json', sampling_recipe['topology_bank'])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    critic = None
    if experiment.config.actor_critic.enabled:
        from .actor_critic import ActorCriticState
        critic = ActorCriticState(experiment.config.actor_critic, config, directory)
        batch_options['actor_critic'] = critic
        sampling_recipe = dict(sampling_recipe or {}, actor_critic=dict(
            **asdict(experiment.config.actor_critic),
            target='measured simulator RMS reward, no bootstrapped returns',
            critic_inputs='context, requested density and executed mask, strengths and phases',
            holdout='one entire current target/detuning context per M; all density/action siblings excluded',
            sampling='balanced fitting records by M and actual physical density band',
            actor_update_source='one true-reward update plus gated fresh-current-policy critic-reward updates',
            actor_gate='warmup, current held-out MAE and within-context/density pairwise ranking',
            experience_archive='experiences/batch_*.npz; archives are never read by the trainer',
            extra_simulations=0))
    reward_scale_state = {}
    history = make_history(config, phase_correction=phase_correction,
                           fixed_topology_bank=topology_bank is not None, actor_critic=critic is not None)
    bank = make_validation_bank(config)
    seen_topologies = unseen_topologies = None
    if topology_bank is not None:
        seen_topologies = {size: topology_bank.validation_masks(size, len(pair[0]), seen=True,
            seed=config.random_seed+30000+size) for size, pair in bank.items()}
        unseen_topologies = {size: topology_bank.validation_masks(size, len(pair[0]), seen=False,
            seed=config.random_seed+30000+size) for size, pair in bank.items()}
    write_run_manifest(directory / "run.json", name=experiment.name,
                       public_config=experiment.config, backend_config=config,
                       initialization=initialization)
    validation = {}
    seen_validation = {}
    best = -np.inf
    pool = None
    if config.n_jobs > 1:
        pool = mp.get_context("spawn").Pool(
            processes=config.n_jobs,
            initializer=core._initialize_simulation_worker)
    try:
        for iteration in range(config.training_iterations):
            set_learning_rate(optimizer, config, iteration,
                              decay_end=experiment.config.schedules.learning_rate_decay_end)
            metrics = run_training_batch(policy, optimizer, config, rng, iteration,
                simulation_pool=pool, reward_scale_state=reward_scale_state, **batch_options)
            state = None
            if band_sampling or precision_enabled or anneal_to_mean:
                from .exploration import exploration_state
                state = exploration_state(policy)
            current_config = policy.config if band_sampling or precision_enabled or anneal_to_mean else config
            validate_now = (iteration == 0 or (iteration+1) % config.validation_interval == 0
                            or iteration+1 == config.training_iterations)
            if getattr(config,'exhaustive_symmetric_topologies',False):
                metrics['validation_seconds']=0.
            validation_start=perf_counter()
            if validate_now:
                options = dict(topologies=unseen_topologies) if topology_bank is not None else {}
                validation = evaluate_bank(policy, config, bank, iteration, pool, **options)
                if topology_bank is not None:
                    seen_validation = evaluate_bank(policy, config, bank, iteration, pool,
                                                    topologies=seen_topologies)
                quality = float(np.mean([value["training_reward"] for value in validation.values()]))
                if quality > best:
                    best = quality
                    save(config.best_checkpoint_file, policy, optimizer, current_config, iteration+1, quality,
                         sampling_recipe=sampling_recipe, exploration_state=state)
            if getattr(config,'exhaustive_symmetric_topologies',False) and validate_now:
                metrics['validation_seconds']=perf_counter()-validation_start
            options = dict(seen_validation=seen_validation) if topology_bank is not None else {}
            record_history(history, metrics, validation, config, **options)
            save(config.current_checkpoint_file, policy, optimizer, current_config, iteration+1,
                 history["held_out_reward"][-1], sampling_recipe=sampling_recipe, exploration_state=state)
            if critic is not None:
                critic.save(iteration+1, snapshot=validate_now)
            plot_now = live_plot and (iteration == 0 or (iteration+1) % config.plot_update_interval == 0
                                     or iteration+1 == config.training_iterations)
            save_progress(history, current_config, directory, write_history=validate_now, live_plot=plot_now,
                          sampling_recipe=sampling_recipe, plot_settings=experiment.config.validation,
                          exploration_state=state)
            print_progress(iteration, config, metrics, history)
            if metrics.get('mean_action_only'):
                print("Final batch used exact mean actions; optimizer update skipped.")
        save(config.final_checkpoint_file, policy, optimizer, policy.config,
             config.training_iterations, history["held_out_reward"][-1], sampling_recipe=sampling_recipe,
             exploration_state=state)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    from .api import CouplingModel
    model = CouplingModel(policy, policy.config,
                          metadata={"iteration": config.training_iterations},
                          output_directory=directory)
    return TrainingResult(model, history, directory, experiment.config.validation)
