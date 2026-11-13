"""Small, separate sparse-graph prototype; dense physics is limited to M <= 100.

Nodes are sorted by detuning. A fixed chain 0 -> 1 -> ... -> M-1 is always
present. Each receiver chooses at most K additional sources from C deterministic
cyclic rank neighbours. No all-pairs operation is used in the policy.

The existing context encoder, curricula, rewards, REINFORCE/PCGrad utilities and
physical simulator are reused. Only the guarded simulation adapter densifies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl import conditional_coupling_designer_variable_m as base
from rl.archive import conditional_coupling_sparse_tolerance as legacy
from rl.paths import MODEL_DIR

CHECKPOINT_FORMAT = "conditional_coupling_sparse_graph_prototype_v1"
DENSE_SIMULATION_LIMIT = 100


@dataclass
class SparseGraphConfig(legacy.SparseToleranceConfig):
    """Reuse physical/curriculum settings, with two new graph controls."""

    max_extra_edges_per_node: int = 4
    optional_candidates_per_node: int = 8
    encoder_architecture: str = "gnn"
    scale_aware_detuning_features: bool = True
    training_iterations: int = 3000
    detuning_curriculum_iterations: int = 600
    normalize_incoming_coupling_by_degree: bool = True

    def __post_init__(self):
        # Give even an uncustomized prototype its own checkpoint family.
        for kind in ("best", "current", "final"):
            field = f"{kind}_checkpoint_file"
            if getattr(self, field) is None:
                setattr(self, field, str(MODEL_DIR / f"sparse_graph_prototype_{kind}.pt"))
        super().__post_init__()
        if not (1 <= self.max_extra_edges_per_node <= self.optional_candidates_per_node):
            raise ValueError("Require 1 <= max_extra_edges_per_node <= optional_candidates_per_node")
        if not self.scale_aware_detuning_features:
            raise ValueError("This prototype requires scale-aware features")
        if self.balanced_coupling or self.edge_conditioned_log_std:
            raise ValueError("Prototype uses independent directed rates and shared action stds")


def candidate_sources(n_lasers: int, width: int) -> np.ndarray:
    """(M, min(C,M-1)) padded source indices, built without an M-by-M grid.

    Try cyclic rank offsets +1,-1,+2,-2,...; omit self and the incoming chain
    edge. Wrapping includes long links between detuning extremes. -1 is padding.
    The finite loop visits O(M*C) possible pairs, including for small M.
    """
    if n_lasers < 2 or width < 1:
        raise ValueError("Require M >= 2 and C >= 1")
    width = min(width, n_lasers - 1)
    result = np.full((n_lasers, width), -1, dtype=np.int64)
    for receiver in range(n_lasers):
        sources = []
        seen = {receiver}
        if receiver:
            seen.add(receiver - 1)
        for distance in range(1, min(n_lasers, width + 2)):
            for sign in (1, -1):
                source = (receiver + sign * distance) % n_lasers
                if source not in seen:
                    sources.append(source)
                    seen.add(source)
                if len(sources) == width:
                    break
            if len(sources) == width:
                break
        result[receiver, :len(sources)] = sources
    return result


def optional_quotas(q: float, capacities: np.ndarray) -> np.ndarray:
    """Allocate exactly round-half-up(q * sum(capacities)) optional edges.

    Cumulative rounding distributes the integer budget over receivers without
    exceeding any receiver's capacity; it is deterministic, not a learned action.
    """
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be finite and in [0,1]")
    cumulative = np.r_[0, np.cumsum(capacities)]
    return np.diff(np.floor(q * cumulative + 0.5).astype(np.int64))


class SparseMessageLayer(base.GraphMessagePassingLayer):
    """Reuse the existing MLPs, averaging by actual candidate indegree."""

    def forward(self, states, edge_features, receivers, sources):
        messages = self.message_mlp(torch.cat(
            (states[sources], states[receivers], edge_features), dim=-1
        ))
        aggregated = torch.zeros_like(states).index_add(0, receivers, messages)
        degree = torch.bincount(receivers, minlength=len(states)).clamp_min(1)
        aggregated = aggregated / degree[:, None]
        update = self.update_mlp(torch.cat((states, aggregated), dim=-1))
        return self.normalization(states + update)


def _mlp(widths):
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _pair_features(features, receivers, sources):
    a, b = features[sources], features[receivers]
    difference = 0.5 * (a[:, 2] - b[:, 2])
    return torch.stack((a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1],
                        a[:, 1]*b[:, 0] - a[:, 0]*b[:, 1],
                        difference, difference.abs()), dim=-1)


class SparseGraphPolicy(nn.Module):
    """Bounded message passing, optional-edge choices, and selected-edge rates."""

    def __init__(self, config: SparseGraphConfig):
        super().__init__()
        self.config = config
        width = config.node_embedding_dim
        self.node_encoder = _mlp([5, *config.node_hidden_sizes, width])
        self.messages = nn.ModuleList(SparseMessageLayer(
            width, config.gnn_message_hidden_size
        ) for _ in range(config.gnn_message_passing_steps))
        # The global mean and bounded M features keep every edge coordinated.
        edge_width = 3 * width + 4 + 1 + 2
        self.topology_head = _mlp([edge_width, width, 1])
        self.edge_head = _mlp([edge_width, *config.edge_hidden_sizes, 2])
        self.q_head = _mlp([width + 2, width, 1])
        self.log_std = nn.Parameter(torch.full((2,), config.initial_log_std))
        self.q_log_std = nn.Parameter(torch.tensor(config.initial_q_log_std))
        self.minimum_log_std = config.minimum_log_std
        self.maximum_log_std = config.maximum_log_std
        self.maximum_q_log_std = config.maximum_q_log_std
        with torch.no_grad():
            self.edge_head[-1].weight.mul_(0.01)
            self.edge_head[-1].bias.copy_(torch.tensor([
                np.log(config.initial_kappa_fraction/(1-config.initial_kappa_fraction)), 0.0
            ]))
            self.q_head[-1].weight.zero_()
            fraction = (config.initial_q_mean-config.mild_minimum_q)/(1-config.mild_minimum_q)
            self.q_head[-1].bias.fill_(np.log(fraction/(1-fraction)))

    def encode(self, features):
        """One context (M,5). All intermediate pair arrays have <= M*(C+1) rows."""
        if features.ndim != 2 or features.shape[1] != 5:
            raise ValueError("features must have shape (M,5)")
        m = len(features)
        candidates_np = candidate_sources(m, self.config.optional_candidates_per_node)
        candidates = torch.as_tensor(candidates_np, device=features.device)
        chain_r = torch.arange(1, m, device=features.device)
        chain_s = chain_r - 1
        optional_r, optional_slot = torch.where(candidates >= 0)
        receivers = torch.cat((chain_r, optional_r))
        sources = torch.cat((chain_s, candidates[optional_r, optional_slot]))
        pairs = _pair_features(features, receivers, sources)
        states = self.node_encoder(features)
        for layer in self.messages:
            states = layer(states, pairs, receivers, sources)
        capacities = np.minimum((candidates_np >= 0).sum(axis=1),
                                self.config.max_extra_edges_per_node)
        return features, states, candidates, capacities

    def _edge_inputs(self, encoded, receivers, sources, q):
        features, states, _, _ = encoded
        m = len(states)
        shared = torch.cat((states.mean(dim=0), states.new_tensor([
            q, 1.0/(m-1), 1.0/(m-1)**2
        ]))).expand(len(sources), -1)
        return torch.cat((states[sources], states[receivers],
                          _pair_features(features, receivers, sources), shared), dim=-1)

    def draw(self, encoded, *, deterministic=False, minimum_q=0.0, q_override=None):
        """Sample latent actions with their joint REINFORCE log probability.

        Optional selections follow sequential categorical sampling without
        replacement (Plackett-Luce), in parallel across receivers. There are at
        most K rounds. Only actual choices contribute to log probability.
        Entropy is reported in latent/action space, not as physical phase entropy.
        """
        features, states, candidates, capacities = encoded
        m = len(states)
        zero = states.sum() * 0.0
        log_prob, entropy = zero, zero
        if q_override is not None:
            q = float(q_override)
        elif minimum_q == 1.0:
            q = 1.0
        else:
            q_input = torch.cat((states.mean(dim=0), states.new_tensor([1/(m-1), 1/(m-1)**2])))
            distribution = torch.distributions.Normal(
                self.q_head(q_input).squeeze(-1),
                self.q_log_std.clamp(self.config.minimum_q_log_std, self.maximum_q_log_std).exp()
            )
            latent = distribution.mean if deterministic else distribution.sample()
            q = float((minimum_q + (1-minimum_q)*latent.sigmoid()).detach().cpu())
            if not deterministic:
                log_prob = log_prob + distribution.log_prob(latent)
                entropy = entropy + distribution.entropy()
        quotas = torch.as_tensor(optional_quotas(q, capacities), device=states.device)
        sources = [torch.arange(m-1, device=states.device)]
        receivers = [sources[0]+1]
        if bool((quotas > 0).any()):
            candidate_r, candidate_slot = torch.where(candidates >= 0)
            scores = states.new_full(candidates.shape, -torch.inf)
            scores[candidate_r, candidate_slot] = self.topology_head(self._edge_inputs(
                encoded, candidate_r, candidates[candidate_r, candidate_slot], q
            )).squeeze(-1)
            available = candidates >= 0
            for rank in range(int(quotas.max().item())):
                active_r = torch.where(quotas > rank)[0]
                logits = scores[active_r].masked_fill(~available[active_r], -torch.inf)
                distribution = torch.distributions.Categorical(logits=logits)
                slot = logits.argmax(dim=1) if deterministic else distribution.sample()
                sources.append(candidates[active_r, slot])
                receivers.append(active_r)
                if not deterministic:
                    log_prob = log_prob + distribution.log_prob(slot).sum()
                    entropy = entropy + distribution.entropy().sum()
                available = available.clone()
                available[active_r, slot] = False
        sources, receivers = torch.cat(sources), torch.cat(receivers)
        means = self.edge_head(self._edge_inputs(encoded, receivers, sources, q))
        means = torch.stack((6.0*torch.tanh(means[:, 0]/6.0), means[:, 1]), dim=-1)
        distribution = torch.distributions.Normal(
            means, self.log_std.clamp(self.minimum_log_std, self.maximum_log_std).exp()
        )
        actions = distribution.mean if deterministic else distribution.sample()
        if not deterministic:
            log_prob = log_prob + distribution.log_prob(actions).sum()
            entropy = entropy + distribution.entropy().sum()
        limit = self.config.maximum_kappa_per_ns
        if self.config.normalize_incoming_coupling_by_degree:
            # Normalize by the maximum sparse indegree, not M-1 or the active
            # degree: coupling bounds then remain useful at large M and fixed q.
            degree_cap = capacities + (np.arange(m) > 0)
            limit = limit / torch.as_tensor(degree_cap, device=states.device)[receivers]
        kappa = limit * actions[:, 0].clamp(-8, 8).sigmoid()
        phi = torch.atan2(actions[:, 1].sin(), actions[:, 1].cos())
        return dict(q=q, source_indices=sources, receiver_indices=receivers,
                    kappa_per_ns=kappa, phi_p_rad=phi, log_probability=log_prob,
                    latent_entropy=entropy, sparse_edge_budget=m-1+int(capacities.sum()))


def infer_sparse_graph(policy, target, detunings, tolerance, *, q_override=None):
    """One deterministic inference; returns sorted-node edge lists, no simulation."""
    target, detunings = np.asarray(target, float), np.asarray(detunings, float)
    if target.ndim != 1 or target.shape != detunings.shape or len(target) < 2:
        raise ValueError("Supply matching one-dimensional target/detuning arrays, M >= 2")
    if not np.isfinite(target).all() or not np.isfinite(detunings).all() or not np.isfinite(tolerance):
        raise ValueError("Context must be finite")
    order = np.argsort(detunings, kind="stable")
    target, detunings = target[order], detunings[order] - detunings.mean()
    config = replace(policy.config, n_lasers=len(target))
    policy.eval()
    with torch.inference_mode():
        features = legacy.encode_sparse_tolerance_context(target, detunings, tolerance, policy, config)[0]
        draw = policy.draw(policy.encode(features), deterministic=True, q_override=q_override)
    result = {key: value.cpu().numpy().copy() if torch.is_tensor(value) else value
              for key, value in draw.items() if key not in {"log_probability", "latent_entropy"}}
    result.update(target_phases_rad=target, detuning_distribution_ghz=detunings,
                  sorted_to_original=order, allowable_error_deg=float(tolerance))
    return result


def topology_summary(design):
    m = len(design["target_phases_rad"])
    sources, receivers = design["source_indices"], design["receiver_indices"]
    return dict(n_lasers=m, q=design["q"], active_edges=len(sources),
                sparse_edge_budget=design["sparse_edge_budget"],
                mean_indegree=len(sources)/m,
                max_indegree=int(np.bincount(receivers, minlength=m).max()),
                max_outdegree=int(np.bincount(sources, minlength=m).max()),
                edge_array_bytes=sum(design[key].nbytes for key in
                    ("source_indices", "receiver_indices", "kappa_per_ns", "phi_p_rad")))


def dense_matrices(design, n_lasers):
    """Only compatibility boundary: kappa[receiver, source]. Guard before allocation."""
    if not 2 <= n_lasers <= DENSE_SIMULATION_LIMIT:
        raise ValueError("Dense adapter supports only 2 <= M <= 100; large-M dynamics are not implemented")
    def array(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    r, s = array(design["receiver_indices"]), array(design["source_indices"])
    kappa, phi = np.zeros((2, n_lasers, n_lasers))
    kappa[r, s], phi[r, s] = array(design["kappa_per_ns"]), array(design["phi_p_rad"])
    return kappa, phi


def simulate_graph_batch(targets, detunings, draws, config, *, simulation_pool=None):
    """Reuse the unchanged simulator for rectangular lists [target][candidate]."""
    if config.n_lasers > DENSE_SIMULATION_LIMIT:
        raise ValueError("M > 100 supports inference only in this prototype")
    matrices = [[dense_matrices(draw, config.n_lasers) for draw in row] for row in draws]
    kappa = np.asarray([[pair[0] for pair in row] for row in matrices])
    phi = np.asarray([[pair[1] for pair in row] for row in matrices])
    return base.simulate_candidates_parallel(targets, kappa, phi, config,
        detuning_distributions_ghz=detunings, simulation_pool=simulation_pool)


def save_checkpoint(filename, policy, optimizer, *, iteration, metrics):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    torch.save(dict(format=CHECKPOINT_FORMAT, config=asdict(policy.config),
                    policy_state_dict=policy.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    iteration=iteration, metrics=metrics), temporary)
    os.replace(temporary, filename)


def load_checkpoint(filename, device="cpu"):
    saved = torch.load(filename, map_location=device, weights_only=False)
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a sparse-graph prototype checkpoint; old dense-prune models are incompatible")
    config = SparseGraphConfig(**{**saved["config"], "device": device})
    policy = SparseGraphPolicy(config).to(device)
    policy.load_state_dict(saved["policy_state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    return policy, optimizer, dict(iteration=saved["iteration"], metrics=saved["metrics"])


def train_step(policy, optimizer, rng, iteration, *, simulation_pool=None):
    """One REINFORCE update; per-size gradients use the existing PCGrad helper."""
    config = policy.config
    if max(config.training_n_lasers) > DENSE_SIMULATION_LIMIT:
        raise ValueError("Prototype training supports M <= 100 only")
    policy.train()
    policy.minimum_log_std = base.minimum_log_std_at(iteration, config)
    policy.maximum_log_std = base.maximum_log_std_at(iteration, config)
    policy.maximum_q_log_std = legacy.q_log_std_ceiling_at(iteration, config)
    phase_only = legacy.training_stage_at(iteration, config) == "dense"
    minimum_q = legacy.minimum_q_at(iteration, config)
    sizes = config.training_n_lasers
    assignments = [sizes[(iteration*config.targets_per_batch+i) % len(sizes)]
                   for i in range(config.targets_per_batch)]
    losses, metrics = [], []
    for m in sorted(set(assignments)):
        local = replace(config, n_lasers=m)
        count = assignments.count(m)
        targets = base.sample_target_phases(count, rng, m,
            config.structured_target_fraction, config.structured_target_jitter_rad)
        detunings = base.sample_detuning_distributions(count, rng, local, iteration=iteration)
        targets, detunings = base.sort_by_detuning(targets, detunings)
        tolerances = legacy.sample_allowable_tolerances(count, iteration, config, rng)
        features = legacy.encode_sparse_tolerance_context(targets, detunings, tolerances, policy, local)
        joint_count = config.candidates_per_target if phase_only else (
            config.candidates_per_target-config.dense_replay_candidates)
        draws = []
        for features_row in features:
            encoded = policy.encode(features_row)
            draws.append([policy.draw(encoded, minimum_q=minimum_q,
                q_override=1.0 if phase_only or c >= joint_count else None)
                for c in range(config.candidates_per_target)])
        simulation = simulate_graph_batch(targets, detunings, draws, local, simulation_pool=simulation_pool)
        errors = legacy.maximum_circular_phase_error_deg(targets, simulation["aligned_achieved_phases_rad"])
        q = np.array([[draw["q"] for draw in row] for row in draws])
        rewards, feasible = legacy.tiered_sparse_reward(simulation["phase_reward"], errors,
            tolerances, q, config.infeasible_reward_tau_deg)
        if phase_only:
            rewards = simulation["phase_reward"]
        else:
            rewards[:, joint_count:] = simulation["phase_reward"][:, joint_count:]
        advantages = np.zeros_like(rewards)
        for start, stop in ((0, joint_count), (joint_count, config.candidates_per_target)):
            if stop > start:
                group = rewards[:, start:stop]
                advantages[:, start:stop] = (group-group.mean(axis=1, keepdims=True))/(group.std(axis=1, keepdims=True)+1e-8)
        log_probs = torch.stack([torch.stack([d["log_probability"] for d in row]) for row in draws])
        losses.append(-(log_probs * torch.as_tensor(advantages, device=features.device)).mean())
        metrics.append(dict(phase_reward=float(simulation["phase_reward"][:, :joint_count].mean()),
            reward=float(rewards[:, :joint_count].mean()), satisfaction=float(feasible[:, :joint_count].mean()),
            q=float(q[:, :joint_count].mean()),
            active_edges=float(np.mean([len(d["source_indices"]) for row in draws for d in row[:joint_count]])),
            latent_entropy=float(torch.stack([d["latent_entropy"].detach() for row in draws for d in row]).mean().cpu())))
    parameters = list(policy.parameters())
    optimizer.zero_grad(set_to_none=True)
    if config.gradient_combination_mode == "pcgrad" and len(losses) > 1:
        gradients = []
        for loss in losses:
            values = torch.autograd.grad(loss, parameters, allow_unused=True)
            gradients.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                        for p, g in zip(parameters, values)]))
        combined, _ = base.combine_task_gradients_pcgrad(torch.stack(gradients), config.per_size_gradient_clip, rng)
        offset = 0
        for parameter in parameters:
            parameter.grad = combined[offset:offset+parameter.numel()].reshape_as(parameter).clone()
            offset += parameter.numel()
    else:
        torch.stack(losses).mean().backward()
    torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
    optimizer.step()
    with torch.no_grad():
        policy.log_std.clamp_(policy.minimum_log_std, policy.maximum_log_std)
        policy.q_log_std.clamp_(config.minimum_q_log_std, policy.maximum_q_log_std)
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def validate(policy, iteration, *, simulation_pool=None):
    """Fixed targets/patterns; report deterministic automatic and maximum-budget results."""
    config = policy.config
    rows = []
    for m in config.training_n_lasers:
        local = replace(config, n_lasers=m)
        targets = base.sample_target_phases(config.held_out_target_count,
            np.random.default_rng(config.random_seed+10000+m), m,
            config.structured_target_fraction, config.structured_target_jitter_rad)
        detunings = base.sample_detuning_distributions(len(targets),
            np.random.default_rng(config.random_seed+20000+m), local,
            iteration=iteration if config.validation_follows_detuning_curriculum else None)
        targets, detunings = base.sort_by_detuning(targets, detunings)
        tolerance = config.held_out_strict_tolerance_deg
        draws = [[infer_sparse_graph(policy, t, d, tolerance, q_override=q)
                  for q in (None, 1.0)] for t, d in zip(targets, detunings)]
        sim = simulate_graph_batch(targets, detunings, draws, local, simulation_pool=simulation_pool)
        errors = legacy.maximum_circular_phase_error_deg(targets, sim["aligned_achieved_phases_rad"])
        qs = np.array([[d["q"] for d in row] for row in draws])
        reward, feasible = legacy.tiered_sparse_reward(sim["phase_reward"], errors,
            tolerance, qs, config.infeasible_reward_tau_deg)
        rows.append(dict(auto_reward=float(reward[:, 0].mean()), auto_phase=float(sim["phase_reward"][:, 0].mean()),
            auto_satisfaction=float(feasible[:, 0].mean()), auto_q=float(qs[:, 0].mean()),
            max_budget_phase=float(sim["phase_reward"][:, 1].mean())))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train(policy, *, progress_file=None):
    """New checkpoint family; only explicitly calling this function starts training."""
    config = policy.config
    if max(config.training_n_lasers) > DENSE_SIMULATION_LIMIT:
        raise ValueError("Training with M > 100 is not implemented")
    rng = base.set_random_seed(config.random_seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    history, latest, best, previous_stage = [], {}, -np.inf, None
    pool = None
    if config.n_jobs > 1:
        pool = mp.get_context("spawn").Pool(min(config.n_jobs, config.targets_per_batch),
            initializer=base._initialize_simulation_worker)
    try:
        for iteration in range(config.training_iterations):
            lr = legacy.learning_rate_at(iteration, config)
            if lr is not None or iteration == config.learning_rate_switch_iteration:
                for group in optimizer.param_groups:
                    group["lr"] = lr if lr is not None else config.learning_rate_after_switch
            metrics = train_step(policy, optimizer, rng, iteration, simulation_pool=pool)
            stage = legacy.training_stage_at(iteration, config)
            if stage != previous_stage:
                best = -np.inf
                previous_stage = stage
            if iteration == 0 or (iteration+1) % config.validation_interval == 0 or iteration+1 == config.training_iterations:
                latest = validate(policy, iteration, simulation_pool=pool)
                score = latest["max_budget_phase"] if stage == "dense" else latest["auto_reward"]
                if score > best:
                    best = score
                    save_checkpoint(config.best_checkpoint_file, policy, optimizer,
                        iteration=iteration+1, metrics=dict(stage=stage, **latest))
            history.append(dict(iteration=iteration+1, stage="maximum_budget" if stage == "dense" else stage,
                                learning_rate=optimizer.param_groups[0]["lr"], **metrics, **latest))
            save_checkpoint(config.current_checkpoint_file, policy, optimizer,
                            iteration=iteration+1, metrics=history[-1])
            if progress_file is not None:
                import json
                destination = Path(progress_file)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps(history, indent=2)+"\n")
                os.replace(temporary, destination)
            print(f"{iteration+1}/{config.training_iterations} {history[-1]['stage']} "
                  f"phase={metrics['phase_reward']:.3f} reward={metrics['reward']:.3f} "
                  f"q={metrics['q']:.3f} auto_satisfied={latest.get('auto_satisfaction', float('nan')):.3f}", flush=True)
        save_checkpoint(config.final_checkpoint_file, policy, optimizer,
                        iteration=config.training_iterations, metrics=history[-1])
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return policy, optimizer, history
