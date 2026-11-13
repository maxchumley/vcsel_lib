"""Learned connected edge-list policy; existing dense physics is used only for M<=100.

The policy encodes O(M) local context edges once. It learns a root, a tree
insertion order, parents and directions, then chooses optional directed edges
globally. No topology degree cap or all-pairs policy head is used. At inference,
the factorized optional-edge scores are enumerated with a heap in O(E log M).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import heapq
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from rl import conditional_coupling_designer_variable_m as base
from rl.archive import conditional_coupling_sparse_tolerance as legacy
from rl.archive import conditional_coupling_sparse_graph_tolerance as graph
from rl.paths import MODEL_DIR, RESULTS_DIR

CHECKPOINT_FORMAT = "conditional_coupling_learned_tree_scale_aware_v1"
DENSE_SIMULATION_LIMIT = graph.DENSE_SIMULATION_LIMIT


@dataclass
class LearnedTreeConfig(legacy.SparseToleranceConfig):
    scale_aware_detuning_features: bool = True
    normalize_incoming_coupling_by_degree: bool = True
    training_iterations: int = 3000
    detuning_curriculum_iterations: int = 600

    def __post_init__(self):
        for kind in ("best", "current", "final"):
            field = f"{kind}_checkpoint_file"
            if getattr(self, field) is None:
                setattr(self, field, str(MODEL_DIR / f"learned_tree_{kind}.pt"))
        super().__post_init__()
        if not self.scale_aware_detuning_features:
            raise ValueError("Learned-tree policy requires scale-aware detuning features")
        if self.balanced_coupling or self.edge_conditioned_log_std:
            raise ValueError("Learned-tree policy requires independent directed actions and shared stds")


def q_to_edge_count(q: float, m: int) -> int:
    """Nonlinear budget: q=0 -> M-1, q=1 -> M(M-1), fixed q<1 -> O(M)."""
    if m < 2 or not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("Require M>=2 and finite q in [0,1]")
    minimum, maximum = m - 1, m * (m - 1)
    if q == 1.0:
        return maximum
    count = int(np.floor(minimum / (1.0 - q * (m - 1) / m) + 0.5))
    return min(maximum, max(minimum, count))


class LearnedTreePolicy(nn.Module):
    """One encoder pass, node-level topology scores, selected-pair value head."""

    def __init__(self, config: LearnedTreeConfig):
        super().__init__()
        self.config = config
        width = config.node_embedding_dim
        self.node_encoder = graph._mlp([5, *config.node_hidden_sizes, width])
        self.messages = nn.ModuleList(graph.SparseMessageLayer(
            width, config.gnn_message_hidden_size
        ) for _ in range(config.gnn_message_passing_steps))
        self.root_head = nn.Linear(width, 1)
        self.join_head = nn.Linear(width, 1)
        self.parent_head = nn.Linear(width, 1)
        self.receiver_head = nn.Linear(width, 1)
        self.source_head = nn.Linear(width, 1)
        edge_width = 3 * width + 4 + 1 + 2
        self.direction_head = graph._mlp([edge_width, width, 1])
        self.edge_head = graph._mlp([edge_width, *config.edge_hidden_sizes, 2])
        self.q_head = graph._mlp([width + 2, width, 1])
        self.log_std = nn.Parameter(torch.full((2,), config.initial_log_std))
        self.q_log_std = nn.Parameter(torch.tensor(config.initial_q_log_std))
        self.minimum_log_std = config.minimum_log_std
        self.maximum_log_std = config.maximum_log_std
        self.maximum_q_log_std = config.maximum_q_log_std
        with torch.no_grad():
            self.edge_head[-1].weight.mul_(0.01)
            fraction = config.initial_kappa_fraction
            self.edge_head[-1].bias.copy_(torch.tensor([
                np.log(fraction / (1.0 - fraction)), 0.0
            ]))
            self.q_head[-1].weight.zero_()
            fraction = ((config.initial_q_mean - config.mild_minimum_q)
                        / (1.0 - config.mild_minimum_q))
            self.q_head[-1].bias.fill_(np.log(fraction / (1.0 - fraction)))

    def encode(self, features):
        if features.ndim != 2 or features.shape[1] != 5 or len(features) < 2:
            raise ValueError("features must have shape (M,5), M>=2")
        m = len(features)
        left = torch.arange(m - 1, device=features.device)
        right = left + 1
        receivers = torch.cat((left, right))
        sources = torch.cat((right, left))
        pairs = graph._pair_features(features, receivers, sources)
        states = self.node_encoder(features)
        for layer in self.messages:
            states = layer(states, pairs, receivers, sources)
        return features, states

    def _edge_inputs(self, encoded, receivers, sources, q):
        features, states = encoded
        m = len(states)
        shared = torch.cat((states.mean(dim=0), states.new_tensor([
            q, 1.0 / (m - 1), 1.0 / (m - 1) ** 2
        ]))).expand(len(sources), -1)
        return torch.cat((states[sources], states[receivers],
                          graph._pair_features(features, receivers, sources), shared), dim=-1)

    def _q(self, states, minimum_q, deterministic, q_override):
        zero = states.sum() * 0.0
        if q_override is not None:
            q = float(q_override)
            if not np.isfinite(q) or not 0 <= q <= 1:
                raise ValueError("q_override must be finite and in [0,1]")
            return q, zero, zero
        if minimum_q == 1.0:
            return 1.0, zero, zero
        m = len(states)
        q_input = torch.cat((states.mean(dim=0), states.new_tensor([
            1.0 / (m - 1), 1.0 / (m - 1) ** 2
        ])))
        distribution = torch.distributions.Normal(
            self.q_head(q_input).squeeze(-1),
            self.q_log_std.clamp(self.config.minimum_q_log_std,
                                 self.maximum_q_log_std).exp())
        latent = distribution.mean if deterministic else distribution.sample()
        q = float((minimum_q + (1.0 - minimum_q) * latent.sigmoid()).detach().cpu())
        return (q, zero, zero) if deterministic else (
            q, distribution.log_prob(latent), distribution.entropy())

    def _backbone(self, encoded, q, deterministic):
        _, states = encoded
        m = len(states)
        zero = states.sum() * 0.0
        root_logits = self.root_head(states).squeeze(-1)
        join_logits = self.join_head(states).squeeze(-1)
        parent_logits = self.parent_head(states).squeeze(-1)
        log_prob, entropy = zero, zero
        if deterministic:
            root = int(root_logits.argmax())
            # Sorting once and maintaining the best available parent avoids
            # O(M^2) scoring at inference.
            order = torch.argsort(join_logits, descending=True).tolist()
            order.remove(root)
            parent = root
            sources, receivers = [], []
            for node in order:
                parents = states.new_tensor([node])
                ancestors = states.new_tensor([parent])
                direction = self.direction_head(self._edge_inputs(
                    encoded, parents.long(), ancestors.long(), q)).squeeze()
                if bool(direction >= 0):
                    sources.append(parent); receivers.append(node)
                else:
                    sources.append(node); receivers.append(parent)
                if bool(parent_logits[node] > parent_logits[parent]):
                    parent = node
            return root, torch.tensor(sources, device=states.device), torch.tensor(
                receivers, device=states.device), log_prob, entropy
        if m > DENSE_SIMULATION_LIMIT:
            raise ValueError("Stochastic learned-tree draws are training-only for M<=100")
        root_dist = torch.distributions.Categorical(logits=root_logits)
        root_tensor = root_dist.sample()
        root = int(root_tensor)
        log_prob = log_prob + root_dist.log_prob(root_tensor)
        entropy = entropy + root_dist.entropy()
        connected = torch.zeros(m, dtype=torch.bool, device=states.device)
        connected[root] = True
        sources, receivers = [], []
        for _ in range(m - 1):
            node_dist = torch.distributions.Categorical(
                logits=join_logits.masked_fill(connected, -torch.inf))
            node_tensor = node_dist.sample()
            node = int(node_tensor)
            parent_dist = torch.distributions.Categorical(
                logits=parent_logits.masked_fill(~connected, -torch.inf))
            parent_tensor = parent_dist.sample()
            parent = int(parent_tensor)
            log_prob = log_prob + node_dist.log_prob(node_tensor) + parent_dist.log_prob(parent_tensor)
            entropy = entropy + node_dist.entropy() + parent_dist.entropy()
            child_index = torch.tensor([node], device=states.device)
            parent_index = torch.tensor([parent], device=states.device)
            direction_dist = torch.distributions.Bernoulli(logits=self.direction_head(
                self._edge_inputs(encoded, child_index, parent_index, q)).squeeze())
            direction = direction_dist.sample()
            log_prob = log_prob + direction_dist.log_prob(direction)
            entropy = entropy + direction_dist.entropy()
            if bool(direction):
                sources.append(parent); receivers.append(node)
            else:
                sources.append(node); receivers.append(parent)
            connected = connected.clone()
            connected[node] = True
        return root, torch.tensor(sources, device=states.device), torch.tensor(
            receivers, device=states.device), log_prob, entropy

    @staticmethod
    def _top_optional_edges(receiver_scores, source_scores, used, needed):
        """Exact highest-scoring unused factorized pairs, without a pair grid."""
        m = len(receiver_scores)
        ordered_sources = sorted(range(m), key=lambda s: (-source_scores[s], s))
        heap = []
        def push(receiver, cursor):
            while cursor < m and ordered_sources[cursor] in used[receiver]:
                cursor += 1
            if cursor < m:
                source = ordered_sources[cursor]
                heapq.heappush(heap, (-(receiver_scores[receiver] + source_scores[source]),
                                      receiver, cursor))
        for receiver in range(m):
            push(receiver, 0)
        receivers, sources = [], []
        for _ in range(needed):
            if not heap:
                raise RuntimeError("Fewer unused edges than requested")
            _, receiver, cursor = heapq.heappop(heap)
            source = ordered_sources[cursor]
            receivers.append(receiver); sources.append(source)
            push(receiver, cursor + 1)
        return sources, receivers

    @staticmethod
    def _sample_optional_edges(receiver_logits, source_logits, used, needed):
        """Vectorized Plackett-Luce draw for small-M training only.

        The M-by-M workspace here is bounded by DENSE_SIMULATION_LIMIT; it is
        never created by deterministic inference, which uses the heap above.
        Gumbel top-k samples an ordered without-replacement list, and the
        log-probability below is its exact sequential Plackett-Luce density.
        """
        m = len(receiver_logits)
        if m > DENSE_SIMULATION_LIMIT:
            raise ValueError("Stochastic optional-edge sampling supports M<=100 only")
        allowed = torch.ones((m, m), dtype=torch.bool, device=receiver_logits.device)
        for receiver, invalid in enumerate(used):
            allowed[receiver, list(invalid)] = False
        scores = (receiver_logits[:, None] + source_logits[None, :]).reshape(-1)
        valid_indices = torch.where(allowed.reshape(-1))[0]
        logits = scores[valid_indices]
        noise = torch.rand_like(logits).clamp(1e-7, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(noise))
        selected = torch.topk(logits + gumbel, needed).indices
        chosen_logits = logits[selected]
        remainder = torch.ones(len(logits), dtype=torch.bool, device=logits.device)
        remainder[selected] = False
        remainder_logmass = torch.logsumexp(logits[remainder], dim=0)
        reverse_logmass = torch.logcumsumexp(
            torch.cat((remainder_logmass[None], chosen_logits.flip(0))), dim=0)
        denominators = reverse_logmass[1:].flip(0)
        log_prob = (chosen_logits - denominators).sum()
        # Diagnostic only; exact entropy of the ordered top-k law is expensive.
        entropy_proxy = torch.distributions.Categorical(logits=logits).entropy() * needed
        flat_pairs = valid_indices[selected]
        receivers = torch.div(flat_pairs, m, rounding_mode="floor").tolist()
        sources = (flat_pairs % m).tolist()
        return sources, receivers, log_prob, entropy_proxy

    def draw(self, encoded, *, deterministic=False, minimum_q=0.0, q_override=None):
        _, states = encoded
        m = len(states)
        q, log_prob, entropy = self._q(states, minimum_q, deterministic, q_override)
        count = q_to_edge_count(q, m)
        if count == m * (m - 1):
            # No topology choice when every directed pair is present.
            receivers = torch.arange(m, device=states.device).repeat_interleave(m)
            sources = torch.arange(m, device=states.device).repeat(m)
            keep = receivers != sources
            receivers, sources = receivers[keep], sources[keep]
            root = int(self.root_head(states).squeeze(-1).argmax())
        else:
            root, sources, receivers, tree_log_prob, tree_entropy = self._backbone(
                encoded, q, deterministic)
            log_prob, entropy = log_prob + tree_log_prob, entropy + tree_entropy
            optional = count - (m - 1)
            if optional:
                receiver_scores = self.receiver_head(states).squeeze(-1)
                source_scores = self.source_head(states).squeeze(-1)
                used = [{r} for r in range(m)]
                for source, receiver in zip(sources.tolist(), receivers.tolist()):
                    used[receiver].add(source)
                if deterministic:
                    extra_s, extra_r = self._top_optional_edges(
                        receiver_scores.tolist(), source_scores.tolist(), used, optional)
                else:
                    extra_s, extra_r, lp, ent = self._sample_optional_edges(
                        receiver_scores, source_scores, used, optional)
                    log_prob, entropy = log_prob + lp, entropy + ent
                sources = torch.cat((sources, torch.tensor(extra_s, device=states.device)))
                receivers = torch.cat((receivers, torch.tensor(extra_r, device=states.device)))
        means = self.edge_head(self._edge_inputs(encoded, receivers, sources, q))
        means = torch.stack((6.0 * torch.tanh(means[:, 0] / 6.0), means[:, 1]), dim=-1)
        distribution = torch.distributions.Normal(
            means, self.log_std.clamp(self.minimum_log_std, self.maximum_log_std).exp())
        actions = distribution.mean if deterministic else distribution.sample()
        if not deterministic:
            log_prob = log_prob + distribution.log_prob(actions).sum()
            entropy = entropy + distribution.entropy().sum()
        limit = self.config.maximum_kappa_per_ns
        if self.config.normalize_incoming_coupling_by_degree:
            indegree = torch.bincount(receivers, minlength=m).clamp_min(1)
            limit = limit / indegree[receivers]
        kappa = limit * actions[:, 0].clamp(-8, 8).sigmoid()
        phi = torch.atan2(actions[:, 1].sin(), actions[:, 1].cos())
        return dict(q=q, root_index=root, source_indices=sources,
                    receiver_indices=receivers, kappa_per_ns=kappa, phi_p_rad=phi,
                    log_probability=log_prob, latent_entropy=entropy,
                    sparse_edge_budget=m * (m - 1))


def infer_learned_tree(policy, target, detunings, tolerance, *, q_override=None):
    """Single deterministic edge-list inference, retaining original node indices."""
    target, detunings = np.asarray(target, float), np.asarray(detunings, float)
    if target.ndim != 1 or target.shape != detunings.shape or len(target) < 2:
        raise ValueError("Supply matching one-dimensional target/detuning arrays, M>=2")
    if not np.isfinite(target).all() or not np.isfinite(detunings).all() or not np.isfinite(tolerance):
        raise ValueError("Context must be finite")
    order = np.argsort(detunings, kind="stable")
    target, detunings = target[order], detunings[order] - detunings.mean()
    config = replace(policy.config, n_lasers=len(target))
    policy.eval()
    with torch.inference_mode():
        features = legacy.encode_sparse_tolerance_context(
            target, detunings, tolerance, policy, config)[0]
        draw = policy.draw(policy.encode(features), deterministic=True,
                           q_override=q_override)
    result = {key: value.cpu().numpy().copy() if torch.is_tensor(value) else value
              for key, value in draw.items() if key not in {"log_probability", "latent_entropy"}}
    result.update(target_phases_rad=target, detuning_distribution_ghz=detunings,
                  sorted_to_original=order, allowable_error_deg=float(tolerance))
    return result


dense_matrices = graph.dense_matrices
simulate_graph_batch = graph.simulate_graph_batch
topology_summary = graph.topology_summary


def train_step(policy, optimizer, rng, iteration, *, simulation_pool=None):
    """One update with all M-groups submitted to the worker pool together."""
    config = policy.config
    if max(config.training_n_lasers) > DENSE_SIMULATION_LIMIT:
        raise ValueError("Training supports M<=100 only")
    policy.train()
    policy.minimum_log_std = base.minimum_log_std_at(iteration, config)
    policy.maximum_log_std = base.maximum_log_std_at(iteration, config)
    policy.maximum_q_log_std = legacy.q_log_std_ceiling_at(iteration, config)
    phase_only = legacy.training_stage_at(iteration, config) == "dense"
    minimum_q = legacy.minimum_q_at(iteration, config)
    sizes = config.training_n_lasers
    assignments = [sizes[(iteration * config.targets_per_batch + i) % len(sizes)]
                   for i in range(config.targets_per_batch)]
    prepared, work_items = [], []
    for m in sorted(set(assignments)):
        local = replace(config, n_lasers=m)
        count = assignments.count(m)
        targets = base.sample_target_phases(count, rng, m,
            config.structured_target_fraction, config.structured_target_jitter_rad)
        detunings = base.sample_detuning_distributions(count, rng, local, iteration=iteration)
        targets, detunings = base.sort_by_detuning(targets, detunings)
        tolerances = legacy.sample_allowable_tolerances(count, iteration, config, rng)
        features = legacy.encode_sparse_tolerance_context(
            targets, detunings, tolerances, policy, local)
        joint_count = (config.candidates_per_target if phase_only else
                       config.candidates_per_target - config.dense_replay_candidates)
        draws = []
        for features_row in features:
            encoded = policy.encode(features_row)
            draws.append([policy.draw(encoded, minimum_q=minimum_q,
                q_override=1.0 if phase_only or candidate >= joint_count else None)
                for candidate in range(config.candidates_per_target)])
        matrices = [[dense_matrices(draw, m) for draw in row] for row in draws]
        kappa = np.asarray([[pair[0] for pair in row] for row in matrices])
        phi = np.asarray([[pair[1] for pair in row] for row in matrices])
        group_index = len(prepared)
        for item in base.split_simulation_batch(
                targets, kappa, phi, local,
                detuning_distributions_ghz=detunings, n_jobs=count):
            work_items.append((group_index, 0, config.candidates_per_target, item))
        prepared.append(dict(m=m, targets=targets, tolerances=tolerances,
                             features=features, draws=draws, joint_count=joint_count,
                             results=[]))
    # Match the older runner: dispatch every target across all M-groups in a
    # single pool call, rather than waiting for one size before starting next.
    work_items.sort(key=lambda item: prepared[item[0]]["m"], reverse=True)
    if config.n_jobs == 1:
        results = [base.simulate_indexed_target_chunk(item) for item in work_items]
    elif simulation_pool is not None:
        results = list(simulation_pool.imap_unordered(
            base.simulate_indexed_target_chunk, work_items))
    else:
        temporary = mp.get_context("spawn").Pool(
            min(config.n_jobs, len(work_items)),
            initializer=base._initialize_simulation_worker)
        try:
            results = list(temporary.imap_unordered(
                base.simulate_indexed_target_chunk, work_items))
        finally:
            temporary.close()
            temporary.join()
    for result in results:
        prepared[int(result["size_batch_index"])]["results"].append(result)

    losses, metrics = [], []
    for group in prepared:
        m, targets, draws = group["m"], group["targets"], group["draws"]
        count, candidates = len(targets), config.candidates_per_target
        phase_reward = np.empty((count, candidates), dtype=float)
        aligned_phases = np.empty((count, candidates, m), dtype=float)
        coverage = np.zeros((count, candidates), dtype=bool)
        for result in group["results"]:
            start, stop = int(result["start_index"]), int(result["stop_index"])
            first = int(result["candidate_start_index"])
            last = int(result["candidate_stop_index"])
            phase_reward[start:stop, first:last] = result["phase_reward"]
            aligned_phases[start:stop, first:last] = result["aligned_achieved_phases_rad"]
            if coverage[start:stop, first:last].any():
                raise RuntimeError("Duplicate physics result for a target/candidate range")
            coverage[start:stop, first:last] = True
        if not coverage.all():
            raise RuntimeError("Missing physics result for a target/candidate range")
        errors = legacy.maximum_circular_phase_error_deg(targets, aligned_phases)
        q = np.array([[draw["q"] for draw in row] for row in draws])
        rewards, feasible = legacy.tiered_sparse_reward(
            phase_reward, errors, group["tolerances"], q,
            config.infeasible_reward_tau_deg)
        joint_count = group["joint_count"]
        if phase_only:
            rewards = phase_reward
        else:
            rewards[:, joint_count:] = phase_reward[:, joint_count:]
        advantages = np.zeros_like(rewards)
        for start, stop in ((0, joint_count), (joint_count, candidates)):
            if stop > start:
                reward_group = rewards[:, start:stop]
                advantages[:, start:stop] = (
                    (reward_group - reward_group.mean(axis=1, keepdims=True))
                    / (reward_group.std(axis=1, keepdims=True) + 1e-8))
        log_probs = torch.stack([
            torch.stack([draw["log_probability"] for draw in row]) for row in draws])
        losses.append(-(log_probs * torch.as_tensor(
            advantages, device=group["features"].device)).mean())
        metrics.append(dict(
            phase_reward=float(phase_reward[:, :joint_count].mean()),
            reward=float(rewards[:, :joint_count].mean()),
            satisfaction=float(feasible[:, :joint_count].mean()),
            q=float(q[:, :joint_count].mean()),
            active_edges=float(np.mean([len(draw["source_indices"])
                for row in draws for draw in row[:joint_count]])),
            latent_entropy=float(torch.stack([
                draw["latent_entropy"].detach() for row in draws
                for draw in row]).mean().cpu())))
    parameters = list(policy.parameters())
    optimizer.zero_grad(set_to_none=True)
    if config.gradient_combination_mode == "pcgrad" and len(losses) > 1:
        gradients = []
        for loss in losses:
            values = torch.autograd.grad(loss, parameters, allow_unused=True)
            gradients.append(torch.cat([
                (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
                for parameter, gradient in zip(parameters, values)]))
        combined, _ = base.combine_task_gradients_pcgrad(
            torch.stack(gradients), config.per_size_gradient_clip, rng)
        offset = 0
        for parameter in parameters:
            parameter.grad = combined[offset:offset + parameter.numel()].reshape_as(parameter).clone()
            offset += parameter.numel()
    else:
        torch.stack(losses).mean().backward()
    torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
    optimizer.step()
    with torch.no_grad():
        policy.log_std.clamp_(policy.minimum_log_std, policy.maximum_log_std)
        policy.q_log_std.clamp_(config.minimum_q_log_std, policy.maximum_q_log_std)
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def save_checkpoint(filename, policy, optimizer, *, iteration, metrics):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    torch.save(dict(format=CHECKPOINT_FORMAT, config=asdict(policy.config),
                    policy_state_dict=policy.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    iteration=iteration, metrics=metrics), temporary)
    os.replace(temporary, filename)


def load_checkpoint(filename, device="cpu"):
    saved = torch.load(filename, map_location=device, weights_only=False)
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a learned-tree checkpoint; older models are incompatible")
    config = LearnedTreeConfig(**{**saved["config"], "device": device})
    policy = LearnedTreePolicy(config).to(device)
    policy.load_state_dict(saved["policy_state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    return policy, optimizer, dict(iteration=saved["iteration"], metrics=saved["metrics"])


def validate(policy, iteration, *, simulation_pool=None):
    config = policy.config
    rows = []
    for m in config.training_n_lasers:
        local = replace(config, n_lasers=m)
        targets = base.sample_target_phases(config.held_out_target_count,
            np.random.default_rng(config.random_seed + 10000 + m), m,
            config.structured_target_fraction, config.structured_target_jitter_rad)
        detunings = base.sample_detuning_distributions(len(targets),
            np.random.default_rng(config.random_seed + 20000 + m), local,
            iteration=iteration if config.validation_follows_detuning_curriculum else None)
        targets, detunings = base.sort_by_detuning(targets, detunings)
        tolerance = config.held_out_strict_tolerance_deg
        draws = [[infer_learned_tree(policy, t, d, tolerance, q_override=q)
                  for q in (None, 1.0)] for t, d in zip(targets, detunings)]
        sim = simulate_graph_batch(targets, detunings, draws, local,
                                   simulation_pool=simulation_pool)
        errors = legacy.maximum_circular_phase_error_deg(
            targets, sim["aligned_achieved_phases_rad"])
        qs = np.array([[d["q"] for d in row] for row in draws])
        reward, feasible = legacy.tiered_sparse_reward(
            sim["phase_reward"], errors, tolerance, qs,
            config.infeasible_reward_tau_deg)
        rows.append(dict(m=m, auto_reward=float(reward[:, 0].mean()),
            auto_phase=float(sim["phase_reward"][:, 0].mean()),
            auto_satisfaction=float(feasible[:, 0].mean()),
            auto_q=float(qs[:, 0].mean()),
            max_budget_phase=float(sim["phase_reward"][:, 1].mean())))
    overall = {key: float(np.mean([row[key] for row in rows]))
               for key in rows[0] if key != "m"}
    overall.update({f"max_budget_phase_M{row['m']}": row["max_budget_phase"]
                    for row in rows})
    return overall


def update_training_figure(history, config):
    """Same two-panel saved/live progress layout as the sparse-scale-aware run."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    iterations = np.array([entry["iteration"] for entry in history])
    axes[0].plot(iterations, [entry["reward"] for entry in history],
                 label="total reward")
    axes[0].plot(iterations, [entry["q"] for entry in history],
                 color="tab:green", linestyle="--", label="mean q")
    axes[0].set(title="REINFORCE training", xlabel="iteration",
                ylabel="mean reward / q", ylim=(0.0, 1.0))
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.25)
    for m in config.training_n_lasers:
        key = f"max_budget_phase_M{m}"
        values = [entry.get(key, np.nan) for entry in history]
        if np.isfinite(values).any():
            axes[1].plot(iterations, values, label=f"M={m}")
    axes[1].plot(iterations,
                 [entry.get("max_budget_phase", np.nan) for entry in history],
                 color="black", linewidth=2.0, label="overall")
    axes[1].set(title=f"Dense held-out targets (T={config.held_out_strict_tolerance_deg:g} deg)",
                xlabel="iteration", ylabel="mean phase reward", ylim=(0.0, 1.0))
    axes[1].legend(loc="lower left", ncol=2)
    axes[1].grid(alpha=0.25)
    return figure


def save_training_figure(history, config, figure_directory=None):
    """Overwrite the two-panel progress PNG in the results directory."""
    figure = update_training_figure(history, config)
    figure_dir = RESULTS_DIR if figure_directory is None else Path(figure_directory)
    figure_dir.mkdir(parents=True, exist_ok=True)
    filename = figure_dir / (Path(config.current_checkpoint_file).stem
                             + "_training_progress.png")
    figure.savefig(filename, dpi=150)
    from IPython import get_ipython
    if config.jupyter_mode and get_ipython() is not None:
        from IPython.display import clear_output, display
        clear_output(wait=True)
        display(figure)
    plt.close(figure)
    return filename


def train(policy, *, figure_directory=None, live_plot=True):
    """Train a separate checkpoint family; never runs at module import."""
    config = policy.config
    if max(config.training_n_lasers) > DENSE_SIMULATION_LIMIT:
        raise ValueError("Training with M>100 is not implemented")
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
            if (iteration == 0 or (iteration + 1) % config.validation_interval == 0
                    or iteration + 1 == config.training_iterations):
                latest = validate(policy, iteration, simulation_pool=pool)
                score = latest["max_budget_phase"] if stage == "dense" else latest["auto_reward"]
                if score > best:
                    best = score
                    save_checkpoint(config.best_checkpoint_file, policy, optimizer,
                                    iteration=iteration + 1, metrics=dict(stage=stage, **latest))
            history.append(dict(iteration=iteration + 1,
                stage="maximum_budget" if stage == "dense" else stage,
                learning_rate=optimizer.param_groups[0]["lr"], **metrics, **latest))
            save_checkpoint(config.current_checkpoint_file, policy, optimizer,
                            iteration=iteration + 1, metrics=history[-1])
            if live_plot and (iteration == 0
                              or (iteration + 1) % config.plot_update_interval == 0
                              or iteration + 1 == config.training_iterations):
                save_training_figure(history, config, figure_directory)
            print(f"{iteration + 1}/{config.training_iterations} {history[-1]['stage']} "
                  f"phase={metrics['phase_reward']:.3f} reward={metrics['reward']:.3f} "
                  f"q={metrics['q']:.3f} auto_satisfied="
                  f"{latest.get('auto_satisfaction', float('nan')):.3f}", flush=True)
        save_checkpoint(config.final_checkpoint_file, policy, optimizer,
                        iteration=config.training_iterations, metrics=history[-1])
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return policy, optimizer, history
