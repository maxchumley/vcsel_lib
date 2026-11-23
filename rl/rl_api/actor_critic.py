"""One-step reward modelling for the conditional, reciprocal density actor.

The critic learns measured RMS rewards, never a bootstrapped return. Historical
designs train only the critic. Additional REINFORCE updates always score fresh
current-policy samples, with detached critic rewards and no simulator calls.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ._compat import core
from .advantages import pair_excluded_advantages
from .density_control import density_band_ids


HISTORY_KEYS = (
    'critic_loss', 'critic_mae', 'critic_rank_accuracy', 'critic_rank_pairs',
    'critic_gate_open', 'critic_actor_updates', 'critic_buffer_size',
    'critic_train_count', 'critic_virtual_reward',
)


class RewardCritic(nn.Module):
    """Independent GNN consuming the executed graph, strengths and phases."""
    def __init__(self, strength_scale=1., width=64):
        super().__init__()
        self.strength_scale = strength_scale
        self.nodes = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.messages = nn.ModuleList([
            core.GraphMessagePassingLayer(width, width, edge_feature_width=8) for _ in range(3)])
        self.edges = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
        self.readout = nn.Sequential(nn.Linear(2*width+4, 128), nn.SiLU(),
                                     nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, features, kappa, phi, mask, fraction):
        size = features.shape[1]
        receiver, source = (torch.as_tensor(v, device=features.device)
                            for v in core.directed_link_indices(size))
        cosine, sine, detuning = features[..., 0], features[..., 1], features[..., 2]
        difference = .5*(detuning[:, source]-detuning[:, receiver])
        enabled = mask[:, receiver, source]
        strengths = kappa[:, receiver, source]*enabled
        phases = phi[:, receiver, source]
        local = torch.stack((
            cosine[:, source]*cosine[:, receiver]+sine[:, source]*sine[:, receiver],
            sine[:, source]*cosine[:, receiver]-cosine[:, source]*sine[:, receiver],
            difference, difference.abs(), torch.log1p(strengths/self.strength_scale),
            enabled, enabled*torch.cos(phases), enabled*torch.sin(phases)), dim=-1)
        states = self.nodes(features)
        for layer in self.messages:
            states = layer(states, local, receiver, source)
        globals_ = torch.stack((fraction, enabled.mean(1),
            fraction.new_full(fraction.shape, size/9.),
            torch.log1p(strengths.sum(1)/size/self.strength_scale)), dim=-1)
        return torch.sigmoid(self.readout(torch.cat((
            states.mean(1), self.edges(local).mean(1), globals_), dim=1))).squeeze(-1)


def experience(item, rewards, iteration):
    """Withhold one whole target/detuning context per size, including siblings."""
    rows, candidates = rewards.shape
    size = item['n_lasers']
    trainable = np.ones((rows, candidates), dtype=bool)
    trainable[(iteration+size) % rows] = False
    mask = np.zeros_like(item['kappa'], dtype=np.uint8)
    receiver, source = core.directed_link_indices(size)
    mask[..., receiver, source] = item['gates'].detach().cpu().numpy().astype(np.uint8)
    repeat = lambda value: np.repeat(np.asarray(value)[:, None], candidates, axis=1)
    values = dict(features=repeat(item['features'].detach().cpu().numpy()),
        targets=repeat(item['targets']), detunings=repeat(item['detunings']),
        kappa=item['kappa'], phi=item['phi'], mask=mask,
        fraction=item['density_fractions'], reward=rewards, trainable=trainable)
    result = {key: np.asarray(value).reshape((-1,)+value.shape[2:]).astype(
        bool if key=='trainable' else np.uint8 if key=='mask' else np.float32)
        for key, value in values.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Critic experience must be finite')
    if np.any((result['reward']<0) | (result['reward']>1)):
        raise ValueError('Critic targets must be measured RMS rewards in [0,1]')
    return result


class DesignReplay:
    """Bounded numeric rings per array size; retain failures and successes alike."""
    def __init__(self, sizes, capacity):
        self.capacity = capacity
        quotient, remainder = divmod(capacity, len(sizes))
        if quotient < 1:
            raise ValueError('Replay capacity must accommodate every array size')
        self.limits = {size: quotient+int(index<remainder) for index, size in enumerate(sizes)}
        self.data, self.counts, self.cursors = {}, {}, {}

    def __len__(self):
        return sum(self.counts.values())

    def clear(self):
        self.counts.clear()
        self.cursors.clear()

    def add(self, size, batch):
        limit = self.limits[size]
        if size not in self.data:
            self.data[size] = {key: np.empty((limit,)+value.shape[1:], dtype=value.dtype)
                               for key, value in batch.items()}
        length = len(batch['reward'])
        if length > limit:
            batch = {key: value[-limit:] for key, value in batch.items()}
            length = limit
        cursor = self.cursors.get(size, 0)
        indices = (cursor+np.arange(length)) % limit
        for key, value in batch.items():
            self.data[size][key][indices] = value
        self.cursors[size] = (cursor+length) % limit
        self.counts[size] = min(limit, self.counts.get(size, 0)+length)

    def strata(self):
        groups = []
        for size, count in self.counts.items():
            data = self.data[size]
            bands = density_band_ids(data['mask'][:count].sum((1, 2)), size)
            for band in range(3):
                indices = np.flatnonzero(data['trainable'][:count] & (bands==band))
                if len(indices):
                    groups.append((size, indices))
        return groups

    def sample(self, count, rng, groups):
        """Balance size/density strata; sample only critic-fitting records."""
        selected = {}
        assignments = rng.choice(sorted({size for size, _ in groups}), size=count)
        for size in np.unique(assignments):
            local = [indices for m, indices in groups if m==size]
            bands = rng.integers(len(local), size=np.count_nonzero(assignments==size))
            for band, indices in enumerate(local):
                amount = int(np.count_nonzero(bands==band))
                if amount:
                    selected.setdefault(size, []).append(rng.choice(indices, size=amount))
        return {size: {key: value[np.concatenate(indices)] for key, value in self.data[size].items()}
                for size, indices in selected.items()}

    def snapshot(self):
        return {f'M{size}_{key}': value[:self.counts.get(size, 0)]
                for size, data in self.data.items() for key, value in data.items()}


def write_archive(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def ranking_metrics(predicted, rewards, trainable, actions):
    """Compare candidates within one held-out target/detuning/density group."""
    predicted, rewards = np.asarray(predicted), np.asarray(rewards)
    held = ~trainable.reshape(rewards.shape)
    mae = float(np.abs(predicted[held]-rewards[held]).mean())
    predictions = predicted.reshape(-1, actions)
    truths = rewards.reshape(-1, actions)
    withheld = held.reshape(-1, actions).all(axis=1)
    left, right = np.triu_indices(actions, 1)
    actual = (truths[withheld, :][:, left]-truths[withheld, :][:, right]).ravel()
    estimate = (predictions[withheld, :][:, left]-predictions[withheld, :][:, right]).ravel()
    meaningful = np.abs(actual)>1e-3
    actual, estimate = actual[meaningful], estimate[meaningful]
    credit = (actual*estimate>0).astype(float)+.5*(estimate==0)
    return mae, float(credit.sum()), len(actual), int(held.sum())


class ActorCriticState:
    """Persistent critic and optional replay, independent of the inference actor."""
    def __init__(self, settings, config, directory):
        self.settings, self.directory = settings, Path(directory)
        self.device = torch.device(config.device)
        # Preserve actor initialization AND its future sampling stream.
        devices = [self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.random_seed+700001)
            self.critic = RewardCritic(config.softplus_strength_scale_per_ns).to(self.device)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=settings.critic_learning_rate)
        capacity = settings.replay_capacity if settings.use_replay else (
            config.targets_per_batch*config.candidates_per_target)
        self.buffer = DesignReplay(config.training_n_lasers, capacity)
        self.rng = np.random.default_rng(config.random_seed+700002)
        self.virtual_reward_scale = {}

    def predict(self, data):
        convert = lambda key: torch.as_tensor(data[key], dtype=torch.float32, device=self.device)
        return self.critic(*(convert(key) for key in ('features','kappa','phi','mask','fraction')))

    def fit(self):
        self.critic.train()
        groups, losses = self.buffer.strata(), []
        for _ in range(self.settings.critic_updates):
            minibatch = self.buffer.sample(self.settings.critic_batch_size, self.rng, groups)
            self.optimizer.zero_grad(set_to_none=True)
            loss = sum((self.predict(data)-torch.as_tensor(data['reward'], device=self.device)).square().sum()
                       for data in minibatch.values())/self.settings.critic_batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        # Prevent stale supervised gradients being mistaken for actor gradients.
        self.optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses))

    def update(self, policy, optimizer, config, prepared, rewards, iteration):
        batches = {item['n_lasers']: experience(item, value, iteration)
                   for item, value in zip(prepared, rewards)}
        if not self.settings.use_replay:
            self.buffer.clear()
        for size, batch in batches.items():
            self.buffer.add(size, batch)
        write_archive(self.directory/'experiences'/f'batch_{iteration+1:06d}.npz',
                      {f'M{size}_{key}': value for size, batch in batches.items() for key, value in batch.items()})
        metrics = dict.fromkeys(HISTORY_KEYS, 0.)
        metrics.update(critic_loss=self.fit(), critic_buffer_size=len(self.buffer),
                       critic_train_count=sum(len(indices) for _, indices in self.buffer.strata()))
        self.critic.eval()
        errors, credits, pairs, counts = [], [], [], []
        with torch.inference_mode():
            for item, value in zip(prepared, rewards):
                data = batches[item['n_lasers']]
                predicted = self.predict(data).cpu().numpy().reshape(value.shape)
                error, credit, pair_count, count = ranking_metrics(
                    predicted, value, data['trainable'], config.actions_per_topology)
                errors.append(error*count)
                credits.append(credit)
                pairs.append(pair_count)
                counts.append(count)
        metrics.update(critic_mae=sum(errors)/sum(counts), critic_rank_pairs=sum(pairs),
                       critic_rank_accuracy=sum(credits)/sum(pairs) if sum(pairs) else .5)
        ready = (iteration+1>=self.settings.warmup_iterations and sum(pairs)>=16
            and metrics['critic_mae']<=self.settings.maximum_reward_mae
            and metrics['critic_rank_accuracy']>=self.settings.minimum_rank_accuracy
            and getattr(config, 'exploration_std_scale', 1.)>0.)
        metrics['critic_gate_open'] = float(ready)
        if ready:
            from .conditional_phases import sample_conditional_density
            from .training_batch import pcgrad_update
            devices = [self.device.index or 0] if self.device.type=='cuda' else []
            # Virtual updates cannot consume the real simulator/context RNGs.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(config.random_seed+800001+iteration)
                update_rng = np.random.default_rng(config.random_seed+900001+iteration)
                virtual_means = []
                for _ in range(self.settings.actor_updates):
                    virtual, values = [], []
                    for item in prepared:
                        _, _, score, mask, decoded = sample_conditional_density(policy,
                            item['features'], item['config'], item['density_fractions'], mirrored=True)
                        kappa, phi, count, rho, shortfall = decoded
                        sample = dict(item, log_prob=score, gates=mask, kappa=kappa, phi=phi,
                                      counts=count, rho=rho, shortfall=shortfall)
                        # No historic action log-probability is ever used here.
                        data = experience(sample, np.zeros_like(rewards[len(virtual)]), iteration)
                        with torch.inference_mode():
                            value = self.predict(data).cpu().numpy().reshape(score.shape)
                        virtual.append(sample)
                        values.append(value)
                    advantages = pair_excluded_advantages(values, config, self.virtual_reward_scale)
                    pcgrad_update(policy, optimizer, config, update_rng, virtual, advantages)
                    virtual_means.append(float(np.mean([value.mean() for value in values])))
                    metrics['critic_actor_updates'] += 1
                metrics['critic_virtual_reward'] = float(np.mean(virtual_means))
        return metrics

    def save(self, iteration, *, snapshot=False):
        path = self.directory/'critic_current.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(iteration=iteration, settings=asdict(self.settings),
            critic_state_dict=self.critic.state_dict(), optimizer_state_dict=self.optimizer.state_dict(),
            sampling_rng=self.rng.bit_generator.state), temporary)
        temporary.replace(path)
        if snapshot:
            write_archive(self.directory/'critic_buffer_latest.npz', self.buffer.snapshot())
