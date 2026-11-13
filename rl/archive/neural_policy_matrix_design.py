# %% Imports and settings
"""Replay-buffered TD3 control of time-varying VCSEL coupling matrices.

Every episode starts from the noise-free free-running delay history. At each
control step a neural actor observes the requested phases and current laser
state, then proposes small changes to every directed kappa and phi_p link.
Both matrices are linearly ramped during one short integration interval. The
final full delay history is passed into the next interval, so the DDE
trajectory is continuous and is never restarted within an episode.
"""

import copy
from collections import deque
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from torch import nn

# Notebook kernels may start inside ``rl/`` instead of the repository root.
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "rl" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from vcsel_lib import VCSEL

from rl.paths import MODEL_DIR


# ----------------- Requested behavior used for validation -----------------
N_lasers = 10
requested_phases = np.linspace(-np.pi, np.pi, N_lasers)


# ----------------- Dynamic-control environment -----------------
parallel_environments = 10
episode_control_steps = 50
# Fifty decisions span 100 ns, long enough for the delayed array to settle.
control_interval_tau = 2.0

# The actor changes each directed link by at most these amounts per step.
# Both controls vary continuously through bounded, piecewise-small updates.
maximum_kappa_ns = 25.0
maximum_kappa_change_per_step_ns = 0.5
maximum_phi_p_change_per_step_rad = 0.10
initial_phi_p = 0.0
controller_history_steps = 5

frequency_tolerance_ghz = 0.10
phase_improvement_weight = 1.0
terminal_reward_weight = 2.0
kappa_magnitude_penalty_weight = 0.05
kappa_change_penalty_weight = 0.01
phi_p_change_penalty_weight = 0.01
success_phase_score = 0.95
success_worst_phase_score = 0.90
success_dwell_steps = 5
success_bonus = 0.25


# ----------------- Replay-buffered TD3 -----------------
n_training_episodes = 200
replay_capacity = 100_000
replay_batch_size = 512
learning_starts = 5_000
n_step_return = 5
discount_factor = 0.99
gradient_updates_per_control_step = 2
actor_learning_rate = 1.0e-4
critic_learning_rate = 3.0e-4
gradient_clip = 1.0
target_update_rate = 0.005
policy_update_delay = 2
exploration_noise = 0.20
target_policy_noise = 0.10
target_noise_clip = 0.30
hidden_units = (512, 512, 256)
random_seed = 7
evaluation_every = 1
evaluation_targets = 4

show_inline_plot = True
plot_every = 1
controller_file = MODEL_DIR / "dynamic_coupling_td3.pt"


# ----------------- Physical setup: explicit numerical values -----------------
alpha = 2.0
tau_p = 5.4e-12
tau_n = 0.25e-9
g0 = 8.75e-4 * 1e9
N0 = 2.86e5
s = 4.0e-6
q = 1.602e-19
beta = 1.0e-3
tau = 1.0e-9
eta = 0.9
I = eta * 3.0 * q / tau_n * (N0 + 1.0 / (g0 * tau_p))

detuning_span_ghz = 2.0
delta = (
    np.linspace(-0.5, 0.5, N_lasers)
    * detuning_span_ghz
    * 2.0
    * np.pi
    * 1e9
)

dt = tau_p
save_every = 10


# ----------------- Action and observation layout -----------------
# Matrix entry [receiver, source]. Every directed off-diagonal magnitude and
# phase is independently controlled; the diagonal remains zero.
aMAT = np.ones((N_lasers, N_lasers)) - np.eye(N_lasers)
link_i, link_j = np.where(aMAT > 0)
n_links = len(link_i)
action_size = 2 * n_links

# target phase sin/cos: 18
# current phase sin/cos: 18
# explicit target-minus-current phase error sin/cos: 18
# recent phase-error sin/cos + relative-frequency history: 5 * 27
# normalized photon powers: 10
# current directed kappa values: 90
# current directed phi_p cosine/sine values: 180
# fraction of the episode completed: 1
observation_size = (
    2 * (N_lasers - 1)
    + 2 * (N_lasers - 1)
    + 2 * (N_lasers - 1)
    + controller_history_steps * 3 * (N_lasers - 1)
    + N_lasers
    + n_links
    + 2 * n_links
    + 1
)


def phase_features(phases):
    """Return wrap-safe phases relative to laser 1."""
    phases = np.asarray(phases, dtype=float)
    single = phases.ndim == 1
    if single:
        phases = phases[None, :]
    if phases.ndim != 2 or phases.shape[1] != N_lasers:
        raise ValueError(
            f"phases must have shape ({N_lasers},) or "
            f"(batch, {N_lasers})"
        )
    relative = np.angle(
        np.exp(1j * (phases - phases[:, :1]))
    )[:, 1:]
    features = np.c_[np.cos(relative), np.sin(relative)]
    return features[0] if single else features


def phase_error_features(phases, targets):
    """Return wrap-safe target-minus-current relative phase errors."""
    phases = np.asarray(phases, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if phases.ndim == 1:
        phases = phases[None, :]
    if targets.ndim == 1:
        targets = targets[None, :]
    current_relative = np.angle(
        np.exp(1j * (phases - phases[:, :1]))
    )[:, 1:]
    target_relative = np.angle(
        np.exp(1j * (targets - targets[:, :1]))
    )[:, 1:]
    error = np.angle(
        np.exp(1j * (target_relative - current_relative))
    )
    return np.c_[np.cos(error), np.sin(error)]


def sample_random_targets(rng, count):
    """Sample phase relationships uniformly with laser 1 as reference."""
    targets = np.zeros((count, N_lasers))
    targets[:, 1:] = rng.uniform(
        -np.pi, np.pi, (count, N_lasers - 1)
    )
    return targets


def links_to_matrices(link_values):
    """Place directed link values into zero-diagonal matrices."""
    values = np.asarray(link_values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    matrices = np.zeros((len(values), N_lasers, N_lasers))
    matrices[:, link_i, link_j] = values
    return matrices


def phase_behavior_scores(phases, targets):
    """Return smooth mean and worst-laser requested-phase scores."""
    phases = np.asarray(phases, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if phases.ndim == 2:
        phases = phases[:, :, None]
    if phases.ndim != 3:
        raise ValueError("phases must have shape (batch,N) or (batch,N,time)")
    desired = np.angle(
        np.exp(1j * (targets - targets[:, :1]))
    )[:, :, None]
    relative = np.angle(
        np.exp(1j * (phases - phases[:, :1]))
    )
    phase_error = np.angle(
        np.exp(1j * (relative - desired))
    )
    per_laser = np.mean(
        0.5 * (1.0 + np.cos(phase_error[:, 1:])),
        axis=2,
    )
    return np.mean(per_laser, axis=1), np.min(per_laser, axis=1)


# ----------------- Closed-loop vectorized environment -----------------
class DynamicCouplingEnvironment:
    """Vectorized, delay-history-preserving VCSEL control environment."""

    def __init__(self, n_envs=parallel_environments, seed=random_seed):
        self.n_envs = int(n_envs)
        self.rng = np.random.default_rng(seed)
        # integrate() receives 2*tau of history plus one new control interval.
        chunk_Tmax = (2.0 + control_interval_tau) * tau
        self.phys = {
            "tau_p": tau_p,
            "tau_n": tau_n,
            "g0": g0,
            "N0": N0,
            "s": s,
            "beta": beta,
            "kappa_c_mat": np.zeros((N_lasers, N_lasers)),
            "phi_p_mat": np.full(
                (N_lasers, N_lasers), initial_phi_p
            ),
            "I": I,
            "q": q,
            "alpha": alpha,
            "delta": delta,
            "coupling": 1.0,
            "self_feedback": 0.0,
            "noise_amplitude": 0.0,
            "dt": dt,
            "Tmax": chunk_Tmax,
            "tau": tau,
            "N_lasers": N_lasers,
            "save_every": save_every,
            "show_output_size_message": False,
        }
        self.vcsel = VCSEL(self.phys)
        self.nd = self.vcsel.scale_params()
        self.nd["kappa_case_dependent"] = True
        self.nd["phi_p"] = np.full(
            (self.n_envs, N_lasers, N_lasers), initial_phi_p
        )
        self.delay_time = 2.0 * tau
        self.reset()

    def reset(self, targets=None):
        """Start every environment from a newly generated FR history."""
        if targets is None:
            targets = sample_random_targets(self.rng, self.n_envs)
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 1:
            targets = np.repeat(targets[None, :], self.n_envs, axis=0)
        if targets.shape != (self.n_envs, N_lasers):
            raise ValueError(
                f"targets must have shape ({self.n_envs}, {N_lasers})"
            )
        self.targets = targets.copy()
        self.history, frequency_history, _, _ = (
            self.vcsel.generate_history(
                self.nd, shape="FR", n_cases=self.n_envs
            )
        )
        self.kappa_links_ns = np.zeros((self.n_envs, n_links))
        self.phi_p_links = np.full(
            (self.n_envs, n_links), initial_phi_p
        )
        self.control_step = 0
        self.success_counter = np.zeros(self.n_envs, dtype=int)
        self.last_frequency_ghz = frequency_history[:, :, -1]
        self.last_state = self.history[:, :, -1]
        self.previous_phase_score, _ = phase_behavior_scores(
            self.last_state[:, 2::3], self.targets
        )
        initial_control_features = self._control_features(
            self.last_state, self.last_frequency_ghz
        )
        self.recent_control_features = np.repeat(
            initial_control_features[:, None, :],
            controller_history_steps,
            axis=1,
        )
        return self._observation(
            self.last_state, self.last_frequency_ghz
        )

    def _control_features(self, state, frequencies_ghz):
        """Features retained across recent control decisions."""
        current_phases = state[:, 2::3]
        relative_frequency = (
            frequencies_ghz[:, 1:] - frequencies_ghz[:, :1]
        ) / detuning_span_ghz
        return np.c_[
            phase_error_features(current_phases, self.targets),
            np.clip(relative_frequency, -3.0, 3.0),
        ]

    def _observation(self, state, frequencies_ghz):
        current_phases = state[:, 2::3]
        photons = np.maximum(state[:, 1::3], 1e-12)
        photon_scale = float(np.asarray(self.nd["sbar"]).reshape(-1)[0])
        normalized_photons = np.clip(
            np.log(photons / max(photon_scale, 1e-12)) / 5.0,
            -2.0,
            2.0,
        )
        observation = np.c_[
            phase_features(self.targets),
            phase_features(current_phases),
            phase_error_features(current_phases, self.targets),
            self.recent_control_features.reshape(self.n_envs, -1),
            normalized_photons,
            self.kappa_links_ns / maximum_kappa_ns,
            np.cos(self.phi_p_links),
            np.sin(self.phi_p_links),
            np.full(
                (self.n_envs, 1),
                self.control_step / episode_control_steps,
            ),
        ]
        if observation.shape != (self.n_envs, observation_size):
            raise RuntimeError("unexpected controller observation shape")
        return observation.astype(np.float32)

    def step(self, raw_actions, record=False):
        """Apply bounded kappa/phi_p changes and integrate one chunk."""
        raw_actions = np.asarray(raw_actions, dtype=float)
        if raw_actions.shape != (self.n_envs, action_size):
            raise ValueError(
                f"actions must have shape ({self.n_envs}, {action_size})"
            )
        normalized_change = np.clip(raw_actions, -1.0, 1.0)
        delta_kappa_ns = (
            maximum_kappa_change_per_step_ns
            * normalized_change[:, :n_links]
        )
        delta_phi_p = (
            maximum_phi_p_change_per_step_rad
            * normalized_change[:, n_links:]
        )
        old_kappa = self.kappa_links_ns.copy()
        old_phi_p = self.phi_p_links.copy()
        self.kappa_links_ns = np.clip(
            self.kappa_links_ns + delta_kappa_ns,
            0.0,
            maximum_kappa_ns,
        )
        self.phi_p_links = np.angle(
            np.exp(1j * (self.phi_p_links + delta_phi_p))
        )
        actual_kappa_change = self.kappa_links_ns - old_kappa
        actual_phi_p_change = np.angle(
            np.exp(1j * (self.phi_p_links - old_phi_p))
        )
        old_kappa_matrices_ns = links_to_matrices(old_kappa)
        kappa_matrices_ns = links_to_matrices(self.kappa_links_ns)
        old_phi_p_matrices = links_to_matrices(old_phi_p)
        phi_p_matrices = links_to_matrices(self.phi_p_links)
        # The first 2*tau entries are the supplied history. The new part of
        # this chunk linearly ramps every link from its previous value to its
        # newly commanded value, so neither control has step discontinuities.
        chunk_ramp = np.zeros(int(self.nd["steps"]))
        first_new_index = 2 * int(self.nd["delay_steps"]) - 1
        chunk_ramp[first_new_index:] = np.linspace(
            0.0,
            1.0,
            len(chunk_ramp) - first_new_index,
        )
        kappa_series_ns = (
            old_kappa_matrices_ns[None, :, :, :]
            + chunk_ramp[:, None, None, None]
            * (
                kappa_matrices_ns - old_kappa_matrices_ns
            )[None, :, :, :]
        )
        phi_p_series = (
            old_phi_p_matrices[None, :, :, :]
            + chunk_ramp[:, None, None, None]
            * links_to_matrices(actual_phi_p_change)[None, :, :, :]
        )
        nd_run = dict(self.nd)
        nd_run["kappa"] = kappa_series_ns * 1e9 * tau_p
        nd_run["kappa_case_dependent"] = True
        nd_run["phi_p"] = phi_p_series
        (
            t_chunk,
            states,
            frequencies,
            final_history,
        ) = self.vcsel.integrate(
            self.history,
            nd=nd_run,
            progress=False,
            max_iter=1,
            smooth_freqs=False,
            return_final_history=True,
        )
        self.history = final_history
        frequencies_ghz = frequencies / (
            2.0 * np.pi * tau_p * 1e9
        )
        new_mask = t_chunk >= (
            self.delay_time - 0.5 * dt
        )
        if not np.any(new_mask):
            new_mask[-1] = True
        new_states = states[:, :, new_mask]
        new_frequencies = frequencies_ghz[:, :, new_mask]
        self.last_state = new_states[:, :, -1]
        self.last_frequency_ghz = np.mean(
            new_frequencies, axis=2
        )

        phases = new_states[:, 2::3, :]
        phase_score, worst_phase = phase_behavior_scores(
            phases, self.targets
        )
        mean_phase = phase_score
        common_frequency = new_frequencies.mean(
            axis=1, keepdims=True
        )
        frequency_spread = np.sqrt(
            np.mean(
                (new_frequencies - common_frequency) ** 2,
                axis=(1, 2),
            )
        )
        # This rational score retains useful distinctions after lock is lost;
        # the previous exponential score numerically collapsed to zero.
        frequency_score = 1.0 / (
            1.0
            + frequency_spread / frequency_tolerance_ghz
        )
        kappa_penalty = np.mean(
            (self.kappa_links_ns / maximum_kappa_ns) ** 2,
            axis=1,
        )
        kappa_change_penalty = np.mean(
            (
                actual_kappa_change
                / maximum_kappa_change_per_step_ns
            )
            ** 2,
            axis=1,
        )
        phi_p_change_penalty = np.mean(
            (
                actual_phi_p_change
                / maximum_phi_p_change_per_step_rad
            )
            ** 2,
            axis=1,
        )
        phase_improvement = phase_score - self.previous_phase_score
        reward = (
            phase_score
            + phase_improvement_weight * phase_improvement
        )
        reward -= (
            kappa_magnitude_penalty_weight * kappa_penalty
            + kappa_change_penalty_weight * kappa_change_penalty
            + phi_p_change_penalty_weight * phi_p_change_penalty
        )
        terminal_step = self.control_step + 1 >= episode_control_steps
        if terminal_step:
            reward += terminal_reward_weight * phase_score
        successful = (
            (phase_score >= success_phase_score)
            & (worst_phase >= success_worst_phase_score)
        )
        self.success_counter = np.where(
            successful, self.success_counter + 1, 0
        )
        sustained_success = (
            self.success_counter >= success_dwell_steps
        )
        reward += success_bonus * sustained_success
        finite = np.all(
            np.isfinite(new_states), axis=(1, 2)
        )
        finite &= np.all(
            np.isfinite(new_frequencies), axis=(1, 2)
        )
        reward = np.where(finite, reward, -1.0)
        self.previous_phase_score = phase_score.copy()
        self.control_step += 1
        new_control_features = self._control_features(
            self.last_state, self.last_frequency_ghz
        )
        self.recent_control_features = np.concatenate(
            (
                self.recent_control_features[:, 1:, :],
                new_control_features[:, None, :],
            ),
            axis=1,
        )
        observation = self._observation(
            self.last_state, self.last_frequency_ghz
        )
        info = {
            "phase_score": phase_score,
            "mean_phase_score": mean_phase,
            "worst_phase_score": worst_phase,
            "frequency_score": frequency_score,
            "frequency_spread_ghz": frequency_spread,
            "phase_improvement": phase_improvement,
            "kappa_penalty": kappa_penalty,
            "kappa_change_penalty": kappa_change_penalty,
            "phi_p_change_penalty": phi_p_change_penalty,
            "success": sustained_success.astype(float),
            "kappa_ns": kappa_matrices_ns,
            "phi_p": phi_p_matrices,
        }
        if record:
            series_indices = np.clip(
                np.rint(t_chunk / dt).astype(int),
                0,
                len(chunk_ramp) - 1,
            )
            info.update(
                {
                    "time_chunk": t_chunk[new_mask] - self.delay_time,
                    "state_chunk": new_states,
                    "frequency_chunk_ghz": new_frequencies,
                    "kappa_chunk_ns": kappa_series_ns[
                        series_indices[new_mask]
                    ],
                    "phi_p_chunk": phi_p_series[
                        series_indices[new_mask]
                    ],
                }
            )
        return observation, reward.astype(np.float32), info


# ----------------- Replay-buffered TD3 -----------------
def make_mlp(input_size, output_size, output_tanh=False):
    layers = []
    previous = input_size
    for width in hidden_units:
        layer = nn.Linear(previous, width)
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.ReLU()))
        previous = width
    output = nn.Linear(previous, output_size)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.zeros_(output.bias)
    layers.append(output)
    if output_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = make_mlp(
            observation_size, action_size, output_tanh=True
        )

    def forward(self, observation):
        return self.network(observation)


class TwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = observation_size + action_size
        self.q1 = make_mlp(input_size, 1)
        self.q2 = make_mlp(input_size, 1)

    def forward(self, observation, action):
        inputs = torch.cat((observation, action), dim=1)
        return self.q1(inputs).squeeze(1), self.q2(inputs).squeeze(1)

    def first(self, observation, action):
        return self.q1(torch.cat((observation, action), dim=1)).squeeze(1)


class ReplayBuffer:
    """Fixed-size float32 replay memory for off-policy learning."""

    def __init__(self, capacity=None):
        self.capacity = int(
            replay_capacity if capacity is None else capacity
        )
        self.observations = np.empty(
            (self.capacity, observation_size), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(
            (self.capacity, action_size), dtype=np.float32
        )
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.discounts = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, observation, action, reward, next_observation, discount, done):
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.discounts[index] = discount
        self.dones[index] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch_size=None):
        if batch_size is None:
            batch_size = replay_batch_size
        indices = rng.integers(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(array[indices], dtype=torch.float32)
            for array in (
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.discounts,
                self.dones,
            )
        )

    def __len__(self):
        return self.size


class NStepReplayWriter:
    """Convert parallel trajectories into n-step replay transitions."""

    def __init__(self, n_envs, replay_buffer):
        self.queues = [deque() for _ in range(n_envs)]
        self.replay_buffer = replay_buffer

    def _write_oldest(self, queue):
        reward_sum = 0.0
        discount = 1.0
        terminal = False
        next_observation = queue[0][3]
        used = 0
        for _, _, reward, candidate_next, done in queue:
            reward_sum += discount * float(reward)
            used += 1
            next_observation = candidate_next
            terminal = bool(done)
            discount *= discount_factor
            if terminal or used >= n_step_return:
                break
        observation, action = queue[0][:2]
        self.replay_buffer.add(
            observation,
            action,
            reward_sum,
            next_observation,
            discount,
            terminal,
        )
        queue.popleft()

    def append(self, observations, actions, rewards, next_observations, dones):
        for env_index, queue in enumerate(self.queues):
            queue.append(
                (
                    observations[env_index].copy(),
                    actions[env_index].copy(),
                    float(rewards[env_index]),
                    next_observations[env_index].copy(),
                    bool(dones[env_index]),
                )
            )
            if len(queue) >= n_step_return:
                self._write_oldest(queue)
            if dones[env_index]:
                while queue:
                    self._write_oldest(queue)


class TD3Agent:
    def __init__(self):
        self.actor = Actor()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TwinCritic()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )
        self.gradient_step = 0

    def deterministic_actions(self, observations):
        with torch.no_grad():
            return self.actor(
                torch.as_tensor(observations, dtype=torch.float32)
            ).cpu().numpy()

    def train_step(self, replay_buffer, rng):
        batch = replay_buffer.sample(rng)
        (
            observations,
            actions,
            rewards,
            next_observations,
            discounts,
            dones,
        ) = batch
        with torch.no_grad():
            noise = torch.randn_like(actions) * target_policy_noise
            noise.clamp_(-target_noise_clip, target_noise_clip)
            next_actions = (
                self.actor_target(next_observations) + noise
            ).clamp(-1.0, 1.0)
            next_q1, next_q2 = self.critic_target(
                next_observations, next_actions
            )
            target_q = rewards + discounts * (1.0 - dones) * torch.minimum(
                next_q1, next_q2
            )
        q1, q2 = self.critic(observations, actions)
        critic_loss = torch.mean((q1 - target_q) ** 2) + torch.mean(
            (q2 - target_q) ** 2
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), gradient_clip
        )
        self.critic_optimizer.step()

        self.gradient_step += 1
        actor_loss = np.nan
        if self.gradient_step % policy_update_delay == 0:
            predicted_actions = self.actor(observations)
            actor_objective = -self.critic.first(
                observations, predicted_actions
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), gradient_clip
            )
            self.actor_optimizer.step()
            actor_loss = float(actor_objective.detach())
            with torch.no_grad():
                for target, source in (
                    (self.actor_target, self.actor),
                    (self.critic_target, self.critic),
                ):
                    for target_parameter, source_parameter in zip(
                        target.parameters(), source.parameters()
                    ):
                        target_parameter.mul_(1.0 - target_update_rate)
                        target_parameter.add_(
                            source_parameter, alpha=target_update_rate
                        )
        return actor_loss, float(critic_loss.detach())


def evaluate_controller(actor, environment, targets):
    observation = environment.reset(targets)
    info = None
    for _ in range(episode_control_steps):
        with torch.no_grad():
            actions = actor(
                torch.as_tensor(observation, dtype=torch.float32)
            ).cpu().numpy()
        observation, _, info = environment.step(actions)
    return {
        "phase_score": float(np.mean(info["phase_score"])),
        "frequency_score": float(np.mean(info["frequency_score"])),
        "success_rate": float(np.mean(info["success"])),
        "requested_phase_score": float(info["phase_score"][0]),
        "requested_frequency_score": float(info["frequency_score"][0]),
        "requested_success_rate": float(info["success"][0]),
        "requested_kappa_ns": info["kappa_ns"][0].copy(),
        "requested_phi_p": info["phi_p"][0].copy(),
    }


def plot_training_progress(histories, episode, final_kappa, final_phi_p):
    clear_output(wait=True)
    x = np.arange(1, episode + 2)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    reward_history = histories["episode_reward"][: episode + 1]
    axes[0, 0].plot(x, reward_history, alpha=0.35)
    window = min(20, episode + 1)
    moving = np.convolve(
        reward_history, np.ones(window) / window, mode="valid"
    )
    axes[0, 0].plot(
        x[window - 1 :], moving, linewidth=2, label=f"{window}-episode mean"
    )
    axes[0, 0].set(
        xlabel="training episode",
        ylabel="mean reward per step",
        title="Replay-buffered TD3 reward",
    )
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    for key, label, style in (
        ("phase_score", "training phase", {"alpha": 0.35}),
        ("frequency_score", "training frequency", {"alpha": 0.35}),
        ("requested_phase_score", "requested phase", {"linewidth": 2}),
        ("requested_frequency_score", "requested frequency", {"linewidth": 2}),
    ):
        axes[0, 1].plot(x, histories[key][: episode + 1], label=label, **style)
    axes[0, 1].set(
        xlabel="training episode",
        ylabel="score",
        ylim=(0.0, 1.0),
        title="Training and deterministic evaluation",
    )
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(
        x, histories["critic_loss"][: episode + 1], label="twin critic"
    )
    axes[0, 2].plot(
        x, histories["actor_loss"][: episode + 1], label="actor"
    )
    axes[0, 2].set(
        xlabel="training episode", ylabel="loss", title="TD3 optimization"
    )
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(
        x, histories["mean_kappa_ns"][: episode + 1], label=r"mean $\kappa$"
    )
    axes[1, 0].plot(
        x,
        histories["mean_kappa_change_ns"][: episode + 1],
        label=r"mean $|\Delta\kappa|$",
    )
    axes[1, 0].set(
        xlabel="training episode",
        ylabel=r"ns$^{-1}$",
        title="Continuous control effort",
    )
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    phi_axis = axes[1, 0].twinx()
    phi_axis.plot(
        x,
        histories["mean_phi_p_change_rad"][: episode + 1],
        color="tab:red",
    )
    phi_axis.set_ylabel(r"mean $|\Delta\phi_p|$ (rad)", color="tab:red")

    kappa_image = axes[1, 1].imshow(
        final_kappa.T, origin="upper", vmin=0.0, vmax=maximum_kappa_ns
    )
    axes[1, 1].set(
        xlabel="receiver",
        ylabel="source",
        title=r"Requested-target final $\kappa$ (ns$^{-1}$)",
    )
    fig.colorbar(kappa_image, ax=axes[1, 1])
    phi_image = axes[1, 2].imshow(
        final_phi_p.T,
        origin="upper",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1, 2].set(
        xlabel="receiver",
        ylabel="source",
        title=r"Requested-target final $\phi_p$ (rad)",
    )
    fig.colorbar(phi_image, ax=axes[1, 2])
    fig.suptitle(
        f"episode {episode + 1}/{n_training_episodes} | "
        f"{parallel_environments} FR simulations | "
        f"replay {int(histories['replay_size'][episode])}/{replay_capacity} | "
        f"requested phase {histories['requested_phase_score'][episode]:.3f}"
    )
    if "agg" not in plt.get_backend().lower():
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


def train_dynamic_controller():
    """Train all directed kappa/phi_p controls with replay-buffered TD3."""
    rng = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
    torch.set_num_threads(1)
    agent = TD3Agent()
    replay_buffer = ReplayBuffer()
    replay_writer = NStepReplayWriter(parallel_environments, replay_buffer)
    environment = DynamicCouplingEnvironment(
        parallel_environments, seed=random_seed + 1
    )
    validation_environment = DynamicCouplingEnvironment(
        evaluation_targets, seed=random_seed + 2
    )
    validation_targets = sample_random_targets(rng, evaluation_targets)
    validation_targets[0] = np.asarray(requested_phases)
    history_names = (
        "episode_reward",
        "phase_score",
        "worst_phase_score",
        "frequency_score",
        "success_rate",
        "validation_phase_score",
        "validation_frequency_score",
        "validation_success_rate",
        "requested_phase_score",
        "requested_frequency_score",
        "requested_success_rate",
        "mean_kappa_ns",
        "mean_kappa_change_ns",
        "mean_phi_p_change_rad",
        "actor_loss",
        "critic_loss",
        "replay_size",
    )
    histories = {
        name: np.full(n_training_episodes, np.nan) for name in history_names
    }
    requested_kappa_ns = np.zeros((N_lasers, N_lasers))
    requested_phi_p = np.zeros((N_lasers, N_lasers))
    total_transitions = 0
    print(
        f"Training replay-buffered TD3 with {parallel_environments} parallel "
        "free-running episodes."
    )
    print(
        f"Controlling {n_links} kappa and {n_links} phi_p links "
        f"({action_size} actions); learning starts at {learning_starts} "
        f"stored transitions."
    )
    for episode in range(n_training_episodes):
        targets = sample_random_targets(rng, parallel_environments)
        observation = environment.reset(targets)
        episode_return = np.zeros(parallel_environments, dtype=np.float32)
        phase_scores = []
        worst_phase_scores = []
        frequency_scores = []
        success_scores = []
        mean_kappa_changes = []
        mean_phi_p_changes = []
        actor_losses = []
        critic_losses = []
        for step in range(episode_control_steps):
            if total_transitions < learning_starts:
                actions = rng.uniform(
                    -1.0, 1.0, (parallel_environments, action_size)
                )
            else:
                actions = agent.deterministic_actions(observation)
                actions += rng.normal(
                    0.0, exploration_noise, actions.shape
                )
                actions = np.clip(actions, -1.0, 1.0)
            next_observation, reward, info = environment.step(actions)
            done = np.full(
                parallel_environments,
                step + 1 == episode_control_steps,
            )
            replay_writer.append(
                observation, actions, reward, next_observation, done
            )
            total_transitions += parallel_environments
            if len(replay_buffer) >= max(learning_starts, replay_batch_size):
                for _ in range(gradient_updates_per_control_step):
                    actor_loss, critic_loss = agent.train_step(
                        replay_buffer, rng
                    )
                    if np.isfinite(actor_loss):
                        actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
            episode_return += reward
            phase_scores.append(info["phase_score"])
            worst_phase_scores.append(info["worst_phase_score"])
            frequency_scores.append(info["frequency_score"])
            success_scores.append(info["success"])
            mean_kappa_changes.append(
                np.mean(
                    np.abs(actions[:, :n_links])
                    * maximum_kappa_change_per_step_ns
                )
            )
            mean_phi_p_changes.append(
                np.mean(
                    np.abs(actions[:, n_links:])
                    * maximum_phi_p_change_per_step_rad
                )
            )
            observation = next_observation

        histories["episode_reward"][episode] = (
            np.mean(episode_return) / episode_control_steps
        )
        histories["phase_score"][episode] = np.mean(phase_scores[-1])
        histories["worst_phase_score"][episode] = np.mean(
            worst_phase_scores[-1]
        )
        histories["frequency_score"][episode] = np.mean(
            frequency_scores[-1]
        )
        histories["success_rate"][episode] = np.mean(success_scores[-1])
        histories["mean_kappa_ns"][episode] = np.mean(
            environment.kappa_links_ns
        )
        histories["mean_kappa_change_ns"][episode] = np.mean(
            mean_kappa_changes
        )
        histories["mean_phi_p_change_rad"][episode] = np.mean(
            mean_phi_p_changes
        )
        histories["actor_loss"][episode] = (
            np.mean(actor_losses) if actor_losses else np.nan
        )
        histories["critic_loss"][episode] = (
            np.mean(critic_losses) if critic_losses else np.nan
        )
        histories["replay_size"][episode] = len(replay_buffer)

        should_evaluate = (
            episode == 0
            or (episode + 1) % evaluation_every == 0
            or episode + 1 == n_training_episodes
        )
        if should_evaluate:
            evaluation = evaluate_controller(
                agent.actor, validation_environment, validation_targets
            )
            requested_kappa_ns = evaluation["requested_kappa_ns"]
            requested_phi_p = evaluation["requested_phi_p"]
            for name in (
                "validation_phase_score",
                "validation_frequency_score",
                "validation_success_rate",
                "requested_phase_score",
                "requested_frequency_score",
                "requested_success_rate",
            ):
                key = (
                    name.removeprefix("validation_")
                    if name.startswith("validation_")
                    else name
                )
                histories[name][episode] = evaluation[key]
        elif episode > 0:
            for name in (
                "validation_phase_score",
                "validation_frequency_score",
                "validation_success_rate",
                "requested_phase_score",
                "requested_frequency_score",
                "requested_success_rate",
            ):
                histories[name][episode] = histories[name][episode - 1]
        print(
            f"episode {episode + 1:3d}/{n_training_episodes} | "
            f"replay {len(replay_buffer):6d} | "
            f"reward {histories['episode_reward'][episode]:.3f} | "
            f"phase {histories['phase_score'][episode]:.3f} | "
            f"requested {histories['requested_phase_score'][episode]:.3f}"
        )
        if show_inline_plot and (
            (episode + 1) % plot_every == 0 or episode == 0
        ):
            plot_training_progress(
                histories, episode, requested_kappa_ns, requested_phi_p
            )

    checkpoint = {
        "training_method": "full_matrix_td3_nstep_v1",
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        "observation_size": observation_size,
        "action_size": action_size,
        "hidden_units": hidden_units,
        "controller_history_steps": controller_history_steps,
        "controlled_link_i": link_i.copy(),
        "controlled_link_j": link_j.copy(),
        "n_step_return": n_step_return,
        "requested_phases": np.asarray(requested_phases),
        "histories": histories,
    }
    Path(controller_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, controller_file)
    print(f"Saved the TD3 actor to {controller_file}")
    return agent.actor, histories


def load_dynamic_controller(filename=controller_file):
    checkpoint = torch.load(
        filename, map_location="cpu", weights_only=False
    )
    if checkpoint.get("training_method") != "full_matrix_td3_nstep_v1":
        raise ValueError("Saved file is not a compatible TD3 controller")
    if (
        int(checkpoint["observation_size"]) != observation_size
        or int(checkpoint["action_size"]) != action_size
    ):
        raise ValueError("Controller dimensions changed; retrain it")
    actor = Actor()
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


# Deterministic validation with time-varying coupling
def simulate_dynamic_controller(
    target_phases=None,
    filename=controller_file,
):
    """Run one FR episode with the deterministic trained controller."""
    from scipy.constants import c, hbar

    if target_phases is None:
        target_phases = requested_phases
    controller = load_dynamic_controller(filename)
    environment = DynamicCouplingEnvironment(1, seed=random_seed + 50_000)
    observation = environment.reset(np.asarray(target_phases))
    time_chunks = []
    state_chunks = []
    frequency_chunks = []
    kappa_chunks = []
    phi_p_chunks = []
    elapsed = 0.0
    for _ in range(episode_control_steps):
        with torch.no_grad():
            action = controller(
                torch.as_tensor(observation, dtype=torch.float32)
            )
        observation, _, info = environment.step(
            action.cpu().numpy(), record=True
        )
        local_time = info["time_chunk"] + elapsed
        elapsed = local_time[-1] + dt * save_every
        time_chunks.append(local_time)
        state_chunks.append(info["state_chunk"][0])
        frequency_chunks.append(info["frequency_chunk_ghz"][0])
        kappa_chunks.append(info["kappa_chunk_ns"][:, 0])
        phi_p_chunks.append(info["phi_p_chunk"][:, 0])

    time_s = np.concatenate(time_chunks)
    states = np.concatenate(state_chunks, axis=1)
    frequencies_ghz = np.concatenate(frequency_chunks, axis=1)
    kappa_history = np.concatenate(kappa_chunks, axis=0)
    phi_p_history = np.angle(
        np.exp(1j * np.concatenate(phi_p_chunks, axis=0))
    )
    photons = states[1::3]
    phases = states[2::3]
    relative_phases = np.angle(
        np.exp(1j * (phases - phases[:1]))
    )
    omega0 = 2.0 * np.pi * c / 910e-9
    intensity_to_mw = 1e3 * hbar * omega0 / (g0 * tau_n * tau_p)
    laser_power_mw = photons * intensity_to_mw
    total_field = np.sum(
        np.sqrt(np.maximum(photons, 0.0)) * np.exp(1j * phases),
        axis=0,
    )
    total_power_mw = np.abs(total_field) ** 2 * intensity_to_mw

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 20), dpi=180, sharex=True
    )
    time_ns = time_s * 1e9
    for laser in range(N_lasers):
        axes[0].plot(time_ns, frequencies_ghz[laser], linewidth=1.5)
        axes[1].plot(time_ns, relative_phases[laser], linewidth=1.2)
        axes[2].plot(time_ns, laser_power_mw[laser], linewidth=1.5)
    axes[0].set_ylabel(r"$\dot{\phi}$ (GHz)", fontsize=18)
    axes[0].set_title(
        "Closed-loop validation from the free-running state",
        fontsize=22,
    )
    axes[1].set_ylabel(
        r"wrapped $\phi_i-\phi_1$ (rad)", fontsize=18
    )
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].set_yticks([-np.pi, 0.0, np.pi])
    axes[1].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    axes[2].plot(
        time_ns,
        total_power_mw,
        color="green",
        linewidth=2.2,
        label=r"$P_{\rm total}$",
    )
    axes[2].set_ylabel("Output power (mW)", fontsize=18)

    active_link_history = kappa_history[:, link_i, link_j]
    axes[3].plot(
        time_ns,
        np.mean(active_link_history, axis=1),
        color="black",
        linewidth=2,
        label=r"mean $\kappa$",
    )
    axes[3].fill_between(
        time_ns,
        np.min(active_link_history, axis=1),
        np.max(active_link_history, axis=1),
        alpha=0.25,
        label="link range",
    )
    axes[3].set_ylabel(r"$\kappa$ (ns$^{-1}$)", fontsize=18)
    axes[3].set_ylim(0.0, maximum_kappa_ns)
    axes[3].legend(fontsize=12)

    active_phi_p_history = phi_p_history[:, link_i, link_j]
    for link in range(min(10, n_links)):
        axes[4].plot(
            time_ns,
            active_phi_p_history[:, link],
            linewidth=1.2,
            label=f"link {link + 1}",
        )
    axes[4].set_ylabel(r"$\phi_p$ (rad)", fontsize=18)
    axes[4].set_xlabel("Time (ns)", fontsize=18)
    axes[4].set_ylim(-np.pi, np.pi)
    axes[4].set_yticks([-np.pi, 0.0, np.pi])
    axes[4].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    axes[4].legend(ncol=5, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=14)
    plt.show()
    return {
        "time_s": time_s,
        "states": states,
        "frequencies_ghz": frequencies_ghz,
        "kappa_history_ns": kappa_history,
        "phi_p_history": phi_p_history,
        "target_phases": np.asarray(target_phases),
    }


if __name__ == "__main__":
    train_dynamic_controller()
    validation_result = simulate_dynamic_controller()
