"""Monte-Carlo policy gradient (REINFORCE) for discrete-action Gymnasium tasks.

The algorithm is deliberately the simplest one that works: it isolates the
effect of the *policy class* (quantum circuit vs. neural network) from the
machinery of more elaborate agents (critics, replay buffers, target networks).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn

from qrl.policies import (
    MLPPolicy,
    QuantumPolicy,
    count_trainable_parameters,
    normalise_observation,
)


@dataclass(frozen=True)
class TrainConfig:
    """Everything needed to reproduce one training run."""

    name: str
    policy: str  # "quantum" or "mlp"
    seed: int = 0
    env_id: str = "CartPole-v1"
    n_updates: int = 100
    episodes_per_update: int = 10
    gamma: float = 0.99
    learning_rate: float = 0.01
    # Quantum policy only.
    n_layers: int = 5
    trainable_input_scaling: bool = True
    trainable_output_scale: bool = True
    scaling_learning_rate: float = 0.1  # for input scalings and output scale
    # MLP policy only.
    hidden_sizes: tuple[int, ...] = (10,)


@dataclass
class TrainResult:
    config: dict[str, Any]
    n_parameters: int
    episode_returns: list[list[float]]  # [update][episode], on-policy
    wall_time_s: float
    state_dict: dict[str, Tensor] = field(repr=False, default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable summary (the weights are saved separately)."""
        return {
            "config": self.config,
            "n_parameters": self.n_parameters,
            "episode_returns": self.episode_returns,
            "wall_time_s": self.wall_time_s,
        }


def build_policy(config: TrainConfig) -> nn.Module:
    if config.policy == "quantum":
        return QuantumPolicy(
            n_layers=config.n_layers,
            trainable_input_scaling=config.trainable_input_scaling,
            trainable_output_scale=config.trainable_output_scale,
        )
    if config.policy == "mlp":
        return MLPPolicy(hidden_sizes=tuple(config.hidden_sizes))
    raise ValueError(f"unknown policy type: {config.policy!r}")


def build_optimizer(policy: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Adam; the quantum policy's scaling parameters get their own step size.

    Rotation angles live on a 2*pi-periodic scale while the input scalings and
    the output scale are unbounded multipliers, so a single learning rate fits
    both poorly; separate step sizes for these groups follow Jerbi et al. (2021)
    and Skolik et al. (2022).
    """
    if isinstance(policy, QuantumPolicy):
        groups = [{"params": [policy.theta], "lr": config.learning_rate}]
        scalings = [p for p in (policy.input_scaling, policy.output_scale) if p.requires_grad]
        if scalings:
            groups.append({"params": scalings, "lr": config.scaling_learning_rate})
        return torch.optim.Adam(groups)
    return torch.optim.Adam(policy.parameters(), lr=config.learning_rate)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Reward-to-go ``G_t = sum_{k >= t} gamma^(k - t) r_k``."""
    returns = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


@dataclass
class Episode:
    observations: np.ndarray  # (T, obs_dim), raw environment observations
    actions: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())


@torch.no_grad()
def collect_episodes(
    envs: list[gym.Env],
    policy: Callable[[Tensor], Tensor],
    generator: torch.Generator,
) -> list[Episode]:
    """Run one episode in each environment, batching the policy evaluations.

    ``policy`` maps normalised observations to action logits; actions are
    sampled from the corresponding softmax distribution.

    All environments that are still running are stepped together so the
    policy (in particular the circuit simulation) is evaluated once per time
    step rather than once per environment.
    """
    observations = [env.reset()[0] for env in envs]
    buffers: list[tuple[list, list, list]] = [([], [], []) for _ in envs]
    active = list(range(len(envs)))

    while active:
        batch = torch.as_tensor(np.stack([observations[i] for i in active]))
        logits = policy(normalise_observation(batch))
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1, generator=generator).squeeze(-1).tolist()

        still_active = []
        for i, action in zip(active, actions, strict=True):
            next_observation, reward, terminated, truncated, _ = envs[i].step(action)
            obs_buf, act_buf, rew_buf = buffers[i]
            obs_buf.append(observations[i])
            act_buf.append(action)
            rew_buf.append(reward)
            observations[i] = next_observation
            if not (terminated or truncated):
                still_active.append(i)
        active = still_active

    return [
        Episode(np.asarray(obs), np.asarray(act, dtype=np.int64), np.asarray(rew, dtype=np.float64))
        for obs, act, rew in buffers
    ]


def policy_gradient_loss(policy: nn.Module, episodes: list[Episode], gamma: float) -> Tensor:
    """REINFORCE surrogate loss with standardised reward-to-go.

    Subtracting the batch mean is a constant baseline (it leaves the gradient
    unbiased); dividing by the standard deviation only rescales the step.
    """
    observations = torch.as_tensor(np.concatenate([ep.observations for ep in episodes]))
    actions = torch.as_tensor(np.concatenate([ep.actions for ep in episodes]))
    returns = torch.as_tensor(
        np.concatenate([discounted_returns(ep.rewards, gamma) for ep in episodes])
    )
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.log_softmax(policy(normalise_observation(observations)), dim=-1)
    chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return -(chosen * advantages).sum() / len(episodes)


def make_envs(env_id: str, n_envs: int, seed: int) -> list[gym.Env]:
    """Create independently seeded environments."""
    envs = [gym.make(env_id) for _ in range(n_envs)]
    for index, env in enumerate(envs):
        env.reset(seed=seed * 1000 + index)
    return envs


def train(config: TrainConfig, verbose: bool = False) -> TrainResult:
    """Train one policy with REINFORCE for a fixed budget of episodes."""
    torch.manual_seed(config.seed)
    sampler = torch.Generator().manual_seed(config.seed)
    policy = build_policy(config)
    optimizer = build_optimizer(policy, config)
    envs = make_envs(config.env_id, config.episodes_per_update, config.seed)

    history: list[list[float]] = []
    start = time.perf_counter()
    for update in range(config.n_updates):
        episodes = collect_episodes(envs, policy, sampler)
        loss = policy_gradient_loss(policy, episodes, config.gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append([ep.total_reward for ep in episodes])
        if verbose and (update + 1) % 10 == 0:
            print(
                f"[{config.name} seed={config.seed}] update {update + 1:4d} "
                f"mean return {np.mean(history[-1]):7.1f}",
                flush=True,
            )
    for env in envs:
        env.close()

    return TrainResult(
        config=asdict(config),
        n_parameters=count_trainable_parameters(policy),
        episode_returns=history,
        wall_time_s=time.perf_counter() - start,
        state_dict={k: v.detach().clone() for k, v in policy.state_dict().items()},
    )
