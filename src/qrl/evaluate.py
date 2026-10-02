"""Evaluation of trained policies, including finite-shot circuit estimates."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from qrl.policies import QuantumPolicy
from qrl.reinforce import collect_episodes, make_envs


def evaluate_quantum_policy(
    policy: QuantumPolicy,
    n_episodes: int,
    seed: int,
    shots: int | None = None,
    env_id: str = "CartPole-v1",
) -> np.ndarray:
    """Returns of ``n_episodes`` evaluation episodes.

    With ``shots=None`` the circuit expectation is exact, as during training.
    With an integer ``shots`` every decision instead uses an estimate from that
    many measurement samples, which is what the policy would have access to on
    a quantum device.  The trained parameters are not adapted to the noise.
    """
    generator = torch.Generator().manual_seed(seed)

    def logits(observation: Tensor) -> Tensor:
        expectation = policy.expectation(observation, shots=shots, generator=generator)
        return policy.logits_from_expectation(expectation)

    envs = make_envs(env_id, n_episodes, seed)
    episodes = collect_episodes(envs, logits, generator)
    for env in envs:
        env.close()
    return np.array([episode.total_reward for episode in episodes])
