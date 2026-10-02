import numpy as np
import pytest
import torch

from qrl.policies import normalise_observation
from qrl.reinforce import (
    Episode,
    TrainConfig,
    build_optimizer,
    build_policy,
    discounted_returns,
    policy_gradient_loss,
    train,
)


def test_discounted_returns_hand_computed():
    returns = discounted_returns(np.array([1.0, 1.0, 1.0]), gamma=0.5)
    np.testing.assert_allclose(returns, [1.75, 1.5, 1.0])


def test_discounted_returns_without_discount_is_reverse_cumsum():
    rewards = np.array([2.0, -1.0, 4.0, 0.5])
    np.testing.assert_allclose(discounted_returns(rewards, 1.0), np.cumsum(rewards[::-1])[::-1])


def test_loss_gradient_matches_the_score_function_formula():
    """Check autograd against sum_t A_t * grad log pi(a_t | s_t), by hand."""
    torch.manual_seed(0)
    config = TrainConfig("t", "mlp", hidden_sizes=())  # linear-softmax policy
    policy = build_policy(config)
    rng = np.random.default_rng(0)
    episodes = [
        Episode(rng.normal(size=(t, 4)), rng.integers(0, 2, size=t), np.ones(t)) for t in (3, 5)
    ]
    policy_gradient_loss(policy, episodes, gamma=0.9).backward()
    weight_grad = policy.network[0].weight.grad

    observations = torch.as_tensor(np.concatenate([e.observations for e in episodes]))
    actions = np.concatenate([e.actions for e in episodes])
    returns = torch.as_tensor(
        np.concatenate([discounted_returns(e.rewards, 0.9) for e in episodes])
    )
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
    inputs = normalise_observation(observations)
    with torch.no_grad():
        probs = torch.softmax(policy(inputs), -1)
    one_hot = torch.nn.functional.one_hot(torch.as_tensor(actions), 2).to(probs.dtype)
    # d log pi(a|s) / d W = (onehot(a) - pi) s^T for a linear-softmax policy.
    expected = -((advantages[:, None] * (one_hot - probs)).T @ inputs) / len(episodes)
    assert torch.allclose(weight_grad, expected, atol=1e-10)


def test_quantum_optimizer_uses_separate_learning_rates():
    config = TrainConfig("t", "quantum", learning_rate=0.01, scaling_learning_rate=0.1)
    optimizer = build_optimizer(build_policy(config), config)
    assert [group["lr"] for group in optimizer.param_groups] == [0.01, 0.1]


@pytest.mark.parametrize("policy", ["quantum", "mlp"])
def test_training_is_reproducible_and_seed_dependent(policy):
    config = TrainConfig("t", policy, seed=1, n_updates=3, episodes_per_update=4, n_layers=2)
    first = train(config)
    second = train(config)
    assert first.episode_returns == second.episode_returns
    for key, value in first.state_dict.items():
        assert torch.equal(value, second.state_dict[key])

    other = train(TrainConfig("t", policy, seed=2, n_updates=3, episodes_per_update=4, n_layers=2))
    assert other.episode_returns != first.episode_returns


def test_training_improves_a_small_mlp():
    """Smoke test: 40 updates should clearly beat a random policy (~22)."""
    result = train(TrainConfig("t", "mlp", seed=0, n_updates=40, learning_rate=0.03))
    assert np.mean(result.episode_returns[-5:]) > 60
