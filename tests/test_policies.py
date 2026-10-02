import pytest
import torch

from qrl import simulator as sim
from qrl.policies import (
    MLPPolicy,
    QuantumPolicy,
    count_trainable_parameters,
    normalise_observation,
)


def _observations(batch: int = 8) -> torch.Tensor:
    return torch.empty(batch, 4, dtype=sim.REAL).uniform_(-1, 1)


@pytest.mark.parametrize("policy", [QuantumPolicy(n_layers=2), MLPPolicy((10,))])
def test_policies_return_two_logits_per_observation(policy):
    logits = policy(_observations(8))
    assert logits.shape == (8, 2)
    assert torch.isfinite(logits).all()


def test_quantum_expectation_is_a_valid_expectation_value():
    expectation = QuantumPolicy(n_layers=5).expectation(_observations(64))
    assert expectation.abs().max().item() <= 1.0 + 1e-12


def test_quantum_state_stays_normalised():
    state = QuantumPolicy(n_layers=5).final_state(_observations(16))
    norms = sim.probabilities(state).sum(-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)


def test_parameter_budgets():
    """The matched MLP must have at least as many parameters as the circuit."""
    quantum = count_trainable_parameters(QuantumPolicy(n_layers=5))
    assert quantum == 2 * 4 * 6 + 4 * 5 + 1 == 69
    assert count_trainable_parameters(MLPPolicy((10,))) == 72
    assert count_trainable_parameters(MLPPolicy((64, 64))) == 4610


def test_ablation_flags_freeze_the_right_parameters():
    policy = QuantumPolicy(trainable_input_scaling=False, trainable_output_scale=False)
    assert count_trainable_parameters(policy) == 48
    optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.5)
    policy(_observations()).sum(0)[0].backward()
    optimizer.step()
    assert torch.equal(policy.input_scaling, torch.ones_like(policy.input_scaling))
    assert policy.output_scale.item() == 1.0


def test_every_trainable_parameter_receives_gradient():
    """No parameter of the full circuit is dead for a generic input."""
    torch.manual_seed(3)
    policy = QuantumPolicy(n_layers=5)
    policy(_observations(32))[:, 0].sum().backward()
    for name, parameter in policy.named_parameters():
        assert (parameter.grad.abs() > 1e-9).all(), name


def test_normalisation_maps_termination_limits_to_unit_scale():
    limits = torch.tensor([[2.4, 0.0, 0.2095, 0.0]])
    assert torch.allclose(normalise_observation(limits)[0, [0, 2]], torch.ones(2, dtype=sim.REAL))


def test_wrong_feature_count_is_rejected():
    with pytest.raises(ValueError):
        QuantumPolicy(n_layers=1)(torch.zeros(2, 3, dtype=sim.REAL))
