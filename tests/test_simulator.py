"""The simulator is checked against PennyLane's reference device."""

import math

import numpy as np
import pytest
import torch

from qrl import simulator as sim
from qrl.policies import QuantumPolicy

qml = pytest.importorskip("pennylane")


@pytest.mark.parametrize("builder, reference", [(sim.rx, "RX"), (sim.ry, "RY"), (sim.rz, "RZ")])
def test_rotation_matrices_match_pennylane(builder, reference):
    for angle in (-2.3, 0.0, 0.4, math.pi):
        ours = builder(torch.tensor(angle, dtype=sim.REAL)).numpy()
        theirs = qml.matrix(getattr(qml, reference)(angle, wires=0))
        np.testing.assert_allclose(ours, theirs, atol=1e-12)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
def test_cz_ring_matches_pennylane(n_qubits):
    pairs = [(0, 1)] if n_qubits == 2 else [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    reference = np.eye(2**n_qubits)
    for a, b in pairs:
        reference = qml.matrix(qml.CZ(wires=[a, b]), wire_order=range(n_qubits)) @ reference
    np.testing.assert_allclose(
        sim.cz_ring_diagonal(n_qubits).flatten().numpy(), np.diag(reference), atol=1e-12
    )


def test_zero_state_is_normalised_basis_state():
    state = sim.zero_state(3, 4)
    probs = sim.probabilities(state)
    assert torch.allclose(probs[:, 0], torch.ones(3, dtype=sim.REAL))
    assert torch.allclose(probs.sum(-1), torch.ones(3, dtype=sim.REAL))


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("batched_gates", [False, True])
def test_gate_layer_matches_explicit_kronecker_product(n_qubits, batched_gates):
    torch.manual_seed(n_qubits)
    batch = 3
    state = torch.randn((batch,) + (2,) * n_qubits, dtype=sim.COMPLEX)
    shape = (batch, n_qubits) if batched_gates else (n_qubits,)
    gates = sim.rz(torch.randn(shape, dtype=sim.REAL)) @ sim.rx(torch.randn(shape, dtype=sim.REAL))

    ours = sim.apply_single_qubit_layer(state, gates).flatten(1).numpy()
    for b in range(batch):
        per_qubit = (gates[b] if batched_gates else gates).numpy()
        full = np.eye(1)
        for q in range(n_qubits):
            full = np.kron(full, per_qubit[q])
        np.testing.assert_allclose(ours[b], full @ state[b].flatten().numpy(), atol=1e-12)


def test_apply_gate_targets_the_right_qubit():
    """X on qubit 1 of |000> gives |010>, i.e. basis index 2 (qubit 0 is MSB)."""
    flip = sim.rx(torch.tensor(math.pi, dtype=sim.REAL))
    probs = sim.probabilities(sim.apply_gate(sim.zero_state(1, 3), flip, qubit=1))
    assert probs.argmax().item() == 2
    assert probs.max().item() == pytest.approx(1.0)


def _pennylane_policy_circuit(policy: QuantumPolicy):
    """The circuit of ``QuantumPolicy`` written independently in PennyLane."""
    n = policy.n_qubits
    device = qml.device("default.qubit", wires=n)

    @qml.qnode(device, diff_method="parameter-shift")
    def circuit(theta, scaling, observation):
        for layer in range(policy.n_layers):
            for q in range(n):
                qml.RY(theta[layer, q, 0], wires=q)
                qml.RZ(theta[layer, q, 1], wires=q)
            for q in range(n):
                qml.CZ(wires=[q, (q + 1) % n])
            for q in range(n):
                qml.RX(scaling[layer, q] * observation[q], wires=q)
        for q in range(n):
            qml.RZ(theta[-1, q, 0], wires=q)
            qml.RY(theta[-1, q, 1], wires=q)
        return qml.expval(qml.prod(*[qml.PauliZ(q) for q in range(n)]))

    return circuit


@pytest.mark.parametrize("n_layers", [1, 3, 5])
def test_policy_expectation_matches_pennylane(n_layers):
    torch.manual_seed(n_layers)
    policy = QuantumPolicy(n_layers=n_layers)
    with torch.no_grad():
        policy.input_scaling.uniform_(0.5, 2.0)
    observations = torch.empty(6, 4, dtype=sim.REAL).uniform_(-1.5, 1.5)

    ours = policy.expectation(observations).detach().numpy()
    circuit = _pennylane_policy_circuit(policy)
    theta = policy.theta.detach().numpy()
    scaling = policy.input_scaling.detach().numpy()
    theirs = [float(circuit(theta, scaling, obs)) for obs in observations.numpy()]
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_autograd_gradient_matches_parameter_shift():
    """Backprop through the simulator equals the hardware-compatible gradient."""
    torch.manual_seed(7)
    policy = QuantumPolicy(n_layers=2)
    observation = torch.tensor([[0.3, -0.8, 0.1, 0.6]], dtype=sim.REAL)

    policy.expectation(observation).sum().backward()
    ours = policy.theta.grad.numpy()

    circuit = _pennylane_policy_circuit(policy)
    theta = qml.numpy.array(policy.theta.detach().numpy(), requires_grad=True)
    scaling = qml.numpy.array(policy.input_scaling.detach().numpy(), requires_grad=False)
    obs = qml.numpy.array(observation[0].numpy(), requires_grad=False)
    theirs = qml.grad(circuit)(theta, scaling, obs)  # only theta is trainable
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_sampled_expectation_converges_to_exact_value():
    torch.manual_seed(0)
    policy = QuantumPolicy(n_layers=3)
    observations = torch.empty(4, 4, dtype=sim.REAL).uniform_(-1, 1)
    with torch.no_grad():
        exact = policy.expectation(observations)
        generator = torch.Generator().manual_seed(0)
        shots = 200_000
        estimate = policy.expectation(observations, shots=shots, generator=generator)
    standard_error = torch.sqrt((1 - exact**2) / shots)
    assert torch.all((estimate - exact).abs() < 5 * standard_error + 1e-9)


def test_sampled_expectation_rejects_non_positive_shots():
    with pytest.raises(ValueError):
        sim.sampled_parity_expectation(sim.zero_state(1, 2), shots=0)
