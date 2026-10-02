"""Policies: a data re-uploading quantum circuit and a classical MLP.

Both policies map a *normalised* CartPole observation (see
:func:`normalise_observation`) to two action logits, so the training loop
treats them identically.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from qrl import simulator as sim

N_ACTIONS = 2
OBS_DIM = 4

# CartPole terminates at |x| > 2.4 and |angle| > 0.2095 rad, so the first and
# third features are divided by those limits.  The two velocities are unbounded;
# 2.5 is the scale used for them in the TensorFlow Quantum implementation of
# the agents of Jerbi et al. (2021) and keeps typical values inside [-1, 1].
OBSERVATION_SCALE = torch.tensor([2.4, 2.5, 0.2095, 2.5], dtype=sim.REAL)


def normalise_observation(observation: Tensor) -> Tensor:
    """Rescale raw CartPole observations to roughly ``[-1, 1]``."""
    return observation.to(sim.REAL) / OBSERVATION_SCALE


class QuantumPolicy(nn.Module):
    r"""Softmax policy whose only non-linearity is a parametrised circuit.

    The circuit uses one qubit per observation feature and ``n_layers``
    data re-uploading layers::

        |0> --[RY RZ]--*--[RX(lam * s)]-- ... --[RZ RY]-- <Z Z Z Z>
                       |  (CZ ring)
              `-------- repeated n_layers times -------'

    * ``[RY RZ]`` are trainable rotations (``theta``),
    * the CZ ring entangles nearest neighbours,
    * ``RX(lam * s_i)`` encodes feature ``s_i`` on qubit ``i``; ``lam`` is a
      trainable *input scaling* (one per qubit and layer),
    * a final ``[RZ RY]`` layer rotates the measurement basis.

    The gate order is chosen so that no trainable angle is trivially
    redundant (e.g. an RZ acting on ``|0>`` or directly before a Z measurement).

    The circuit output is the parity expectation ``e(s) = <Z^{\otimes n}>`` in
    ``[-1, 1]`` and the policy is ``pi(a | s) = softmax(w * [e, -e])_a``, where
    ``w`` is a trainable inverse temperature.  Without ``w`` the policy can
    never become more deterministic than ``sigmoid(2) ~= 0.88``.

    Parameters
    ----------
    n_layers:
        Number of data re-uploading layers.
    trainable_input_scaling:
        If ``False`` the scalings ``lam`` are frozen at 1 (ablation).
    trainable_output_scale:
        If ``False`` the inverse temperature ``w`` is frozen at its initial
        value (ablation).
    """

    def __init__(
        self,
        n_layers: int = 5,
        trainable_input_scaling: bool = True,
        trainable_output_scale: bool = True,
        initial_output_scale: float = 1.0,
        n_qubits: int = OBS_DIM,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # theta[l, q] = (first angle, second angle) of trainable block l.
        theta = torch.empty(n_layers + 1, n_qubits, 2, dtype=sim.REAL).uniform_(-math.pi, math.pi)
        self.theta = nn.Parameter(theta)
        self.input_scaling = nn.Parameter(
            torch.ones(n_layers, n_qubits, dtype=sim.REAL),
            requires_grad=trainable_input_scaling,
        )
        self.output_scale = nn.Parameter(
            torch.tensor(initial_output_scale, dtype=sim.REAL),
            requires_grad=trainable_output_scale,
        )
        self.register_buffer("_cz_ring", sim.cz_ring_diagonal(n_qubits), persistent=False)

    # -- circuit ------------------------------------------------------------

    def final_state(self, observation: Tensor) -> Tensor:
        """Simulate the circuit and return the state before measurement."""
        if observation.shape[-1] != self.n_qubits:
            raise ValueError(f"expected {self.n_qubits} features, got {observation.shape[-1]}")
        observation = observation.to(sim.REAL)

        # Build every gate up front (one vectorised call per gate type) so the
        # loop below is just L contractions.  Matrix products act right-to-left:
        # in the repeated block RY is applied first, then RZ.
        trainable = sim.rz(self.theta[:-1, :, 1]) @ sim.ry(self.theta[:-1, :, 0])
        encoding = sim.rx(self.input_scaling * observation[:, None, :])
        final = sim.ry(self.theta[-1, :, 1]) @ sim.rz(self.theta[-1, :, 0])

        state = sim.zero_state(observation.shape[0], self.n_qubits)
        for layer in range(self.n_layers):
            state = sim.apply_single_qubit_layer(state, trainable[layer])
            state = sim.apply_diagonal(state, self._cz_ring)
            state = sim.apply_single_qubit_layer(state, encoding[:, layer])
        return sim.apply_single_qubit_layer(state, final)

    def expectation(
        self,
        observation: Tensor,
        shots: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Parity expectation, exact (``shots=None``) or estimated from samples."""
        state = self.final_state(observation)
        if shots is None:
            return sim.parity_expectation(state)
        return sim.sampled_parity_expectation(state, shots, generator)

    # -- policy -------------------------------------------------------------

    def logits_from_expectation(self, expectation: Tensor) -> Tensor:
        return self.output_scale * torch.stack([expectation, -expectation], dim=-1)

    def forward(self, observation: Tensor) -> Tensor:
        """Action logits, shape ``(batch, 2)``."""
        return self.logits_from_expectation(self.expectation(observation))


class MLPPolicy(nn.Module):
    """Fully connected tanh network producing two action logits."""

    def __init__(self, hidden_sizes: tuple[int, ...] = (10,)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = OBS_DIM
        for hidden in hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, N_ACTIONS))
        self.network = nn.Sequential(*layers).to(sim.REAL)

    def forward(self, observation: Tensor) -> Tensor:
        return self.network(observation.to(sim.REAL))


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
