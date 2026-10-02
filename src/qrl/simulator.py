"""A small, batched, differentiable state-vector simulator.

The policies in this project use at most a handful of qubits, so an exact
state-vector simulation is cheap.  Writing it directly in PyTorch (instead of
calling a quantum SDK inside the training loop) has two advantages:

* every environment state in a batch is simulated in a single tensor
  operation, which is what makes multi-seed experiments feasible on a CPU;
* gradients come from reverse-mode autodiff, which is exact for a noiseless
  simulation and far cheaper than the parameter-shift rule.

Conventions
-----------
A batch of ``n``-qubit states is stored as a complex tensor of shape
``(batch, 2, ..., 2)`` with one axis per qubit.  Qubit 0 is the first qubit
axis, i.e. the most significant bit of a computational-basis label.  This is
the same ordering PennyLane uses, which the test-suite relies on when it
cross-checks this module against ``default.qubit``.
"""

from __future__ import annotations

import torch
from torch import Tensor

COMPLEX = torch.complex128
REAL = torch.float64


def zero_state(batch_size: int, n_qubits: int) -> Tensor:
    """Return ``batch_size`` copies of ``|0...0>``."""
    state = torch.zeros((batch_size,) + (2,) * n_qubits, dtype=COMPLEX)
    state[(slice(None),) + (0,) * n_qubits] = 1.0
    return state


# --------------------------------------------------------------------------
# Single-qubit rotations.  Each takes angles of arbitrary shape ``S`` and
# returns matrices of shape ``S + (2, 2)``.
# --------------------------------------------------------------------------


def rx(angle: Tensor) -> Tensor:
    """``exp(-i * angle / 2 * X)``."""
    c = torch.cos(angle / 2).to(COMPLEX)
    s = -1j * torch.sin(angle / 2).to(COMPLEX)
    return torch.stack([torch.stack([c, s], -1), torch.stack([s, c], -1)], -2)


def ry(angle: Tensor) -> Tensor:
    """``exp(-i * angle / 2 * Y)``."""
    c = torch.cos(angle / 2).to(COMPLEX)
    s = torch.sin(angle / 2).to(COMPLEX)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def rz(angle: Tensor) -> Tensor:
    """``exp(-i * angle / 2 * Z)``."""
    phase = torch.exp(-0.5j * angle.to(COMPLEX))
    zero = torch.zeros_like(phase)
    return torch.stack([torch.stack([phase, zero], -1), torch.stack([zero, phase.conj()], -1)], -2)


# --------------------------------------------------------------------------
# Applying gates.
# --------------------------------------------------------------------------


def _kronecker_product(gates: Tensor) -> Tensor:
    """Kronecker product of ``k`` single-qubit gates along axis ``-3``.

    ``(..., k, 2, 2) -> (..., 2**k, 2**k)``, with ``gates[..., 0, :, :]`` acting
    on the most significant qubit.
    """
    product = gates[..., 0, :, :]
    for q in range(1, gates.shape[-3]):
        gate = gates[..., q, :, :]
        size = 2 * product.shape[-1]
        product = product[..., :, None, :, None] * gate[..., None, :, None, :]
        product = product.reshape(*product.shape[:-4], size, size)
    return product


def apply_single_qubit_layer(state: Tensor, gates: Tensor) -> Tensor:
    """Apply one single-qubit gate to every qubit.

    Parameters
    ----------
    state:
        Tensor of shape ``(batch, 2, ..., 2)``.
    gates:
        Either ``(n_qubits, 2, 2)`` (the same gates for the whole batch, e.g.
        trainable rotations) or ``(batch, n_qubits, 2, 2)`` (gates that depend
        on the batch element, e.g. data-encoding rotations).  ``gates[..., q,
        :, :]`` acts on qubit ``q``.

    Notes
    -----
    The qubits are split into a "row" half and a "column" half, so the state
    becomes a matrix ``S`` and the layer ``A (x) B`` acts as ``A S B^T``.  Two
    small matrix products are much cheaper than either a ``2^n x 2^n`` matrix
    or ``n`` separate tensor contractions.
    """
    n_qubits = state.dim() - 1
    if gates.shape[-3:] != (n_qubits, 2, 2):
        raise ValueError(f"expected gates for {n_qubits} qubits, got {tuple(gates.shape)}")
    if n_qubits == 1:
        return (gates[..., 0, :, :] @ state.unsqueeze(-1)).squeeze(-1)

    half = n_qubits // 2
    rows = _kronecker_product(gates[..., :half, :, :])
    columns = _kronecker_product(gates[..., half:, :, :])
    matrix = state.reshape(-1, 2**half, 2 ** (n_qubits - half))
    return (rows @ matrix @ columns.transpose(-1, -2)).reshape(state.shape)


def apply_gate(state: Tensor, gate: Tensor, qubit: int) -> Tensor:
    """Apply a single ``(2, 2)`` gate to one qubit (used mainly in tests)."""
    n_qubits = state.dim() - 1
    identity = torch.eye(2, dtype=COMPLEX)
    gates = torch.stack([gate if q == qubit else identity for q in range(n_qubits)])
    return apply_single_qubit_layer(state, gates)


def cz_ring_diagonal(n_qubits: int) -> Tensor:
    """Diagonal of a ring of CZ gates, shaped ``(2,) * n_qubits``.

    CZ gates are diagonal and commute with each other, so the whole
    entangling layer is a single vector of ``+1/-1`` phases: the amplitude of a
    basis state is multiplied by ``-1`` once for every neighbouring pair of
    qubits that are both in ``|1>``.  For two qubits the "ring" is a single CZ.
    """
    if n_qubits < 2:
        raise ValueError("an entangling layer needs at least two qubits")
    pairs = [(0, 1)] if n_qubits == 2 else [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    labels = torch.arange(2**n_qubits)
    bits = torch.stack([(labels >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)])
    n_active_pairs = sum(bits[a] * bits[b] for a, b in pairs)
    signs = 1.0 - 2.0 * (n_active_pairs % 2).to(REAL)
    return signs.reshape((2,) * n_qubits).to(COMPLEX)


def apply_diagonal(state: Tensor, diagonal: Tensor) -> Tensor:
    """Apply a diagonal unitary given as a ``(2,) * n_qubits`` tensor."""
    return state * diagonal


# --------------------------------------------------------------------------
# Measurement.
# --------------------------------------------------------------------------


def probabilities(state: Tensor) -> Tensor:
    """Born-rule probabilities, shape ``(batch, 2**n_qubits)``."""
    return state.flatten(1).abs().square()


def parity_signs(n_qubits: int) -> Tensor:
    """Eigenvalues of ``Z x Z x ... x Z`` in computational-basis order."""
    labels = torch.arange(2**n_qubits)
    ones = sum((labels >> q) & 1 for q in range(n_qubits))
    return 1.0 - 2.0 * (ones % 2).to(REAL)


def parity_expectation(state: Tensor) -> Tensor:
    """Exact ``<Z x Z x ... x Z>`` for each state in the batch."""
    n_qubits = state.dim() - 1
    return probabilities(state) @ parity_signs(n_qubits)


def sampled_parity_expectation(
    state: Tensor, shots: int, generator: torch.Generator | None = None
) -> Tensor:
    """Estimate ``<Z x ... x Z>`` from ``shots`` computational-basis samples.

    This is what a quantum device would actually return: the estimate is the
    mean of ``shots`` outcomes in ``{-1, +1}``, so its standard error is
    ``sqrt((1 - <O>^2) / shots)``.
    """
    if shots < 1:
        raise ValueError("shots must be a positive integer")
    n_qubits = state.dim() - 1
    probs = probabilities(state)
    probs = probs / probs.sum(-1, keepdim=True)  # guard against round-off
    outcomes = torch.multinomial(probs, shots, replacement=True, generator=generator)
    return parity_signs(n_qubits)[outcomes].mean(-1)
