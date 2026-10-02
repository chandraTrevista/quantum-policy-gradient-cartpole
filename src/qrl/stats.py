"""Summary statistics for multi-seed reinforcement-learning experiments.

RL results are noisy and seed counts are small, so point estimates alone are
misleading.  Following Agarwal et al. (2021) the headline numbers are
interquartile means (IQM) across seeds with percentile-bootstrap confidence
intervals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SOLVED_THRESHOLD = 475.0  # CartPole-v1: mean return over 100 consecutive episodes
SOLVED_WINDOW = 100


def interquartile_mean(values: np.ndarray) -> float:
    """Mean of the middle 50% of ``values`` (25% trimmed from each end)."""
    ordered = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = len(ordered)
    if n == 0:
        raise ValueError("cannot take the IQM of an empty array")
    cut = n // 4
    return float(ordered[cut : n - cut].mean())


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = interquartile_mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval, resampling over seeds."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    estimates = np.array([statistic(values[row]) for row in indices])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def learning_curve(episode_returns: list[list[float]]) -> np.ndarray:
    """Mean on-policy return of each update's batch, shape ``(n_updates,)``."""
    return np.asarray(episode_returns, dtype=np.float64).mean(axis=1)


def final_return(episode_returns: list[list[float]], last_episodes: int = 100) -> float:
    """Mean return over the last ``last_episodes`` training episodes."""
    flat = np.asarray(episode_returns, dtype=np.float64).ravel()
    return float(flat[-last_episodes:].mean())


def mean_training_return(episode_returns: list[list[float]]) -> float:
    """Mean return over every training episode (area under the learning curve)."""
    return float(np.mean(episode_returns))


def episodes_to_solve(episode_returns: list[list[float]]) -> float:
    """Episodes until the trailing 100-episode mean first reaches 475.

    Returns ``nan`` if the run never meets the criterion.
    """
    flat = np.asarray(episode_returns, dtype=np.float64).ravel()
    if len(flat) < SOLVED_WINDOW:
        return float("nan")
    window_means = np.convolve(flat, np.ones(SOLVED_WINDOW) / SOLVED_WINDOW, mode="valid")
    solved = np.flatnonzero(window_means >= SOLVED_THRESHOLD)
    return float(solved[0] + SOLVED_WINDOW) if len(solved) else float("nan")
