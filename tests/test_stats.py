import math

import numpy as np
import pytest

from qrl import stats


def test_iqm_drops_the_outer_quartiles():
    values = np.array([0, 1, 2, 3, 4, 5, 6, 1000])
    assert stats.interquartile_mean(values) == pytest.approx(np.mean([2, 3, 4, 5]))


def test_iqm_of_constant_is_constant():
    assert stats.interquartile_mean(np.full(10, 3.5)) == 3.5


def test_iqm_rejects_empty_input():
    with pytest.raises(ValueError):
        stats.interquartile_mean(np.array([]))


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    values = np.random.default_rng(0).normal(100, 10, size=10)
    low, high = stats.bootstrap_ci(values, n_resamples=2000)
    assert low <= stats.interquartile_mean(values) <= high
    assert (low, high) == stats.bootstrap_ci(values, n_resamples=2000)


def test_bootstrap_interval_collapses_for_constant_data():
    assert stats.bootstrap_ci(np.full(10, 500.0), n_resamples=200) == (500.0, 500.0)


def test_curve_and_scalar_summaries():
    history = [[10.0, 20.0], [30.0, 50.0]]
    np.testing.assert_allclose(stats.learning_curve(history), [15.0, 40.0])
    assert stats.final_return(history, last_episodes=2) == 40.0
    assert stats.mean_training_return(history) == 27.5


def test_episodes_to_solve():
    never = [[100.0] * 10 for _ in range(20)]
    assert math.isnan(stats.episodes_to_solve(never))
    # 50 poor episodes, then perfect ones: the trailing-100 mean reaches 475
    # once at most 5 of the last 100 episodes are zeros, i.e. after 145.
    history = [[0.0] * 10 for _ in range(5)] + [[500.0] * 10 for _ in range(20)]
    assert stats.episodes_to_solve(history) == 145
    assert math.isnan(stats.episodes_to_solve([[500.0] * 10]))
