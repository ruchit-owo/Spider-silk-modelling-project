"""Tests for the scoring functions.

The within-vector R2 is checked against sklearn rather than against itself:
the paper's number is whatever sklearn.metrics.r2_score returns, so agreeing
with our own algebra would prove nothing.
"""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from silkrepro import metrics


def test_r2_matches_sklearn_on_random_vectors():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = rng.uniform(0, 1, size=8)
        p = rng.uniform(0, 1, size=8)
        assert metrics.r2_within_vector(t, p) == pytest.approx(r2_score(t, p), abs=1e-12)


def test_r2_perfect_prediction_is_one():
    t = [0.25, 0.252, 0.2, 0.151, 0.31, 0.203, 0.293, 0.265]
    assert metrics.r2_within_vector(t, t) == pytest.approx(1.0)


def test_r2_argument_order_matters():
    """Guards against the swap. The denominator is the variance of the first
    argument, so r2(a, b) != r2(b, a) in general. The released notebook's
    validate() passes (predictions, ground truth); generate_new() passes
    (target, predictions). We always use (target, predictions)."""
    t = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p = np.array([0.3, 0.3, 0.35, 0.4, 0.45, 0.5, 0.5, 0.55])
    assert metrics.r2_within_vector(t, p) != pytest.approx(metrics.r2_within_vector(p, t))


def test_r2_degenerate_target():
    t = [0.5] * 8
    assert metrics.r2_within_vector(t, t) == 1.0
    assert metrics.r2_within_vector(t, [0.6] * 8) == 0.0


def test_target_spread_flags_tight_targets():
    tight = [0.25, 0.252, 0.2, 0.251, 0.25, 0.253, 0.25, 0.265]
    wide = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.9]
    assert metrics.target_spread(tight) < metrics.target_spread(wide)


def test_r2_across_dataset_shapes_and_values():
    rng = np.random.default_rng(1)
    y = rng.uniform(size=(50, 3))
    r2 = metrics.r2_across_dataset(y, y)
    assert r2.shape == (3,)
    assert np.allclose(r2, 1.0)

    # Predicting the column mean gives R2 = 0 by construction.
    r2_mean = metrics.r2_across_dataset(y, np.tile(y.mean(axis=0), (50, 1)))
    assert np.allclose(r2_mean, 0.0, atol=1e-12)


def test_best_of_n_is_monotone_and_bounded():
    rng = np.random.default_rng(2)
    scores = rng.normal(0.3, 0.25, size=500)
    curve = metrics.best_of_n_curve(scores, [1, 2, 8, 64, 256], rng=rng)

    means = [curve[n]["mean"] for n in (1, 2, 8, 64, 256)]
    assert all(a <= b + 1e-9 for a, b in zip(means, means[1:])), "best-of-N must not decrease"
    assert curve[1]["mean"] == pytest.approx(scores.mean(), abs=0.05)
    assert curve[256]["mean"] <= scores.max() + 1e-9


def test_n_required_to_reach():
    scores = [0.1] * 99 + [0.9]  # 1% hit rate
    n = metrics.n_required_to_reach(scores, 0.9)
    assert n is not None and 60 < n < 80  # ceil(log .5 / log .99) = 69
    assert metrics.n_required_to_reach(scores, 1.5) is None


def test_parse_stats_rate():
    s = metrics.ParseStats(attempted=10, parsed=7, rejected_unparseable=3)
    assert s.parse_rate == pytest.approx(0.7)
    assert s.to_dict()["parse_rate"] == pytest.approx(0.7)
