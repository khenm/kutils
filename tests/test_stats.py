"""Tests for lab_utils.stats: summarize, bootstrap_ci, permutation_test."""

import numpy as np
import pytest

from lab_utils.stats import bootstrap_ci, permutation_test, summarize


def test_summarize_basic():
    s = summarize([1.0, 3.0])
    assert s.n == 2
    assert s.mean == 2.0
    assert s.std == pytest.approx(np.sqrt(2.0), abs=1e-6)  # sample std, ddof=1
    assert s.min == 1.0
    assert s.max == 3.0


def test_summarize_single_value_has_no_spread():
    s = summarize([0.5])
    assert s.n == 1
    assert s.mean == 0.5
    assert s.std == 0.0


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_bootstrap_ci_covers_mean():
    values = [0.90, 0.85, 0.88, 0.91, 0.87]
    ci = bootstrap_ci(values, seed=1)
    assert ci.lo <= np.mean(values) <= ci.hi
    assert ci.level == 0.95


def test_bootstrap_ci_seeded_reproducible():
    values = [0.90, 0.85, 0.88, 0.91, 0.87]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_permutation_test_identical_groups_p_is_one():
    values = [0.90, 0.85, 0.88, 0.91, 0.87]
    result = permutation_test(values, values)
    assert result.observed_diff == 0.0
    assert result.p_value == 1.0


def test_permutation_test_detects_separation():
    a = [0.90, 0.88, 0.91, 0.89, 0.90]
    b = [0.83, 0.81, 0.84, 0.82, 0.83]
    result = permutation_test(a, b)
    assert result.observed_diff > 0
    assert result.p_value < 0.05
    # 5+5 seeds: C(10,5)=252 distinct relabelings -> exact, no Monte Carlo noise.
    assert result.exact is True


def test_permutation_test_small_samples_are_exact():
    a = [0.9, 0.8]
    b = [0.7, 0.6]
    result = permutation_test(a, b)
    assert result.exact is True
    assert result.n_permutations == 6  # C(4,2)


def test_permutation_test_type1_error_calibrated():
    """Under the null (identical distributions), rejections ~ alpha.

    Verifies our own test's false-positive rate rather than trusting any
    external implementation.
    """
    rng = np.random.default_rng(0)
    rejections = 0
    trials = 100
    for i in range(trials):
        a = rng.normal(0, 1, 8)
        b = rng.normal(0, 1, 8)
        if permutation_test(a, b, n_permutations=2000, seed=i).p_value < 0.05:
            rejections += 1
    # Deterministic given the seeded trials; allow generous slack over 5%.
    assert rejections <= 15, f"{rejections}/100 false rejections at alpha=0.05"
