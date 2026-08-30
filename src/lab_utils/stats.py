"""Run-level statistics: spread summaries, bootstrap CIs, permutation tests.

Thin wrappers around scipy (`bootstrap`, `permutation_test`) so the math is
maintained upstream. scipy automatically enumerates every relabeling when the
count is small (exact test, no Monte Carlo noise), matching our old
hand-written behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import stats

DEFAULT_SEED = 42


@dataclass
class Summary:
    """Spread statistics over a sample of per-seed values."""

    n: int
    mean: float
    std: float
    min: float
    max: float


def summarize(values: Sequence[float]) -> Summary:
    """Mean and sample std (ddof=1); std is 0.0 for a single value."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("summarize() needs at least one value")
    return Summary(
        n=arr.size,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
        min=float(arr.min()),
        max=float(arr.max()),
    )


@dataclass
class BootstrapCI:
    """Percentile-bootstrap confidence interval for the mean."""

    level: float
    lo: float
    hi: float
    n_boot: int
    seed: int


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    level: float = 0.95,
    n_boot: int = 1000,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Percentile bootstrap CI of the mean (scipy.stats.bootstrap)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci() needs at least one value")
    result = stats.bootstrap(
        (arr,),
        np.mean,
        confidence_level=level,
        n_resamples=n_boot,
        method="percentile",
        rng=np.random.default_rng(seed),
    )
    ci = result.confidence_interval
    return BootstrapCI(level=level, lo=float(ci.low), hi=float(ci.high), n_boot=n_boot, seed=seed)


@dataclass
class PermutationResult:
    """Result of a two-condition permutation test on the difference of means."""

    observed_diff: float
    p_value: float
    n_permutations: int
    exact: bool
    seed: int


def _diff_of_means(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(x) - np.mean(y))


def permutation_test(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
    *,
    n_permutations: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> PermutationResult:
    """Is the A-vs-B gap distinguishable from seed noise?

    Wraps `scipy.stats.permutation_test` (two-sided, independent groups),
    which enumerates every relabeling when the count is <= `n_permutations`
    (exact) and resamples otherwise.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("permutation_test() needs values from both conditions")

    result = stats.permutation_test(
        (a, b),
        _diff_of_means,
        n_resamples=n_permutations,
        alternative="two-sided",
        rng=np.random.default_rng(seed),
    )
    return PermutationResult(
        observed_diff=float(result.statistic),
        p_value=float(result.pvalue),
        n_permutations=len(result.null_distribution),
        exact=comb(a.size + b.size, a.size) <= n_permutations,
        seed=seed,
    )
