#!/usr/bin/env python3
"""
Statistics for evaluations with several runs per plant.

Two sources of variation are mixed here. Plants differ from each other (that is the population the
paper reports on), and runs of the same plant differ because a planner can sample viewpoints at
random. A hierarchical bootstrap resamples both: first the plants, then the runs inside each plant.

Curves (a metric per viewpoint) get a *simultaneous* confidence band instead of a pointwise one.
The viewpoints are strongly correlated because the metrics are cumulative, so a pointwise band says
nothing about the curve as a whole. A simultaneous band holds for all viewpoints at once, which is
what you need to compare methods over the whole curve.

Author : Thomas Versmissen <thomas.versmissen@wur.nl>
License: MIT - see LICENSE for full text
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

EPS = 1e-12


def pad_curves(curves: Sequence[np.ndarray], length: int) -> np.ndarray:
    """Stack curves of different lengths into (n, length), repeating the last value.

    Runs that stopped early (the planner reached the top of the plant) keep their final value,
    the same carry forward that the figures already used over plants.
    """
    out = np.full((len(curves), length), np.nan)
    for i, c in enumerate(curves):
        c = np.asarray(c, dtype=float)
        k = min(len(c), length)
        out[i, :k] = c[:k]
        if k and k < length:
            out[i, k:] = c[k - 1]
    return out


def bootstrap(values: List[np.ndarray], statistic: Callable[[np.ndarray], np.ndarray],
              n_boot: int = 2000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Hierarchical bootstrap over plants and runs.

    values: one array per plant, shape (runs, ...) - the runs of that plant
    statistic: maps an array (plants, ...) of per plant means to the reported number(s)

    Returns the estimate on the real data and the bootstrap replicates (n_boot, ...).
    """
    rng = np.random.default_rng(seed)
    per_plant = np.array([np.nanmean(v, axis=0) for v in values])
    estimate = statistic(per_plant)

    n_plants = len(values)
    replicates = np.empty((n_boot,) + np.shape(estimate))
    for b in range(n_boot):
        plant_idx = rng.integers(0, n_plants, n_plants)
        means = []
        for p in plant_idx:
            runs = values[p]
            run_idx = rng.integers(0, len(runs), len(runs))
            means.append(np.nanmean(np.asarray(runs)[run_idx], axis=0))
        replicates[b] = statistic(np.array(means))
    return estimate, replicates


def pointwise_interval(replicates: np.ndarray, ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile interval per point. Valid per viewpoint, not for a whole curve."""
    alpha = (1.0 - ci) / 2.0
    return (np.percentile(replicates, 100 * alpha, axis=0),
            np.percentile(replicates, 100 * (1 - alpha), axis=0))


def supt_band(estimate: np.ndarray, replicates: np.ndarray, ci: float = 0.95
              ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simultaneous band, sup-t method (Montiel Olea and Plagborg-Moller, 2019).

    Scale every viewpoint by its own bootstrap spread, then take the width that keeps 95% of the
    bootstrap curves inside over all viewpoints at once. Returns (low, high, factor); the factor is
    about 1.96 for a single point and grows with the number of correlated viewpoints.
    """
    sigma = replicates.std(axis=0, ddof=1)
    safe = np.where(sigma > EPS, sigma, 1.0)
    t = np.max(np.abs(replicates - estimate) / safe, axis=tuple(range(1, replicates.ndim)))
    factor = float(np.percentile(t, 100 * ci))
    return estimate - factor * sigma, estimate + factor * sigma, factor


def chi2_band(estimate: np.ndarray, replicates: np.ndarray, ci: float = 0.95
              ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simultaneous band from the classical confidence ellipsoid (Scheffe).

    Projects the ellipsoid (x - estimate)' S^-1 (x - estimate) <= chi2(dim, ci) on every viewpoint.
    Assumes the bootstrap distribution is roughly normal and is wider than sup-t.
    """
    sigma = replicates.std(axis=0, ddof=1)
    dim = int(np.prod(np.shape(estimate)))
    factor = float(np.sqrt(chi2.ppf(ci, dim)))
    return estimate - factor * sigma, estimate + factor * sigma, factor


def quantile_band(estimate: np.ndarray, replicates: np.ndarray, ci: float = 0.95
                  ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simultaneous band from bootstrap percentiles (Montiel Olea and Plagborg-Moller, 2019, algorithm 2).

    Starts from the equal tailed percentile interval at every point and widens all of them by the
    same tail level until a fraction ci of the bootstrap curves lies inside at every point. Unlike
    sup-t the band can be asymmetric and never leaves the range of the bootstrap values, which
    matters for metrics close to 0 or 1. Returns (low, high, tail), tail the percentile level used.
    """
    reps = replicates.reshape(len(replicates), -1)
    alpha = 1.0 - ci

    def limits(tail):
        return (np.percentile(reps, 100 * tail, axis=0),
                np.percentile(reps, 100 * (1 - tail), axis=0))

    def covered(tail):
        lo, hi = limits(tail)
        return np.mean(np.all((reps >= lo) & (reps <= hi), axis=1))

    # the narrowest candidate is the pointwise interval; the search goes down to tail 0 (the full
    # range of the draws) and not only to Bonferroni, because with a finite number of draws even
    # the Bonferroni percentiles can leave slightly more than 1 - ci of the curves outside
    small, large = 0.0, alpha / 2
    if covered(large) >= ci:
        tail = large
    else:
        # a larger tail level gives a narrower band that covers fewer curves, so bisect
        for _ in range(40):
            mid = (small + large) / 2
            if covered(mid) >= ci:
                small = mid
            else:
                large = mid
        tail = small
    lo, hi = limits(tail)
    return lo.reshape(np.shape(estimate)), hi.reshape(np.shape(estimate)), float(tail)


def band(estimate: np.ndarray, replicates: np.ndarray, kind: str = "quantile", ci: float = 0.95
         ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Confidence band of the chosen kind: quantile, supt, chi2 or pointwise."""
    if kind == "quantile":
        return quantile_band(estimate, replicates, ci)
    if kind == "supt":
        return supt_band(estimate, replicates, ci)
    if kind == "chi2":
        return chi2_band(estimate, replicates, ci)
    if kind == "pointwise":
        low, high = pointwise_interval(replicates, ci)
        return low, high, None
    raise ValueError(f"unknown band kind: {kind}")


def median_ci(values: List[np.ndarray], n_boot: int = 2000, seed: int = 0, ci: float = 0.95
              ) -> Tuple[float, float, float]:
    """Median over plants of the per plant mean, with a bootstrap interval.

    values: one array of run values per plant.
    """
    estimate, replicates = bootstrap(values, lambda x: np.median(x, axis=0), n_boot, seed)
    low, high = pointwise_interval(replicates, ci)
    return float(estimate), float(low), float(high)


def mean_run_sd(values: List[np.ndarray]) -> float:
    """Mean over plants of the standard deviation between the runs of one plant.

    Says how much a single run of one plant can differ from the average of that plant.
    """
    sds = [np.std(np.asarray(v), ddof=1) for v in values if len(np.asarray(v)) > 1]
    return float(np.mean(sds)) if sds else 0.0


def paired_difference(values_a: List[np.ndarray], values_b: List[np.ndarray],
                      n_boot: int = 2000, seed: int = 0, ci: float = 0.95
                      ) -> Tuple[float, float, float, int, int]:
    """Compare two methods on the same plants.

    values_a[i] and values_b[i] are the runs of the same plant i. Per plant the difference of the
    run means is taken, and the median of those differences is reported with a bootstrap interval
    (plants resampled, runs resampled inside each plant, both methods independently).

    Pairing removes the plant to plant variation, which is the largest source of spread, so this is
    much more sensitive than comparing two separate medians. Returns
    (median difference, low, high, number of plants where a is better, number where b is better).
    """
    diffs = np.array([np.mean(a) - np.mean(b) for a, b in zip(values_a, values_b)])
    estimate = float(np.median(diffs))
    rng = np.random.default_rng(seed)
    n = len(values_a)
    replicates = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.integers(0, n, n)
        replicates[k] = np.median([
            np.mean(rng.choice(values_a[i], len(values_a[i]))) -
            np.mean(rng.choice(values_b[i], len(values_b[i]))) for i in pick])
    low, high = pointwise_interval(replicates, ci)
    return estimate, float(low), float(high), int((diffs > 0).sum()), int((diffs < 0).sum())
