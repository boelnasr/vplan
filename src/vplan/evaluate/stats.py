"""Interval statistics for the evaluation report (spec §14): Wilson score intervals for proportions, and a paired
percentile bootstrap over tasks for the difference between two configurations on the same tasks."""
from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import NamedTuple

import numpy as np

Z95 = NormalDist().inv_cdf(0.975)  # two-sided 95% normal quantile, 1.959964
PERCENTILES = (2.5, 97.5)  # the 95% percentile-bootstrap interval
_BOOT_CHUNK = 1000  # resamples drawn per block, so memory stays at about 1000 x n integers


class PairedDifference(NamedTuple):
    mean: float  # observed mean of (a - b) over the paired tasks
    low: float  # 2.5th percentile of the bootstrap means
    high: float  # 97.5th percentile of the bootstrap means
    n: int  # paired tasks


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer count, got {type(value).__name__}")
    return int(value)


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials.

    n = 0 carries no information, so the interval is the whole unit interval (0.0, 1.0). The bounds at k = 0 and
    k = n are exactly 0 and 1 (the formula gives them up to rounding)."""
    k, n = _count("k", k), _count("n", n)
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    p, z2 = k / n, Z95 * Z95
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return (0.0 if k == 0 else center - half, 1.0 if k == n else center + half)


def _binary(name: str, values: Sequence[int | bool]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.dtype.kind not in "biuf" or not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must be a flat sequence of 0/1 outcomes")
    return arr.astype(np.int64)


def paired_bootstrap(a: Sequence[int | bool], b: Sequence[int | bool], n_boot: int = 10_000,
                     seed: int = 0) -> PairedDifference:
    """Mean paired difference mean(a - b) over tasks, with a 95% percentile-bootstrap interval.

    a[i] and b[i] are the 0/1 outcomes of two configurations on task i. Tasks are resampled with replacement n_boot
    times; spec §14 has one scene and one attempt per configuration per task, so tasks are the resampling unit and no
    further clustering is needed. Deterministic for a given seed (numpy PCG64)."""
    x, y = _binary("a", a), _binary("b", b)
    if len(x) != len(y):
        raise ValueError(f"a and b must pair the same tasks: {len(x)} != {len(y)} outcomes")
    if len(x) == 0:
        raise ValueError("the paired bootstrap needs at least one task")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    d, n = x - y, len(x)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, _BOOT_CHUNK):
        stop = min(n_boot, start + _BOOT_CHUNK)
        means[start:stop] = d[rng.integers(0, n, size=(stop - start, n))].mean(axis=1)
    low, high = np.percentile(means, PERCENTILES)
    return PairedDifference(float(d.mean()), float(low), float(high), n)
