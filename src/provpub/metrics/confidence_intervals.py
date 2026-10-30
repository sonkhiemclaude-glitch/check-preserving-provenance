"""Intervals over seeds."""
from __future__ import annotations

import numpy as np
from scipy import stats


def t_interval(x, level: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a Student-t interval over independent seeds.
    NaNs are dropped; fewer than two values give half-width NaN (n=0) or 0 (n=1)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a[0]), 0.0
    h = stats.t.ppf(0.5 + level / 2, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(h)


def bootstrap_interval(x, level: float = 0.95, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean: (mean, low, high). Offered for real-data runs."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(a.mean()), float(lo), float(hi)
