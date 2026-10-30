"""Evaluation metrics.

All functions work on the outcome table returned by ``checks.compare``:
columns family, cid, key, outcome_raw, outcome_san, same, with outcomes
PASS=1, FAIL=0, UNDET=-1 (missing release instances are UNDET).
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..data.schemas import FAIL, PASS, UNDET


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def preservation_by_family(m: pd.DataFrame) -> pd.DataFrame:
    """Share of check instances whose outcome is unchanged, per family.

    Verdict-equivalence holds for a family iff the share is 1.
    Empty families give NaN, not 1.
    """
    g = m.groupby("family")["same"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "same"})
    g["unchanged"] = [_safe_div(s, n) for s, n in zip(g["same"], g["n"])]
    return g.reset_index()


def verdict_transitions(m: pd.DataFrame) -> dict[str, float]:
    """Split of changed outcomes: to UNDET (cannot verify) versus
    wrong verdicts (PASS->FAIL, FAIL->PASS), as shares of all instances."""
    r, s = m["outcome_raw"], m["outcome_san"]
    n = len(m)
    return {
        "n": n,
        "changed": _safe_div((~m["same"].astype(bool)).sum(), n),
        "to_undet": _safe_div(((r != UNDET) & (s == UNDET)).sum(), n),
        "pass_undet": _safe_div(((r == PASS) & (s == UNDET)).sum(), n),
        "fail_undet": _safe_div(((r == FAIL) & (s == UNDET)).sum(), n),
        "pass_fail": _safe_div(((r == PASS) & (s == FAIL)).sum(), n),
        "fail_pass": _safe_div(((r == FAIL) & (s == PASS)).sum(), n),
    }


def runs_of_key(k) -> list[int]:
    """Runs referenced by a check-instance key (run id, or tuple of runs for a replica group)."""
    if isinstance(k, tuple):
        return [int(x) for x in k]
    if k is None or (isinstance(k, float) and math.isnan(k)) or k == -1:
        return []
    return [int(k)]


def detected_runs(m: pd.DataFrame, family: str | None = None) -> set[int]:
    """Runs with at least one FAIL in the release (optionally in one family)."""
    sub = m[m["outcome_san"] == FAIL]
    if family is not None:
        sub = sub[sub["family"] == family]
    out: set[int] = set()
    for k in sub["key"]:
        out.update(runs_of_key(k))
    return out


def detection_rate(m: pd.DataFrame, deviations: Iterable[tuple[int, str]], dev_family: dict[str, str]) -> float:
    """Share of injected deviations whose mapped check family FAILs on the
    deviating run in the release."""
    devs = list(deviations)
    if not devs:
        return float("nan")
    by_fam = {f: detected_runs(m, f) for f in set(dev_family.values())}
    hits = sum(int(r) in by_fam[dev_family[kind]] for r, kind in devs)
    return hits / len(devs)


def false_alarm_rate(m: pd.DataFrame, all_runs: Iterable[int], deviating_runs: Iterable[int]) -> tuple[int, int, float]:
    """Runs without an injected deviation that fail a check on the release where
    the raw outcome passed.
    Returns (false-alarm runs, clean runs, rate)."""
    dev = {int(r) for r in deviating_runs}
    clean = {int(r) for r in all_runs} - dev
    fa: set[int] = set()
    for k in m.loc[(m["outcome_raw"] == PASS) & (m["outcome_san"] == FAIL), "key"]:
        fa.update(x for x in runs_of_key(k) if x in clean)
    return len(fa), len(clean), _safe_div(len(fa), len(clean))


def flip_probability(x: int, theta: int, alpha: float) -> float:
    """Proposition 4: P([x+Z >= theta] != [x >= theta]) for Z ~ DLap(alpha).

    alpha^(x-theta+1)/(1+alpha) if x >= theta, alpha^(theta-x)/(1+alpha) otherwise.
    """
    if not 0 <= alpha < 1:
        raise ValueError("alpha must lie in [0, 1)")
    k = (x - theta + 1) if x >= theta else (theta - x)
    return alpha ** k / (1 + alpha)


def expected_flips(values: Iterable[float], theta: float, alpha: float) -> float:
    """Expected number of flipped comparisons over independent draws."""
    return float(sum(flip_probability(int(v), int(theta), alpha) for v in values))


def value_coincidence_bound(alphas: Iterable[float]) -> float:
    """Proposition 3: upper bound on P(two different partitions publish the same
    noisy pair) = prod_i (1-a_i)(1+a_i^2)/(1+a_i)^3."""
    out = 1.0
    for a in alphas:
        out *= (1 - a) * (1 + a * a) / (1 + a) ** 3
    return out


def timed_estimate(u: float, w: float, s_r: float, delta: float) -> float:
    """Theorem 2 estimator: d_hat = (floor((w-s_r)/delta) - floor((u-s_r)/delta)) * delta."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    return (math.floor((w - s_r) / delta) - math.floor((u - s_r) / delta)) * delta


def timed_outcome_may_change(d: float, limit: float, delta: float) -> bool:
    """Theorem 2: the outcome of [d <= L] can change only if |d - L| < delta."""
    return abs(d - limit) < delta


def per_record_budget(m_p: Iterable[int], eps: float) -> float:
    """Theorem 4(a): budget of a record in input-set scope, sum_v m_{p_v} * eps
    (pass the m_p of every version that contains the record)."""
    return float(sum(m_p)) * eps


def attack_chance_levels(y: np.ndarray) -> dict[str, float]:
    """Majority-class baseline for an attack target vector."""
    y = np.asarray(y)
    if len(y) == 0:
        return {"majority": float("nan")}
    _, c = np.unique(y, return_counts=True)
    return {"majority": float(c.max(axis=0) / len(y))}
