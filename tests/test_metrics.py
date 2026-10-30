"""Metric formulas on controlled examples."""
import math

import numpy as np
import pandas as pd
import pytest

from provpub.data.schemas import FAIL, PASS, UNDET
from provpub.metrics import (
    detection_rate,
    expected_flips,
    false_alarm_rate,
    flip_probability,
    per_record_budget,
    preservation_by_family,
    t_interval,
    timed_estimate,
    timed_outcome_may_change,
    value_coincidence_bound,
    verdict_transitions,
)


def _m(rows):
    df = pd.DataFrame(rows, columns=["family", "cid", "key", "outcome_raw", "outcome_san"])
    df["same"] = df["outcome_raw"] == df["outcome_san"]
    return df


M = _m([("conf", "c", 1, PASS, PASS), ("conf", "c", 2, PASS, FAIL), ("conf", "c", 3, FAIL, UNDET),
        ("thr", "t", 1, PASS, UNDET), ("thr", "t", 4, FAIL, PASS), ("rep_inputs", "agree", (2, 3), PASS, FAIL)])


def test_preservation_by_family():
    p = preservation_by_family(M).set_index("family")["unchanged"]
    assert p["conf"] == pytest.approx(1 / 3) and p["thr"] == 0.0 and p["rep_inputs"] == 0.0


def test_verdict_transitions():
    t = verdict_transitions(M)
    assert t["n"] == 6
    assert t["changed"] == pytest.approx(5 / 6)
    assert t["pass_undet"] == pytest.approx(1 / 6) and t["fail_undet"] == pytest.approx(1 / 6)
    assert t["pass_fail"] == pytest.approx(2 / 6) and t["fail_pass"] == pytest.approx(1 / 6)
    empty = verdict_transitions(M.iloc[0:0])
    assert math.isnan(empty["changed"])


def test_detection_and_false_alarms():
    devs = [(2, "order"), (4, "model")]
    fam = {"order": "conf", "model": "vocab"}
    assert detection_rate(M, devs, fam) == 0.5
    fa, clean, rate = false_alarm_rate(M, [1, 2, 3, 4, 5], [2])
    assert (fa, clean) == (1, 4) and rate == 0.25          # run 3 via the replica group (2,3)
    assert math.isnan(detection_rate(M, [], fam))


def test_flip_probability_against_monte_carlo():
    alpha = 0.8
    rng = np.random.default_rng(0)
    g1 = rng.geometric(1 - alpha, 200000) - 1
    g2 = rng.geometric(1 - alpha, 200000) - 1
    z = g1 - g2
    for x, th in ((5, 3), (2, 3), (3, 3)):
        mc = np.mean((x + z >= th) != (x >= th))
        assert mc == pytest.approx(flip_probability(x, th, alpha), abs=0.005)
    assert expected_flips([5, 2], 3, alpha) == pytest.approx(flip_probability(5, 3, alpha) + flip_probability(2, 3, alpha))
    with pytest.raises(ValueError):
        flip_probability(1, 1, 1.0)


def test_value_coincidence_bound_against_monte_carlo():
    alpha = 0.7
    rng = np.random.default_rng(1)
    n = 400000
    z1 = (rng.geometric(1 - alpha, n) - 1) - (rng.geometric(1 - alpha, n) - 1)
    z2 = (rng.geometric(1 - alpha, n) - 1) - (rng.geometric(1 - alpha, n) - 1)
    exact_same_value = np.mean(z1 == z2)        # partitions with equal true value: worst case
    assert exact_same_value <= value_coincidence_bound([alpha]) + 0.003


def test_timed_estimate_bound():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        s_r = rng.uniform(0, 1e5)
        u = s_r + rng.uniform(0, 1e4)
        w = u + rng.uniform(0, 5e3)
        delta = rng.choice([1.0, 60.0, 600.0])
        assert abs(timed_estimate(u, w, s_r, delta) - (w - u)) < delta
    assert not timed_outcome_may_change(1500, 1200, 60) and timed_outcome_may_change(1500, 1200, 600)


def test_budget_and_intervals():
    assert per_record_budget([3, 4, 5], 0.5) == 6.0
    m, h = t_interval([1.0, 2.0, 3.0])
    assert m == 2.0 and h == pytest.approx(4.302652729749464 * 1.0 / math.sqrt(3), rel=1e-6)
    assert t_interval([2.0])[1] == 0.0 and math.isnan(t_interval([])[0])
