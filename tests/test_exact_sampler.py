"""Optional exact discrete-Laplace sampler."""
import math
from fractions import Fraction

import numpy as np

from provpub.data.synthetic import w2_generate, w2_spec
from provpub.methods.exact_dlap import PRFStream, discrete_laplace_exact, keyed_exact_noise, rational_scale
from provpub.methods.operators import KeyRing, dpn

KEY = bytes(range(32))


def test_prf_stream_uniform_and_deterministic():
    a = [PRFStream(KEY, "x").randbelow(7) for _ in range(3)]
    s1, s2 = PRFStream(KEY, "lbl"), PRFStream(KEY, "lbl")
    v1 = [s1.randbelow(1000) for _ in range(200)]
    v2 = [s2.randbelow(1000) for _ in range(200)]
    assert v1 == v2 and all(0 <= x < 1000 for x in v1)
    assert len(set(a)) >= 1


def test_exact_sampler_matches_pmf():
    scale = Fraction(3, 2)                       # alpha = exp(-2/3)
    alpha = math.exp(-1 / float(scale))
    st = PRFStream(KEY, "pmf")
    xs = np.array([discrete_laplace_exact(st, scale) for _ in range(20000)])
    for z in (-2, -1, 0, 1, 2):
        p = (1 - alpha) / (1 + alpha) * alpha ** abs(z)
        se = math.sqrt(p * (1 - p) / len(xs))
        assert abs((xs == z).mean() - p) < 5 * se
    assert abs(xs.mean()) < 0.1


def test_keyed_noise_is_memoised_by_label():
    assert keyed_exact_noise(KEY, "a", 1000.0, 0.5) == keyed_exact_noise(KEY, "a", 1000.0, 0.5)
    vals = {keyed_exact_noise(KEY, f"l{i}", 1000.0, 0.5) for i in range(20)}
    assert len(vals) > 10
    assert rational_scale(1000.0, 0.5) == Fraction(2000)


def test_dpn_exact_sampler_keeps_memoisation():
    g = w2_generate(60, seed=2)
    spec = w2_spec()
    h = dpn(g, spec, KeyRing(KEY), 1.0, unit="bucket", sampler="exact")
    src = g.ent["type"] == spec.source_type
    df = h.ent.loc[src, ["part_id", "n_records"]]
    assert df.groupby("part_id")["n_records"].nunique().max() == 1
