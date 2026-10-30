"""Exact discrete Laplace sampler driven by a keyed PRF.

This is the default sampler for two-sided geometric noise. The historical
double-precision inversion sampler remains available explicitly for reproducing
older result files.

Randomness is not drawn from an RNG but from HMAC-SHA256 in counter mode
under a label-specific key, so the result is still a deterministic function
of (release key, memo label): memoisation and shard-parallel processing keep
working.

The scale ``sens / eps_attr`` is represented as a rational number t/s with
``Fraction.limit_denominator(max_den)``; the sampled distribution is exactly
DLap(t/s). For the epsilon and sensitivity values used in the paper, the
requested scales are represented exactly by these rationals. Other parameter
values may be approximated; the resulting effective privacy parameter should
then be accounted for using the actual rational scale.
"""
from __future__ import annotations

import hmac
from fractions import Fraction


class PRFStream:
    """Deterministic stream of uniform integers from HMAC-SHA256(key, label|counter)."""

    def __init__(self, key: bytes, label: str):
        self._key = hmac.digest(key, f"dlap-exact|{label}".encode(), "sha256")
        self._ctr = 0
        self._buf = b""

    def _bytes(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._buf += hmac.digest(self._key, self._ctr.to_bytes(8, "big"), "sha256")
            self._ctr += 1
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def randbelow(self, n: int) -> int:
        """Uniform integer in [0, n) by rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("n must be positive")
        if n == 1:
            return 0
        k = (n - 1).bit_length()
        nbytes = (k + 7) // 8
        mask = (1 << k) - 1
        while True:
            x = int.from_bytes(self._bytes(nbytes), "big") & mask
            if x < n:
                return x


def bernoulli_rational(stream: PRFStream, num: int, den: int) -> int:
    """Bernoulli(num/den) for 0 <= num <= den."""
    return 1 if stream.randbelow(den) < num else 0


def bernoulli_exp_neg(stream: PRFStream, gamma: Fraction) -> int:
    """Bernoulli(exp(-gamma)) for rational gamma >= 0 (CKS20 Algorithm 1)."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    while gamma > 1:
        if bernoulli_exp_neg(stream, Fraction(1)) == 0:
            return 0
        gamma -= 1
    k = 1
    while True:
        g = gamma / k
        if bernoulli_rational(stream, g.numerator, g.denominator) == 0:
            break
        k += 1
    return k % 2


def discrete_laplace_exact(stream: PRFStream, scale: Fraction) -> int:
    """Sample Z with P(Z=z) proportional to exp(-|z|/scale) (CKS20 Algorithm 2).

    scale = t/s with integers t, s >= 1.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    t, s = scale.numerator, scale.denominator
    while True:
        u = stream.randbelow(t)
        if bernoulli_exp_neg(stream, Fraction(u, t)) == 0:
            continue
        v = 0
        while bernoulli_exp_neg(stream, Fraction(1)) == 1:
            v += 1
        x = u + t * v
        y = x // s
        b = bernoulli_rational(stream, 1, 2)
        if b == 1 and y == 0:
            continue
        return (1 - 2 * b) * y


def rational_scale(sens: float, eps_attr: float, max_den: int = 10**6) -> Fraction:
    """Rational approximation of the noise scale sens/eps_attr (alpha = exp(-1/scale))."""
    if sens <= 0 or eps_attr <= 0:
        raise ValueError("sensitivity and epsilon must be positive")
    return Fraction(sens / eps_attr).limit_denominator(max_den)


def keyed_exact_noise(key: bytes, label: str, sens: float, eps_attr: float) -> int:
    """Deterministic exact discrete-Laplace noise for one memo label."""
    return discrete_laplace_exact(PRFStream(key, label), rational_scale(sens, eps_attr))


__all__ = ["PRFStream", "bernoulli_exp_neg", "discrete_laplace_exact", "keyed_exact_noise", "rational_scale"]
