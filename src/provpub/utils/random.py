"""Seeds and keys.

The sanitiser itself draws no random numbers: all its randomness is derived
from the release key by HMAC. Seeds only
control the synthetic workload generators and the attack classifiers.
Experiment keys are derived deterministically from names and seeds exactly as
in the reference runs, so outputs are reproducible; they are NOT secret and must
never be used for a real release.
"""
from __future__ import annotations

import hashlib
import os


def experiment_key(prefix: str, seed: int, length: int = 32) -> bytes:
    """Deterministic, PUBLIC experiment key: (f"{prefix}-{seed}" repeated)[:length].

    Matches the keys used for the reference results, e.g.
    experiment_key('release-key-W2', 3) == (b'release-key-W2-3' * 2)[:32].
    """
    raw = f"{prefix}-{seed}".encode()
    reps = -(-length // len(raw))
    return (raw * reps)[:length]


def new_release_key(nbytes: int = 32) -> bytes:
    """Fresh secret key for a real release (os.urandom). Store it separately
    from the release; it is the additional information under GDPR Art. 4(5)."""
    return os.urandom(nbytes)


def key_fingerprint(key: bytes) -> str:
    """Short public fingerprint for provenance records (never log the key)."""
    return hashlib.sha256(b"fp|" + key).hexdigest()[:16]
