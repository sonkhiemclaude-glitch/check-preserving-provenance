"""Determinism under fixed seeds and keys; shard-parallel identity; key handling."""
import pandas as pd

from provpub.data.synthetic import w2_generate, w2_spec
from provpub.experiments.e4 import canon_hash
from provpub.methods.operators import sanitise
from provpub.utils.random import experiment_key, key_fingerprint

KEY = b"0123456789abcdef0123456789abcdef"


def test_same_seed_same_release():
    spec = w2_spec()
    a = sanitise(w2_generate(60, seed=9), spec, KEY)
    b = sanitise(w2_generate(60, seed=9), spec, KEY)
    assert canon_hash(a) == canon_hash(b)


def test_different_key_different_release():
    spec = w2_spec()
    g = w2_generate(40, seed=9)
    assert canon_hash(sanitise(g, spec, KEY)) != canon_hash(sanitise(g, spec, b"x" * 32))


def test_experiment_keys_match_reference_runs():
    assert experiment_key("release-key-W2", 3) == (b"release-key-W2-3" * 2)[:32]
    assert experiment_key("target-key", 0) == (b"target-key-0" * 3)[:32]
    assert len(key_fingerprint(KEY)) == 16 and key_fingerprint(KEY) not in KEY.hex()


def test_e1_rows_reproducible(tmp_path):
    from provpub.experiments.e1_e3 import main
    kw = dict(seeds=[0], configs=["RAW", "FULL-bucket"], runs={"W1": 15, "W2": 15, "W3": 8})
    main(out_dir=str(tmp_path / "a"), **kw)
    main(out_dir=str(tmp_path / "b"), **kw)
    for f in ("e1_utility.csv", "e3_deviations.csv", "e1_bounds.csv"):
        x = pd.read_csv(tmp_path / "a" / f).drop(columns=["t_sanitise"], errors="ignore")
        y = pd.read_csv(tmp_path / "b" / f).drop(columns=["t_sanitise"], errors="ignore")
        pd.testing.assert_frame_equal(x, y)


def test_e4_parallel_equals_sequential_smoke(tmp_path):
    """E4 worker processes run from a source checkout and produce identical releases."""
    from provpub.experiments.e4 import main
    s = main(sizes=[5000], workers=[1, 2], workloads=["W2"], reps=1, out_dir=str(tmp_path), tag="t")
    df = pd.read_csv(tmp_path / "e4_scaling_t.csv")
    assert s["failed_measurements"] == 0 and len(df) == 2
    assert df["hash"].nunique() == 1
