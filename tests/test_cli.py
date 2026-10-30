"""CLI smoke tests on synthetic data."""
import json
import os
import subprocess
import sys

import pandas as pd

from provpub.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_help_runs():
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run([sys.executable, "-m", "provpub.cli", "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "sanitize" in r.stdout


def test_generate_sanitize_check(tmp_path):
    raw, spec, rel, rep, out = (str(tmp_path / x) for x in ("raw.json", "spec.yaml", "rel.json", "rep.json", "o.csv"))
    key = tmp_path / "key.bin"
    key.write_bytes(b"k" * 32)
    assert main(["generate", "--workload", "W3", "--runs", "15", "--out", raw, "--spec-out", spec]) == 0
    assert main(["sanitize", "--input", raw, "--spec", spec, "--out", rel, "--key-file", str(key),
                 "--report", rep]) == 0
    r = json.load(open(rep))
    assert r["fail_pass"] == 0.0 and r["to_undet"] == 0.0
    assert main(["check", "--input", rel, "--spec", spec, "--out", out]) == 0
    assert len(pd.read_csv(out)) > 0


def test_sanitize_requires_key(tmp_path):
    raw, spec = str(tmp_path / "raw.json"), str(tmp_path / "s.yaml")
    main(["generate", "--workload", "W1", "--runs", "5", "--out", raw, "--spec-out", spec])
    assert main(["sanitize", "--input", raw, "--spec", spec, "--out", str(tmp_path / "r.json")]) == 2


def test_run_rejects_bad_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("release: {key: secret}\n")
    assert main(["run", "--config", str(cfg)]) == 2
    cfg.write_text("shell: rm -rf /\n")
    assert main(["run", "--config", str(cfg)]) == 2
