"""Failures must leave evidence and must not become successful experiment runs."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from provpub.experiments import e4
from provpub.experiments.real_data import read_release_key
from provpub.experiments.runner import load_config, run


def test_stage_failure_keeps_configuration_and_source_hashes(tmp_path):
    cfg = load_config(None, {"run": {"out_dir": str(tmp_path)}})
    with pytest.raises(ValueError, match="real-data"):
        run(cfg, ["real"])
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["summary"]["status"] == "failed"
    assert meta["summary"]["active_stage"] == "real"
    assert meta["summary"]["error_type"] == "ValueError"
    assert "src/provpub/methods/operators.py" in meta["source_sha256"]
    assert "analysis/e5_study_analysis.py" in meta["source_sha256"]
    assert all(".env" not in p for p in meta["source_sha256"])
    assert (tmp_path / "config_used.yaml").is_file()


def test_binary_release_key_keeps_all_bytes(tmp_path):
    key = b"\n" + bytes(range(32)) + b"\r\t"
    path = tmp_path / "release.key"
    path.write_bytes(key)
    assert read_release_key(str(path)) == key


@pytest.mark.parametrize("failure", ["hash", "process"])
def test_e4_rejects_inconsistent_or_partial_measurements(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(e4, "ensure_data", lambda *a: None)
    monkeypatch.setattr(e4, "ensure_shards", lambda *a: None)

    def measurement(cmd, **kwargs):
        workers = int(cmd[6])
        if failure == "process" and workers == 2:
            return SimpleNamespace(returncode=1, stderr="worker failed", stdout="")
        row = {"workload": "W2", "target_records": 5000, "workers": workers,
               "hash": str(workers) if failure == "hash" else "same", "t_sanitise": 1.0}
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(row))

    monkeypatch.setattr(e4.subprocess, "run", measurement)
    with pytest.raises(RuntimeError, match="hash mismatch" if failure == "hash" else "measurements failed"):
        e4.main(sizes=[5000], workers=[1, 2], workloads=["W2"], reps=1, out_dir=str(tmp_path), tag="failure")
    rows = pd.read_csv(tmp_path / "e4_scaling_failure.csv")
    assert len(rows) == 2
    if failure == "process":
        assert rows["error"].notna().sum() == 1
