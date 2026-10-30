"""Runner, labelling of synthetic outputs, and result integrity of the repository."""
import json
import os

import pytest

from provpub.experiments.runner import load_config, run
from provpub.utils.io import SYNTHETIC_LABEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_shipped_configs_load():
    for f in os.listdir(os.path.join(ROOT, "configs")):
        if f.endswith(".yaml"):
            cfg = load_config(os.path.join(ROOT, "configs", f))
            assert cfg["run"]["kind"] in ("smoke", "reference-protocol", "real-data")
            assert "key" not in cfg["release"]


def test_smoke_run_is_labelled(tmp_path):
    cfg = load_config(None, {"run": {"kind": "smoke", "out_dir": str(tmp_path / "o")},
                             "e1": {"runs": {"W1": 12, "W2": 12, "W3": 6}, "configs": ["RAW", "VG+TC+BS"]},
                             "tables": {"table3_seeds": [0]}})
    run(cfg, ["e1", "tables"])
    meta = json.load(open(tmp_path / "o" / "run_metadata.json"))
    assert meta["run_kind"] == "smoke" and meta["label"] == SYNTHETIC_LABEL
    tab = tmp_path / "o" / "tables"
    for f in os.listdir(tab):
        if f.endswith((".tex", ".csv")):
            assert open(tab / f).readline().strip().endswith(SYNTHETIC_LABEL), f


def test_runner_refuses_to_mix_run_kinds(tmp_path):
    out = str(tmp_path / "o")
    cfg = load_config(None, {"run": {"kind": "smoke", "out_dir": out}, "e1": {"runs": {"W1": 5, "W2": 5, "W3": 3},
                                                                           "configs": ["RAW"]}})
    run(cfg, ["e1"])
    cfg2 = load_config(None, {"run": {"kind": "reference-protocol", "out_dir": out}})
    with pytest.raises(ValueError):
        run(cfg2, ["tables"])


def test_real_stage_requires_real_kind_and_root(tmp_path):
    cfg = load_config(None, {"run": {"kind": "smoke", "out_dir": str(tmp_path / "x")}})
    with pytest.raises(ValueError):
        run(cfg, ["real"])


def test_no_result_files_shipped_as_real_results():
    """outputs/ must hold only .gitkeep or runs labelled as synthetic smoke tests."""
    out = os.path.join(ROOT, "outputs")
    for entry in os.listdir(out):
        p = os.path.join(out, entry)
        if entry in (".gitkeep", "README.md"):
            continue
        assert os.path.isdir(p), f"unexpected file in outputs/: {entry}"
        meta = os.path.join(p, "run_metadata.json")
        assert os.path.exists(meta), f"{entry} has no run_metadata.json"
        assert json.load(open(meta))["run_kind"] == "smoke", f"{entry} is not a smoke run"
