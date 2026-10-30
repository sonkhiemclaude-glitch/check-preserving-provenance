"""Config-driven experiment runner (used by the CLI and scripts/).

A config is a YAML file with the sections documented in configs/default.yaml.
Configs contain only data (paths, seeds, grids); they never contain commands
or keys. Stages:

  e1           E1/E3 check preservation, verdict split, deviations
  e2           E2 attacks A1-A5, eps and Delta sweeps, group budget
  sensitivity  E2 sensitivity run
  e4           E4 scaling
  tables       result tables from result CSVs
  plots        Figures 2-4 from result CSVs (smoke runs labelled)
  real         release + verification of real traces under data.root (E5)
"""
from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any

import yaml

from ..utils.io import RUN_KINDS, read_run_kind, write_run_metadata

log = logging.getLogger(__name__)

STAGES = ("e1", "e2", "sensitivity", "e4", "tables", "plots", "real")
TOP_KEYS = {"run", "stages", "e1", "e2", "sensitivity", "e4", "tables", "data", "release"}

DEFAULTS: dict[str, Any] = {
    "run": {"kind": "smoke", "out_dir": "outputs/run", "log_level": "INFO"},
    "stages": ["e1", "tables"],
    "e1": {"seeds": [0], "runs": {"W1": 40, "W2": 50, "W3": 20}, "configs": None, "workloads": None},
    "e2": {"seeds": [0], "n_runs": 250, "eps_grid": [0.1, 1.0], "delta_grid": [60.0, 900.0], "configs": None,
           "w1_runs": 40, "dictionary": True},
    "sensitivity": {"seeds": [0], "n_runs": 250},
    "e4": {"sizes": [2.0e4, 5.0e4], "workers": [1, 2], "workloads": ["W2"], "reps": 1, "tag": "smoke", "cache": None},
    "tables": {"table3_seeds": [0]},
    "data": {"root": None},
    "release": {"key_file": None, "key_env": "PROVPUB_RELEASE_KEY", "write_release": True, "attacks": True,
                "operators": {"ops": ["VG", "TC", "BS", "DPN"], "delta": 60.0, "eps": 1.0, "unit": "bucket",
                              "memo": True, "sampler": "exact"}},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None, overrides: dict | None = None) -> dict:
    """Load a YAML config on top of DEFAULTS and validate it."""
    user: dict = {}
    if path:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    unknown = set(user) - TOP_KEYS
    if unknown:
        raise ValueError(f"unknown config sections {sorted(unknown)}; allowed: {sorted(TOP_KEYS)}")
    cfg = _merge(DEFAULTS, user)
    if overrides:
        cfg = _merge(cfg, overrides)
    if cfg["run"]["kind"] not in RUN_KINDS:
        raise ValueError(f"run.kind must be one of {RUN_KINDS}")
    bad = [s for s in cfg["stages"] if s not in STAGES]
    if bad:
        raise ValueError(f"unknown stages {bad}; allowed: {STAGES}")
    if "key" in cfg["release"]:
        raise ValueError("release.key is not allowed in configs; use release.key_file or release.key_env")
    return cfg


def label_smoke_tables(tab_dir: str) -> None:
    """Prefix every table of a smoke run with the synthetic label (CSV: '#' comment line)."""
    from ..utils.io import SYNTHETIC_LABEL
    for f in os.listdir(tab_dir):
        p = os.path.join(tab_dir, f)
        if not f.endswith((".tex", ".csv")):
            continue
        body = open(p, encoding="utf-8").read()
        mark = ("% " if f.endswith(".tex") else "# ") + SYNTHETIC_LABEL
        if not body.startswith(mark):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(mark + "\n" + body)
    with open(os.path.join(tab_dir, "LABEL.txt"), "w", encoding="utf-8") as fh:
        fh.write(SYNTHETIC_LABEL + "\n")


def run(cfg: dict, stages: list[str] | None = None) -> dict:
    """Run the requested stages and record provenance. Returns a summary."""
    stages = list(stages or cfg["stages"])
    out = cfg["run"]["out_dir"]
    res, tab, fig = (os.path.join(out, x) for x in ("results", "tables", "figures"))
    os.makedirs(res, exist_ok=True)
    kind = cfg["run"]["kind"]
    prev = read_run_kind(out)
    if prev and prev != kind:
        raise ValueError(f"{out} holds a '{prev}' run; refusing to mix it with a '{kind}' run")
    summary: dict[str, Any] = {"stages": {}, "requested_stages": stages, "status": "running"}
    t0 = time.time()
    write_run_metadata(out, cfg, kind, extra={"summary": summary})
    for st in stages:
        summary["active_stage"] = st
        write_run_metadata(out, cfg, kind, extra={"summary": summary})
        try:
            log.info("stage %s", st)
            if st == "e1":
                from .e1_e3 import main as e1
                c = cfg["e1"]
                summary["stages"]["e1"] = e1(seeds=c["seeds"], configs=c["configs"], out_dir=res, runs=c["runs"],
                                             workloads=c["workloads"])
            elif st == "e2":
                from .e2 import main as e2
                c = cfg["e2"]
                summary["stages"]["e2"] = e2(seeds=c["seeds"], out_dir=res, n_runs=c["n_runs"], eps_grid=c["eps_grid"],
                                             delta_grid=c["delta_grid"], configs=c["configs"], w1_runs=c["w1_runs"],
                                             dictionary=c["dictionary"])
            elif st == "sensitivity":
                from .e2 import sensitivity
                c = cfg["sensitivity"]
                summary["stages"]["sensitivity"] = sensitivity(seeds=c["seeds"], out_dir=res, n_runs=c["n_runs"])
            elif st == "e4":
                from .e4 import main as e4
                c = cfg["e4"]
                summary["stages"]["e4"] = e4(sizes=[float(x) for x in c["sizes"]], workers=c["workers"],
                                             workloads=c["workloads"], reps=c["reps"], out_dir=res, tag=c["tag"],
                                             cache=c["cache"])
            elif st == "tables":
                from .tables import make_all
                summary["stages"]["tables"] = make_all(res, tab, runs=cfg["e1"]["runs"],
                                                       seeds_table3=cfg["tables"]["table3_seeds"])
                if kind == "smoke":
                    label_smoke_tables(tab)
            elif st == "plots":
                from .plots import make_all as make_plots
                summary["stages"]["plots"] = make_plots(res, fig, run_kind=kind)
            elif st == "real":
                from .real_data import main as real
                if kind != "real-data":
                    raise ValueError("stage 'real' requires run.kind: real-data")
                if not cfg["data"]["root"]:
                    raise ValueError("stage 'real' requires data.root")
                r = cfg["release"]
                summary["stages"]["real"] = real(cfg["data"]["root"], res, operators=r["operators"],
                                                 key_file=r["key_file"], key_env=r["key_env"],
                                                 write_release=r["write_release"], attacks=r["attacks"])
        except BaseException as exc:
            summary.update(status="failed", error_type=type(exc).__name__, wall_seconds=time.time() - t0)
            write_run_metadata(out, cfg, kind, extra={"summary": summary})
            raise
        write_run_metadata(out, cfg, kind, extra={"summary": summary})
    summary.update(status="completed", active_stage=None)
    summary["wall_seconds"] = time.time() - t0
    write_run_metadata(out, cfg, kind, extra={"summary": summary})
    return summary
