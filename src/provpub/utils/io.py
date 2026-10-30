"""Result files and run provenance."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

RUN_KINDS = ("smoke", "reference-protocol", "real-data")
SYNTHETIC_LABEL = "synthetic smoke-test output, not a reference result"


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "machine": platform.machine(), "cpus": os.cpu_count(),
                            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # Read installed metadata without loading unrelated native libraries. In
    # particular, recording an E1/E4 run must not depend on importing sklearn.
    for mod, distribution in (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                              ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("yaml", "pyyaml")):
        try:
            info[mod] = version(distribution)
        except PackageNotFoundError:
            info[mod] = None
    try:
        from .. import __version__
        info["provpub"] = __version__
    except ImportError:
        pass
    info["threads"] = {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    info["python_dont_write_bytecode"] = os.environ.get("PYTHONDONTWRITEBYTECODE")
    if platform.system() == "Darwin":
        for field, name in (("model", "hw.model"), ("memory_bytes", "hw.memsize"),
                            ("chip", "machdep.cpu.brand_string")):
            result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
            info[field] = result.stdout.strip()
    return info


def source_manifest() -> dict[str, str]:
    """Hash implementation, analysis, scripts and dependencies; exclude secrets and outputs."""
    root = Path(__file__).resolve().parents[3]
    paths = [
        *root.glob("src/**/*.py"),
        *root.glob("analysis/**/*.py"),
        *root.glob("scripts/*.py"),
        root / "pyproject.toml",
        root / "requirements.txt",
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def git_revision(path: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_metadata(out_dir: str, config: dict, run_kind: str, extra: dict | None = None) -> str:
    """Write run_metadata.json and config_used.yaml next to the results."""
    if run_kind not in RUN_KINDS:
        raise ValueError(f"run_kind must be one of {RUN_KINDS}")
    os.makedirs(out_dir, exist_ok=True)
    meta = {"run_kind": run_kind, "label": SYNTHETIC_LABEL if run_kind == "smoke" else
            ("reference protocol on synthetic workloads; compare with results/ only if the config equals the reference config"
             if run_kind == "reference-protocol" else "real-data run; provenance of the data in manifest"),
            "env": env_info(), "git_revision": git_revision(os.getcwd()), "source_sha256": source_manifest()}
    if extra:
        meta.update(extra)
    p = os.path.join(out_dir, "run_metadata.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    if run_kind == "smoke":
        with open(os.path.join(out_dir, "LABEL.txt"), "w", encoding="utf-8") as f:
            f.write(SYNTHETIC_LABEL + "\n")
    return p


def read_run_kind(out_dir: str) -> str | None:
    p = os.path.join(out_dir, "run_metadata.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("run_kind")


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
