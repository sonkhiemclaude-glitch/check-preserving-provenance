"""Explicit provisioning of five public regression datasets; never called by collection.

UCI source pages and licences checked 2026-10-24. Network access occurs only
when the caller explicitly enables it. Raw archives and transformations are
recorded so source-data provenance is separate from agent provenance.
"""
from __future__ import annotations

import io
import json
import shlex
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yaml
from sklearn.datasets import load_diabetes

from .workflow import Dataset, atomic_json, digest

CATALOG = {
    "auto_mpg": {"uci": 9, "slug": "auto+mpg", "member": "auto-mpg.data", "rows": 398,
                 "target": "mpg", "doi": "10.24432/C5859H"},
    "airfoil": {"uci": 291, "slug": "airfoil+self+noise", "member": "airfoil_self_noise.dat", "rows": 1503,
                "target": "scaled_sound_pressure", "doi": "10.24432/C5VW2C"},
    "yacht": {"uci": 243, "slug": "yacht+hydrodynamics", "member": "yacht_hydrodynamics.data", "rows": 308,
              "target": "residuary_resistance", "doi": "10.24432/C5XG7R"},
    "abalone": {"uci": 1, "slug": "abalone", "member": "abalone.data", "rows": 4177,
                "target": "rings", "doi": "10.24432/C55C7W"},
}


def parse_public(name: str, raw: bytes) -> tuple[pd.DataFrame, str]:
    """Parse the documented source schema without fitted or target-based preprocessing."""
    if name == "auto_mpg":
        columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration",
                   "model_year", "origin", "car_name"]
        rows = [shlex.split(line) for line in raw.decode().splitlines() if line.strip()]
        if any(len(row) != len(columns) for row in rows):
            raise ValueError("unexpected Auto MPG source schema")
        frame = pd.DataFrame(rows, columns=columns).drop(columns="car_name")
        frame = frame.replace("?", float("nan")).apply(pd.to_numeric)
        note = "Drop car_name ID; retain missing horsepower for training-only median imputation; no rows removed."
    elif name == "abalone":
        columns = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
                   "viscera_weight", "shell_weight", "rings"]
        frame = pd.read_csv(io.BytesIO(raw), header=None, names=columns)
        sex = frame.pop("sex")
        if not sex.isin(["M", "F", "I"]).all():
            raise ValueError("unexpected Abalone sex category")
        for value in ("M", "F", "I"):
            frame[f"sex_{value}"] = (sex == value).astype(int)
        note = "One-hot sex with the public fixed M/F/I vocabulary; rings is the regression target; no rows removed."
    elif name in ("airfoil", "yacht"):
        columns = (["frequency", "attack_angle", "chord_length", "free_stream_velocity", "displacement_thickness",
                    "scaled_sound_pressure"] if name == "airfoil" else
                   ["buoyancy_center", "prismatic_coefficient", "length_displacement_ratio", "beam_draught_ratio",
                    "length_beam_ratio", "froude_number", "residuary_resistance"])
        frame = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None)
        if len(frame.columns) != len(columns):
            raise ValueError(f"unexpected {name} source schema")
        frame.columns = columns
        note = "Assign documented column names; preserve all rows and numeric source values."
    else:
        raise ValueError("unknown public benchmark")
    return frame, note


def provision(root: Path, download_public: bool = False, sites: int = 12) -> Path:
    """Create a new manifest or validate a previously provisioned directory by hash."""
    if sites < 7 or sites > 24:
        raise ValueError("benchmark simulated sites must be 7..24")
    root = root.resolve()
    manifest, receipt = root / "manifest.yaml", root / "provision.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["sites"] != sites:
            raise ValueError("provisioned site count differs")
        for relative, sha in record["files_sha256"].items():
            p = root / relative
            if not p.is_file() or digest(p.read_bytes()) != sha:
                raise ValueError("provisioned dataset or manifest changed")
        return manifest
    if not download_public:
        raise ValueError("new public suite requires explicit --download-public; collection never downloads data")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("use an empty public-suite destination; partial provisioning is not overwritten")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e5-provision-", dir=root.parent) as staging:
        staged = Path(staging) / "suite"
        staged.mkdir()
        _provision_new(staged, sites)
        staged.rename(root)
    return manifest


def _provision_new(root: Path, sites: int) -> None:
    manifest, receipt = root / "manifest.yaml", root / "provision.json"
    archive_dir = root / "sources"
    archive_dir.mkdir()
    records, entries = [], []
    data = load_diabetes(as_frame=True)
    diabetes = data.data.copy()
    diabetes["target"] = data.target
    diabetes.to_csv(root / "diabetes.csv", index=False)
    source = "scikit-learn bundled diabetes; https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html"
    entries.append({"id": "diabetes", "path": "diabetes.csv", "target": "target", "source": source,
                    "sites": sites, "partition_seed": 42})
    records.append({"id": "diabetes", "rows": len(diabetes), "source": source,
                    "transformation": "Export bundled scaled features and unchanged numeric target; no network download."})
    for name, item in CATALOG.items():
        url = f"https://archive.ics.uci.edu/static/public/{item['uci']}/{item['slug']}.zip"
        with urlopen(url, timeout=60) as response:
            archive = response.read(8 * 1024 * 1024 + 1)
        if len(archive) > 8 * 1024 * 1024:
            raise ValueError("public benchmark archive exceeds size limit")
        (archive_dir / f"{name}.zip").write_bytes(archive)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            member = z.getinfo(str(item["member"]))
            if member.file_size > 16 * 1024 * 1024:
                raise ValueError("public benchmark member exceeds size limit")
            raw = z.read(member)
        frame, note = parse_public(name, raw)
        if len(frame) != item["rows"]:
            raise ValueError(f"unexpected {name} row count")
        target = str(item["target"])
        frame.to_csv(root / f"{name}.csv", index=False)
        source = f"UCI {name}; https://doi.org/{item['doi']}; CC BY 4.0"
        ds = Dataset(name, root / f"{name}.csv", target, source, sites=sites)
        entries.append({"id": name, "path": f"{name}.csv", "target": target, "source": source,
                        "sites": sites, "partition_seed": 42})
        records.append({"id": name, "rows": len(frame), "features": ds.features,
                        "missing_values": int(frame.isna().sum().sum()), "source": source,
                        "source_page": f"https://archive.ics.uci.edu/dataset/{item['uci']}/{item['slug']}",
                        "archive_url": url, "archive_sha256": digest(archive), "member": item["member"],
                        "member_sha256": digest(raw), "transformation": note})
    manifest.write_text(yaml.safe_dump({"datasets": entries}, sort_keys=False))
    record = {"datasets": records, "sites": sites, "partition_seed": 42,
              "files_sha256": {str(p.relative_to(root)): digest(p.read_bytes())
                               for p in sorted(root.rglob('*')) if p.is_file()}}
    atomic_json(receipt, record)
