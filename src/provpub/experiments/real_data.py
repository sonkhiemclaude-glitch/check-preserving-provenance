"""Release and verification of REAL traces placed manually under data.root
(real-trace release and verification).

What it does once real data exist:
  1. load traces and the declared specification (data/adapters.TraceDirectoryAdapter);
  2. validate the graph (utils/validation.py) and the consistent-provenance
     precondition of Theorem 4(c);
  3. sanitise with the configured operators and a release key read from a file
     or an environment variable (never from the config itself);
  4. run all checks on raw and release, write per-instance outcomes and the
     verdict split;
  5. optionally run A1 (site membership) if truth/runs.csv and
     truth/sites_aux.csv exist (auxiliary knowledge of the adversary);
  6. optionally write the release as PROV-JSON.

A2 (shadow release) and A3 (W2-specific timing features) are not defined for
arbitrary real pipelines.
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pandas as pd

from ..checks.checks import align, compare, run_checks
from ..data.adapters import TraceDirectoryAdapter
from ..data.prov_io import to_prov_json
from ..methods.operators import sanitise
from ..metrics.metrics import preservation_by_family, verdict_transitions
from ..utils.random import key_fingerprint
from ..utils.validation import consistent_provenance, validate_graph

log = logging.getLogger(__name__)


def read_release_key(key_file: str | None = None, key_env: str | None = None) -> bytes:
    """Read the secret release key (>= 16 bytes) from a file or environment variable."""
    if key_file:
        with open(key_file, "rb") as f:
            key = f.read()
    elif key_env:
        v = os.environ.get(key_env)
        if not v:
            raise KeyError(f"environment variable {key_env} is not set")
        key = v.encode()
    else:
        raise ValueError("set release.key_file or release.key_env; keys are never stored in configs")
    if len(key) < 16:
        raise ValueError("release key must have at least 16 bytes")
    return key


def _sites_from_aux(path: str):
    """Adversary auxiliary knowledge per site: site, approx_records, row_width, load_seconds."""
    df = pd.read_csv(path)
    need = {"site", "approx_records", "row_width", "load_seconds"}
    if not need <= set(df.columns):
        raise ValueError(f"{path} needs columns {sorted(need)}")
    out = []
    for r in df.itertuples():
        out.append(SimpleNamespace(name=str(r.site), base_records=float(r.approx_records), row_width=float(r.row_width),
                                   latency=0.0, throughput=float(r.approx_records) / max(float(r.load_seconds), 1e-9)))
    return out


def main(root: str, out_dir: str, operators: dict, key_file: str | None = None, key_env: str | None = None,
         write_release: bool = True, attacks: bool = True, seed: int = 0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ad = TraceDirectoryAdapter(root)
    spec = ad.spec()
    g = ad.load()
    problems = validate_graph(g)
    if problems:
        raise ValueError("invalid input graph: " + "; ".join(problems))
    consistent = consistent_provenance(g, spec.source_type)
    key = read_release_key(key_file, key_env)
    h = sanitise(g, spec, key, **operators)
    raw = run_checks(g, spec)
    m = compare(raw, align(h, run_checks(h, spec)))
    m.to_csv(os.path.join(out_dir, "outcomes.csv"), index=False)
    preservation_by_family(m).to_csv(os.path.join(out_dir, "preservation_by_family.csv"), index=False)
    summary = {"runs": int(len(g.runs)), "prov_records": int(g.n_records), "consistent_provenance": consistent,
               "key_fingerprint": key_fingerprint(key), "operators": operators,
               "verdicts": verdict_transitions(m), "source": ad.manifest.get("source")}
    if write_release:
        rel = os.path.join(out_dir, "release.prov.json")
        summary["release_bytes"] = to_prov_json(h, rel)
    truth = ad.truth()
    aux = os.path.join(root, "truth", "sites_aux.csv")
    if attacks and truth is not None and os.path.exists(aux):
        from ..attacks.attacks import membership_attack
        sites = _sites_from_aux(aux)
        auc, per, auc_unlinked = membership_attack(h, spec, truth, sites, list(range(len(sites))), seed=seed)
        summary["A1_membership_auc"] = auc
        summary["A1_membership_auc_unlinked"] = auc_unlinked
        pd.DataFrame([{"site": k, "auc": v} for k, v in per.items()]).to_csv(
            os.path.join(out_dir, "a1_per_site.csv"), index=False)
    else:
        summary["A1_membership_auc"] = None
        log.info("A1 skipped: needs truth/runs.csv and truth/sites_aux.csv under data.root")
    return summary
