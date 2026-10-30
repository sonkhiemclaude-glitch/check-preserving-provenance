#!/usr/bin/env python3
"""Offline operator and scripted-fault evaluation on captured E5 graphs.

This never alters the input capture or calls an LLM. Injected variants are
labelled post-collection faults, not genuine model deviations. Analysis code
and the collector snapshot are hashed independently in the result receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from provpub.checks.checks import align, compare, run_checks  # noqa: E402
from provpub.data.adapters import TraceDirectoryAdapter  # noqa: E402
from provpub.data.schemas import FAIL  # noqa: E402
from provpub.e5.workflow import atomic_json, digest  # noqa: E402
from provpub.experiments.configurations import build  # noqa: E402
from provpub.experiments.real_data import read_release_key  # noqa: E402
from provpub.utils.io import source_manifest  # noqa: E402
from provpub.utils.validation import validate_graph  # noqa: E402

CONFIGS = {
    "RAW": "raw",
    "VG": {"ops": ["VG"]},
    "VG+TC": {"ops": ["VG", "TC"], "delta": 60.0},
    "VG+TC+BS": {"ops": ["VG", "TC", "BS"], "delta": 60.0},
    "FULL-record": {"ops": ["VG", "TC", "BS", "DPN"], "delta": 60.0, "unit": "record", "eps": 1.0, "sampler": "exact"},
    "FULL-bucket": {"ops": ["VG", "TC", "BS", "DPN"], "delta": 60.0, "unit": "bucket", "eps": 1.0, "sampler": "exact"},
    "FULL-bucket-group": {
        "ops": ["VG", "TC", "BS", "DPN"],
        "delta": 60.0,
        "unit": "bucket",
        "eps": 1.0,
        "memo": "group",
        "sampler": "exact",
    },
    "VG-runscope": {"ops": ["VG"], "scopes": "runscope"},
    "SUPPRESS": "suppress",
}
FAULTS = {
    "missing_response_digest": "nd",
    "unapproved_model": "vocab",
    "skip_validation": "conf",
    "missing_internal_producer": "flow",
    "replica_output_change": "rep_outputs",
    "deadline_overrun": "time",
}


def inject(graph, spec, name: str):
    g = graph.copy()
    run = int(g.runs["run"].iloc[0])
    if name == "missing_response_digest":
        idx = g.ent.index[(g.ent["run"] == run) & (g.ent["type"] == "response")][0]
        g.ent.loc[idx, "digest"] = None
    elif name == "unapproved_model":
        idx = g.act.index[(g.act["run"] == run) & g.act["type"].isin(spec.nd_types)][0]
        g.act.loc[idx, "model"] = "SCRIPTED-UNAPPROVED-MODEL"
    elif name == "skip_validation":
        ids = set(g.act.loc[(g.act["run"] == run) & (g.act["type"] == "validate_schema"), "id"])
        if not ids:
            raise ValueError("validation fault needs a captured validation")
        g.act = g.act[~g.act["id"].isin(ids)].copy()
        g.edge = g.edge[~g.edge["src"].isin(ids) & ~g.edge["dst"].isin(ids)].copy()
    elif name == "missing_internal_producer":
        candidates = g.ent[(g.ent["run"] == run) & g.ent["type"].isin(["fitted_model", "table"])]
        model = candidates["id"].iloc[-1]
        g.edge = g.edge[~((g.edge["kind"] == "wasGeneratedBy") & (g.edge["src"] == model))].copy()
    elif name == "replica_output_change":
        groups = g.runs.groupby(spec.replica_attr, sort=False)["run"].agg(list)
        groups = [r for r in groups if len(r) > 1]
        if not groups:
            raise ValueError("output fault needs a replica group")
        run = int(groups[0][-1])
        idx = g.ent.index[(g.ent["run"] == run) & (g.ent["type"] == spec.result_type)][0]
        text = json.dumps({"scripted_fault": "replica_output_change"})
        g.ent.loc[idx, ["text", "digest"]] = [text, digest(text)]
    elif name == "deadline_overrun":
        idx = g.act.index[g.act["run"] == run][-1]
        span = next(r.limit for r in spec.timed if r.kind == "span")
        g.act.loc[idx, "end"] = g.act.loc[g.act["run"] == run, "start"].min() + span + 120
    else:
        raise ValueError("unknown scripted fault")
    problems = validate_graph(g)
    if problems:
        raise ValueError("invalid scripted graph: " + "; ".join(problems))
    return g, run


def evaluate(root: Path, out: Path, key_file: str):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("choose a new offline evaluation output")
    adapter = TraceDirectoryAdapter(str(root))
    graph, spec = adapter.load(), adapter.spec()
    key = read_release_key(key_file)
    raw = run_checks(graph, spec)
    if (raw["outcome"] != 1).any():
        raise ValueError("offline baseline requires a clean captured graph")
    out.mkdir(parents=True, exist_ok=True)
    utility, faults = [], []
    for config, options in CONFIGS.items():
        release = build(graph, spec, "W2", options, key)
        comparison = compare(raw, align(release, run_checks(release, spec)))
        for family, rows in comparison.groupby("family"):
            utility.append(
                {
                    "config": config,
                    "family": family,
                    "instances": len(rows),
                    "preserved": int(rows["same"].sum()),
                    "to_undet": int((rows["outcome_san"] == -1).sum()),
                }
            )
        if config.startswith("FULL-") or config in ("VG", "VG+TC", "VG+TC+BS"):
            exact = comparison[~comparison["family"].isin(["thr", "time"])]
            if not exact["same"].all():
                raise ValueError("unexpected nonnumeric verdict change")
    for name, family in FAULTS.items():
        variant, run = inject(graph, spec, name)
        raw_fault = run_checks(variant, spec)
        for config in ("VG+TC+BS", "FULL-record", "FULL-bucket"):
            release = build(variant, spec, "W2", CONFIGS[config], key)
            comparison = compare(raw_fault, align(release, run_checks(release, spec)))
            target = comparison[(comparison["family"] == family) & (comparison["outcome_raw"] == FAIL)]
            if target.empty:
                raise ValueError(f"fault {name} did not trigger its intended family")
            detected = bool((target["outcome_san"] == FAIL).all())
            faults.append(
                {
                    "fault": name,
                    "family": family,
                    "run": run,
                    "config": config,
                    "raw_fail_instances": len(target),
                    "detected": detected,
                    "scripted_post_collection": True,
                }
            )
            if not detected:
                raise ValueError(f"fault visibility lost for {name}")
    pd.DataFrame(utility).to_csv(out / "operator_utility.csv", index=False)
    pd.DataFrame(faults).to_csv(out / "scripted_faults.csv", index=False)
    receipt = {
        "status": "completed",
        "source_root": str(root.resolve()),
        "runs": len(graph.runs),
        "prov_records": graph.n_records,
        "configurations": list(CONFIGS),
        "faults": list(FAULTS),
        "fault_evaluations": len(faults),
        "all_injected_faults_detected": True,
        "interpretation": "post-collection scripted graph faults on captured real traces; not new LLM deviations",
        "collector_source_sha256": json.loads((root.parent / "run_metadata.json").read_text())["source_sha256"],
        "analysis_dependencies_sha256": source_manifest(),
        "analysis_script_sha256": digest(Path(__file__).read_bytes()),
        "raw_files_sha256": {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()},
        "model_requests": 0,
    }
    atomic_json(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key-file", required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate(args.root, args.out, args.key_file), indent=2))
