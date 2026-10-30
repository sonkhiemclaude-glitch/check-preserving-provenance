#!/usr/bin/env python3
"""Final E5 utility, deviation visibility, A1-A5 and reproducible evidence.

Two workflows are evaluated against their own declared specifications. Only
primary tasks enter the attacks. The dictionary comes from public templates;
shadow labels come from independent public partition versions, not test data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e5_attacks as attacks
import numpy as np
import pandas as pd
import provpub.e5.workflow as e5_workflow
import yaml
from provpub.checks.checks import align, compare, run_checks
from provpub.data.adapters import TraceDirectoryAdapter
from provpub.data.prov_io import to_prov_json
from provpub.data.schemas import concat_graphs, ranks
from provpub.data.spec import lift_spec
from provpub.e5.workflow import (
    Dataset,
    Tools,
    Trace,
    atomic_json,
    canonical,
    digest,
    load_datasets,
)
from provpub.experiments.configurations import CONFIGS, SALT, build
from provpub.experiments.e1_e3 import timed_bound
from provpub.methods.operators import blocks
from provpub.utils.random import key_fingerprint
from provpub.utils.validation import consistent_provenance, validate_graph


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def key_for(root, name):
    directory = root / "release_keys"
    directory.mkdir(exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / f"{name}.key"
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    return path.read_bytes()


def subset(graph, gids):
    selected = set(map(int, gids))
    g = graph.copy()
    g.runs = g.runs[g.runs.run.isin(selected)].copy()
    g.act = g.act[g.act.run.isin(selected)].copy()
    g.ent = g.ent[g.ent.run.isin(selected)].copy()
    g.edge = g.edge[g.edge.run.isin(selected)].copy()
    return g


def merge_workflow(root, cohorts, workflow):
    target = root / "raw_workflows" / workflow
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    (target / "traces").mkdir(exist_ok=True)
    (target / "truth").mkdir(exist_ok=True)
    rows = []
    provenance = []
    totals = {"input_tokens": 0, "output_tokens": 0, "attempts": 0, "live_runs": 0, "replay_runs": 0}
    plan = json.loads((root / "study_plan.json").read_text())
    for cohort in cohorts:
        config = yaml.safe_load((cohort / "config_used.yaml").read_text())
        if config["experiment"]["workflow"] != workflow:
            continue
        state = json.loads((cohort / "collection_state.json").read_text())
        if state["status"] != "completed" or state["source_sha256"] != plan["frozen_source_sha256"]:
            raise ValueError("cohort incomplete or collector version differs")
        shutil.copyfile(cohort / "raw/spec.yaml", target / "spec.yaml")
        for r in state["runs"].values():
            filename = f"{cohort.parent.name}__{cohort.name}__{r['run_file']}"
            p = cohort / "raw/traces" / r["run_file"]
            c = cohort / "raw/captures" / f"run_{r['run']:05d}.json"
            if digest(p.read_bytes()) != r["trace_sha256"] or digest(c.read_bytes()) != r["capture_sha256"]:
                raise ValueError("capture checksum changed")
            shutil.copyfile(p, target / "traces" / filename)
            rows.append(
                {
                    "run_file": filename,
                    "run": r["run"],
                    "user": r["user"],
                    "sites": r["sites"],
                    "mode": r["mode"],
                    "dataset": r["dataset"],
                    "task": r["group"],
                    "workflow": workflow,
                    "source_cohort": str(cohort),
                    "source_run": r["run"],
                    "steps": r["steps"],
                }
            )
            totals["live_runs" if r["mode"] == "live" else "replay_runs"] += 1
            if r["mode"] == "live":
                captures = json.loads(c.read_text())
                if any(x["response"].get("model") != "openai.gpt-oss-120b" for x in captures):
                    raise ValueError("reported model differs from requested model")
        journal = [json.loads(x) for x in (cohort / "requests.jsonl").read_text().splitlines()]
        totals["attempts"] += sum(x["status"] == "started" for x in journal)
        for row in journal:
            if row["status"] == "ok":
                totals["input_tokens"] += row["input_tokens"]
                totals["output_tokens"] += row["output_tokens"]
        provenance.append(
            {
                "cohort": str(cohort),
                "source_sha256": state["source_sha256"],
                "summary": state["summary"],
                "checkpoint_sha256": digest((cohort / "collection_state.json").read_bytes()),
            }
        )
    pd.DataFrame(rows).to_csv(target / "truth/runs.csv", index=False)
    (target / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "prov-json",
                "source": f"E5 full {workflow}",
                "capture_mode": "live and recorded-decision replay",
                "simulated_sites": True,
                "simulated_analyst_profiles": True,
            }
        )
    )
    adapter = TraceDirectoryAdapter(str(target))
    g = adapter.load()
    spec = adapter.spec()
    frame = pd.DataFrame(rows)
    frame["gid"] = [int(r.run) + adapter.run_offsets[r.run_file] for r in frame.itertuples()]
    if validate_graph(g) or not consistent_provenance(g, spec.source_type):
        raise ValueError("invalid merged graph")
    if (run_checks(g, spec)["outcome"] != 1).any():
        raise ValueError("collected raw graph is not clean")
    return target, g, spec, frame, pd.DataFrame(provenance), totals


def shadow_graph(datasets, deterministic_clock=False):
    if not deterministic_clock:
        return _shadow_graph_impl(datasets)
    original_analysis_clock = globals()["time"]
    original_workflow_clock = e5_workflow.time
    counter = 0

    def fixed_time():
        nonlocal counter
        counter += 1
        return 1_700_000_000.0 + counter / 100.0

    clock = SimpleNamespace(time=fixed_time)
    globals()["time"] = clock
    e5_workflow.time = clock
    try:
        return _shadow_graph_impl(datasets)
    finally:
        globals()["time"] = original_analysis_clock
        e5_workflow.time = original_workflow_clock


def _shadow_graph_impl(datasets):
    parts = []
    for di, ds in enumerate(datasets):
        for seed in range(100, 105):
            shadow = Dataset(ds.name, ds.path, ds.target, ds.source, 12, seed, "weighted")
            run = di * 10 + seed - 100
            trace = Trace(run, f"public-shadow-{ds.name}-{seed}", "auxiliary-controller")
            instruction = trace.entity("prompt", "Public independent shadow partition registry, tool-only, no LLM.")
            controller = trace.activity("shadow_controller", time.time(), time.time(), [instruction], [])
            tools = Tools(shadow, shadow.site_ids, trace, 42)
            for site in shadow.site_ids:
                tools.execute("load_partition", {"site": site}, instruction, controller)
            parts.append(trace.graph())
    # Each trace allocates local node IDs; offset them before concatenation.
    from provpub.data.adapters import _offset

    offset = 0
    adjusted = []
    for g in parts:
        g = _offset(g, 0, offset)
        adjusted.append(g)
        offset = int(max(g.act.id.max(), g.ent.id.max())) + 1
    combined = concat_graphs(adjusted)
    # Null paths are absent from the native PROV-JSON wire schema. The shadow
    # controller has no path rule or path value; keep the identical schema.
    if "p_path" in combined.act and combined.act["p_path"].isna().all():
        combined.act = combined.act.drop(columns="p_path")
    return combined


def shadow_latency_records(shadow):
    loads = shadow.act[shadow.act.type == "load_partition"]
    records = []
    seen = set()
    for row in loads.itertuples():
        key = (int(row.run), str(row.site))
        duration = float(row.end) - float(row.start)
        if key in seen or not np.isfinite(duration) or duration < 0:
            raise ValueError("invalid or duplicate shadow load duration")
        seen.add(key)
        records.append({"run": key[0], "site": key[1], "duration_seconds": duration})
    if not records:
        raise ValueError("shadow graph contains no load durations")
    return sorted(records, key=lambda item: (item["run"], item["site"]))


def apply_shadow_latency_records(shadow, records):
    loads = shadow.act[shadow.act.type == "load_partition"]
    indices = {}
    for index, row in loads.iterrows():
        key = (int(row.run), str(row.site))
        if key in indices:
            raise ValueError("duplicate shadow load activity")
        indices[key] = index
    values = {}
    for item in records:
        key = (int(item["run"]), str(item["site"]))
        duration = float(item["duration_seconds"])
        if key in values or not np.isfinite(duration) or duration < 0:
            raise ValueError("invalid or duplicate cached shadow load duration")
        values[key] = duration
    if not indices or set(indices) != set(values):
        raise ValueError("shadow latency cache does not match the generated load activities")
    for key, index in indices.items():
        start = float(shadow.act.at[index, "start"])
        shadow.act.at[index, "end"] = start + values[key]


def evaluate(root):
    os.umask(0o077)
    destination = root / "results/e5_study"
    destination.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "study_plan.json").read_text())
    registry = json.loads((root / "e5_cohorts.json").read_text())
    if registry["traces"] < 500 or registry["prov_records"] < 100000:
        raise ValueError("study volume gate failed")
    cohorts = [Path(p) for p in registry["cohorts"]]
    datasets = []
    for p in sorted((root / "study_inputs").glob("*.yaml")):
        doc = yaml.safe_load(p.read_text())
        if isinstance(doc, dict) and "datasets" in doc:
            datasets += load_datasets(p)
    if len(datasets) != 5:
        raise ValueError("expected five public dataset manifests")
    candidates = attacks.dictionary_candidates(datasets)
    latency_cache_path = destination / "shadow_latency_cache.json"
    if latency_cache_path.exists():
        latency_records = json.loads(latency_cache_path.read_text())
        if not isinstance(latency_records, list):
            raise ValueError("shadow latency cache must be a list of per-run/site durations")
        latency_cache_source = "replayed"
    else:
        measured_shadow = shadow_graph(datasets)
        latency_records = shadow_latency_records(measured_shadow)
        atomic_json(latency_cache_path, latency_records)
        latency_cache_source = "captured_on_first_analysis"
    shadow = shadow_graph(datasets, deterministic_clock=True)
    apply_shadow_latency_records(shadow, latency_records)
    latency_cache_sha256 = digest(latency_cache_path.read_bytes())
    shadow_sources = shadow.ent[shadow.ent.type == "data_partition"].drop_duplicates("part_id")
    thresholds = np.quantile(shadow_sources.n_records.astype(float), [1 / 3, 2 / 3])
    if thresholds[0] >= thresholds[1]:
        raise ValueError("shadow population has no three-class size problem")
    reports = []
    utility = []
    fault_receipts = []
    all_stats = []
    configurations = {
        name: ({**c, "sampler": "exact"} if isinstance(c, dict) and "DPN" in c.get("ops", ()) else c)
        for name, c in CONFIGS.items()
    }
    configurations["FULL-nosize"] = {
        "ops": ["VG", "TC", "BS", "DPN"],
        "delta": 60.0,
        "eps": 1.0,
        "unit": "suppress",
        "sampler": "exact",
    }
    private_types = {
        "W1": {"load_partition", "validate_schema", "tool_batch", "agent_step", "agent_report"},
        "W2": {"load_partition", "validate_schema", "tool_batch", "agent_step", "agent_report"},
    }
    abstract_group = {"W1": {"load_partition", "validate_schema"}, "W2": {"load_partition", "validate_schema"}}
    for workflow in ("explore", "regression"):
        print("MERGE", workflow, flush=True)
        raw_root, g, spec, labels, provenance, usage = merge_workflow(root, cohorts, workflow)
        primary = labels[labels["mode"] == "live"].copy()
        raw_primary = subset(g, primary.gid)
        key = key_for(root, f"{workflow}-target")
        sk = key_for(root, f"{workflow}-shadow")
        raw_checks = run_checks(g, spec)
        ranked = g.copy()
        ranked.act["rank"] = ranks(ranked.act)
        bs = blocks(ranked)
        if len(bs) == 0:
            raise ValueError("BS coverage is still empty")
        auxiliary = {}
        part_datasets = {}
        for ds in datasets:
            auxiliary[ds.name] = {}
            for site in ds.site_ids:
                frame = ds.partition(site)
                n = len(frame)
                width = len(frame.to_csv(index=False).encode()) / n
                # Duration auxiliary is estimated only from labelled training-side
                # public load instrumentation; it is not a target membership label.
                auxiliary[ds.name][site] = {
                    "records": n,
                    "width": width,
                    "load_seconds": float(
                        (
                            shadow.act.loc[shadow.act.site == site, "end"] - shadow.act.loc[shadow.act.site == site, "start"]
                        ).median()
                    ),
                }
            for row in g.ent[(g.ent.type == "data_partition") & g.ent.site.isin(ds.site_ids)].itertuples():
                part_datasets[str(row.part_id)] = ds.name
        wl = "W1" if workflow == "explore" else "W2"
        for config, options in configurations.items():
            print("EVALUATE", workflow, config, flush=True)
            h = build(g, spec, wl, options, key, private_types=private_types, abstract_group=abstract_group)
            hs = build(shadow, spec, wl, options, sk, private_types=private_types, abstract_group=abstract_group)
            view = lift_spec(spec, abstract_group[wl]) if config == "ABSTRACT-lifted" else spec
            raw_view = run_checks(g, view) if config == "ABSTRACT-lifted" else raw_checks
            comparison = compare(raw_view, align(h, run_checks(h, view)))
            for family, part in comparison.groupby("family"):
                utility.append(
                    {
                        "workflow": workflow,
                        "config": config,
                        "family": family,
                        "instances": len(part),
                        "preserved": int(part["same"].sum()),
                        "raw_fail": int((part.outcome_raw == 0).sum()),
                        "to_undet": int((part.outcome_san == -1).sum()),
                        "pass_fail": int(((part.outcome_raw == 1) & (part.outcome_san == 0)).sum()),
                    }
                )
            strong = config in (
                "RAW",
                "VG",
                "VG-salted",
                "TC-60",
                "TC-600",
                "TC-3600",
                "VG+TC+BS",
                "FULL-record",
                "FULL-bucket",
                "FULL-bucket-group",
                "FULL-nosize",
            )
            changed = comparison[~comparison.family.isin(["thr", "time"]) & ~comparison["same"]]
            if config == "FULL-nosize":
                # Suppressed counts leave count-reading replica checks undetermined (Kleene), never wrong.
                changed = changed[~((changed.family == "rep_counts") & (changed.outcome_san == -1))]
            if strong and len(changed):
                raise ValueError(f"unexpected nonnumeric change in {workflow}/{config}")
            bound = timed_bound(
                g, comparison, options.get("delta") if isinstance(options, dict) and "TC" in options.get("ops", ()) else None
            )
            if bound and bound[1]:
                raise ValueError("timed coarsening bound violated")
            inv = attacks.original_runs(h)
            keep = [r for r in h.runs.run if inv[int(r)] in set(primary.gid)]
            hp = subset(h, keep)
            salt = SALT if config == "VG-salted" else None
            a1, p1 = attacks.membership(hp, spec, primary, auxiliary, salt)
            a2, p2 = attacks.size_classes(hp, hs, raw_primary, shadow, spec, key, sk, thresholds, part_datasets)
            a3, p3 = attacks.analyst_profiles(hp, primary, salt)
            a4 = attacks.dictionary(hp, raw_primary, key, candidates, salt)
            a5, p5 = attacks.load_order(hp, raw_primary, spec, key, primary)
            report = {
                "workflow": workflow,
                "config": config,
                "A1": a1,
                "A2": a2,
                "A3_profiles": a3,
                "A4": a4,
                "A5": a5,
                "timed_bound": bound,
                "nonnumeric_preservation": bool(comparison.loc[~comparison.family.isin(["thr", "time"]), "same"].all()),
            }
            reports.append(report)
            folder = destination / "predictions" / workflow / config
            folder.mkdir(parents=True, exist_ok=True)
            for name, pred in [("a1", p1), ("a2", p2), ("a3_profiles", p3), ("a5", p5)]:
                if not pred.empty:
                    pred.to_csv(folder / f"{name}.csv", index=False)
            if config in ("FULL-record", "FULL-bucket", "FULL-nosize"):
                folder = destination / "releases" / workflow / config
                folder.mkdir(parents=True, exist_ok=True)
                to_prov_json(h, str(folder / "release.prov.json"))
                shutil.copyfile(raw_root / "spec.yaml", folder / "spec.yaml")
            atomic_json(destination / "attacks_partial.json", safe_json(reports))
        module_path = Path(__file__).with_name("e5_checks.py")
        module_spec = importlib.util.spec_from_file_location("e5_checks", module_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        # Utility/fault evaluator reads the original collector metadata from the
        # consolidated raw parent, with all contributing snapshots retained.
        atomic_json(raw_root.parent / "run_metadata.json", {"source_sha256": plan["frozen_source_sha256"]})
        fault_out = destination / "faults" / workflow
        if fault_out.exists():
            shutil.rmtree(fault_out)
        receipt = module.evaluate(raw_root, fault_out, str(root / "release_keys" / f"{workflow}-target.key"))
        fault_receipts.append(receipt)
        all_stats.append(
            {
                "workflow": workflow,
                "traces": len(labels),
                "live": len(primary),
                "replay": int((labels["mode"] == "replay").sum()),
                "records": g.n_records,
                "eligible_bs_blocks": len(bs),
                "activities_in_blocks": sum(len(x) for _, x in bs),
                "primary_tasks_only_for_attacks": len(primary),
                "usage": usage,
                "key_fingerprint": key_fingerprint(key),
            }
        )
        provenance.to_json(destination / f"{workflow}_cohort_provenance.json", orient="records", indent=2)
    pd.DataFrame(utility).to_csv(destination / "utility_by_family.csv", index=False)
    atomic_json(destination / "attacks.json", safe_json(reports))
    total_traces = sum(s["traces"] for s in all_stats)
    total_records = sum(s["records"] for s in all_stats)
    if total_traces < 500 or total_records < 100000:
        raise ValueError("post-merge volume differs from registry")
    receipt = {
        "status": "completed",
        "traces": total_traces,
        "prov_records": total_records,
        "workflows": all_stats,
        "attack_configurations": len(configurations),
        "attacks": ["A1", "A2", "A3 simulated profiles", "A4 public initial dictionary", "A5"],
        "uncertainty": {
            "method": "percentile cluster bootstrap, conditional on fitted attack models",
            "replicates": attacks.BOOTSTRAPS,
            "confidence": 0.95,
            "unit": "primary task for A1/A3/A5; source partition version for A2",
        },
        "shadow": {
            "kind": "actual public-data tool executions, no LLM inference",
            "partition_seeds": list(range(100, 105)),
            "records": shadow.n_records,
            "source_versions": len(shadow_sources),
            "class_thresholds": thresholds.tolist(),
            "load_duration_cache_sha256": latency_cache_sha256,
            "load_duration_cache_source": latency_cache_source,
            "absolute_timestamps": "deterministic synthetic clock; measured durations are frozen per run/site",
        },
        "dictionary_candidates": len(candidates),
        "dictionary_sha256": digest(canonical(candidates)),
        "scripted_fault_evaluations": sum(r["fault_evaluations"] for r in fault_receipts),
        "all_scripted_faults_detected": all(r["all_injected_faults_detected"] for r in fault_receipts),
        "collector_source_sha256": plan["frozen_source_sha256"],
        "analysis_sha256": {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("e5_*.py")},
        "limitations": [
            "Analyst labels are simulated profiles, not real people.",
            "Dictionary covers public initial templates, not arbitrary later histories.",
            "Membership auxiliary knowledge is conditional on a known public dataset and approximate per-site size/latency.",
            "Replays are excluded from attack training and uncertainty.",
            "Attack success is empirical, not a privacy proof.",
        ],
    }
    atomic_json(destination / "receipt.json", safe_json(receipt))
    print(json.dumps(safe_json(receipt), indent=2), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    evaluate(parser.parse_args().root.resolve())
