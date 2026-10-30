#!/usr/bin/env python3
"""Run the offline E5 held-out-dataset size-class transfer analysis.

For each public dataset, train the E5 size-class attacker on independent
shadow partitions from the other four datasets, then score the held-out
dataset's live traces. An all-five-dataset shadow is recorded as a
known-dataset reference. No LLM/API calls are made.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
import yaml


def apply_frozen_shadow_latency(shadow, records) -> None:
    """Apply and validate the exact run/site duration cache from E5."""
    loads = shadow.act[shadow.act["type"] == "load_partition"]
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
            raise ValueError("invalid or duplicate duration in frozen shadow cache")
        values[key] = duration
    if not indices or set(indices) != set(values):
        raise ValueError("frozen shadow cache does not match generated run/site activities")
    for key, index in indices.items():
        start = float(shadow.act.at[index, "start"])
        shadow.act.at[index, "end"] = start + values[key]


def build_deterministic_shadow(datasets, analysis_module, workflow_module):
    """Use the same deterministic clock protocol across E5 source snapshots."""
    counter = 0

    def fixed_time() -> float:
        nonlocal counter
        counter += 1
        return 1_700_000_000.0 + counter / 100.0

    clock = SimpleNamespace(time=fixed_time)
    old_analysis_clock, old_workflow_clock = analysis_module.time, workflow_module.time
    analysis_module.time = clock
    workflow_module.time = clock
    try:
        return analysis_module.shadow_graph(datasets)
    finally:
        analysis_module.time = old_analysis_clock
        workflow_module.time = old_workflow_clock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="private Mac Studio E5 study root")
    parser.add_argument("--out", type=Path, required=True, help="aggregate-only output directory")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    source_root = root / "code"
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "analysis"))

    from provpub.data.adapters import TraceDirectoryAdapter
    from provpub.e5.workflow import load_datasets
    from provpub.experiments.configurations import build
    from provpub.methods import operators
    from e5_attacks import node_map, size_classes
    import provpub.e5.workflow as e5_workflow
    import e5_study_analysis as e5_analysis

    subset = e5_analysis.subset

    if inspect.signature(operators.dpn).parameters["sampler"].default != "exact":
        raise RuntimeError("exact discrete-Laplace sampler is not the library default")

    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifests = []
    datasets = []
    for manifest in sorted((root / "study_inputs").glob("*.yaml")):
        doc = yaml.safe_load(manifest.read_text())
        if isinstance(doc, dict) and "datasets" in doc:
            manifests.append(manifest)
            datasets.extend(load_datasets(manifest))
    if len(datasets) != 5 or len({ds.name for ds in datasets}) != 5:
        raise ValueError("expected exactly five unique public datasets")

    latency_path = root / "results" / "e5_study" / "shadow_latency_cache.json"
    if not latency_path.is_file():
        raise FileNotFoundError("the frozen E5 shadow-latency cache is required")
    latency = json.loads(latency_path.read_text())
    shadow = build_deterministic_shadow(datasets, e5_analysis, e5_workflow)
    apply_frozen_shadow_latency(shadow, latency)

    configs = {
        "FULL-record": {"ops": ("VG", "TC", "BS", "DPN"), "delta": 60.0, "unit": "record", "eps": 1.0},
        "FULL-bucket": {"ops": ("VG", "TC", "BS", "DPN"), "delta": 60.0, "unit": "bucket", "eps": 1.0},
        "FULL-bucket-group": {
            "ops": ("VG", "TC", "BS", "DPN"), "delta": 60.0,
            "unit": "bucket", "eps": 1.0, "memo": "group",
        },
    }
    all_shadow_runs = set(map(int, shadow.runs["run"]))
    rows = []
    # A2 is reported for the regression workflow in the E5 manuscript table.
    workflows = ("regression",)
    for workflow in workflows:
        adapter = TraceDirectoryAdapter(str(root / "raw_workflows" / workflow))
        raw = adapter.load()
        spec = adapter.spec()
        labels = pd.read_csv(root / "raw_workflows" / workflow / "truth" / "runs.csv")
        labels["gid"] = [
            int(row.run) + adapter.run_offsets[str(row.run_file)]
            for row in labels.itertuples(index=False)
        ]
        live = labels[labels["mode"] == "live"].copy()
        wl = "W1" if workflow == "explore" else "W2"
        target_key = (root / "release_keys" / f"{workflow}-target.key").read_bytes()
        shadow_key = (root / "release_keys" / f"{workflow}-shadow.key").read_bytes()

        for config_name, options in configs.items():
            # Build once with the same target key and full E5 graph used by the
            # main study, then score only live traces from the held-out dataset.
            target_release = build(raw, spec, wl, options, target_key)
            inverse = {int(k): int(v) for k, v in target_release.meta.get("run_map_inv", {}).items()}
            for test_i, test_ds in enumerate(datasets):
                test_labels = live[live["dataset"] == test_ds.name].copy()
                if test_labels.empty:
                    raise ValueError(f"no live tasks for {workflow}/{test_ds.name}")
                target_gids = set(map(int, test_labels["gid"]))
                target_run_ids = [r for r in target_release.runs["run"] if inverse.get(int(r), int(r)) in target_gids]
                raw_target = subset(raw, target_gids)
                h_target = subset(target_release, target_run_ids)
                target_parts = raw_target.ent[raw_target.ent["type"] == spec.source_type]
                part_dataset = {str(part_id): test_ds.name for part_id in target_parts["part_id"].astype(str)}

                for protocol in ("known-dataset-reference", "leave-one-dataset-out"):
                    if protocol == "known-dataset-reference":
                        shadow_runs = all_shadow_runs
                        train_names = [ds.name for ds in datasets]
                    else:
                        shadow_runs = {
                            int(run)
                            for di, ds in enumerate(datasets) if ds.name != test_ds.name
                            for run in range(di * 10, di * 10 + 5)
                        }
                        train_names = [ds.name for ds in datasets if ds.name != test_ds.name]
                    raw_shadow = subset(shadow, shadow_runs)
                    h_shadow = build(raw_shadow, spec, wl, options, shadow_key)
                    source = raw_shadow.ent[raw_shadow.ent["type"] == spec.source_type].drop_duplicates("part_id")
                    thresholds = np.quantile(source["n_records"].astype(float), [1 / 3, 2 / 3])
                    if not thresholds[0] < thresholds[1]:
                        raise ValueError(f"non-distinct size thresholds in {workflow}/{test_ds.name}/{protocol}")
                    result, _ = size_classes(
                        h_target, h_shadow, raw_target, raw_shadow, spec,
                        target_key, shadow_key, thresholds, part_dataset,
                    )
                    rows.append({
                        "workflow": workflow,
                        "config": config_name,
                        "heldout_dataset": test_ds.name,
                        "protocol": protocol,
                        "training_datasets": ";".join(train_names),
                        "target_tasks": int(test_labels["gid"].nunique()),
                        "threshold_small_medium": float(thresholds[0]),
                        "threshold_medium_large": float(thresholds[1]),
                        "accuracy": result["estimate"],
                        "ci_low_source_partition_cluster": result["ci_low"],
                        "ci_high_source_partition_cluster": result["ci_high"],
                        "majority_baseline": result["baseline"],
                        "source_versions": result["clusters"],
                        "target_classes_observed": ";".join(map(str, result.get("target_classes_observed", []))),
                    })
                    print(f"{workflow} {config_name} {test_ds.name} {protocol}: {result['estimate']}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "e5_heldout_size_transfer.csv", index=False)
    summary_rows = []
    for (workflow, config, protocol), group in frame.groupby(["workflow", "config", "protocol"], sort=True):
        summary_rows.append({
            "workflow": workflow,
            "config": config,
            "protocol": protocol,
            "dataset_folds": int(group["heldout_dataset"].nunique()),
            "mean_dataset_accuracy": float(group["accuracy"].mean()),
            "min_dataset_accuracy": float(group["accuracy"].min()),
            "max_dataset_accuracy": float(group["accuracy"].max()),
            "mean_majority_baseline": float(group["majority_baseline"].mean()),
        })
    summary = {
        "status": "completed",
        "study": "E5 held-out-dataset size-class transfer",
        "method": "train on independent public shadow partitions from four datasets; evaluate on live tasks from the fifth; compare with all-five-dataset shadow reference",
        "workflows": list(workflows),
        "datasets": [ds.name for ds in datasets],
        "configs": list(configs),
        "exact_sampler": True,
        "api_calls": 0,
        "raw_traces_exported": False,
        "shadow_cache_sha256": __import__("hashlib").sha256(latency_path.read_bytes()).hexdigest(),
        "result_rows": len(frame),
        "summary": summary_rows,
        "started_unix": started,
        "completed_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
    }
    (out / "e5_heldout_size_transfer_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
