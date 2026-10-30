#!/usr/bin/env python3
"""Offline, graph-level repeated-release study on the frozen E5 live traces.

The study never calls a model API. It publishes aggregate metrics only; raw
traces and release keys stay on the Mac Studio. The experiment is a controlled
version-update stress test, not evidence from production institutions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import secrets
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from provpub.checks.checks import align, compare, run_checks
from provpub.data.adapters import TraceDirectoryAdapter
from provpub.data.schemas import Graph
from provpub.methods.operators import KeyRing, dpn, sanitise


RELEASE_SIZE = 40
OVERLAPS = (0.25, 0.50, 1.00)
RELEASE_COUNTS = (2, 5)
KEY_POLICIES = ("reused", "fresh")
MEMO_POLICIES = ("partition", "input_set")
OPS = ("VG", "TC", "BS", "DPN")
EPS = 1.0
UPDATE_STEP_RECORDS = 1
UPDATE_STEP_BYTES = 128


def subset_runs(g: Graph, run_ids: set[int]) -> Graph:
    return Graph(
        g.runs[g.runs.run.isin(run_ids)].reset_index(drop=True).copy(),
        g.act[g.act.run.isin(run_ids)].reset_index(drop=True).copy(),
        g.ent[g.ent.run.isin(run_ids)].reset_index(drop=True).copy(),
        g.edge[g.edge.run.isin(run_ids)].reset_index(drop=True).copy(),
        dict(g.meta),
    )


def live_rows(root: Path, adapter: TraceDirectoryAdapter) -> pd.DataFrame:
    path = root / "truth" / "runs.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["mode"] == "live"].copy()
    frame["gid"] = [
        int(row.run) + int(adapter.run_offsets[row.run_file])
        for row in frame.itertuples()
    ]
    counts = frame.groupby("dataset").size().to_dict()
    if len(frame) != 200 or set(counts.values()) != {40}:
        raise ValueError(f"expected 200 live tasks, 40 per dataset; found {counts}")
    return frame.sort_values(["dataset", "gid"]).reset_index(drop=True)


def build_release_sets(frame: pd.DataFrame, overlap: float, n_releases: int) -> list[set[int]]:
    if overlap == 0.25:
        core_per_dataset, unique_per_release = 2, 6
    elif overlap == 0.50:
        core_per_dataset, unique_per_release = 4, 4
    elif overlap == 1.00:
        core_per_dataset, unique_per_release = 8, 0
    else:
        raise ValueError(overlap)

    releases: list[set[int]] = [set() for _ in range(n_releases)]
    for _, dataset_rows in frame.groupby("dataset", sort=True):
        ids = dataset_rows.gid.astype(int).tolist()
        core = ids[:core_per_dataset]
        pool = ids[core_per_dataset:]
        for ri in range(n_releases):
            start = ri * unique_per_release
            chosen = core + pool[start : start + unique_per_release]
            if len(chosen) != 8:
                raise ValueError("release must contain eight runs per dataset")
            releases[ri].update(chosen)
    if any(len(x) != RELEASE_SIZE for x in releases):
        raise ValueError("release size gate failed")
    expected = round(RELEASE_SIZE * overlap)
    if any(len(a & b) != expected for a, b in itertools.combinations(releases, 2)):
        raise ValueError("release overlap gate failed")
    return releases


def choose_update_partitions(g: Graph, run_dataset: dict[int, str], source_type: str) -> set[str]:
    src = g.ent[g.ent.type == source_type].copy()
    src["dataset"] = src.run.map(run_dataset)
    selected: set[str] = set()
    for dataset, group in src.groupby("dataset", sort=True):
        parts = sorted(group.part_id.astype(str).unique())
        if not parts:
            raise ValueError(f"no source partitions for {dataset}")
        selected.add(parts[0])
    if len(selected) != 5:
        raise ValueError("expected one updated public partition in each dataset")
    return selected


def versioned_copy(g: Graph, source_type: str, changed_parts: set[str], version: int) -> Graph:
    h = g.copy()
    src = h.ent.type == source_type
    changed = src & h.ent.part_id.astype(str).isin(changed_parts)
    if version:
        h.ent.loc[changed, "n_records"] = h.ent.loc[changed, "n_records"].astype(int) + version * UPDATE_STEP_RECORDS
        h.ent.loc[changed, "n_bytes"] = h.ent.loc[changed, "n_bytes"].astype(int) + version * UPDATE_STEP_BYTES
    return h


def input_set_labels(g: Graph, source_type: str) -> dict[int, str]:
    src = g.ent[g.ent.type == source_type]
    members: dict[int, list[str]] = defaultdict(list)
    for run, part in zip(src.run, src.part_id):
        members[int(run)].append(str(part))
    return {run: json.dumps(sorted(parts)) for run, parts in members.items()}


def structural_fingerprints(g: Graph) -> dict[int, str]:
    """One-hop typed graph fingerprint invariant to node and run identifiers."""
    result: dict[int, str] = {}
    run_ids = g.runs.run.astype(int).tolist()
    for run in run_ids:
        acts = g.act[g.act.run == run]
        ents = g.ent[g.ent.run == run]
        edges = g.edge[g.edge.run == run]
        labels = {int(r.id): f"A:{r.type}" for r in acts.itertuples()}
        labels.update({int(r.id): f"E:{r.type}" for r in ents.itertuples()})
        adj: dict[int, list[tuple[str, str, str]]] = {node: [] for node in labels}
        for edge in edges.itertuples():
            src, dst = int(edge.src), int(edge.dst)
            if src in labels and dst in labels:
                adj[src].append(("out", str(edge.kind), labels[dst]))
                adj[dst].append(("in", str(edge.kind), labels[src]))
        node_signatures = sorted(
            (labels[node], tuple(sorted(adj[node]))) for node in labels
        )
        payload = json.dumps(node_signatures, separators=(",", ":"), ensure_ascii=True)
        result[run] = hashlib.sha256(payload.encode()).hexdigest()
    return result


def link_metrics(graphs: list[Graph], release_sets: list[set[int]]) -> dict[str, object]:
    rows = []
    for i, j in itertools.combinations(range(len(graphs)), 2):
        a, b = graphs[i], graphs[j]
        inv_a = {int(k): int(v) for k, v in a.meta["run_map_inv"].items()}
        inv_b = {int(k): int(v) for k, v in b.meta["run_map_inv"].items()}
        true_pairs = release_sets[i] & release_sets[j]
        reverse_a = {origin: published for published, origin in inv_a.items()}
        reverse_b = {origin: published for published, origin in inv_b.items()}

        id_pairs = []
        for origin in sorted(set(reverse_a) & set(reverse_b)):
            if reverse_a[origin] == reverse_b[origin]:
                id_pairs.append((origin, origin))
        id_true = sum(x == y and x in true_pairs for x, y in id_pairs)

        fp_a, fp_b = structural_fingerprints(a), structural_fingerprints(b)
        by_fp_a: dict[str, list[int]] = defaultdict(list)
        by_fp_b: dict[str, list[int]] = defaultdict(list)
        for published, fp in fp_a.items():
            by_fp_a[fp].append(inv_a[int(published)])
        for published, fp in fp_b.items():
            by_fp_b[fp].append(inv_b[int(published)])
        fp_pairs = []
        for fp in set(by_fp_a) & set(by_fp_b):
            fp_pairs.extend(itertools.product(by_fp_a[fp], by_fp_b[fp]))
        fp_true = sum(x == y and x in true_pairs for x, y in fp_pairs)
        rows.append({
            "release_i": i,
            "release_j": j,
            "true_shared_runs": len(true_pairs),
            "id_candidates": len(id_pairs),
            "id_true_links": id_true,
            "id_precision": id_true / len(id_pairs) if id_pairs else None,
            "id_recall": id_true / len(true_pairs) if true_pairs else None,
            "fingerprint_candidates": len(fp_pairs),
            "fingerprint_true_links": fp_true,
            "fingerprint_precision": fp_true / len(fp_pairs) if fp_pairs else None,
            "fingerprint_recall": fp_true / len(true_pairs) if true_pairs else None,
        })

    def avg(field: str):
        vals = [row[field] for row in rows if row[field] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "pairwise_release_pairs": len(rows),
        "mean_true_shared_runs_per_pair": float(np.mean([r["true_shared_runs"] for r in rows])),
        "run_id_candidate_pairs": int(sum(r["id_candidates"] for r in rows)),
        "run_id_precision_mean": avg("id_precision"),
        "run_id_recall_mean": avg("id_recall"),
        "structural_candidate_pairs": int(sum(r["fingerprint_candidates"] for r in rows)),
        "structural_precision_mean": avg("fingerprint_precision"),
        "structural_recall_mean": avg("fingerprint_recall"),
        "pair_details": rows,
    }


def count_observations(raw: Graph, noisy: Graph, source_type: str) -> dict[tuple[str, str], tuple[int, int]]:
    src = raw.ent[raw.ent.type == source_type]
    out = noisy.ent.loc[src.index]
    input_sets = input_set_labels(raw, source_type)
    grouped: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for idx, row in src.iterrows():
        label = (str(row.part_id), input_sets[int(row.run)])
        grouped[label].add((int(row.n_records), int(out.at[idx, "n_records"])))
    result = {}
    for label, values in grouped.items():
        if len(values) != 1:
            raise ValueError("same partition/input-set has inconsistent counts or memoised noise")
        result[label] = next(iter(values))
    return result


def memo_multiplicity(observations: list[dict[tuple[str, str], tuple[int, int]]],
                     memo_scope: str) -> dict[str, object]:
    """Count distinct memo labels per partition in each release."""
    per_release: list[dict[str, int]] = []
    for observed in observations:
        counts: dict[str, int] = defaultdict(int)
        for pid, _input_set in observed:
            counts[pid] = 1 if memo_scope == "partition" else counts[pid] + 1
        per_release.append(dict(counts))
    totals: dict[str, int] = defaultdict(int)
    for release_counts in per_release:
        for pid, n_labels in release_counts.items():
            totals[pid] += n_labels
    return {
        "max_distinct_memo_labels_per_partition_by_release": [
            max(release.values(), default=0) for release in per_release
        ],
        "fresh_key_worst_case_basic_composition_epsilon": max(totals.values(), default=0) * EPS,
        "fresh_key_composition_is_partition_release_count_epsilon": all(
            len(set(release.values())) <= 1 and set(release.values()).issubset({1})
            for release in per_release
        ),
    }


def count_delta_metrics(observations: list[dict[tuple[str, str], tuple[int, int]]],
                        release_sets: list[set[int]],
                        changed_parts: set[str]) -> list[dict[str, object]]:
    rows = []
    for i, j in itertools.combinations(range(len(observations)), 2):
        a, b = observations[i], observations[j]
        pids_a = {pid for pid, _ in a}
        pids_b = {pid for pid, _ in b}
        for pid in pids_a & pids_b:
            a_rows = [(iset, val) for (part, iset), val in a.items() if part == pid]
            b_rows = [(iset, val) for (part, iset), val in b.items() if part == pid]
            for (set_a, (count_a, noisy_a)), (set_b, (count_b, noisy_b)) in itertools.product(a_rows, b_rows):
                rows.append({
                    "release_i": i,
                    "release_j": j,
                    "partition_updated": pid in changed_parts,
                    "input_set_unchanged": set_a == set_b,
                    "true_delta": count_b - count_a,
                    "estimated_delta": noisy_b - noisy_a,
                })
    return rows


def summarize_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame["abs_error"] = (frame.estimated_delta - frame.true_delta).abs()
    summary = []
    keys = ["partition_updated", "input_set_unchanged"]
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        updated, same_set = group_key
        summary.append({
            "partition_updated": bool(updated),
            "input_set_unchanged": bool(same_set),
            "pairs": int(len(group)),
            "exact_delta_recovery_rate": float((group.estimated_delta == group.true_delta).mean()),
            "mean_absolute_error_records": float(group.abs_error.mean()),
            "p95_absolute_error_records": float(group.abs_error.quantile(0.95)),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.audit_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    private = args.audit_root / "private"
    keys_dir = private / "release_keys"
    keys_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    keys_dir.chmod(0o700)

    adapter = TraceDirectoryAdapter(str(args.data_root))
    raw_all = adapter.load()
    spec = adapter.spec()
    frame = live_rows(args.data_root, adapter)
    run_dataset = dict(zip(frame.gid.astype(int), frame.dataset.astype(str)))
    source_type = spec.source_type
    full_source = raw_all.ent[raw_all.ent.type == source_type].copy()
    full_source["dataset"] = full_source.run.map(run_dataset)
    # Public partition roots must be stable over the task replays used here.
    if (full_source.groupby("part_id").n_records.nunique() > 1).any():
        raise ValueError("E5 has varying n_records for a stable partition identifier")
    changed_parts = choose_update_partitions(raw_all, run_dataset, source_type)

    results: list[dict[str, object]] = []
    private_key_fingerprints: dict[str, list[str]] = {}
    for overlap, n_releases, key_policy, memo_policy in itertools.product(
        OVERLAPS, RELEASE_COUNTS, KEY_POLICIES, MEMO_POLICIES
    ):
        release_sets = build_release_sets(frame, overlap, n_releases)
        memo = True if memo_policy == "partition" else "group"
        condition = f"o{int(overlap * 100)}_r{n_releases}_{key_policy}_{memo_policy}"
        print(f"START {condition}", flush=True)
        reused_key = None
        if key_policy == "reused":
            key_path = keys_dir / f"{condition}_shared.key"
            if not key_path.exists():
                key_path.write_bytes(secrets.token_bytes(32))
                key_path.chmod(0o600)
            reused_key = key_path.read_bytes()
        key_fingerprints: list[str] = []
        full_graphs: list[Graph] = []
        numeric_observations = []
        utilities = []
        for ri, selected_runs in enumerate(release_sets):
            raw = versioned_copy(subset_runs(raw_all, selected_runs), source_type, changed_parts, ri)
            if key_policy == "reused":
                key = reused_key
            else:
                key_path = keys_dir / f"{condition}_release_{ri}.key"
                if not key_path.exists():
                    key_path.write_bytes(secrets.token_bytes(32))
                    key_path.chmod(0o600)
                key = key_path.read_bytes()
            key_fingerprints.append(hashlib.sha256(key).hexdigest())
            full = sanitise(
                raw, spec, key, ops=OPS, delta=60.0, date_granularity=7 * 86400,
                eps=EPS, unit="record", memo=memo, sampler="exact",
            )
            dp_only = dpn(raw, spec, KeyRing(key), eps=EPS, unit="record", memo=memo,
                          sampler="exact", clamp=False)
            raw_checks = run_checks(raw, spec)
            sanitized_checks = align(full, run_checks(full, spec))
            comparison = compare(raw_checks, sanitized_checks)
            threshold = comparison[comparison.family == "thr"]
            utilities.append({
                "release": ri,
                "checks_total": int(len(comparison)),
                "checks_same_fraction": float(comparison.same.mean()) if len(comparison) else None,
                "threshold_checks": int(len(threshold)),
                "threshold_same_fraction": float(threshold.same.mean()) if len(threshold) else None,
            })
            full_graphs.append(full)
            numeric_observations.append(count_observations(raw, dp_only, source_type))

        link = link_metrics(full_graphs, release_sets)
        delta_rows = count_delta_metrics(numeric_observations, release_sets, changed_parts)
        accounting = memo_multiplicity(numeric_observations, memo_policy)
        if key_policy != "fresh":
            accounting["fresh_key_worst_case_basic_composition_epsilon"] = None
        private_key_fingerprints[condition] = key_fingerprints
        results.append({
            "overlap_fraction": overlap,
            "release_count": n_releases,
            "runs_per_release": RELEASE_SIZE,
            "key_policy": key_policy,
            "memo_scope": memo_policy,
            "epsilon_per_release": EPS,
            "privacy_accounting": {
                **accounting,
                "fresh_keys": "basic composition sums epsilon over distinct memo labels that expose a protected partition across releases",
                "reused_key": "no finite cumulative bound assigned when a changed count reuses a memo label; paired releases can expose the exact change",
            },
            "sampler": "exact discrete Laplace",
            "numeric_mechanism": "record-level root counts and byte sizes, epsilon split equally; no clamping",
            "changed_public_partitions": 5,
            "controlled_update_per_release": {"records_per_selected_partition": UPDATE_STEP_RECORDS,
                                                "bytes_per_selected_partition": UPDATE_STEP_BYTES},
            "linkability": link,
            "count_differencing": summarize_deltas(delta_rows),
            "count_differencing_pairs_are_correlated": True,
            "check_utility_by_release": utilities,
        })
        print(f"DONE {condition}", flush=True)

    output = {
        "status": "completed",
        "study": "E7 graph-level repeated-release stress test",
        "source": "frozen E5 explore traces; only 200 live tasks, 50 replays excluded",
        "data_scope": "five public datasets, simulated sites and analyst profiles; one LLM/model cohort",
        "api_calls": 0,
        "release_design": "40 runs per release; the same core task set gives exact 25%, 50% or 100% pairwise run overlap",
        "release_counts": list(RELEASE_COUNTS),
        "key_policies": list(KEY_POLICIES),
        "memo_scopes": list(MEMO_POLICIES),
        "operator_stack": ["exact DPN", "VG", "TC-60", "BS"],
        "updates": "one stable public partition per dataset gains one record and 128 bytes per release version; controlled metadata update",
        "interpretation_limits": [
            "Controlled offline stress test, not production deployment evidence.",
            "Structural fingerprint uses released activity/entity types and typed one-hop adjacency only.",
            "Count differencing is scored with true partition correspondence; this is an oracle-assisted attack metric, not proof that every attacker can align partitions.",
            "Pairwise count comparisons share runs and partition IDs; pairs are not independent observations.",
            "Basic composition applies to independent fresh releases; reused memoized noise is evaluated empirically and is not assigned a cumulative DP budget here.",
        ],
        "conditions": results,
    }
    out_path = args.audit_root / "e7_graph_level_results.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    out_path.chmod(0o600)

    flat = []
    for condition in results:
        link = condition["linkability"]
        for delta in condition["count_differencing"]:
            flat.append({
                "overlap_fraction": condition["overlap_fraction"],
                "release_count": condition["release_count"],
                "key_policy": condition["key_policy"],
                "memo_scope": condition["memo_scope"],
                "fresh_key_worst_case_basic_composition_epsilon": condition["privacy_accounting"]["fresh_key_worst_case_basic_composition_epsilon"],
                "max_memo_labels_per_partition_per_release": max(condition["privacy_accounting"]["max_distinct_memo_labels_per_partition_by_release"]),
                "true_shared_runs_per_pair": link["mean_true_shared_runs_per_pair"],
                "run_id_precision": link["run_id_precision_mean"],
                "run_id_recall": link["run_id_recall_mean"],
                "structural_precision": link["structural_precision_mean"],
                "structural_recall": link["structural_recall_mean"],
                **delta,
            })
    csv_path = args.audit_root / "e7_graph_level_summary.csv"
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    csv_path.chmod(0o600)
    fingerprint_path = private / "release_key_fingerprints.json"
    fingerprint_path.write_text(json.dumps(private_key_fingerprints, indent=2, sort_keys=True) + "\n")
    fingerprint_path.chmod(0o600)
    print(json.dumps({
        "status": "completed",
        "conditions": len(results),
        "json": str(out_path),
        "csv": str(csv_path),
        "source_live_runs": len(frame),
        "source_replays_excluded": 50,
        "source_records_read": int(raw_all.n_records),
        "key_files_private": str(keys_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
