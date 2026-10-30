#!/usr/bin/env python3
"""Durable E5: batch/EDA preflight, 400 live + 100 replay, then offline analysis.

A supervisor owns its children independently of SSH/chat. Credentials remain
in ../private/bedrock.env; only file paths and bounded budgets enter the plan.
The ten dataset/workflow strata have disjoint task namespaces. Additional
volume, if needed, is added equally to all strata, never selected by verdict.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def execute(root: Path, concurrency: int):
    code, python = root / "code", root / ".venv/bin/python"
    state_dir, logs = root / "state", root / "logs"
    state_dir.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    lock = (state_dir / "e5_study.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(code / "src"))
    from provpub.e5.workflow import ANALYST_PROFILES
    from provpub.utils.io import source_manifest

    public_source = source_manifest()
    credential = root / "private/bedrock.env"
    env = {
        **os.environ,
        "PYTHONPATH": str(code / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "E5_MAX_REQUESTS": "2000",
        "E5_MAX_TOTAL_TOKENS": "3000000",
        "E5_MAX_OUTPUT_TOKENS": "2048",
    }
    state = {
        "status": "running",
        "stage": "preflight",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "jobs": {},
        "source_sha256": public_source,
        "minimum_traces": 500,
        "minimum_records": 100000,
    }
    mutex = threading.Lock()

    def update(**kw):
        with mutex:
            state.update(kw)
            state["heartbeat_unix"] = time.time()
            atomic(state_dir / "e5_study.json", state)

    def job_status(name, **kw):
        with mutex:
            state["jobs"].setdefault(name, {}).update(kw)
        update()

    original = root / "data/manifest.yaml"
    datasets = yaml.safe_load(original.read_text())["datasets"]
    generated = root / "study_inputs"
    generated.mkdir(exist_ok=True)
    manifests = []
    for item in datasets:
        entry = {
            **item,
            "path": str((original.parent / item["path"]).resolve()),
            "sites": 12,
            "partition_seed": 42,
            "partition_strategy": "weighted",
        }
        p = generated / f"{item['id']}.yaml"
        p.write_text(yaml.safe_dump({"datasets": [entry]}, sort_keys=False))
        manifests.append((entry["id"], p))
    if len(manifests) != 5:
        raise ValueError("study requires exactly five provisioned datasets")
    strata = [(wf, di, name, path) for wf in ("explore", "regression") for di, (name, path) in enumerate(manifests)]
    plan = {
        "frozen_source_sha256": public_source,
        "datasets": [name for name, _ in manifests],
        "workflows": ["explore", "regression"],
        "simulated_analyst_profiles": ANALYST_PROFILES,
        "sites_per_task": 10,
        "live_tasks_per_stratum": 40,
        "replica_every": 4,
        "initial_traces": 500,
        "max_steps_per_task": 48,
        "max_output_tokens": 2048,
        "agent_context_protocol": "per-request deterministic state summary; tool schemas expose only unmet phases",
        "top_up_policy": "if measured records <100000, add 8 live + 2 replay per stratum, up to three waves",
        "attacker_known_fraction": 0.25,
        "attack_split_seeds": [0, 1, 2, 3, 4],
        "bootstrap_unit": "primary task; replay excluded",
        "candidate_dictionary": "public initial-message templates only",
        "operations": {
            "ops": ["VG", "TC", "BS", "DPN"],
            "delta": 60.0,
            "eps": 1.0,
            "unit": "record",
            "memo": True,
            "sampler": "exact",
        },
    }
    atomic(root / "study_plan.json", plan)

    def collect(phase, wf, di, name, manifest, live_runs, replica_every, offset):
        label = f"{phase}_{wf}_{name}"
        cfg = {
            "live_runs": live_runs,
            "sites_per_run": 10,
            "max_steps": 48,
            "data_seed": 42,
            "replica_every": replica_every,
            "workflow": wf,
            "batch_tools": True,
            "analyst_profiles": True,
            "private_assignments": True,
            "task_offset": offset,
            "operators": plan["operations"],
        }
        config = generated / f"{label}.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        out = root / "runs" / phase / f"{wf}_{name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while attempt < 3:
            if source_manifest() != public_source:
                raise RuntimeError("frozen collector source changed")
            cmd = [
                str(python),
                "scripts/run_e5.py",
                "--env",
                str(credential),
                "--config",
                str(config),
                "--datasets",
                str(manifest),
                "--out",
                str(out),
            ]
            if (out / "collection_state.json").exists():
                cmd += ["--resume"]
            before = credential.stat().st_mtime_ns
            with (logs / f"{label}.log").open("a") as log:
                child = subprocess.Popen(cmd, cwd=code, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                job_status(label, status="running", pid=child.pid, attempt=attempt + 1, output=str(out))
                while child.poll() is None:
                    cp = out / "collection_state.json"
                    if cp.exists():
                        try:
                            completed = len(json.loads(cp.read_text())["completed"])
                        except (ValueError, KeyError):
                            completed = None
                        job_status(label, completed_traces=completed)
                    time.sleep(3)
            if child.returncode == 0:
                summary = json.loads((out / "e5_summary.json").read_text())
                job_status(label, status="completed", summary=summary)
                return out
            cp = out / "collection_state.json"
            failure = json.loads(cp.read_text()) if cp.exists() else {}
            error = failure.get("provider_error", {})
            if error.get("http_status") in (401, 403):
                job_status(label, status="waiting_credentials", http_status=error["http_status"])
                while credential.stat().st_mtime_ns == before:
                    update()
                    time.sleep(10)
                # Short-term keys expire mid-study; a refresh is not a failed attempt.
                continue
            if error.get("code") == "request_or_token_budget_exhausted":
                raise RuntimeError(f"{label}: budget exhausted; usage was preserved")
            elif child.returncode != 3:
                job_status(label, status="failed", exit_code=child.returncode, error_type=failure.get("error_type"))
                raise RuntimeError(f"{label}: non-provider failure, see log and checkpoint")
            attempt += 1
            time.sleep(2)
        raise RuntimeError(f"{label}: three bounded attempts failed; no model fallback used")

    def wave(phase, live_runs, replica_every, extra_offset):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    collect,
                    phase,
                    wf,
                    di,
                    name,
                    path,
                    live_runs,
                    replica_every,
                    extra_offset + (0 if wf == "explore" else 100000) + 1000 * di,
                )
                for wf, di, name, path in strata
            ]
            return [f.result() for f in futures]

    try:
        update()
        wave("preflight", 1, 1, 900000)
        update(stage="main", preflight_status="completed")
        cohorts = wave("main", 40, 4, 0)

        def volume():
            ss = [json.loads((p / "e5_summary.json").read_text()) for p in cohorts]
            return sum(s["live_runs"] + s["replay_runs"] for s in ss), sum(s["evaluation"]["prov_records"] for s in ss)

        traces, records = volume()
        for n in range(3):
            if records >= 100000:
                break
            update(stage=f"volume_topup_{n}", traces=traces, prov_records=records)
            cohorts += wave(f"topup_{n}", 8, 4, 40 + 8 * n)
            traces, records = volume()
        if traces < 500 or records < 100000:
            raise RuntimeError("minimum study volume was not attained")
        atomic(root / "e5_cohorts.json", {"cohorts": [str(p) for p in cohorts], "traces": traces, "prov_records": records})
        update(stage="offline_analysis", traces=traces, prov_records=records)
        analysis = code / "analysis/e5_study_analysis.py"
        deadline = time.monotonic() + 3600
        while not analysis.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("analysis executable not provisioned")
            update()
            time.sleep(10)
        with (logs / "e5_analysis.log").open("a") as log:
            child = subprocess.Popen(
                [str(python), str(analysis), "--root", str(root)],
                cwd=code,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while child.poll() is None:
                update()
                time.sleep(3)
            if child.returncode:
                raise RuntimeError("offline analysis failed; see e5_analysis.log")
        receipt = json.loads((root / "results/e5_study/receipt.json").read_text())
        if receipt.get("status") != "completed":
            raise RuntimeError("analysis did not issue a completion receipt")
        update(
            status="completed", stage="done", ended_unix=time.time(), analysis_receipt=str(root / "results/e5_study/receipt.json")
        )
    except Exception as exc:
        update(status="failed", error=str(exc), ended_unix=time.time())
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    if a.execute:
        execute(root, a.concurrency)
    else:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        with (root / "logs/e5_supervisor.log").open("a") as log:
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--root",
                    str(root),
                    "--concurrency",
                    str(a.concurrency),
                    "--execute",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"pid": p.pid, "status_file": str(root / "state/e5_study.json")}))


if __name__ == "__main__":
    main()
