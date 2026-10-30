#!/usr/bin/env python3
"""Durable full E2/E4 reproduction: per-seed checkpoints and independent supervision.

No API credentials or network calls. E4 follows E2 so the scaling measurements
are not contending with the ten E2 jobs. Re-launching skips completed seeds.
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

import pandas as pd
import yaml


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def execute(root: Path, reference: Path, concurrency: int):
    code, python = root / "code", root / ".venv/bin/python"
    state_dir, logs = root / "state", root / "logs"
    state_dir.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    lock = (state_dir / "reference_suite.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {
        **os.environ,
        "PYTHONPATH": str(code / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    state = {"status": "running", "pid": os.getpid(), "started_unix": time.time(), "stage": "E2", "seeds": {}}
    mutex = threading.Lock()

    def update(**kw):
        with mutex:
            state.update(kw)
            state["heartbeat_unix"] = time.time()
            atomic(state_dir / "reference_suite.json", state)

    update()

    def command(args, name):
        with (logs / f"{name}.log").open("a") as output:
            child = subprocess.Popen(
                [str(python), *args], cwd=code, env=env, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
            while child.poll() is None:
                update()
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"{name} exited {child.returncode}; see its log")

    generated = root / "generated_configs"
    generated.mkdir(exist_ok=True)
    suite = root / "runs/reference_suite"
    suite.mkdir(parents=True, exist_ok=True)

    def seed_job(seed):
        out = suite / f"seed_{seed:02d}"
        metadata = out / "run_metadata.json"
        if metadata.exists() and json.loads(metadata.read_text()).get("summary", {}).get("status") == "completed":
            status = "completed_from_checkpoint"
        else:
            cfg = yaml.safe_load((code / "configs/e2.yaml").read_text())
            cfg["run"]["out_dir"] = str(out)
            cfg["stages"] = ["e2"] + (["sensitivity"] if seed < 5 else [])
            cfg["e2"]["seeds"] = [seed]
            cfg["sensitivity"]["seeds"] = [seed]
            path = generated / f"e2_seed_{seed:02d}.yaml"
            path.write_text(yaml.safe_dump(cfg))
            with mutex:
                state["seeds"][str(seed)] = "running"
            update()
            command(["scripts/run_experiment.py", "--config", str(path)], f"e2_seed_{seed:02d}")
            status = "completed"
        with mutex:
            state["seeds"][str(seed)] = status
        update()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _result in pool.map(seed_job, range(10)):
                pass
        merged = suite / "results"
        merged.mkdir(exist_ok=True)
        names = sorted({p.name for p in suite.glob("seed_*/results/e2_*.csv")})
        for name in names:
            frames = [pd.read_csv(p) for p in sorted(suite.glob(f"seed_*/results/{name}"))]
            pd.concat(frames, ignore_index=True).to_csv(merged / name, index=False)
        command(
            [
                "scripts/compare_reference.py",
                "--measured",
                str(merged),
                "--reference",
                str(reference),
                "--out",
                str(suite / "reference_comparison.json"),
            ],
            "e2_reference_comparison",
        )
        atomic(
            suite / "run_metadata.json",
            {
                "run_kind": "reference-protocol",
                "summary": {"status": "completed"},
                "protocol": "E2 seeds 0-9 plus sensitivity seeds 0-4",
            },
        )
        command(["scripts/make_tables.py", "--input", str(merged), "--output", str(suite / "tables")], "e2_tables")
        command(["scripts/make_plots.py", "--input", str(merged), "--output", str(suite / "figures")], "e2_plots")
        update(stage="E4", e2_status="completed_and_reference_matched")
        command(
            ["scripts/run_experiment.py", "--config", "configs/e4.yaml", "--out", str(root / "runs/e4_full")],
            "e4_full",
        )
        path = root / "runs/e4_full/results/e4_scaling_reference_protocol.csv"
        df = pd.read_csv(path)
        if len(df) != 60 or ("error" in df and df["error"].notna().any()):
            raise RuntimeError("E4 did not produce all 60 successful measurements")
        if (df.groupby(["workload", "target_records"])["hash"].nunique() != 1).any():
            raise RuntimeError("E4 sequential/parallel release hashes differ")
        update(status="completed", stage="done", e4_measurements=len(df), e4_hash_equality=True, ended_unix=time.time())
    except Exception as exc:
        update(status="failed", error=str(exc), ended_unix=time.time())
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    reference = args.reference.resolve()
    if args.execute:
        execute(root, reference, args.concurrency)
    else:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        with (root / "logs/reference_supervisor.log").open("a") as output:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--execute",
                    "--root",
                    str(root),
                    "--reference",
                    str(reference),
                    "--concurrency",
                    str(args.concurrency),
                ],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        print(json.dumps({"pid": child.pid, "status_file": str(root / "state/reference_suite.json")}))


if __name__ == "__main__":
    main()
