"""E4: sanitisation time and peak memory vs. PROV records and workers
.

Each measurement runs in a fresh Python process (worker mode) so that peak
memory is per measurement. Shards are sets of whole runs; all operator
randomness derives from the release key, so the parallel release equals the
sequential one, which is checked by hashing canonical tables.

Reference protocol: W2 and W3; 1e5, 2.5e5, 5e5, 1e6, 2e6 records; 1 and 2 workers;
3 repetitions; full composition, bucket level, eps=1, Delta=60 s; data seed 42
(configs/e4.yaml). The extended grid (more workers, larger sizes)
is configs/e4_extended.yaml.

Memory: Linux uses VmHWM from /proc/self/status (reset by exec). macOS has no
/proc; there ru_maxrss of the fresh worker process is used (bytes on macOS).
The method is recorded in column ``mem_method``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import platform
import resource
import subprocess
import sys
import time

import pandas as pd

log = logging.getLogger(__name__)

REC_PER_RUN = {"W2": 97.0, "W3": 250.0}
KEY = b"e4-release-key-0123456789abcdef!"   # public experiment key of E4
DATA_SEED = 42


def data_path(wl: str, size: float, cache: str) -> str:
    return os.path.join(cache, f"{wl}_{int(size)}.pkl")


def ensure_data(wl: str, size: float, cache: str) -> str:
    from ..data.synthetic import GENERATORS
    p = data_path(wl, size, cache)
    if not os.path.exists(p):
        gen, _ = GENERATORS[wl]
        g = gen(int(round(size / REC_PER_RUN[wl])), seed=DATA_SEED)
        with open(p, "wb") as f:
            pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def peak_rss_mb() -> tuple[float, str]:
    """Peak resident set size of this process in MB, and the method used."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0, "VmHWM"
    except OSError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return r / (1024.0 * 1024.0), "ru_maxrss_bytes"
    return r / 1024.0, "ru_maxrss_kb"


def shard_path(wl, size, n, i, cache, kind="in"):
    return os.path.join(cache, f"{wl}_{int(size)}_{kind}_{n}_{i}.pkl")


def ensure_shards(wl, size, n, cache):
    from ..data.schemas import split_by_runs
    if n == 1 or os.path.exists(shard_path(wl, size, n, n - 1, cache)):
        return
    with open(data_path(wl, size, cache), "rb") as f:
        g = pickle.load(f)
    for i, part in enumerate(split_by_runs(g, n)):
        with open(shard_path(wl, size, n, i, cache), "wb") as f:
            pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)


def _shard_job(args):
    """Load one shard, sanitise it, write the released shard (same I/O as the sequential path)."""
    wl, size, n, i, cache = args
    from ..data.synthetic import GENERATORS
    from ..methods import operators as O
    with open(shard_path(wl, size, n, i, cache), "rb") as f:
        part = pickle.load(f)
    spec = GENERATORS[wl][1]()
    h = O.sanitise(part, spec, KEY, delta=60.0, unit="bucket", eps=1.0)
    with open(shard_path(wl, size, n, i, cache, "out"), "wb") as f:
        pickle.dump(h, f, protocol=pickle.HIGHEST_PROTOCOL)
    return peak_rss_mb()[0]


def canon_hash(h) -> str:
    """SHA-256 of the canonical tables (runs, activities, entities, relations)."""
    m = hashlib.sha256()
    for tab, key in (("runs", "run"), ("act", "id"), ("ent", "id")):
        df = getattr(h, tab).sort_values(key).reset_index(drop=True)
        m.update(df.astype(str).to_csv(index=False).encode())
    e = h.edge.sort_values(["kind", "src", "dst"]).reset_index(drop=True)
    m.update(e.astype(str).to_csv(index=False).encode())
    return m.hexdigest()


def worker(wl: str, size: float, workers: int, cache: str, export: bool) -> dict:
    """One measurement (runs inside a fresh process)."""
    from multiprocessing import get_context

    from ..checks.checks import run_checks
    from ..data.prov_io import to_prov_json
    from ..data.schemas import concat_graphs
    from ..data.synthetic import GENERATORS
    from ..methods import operators as O
    spec = GENERATORS[wl][1]()
    out: dict = {"workload": wl, "target_records": size, "workers": workers}
    t0 = time.perf_counter()
    if workers == 1:
        with open(data_path(wl, size, cache), "rb") as f:
            g = pickle.load(f)
        ring = O.KeyRing(KEY)
        ts = {"load": time.perf_counter() - t0}
        t = time.perf_counter(); h = O.dpn(g, spec, ring, 1.0, unit="bucket"); ts["DPN"] = time.perf_counter() - t
        t = time.perf_counter(); h = O.vg(h, spec, ring); ts["VG"] = time.perf_counter() - t
        t = time.perf_counter(); h = O.tc(h, 60.0, 7 * 86400); ts["TC"] = time.perf_counter() - t
        t = time.perf_counter(); h = O.bs(h, ring); ts["BS"] = time.perf_counter() - t
        with open(os.path.join(cache, f"seq_out_{wl}_{int(size)}.pkl"), "wb") as f:
            pickle.dump(h, f, protocol=pickle.HIGHEST_PROTOCOL)
        out.update({f"t_{k}": v for k, v in ts.items()})
        out.update(records=g.n_records, runs=len(g.runs), nodes=g.n_nodes)
    else:
        method = "spawn" if platform.system() == "Darwin" else "fork"
        out["start_method"] = method
        with get_context(method).Pool(workers) as pool:
            child_mem = pool.map(_shard_job, [(wl, size, workers, i, cache) for i in range(workers)])
        out["peak_rss_mb_worker_max"] = max(child_mem)
    out["t_sanitise"] = time.perf_counter() - t0
    out["peak_rss_mb_sanitise"], out["mem_method"] = peak_rss_mb()
    if workers > 1:
        parts = []
        for i in range(workers):
            with open(shard_path(wl, size, workers, i, cache, "out"), "rb") as f:
                parts.append(pickle.load(f))
        h = concat_graphs(parts)
        out.update(records=h.n_records, runs=len(h.runs), nodes=h.n_nodes)
    out["hash"] = canon_hash(h)
    if workers == 1:
        t = time.perf_counter(); run_checks(h, spec); out["t_verify_release"] = time.perf_counter() - t
        if export:
            p = os.path.join(cache, f"release_{wl}_{int(size)}.json")
            t = time.perf_counter(); out["export_bytes"] = to_prov_json(h, p); out["t_export"] = time.perf_counter() - t
            os.remove(p)
    return out


def main(sizes=(1e5, 2.5e5, 5e5, 1e6, 2e6), workers=(1, 2), workloads=("W2", "W3"), reps: int = 3,
         out_dir: str = "outputs/e4", tag: str = "local", cache: str | None = None) -> dict:
    """Run E4; writes e4_scaling_<tag>.csv. ``cache`` holds generated data and shards
    (several GB at 2e6 records); it defaults to <out_dir>/cache and can be deleted afterwards."""
    cache = cache or os.path.join(out_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    rows: list[dict] = []
    csv = os.path.join(out_dir, f"e4_scaling_{tag}.csv")
    # the worker runs in a fresh interpreter; make the package importable there
    # even when it is used from a source checkout without installation
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([pkg_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for wl in workloads:
        for size in sizes:
            ensure_data(wl, size, cache)
            for w in workers:
                ensure_shards(wl, size, w, cache)
            for rep in range(reps):
                for w in workers:
                    cmd = [sys.executable, "-m", "provpub.experiments.e4", "--worker", wl, str(size), str(w),
                           "1" if (rep == 0 and w == 1) else "0", "--cache", cache]
                    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    if res.returncode != 0:
                        log.error("E4 worker failed: %s", res.stderr[-2000:])
                        rows.append({"workload": wl, "target_records": size, "workers": w, "rep": rep,
                                     "error": res.stderr[-300:]})
                        pd.DataFrame(rows).to_csv(csv, index=False)
                        continue
                    r = json.loads(res.stdout.strip().splitlines()[-1])
                    r["rep"] = rep
                    rows.append(r)
                    log.info("E4 %s %.0f workers=%d rep=%d %.2fs", wl, size, w, rep, r["t_sanitise"])
                    pd.DataFrame(rows).to_csv(csv, index=False)
            hashes = {r["hash"] for r in rows if r.get("workload") == wl
                      and r.get("target_records") == size and "hash" in r}
            if len(hashes) > 1:
                raise RuntimeError(f"E4 release hash mismatch for {wl} at {size:g} records; see {csv}")
    failed = sum(1 for r in rows if "error" in r)
    if failed:
        raise RuntimeError(f"{failed} E4 measurements failed; partial results saved in {csv}")
    return {"sizes": list(sizes), "workers": list(workers), "workloads": list(workloads), "reps": reps, "tag": tag,
            "failed_measurements": failed, "hash_equality": True}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="E4 worker mode (internal)")
    ap.add_argument("--worker", nargs=4, metavar=("WL", "SIZE", "WORKERS", "EXPORT"), required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    wl, size, w, ex = a.worker
    print(json.dumps(worker(wl, float(size), int(w), a.cache, ex == "1")))


if __name__ == "__main__":
    _cli()
