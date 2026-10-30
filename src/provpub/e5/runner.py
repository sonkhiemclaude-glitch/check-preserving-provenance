"""Checkpointed E5 collection and end-to-end verification, with mock/live separation."""
from __future__ import annotations

import hmac
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..checks.checks import run_checks
from ..data.adapters import TraceDirectoryAdapter
from ..data.prov_io import to_prov_json
from ..data.spec_io import save_spec_yaml
from ..experiments.real_data import main as evaluate
from ..utils.io import env_info, source_manifest, write_run_metadata
from ..utils.validation import validate_graph
from .provider import BedrockClient, ProviderConfig, ProviderError
from .workflow import (
    ANALYST_PROFILES,
    MockClient,
    ReplayClient,
    atomic_json,
    canonical,
    digest,
    load_datasets,
    make_spec,
    run_agent,
)

DEFAULTS: dict[str, Any] = {"live_runs": 2, "sites_per_run": 3, "max_steps": 16, "data_seed": 42, "replica_every": 1,
            "workflow": "regression", "batch_tools": False, "analyst_profiles": False,
            "private_assignments": False, "task_offset": 0,
            "operators": {"ops": ["VG", "TC", "BS", "DPN"], "delta": 60.0, "eps": 1.0,
                          "unit": "bucket", "memo": True, "sampler": "exact"}}


def load_config(path: Path | None) -> dict:
    user = yaml.safe_load(path.read_text()) if path else {}
    user = user or {}
    if not isinstance(user, dict) or set(user) - set(DEFAULTS):
        raise ValueError(f"E5 config fields must be {sorted(DEFAULTS)}; credentials come only from .env")
    cfg = {**DEFAULTS, **user, "operators": {**DEFAULTS["operators"], **user.get("operators", {})}}
    for name, lo, hi in (("live_runs", 1, 10000), ("sites_per_run", 2, 12), ("max_steps", 4, 100),
                         ("data_seed", 0, 2**31 - 1), ("replica_every", 0, 10000)):
        if not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or not lo <= cfg[name] <= hi:
            raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    if cfg["workflow"] not in ("regression", "explore"):
        raise ValueError("workflow must be regression or explore")
    if any(not isinstance(cfg[k], bool) for k in ("batch_tools", "analyst_profiles", "private_assignments")):
        raise ValueError("batch_tools, analyst_profiles and private_assignments must be booleans")
    if not isinstance(cfg["task_offset"], int) or isinstance(cfg["task_offset"], bool) or cfg["task_offset"] < 0:
        raise ValueError("task_offset must be a nonnegative integer")
    allowed = {"ops", "delta", "eps", "unit", "memo", "sampler", "max_row_bytes", "bucket_width", "clamp"}
    if set(cfg["operators"]) - allowed:
        raise ValueError("unknown E5 sanitiser option; secrets and custom keys are not allowed in YAML")
    if cfg["operators"]["sampler"] != "exact":
        raise ValueError("E5 requires the exact sampler; inversion remains available for the reference protocol")
    if set(cfg["operators"]["ops"]) != {"VG", "TC", "BS", "DPN"}:
        raise ValueError("E5 full-pipeline config must enable VG, TC, BS and DPN")
    for name in ("delta", "eps"):
        value = cfg["operators"][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("E5 delta and epsilon must be finite positive numbers")
    if cfg["operators"]["unit"] not in ("record", "bucket"):
        raise ValueError("E5 full-pipeline unit must be record or bucket")
    if cfg["operators"]["memo"] is not True and cfg["operators"]["memo"] != "group":
        raise ValueError("E5 full-pipeline requires partition or input-set memoisation")
    if "clamp" in cfg["operators"] and not isinstance(cfg["operators"]["clamp"], bool):
        raise ValueError("clamp must be boolean")
    width = cfg["operators"].get("bucket_width", [1000, 500000])
    if not isinstance(width, (list, tuple)) or len(width) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in width):
        raise ValueError("bucket_width requires two finite positive numbers")
    bound = cfg["operators"].get("max_row_bytes", 4096)
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 1:
        raise ValueError("max_row_bytes must be a positive integer")
    return cfg


@contextmanager
def _exclusive_run(out: Path):
    """OS lock is released after interruption or a killed process (macOS/Linux)."""
    import fcntl
    handle = (out / ".run.lock").open("a")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("this E5 output is already in use by another process") from None
        yield
    finally:
        handle.close()


def coverage(graph, spec) -> dict:
    """Reject vacuous/missing-content evidence before accepting a collected run."""
    problems = validate_graph(graph)
    llm = graph.act[graph.act["type"].isin(spec.nd_types)]
    generated = graph.edge[graph.edge["kind"] == "wasGeneratedBy"]
    responses = graph.ent[graph.ent["type"] == "response"].set_index("id")
    captured = set(generated[generated["src"].isin(responses.index)]["dst"])
    if llm.empty or set(llm["id"]) - captured:
        problems.append("missing response entity for a model invocation")
    for row in responses.itertuples():
        if not isinstance(row.text, str) or row.digest != digest(row.text):
            problems.append("response content or content digest is missing/inconsistent")
    for row in graph.ent[graph.ent["type"] == "prompt"].itertuples():
        if not isinstance(row.text, str) or row.digest != digest(row.text):
            problems.append("prompt content or content digest is missing/inconsistent")
    if set(graph.act["type"]) - set(spec.alphabet):
        problems.append("undeclared activity type")
    source_count = int((graph.ent["type"] == spec.source_type).sum())
    if source_count < 2:
        problems.append("no multi-site source lineage")
    used = graph.edge[graph.edge["kind"] == "used"]
    internal = set(graph.ent.loc[~graph.ent["type"].isin(spec.external_types), "id"])
    internal_uses = int(used["dst"].isin(internal).sum())
    if internal_uses == 0:
        problems.append("data-flow check would have no internal lineage")
    raw = run_checks(graph, spec)
    if (raw["outcome"] != 1).any():
        problems.append("clean collected run does not satisfy its declared specification")
    if problems:
        raise ValueError("E5 trace coverage failed: " + "; ".join(sorted(set(problems))))
    return {"llm_invocations": len(llm), "responses": len(responses), "source_partitions": source_count,
            "internal_uses": internal_uses, "check_instances": len(raw)}


def _request_audit(path: Path):
    def audit(event: dict):
        with path.open("a") as handle:
            handle.write(canonical(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return audit


def _restore_usage(client: BedrockClient, path: Path) -> None:
    if not path.exists():
        return
    successes = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A partial final journal line must never reset a previously used budget.
            raise ValueError("request journal is incomplete; review before resuming") from None
        client.requests = max(client.requests, row["attempt"])
        if row["status"] == "ok":
            successes[row["attempt"]] = row["input_tokens"] + row["output_tokens"]
    client.total_tokens = sum(successes.values())


def _release_key(out: Path, resume: bool = False, purpose: str = "release") -> Path:
    # Separate API credentials from release keys, and both from publishable artifacts.
    if purpose not in ("release", "assignment"):
        raise ValueError("unsupported key purpose")
    directory = out.parent / f".provpub-{purpose}-keys"
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    key = directory / (digest(str(out.resolve()))[:24] + ".key")
    if not key.exists():
        if resume:
            raise ValueError("release key is missing; refusing to silently create a different release")
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(os.urandom(32))
    if len(key.read_bytes()) < 16:
        raise ValueError("invalid release key file")
    key.chmod(0o600)
    return key


def run(config: dict, manifest: Path, out: Path, provider: ProviderConfig | None = None,
        mock: bool = False, resume: bool = False, client=None) -> dict:
    datasets = load_datasets(manifest.resolve())
    if any(config["sites_per_run"] > d.sites for d in datasets):
        raise ValueError("sites_per_run exceeds a dataset's simulated site count")
    if config["operators"]["unit"] == "record" and any(
            d.max_row_bytes > config["operators"].get("max_row_bytes", 4096) for d in datasets):
        raise ValueError("dataset row bytes exceed the declared record-level sensitivity bound")
    if not mock and provider is None:
        raise ValueError("live collection needs an explicit provider configuration")
    if provider:
        provider.validate(require_key=not mock)
    temperature: float | None
    if mock:
        model, temperature = "MOCK-NOT-A-LIVE-LLM", 0.0
        provider_public = {"mode": "mock"}
    else:
        assert provider is not None
        model, temperature = provider.model_id, provider.temperature
        provider_public = provider.public()
    environment = env_info()
    public = {"experiment": config, "provider": provider_public,
              "datasets": [d.identity() for d in datasets], "collection_mode": "mock" if mock else "live",
              "execution_environment": {k: environment[k] for k in
                                        ("python", "numpy", "pandas", "sklearn", "scipy", "machine", "platform", "threads")},
              "replica_protocol": "replay captured model decisions and re-execute local tools; not fresh inference"}
    scientific = json.loads(canonical(public))
    if not mock:
        # A user can explicitly raise a request/token ceiling to resume. Existing
        # usage is restored from the journal and never reset by the new ceiling.
        for name in ("max_requests", "max_total_tokens"):
            scientific["provider"].pop(name)
    fingerprint = digest(canonical(scientific))
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("E5 output exists; choose a new output or pass --resume")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    out.chmod(0o700)
    with _exclusive_run(out):
        return _run_locked(config, datasets, out, provider, mock, resume, client, public, fingerprint, model, temperature)


def _run_locked(config, datasets, out, provider, mock, resume, client, public, fingerprint, model, temperature):
    state_path = out / "collection_state.json"
    if resume:
        if not state_path.exists():
            raise ValueError("no collection checkpoint exists at the requested output")
        state = json.loads(state_path.read_text())
        if state["fingerprint"] != fingerprint or state["source_sha256"] != source_manifest():
            raise ValueError("configuration, source or dataset changed; use a new E5 output")
        for record in state["runs"].values():
            for path, expected in ((out / "raw/traces" / record["run_file"], record["trace_sha256"]),
                                   (out / "raw/captures" / f"run_{record['run']:05d}.json", record["capture_sha256"])):
                if not path.is_file() or digest(path.read_bytes()) != expected:
                    raise ValueError("completed trace or capture changed; refusing resume")
        if state["status"] == "completed":
            for relative, sha in state.get("result_sha256", {}).items():
                path = out / relative
                if not path.is_file() or digest(path.read_bytes()) != sha:
                    raise ValueError("completed evaluation artifact changed; refusing resume")
            return state["summary"]
    else:
        state = {"fingerprint": fingerprint, "source_sha256": source_manifest(), "status": "running", "completed": [],
                 "runs": {}, "provider_probe": None}
    raw = out / "raw"
    for folder in (raw / "traces", raw / "captures", raw / "truth"):
        folder.mkdir(parents=True, exist_ok=True)
    spec = make_spec(model, temperature, config["workflow"], config["batch_tools"])
    save_spec_yaml(spec, str(raw / "spec.yaml"))
    raw_manifest = {"format": "prov-json", "source": "MOCK fixture; not live LLM evidence" if mock else "E5 live API agent",
                    "capture_mode": "mock" if mock else "live", "simulated_sites": True,
                    "replica_protocol": public["replica_protocol"], "datasets": public["datasets"],
                    "nd_requires_seed": False, "notes": "All prompts, responses and used/generated lineage are captured."}
    (raw / "manifest.yaml").write_text(yaml.safe_dump(raw_manifest, sort_keys=False))
    kind = "smoke" if mock else "real-data"
    summary = {"status": "running", "live_runs": 0, "replay_runs": 0, "mock_runs": 0, "model_requests": 0}
    if not mock:
        client = client or BedrockClient(provider)
        client.audit = _request_audit(out / "requests.jsonl")
        _restore_usage(client, out / "requests.jsonl")
    key = _release_key(out, resume=resume)
    assignment = _release_key(out, resume=resume, purpose="assignment").read_bytes() if config["private_assignments"] else None
    t0 = time.monotonic()
    try:
        state["status"] = "running"
        state.pop("error_type", None)
        state.pop("provider_error", None)
        atomic_json(state_path, state)
        write_run_metadata(str(out), public, kind, {"summary": summary, "e5_config_sha256": fingerprint})
        if not mock and not state["provider_probe"]:
            state["provider_probe"] = client.probe()
            atomic_json(state_path, state)
        run_index = 0
        for index in range(config["live_runs"]):
            ds = datasets[index % len(datasets)]
            task_index = config["task_offset"] + index
            group = f"{ds.name}-{config['workflow']}-task-{task_index}"
            selection_seed = (int.from_bytes(hmac.digest(assignment, f"sites|{group}".encode(), "sha256")[:8], "big")
                              if assignment else config["data_seed"] + task_index)
            rng = __import__("numpy").random.default_rng(selection_seed)
            sites = sorted(rng.choice(ds.site_ids, config["sites_per_run"], replace=False).tolist())
            profile_index = (int.from_bytes(hmac.digest(assignment, f"profile|{group}".encode(), "sha256")[:8], "big")
                             if assignment else task_index)
            profile = list(ANALYST_PROFILES)[profile_index % len(ANALYST_PROFILES)] if config["analyst_profiles"] else None
            variants = [False]
            if config["replica_every"] and (index + 1) % config["replica_every"] == 0:
                variants.append(True)
            primary = run_index
            for replay in variants:
                run_id = run_index
                run_index += 1
                name = f"run_{run_id:05d}.prov.json"
                if run_id in state["completed"]:
                    path = raw / "traces" / name
                    record = state["runs"][str(run_id)]
                    capture = raw / "captures" / f"run_{run_id:05d}.json"
                    if not path.is_file() or digest(path.read_bytes()) != record["trace_sha256"]:
                        raise ValueError("completed trace is missing or changed; refusing resume")
                    if not capture.is_file() or digest(capture.read_bytes()) != record["capture_sha256"]:
                        raise ValueError("completed capture is missing or changed; refusing resume")
                    continue
                journal = raw / "captures" / f"run_{run_id:05d}.jsonl"
                if journal.exists():
                    journal.rename(journal.with_name(journal.stem + f".previous-{time.time_ns()}.jsonl"))
                if replay:
                    captures = json.loads((raw / "captures" / f"run_{primary:05d}.json").read_text())
                    agent = ReplayClient(captures)
                else:
                    agent = MockClient() if mock else client
                graph, detail = run_agent(ds, sites, agent, model, temperature, run_id, group,
                                          raw / "captures", config["data_seed"], config["max_steps"],
                                          config["workflow"], config["batch_tools"], profile)
                counts = coverage(graph, spec)
                if config["batch_tools"]:
                    from ..data.schemas import ranks
                    from ..methods.operators import blocks
                    probe_graph = graph.copy()
                    probe_graph.act["rank"] = ranks(probe_graph.act)
                    counts["bs_blocks"] = len(blocks(probe_graph))
                    if not counts["bs_blocks"]:
                        raise ValueError("batch workflow has no eligible BS block")
                target = raw / "traces" / name
                to_prov_json(graph, str(target.with_suffix(".tmp")))
                target.with_suffix(".tmp").replace(target)
                mode = "replay" if replay else "mock" if mock else "live"
                record = {"run": run_id, "run_file": name, "sites": ";".join(sites), "user": profile or "benchmark-analyst",
                          "workflow": config["workflow"], "profile": profile, "task_index": task_index,
                          "dataset": ds.name, "mode": mode, "group": group, "coverage": counts,
                          "steps": detail["steps"], "model_calls": detail["model_calls"] if not mock else 0,
                          "metrics": detail["metrics"], "trace_sha256": digest(target.read_bytes()),
                          "capture_sha256": digest((raw / "captures" / f"run_{run_id:05d}.json").read_bytes())}
                state["runs"][str(run_id)] = record
                state["completed"].append(run_id)
                atomic_json(state_path, state)
                print(f"E5 {mode} run {run_id} captured: {graph.n_records} PROV records", flush=True)
        records = list(state["runs"].values())
        pd.DataFrame(records).drop(columns=["coverage", "metrics"]).to_csv(out / "runs.csv", index=False)
        pd.DataFrame([{"run_file": r["run_file"], "run": r["run"], "user": r["user"], "sites": r["sites"]}
                      for r in records]).to_csv(raw / "truth/runs.csv", index=False)
        ad = TraceDirectoryAdapter(str(raw))
        combined = ad.load()
        all_checks = run_checks(combined, spec)
        replica_checks = int(all_checks["family"].str.startswith("rep_").sum())
        if any(r["mode"] == "replay" for r in records) and not replica_checks:
            raise ValueError("replica check coverage is empty")
        evaluation = evaluate(str(raw), str(out / "results"), operators=config["operators"],
                              key_file=str(key), attacks=False)
        # This protocol predicts exact preservation outside thresholds and near-limit timed checks.
        measured = pd.read_csv(out / "results/outcomes.csv")
        exact = measured[~measured["family"].isin(["thr", "time"])]
        if not exact["same"].all():
            raise ValueError("E5 nonnumeric verdict preservation failed; inspect outcomes.csv")
        summary = {"status": "completed", "live_runs": sum(r["mode"] == "live" for r in records),
                   "mock_runs": sum(r["mode"] == "mock" for r in records),
                   "replay_runs": sum(r["mode"] == "replay" for r in records),
                   "model_requests": 0 if mock else client.requests,
                   "model_tokens": 0 if mock else client.total_tokens, "replica_check_instances": replica_checks,
                   "wall_seconds_this_session": time.monotonic() - t0, "evaluation": evaluation,
                   "attacks": "not run; generic E5 attack protocols require a separate sufficient-sized population"}
        state.update(status="completed", summary=summary,
                     result_sha256={str(p.relative_to(out)): digest(p.read_bytes())
                                    for p in (out / "results").iterdir() if p.is_file()})
        atomic_json(state_path, state)
        atomic_json(out / "e5_summary.json", summary)
        write_run_metadata(str(out), public, kind, {"summary": summary, "e5_config_sha256": fingerprint})
        return summary
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        if isinstance(exc, ProviderError):
            state["provider_error"] = {"code": exc.code, "http_status": exc.status}
        atomic_json(state_path, state)
        failure = {"status": "failed", "completed_runs": len(state["completed"]), "error_type": type(exc).__name__}
        if isinstance(exc, ProviderError):
            failure["provider_error"] = exc.code
        write_run_metadata(str(out), public, kind, {"summary": failure, "e5_config_sha256": fingerprint})
        raise
