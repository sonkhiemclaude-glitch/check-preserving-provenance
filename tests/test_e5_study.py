"""Batched real tools expose nonvacuous shuffle blocks, EDA results and replay."""

import numpy as np
import pandas as pd
import pytest
import yaml

from provpub.checks.checks import run_checks
from provpub.data.schemas import ranks
from provpub.e5.runner import load_config, run
from provpub.e5.workflow import Dataset, MockClient, ReplayClient, make_spec, run_agent
from provpub.methods.operators import blocks


def frame_manifest(root):
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(360, 5)), columns=["a", "b", "c", "d", "target"])
    df.to_csv(root / "data.csv", index=False)
    manifest = root / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "id": "fixture",
                        "path": "data.csv",
                        "target": "target",
                        "source": "offline test fixture",
                        "sites": 12,
                        "partition_seed": 42,
                        "partition_strategy": "weighted",
                    }
                ]
            }
        )
    )
    return manifest


def test_weighted_partition_versions_are_disjoint_and_repeatable(tmp_path):
    frame_manifest(tmp_path)
    a = Dataset("x", tmp_path / "data.csv", "target", "fixture", 12, 42, "weighted")
    b = Dataset("x", tmp_path / "data.csv", "target", "fixture", 12, 42, "weighted")
    assert min(map(len, a.indices)) >= 12 and len({len(x) for x in a.indices}) > 2
    assert sorted(np.concatenate(a.indices)) == list(range(360))
    assert all(np.array_equal(x, y) for x, y in zip(a.indices, b.indices))


@pytest.mark.parametrize("workflow", ["regression", "explore"])
def test_batch_workflow_has_actual_shuffle_blocks_and_replays(tmp_path, workflow):
    frame_manifest(tmp_path)
    ds = Dataset("fixture", tmp_path / "data.csv", "target", "fixture", 12, 42, "weighted")
    captures = tmp_path / "capture"
    captures.mkdir()
    graph, result = run_agent(
        ds,
        ds.site_ids[:10],
        MockClient(),
        "MOCK",
        0.0,
        0,
        "g",
        captures,
        workflow=workflow,
        batch_tools=True,
        profile="profile-2",
    )
    assert (run_checks(graph, make_spec("MOCK", 0.0, workflow, True))["outcome"] == 1).all()
    graph.act["rank"] = ranks(graph.act)
    candidates = blocks(graph)
    assert len(candidates) >= 2 and sum(len(x) for _, x in candidates) >= 20
    assert result["metrics"] and graph.n_records >= 200
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay, rr = run_agent(
        ds,
        ds.site_ids[:10],
        ReplayClient(result["captures"]),
        "MOCK",
        0.0,
        1,
        "g",
        replay_dir,
        workflow=workflow,
        batch_tools=True,
        profile="profile-2",
    )
    assert rr["metrics"] == result["metrics"] and replay.n_records == graph.n_records


@pytest.mark.parametrize("workflow", ["regression", "explore"])
def test_new_cohort_runner_offline(tmp_path, workflow):
    manifest = frame_manifest(tmp_path / "input")
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "live_runs": 1,
                "sites_per_run": 10,
                "max_steps": 20,
                "replica_every": 1,
                "workflow": workflow,
                "batch_tools": True,
                "analyst_profiles": True,
                "task_offset": 1000,
            }
        )
    )
    result = run(load_config(path), manifest, tmp_path / "out", mock=True)
    assert result["status"] == "completed" and result["mock_runs"] == 1 and result["replay_runs"] == 1
    assert result["evaluation"]["prov_records"] >= 400 and result["replica_check_instances"] == 3


def test_rejected_early_profile_is_not_a_completed_profile(tmp_path):
    from provpub.e5.workflow import Tools, Trace

    frame_manifest(tmp_path)
    ds = Dataset("fixture", tmp_path / "data.csv", "target", "fixture", 12, 42, "weighted")
    trace = Trace(0, "g", "profile-0")
    prompt = trace.entity("prompt", "test input")
    decision = trace.activity("agent_step", 0.0, 1.0, [prompt], [])
    tools = Tools(ds, ds.site_ids[:10], trace, 42, "explore", True)
    tools.execute("load_partitions", {"sites": ds.site_ids[:10]}, prompt, decision)
    tools.execute("validate_partitions", {"sites": ds.site_ids[:10]}, prompt, decision)
    tools.execute("harmonise", {}, prompt, decision)
    rejected, _ = tools.execute("profile_features", {"features": ds.features[:3]}, prompt, decision)
    assert not rejected["ok"] and (trace.graph().act.type == "profile_feature").sum() == 0
    assert (trace.graph().act.type == "tool_call_rejected").sum() == 1
    tools.execute("describe_data", {}, prompt, decision)
    ok, _ = tools.execute("profile_features", {"features": ds.features[:3]}, prompt, decision)
    assert ok["ok"] and tools.finished and len(tools.profiles) == 3
    assert (trace.graph().act.type == "profile_feature").sum() == 3


def test_assignment_seed_is_private_separate_and_stable_on_resume(tmp_path):
    import json
    import stat

    manifest = frame_manifest(tmp_path / "input")
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "live_runs": 2,
                "sites_per_run": 10,
                "max_steps": 20,
                "replica_every": 1,
                "workflow": "explore",
                "batch_tools": True,
                "analyst_profiles": True,
                "private_assignments": True,
            }
        )
    )
    cfg = load_config(path)
    out = tmp_path / "out"
    run(cfg, manifest, out, mock=True)
    key = next((tmp_path / ".provpub-assignment-keys").glob("*.key"))
    release = next((tmp_path / ".provpub-release-keys").glob("*.key"))
    assert key.read_bytes() != release.read_bytes() and stat.S_IMODE(key.stat().st_mode) == 0o600
    state = json.loads((out / "collection_state.json").read_text())
    assert all(key.read_bytes() not in p.read_bytes() for p in out.rglob("*") if p.is_file())
    run(cfg, manifest, out, mock=True, resume=True)
    assert json.loads((out / "collection_state.json").read_text())["runs"] == state["runs"]
    assert state["runs"]["0"]["user"] == state["runs"]["1"]["user"]
    assert state["runs"]["2"]["user"] == state["runs"]["3"]["user"]
