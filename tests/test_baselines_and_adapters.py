"""Baselines (specified vs stub vs dummy) and real-trace adapters."""
import json
import os

import pandas as pd
import pytest
import yaml

from provpub.baselines import (
    DavidsonModulePrivacy,
    DPEventLogRelease,
    DropAllRuns,
    IdentityRelease,
    ModuleHiding,
    ProvAbstraction,
    StructuralAbstraction,
    SuppressAll,
)
from provpub.checks.checks import align, compare, run_checks
from provpub.data.adapters import (
    FlowceptAdapter,
    TraceDirectoryAdapter,
    load_graph,
    otlp_json_to_spans,
    spans_to_otlp_json,
)
from provpub.data.prov_io import to_otel_spans, to_prov_json
from provpub.data.spec_io import save_spec_yaml
from provpub.data.synthetic import w2_generate, w2_spec
from provpub.experiments.configurations import ABSTRACT_GROUP, PRIVATE_TYPES

KEY = bytes(range(32))


def test_kinds_are_explicit():
    assert SuppressAll.kind == "baseline"
    assert IdentityRelease.kind == "dummy" and DropAllRuns.kind == "dummy"
    for stub in (DavidsonModulePrivacy, ProvAbstraction, DPEventLogRelease):
        assert stub.kind == "adapter-stub"
        with pytest.raises(NotImplementedError):
            stub().release(w2_generate(5, seed=0), w2_spec(), KEY)


def test_baselines_lose_checks_as_reported():
    g = w2_generate(80, seed=1)
    spec = w2_spec()
    raw = run_checks(g, spec)
    res = {}
    for name, mech in (("suppress", SuppressAll()), ("modhide", ModuleHiding(PRIVATE_TYPES["W2"])),
                       ("abstract", StructuralAbstraction(ABSTRACT_GROUP["W2"]))):
        h = mech.release(g, spec, KEY)
        m = compare(raw, align(h, run_checks(h, spec)))
        res[name] = m
    # suppress-all and module hiding only make checks unverifiable
    for name in ("suppress", "modhide"):
        m = res[name]
        changed = m[~m["same"]]
        assert len(changed) > 0 and (changed["outcome_san"] == -1).all()
    # abstraction against the unlifted specification produces wrong verdicts
    m = res["abstract"]
    assert ((m["outcome_raw"] == 1) & (m["outcome_san"] == 0)).any()


def test_dummy_drop_all_makes_everything_undetermined():
    g = w2_generate(10, seed=0)
    spec = w2_spec()
    m = compare(run_checks(g, spec), run_checks(DropAllRuns().release(g, spec, KEY), spec))
    assert (m["outcome_san"] == -1).all()


def test_otlp_conversion_preserves_checks():
    g = w2_generate(25, seed=4)
    spec = w2_spec()
    doc = spans_to_otlp_json(to_otel_spans(g))
    assert "resourceSpans" in doc
    g2 = load_graph_from_doc(doc)
    a = run_checks(g, spec)
    b = run_checks(g2, spec)
    assert sorted(a["outcome"].tolist()) == sorted(b["outcome"].tolist())


def load_graph_from_doc(doc):
    from provpub.data.prov_io import from_otel_spans
    return from_otel_spans(otlp_json_to_spans(doc))


def test_otlp_without_lineage_synthesises_prompt_entities():
    doc = {"resourceSpans": [{"resource": {"attributes": [{"key": "enduser.id", "value": {"stringValue": "u1"}}]},
                              "scopeSpans": [{"spans": [
        {"traceId": "aa", "spanId": "r", "name": "invoke_agent", "startTimeUnixNano": "1", "endTimeUnixNano": "9",
         "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "invoke_agent"}}]},
        {"traceId": "aa", "spanId": "c1", "parentSpanId": "r", "name": "plan", "startTimeUnixNano": "2",
         "endTimeUnixNano": "3", "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": "m1"}},
            {"key": "gen_ai.input.messages", "value": {"stringValue": "hello"}}]},
        {"traceId": "aa", "spanId": "t1", "parentSpanId": "r", "name": "load", "startTimeUnixNano": "4",
         "endTimeUnixNano": "5", "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "sql"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": "{\"path\": \"/data/x\"}"}}]}]}]}]}
    g = load_graph_from_doc(doc)
    assert list(g.act["type"]) == ["plan", "load"]
    assert set(g.ent["type"]) == {"prompt", "response"}
    assert ((g.edge["kind"] == "wasInformedBy")).sum() == 1
    assert g.runs["user"].tolist() == ["u1"]


def _make_root(tmp_path, fmt="prov-json", n_files=2):
    root = tmp_path / "real"
    (root / "traces").mkdir(parents=True)
    spec = w2_spec()
    save_spec_yaml(spec, str(root / "spec.yaml"))
    rows = []
    for i in range(n_files):
        g = w2_generate(15, seed=10 + i)
        p = root / "traces" / f"part{i}.json"
        if fmt == "prov-json":
            to_prov_json(g, str(p))
        else:
            p.write_text(json.dumps(to_otel_spans(g)))
        for r in g.runs["run"]:
            t = g.meta["truth"][int(r)]
            rows.append({"run_file": p.name, "run": int(r), "user": t["user"], "sites": ";".join(t["sites"])})
    (root / "manifest.yaml").write_text(yaml.safe_dump({"format": fmt, "source": "synthetic test fixture"}))
    (root / "truth").mkdir()
    pd.DataFrame(rows).to_csv(root / "truth" / "runs.csv", index=False)
    return root


@pytest.mark.parametrize("fmt", ["prov-json", "provpub-spans"])
def test_trace_directory_adapter_merges_files(tmp_path, fmt):
    root = _make_root(tmp_path, fmt)
    ad = TraceDirectoryAdapter(str(root))
    g = ad.load()
    assert len(g.runs) == 30 and g.runs["run"].is_unique
    assert pd.concat([g.act["id"], g.ent["id"]]).is_unique
    truth = ad.truth()
    assert set(truth) == set(g.runs["run"].astype(int))
    out = run_checks(g, ad.spec())
    assert len(out) > 0


def test_adapter_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        TraceDirectoryAdapter(str(tmp_path / "missing"))
    (tmp_path / "r").mkdir()
    with pytest.raises(FileNotFoundError):
        TraceDirectoryAdapter(str(tmp_path / "r"))
    (tmp_path / "r" / "manifest.yaml").write_text("format: csv\n")
    with pytest.raises(ValueError):
        TraceDirectoryAdapter(str(tmp_path / "r"))
    with pytest.raises(NotImplementedError):
        FlowceptAdapter("x").to_graph()
    with pytest.raises(ValueError):
        load_graph("x.json", "graphml")


def test_real_data_stage_on_fixture(tmp_path, monkeypatch):
    from provpub.experiments.real_data import main as real
    root = _make_root(tmp_path, "prov-json", n_files=1)
    monkeypatch.setenv("PPUB_TEST_KEY", "0123456789abcdef0123456789abcdef")
    out = tmp_path / "out"
    s = real(str(root), str(out), operators={"ops": ["VG", "TC", "BS", "DPN"], "delta": 60.0, "eps": 1.0,
                                              "unit": "bucket", "memo": True, "sampler": "exact"},
             key_env="PPUB_TEST_KEY", attacks=False)
    assert os.path.exists(out / "outcomes.csv") and os.path.exists(out / "release.prov.json")
    assert s["verdicts"]["fail_pass"] == 0.0 and s["verdicts"]["to_undet"] == 0.0
    with pytest.raises(ValueError):
        real(str(root), str(out), operators={}, key_file=None, key_env=None)
