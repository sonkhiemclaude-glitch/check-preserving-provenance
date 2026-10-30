"""Provider failures, honest trace capture, replay and full offline E5 evaluation."""
import json
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import pytest
import yaml

from provpub.checks.checks import run_checks
from provpub.data.spec_io import spec_from_dict, spec_to_dict
from provpub.e5.provider import BedrockClient, ProviderConfig, ProviderError, _converse_payload, read_env
from provpub.e5.runner import coverage, load_config, run
from provpub.e5.workflow import Dataset, MockClient, ReplayClient, Tools, Trace, load_datasets, make_spec, run_agent

TOKEN = "test-secret-do-not-log"


def provider(**kw):
    cfg = ProviderConfig("gpt-4.1-mini", "https://bedrock-mantle.us-east-1.api.aws/v1", api_key=TOKEN)
    return replace(cfg, **kw)


def reply(name="provenance_probe", arguments=None):
    return {"model": "test-model-snapshot", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {"value": 7})}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3}}


def test_env_is_data_and_credentials_are_redacted(tmp_path):
    f = tmp_path / ".env"
    f.write_text('Bedrock_API="test-secret-do-not-log"\nCUSTOM="$(touch NO_SUCH_FILE)"\n')
    env = read_env(f)
    assert env["CUSTOM"] == "$(touch NO_SUCH_FILE)" and not (tmp_path / "NO_SUCH_FILE").exists()
    cfg = ProviderConfig.from_env(env)
    cfg.validate()
    assert TOKEN not in repr(cfg) and TOKEN not in json.dumps(cfg.public())
    f.write_text('X=1\nX=2\n')
    with pytest.raises(ValueError, match="duplicate"):
        read_env(f)


@pytest.mark.parametrize("base", ["http://example.com/v1", "https://u:p@example.com/v1", "https://x.com/v1?k=secret"])
def test_unsafe_endpoint_rejected(base):
    with pytest.raises(ValueError):
        provider(base_url=base).validate()


def test_gateway_requires_explicit_mode_and_no_openai_credential_fallback():
    with pytest.raises(ValueError, match="AWS endpoint"):
        provider(base_url="https://gateway.example/v1").validate()
    provider(base_url="https://gateway.example/v1", api_mode="openai-compatible").validate()
    with pytest.raises(ValueError, match="API_KEY|Bedrock_API"):
        ProviderConfig.from_env({"OPENAI_API_KEY": TOKEN}).validate()


def test_chat_probe_retry_budget_and_frozen_model():
    observed = []
    def transport(url, payload, key, timeout):
        observed.append((url, payload))
        assert key == TOKEN and TOKEN not in json.dumps(payload)
        if len(observed) == 1:
            raise HTTPError(url, 429, TOKEN, {"Retry-After": "0"}, None)
        return reply()
    client = BedrockClient(provider(), transport=transport, sleep=lambda _: None)
    result = client.probe()
    assert result["tool_calling"] and client.requests == 2 and client.total_tokens == 8
    assert {p["model"] for _, p in observed} == {"gpt-4.1-mini"}
    assert TOKEN not in json.dumps(client.events)
    assert observed[0][0].endswith('/v1/chat/completions')
    client.config = replace(client.config, max_requests=2)
    with pytest.raises(ProviderError, match="budget_exhausted"):
        client.probe()


@pytest.mark.parametrize("error", [401, 403, 404])
def test_access_and_model_failures_do_not_retry_or_leak(error):
    count = []
    def transport(*args):
        count.append(1)
        raise HTTPError(args[0], error, TOKEN, {}, None)
    client = BedrockClient(provider(), transport=transport)
    with pytest.raises(ProviderError) as exc:
        client.probe()
    assert len(count) == 1 and TOKEN not in str(exc.value)


def test_network_error_does_not_leak_credentials():
    def transport(*args):
        raise URLError(TOKEN)
    with pytest.raises(ProviderError, match="network_error") as exc:
        BedrockClient(provider(max_retries=0), transport=transport).probe()
    assert TOKEN not in str(exc.value)


def test_converse_tool_use_and_tool_result_mapping():
    cfg = provider(api_mode="bedrock-converse", base_url="https://bedrock-runtime.us-east-1.amazonaws.com")
    tool = {"type": "function", "function": {"name": "x", "description": "x", "parameters": {"type": "object"}}}
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {
                    "name": "x", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c", "content": "done"}]
    body = _converse_payload(messages, [tool], cfg)
    assert body["messages"][-1]["content"][0]["toolResult"]["toolUseId"] == "c"
    def transport(url, payload, key, timeout):
        assert url.endswith('/model/gpt-4.1-mini/converse')
        return {"stopReason": "tool_use", "output": {"message": {"content": [{"toolUse": {
            "toolUseId": "a", "name": "provenance_probe", "input": {"value": 7}}}]}},
            "usage": {"inputTokens": 5, "outputTokens": 3}}
    assert BedrockClient(cfg, transport=transport).probe()["tool_calling"]


def test_probe_refuses_text_without_tool_call():
    with pytest.raises(ProviderError, match="probe_failed"):
        BedrockClient(provider(), transport=lambda *a: {"choices": [{"message": {"content": "OK"}}]}).probe()


def dataset_manifest(tmp_path: Path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=192)
    pd.DataFrame({"x": x, "z": rng.normal(size=192), "target": 2 * x + rng.normal(size=192)}).to_csv(
        tmp_path / "fixture.csv", index=False)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"datasets": [{"id": "fixture", "path": "fixture.csv", "target": "target",
                                                       "source": "offline unit-test fixture", "sites": 8}]}))
    return manifest


def test_e5_mock_capture_replay_sanitise_and_resume(tmp_path):
    manifest = dataset_manifest(tmp_path)
    cfg = load_config(None)
    cfg["live_runs"] = 1
    out = tmp_path / "output"
    result = run(cfg, manifest, out, mock=True)
    assert result["status"] == "completed" and result["mock_runs"] == 1 and result["replay_runs"] == 1
    assert result["live_runs"] == 0 and result["model_requests"] == 0
    assert result["replica_check_instances"] == 3
    metadata = json.loads((out / "run_metadata.json").read_text())
    assert metadata["run_kind"] == "smoke" and metadata["summary"]["status"] == "completed"
    assert (out / "LABEL.txt").exists()
    captures = json.loads((out / "raw/captures/run_00000.json").read_text())
    assert all(r["mode"] == "mock" for r in captures)
    checkpoints = json.loads((out / "collection_state.json").read_text())
    assert checkpoints["completed"] == [0, 1]
    measured = pd.read_csv(out / "results/outcomes.csv")
    assert measured.loc[~measured.family.isin(['thr', 'time']), "same"].all()
    assert run(cfg, manifest, out, mock=True, resume=True) == result
    assert not any(out.rglob('*.key')) and not any(out.rglob('.env'))
    with pytest.raises(FileExistsError):
        run(cfg, manifest, out, mock=True)
    changed = {**cfg, "sites_per_run": 2}
    with pytest.raises(ValueError, match="changed"):
        run(changed, manifest, out, mock=True, resume=True)


def test_hosted_spec_declares_seed_absence_and_checks_real_response(tmp_path):
    manifest = dataset_manifest(tmp_path)
    ds = Dataset("fixture", tmp_path / "fixture.csv", "target", "test fixture")
    capture = tmp_path / "capture"
    capture.mkdir()
    g, detail = run_agent(ds, ds.site_ids[:3], MockClient(), "mock", 0.0, 0, "g", capture)
    spec = make_spec("mock", 0.0)
    assert not spec.nd_requires_seed and not spec_from_dict(spec_to_dict(spec)).nd_requires_seed
    assert g.act[g.act.type == 'agent_step']["seed"].isna().all()
    assert coverage(g, spec)["responses"] > 0
    g.ent.loc[g.ent.type == "response", "digest"] = None
    with pytest.raises(ValueError, match="response"):
        coverage(g, spec)
    assert (run_checks(g, spec).query("family == 'nd'")["outcome"] == 0).all()
    replay = ReplayClient(detail["captures"])
    with pytest.raises(ProviderError, match="replay_prompt"):
        replay.complete([{"role": "user", "content": "changed"}], [])
    assert manifest.exists()


def test_scripted_spec_still_requires_seed():
    from provpub.data.synthetic import w2_spec
    assert w2_spec().nd_requires_seed


def test_resuming_changed_capture_is_rejected(tmp_path):
    manifest = dataset_manifest(tmp_path)
    cfg = {**load_config(None), "live_runs": 1}
    out = tmp_path / "out"
    run(cfg, manifest, out, mock=True)
    state = json.loads((out / 'collection_state.json').read_text())
    state['status'] = 'failed'
    (out / 'collection_state.json').write_text(json.dumps(state))
    (out / 'raw/captures/run_00000.json').write_text('[]')
    with pytest.raises(ValueError, match='capture'):
        run(cfg, manifest, out, mock=True, resume=True)


def test_missing_otlp_response_is_not_given_a_fabricated_digest():
    from provpub.data.adapters import otlp_json_to_spans
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [{"traceId": "t", "spanId": "s",
           "name": "agent_step", "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
           {"key": "gen_ai.input.messages", "value": {"stringValue": "actual prompt"}}]}]}]}]}
    spans = otlp_json_to_spans(doc)
    response = spans[0]["attributes"]["ppub.generated"][0]
    assert response["text"] is None and response["digest"] is None


def test_live_transport_fixture_checkpoint_resume_preserves_request_budget(tmp_path):
    manifest = dataset_manifest(tmp_path)
    cfg = {**load_config(None), "live_runs": 1}
    counts = []
    fixture = MockClient()
    def transport(url, payload, key, timeout):
        counts.append(1)
        if len(counts) > 3:
            checkpoint = json.loads((out / 'collection_state.json').read_text())
            assert checkpoint['status'] == 'running' and 'provider_error' not in checkpoint
        if [tool["function"]["name"] for tool in payload["tools"]] == ["provenance_probe"]:
            return reply()
        if len(counts) == 3:
            raise HTTPError(url, 503, "temporary", {}, None)
        result = fixture.complete(payload["messages"], payload["tools"])
        return {"model": "fixture-only-not-real-inference", "choices": [{"message": result.message}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    pcfg = provider(max_retries=0)
    out = tmp_path / 'live-fixture'
    with pytest.raises(ProviderError):
        run(cfg, manifest, out, provider=pcfg, client=BedrockClient(pcfg, transport=transport))
    assert json.loads((out / 'collection_state.json').read_text())['status'] == 'failed'
    first_requests = len(counts)
    fixture = MockClient()
    result = run(cfg, manifest, out, provider=pcfg, resume=True,
                 client=BedrockClient(pcfg, transport=transport))
    assert result['status'] == 'completed' and result['live_runs'] == 1 and result['replay_runs'] == 1
    assert result['model_requests'] == len(counts) and len(counts) > first_requests
    assert len(list((out / 'raw/captures').glob('*.previous-*.jsonl'))) == 1
    assert TOKEN not in (out / 'run_metadata.json').read_text()
    assert TOKEN not in (out / 'requests.jsonl').read_text()


def test_reasoning_model_uses_recorded_compatible_parameters():
    cfg = ProviderConfig.from_env({'Bedrock_API': TOKEN, 'BEDROCK_MODEL_ID': 'openai.gpt-5.6-luna'})
    observed = []
    def transport(url, payload, key, timeout):
        observed.append(payload)
        return reply()
    client = BedrockClient(cfg, transport=transport)
    client.probe()
    assert cfg.temperature is None and cfg.reasoning_effort == 'none'
    assert 'temperature' not in observed[0] and 'seed' not in observed[0]
    assert observed[0]['max_completion_tokens'] == 1024 and 'max_tokens' not in observed[0]
    assert observed[0]['model'] == 'openai.gpt-5.6-luna'


def test_affordable_default_uses_oss_route_and_low_reasoning():
    cfg = ProviderConfig.from_env({'Bedrock_API': TOKEN})
    observed = []
    def transport(url, payload, key, timeout):
        observed.append((url, payload))
        return reply()
    BedrockClient(cfg, transport=transport).probe()
    url, payload = observed[0]
    assert cfg.model_id == 'openai.gpt-oss-120b'
    assert url == 'https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions'
    assert payload['reasoning_effort'] == 'low' and payload['temperature'] == 0
    assert payload['max_completion_tokens'] == 1024 and 'max_tokens' not in payload


def test_malformed_response_is_captured_and_charged_to_token_budget():
    raw = reply('evaluate_model<|channel|>commentary')
    raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = []
    client = BedrockClient(provider(), transport=lambda *args: raw)
    with pytest.raises(ProviderError, match='invalid_tool_call'):
        client.probe()
    assert client.requests == 1 and client.total_tokens == 8
    assert client.events[-1]['status'] == 'response_rejected'
    assert client.events[-1]['response'] == raw
    assert TOKEN not in json.dumps(client.events)


def test_record_sensitivity_bound_is_checked_before_live_call(tmp_path):
    manifest = dataset_manifest(tmp_path)
    cfg = load_config(None)
    cfg['operators'].update(unit='record', max_row_bytes=1)
    with pytest.raises(ValueError, match='sensitivity'):
        run(cfg, manifest, tmp_path / 'not-created', provider=provider())
    assert not (tmp_path / 'not-created').exists()


def test_completed_release_cannot_be_changed_and_resumed(tmp_path):
    manifest = dataset_manifest(tmp_path)
    cfg = {**load_config(None), 'live_runs': 1}
    out = tmp_path / 'release-change'
    run(cfg, manifest, out, mock=True)
    path = out / 'results/release.prov.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='evaluation artifact changed'):
        run(cfg, manifest, out, mock=True, resume=True)


def test_missing_site_validation_error_identifies_the_recovery_action(tmp_path):
    ds = load_datasets(dataset_manifest(tmp_path))[0]
    sites = ds.site_ids[:3]
    trace = Trace(0, 'recovery-test', 'test-user')
    response = trace.entity('response', 'fixture decision')
    activity = trace.activity('agent_step', 0., 1., [], [response])
    tools = Tools(ds, sites, trace, 42)
    for site in sites:
        result, _ = tools.execute('load_partition', {'site': site}, response, activity)
        assert result['ok']
    for site in sites[:-1]:
        tools.execute('validate_schema', {'site': site}, response, activity)
    result, _ = tools.execute('harmonise', {}, response, activity)
    assert not result['ok'] and result['pending_validation_sites'] == [sites[-1]]
    assert result['pending_load_sites'] == [] and not result['table_ready']
    tools.execute('validate_schema', {'site': sites[-1]}, response, activity)
    result, _ = tools.execute('harmonise', {}, response, activity)
    assert result['ok'] and result['rows'] > 0


def test_batch_tool_schemas_hide_completed_phases_and_bound_agent_state(tmp_path):
    ds = load_datasets(dataset_manifest(tmp_path))[0]
    sites = ds.site_ids[:3]
    trace = Trace(0, 'state-guidance-test', 'test-user')
    response = trace.entity('response', 'fixture decision')
    activity = trace.activity('agent_step', 0., 1., [], [response])
    tools = Tools(ds, sites, trace, 42, workflow='explore', batch_tools=True)

    def names():
        return {item['function']['name'] for item in tools.schemas()}

    assert names() == {'load_partitions'}
    result, _ = tools.execute('load_partitions', {'sites': sites}, response, activity)
    assert result['ok'] and names() == {'validate_partitions'}
    rejected, _ = tools.execute('load_partitions', {'sites': sites}, response, activity)
    assert not rejected['ok'] and 'validate' in rejected['next_action']

    result, _ = tools.execute('validate_partitions', {'sites': sites}, response, activity)
    assert result['ok'] and names() == {'harmonise'}
    result, _ = tools.execute('harmonise', {}, response, activity)
    assert result['ok'] and names() == {'describe_data'}
    result, _ = tools.execute('describe_data', {}, response, activity)
    assert result['ok'] and names() == {'profile_features'}
    feature_schema = tools.schemas()[0]['function']['parameters']['properties']['features']
    assert feature_schema['maxItems'] == min(3, len(ds.features))
    result, _ = tools.execute('profile_features', {'features': ds.features[:3]}, response, activity)
    assert result['ok'] and tools.finished and names() == set()


def test_bad_json_and_unknown_tool_are_rejected_then_recover_and_replay(tmp_path):
    from provpub.e5.provider import Completion
    ds = load_datasets(dataset_manifest(tmp_path))[0]
    class RecoveringFixture(MockClient):
        def __init__(self):
            super().__init__()
            self.bad = 0
        def complete(self, messages, tools):
            if self.bad < 2:
                name = 'load_partition' if self.bad == 0 else 'unknown<|channel|>tool'
                args = '{"site": broken}' if self.bad == 0 else '{}'
                self.bad += 1
                message = reply(name)['choices'][0]['message']
                message['tool_calls'][0]['function']['arguments'] = args
                return Completion(message, {'choices': [{'message': message}]}, mock=True)
            return super().complete(messages, tools)
    captures = tmp_path / 'captures'
    captures.mkdir()
    graph, detail = run_agent(ds, ds.site_ids[:3], RecoveringFixture(), 'MOCK', 0., 0, 'test-group', captures)
    counts = coverage(graph, make_spec('MOCK', 0.))
    assert counts['source_partitions'] == 3 and (graph.ent['type'] == 'tool_error').sum() == 2
    assert (graph.act['type'] == 'tool_call_rejected').sum() == 2
    records = json.loads((captures / 'run_00000.json').read_text())
    assert records[0]['message']['tool_calls'][0]['function']['arguments'] == '{"site": broken}'
    state_message = records[1]['messages'][-1]['content']
    feedback = json.loads(state_message.split(': ', 1)[1].split('\n', 1)[0])
    assert feedback['pending_load_sites'] == ds.site_ids[:3]
    assert 'Next: load only the pending sites' in state_message
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()
    replay, replay_detail = run_agent(ds, ds.site_ids[:3], ReplayClient(records), 'MOCK', 0., 1, 'test-group', replay_dir)
    assert detail['metrics'] == replay_detail['metrics'] and replay.n_records == graph.n_records


def test_rejected_batch_arguments_are_fed_back_then_recover_and_replay(tmp_path):
    # Live Airfoil preflight: a scalar instead of a list was retried verbatim until the step limit.
    from provpub.e5.provider import Completion
    ds = load_datasets(dataset_manifest(tmp_path))[0]

    class ScalarFeatureFixture(MockClient):
        def __init__(self):
            super().__init__()
            self.seen = []

        def complete(self, messages, tools):
            self.seen.append(messages[-1]['content'])
            names = {t['function']['name'] for t in tools}
            if 'profile_features' in names and 'previous tool call' not in messages[-1]['content']:
                feature = tools[0]['function']['parameters']['properties']['features']['items']['enum'][0]
                message = reply('profile_features', {'features': feature})['choices'][0]['message']
                return Completion(message, {'choices': [{'message': message}]}, mock=True)
            return super().complete(messages, tools)

    captures = tmp_path / 'captures'
    captures.mkdir()
    client = ScalarFeatureFixture()
    graph, detail = run_agent(ds, ds.site_ids[:3], client, 'MOCK', 0., 0, 'g', captures,
                              workflow='explore', batch_tools=True)
    feedback = next(m for m in client.seen if 'previous tool call' in m)
    note = json.loads(feedback.split('current): ', 1)[1].split('\n', 1)[0])
    assert note[0]['tool'] == 'profile_features' and note[0]['error'] == 'invalid_batch_arguments'
    assert 'JSON array' in note[0]['expected_arguments']['features']
    assert '"features": "' in note[0]['sent_arguments']
    assert (graph.ent['type'] == 'tool_error').sum() == 1 and len(detail['metrics']['profiles']) == min(3, len(ds.features))
    records = json.loads((captures / 'run_00000.json').read_text())
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()
    replay, replay_detail = run_agent(ds, ds.site_ids[:3], ReplayClient(records), 'MOCK', 0., 1, 'g', replay_dir,
                                      workflow='explore', batch_tools=True)
    assert detail['metrics'] == replay_detail['metrics'] and replay.n_records == graph.n_records
