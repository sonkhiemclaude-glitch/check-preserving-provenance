"""Adapters for real provenance traces.

Real traces are expected to be placed manually under ``data.root`` on the
machine that runs the experiments (see docs/data_format.md). Nothing in this
module downloads data.

Supported inputs
----------------
``prov-json``      PROV-JSON documents written with provpub conventions
                   (``ppub:run``, ``ppub:seq``, PROV-AGENT ``prov:type``s);
                   see data/prov_io.py. Generic PROV-JSON without these
                   attributes is NOT supported (no run membership, no order).
``provpub-spans``  JSON list of GenAI-style span dicts as written by
                   ``to_otel_spans`` (lineage in ``ppub.used``/``ppub.generated``).
``otlp-json``      OpenTelemetry OTLP/JSON trace exports
                   (``resourceSpans[].scopeSpans[].spans[]``). Converted by
                   ``otlp_json_to_spans`` under the documented assumptions below.
                   Implemented and unit-tested on synthetic OTLP documents only;
                   NOT validated against exports of a real agent framework.
Flowcept           ``FlowceptAdapter`` is an adapter stub: Flowcept 1.0.3 has no
                   PROV-JSON exporter and its task-object schema was not verified
                   in enough detail to write a mapping without guessing.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml

from .prov_io import from_otel_spans, from_prov_json
from .schemas import Graph, concat_graphs
from .spec import Spec
from .spec_io import load_spec_yaml

log = logging.getLogger(__name__)

FORMATS = ("prov-json", "provpub-spans", "otlp-json")
LLM_OPERATIONS = {"chat", "text_completion", "generate_content"}
TOOL_OPERATIONS = {"execute_tool"}
RUN_OPERATIONS = {"invoke_workflow", "invoke_agent", "create_agent"}


# ------------------------------------------------------------------ OTLP/JSON

def _otlp_value(v: dict | None) -> Any:
    """Decode an OTLP/JSON AnyValue."""
    if v is None:
        return None
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    if "arrayValue" in v:
        return [_otlp_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _otlp_value(kv.get("value")) for kv in v["kvlistValue"].get("values", [])}
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _attrs(lst: list[dict] | None) -> dict[str, Any]:
    return {kv["key"]: _otlp_value(kv.get("value")) for kv in (lst or [])}


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def otlp_json_to_spans(doc: dict, type_attr: str = "ppub.type", config_attr: str = "ppub.config",
                       user_attrs: tuple[str, ...] = ("enduser.id", "user.id"),
                       path_arg: str = "path") -> list[dict]:
    """Convert an OTLP/JSON trace document to provpub span dicts.

    Assumptions (documented in docs/data_format.md; author verification needed
    for any real framework):
      * one trace = one run; runs are numbered in order of first appearance;
      * spans whose ``gen_ai.operation.name`` is in RUN_OPERATIONS and that have
        no parent in the trace carry run attributes (user, replica attribute);
      * ``chat``/``text_completion``/``generate_content`` spans are model
        invocations; ``execute_tool`` spans are tool executions; the activity
        type is the ``type_attr`` attribute if present, else the span name;
      * prompts and responses are taken from ``gen_ai.input.messages`` /
        ``gen_ai.output.messages`` (content captured only if the exporter was
        configured to capture it; otherwise the entities carry no text);
      * a tool execution is informed by the latest model invocation of the same
        trace that started before it (PROV-AGENT: AgentTool wasInformedBy
        AIModelInvocation);
      * data lineage is read from ``ppub.used`` / ``ppub.generated`` if present;
        standard GenAI spans carry no data lineage, so data-flow and replica
        input checks need these attributes or are vacuous.
    """
    raw = []
    for rs in doc.get("resourceSpans", []):
        res = _attrs(rs.get("resource", {}).get("attributes"))
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                raw.append((sp, res))
    if not raw:
        raise ValueError("no spans found in OTLP/JSON document")
    run_of: dict[str, int] = {}
    for sp, _ in raw:
        run_of.setdefault(sp["traceId"], len(run_of))
    raw.sort(key=lambda x: (run_of[x[0]["traceId"]], int(x[0].get("startTimeUnixNano", 0))))
    next_id = 1
    sid_map: dict[str, int] = {}
    ent_id = 10**9
    spans: list[dict] = []
    last_llm: dict[int, int] = {}
    runs_seen: set[int] = set()
    for sp, res in raw:
        a = _attrs(sp.get("attributes"))
        run = run_of[sp["traceId"]]
        op = a.get("gen_ai.operation.name")
        if op in RUN_OPERATIONS and not sp.get("parentSpanId") and run not in runs_seen:
            runs_seen.add(run)
            user = next((a.get(k) or res.get(k) for k in user_attrs if a.get(k) or res.get(k)), None)
            spans.append({"trace_id": f"t{run}", "span_id": f"w{run}", "parent_span_id": None,
                          "name": "invoke_workflow",
                          "attributes": {"gen_ai.operation.name": "invoke_workflow",
                                         "ppub.config": a.get(config_attr), "enduser.id": user}})
            continue
        if op not in LLM_OPERATIONS | TOOL_OPERATIONS:
            continue
        sid_map[sp["spanId"]] = next_id
        aid = next_id
        next_id += 1
        attrs: dict[str, Any] = {"ppub.type": a.get(type_attr) or sp.get("name"), "gen_ai.agent.id": a.get("gen_ai.agent.id")}
        used = list(a.get("ppub.used") or [])
        generated = list(a.get("ppub.generated") or [])
        if op in LLM_OPERATIONS:
            attrs.update({"gen_ai.operation.name": "chat",
                          "gen_ai.request.model": a.get("gen_ai.request.model") or a.get("gen_ai.response.model"),
                          "gen_ai.request.temperature": None if a.get("gen_ai.request.temperature") is None
                          else str(a.get("gen_ai.request.temperature")),
                          "gen_ai.request.seed": None if a.get("gen_ai.request.seed") is None
                          else str(a.get("gen_ai.request.seed"))})
            pin = a.get("gen_ai.input.messages")
            pout = a.get("gen_ai.output.messages")
            pin_s = pin if isinstance(pin, str) or pin is None else json.dumps(pin, sort_keys=True)
            pout_s = pout if isinstance(pout, str) or pout is None else json.dumps(pout, sort_keys=True)
            # synthesise prompt/response entities only if lineage attributes do not carry them
            if not any(e.get("type") == "prompt" for e in used):
                used.append({"id": ent_id, "run": run, "type": "prompt", "text": pin_s,
                             "digest": _digest("prompt", pin_s) if pin_s is not None else None})
            if not any(e.get("type") == "response" for e in generated):
                generated.append({"id": ent_id + 1, "run": run, "type": "response", "text": pout_s,
                                  "digest": _digest("resp", pout_s) if pout_s is not None else None})
            ent_id += 2
            last_llm[run] = aid
            informed = []
        else:
            args = a.get("gen_ai.tool.call.arguments")
            if isinstance(args, str):
                try:
                    args_d = json.loads(args)
                except json.JSONDecodeError:
                    args_d = {}
            else:
                args_d = args or {}
            attrs.update({"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": a.get("gen_ai.tool.name"),
                          "gen_ai.tool.call.arguments": json.dumps({"path": args_d.get(path_arg),
                                                                    "value": args_d.get("value"),
                                                                    "site": args_d.get("site")})})
            informed = [f"s{last_llm[run]}"] if run in last_llm else []
        attrs["ppub.used"] = used
        attrs["ppub.generated"] = generated
        attrs["ppub.informed_by"] = informed
        spans.append({"trace_id": f"t{run}", "span_id": f"s{aid}", "parent_span_id": f"w{run}",
                      "name": attrs["ppub.type"], "start_time_unix_nano": int(sp.get("startTimeUnixNano", 0)),
                      "end_time_unix_nano": int(sp.get("endTimeUnixNano", 0)), "seq": None, "attributes": attrs})
    for run in set(run_of.values()) - runs_seen:   # traces without a run span
        spans.append({"trace_id": f"t{run}", "span_id": f"w{run}", "parent_span_id": None, "name": "invoke_workflow",
                      "attributes": {"gen_ai.operation.name": "invoke_workflow", "ppub.config": None, "enduser.id": None}})
    return spans


def spans_to_otlp_json(spans: list[dict]) -> dict:
    """Wrap provpub span dicts in an OTLP/JSON envelope (used to build synthetic
    test documents; real exporters produce this envelope themselves)."""
    def enc(v):
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, float):
            return {"doubleValue": v}
        if isinstance(v, list):
            return {"arrayValue": {"values": [enc(x) for x in v]}}
        if isinstance(v, dict):
            return {"kvlistValue": {"values": [{"key": k, "value": enc(x)} for k, x in v.items()]}}
        return {"stringValue": "" if v is None else str(v)}
    out = []
    for s in spans:
        attrs = []
        for k, v in s["attributes"].items():
            if v is None:
                continue
            attrs.append({"key": k, "value": enc(v)})
        d = {"traceId": s["trace_id"], "spanId": s["span_id"], "name": s["name"], "attributes": attrs}
        if s.get("parent_span_id"):
            d["parentSpanId"] = s["parent_span_id"]
        if "start_time_unix_nano" in s:
            d["startTimeUnixNano"] = str(s["start_time_unix_nano"])
            d["endTimeUnixNano"] = str(s["end_time_unix_nano"])
        out.append(d)
    return {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"scope": {"name": "provpub"}, "spans": out}]}]}


# ------------------------------------------------------------------ files

def load_graph(path: str, fmt: str, **kw: Any) -> Graph:
    """Load one trace file in one of FORMATS."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    if fmt == "prov-json":
        return from_prov_json(path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if fmt == "provpub-spans":
        return from_otel_spans(doc)
    return from_otel_spans(otlp_json_to_spans(doc, **kw))


def _offset(g: Graph, run_off: int, id_off: int) -> Graph:
    h = g.copy()
    for t in (h.runs, h.act, h.ent, h.edge):
        if "run" in t.columns:
            t["run"] = t["run"].astype("int64") + run_off
    for t in (h.act, h.ent):
        t["id"] = t["id"].astype("int64") + id_off
    for c in ("src", "dst"):
        h.edge[c] = h.edge[c].astype("int64") + id_off
    return h


@dataclass
class TraceDirectoryAdapter:
    """Real-trace directory (``data.root``).

    Layout::

        <root>/
          manifest.yaml        format: prov-json | provpub-spans | otlp-json
                               (optional keys: source, collector, collected_at, notes,
                                otlp: {type_attr, config_attr, path_arg})
          spec.yaml            declared specification (data/spec_io.py schema)
          traces/*.json        one or more trace files in the declared format
          truth/runs.csv       OPTIONAL, attacks only: run_file, run, user, sites (';'-separated)
          truth/sites_aux.csv  OPTIONAL, attacks only: site, approx_records, row_width, load_seconds

    Files are merged with disjoint run and node identifiers. The adapter never
    writes into <root>.
    """

    root: str
    manifest: dict = field(default_factory=dict)
    run_offsets: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"data.root does not exist: {self.root}")
        mpath = os.path.join(self.root, "manifest.yaml")
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"missing {mpath} (see docs/data_format.md)")
        with open(mpath, encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f) or {}
        if self.manifest.get("format") not in FORMATS:
            raise ValueError(f"manifest.yaml: format must be one of {FORMATS}")

    def spec(self) -> Spec:
        return load_spec_yaml(os.path.join(self.root, "spec.yaml"))

    def trace_files(self) -> list[str]:
        files = sorted(glob.glob(os.path.join(self.root, "traces", "*.json")))
        if not files:
            raise FileNotFoundError(f"no trace files under {self.root}/traces/")
        return files

    def load(self) -> Graph:
        fmt = self.manifest["format"]
        kw = self.manifest.get("otlp", {}) if fmt == "otlp-json" else {}
        parts, run_off, id_off = [], 0, 0
        for path in self.trace_files():
            g = load_graph(path, fmt, **kw)
            self.run_offsets[os.path.basename(path)] = run_off
            g = _offset(g, run_off, id_off)
            g.meta["source_file"] = os.path.basename(path)
            parts.append(g)
            if len(g.runs):
                run_off = int(g.runs["run"].max()) + 1
            ids = pd.concat([g.act["id"], g.ent["id"]])
            if len(ids):
                id_off = int(ids.max()) + 1
        g = concat_graphs(parts)
        g.meta.update({"source": self.manifest.get("source"), "format": fmt, "files": len(parts)})
        log.info("loaded %d runs, %d PROV records from %d files", len(g.runs), g.n_records, len(parts))
        return g

    def truth(self) -> dict | None:
        """Optional ground truth for the attacks (never used by the sanitiser).

        truth/runs.csv columns: run_file (trace file name), run (run id inside
        that file, as numbered by the loader), user, sites (';'-separated).
        Call after load(), which fixes the per-file run offsets."""
        p = os.path.join(self.root, "truth", "runs.csv")
        if not os.path.exists(p):
            return None
        if not self.run_offsets:
            raise RuntimeError("call load() before truth()")
        df = pd.read_csv(p)
        need = {"run_file", "run", "user", "sites"}
        if not need <= set(df.columns):
            raise ValueError(f"truth/runs.csv needs columns {sorted(need)}")
        out = {}
        for r in df.itertuples():
            if r.run_file not in self.run_offsets:
                raise ValueError(f"truth/runs.csv refers to unknown trace file {r.run_file!r}")
            out[int(r.run) + self.run_offsets[r.run_file]] = {"user": r.user, "sites": str(r.sites).split(";")}
        return out


class FlowceptAdapter:
    """ADAPTER STUB (external dependency): Flowcept task objects -> provpub Graph.

    Expected input (to be confirmed against the Flowcept version in use):
    task records with an identifier, workflow/campaign identifier (-> run),
    ``subtype`` (``ai_model_invocation`` / ``agent_tool``), ``used`` and
    ``generated`` dictionaries (prompt/response for model invocations),
    ``started_at``/``ended_at`` and user/host fields.
    Output: a Graph with the columns in data/synthetic.py ACT_COLS / ENT_COLS.

    Not implemented because the field-level schema of Flowcept 1.0.3 exports was
    not verified for this repository. Export
    Flowcept data to the provpub span format or OTLP/JSON instead, or implement
    ``to_graph`` after checking the schema.
    """

    def __init__(self, path: str):
        self.path = path

    def to_graph(self) -> Graph:
        raise NotImplementedError("FlowceptAdapter is an adapter stub; see its docstring")
