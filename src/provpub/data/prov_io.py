"""W3C PROV-JSON input/output in the PROV-AGENT model, plus an OpenTelemetry
GenAI span adapter.

Mapping:
  LLM step activity          -> prov:Activity, prov:type provagent:AIModelInvocation
  tool activity              -> prov:Activity, prov:type provagent:AgentTool
  agent                      -> prov:Agent,    prov:type provagent:AIAgent
  prompt / response entity   -> prov:Entity,   provagent:Prompt / provagent:ResponseData
  other entities             -> prov:Entity,   provagent:DomainData (ppub:role gives our type)
  run                        -> prov:Activity, prov:type provagent:Workflow (ppub:run)
Relations: used, wasGeneratedBy, wasInformedBy (AgentTool informed by
AIModelInvocation), wasAssociatedWith (activity -> AIAgent).
Published coarse timing uses ppub:rank, ppub:bStart, ppub:bEnd; raw graphs
use prov:startTime / prov:endTime (seconds since epoch, ISO 8601).
"""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd

from .schemas import ACT_TYPES_LLM, ATTR_CLASS, Graph

PREFIX = {"prov": "http://www.w3.org/ns/prov#",
          "provagent": "https://w3id.org/provagent#",
          "ppub": "https://example.org/provpub#",
          "ex": "https://example.org/release#"}


def _iso(t):
    return dt.datetime.fromtimestamp(float(t), tz=dt.timezone.utc).isoformat()


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def to_prov_json(g: Graph, path: str, llm_types=ACT_TYPES_LLM):
    """Stream a graph to a PROV-JSON file. Returns bytes written."""
    act, ent, e, runs = g.act, g.ent, g.edge, g.runs
    attr_a = [c for c in act.columns if c in ATTR_CLASS]
    attr_e = [c for c in ent.columns if c in ATTR_CLASS]
    has_raw_time = "start" in act.columns and pd.api.types.is_numeric_dtype(act["start"])
    with open(path, "w") as f:
        f.write('{"prefix": ' + json.dumps(PREFIX))
        f.write(', "agent": {"ex:agent": {"prov:type": "provagent:AIAgent"}}')
        f.write(', "activity": {')
        first = True
        for r in runs.itertuples(index=False):
            d = {"prov:type": "provagent:Workflow", "ppub:run": int(r.run)}
            if "delta" in g.meta:
                d["ppub:delta"] = float(g.meta["delta"])
            for c in runs.columns:
                if c != "run":
                    v = _clean(getattr(r, c))
                    if v is not None:
                        d[f"ppub:{c}"] = v
            f.write(("" if first else ", ") + json.dumps(f"ex:wf{int(r.run)}") + ": " + json.dumps(d))
            first = False
        cols = list(act.columns)
        for row in act.itertuples(index=False, name=None):
            rec = dict(zip(cols, row))
            typ = rec["type"]
            d = {"prov:type": "provagent:AIModelInvocation" if typ in llm_types else "provagent:AgentTool",
                 "ppub:activityType": typ, "ppub:run": int(rec["run"])}
            if has_raw_time:
                d["prov:startTime"] = _iso(rec["start"])
                d["prov:endTime"] = _iso(rec["end"])
                d["ppub:seq"] = int(rec["seq"])
            for c in ("rank", "b_start", "b_end"):
                if c in rec:
                    d[f"ppub:{c}"] = int(rec[c])
            for c in attr_a:
                v = _clean(rec[c])
                if v is not None:
                    d[f"ppub:{c}"] = v
            f.write(", " + json.dumps(f"ex:a{int(rec['id'])}") + ": " + json.dumps(d))
        f.write("}, \"entity\": {")
        first = True
        cols = list(ent.columns)
        for row in ent.itertuples(index=False, name=None):
            rec = dict(zip(cols, row))
            typ = rec["type"]
            pt = {"prompt": "provagent:Prompt", "response": "provagent:ResponseData"}.get(typ, "provagent:DomainData")
            d = {"prov:type": pt, "ppub:role": typ, "ppub:run": int(rec["run"])}
            for c in attr_e:
                v = _clean(rec[c])
                if v is not None:
                    d[f"ppub:{c}"] = v
            f.write(("" if first else ", ") + json.dumps(f"ex:e{int(rec['id'])}") + ": " + json.dumps(d))
            first = False
        f.write("}")
        ids_act = set(act["id"].tolist())
        for kind, sk, dk in (("used", "prov:activity", "prov:entity"),
                             ("wasGeneratedBy", "prov:entity", "prov:activity"),
                             ("wasInformedBy", "prov:informed", "prov:informant")):
            sub = e[e["kind"] == kind]
            f.write(f', "{kind}": {{')
            first = True
            for k, (s, t) in enumerate(zip(sub["src"].tolist(), sub["dst"].tolist())):
                sp = ("ex:a" if s in ids_act else "ex:e") + str(int(s))
                tp = ("ex:a" if t in ids_act else "ex:e") + str(int(t))
                f.write(("" if first else ", ") + json.dumps(f"_:{kind[0]}{k}") + ": " + json.dumps({sk: sp, dk: tp}))
                first = False
            f.write("}")
        f.write(', "wasAssociatedWith": {')
        first = True
        for k, a in enumerate(act["id"].tolist()):
            f.write(("" if first else ", ") + json.dumps(f"_:w{k}") + ": " + json.dumps({"prov:activity": f"ex:a{int(a)}", "prov:agent": "ex:agent"}))
            first = False
        f.write("}}")
        return f.tell()


def from_prov_json(path: str) -> Graph:
    """Read a PROV-JSON document written by to_prov_json (raw or sanitised)."""
    with open(path) as f:
        doc = json.load(f)
    runs, act, ent, edge = [], [], [], []
    for k, d in doc.get("activity", {}).items():
        if d.get("prov:type") == "provagent:Workflow":
            r = {"run": d["ppub:run"]}
            r.update({c.split(":", 1)[1]: v for c, v in d.items() if c.startswith("ppub:") and c != "ppub:run"})
            runs.append(r)
            continue
        r = {"id": int(k[4:]), "run": d["ppub:run"], "type": d["ppub:activityType"]}
        if "prov:startTime" in d:
            r["start"] = dt.datetime.fromisoformat(d["prov:startTime"]).timestamp()
            r["end"] = dt.datetime.fromisoformat(d["prov:endTime"]).timestamp()
        for c, v in d.items():
            if c.startswith("ppub:") and c not in ("ppub:run", "ppub:activityType"):
                r[c.split(":", 1)[1]] = v
        act.append(r)
    for k, d in doc.get("entity", {}).items():
        r = {"id": int(k[4:]), "run": d["ppub:run"], "type": d["ppub:role"]}
        for c, v in d.items():
            if c.startswith("ppub:") and c not in ("ppub:run", "ppub:role"):
                r[c.split(":", 1)[1]] = v
        ent.append(r)
    for kind, sk, dk in (("used", "prov:activity", "prov:entity"),
                         ("wasGeneratedBy", "prov:entity", "prov:activity"),
                         ("wasInformedBy", "prov:informed", "prov:informant")):
        for _, d in doc.get(kind, {}).items():
            edge.append((kind, int(d[sk][4:]), int(d[dk][4:])))
    A = pd.DataFrame(act)
    E = pd.DataFrame(ent)
    for c in ("n_records", "n_bytes", "out_rows", "out_bytes"):
        if c in E.columns:
            E[c] = E[c].astype(object).where(E[c].notna(), None)
    run_of = dict(zip(A["id"], A["run"]))
    run_of.update(dict(zip(E["id"], E["run"])))
    ED = pd.DataFrame(edge, columns=["kind", "src", "dst"])
    ED["run"] = ED["src"].map(run_of)
    R = pd.DataFrame(runs)
    meta = {}
    if "delta" in R.columns:
        meta["delta"] = float(R["delta"].iloc[0])
        R = R.drop(columns=["delta"])
    return Graph(R, A, E, ED, meta)


# ---------------------------------------------------------------- OpenTelemetry

def to_otel_spans(g: Graph, llm_types=ACT_TYPES_LLM) -> list[dict]:
    """Export a raw graph as OpenTelemetry GenAI-style spans (one trace per run).
    Prompts/responses go to gen_ai.input.messages / gen_ai.output.messages;
    data lineage is carried in custom attributes ppub.used / ppub.generated."""
    act, ent, e = g.act, g.ent, g.edge
    ent_by_id = ent.set_index("id")
    used = e[e["kind"] == "used"].groupby("src")["dst"].apply(list).to_dict()
    gen = e[e["kind"] == "wasGeneratedBy"].groupby("dst")["src"].apply(list).to_dict()
    inf = e[e["kind"] == "wasInformedBy"].groupby("src")["dst"].apply(list).to_dict()
    spans: list[dict] = []
    for r in g.runs.itertuples(index=False):
        spans.append({"trace_id": f"t{int(r.run)}", "span_id": f"w{int(r.run)}", "parent_span_id": None,
                      "name": "invoke_workflow", "attributes": {"gen_ai.operation.name": "invoke_workflow",
                                                                 "ppub.config": r.config, "enduser.id": r.user}})
    for rec in act.to_dict("records"):
        llm = rec["type"] in llm_types
        attrs: dict = {"gen_ai.operation.name": "chat" if llm else "execute_tool", "ppub.type": rec["type"],
                 "gen_ai.agent.id": rec.get("agent")}
        if llm:
            attrs["gen_ai.request.model"] = rec.get("model")
            attrs["gen_ai.request.temperature"] = rec.get("temperature")
            attrs["gen_ai.request.seed"] = rec.get("seed")
        else:
            attrs["gen_ai.tool.name"] = rec.get("tool")
            attrs["gen_ai.tool.call.arguments"] = json.dumps({"path": rec.get("p_path"), "value": rec.get("p_value"),
                                                              "site": rec.get("site")})
        u = used.get(rec["id"], [])
        gn = gen.get(rec["id"], [])
        for k, lst in (("ppub.used", u), ("ppub.generated", gn)):
            attrs[k] = [{c: _clean(v) for c, v in ent_by_id.loc[x].to_dict().items()} | {"id": int(x)} for x in lst]
        attrs["ppub.informed_by"] = [f"s{int(x)}" for x in inf.get(rec["id"], [])]
        spans.append({"trace_id": f"t{int(rec['run'])}", "span_id": f"s{int(rec['id'])}", "parent_span_id": f"w{int(rec['run'])}",
                      "name": rec["type"], "start_time_unix_nano": int(round(rec["start"] * 1e9)),
                      "end_time_unix_nano": int(round(rec["end"] * 1e9)), "seq": int(rec["seq"]), "attributes": attrs})
    return spans


def from_otel_spans(spans: list[dict]) -> Graph:
    """Build a PROV-AGENT-model graph from GenAI-style spans (inverse of to_otel_spans)."""
    runs, act, ent, edge = [], [], [], []
    seen_ent = set()
    for s in spans:
        a = s["attributes"]
        run = int(s["trace_id"][1:])
        if a.get("gen_ai.operation.name") == "invoke_workflow":
            runs.append({"run": run, "config": a.get("ppub.config"), "user": a.get("enduser.id")})
            continue
        aid = int(s["span_id"][1:])
        row = {"id": aid, "run": run, "type": a["ppub.type"], "start": s["start_time_unix_nano"] / 1e9,
               "end": s["end_time_unix_nano"] / 1e9, "seq": s.get("seq"), "agent": a.get("gen_ai.agent.id")}
        if a["gen_ai.operation.name"] == "chat":
            row.update(model=a.get("gen_ai.request.model"), temperature=a.get("gen_ai.request.temperature"),
                       seed=a.get("gen_ai.request.seed"), tool=None)
        else:
            args = json.loads(a.get("gen_ai.tool.call.arguments") or "{}")
            row.update(tool=a.get("gen_ai.tool.name"), p_path=args.get("path"), p_value=args.get("value"),
                       site=args.get("site"))
        act.append(row)
        for k, kind in (("ppub.used", "used"), ("ppub.generated", "wasGeneratedBy")):
            for en in a.get(k, []):
                if en["id"] not in seen_ent:
                    seen_ent.add(en["id"])
                    ent.append(en)
                edge.append((kind, aid, en["id"], run) if kind == "used" else (kind, en["id"], aid, run))
        for x in a.get("ppub.informed_by", []):
            edge.append(("wasInformedBy", aid, int(x[1:]), run))
    from .synthetic import ACT_COLS, ENT_COLS
    A = pd.DataFrame(act).reindex(columns=ACT_COLS)
    # collector sequence numbers must be unique per run; spans without one get a
    # deterministic per-trace order (start time, then span order)
    if A["seq"].isna().any() or A.duplicated(["run", "seq"]).any():
        A["_o"] = range(len(A))
        A = A.sort_values(["run", "start", "_o"], kind="mergesort")
        A["seq"] = A.groupby("run").cumcount() + 1
        A = A.sort_values("_o").drop(columns="_o")
    E = pd.DataFrame(ent).reindex(columns=ENT_COLS)
    for c in ("n_records", "n_bytes", "out_rows", "out_bytes"):
        E[c] = E[c].astype(object).where(E[c].notna(), None)
    return Graph(pd.DataFrame(runs), A, E, pd.DataFrame(edge, columns=["kind", "src", "dst", "run"]), {})
