"""Read and write declared specifications as YAML.

A real deployment must state its specification explicitly; the synthetic
workloads W1-W3 build theirs in code (data/synthetic.py). This module lets
both be written to and read from YAML, so a specification for real traces
can be supplied with ``data.spec_path`` in a config file.

Schema (all keys except ``name`` optional):

    name: my-pipeline
    alphabet: [agent_plan, load_partition, ...]       # activity types
    constraints:                                       # Declare templates
      - {template: init, a: agent_plan}
      - {template: precedence, a: load_partition, b: validate_schema}
    vocab: {tool: [sql_load, ...], model: [...], temperature: ["0"]}
    vocab_rules:
      - {act_type: load_partition, attr: tool, allowed: [sql_load]}
      - {act_type: load_partition, attr: p_path, allowed: [allowed]}
    generalisers:
      p_path: {kind: path_prefix, allowed_prefixes: [/data/consortium/]}
    nd_types: [agent_plan, agent_reason]
    replica_attr: config
    source_type: data_partition
    result_type: result
    external_types: [data_partition, prompt]
    thresholds: [{ent_type: data_partition, attr: n_records, minimum: 500}]
    timed: [{kind: span, limit: 3600}, {kind: max_dur, act_type: load_partition, limit: 120}]

The Spec constructor enforces Assumption 2 (well-typed specification).
"""
from __future__ import annotations

from typing import Any

import yaml

from .spec import Constraint, Spec, ThresholdRule, TimedRule, VocabRule, path_generaliser


def spec_to_dict(spec: Spec) -> dict[str, Any]:
    gens = {}
    for attr, g in spec.generalisers.items():
        if getattr(g, "kind", None) != "path_prefix":
            raise ValueError(f"generaliser for {attr!r} cannot be serialised (only path_prefix is supported)")
        gens[attr] = {"kind": "path_prefix", "allowed_prefixes": list(g.allowed_prefixes)}  # type: ignore[attr-defined]
    return {
        "name": spec.name,
        "alphabet": sorted(spec.alphabet),
        "constraints": [{"template": c.template, "a": c.a, **({"b": c.b} if c.b else {})} for c in spec.constraints],
        "vocab": {k: sorted(map(str, v)) for k, v in spec.vocab.items()},
        "vocab_rules": [{"act_type": r.act_type, "attr": r.attr, "allowed": sorted(map(str, r.allowed))}
                        for r in spec.vocab_rules],
        "generalisers": gens,
        "nd_types": sorted(spec.nd_types),
        "nd_requires_seed": spec.nd_requires_seed,
        "replica_attr": spec.replica_attr,
        "source_type": spec.source_type,
        "result_type": spec.result_type,
        "external_types": sorted(spec.external_types),
        "thresholds": [{"ent_type": t.ent_type, "attr": t.attr, "minimum": t.minimum} for t in spec.thresholds],
        "timed": [{"kind": t.kind, "limit": t.limit, **({"act_type": t.act_type} if t.act_type else {})}
                  for t in spec.timed],
    }


def spec_from_dict(d: dict[str, Any]) -> Spec:
    if "name" not in d:
        raise ValueError("specification needs a 'name'")
    gens = {}
    for attr, gd in (d.get("generalisers") or {}).items():
        if gd.get("kind") != "path_prefix":
            raise ValueError(f"unsupported generaliser kind {gd.get('kind')!r} for {attr!r}")
        gens[attr] = path_generaliser(tuple(gd["allowed_prefixes"]))
    cons = [Constraint(c["template"], c["a"], c.get("b")) for c in d.get("constraints", [])]
    alphabet = set(d.get("alphabet", []))
    for c in cons:
        alphabet.add(c.a)
        if c.b:
            alphabet.add(c.b)
    return Spec(
        name=d["name"],
        alphabet=frozenset(alphabet),
        constraints=cons,
        vocab={k: frozenset(map(str, v)) for k, v in (d.get("vocab") or {}).items()},
        vocab_rules=[VocabRule(r["act_type"], r["attr"], frozenset(map(str, r["allowed"])))
                     for r in d.get("vocab_rules", [])],
        generalisers=gens,
        nd_types=frozenset(d.get("nd_types", [])),
        nd_requires_seed=d.get("nd_requires_seed", True),
        replica_attr=d.get("replica_attr", "config"),
        source_type=d.get("source_type", "data_partition"),
        result_type=d.get("result_type", "result"),
        external_types=frozenset(d.get("external_types", ["data_partition", "prompt"])),
        thresholds=[ThresholdRule(t["ent_type"], t["attr"], float(t["minimum"])) for t in d.get("thresholds", [])],
        timed=[TimedRule(t["kind"], float(t["limit"]), t.get("act_type")) for t in d.get("timed", [])],
    )


def load_spec_yaml(path: str) -> Spec:
    with open(path, encoding="utf-8") as f:
        return spec_from_dict(yaml.safe_load(f))


def save_spec_yaml(spec: Spec, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec_to_dict(spec), f, sort_keys=False)
