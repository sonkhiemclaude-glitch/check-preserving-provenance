"""Columnar provenance graph in the PROV-AGENT model.

A Graph holds four tables:
  runs : one row per pipeline run (run, plus run-level attributes)
  act  : one row per PROV activity (agent step, tool call, model invocation)
  ent  : one row per PROV entity (prompt, response, data partition, result ...)
  edge : one row per PROV relation (kind in EDGE_KINDS, src, dst)

Every activity and entity belongs to exactly one run. Attribute columns are
classified by ATTR_CLASS; operators act on columns by class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Sentinel for a value removed by a suppressing operator. Checks that need a
# suppressed value return UNDETERMINED rather than pass or fail.
SUPP = "⊥supp"

PASS, FAIL, UNDET = 1, 0, -1

EDGE_KINDS = ("used", "wasGeneratedBy", "wasAssociatedWith", "wasInformedBy", "wasDerivedFrom")

# Attribute classes.
#   vocab   : value may belong to the public specification vocabulary
#   path    : value has a public generaliser in the specification
#   text    : free text (prompts, responses)
#   ident   : identifier (site, user, seed, parameter value, config)
#   digest  : content digest of an entity
#   num_root: numeric attribute of a source partition (record count, bytes)
#   num_der : numeric attribute derived from other data (output sizes)
ATTR_CLASS = {
    # run table
    "config": "ident",
    "user": "ident",
    # activities
    "agent": "ident",
    "tool": "vocab",
    "model": "vocab",
    "temperature": "vocab",
    "seed": "ident",
    "site": "ident",
    "p_path": "path",
    "p_value": "ident",
    # entities
    "text": "text",
    "digest": "digest",
    "part_id": "ident",
    "n_records": "num_root",
    "n_bytes": "num_root",
    "out_rows": "num_der",
    "out_bytes": "num_der",
}

# PROV-AGENT-style class of each node type (see prov_io.py for the mapping).
ACT_TYPES_LLM = {"agent_plan", "agent_reason", "agent_report", "agent_step"}


@dataclass
class Graph:
    runs: pd.DataFrame
    act: pd.DataFrame
    ent: pd.DataFrame
    edge: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def copy(self) -> "Graph":
        return Graph(self.runs.copy(), self.act.copy(), self.ent.copy(), self.edge.copy(), dict(self.meta))

    @property
    def n_records(self) -> int:
        """PROV records: activities + entities + relations."""
        return len(self.act) + len(self.ent) + len(self.edge)

    @property
    def n_nodes(self) -> int:
        return len(self.act) + len(self.ent)

    def has_order(self) -> bool:
        return "rank" in self.act.columns or ("start" in self.act.columns and "seq" in self.act.columns)


def ranks(act: pd.DataFrame) -> pd.Series:
    """Per-run rank of each activity (0-based).

    Uses a published 'rank' column if present; otherwise the collector order
    (start time, then collector sequence number). Returns NaN if no order
    information is published.
    """
    if "rank" in act.columns:
        return act["rank"]
    if "start" in act.columns and "seq" in act.columns:
        if not pd.api.types.is_numeric_dtype(act["start"]):  # suppressed
            return pd.Series(np.nan, index=act.index)
        if act.duplicated(["run", "seq"]).any():
            raise ValueError("collector sequence numbers must be unique within a run (Assumption 3)")
        order = act.sort_values(["run", "start", "seq"], kind="mergesort")
        r = order.groupby("run", sort=False).cumcount()
        return r.reindex(act.index)
    return pd.Series(np.nan, index=act.index)


def concat_graphs(gs: list[Graph]) -> Graph:
    return Graph(
        pd.concat([g.runs for g in gs], ignore_index=True),
        pd.concat([g.act for g in gs], ignore_index=True),
        pd.concat([g.ent for g in gs], ignore_index=True),
        pd.concat([g.edge for g in gs], ignore_index=True),
        dict(gs[0].meta) if gs else {},
    )


def split_by_runs(g: Graph, n_parts: int) -> list[Graph]:
    """Partition a graph into n_parts shards by run (runs are independent)."""
    run_ids = g.runs["run"].to_numpy()
    part_of = {r: i % n_parts for i, r in enumerate(run_ids)}
    out = []
    for p in range(n_parts):
        rs = {r for r, q in part_of.items() if q == p}
        out.append(Graph(
            g.runs[g.runs["run"].isin(rs)].reset_index(drop=True),
            g.act[g.act["run"].isin(rs)].reset_index(drop=True),
            g.ent[g.ent["run"].isin(rs)].reset_index(drop=True),
            g.edge[g.edge["run"].isin(rs)].reset_index(drop=True),
            dict(g.meta),
        ))
    return out
