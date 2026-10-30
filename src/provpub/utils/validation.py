"""Input validation for graphs and configs (clear errors instead of silent failures)."""
from __future__ import annotations

import logging

import pandas as pd

from ..data.schemas import EDGE_KINDS, Graph

log = logging.getLogger(__name__)

REQUIRED = {"runs": ["run"], "act": ["id", "run", "type"], "ent": ["id", "run", "type"], "edge": ["kind", "src", "dst"]}


def validate_graph(g: Graph) -> list[str]:
    """Return a list of problems (empty if the graph is well formed)."""
    problems = []
    for tab, cols in REQUIRED.items():
        df = getattr(g, tab)
        miss = [c for c in cols if c not in df.columns]
        if miss:
            problems.append(f"{tab}: missing columns {miss}")
    if problems:
        return problems
    ids = pd.concat([g.act["id"], g.ent["id"]])
    if ids.duplicated().any():
        problems.append("activity/entity ids are not unique")
    bad = set(g.edge["kind"]) - set(EDGE_KINDS)
    if bad:
        problems.append(f"unknown relation kinds {sorted(bad)}")
    known = set(ids)
    dangling = ~(g.edge["src"].isin(known) & g.edge["dst"].isin(known))
    if dangling.any():
        problems.append(f"{int(dangling.sum())} relations reference unknown nodes")
    runs = set(g.runs["run"])
    for tab in ("act", "ent"):
        orphan = ~getattr(g, tab)["run"].isin(runs)
        if orphan.any():
            problems.append(f"{tab}: {int(orphan.sum())} rows belong to no declared run")
    if {"seq"} <= set(g.act.columns) and g.act["seq"].notna().all() and g.act.duplicated(["run", "seq"]).any():
        problems.append("collector sequence numbers repeat within a run (Assumption 3)")
    return problems


def consistent_provenance(g: Graph, source_type: str = "data_partition") -> bool:
    """Theorem 4(c) precondition: every partition identifier has one record count
    and one byte size across the graph. Returns False (and logs) otherwise."""
    if "part_id" not in g.ent.columns:
        return True
    src = g.ent[g.ent["type"] == source_type]
    src = src[src["part_id"].notna()]
    n = src.groupby("part_id")[["n_records", "n_bytes"]].nunique(dropna=False)
    ok = bool((n <= 1).all().all())
    if not ok:
        log.warning("inconsistent provenance: some partition identifiers carry different sizes; "
                    "count agreement may change under DPN")
    return ok


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)
