"""Dummy release mechanisms for tests only. Never reported as results."""
from __future__ import annotations

from ..data.schemas import Graph
from ..data.spec import Spec
from .base import ReleaseMechanism


class IdentityRelease(ReleaseMechanism):
    """Returns a copy of the raw graph (equivalent to the RAW reference row)."""

    kind = "dummy"
    name = "identity"

    def release(self, g: Graph, spec: Spec, key: bytes) -> Graph:
        return g.copy()


class DropAllRuns(ReleaseMechanism):
    """Releases an empty graph; every raw check instance becomes UNDET."""

    kind = "dummy"
    name = "drop_all"

    def release(self, g: Graph, spec: Spec, key: bytes) -> Graph:
        h = g.copy()
        h.runs = h.runs.iloc[0:0]
        h.act = h.act.iloc[0:0]
        h.ent = h.ent.iloc[0:0]
        h.edge = h.edge.iloc[0:0]
        return h
