"""Baselines as evaluated: applied after VG so that
identifier and content handling equals that of the proposed operators.

``private_types`` / ``group_types`` define which modules are private (module
hiding) or collapsed (abstraction); the choices for W1-W3 are in
experiments/configurations.py (PRIVATE_TYPES, ABSTRACT_GROUP).
"""
from __future__ import annotations

from ..data.schemas import Graph
from ..data.spec import Spec
from ..methods.operators import sanitise
from .base import ReleaseMechanism
from .structural import abstract, modhide, suppress_all


class SuppressAll(ReleaseMechanism):
    """Keep node types and relations; remove all values and timing (after VG)."""

    name = "suppress_all"

    def release(self, g: Graph, spec: Spec, key: bytes) -> Graph:
        return suppress_all(sanitise(g, spec, key, ops=("VG",)))


class ModuleHiding(ReleaseMechanism):
    """Hide data attributes of private modules and of the data they touch; keep
    vocabulary attributes (module identity) and timing (after VG)."""

    name = "module_hiding"

    def __init__(self, private_types: set[str]):
        self.private_types = set(private_types)

    def release(self, g: Graph, spec: Spec, key: bytes) -> Graph:
        return modhide(sanitise(g, spec, key, ops=("VG",)), self.private_types)


class StructuralAbstraction(ReleaseMechanism):
    """Collapse group_types per run into one composite activity and pool the
    source partitions (after VG unless values_in_clear)."""

    name = "structural_abstraction"

    def __init__(self, group_types: set[str], values_in_clear: bool = False):
        self.group_types = set(group_types)
        self.values_in_clear = values_in_clear

    def release(self, g: Graph, spec: Spec, key: bytes) -> Graph:
        h = g.copy() if self.values_in_clear else sanitise(g, spec, key, ops=("VG",))
        return abstract(h, spec, self.group_types)
