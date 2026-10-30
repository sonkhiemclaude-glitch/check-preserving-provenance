"""Evaluated release configurations.

Every name here maps to keyword arguments of ``methods.operators.sanitise`` or
to a baseline tag. ``build`` reproduces exactly the releases used for the
reference E1/E2 results (same keys, same order of operators).
"""
from __future__ import annotations

from ..baselines.structural import abstract, modhide, suppress_all
from ..data.schemas import Graph
from ..data.spec import Spec
from ..methods import operators as O

SALT = b"published-salt-v1"   # public salt of the "published salt" ablation (known to the adversary)

# runs per workload chosen so that each release has about 1e5 PROV records
RUNS_1E5 = {"W1": 720, "W2": 1000, "W3": 430}

# private modules for module hiding
PRIVATE_TYPES = {
    "W1": {"agent_step", "load_csv"},
    "W2": {"load_partition", "validate_schema", "agent_plan", "agent_reason", "agent_report"},
    "W3": {"run_simulation", "agent_plan", "agent_reason", "agent_report"},
}
# collapsed modules for structural abstraction
ABSTRACT_GROUP = {
    "W1": {"agent_step", "load_csv"},
    "W2": {"load_partition", "validate_schema"},
    "W3": {"run_simulation"},
}

# E1/E3 configurations
CONFIGS: dict[str, object] = {
    "RAW": "raw",
    "SUPPRESS": "suppress",
    "MODHIDE": "modhide",
    "ABSTRACT": "abstract",
    "ABSTRACT-lifted": "abstract",          # evaluated against the lifted specification
    "VG": dict(ops=("VG",)),
    "VG-salted": dict(ops=("VG",), salted=SALT),
    "VG-runscope": dict(ops=("VG",), scopes="runscope"),
    "TC-60": dict(ops=("TC",), delta=60.0),
    "TC-600": dict(ops=("TC",), delta=600.0),
    "TC-3600": dict(ops=("TC",), delta=3600.0),
    "VG+TC-norank-600": dict(ops=("VG", "TC"), delta=600.0, keep_rank=False),
    "VG+TC+BS": dict(ops=("VG", "TC", "BS"), delta=60.0),
    "FULL-record": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="record", eps=1.0),
    "FULL-bucket": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="bucket", eps=1.0),
    "FULL-bucket-group": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="bucket", eps=1.0, memo="group"),
    "FULL-bucket-fresh": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="bucket", eps=1.0, memo=False),
}

# E2 configurations
E2_CONFIGS: dict[str, object] = {
    "RAW": "raw",
    "SUPPRESS": "suppress",
    "MODHIDE": "modhide",
    "ABSTRACT": "abstract",
    "ABSTRACT-clear": "abstract-clear",
    "VG-salted": dict(ops=("VG",), salted=SALT),
    "VG-release": dict(ops=("VG",), scopes="release"),
    "VG": dict(ops=("VG",)),
    "VG+TC": dict(ops=("VG", "TC"), delta=60.0),
    "VG+TC+BS": dict(ops=("VG", "TC", "BS"), delta=60.0),
    "FULL-record": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="record", eps=1.0),
    "FULL-bucket": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="bucket", eps=1.0),
    "FULL-bucket-group": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="bucket", eps=1.0, memo="group"),
    "FULL-nosize": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="suppress"),
    "NAIVE-DP": dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit="bucket", eps=1.0, memo=False,
                     scopes="release"),
}


def build(g: Graph, spec: Spec, wl: str, cfg: object, key: bytes,
          private_types: dict | None = None, abstract_group: dict | None = None) -> Graph:
    """Produce the release of configuration ``cfg`` for workload ``wl``.

    Baselines are applied after VG (identifier handling equal to the operators);
    'abstract-clear' keeps values in clear (the original abstraction design).
    """
    pt = (private_types or PRIVATE_TYPES)
    ag = (abstract_group or ABSTRACT_GROUP)
    if cfg == "raw":
        return g.copy()
    if cfg in ("suppress", "modhide", "abstract"):
        g = O.sanitise(g, spec, key, ops=("VG",))
    if cfg == "suppress":
        return suppress_all(g)
    if cfg == "modhide":
        return modhide(g, pt[wl])
    if cfg in ("abstract", "abstract-clear"):
        return abstract(g, spec, ag[wl])
    if not isinstance(cfg, dict):
        raise ValueError(f"unknown configuration {cfg!r}")
    kw = dict(cfg)
    sc = kw.get("scopes")
    if sc == "runscope":
        s = O.default_scopes(spec)
        for k in list(s):
            s[k] = "run"
        kw["scopes"] = s
    elif sc == "release":
        s = O.default_scopes(spec, aggregate_inputs=False)
        for k in list(s):
            s[k] = "release"
        kw["scopes"] = s
        kw["aggregate_inputs"] = False
    return O.sanitise(g, spec, key, **kw)
