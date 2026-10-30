"""Declared pipeline specification.

The specification is public: verifiers hold it. It fixes
  - Declare constraints over activity types (untimed, LTLf semantics),
  - a public vocabulary per attribute (approved tools, models, settings),
  - public generalisers (e.g. a path is mapped to 'allowed' or 'other'),
  - which activity types are nondeterministic (LLM invocations),
  - the replica-grouping attribute and the result entity type,
  - optional numeric thresholds and timed constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

DECLARE_TEMPLATES = (
    "init", "existence", "absence2", "response", "precedence", "chain_response",
    "responded_existence", "not_coexistence", "not_succession", "absence",
)


@dataclass(frozen=True)
class Constraint:
    template: str
    a: str
    b: str | None = None

    def __post_init__(self):
        if self.template not in DECLARE_TEMPLATES:
            raise ValueError(self.template)
        if self.b is not None and self.b == self.a:
            raise ValueError("binary Declare templates require two distinct activity types")

    @property
    def cid(self) -> str:
        return f"{self.template}({self.a}{',' + self.b if self.b else ''})"


@dataclass(frozen=True)
class VocabRule:
    """For every activity of type act_type, attribute attr must map into allowed."""
    act_type: str
    attr: str
    allowed: frozenset

    @property
    def cid(self) -> str:
        return f"vocab({self.act_type}.{self.attr})"


@dataclass(frozen=True)
class ThresholdRule:
    """Every source entity of type ent_type in a run has attr >= minimum."""
    ent_type: str
    attr: str
    minimum: float

    @property
    def cid(self) -> str:
        return f"thr({self.ent_type}.{self.attr}>={self.minimum:g})"


@dataclass(frozen=True)
class TimedRule:
    """kind='span': run span <= limit seconds.
    kind='max_dur': every activity of act_type lasts <= limit seconds."""
    kind: str
    limit: float
    act_type: str | None = None

    @property
    def cid(self) -> str:
        return f"time({self.kind}{':' + self.act_type if self.act_type else ''}<={self.limit:g})"


def path_generaliser(allowed_prefixes: tuple[str, ...]) -> Callable[[str], str]:
    cats = {"allowed", "other"}

    def g(v):
        if v is None:
            return None
        if v in cats:  # idempotent on published categories
            return v
        return "allowed" if any(str(v).startswith(p) for p in allowed_prefixes) else "other"
    g.kind = "path_prefix"  # type: ignore[attr-defined]  # recorded so the spec can be serialised (spec_io.py)
    g.allowed_prefixes = tuple(allowed_prefixes)  # type: ignore[attr-defined]
    return g


@dataclass
class Spec:
    name: str
    alphabet: frozenset
    constraints: list[Constraint]
    vocab: dict[str, frozenset]               # attr -> public vocabulary
    vocab_rules: list[VocabRule]
    generalisers: dict[str, Callable]         # attr -> public generaliser
    nd_types: frozenset                       # nondeterministic activity types
    replica_attr: str = "config"              # run attribute defining replica groups
    source_type: str = "data_partition"
    result_type: str = "result"
    external_types: frozenset = frozenset({"data_partition", "prompt"})  # inputs from outside the run
    thresholds: list[ThresholdRule] = field(default_factory=list)
    timed: list[TimedRule] = field(default_factory=list)
    # E1-E4 retain the scripted seed-and-response rule. Hosted APIs that
    # expose no seed can explicitly declare response capture alone for E5.
    nd_requires_seed: bool = True

    def __post_init__(self):
        """Well-typed specification (Assumption 2): vocabulary rules read
        vocabulary-class attributes with allowed sets inside the public
        vocabulary, or generalisable attributes with allowed sets inside the
        generaliser's codomain; thresholds read numeric roots."""
        from .schemas import ATTR_CLASS
        if not isinstance(self.nd_requires_seed, bool):
            raise ValueError("nd_requires_seed must be a boolean")
        for r in self.vocab_rules:
            cls = ATTR_CLASS.get(r.attr)
            if cls == "vocab":
                if r.attr in self.generalisers:
                    raise ValueError(f"{r.cid}: vocabulary attribute must not have a generaliser")
                if not set(r.allowed) <= set(self.vocab.get(r.attr, ())):
                    raise ValueError(f"{r.cid}: allowed set must lie inside the public vocabulary")
            elif cls == "path":
                if r.attr not in self.generalisers:
                    raise ValueError(f"{r.cid}: generalisable attribute needs a generaliser")
            else:
                raise ValueError(f"{r.cid}: vocabulary rules may only read vocabulary or generalisable attributes")
        for v in self.vocab.values():
            if any(str(x).startswith("ps:") for x in v):
                raise ValueError("vocabulary values must not use the pseudonym prefix")
        for t in self.thresholds:
            if ATTR_CLASS.get(t.attr) != "num_root":
                raise ValueError(f"{t.cid}: thresholds read numeric roots only")


def lift_spec(spec: "Spec", group_types: set, composite: str = "prepare_inputs") -> "Spec":
    """Specification lifted to a structural-abstraction view: constraints,
    vocabulary and timed rules that mention collapsed activity types are
    dropped (the verifier of a user view only receives the view)."""
    import dataclasses
    g = set(group_types)
    cons = [c for c in spec.constraints if c.a not in g and (c.b is None or c.b not in g)]
    vr = [r for r in spec.vocab_rules if r.act_type not in g]
    tr = [t for t in spec.timed if t.act_type is None or t.act_type not in g]
    return dataclasses.replace(spec, alphabet=frozenset((set(spec.alphabet) - g) | {composite}),
                               constraints=cons, vocab_rules=vr, timed=tr,
                               nd_types=frozenset(set(spec.nd_types) - g))
