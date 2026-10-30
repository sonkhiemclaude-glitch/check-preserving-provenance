"""Verifier-side checks.

Every check reads only fields that appear in a published graph: node types,
relations, per-run order (rank or collector order), attribute values compared
by equality or vocabulary membership, numeric values against thresholds, and
time offsets or buckets. The same code runs on raw and sanitised graphs.

Outcomes: PASS (1), FAIL (0), UNDET (-1, required input suppressed).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..data.schemas import FAIL, PASS, SUPP, UNDET, Graph, ranks
from ..data.spec import Constraint, Spec

ORDER_FREE = {"existence", "absence", "absence2", "responded_existence", "not_coexistence"}


# ---------------------------------------------------------------- Declare

def declare_holds(c: Constraint, s: list) -> bool:
    t, a, b = c.template, c.a, c.b
    if t == "init":
        return len(s) > 0 and s[0] == a
    if t == "existence":
        return a in s
    if t == "absence":
        return a not in s
    if t == "absence2":
        return s.count(a) <= 1
    if t == "responded_existence":
        return (a not in s) or (b in s)
    if t == "not_coexistence":
        return not (a in s and b in s)
    if t == "response":
        last_b = max((i for i, x in enumerate(s) if x == b), default=-1)
        return all(i < last_b for i, x in enumerate(s) if x == a)
    if t == "precedence":
        first_a = next((i for i, x in enumerate(s) if x == a), None)
        first_b = next((i for i, x in enumerate(s) if x == b), None)
        return first_b is None or (first_a is not None and first_a < first_b)
    if t == "chain_response":
        return all(i + 1 < len(s) and s[i + 1] == b for i, x in enumerate(s) if x == a)
    if t == "not_succession":
        first_a = next((i for i, x in enumerate(s) if x == a), None)
        if first_a is None:
            return True
        return b not in s[first_a + 1:]
    raise ValueError(t)


def _traces(g: Graph):
    """Per-run activity-type sequence in published order; None if order unknown."""
    act = g.act
    rk = ranks(act)
    df = pd.DataFrame({"run": act["run"].to_numpy(), "type": act["type"].to_numpy(), "rk": rk.to_numpy()})
    ordered = not df["rk"].isna().any()
    if ordered:
        df = df.sort_values(["run", "rk"], kind="mergesort")
    out = {}
    for run, typ in zip(*_split(df["run"].to_numpy(), df["type"].to_numpy())):
        out[run] = typ
    return out, ordered


def _split(keys: np.ndarray, vals: np.ndarray):
    if len(keys) == 0:
        return [], []
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    v = vals[order]
    cut = np.flatnonzero(k[1:] != k[:-1]) + 1
    ks = [k[0]] + list(k[cut])
    vs = np.split(v, cut)
    return ks, [list(x) for x in vs]


def check_conformance(g: Graph, spec: Spec, rows: list):
    traces, ordered = _traces(g)
    for run in g.runs["run"].to_numpy():
        s = traces.get(run, [])
        for c in spec.constraints:
            if not ordered and c.template not in ORDER_FREE:
                out = UNDET
            else:
                out = PASS if declare_holds(c, s) else FAIL
            rows.append(("conf", c.cid, run, out))


# ---------------------------------------------------------------- helpers

def _isnull(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _vals_ok(vals, allowed, gen):
    """Kleene semantics: FAIL if a visible value violates the rule (a missing
    value is a violation), else UNDET if a needed value is suppressed, else PASS."""
    fail = undet = False
    for v in vals:
        if isinstance(v, str) and v == SUPP:
            undet = True
            continue
        if _isnull(v):
            fail = True
            continue
        x = gen(v) if gen else v
        if x not in allowed:
            fail = True
    return FAIL if fail else (UNDET if undet else PASS)


def input_signature(digests) -> str:
    """Canonical, injective encoding of a run's multiset of input digests,
    shared by the raw checker and the VG input aggregate."""
    import json
    return json.dumps(sorted("<none>" if _isnull(d) else str(d) for d in digests), ensure_ascii=False)


# ---------------------------------------------------------------- families

def check_vocab(g: Graph, spec: Spec, rows: list):
    act = g.act
    runs = g.runs["run"].to_numpy()
    for rule in spec.vocab_rules:
        sub = act[act["type"] == rule.act_type]
        gen = spec.generalisers.get(rule.attr)
        if rule.attr not in sub.columns:
            per_run = {}
        else:
            ks, vs = _split(sub["run"].to_numpy(), sub[rule.attr].to_numpy(dtype=object))
            per_run = dict(zip(ks, vs))
        for run in runs:
            vals = per_run.get(run, [])
            rows.append(("vocab", rule.cid, run, _vals_ok(vals, rule.allowed, gen)))


def check_flow(g: Graph, spec: Spec, rows: list):
    """Data-flow closure: every used entity is a source or was generated earlier in the same run."""
    act, ent, edge = g.act, g.ent, g.edge
    rk = pd.Series(ranks(act).to_numpy(), index=act["id"].to_numpy())
    has_rank = not rk.isna().any()
    used = edge[edge["kind"] == "used"]           # src = activity, dst = entity
    gen = edge[edge["kind"] == "wasGeneratedBy"]  # src = entity, dst = activity
    gen_by = pd.Series(gen["dst"].to_numpy(), index=gen["src"].to_numpy())
    gen_by = gen_by[~gen_by.index.duplicated()]
    bad: defaultdict = defaultdict(int)
    undet: set = set()
    u_act = used["src"].to_numpy()
    u_ent = used["dst"].to_numpy()
    et = dict(zip(ent["id"].to_numpy().tolist(), ent["type"].to_numpy(dtype=object)))
    er = dict(zip(ent["id"].to_numpy().tolist(), ent["run"].to_numpy().tolist()))
    ar = dict(zip(act["id"].to_numpy().tolist(), act["run"].to_numpy().tolist()))
    rkd = dict(zip(act["id"].to_numpy().tolist(), ranks(act).to_numpy().tolist()))
    gb: dict = {}
    for n, a in zip(gen["src"].to_numpy().tolist(), gen["dst"].to_numpy().tolist()):
        gb.setdefault(n, a)
    ext = spec.external_types
    for a, n in zip(u_act.tolist(), u_ent.tolist()):
        if et.get(n) in ext:
            continue
        r = ar.get(a)
        p = gb.get(n)
        if p is None or ar.get(p) != r or er.get(n) != r:
            bad[r] += 1
            continue
        if not has_rank:
            undet.add(r)
            continue
        if not rkd[p] < rkd[a]:
            bad[r] += 1
    for run in g.runs["run"].to_numpy():
        if bad.get(run):
            out = FAIL
        elif run in undet:
            out = UNDET
        else:
            out = PASS
        rows.append(("flow", "closure", run, out))


def check_nd(g: Graph, spec: Spec, rows: list):
    """Capture a response digest and, by default, a seed (the reference protocol).

    A hosted-API specification can explicitly set nd_requires_seed=False.
    This changes the declared rule; it does not invent a seed or establish
    deterministic fresh inference from the same prompt.
    """
    act, ent, edge = g.act, g.ent, g.edge
    nd = act[act["type"].isin(spec.nd_types)]
    gen = edge[edge["kind"] == "wasGeneratedBy"]
    resp = ent[ent["type"] == "response"][["id", "digest"]]
    m = gen.merge(resp, left_on="src", right_on="id", how="inner")
    dig_by_act = defaultdict(list)
    for a, d in zip(m["dst"].to_numpy(), m["digest"].to_numpy(dtype=object)):
        dig_by_act[a].append(d)
    n_fail: defaultdict = defaultdict(int)
    n_undet: defaultdict = defaultdict(int)
    seeds = nd["seed"].to_numpy(dtype=object) if "seed" in nd.columns else [None] * len(nd)
    for a, r, sd in zip(nd["id"].to_numpy(), nd["run"].to_numpy(), seeds):
        ds = dig_by_act.get(a, [])
        vals = ([sd] if spec.nd_requires_seed else []) + ds
        if (spec.nd_requires_seed and _isnull(sd)) or len(ds) == 0 or any(_isnull(d) for d in ds):
            n_fail[r] += 1          # a missing value is a violation even if others are suppressed
        elif any(isinstance(v, str) and v == SUPP for v in vals):
            n_undet[r] += 1
    for run in g.runs["run"].to_numpy():
        out = FAIL if n_fail.get(run) else (UNDET if n_undet.get(run) else PASS)
        rows.append(("nd", "capture", run, out))


def replica_groups(g: Graph, spec: Spec):
    """Map group-id -> list of runs, using the published replica attribute.
    Returns None if the attribute is suppressed."""
    col = g.runs[spec.replica_attr].to_numpy(dtype=object)
    if any(isinstance(v, str) and v == SUPP for v in col):
        return None
    groups = defaultdict(list)
    for r, c in zip(g.runs["run"].to_numpy(), col):
        groups[c].append(r)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def _sig_by_run(df: pd.DataFrame, col: str):
    out = defaultdict(list)
    for r, v in zip(df["run"].to_numpy(), df[col].to_numpy(dtype=object)):
        out[r].append(v)
    return {r: tuple(sorted(map(str, v))) for r, v in out.items()}


def check_replicas(g: Graph, spec: Spec, rows: list):
    """Replica agreement on inputs (content and record counts) and outputs."""
    groups = replica_groups(g, spec)
    ent = g.ent
    src = ent[ent["type"] == spec.source_type]
    res = ent[ent["type"] == spec.result_type]
    if "input_digest" in g.runs.columns:
        in_sig = dict(zip(g.runs["run"].to_numpy(), g.runs["input_digest"].to_numpy(dtype=object)))
    else:
        per = defaultdict(list)
        for r, d in zip(src["run"].to_numpy(), src["digest"].to_numpy(dtype=object)):
            per[r].append(d)
        in_sig = {r: input_signature(per.get(r, [])) for r in g.runs["run"].to_numpy()}
    cnt_sig = _sig_by_run(src, "n_records")
    out_sig = _sig_by_run(res, "digest")
    if groups is None:
        rows.append(("rep_meta", "suppressed", -1, UNDET))
        return
    for _, members in groups.items():
        key = tuple(members)
        for name, sig in (("inputs", in_sig), ("counts", cnt_sig), ("outputs", out_sig)):
            vals = [sig.get(r, ()) for r in members]
            flat = [x for v in vals if isinstance(v, tuple) for x in v] + [v for v in vals if not isinstance(v, tuple)]
            supp = any(isinstance(x, str) and SUPP in x for x in flat)
            visible_disagree = (not supp) and not all(v == vals[0] for v in vals)
            if visible_disagree:
                out = FAIL
            elif supp:
                out = UNDET
            else:
                out = PASS
            rows.append(("rep_" + name, "agree", key, out))


def check_thresholds(g: Graph, spec: Spec, rows: list):
    for rule in spec.thresholds:
        sub = g.ent[g.ent["type"] == rule.ent_type]
        per = defaultdict(list)
        if rule.attr in sub.columns:
            for r, v in zip(sub["run"].to_numpy(), sub[rule.attr].to_numpy(dtype=object)):
                per[r].append(v)
        for run in g.runs["run"].to_numpy():
            vals = per.get(run, [])
            vis = [v for v in vals if not (isinstance(v, str) and v == SUPP)]
            if any(_isnull(v) or float(v) < rule.minimum for v in vis):
                out = FAIL
            elif len(vis) < len(vals):
                out = UNDET
            else:
                out = PASS
            rows.append(("thr", rule.cid, run, out))


def check_timed(g: Graph, spec: Spec, rows: list):
    act = g.act
    if "b_start" in act.columns:
        delta = float(g.meta["delta"])
        s = act["b_start"].to_numpy(dtype=float) * delta
        e = act["b_end"].to_numpy(dtype=float) * delta
        timed = True
    elif "start" in act.columns and pd.api.types.is_numeric_dtype(act["start"]):
        s = act["start"].to_numpy(dtype=float)
        e = act["end"].to_numpy(dtype=float)
        timed = True
    else:
        timed = False
    runs = act["run"].to_numpy()
    types = act["type"].to_numpy()
    for rule in spec.timed:
        if not timed:
            for run in g.runs["run"].to_numpy():
                rows.append(("time", rule.cid, run, UNDET))
            continue
        df = pd.DataFrame({"run": runs, "s": s, "e": e, "t": types})
        if rule.kind == "span":
            agg = df.groupby("run").agg(smin=("s", "min"), emax=("e", "max"))
            ok = (agg["emax"] - agg["smin"]) <= rule.limit
        else:
            sub = df[df["t"] == rule.act_type]
            agg = (sub["e"] - sub["s"]).groupby(sub["run"]).max()
            ok = agg <= rule.limit
        okd = ok.to_dict()
        for run in g.runs["run"].to_numpy():
            rows.append(("time", rule.cid, run, PASS if okd.get(run, True) else FAIL))


FAMILIES = {
    "conf": check_conformance,
    "vocab": check_vocab,
    "flow": check_flow,
    "nd": check_nd,
    "rep": check_replicas,
    "thr": check_thresholds,
    "time": check_timed,
}


def run_checks(g: Graph, spec: Spec, families=None) -> pd.DataFrame:
    rows: list = []
    for name, fn in FAMILIES.items():
        if families is None or name in families:
            fn(g, spec, rows)
    return pd.DataFrame(rows, columns=["family", "cid", "key", "outcome"])


def align(sanitised: Graph, checks: pd.DataFrame) -> pd.DataFrame:
    """Translate run keys of a sanitised graph back to raw run ids (publisher side)."""
    inv = sanitised.meta.get("run_map_inv")
    if not inv:
        return checks
    def tr(k):
        if isinstance(k, tuple):
            return tuple(sorted(inv[x] for x in k))
        if k == -1:
            return k
        return inv[k]
    out = checks.copy()
    out["key"] = [tr(k) for k in out["key"]]
    return out


def _norm(k):
    if isinstance(k, tuple):
        return repr(tuple(int(x) for x in k))
    return repr(int(k))


def compare(raw: pd.DataFrame, san: pd.DataFrame) -> pd.DataFrame:
    """Join raw and sanitised outcomes on (family, cid, key); raw checks missing
    in the sanitised result count as UNDET (the verifier cannot run them)."""
    r = raw.copy()
    s = san.copy()
    r["k"] = r["key"].map(_norm)
    s["k"] = s["key"].map(_norm)
    s = s[s["family"] != "rep_meta"]
    m = r.merge(s[["family", "cid", "k", "outcome", "key"]].rename(columns={"key": "key_san"}),
                on=["family", "cid", "k"], how="outer", suffixes=("_raw", "_san"), indicator=True)
    # raw instances missing from the release count as UNDET there; instances that
    # exist only in the release are spurious and count as changed outcomes
    m["key"] = m["key"].where(m["_merge"] != "right_only", m["key_san"])
    m["outcome_raw"] = m["outcome_raw"].fillna(-2).astype(int)
    m["outcome_san"] = m["outcome_san"].fillna(UNDET).astype(int)
    m["same"] = (m["outcome_raw"] == m["outcome_san"]) & (m["_merge"] != "right_only")
    return m.drop(columns=["key_san", "_merge"])
