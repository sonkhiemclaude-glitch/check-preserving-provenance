"""Sanitisation operators (paper Sec. 4).

Operators (paper notation):
  VG   vocabulary generalisation with keyed, scope-limited pseudonyms and
       check-sufficient aggregation of input digests
  TC   rank-preserving timestamp coarsening
  BS   block shuffle of rank/time slots among consecutive, independent
       activities of the same type
  DPN  memoised two-sided geometric noise on numeric attributes of source
       partitions; derived numeric attributes suppressed
Baselines live in provpub.baselines. Naive variants for the ablations
(published salt, all run scope, no rank, fresh noise) are selected by parameters.

All randomness is derived from a secret release key with HMAC-SHA256, so a
sanitiser applied shard by shard yields the same release as a sequential
run (shards are sets of whole runs).
"""
from __future__ import annotations

import hashlib
import hmac
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from ..data.schemas import ATTR_CLASS, SUPP, Graph, ranks
from ..data.spec import Spec

PS = "ps:"


class CollisionError(RuntimeError):
    pass


# ------------------------------------------------------------------ keys

def _hm(key: bytes, msg: str) -> bytes:
    return hmac.digest(key, msg.encode("utf-8"), "sha256")


class KeyRing:
    """Release key and derived run-scope keys."""

    def __init__(self, release_key: bytes):
        self.k = release_key
        self._run: dict = {}

    def scope_key(self, scope) -> bytes:
        """release -> K; ("group", v) -> HMAC(K, "grp|v"); run r -> HMAC(K, "run|r").
        Derived keys are full 256-bit HMAC outputs (no truncation)."""
        if scope == "release":
            return self.k
        k = self._run.get(scope)
        if k is None:
            if isinstance(scope, tuple) and scope[0] == "group":
                k = _hm(self.k, f"grp|{scope[1]}")
            else:
                k = _hm(self.k, f"run|{scope}")
            self._run[scope] = k
        return k

    def uniform(self, label: str) -> tuple[float, float]:
        d = _hm(self.k, "dp|" + label)
        a = int.from_bytes(d[:8], "big")
        b = int.from_bytes(d[8:16], "big")
        return (a + 0.5) / 2**64, (b + 0.5) / 2**64

    def permutation(self, label: str, n: int) -> np.ndarray:
        """Fisher-Yates permutation driven by an HMAC counter stream (PRF-derived;
        reaches every permutation under the PRF assumption, up to 2^-64 modulo bias)."""
        perm = np.arange(n)
        for i in range(n - 1, 0, -1):
            r = int.from_bytes(_hm(self.k, f"perm|{label}|{i}")[:16], "big")
            j = r % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm


def salted_pseudonym(salt: bytes, attr: str, value, nbytes: int = 8) -> str:
    """Naive published-salt variant (unkeyed SHA-256). Shared with the attacks,
    which know the construction (Kerckhoffs)."""
    return PS + hashlib.sha256(salt + f"ps|{attr}|{value}".encode()).hexdigest()[: 2 * nbytes]


def pseudonymise(values, scopes, attr: str, ring: KeyRing, nbytes: int = 8,
                 salted: bytes | None = None, check: bool = True):
    """Map each (scope, value) to PS + hex(trunc(HMAC(K_scope, attr|value))).

    Null and suppressed values are kept as they are. With `salted`, an
    unkeyed SHA-256 with a public salt is used instead (naive variant).
    Raises CollisionError if two distinct values in one scope collide.
    """
    out = np.empty(len(values), dtype=object)
    cache: dict = {}
    seen: dict = defaultdict(dict)
    for i, (v, sc) in enumerate(zip(values, scopes)):
        if v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v == SUPP):
            out[i] = v
            continue
        key = (sc, v)
        p = cache.get(key)
        if p is None:
            msg = f"ps|{attr}|{v}"
            if salted is not None:
                p = salted_pseudonym(salted, attr, v, nbytes)
            else:
                p = PS + _hm(ring.scope_key(sc), msg)[:nbytes].hex()
            if check:
                prev = seen[sc].get(p)
                if prev is not None and prev != v:
                    raise CollisionError(f"{attr} collision in scope {sc}")
                seen[sc][p] = v
            cache[key] = p
        out[i] = p
    return out


# ------------------------------------------------------------------ VG

def default_scopes(spec: Spec, aggregate_inputs: bool = True) -> dict:
    """Scope rule: the smallest key scope that contains every pair of runs whose
    values some declared check compares. The replica attribute is compared across
    all runs (grouping) -> release scope; the input aggregate and result digests
    are compared only within a replica group -> group scope; everything else is
    compared at most within a run -> run scope."""
    return {
        ("runs", spec.replica_attr): "release",
        ("runs", "input_digest"): "group",
        ("ent", "digest", spec.result_type): "group",
        ("ent", "digest", spec.source_type): "run" if aggregate_inputs else "release",
        ("runs", "*", "*"): "run",
        ("ent", "*", "*"): "run",
        ("act", "*", "*"): "run",
    }


def _scope_for(scopes: dict, table: str, attr: str, etype: str = "*") -> str:
    for k in ((table, attr, etype), (table, attr, "*"), (table, attr), (table, attr, "*"), (table, "*", "*")):
        if k in scopes:
            return scopes[k]
    return "run"


def vg(g: Graph, spec: Spec, ring: KeyRing, scopes: dict | None = None,
       aggregate_inputs: bool = True, salted: bytes | None = None,
       renumber: bool = True) -> Graph:
    if scopes is None:
        scopes = default_scopes(spec, aggregate_inputs)
    h = g.copy()
    runs, act, ent = h.runs, h.act, h.ent

    # check-sufficient aggregation: one keyed digest of each run's input multiset,
    # computed with the same canonical signature the raw checker uses
    from ..checks.checks import input_signature
    if aggregate_inputs:
        src = g.ent[g.ent["type"] == spec.source_type]
        agg = defaultdict(list)
        for r, d in zip(src["run"].to_numpy(), src["digest"].to_numpy(dtype=object)):
            agg[r].append(d)
        runs["input_digest"] = [input_signature(agg.get(r, [])) for r in runs["run"].to_numpy()]
    group_of = dict(zip(g.runs["run"].to_numpy(), g.runs[spec.replica_attr].astype(str).to_numpy()))

    def sc_list(table, attr, run_col, etypes=None):
        out = []
        for i, r in enumerate(run_col):
            et = etypes[i] if etypes is not None else "*"
            s = _scope_for(scopes, table, attr, et)
            if s == "release":
                out.append("release")
            elif s == "group":
                out.append(("group", group_of[r]))
            else:
                out.append(int(r))
        return out

    # runs table
    for col in [c for c in runs.columns if c != "run"]:
        if col in ("date",):
            continue
        sc = sc_list("runs", col, runs["run"].to_numpy())
        runs[col] = pseudonymise(runs[col].to_numpy(dtype=object), sc, col, ring, salted=salted)

    # activities
    for col in [c for c in act.columns if c in ATTR_CLASS]:
        cls = ATTR_CLASS[col]
        vals = act[col].to_numpy(dtype=object)
        if cls == "vocab":
            voc = spec.vocab.get(col, frozenset())
            vals = np.array([v if (v is None or (isinstance(v, float) and math.isnan(v)) or isinstance(v, str)) else str(v)
                             for v in vals], dtype=object)
            keep = np.array([(v in voc) if isinstance(v, str) else True for v in vals])
            sc = sc_list("act", col, act["run"].to_numpy())
            ps = pseudonymise(vals, sc, col, ring, salted=salted)
            act[col] = np.where(keep, vals, ps)
        elif cls == "path":
            gen = spec.generalisers[col]
            act[col] = [gen(v) if isinstance(v, str) and v != SUPP else v for v in vals]
        elif cls in ("ident", "text", "digest"):
            sc = sc_list("act", col, act["run"].to_numpy())
            act[col] = pseudonymise(vals, sc, col, ring, salted=salted)

    # entities
    et = ent["type"].to_numpy(dtype=object)
    for col in [c for c in ent.columns if c in ATTR_CLASS]:
        cls = ATTR_CLASS[col]
        if cls in ("ident", "text", "digest"):
            sc = sc_list("ent", col, ent["run"].to_numpy(), et)
            ent[col] = pseudonymise(ent[col].to_numpy(dtype=object), sc, col, ring, salted=salted)
        elif cls == "path":
            gen = spec.generalisers[col]
            ent[col] = [gen(v) if isinstance(v, str) and v != SUPP else v for v in ent[col].to_numpy(dtype=object)]

    if renumber:
        h = _renumber(h, ring)
    h.meta.setdefault("ops", []).append("VG")
    return h


def _renumber(h: Graph, ring: KeyRing) -> Graph:
    """Replace run and node identifiers by keyed random labels so that
    identifiers do not reveal collection order. The inverse run map stays with
    the publisher (meta) and is not part of the release."""
    old_runs = h.runs["run"].to_numpy()
    new_runs = np.array([int.from_bytes(_hm(ring.k, f"rid|{r}")[:6], "big") for r in old_runs])
    if len(set(new_runs)) != len(new_runs):
        raise CollisionError("run id collision")
    rmap = dict(zip(old_runs, new_runs))
    prev_inv = h.meta.get("run_map_inv")
    h.meta["run_map_inv"] = {int(n): (prev_inv[int(o)] if prev_inv else int(o)) for o, n in rmap.items()}
    h.runs["run"] = new_runs
    h.act["run"] = h.act["run"].map(rmap)
    h.ent["run"] = h.ent["run"].map(rmap)
    h.edge["run"] = h.edge["run"].map(rmap)
    ids = np.concatenate([h.act["id"].to_numpy(), h.ent["id"].to_numpy()])
    new_ids = np.array([int.from_bytes(_hm(ring.k, f"nid|{i}")[:7], "big") >> 4 for i in ids], dtype=np.int64)  # 52 bits: exact in float64
    if len(set(new_ids)) != len(new_ids):
        raise CollisionError("node id collision")
    nmap = pd.Series(new_ids, index=ids)
    h.act["id"] = nmap.reindex(h.act["id"].to_numpy()).to_numpy()
    h.ent["id"] = nmap.reindex(h.ent["id"].to_numpy()).to_numpy()
    h.edge["src"] = nmap.reindex(h.edge["src"].to_numpy()).to_numpy()
    h.edge["dst"] = nmap.reindex(h.edge["dst"].to_numpy()).to_numpy()
    # publish rows in key-determined order, not collection order (all tables)
    h.edge = h.edge.sort_values(["kind", "src", "dst"], kind="mergesort").reset_index(drop=True)
    h.act = h.act.sort_values("id", kind="mergesort").reset_index(drop=True)
    h.ent = h.ent.sort_values("id", kind="mergesort").reset_index(drop=True)
    h.runs = h.runs.sort_values("run", kind="mergesort").reset_index(drop=True)
    return h


# ------------------------------------------------------------------ TC

def tc(g: Graph, delta: float, date_granularity: float | None = None,
       keep_rank: bool = True) -> Graph:
    """Rank-preserving coarsening. Publishes per-run rank and offset buckets
    floor((t - run_start)/delta); absolute times are dropped and the run start
    is published only at date_granularity (or not at all).
    keep_rank=False gives the naive bucketing variant (ties)."""
    h = g.copy()
    act = h.act
    rk = ranks(act)
    start = act["start"].to_numpy(dtype=float)
    end = act["end"].to_numpy(dtype=float)
    r0 = act.groupby("run")["start"].transform("min").to_numpy(dtype=float)
    act["b_start"] = np.floor((start - r0) / delta).astype(np.int64)
    act["b_end"] = np.floor((end - r0) / delta).astype(np.int64)
    if keep_rank:
        act["rank"] = rk.to_numpy().astype(np.int64)
    run_start = act.groupby("run")["start"].min()
    if date_granularity:
        h.runs["date"] = h.runs["run"].map((run_start // date_granularity).astype(np.int64))
    h.act = act.drop(columns=["start", "end", "seq"])
    h.meta["delta"] = delta
    h.meta.setdefault("ops", []).append("TC" if keep_rank else "TC-norank")
    return h


def naive_order(g: Graph, ring: KeyRing) -> Graph:
    """For the no-rank variant: the verifier orders by start bucket and breaks
    ties by the published (random) node id. Materialise that as a rank."""
    h = g.copy()
    a = h.act.sort_values(["run", "b_start", "id"], kind="mergesort")
    a["rank"] = a.groupby("run").cumcount()
    h.act = a.sort_index()
    return h


# ------------------------------------------------------------------ BS

def blocks(g: Graph):
    """Consecutive activities of one type in a run that are pairwise
    independent (no direct generation-use dependency between members)."""
    a = g.act.sort_values(["run", "rank"], kind="mergesort")
    run = a["run"].to_numpy()
    typ = a["type"].to_numpy(dtype=object)
    ids = a["id"].to_numpy()
    idx = a.index.to_numpy()
    newblk = np.ones(len(a), dtype=bool)
    newblk[1:] = (run[1:] != run[:-1]) | (typ[1:] != typ[:-1])
    bid = np.cumsum(newblk) - 1
    # direct dependencies between activities: a' generated n, a used n
    e = g.edge
    gen = e[e["kind"] == "wasGeneratedBy"][["src", "dst"]].rename(columns={"src": "n", "dst": "prod"})
    use = e[e["kind"] == "used"][["src", "dst"]].rename(columns={"src": "cons", "dst": "n"})
    dep = use.merge(gen, on="n")
    inf = e[e["kind"] == "wasInformedBy"][["src", "dst"]].rename(columns={"src": "cons", "dst": "prod"})
    dep = pd.concat([dep[["prod", "cons"]], inf], ignore_index=True)
    blk_of = pd.Series(bid, index=ids)
    bp = blk_of.reindex(dep["prod"].to_numpy()).to_numpy()
    bc = blk_of.reindex(dep["cons"].to_numpy()).to_numpy()
    dependent = set(bp[(bp == bc) & ~np.isnan(bp.astype(float))].astype(int).tolist())
    sizes = np.bincount(bid)
    out = []
    starts = np.flatnonzero(newblk)
    ends = np.append(starts[1:], len(a))
    for b, (s, t) in enumerate(zip(starts, ends)):
        if sizes[b] >= 2 and b not in dependent:
            out.append((run[s], idx[s:t]))
    return out


def bs(g: Graph, ring: KeyRing) -> Graph:
    """Randomly reassign (rank, b_start, b_end) slots within each block."""
    if "rank" not in g.act.columns:
        raise ValueError("BS expects a rank-coarsened graph (apply TC first)")
    h = g.copy()
    act = h.act
    cols = [c for c in ("rank", "b_start", "b_end") if c in act.columns]
    slots = act[cols].to_numpy().copy()
    pos = {ix: i for i, ix in enumerate(act.index.to_numpy())}
    for run, members in blocks(h):
        rows = np.array([pos[m] for m in members])
        orig = h.meta.get("run_map_inv", {}).get(int(run), int(run))
        first = int(slots[rows, 0].min())          # block position, independent of sharding
        perm = ring.permutation(f"bs|{orig}|{first}|{len(rows)}", len(rows))
        slots[rows] = slots[rows[perm]]
    act[cols] = slots
    h.meta.setdefault("ops", []).append("BS")
    return h


# ------------------------------------------------------------------ DPN

def two_sided_geometric(u1: float, u2: float, alpha: float) -> int:
    """Discrete Laplace sample, P(Z=z) proportional to alpha^|z|, from two uniforms."""
    la = math.log(alpha)
    g1 = math.floor(math.log(u1) / la)
    g2 = math.floor(math.log(u2) / la)
    return g1 - g2


def dpn(g: Graph, spec: Spec, ring: KeyRing, eps: float, unit: str = "record",
        max_row_bytes: int = 4096, bucket_width: tuple[float, float] = (1000, 500_000),
        memo=True, clamp: bool = False, sampler: str = "exact") -> Graph:
    """Two-sided geometric noise on n_records and n_bytes of source partitions
    (paper Sec. 4.4, Theorem 4; alpha_i = exp(-eps_i/delta_i), eps_1 = eps_2 = eps/2).

    sampler='exact' (default) uses the rational-scale exact discrete-Laplace
    sampler of methods/exact_dlap.py. sampler='inversion' remains available
    for reproducing historical results and uses double-precision inversion.

    eps is split equally between the two attributes. Sensitivity:
      unit='record': one record changes n_records by <= 1 and n_bytes by <= max_row_bytes
                     (rows are assumed to be at most max_row_bytes long)
      unit='bucket': metric privacy at the scale of bucket_width (counts, bytes)
    The memo key is the publisher-assigned stable partition identifier
    ('part_id'), never the content, so a change of content cannot merge or
    split memo keys:
      memo=True     partition scope: one draw per partition identifier
      memo='group'  input-set scope: one draw per (identifier, the run's multiset
                    of input identifiers); replicas share noise
      memo=False    fresh draw per occurrence (naive variant)
    Partitions without an identifier are keyed by their node id.
    Values are not clamped (clamp=True clamps at 0 as post-processing).
    Derived numeric attributes are suppressed.
    """
    import json as _json
    h = g.copy()
    ent = h.ent
    is_src = (ent["type"] == spec.source_type).to_numpy()
    ids = ent["id"].to_numpy()
    pid = ent["part_id"].to_numpy(dtype=object) if "part_id" in ent.columns else np.full(len(ent), None, dtype=object)
    pid = np.array([p if isinstance(p, str) and p != SUPP else f"node:{i}" for p, i in zip(pid, ids)], dtype=object)
    if unit == "suppress":       # endpoint of the trade-off: publish no sizes at all
        for col in ("n_records", "n_bytes", "out_rows", "out_bytes"):
            if col in ent.columns:
                vals = ent[col].to_numpy(dtype=object).copy()
                if col.startswith("n_"):
                    vals[is_src] = SUPP
                else:
                    vals[:] = SUPP
                ent[col] = vals
        h.meta.setdefault("ops", []).append("DPN[suppress]")
        return h
    if unit == "record":
        sens = {"n_records": 1.0, "n_bytes": float(max_row_bytes)}
    elif unit == "bucket":
        sens = {"n_records": float(bucket_width[0]), "n_bytes": float(bucket_width[1])}
    else:
        raise ValueError(unit)
    eps_attr = eps / 2.0
    rr = ent["run"].to_numpy()
    if memo == "group":
        sets: dict = {}
        for r_, p_ in zip(rr[is_src], pid[is_src]):
            sets.setdefault(r_, []).append(p_)
        set_key = {r_: _json.dumps(sorted(v)) for r_, v in sets.items()}
    for col in ("n_records", "n_bytes"):
        if col not in ent.columns:
            continue
        alpha = math.exp(-eps_attr / sens[col])
        vals = ent[col].to_numpy(dtype=object).copy()
        for i in np.flatnonzero(is_src):
            v = vals[i]
            if v is None or (isinstance(v, str)):
                continue
            if memo == "group":
                label = f"{col}|pid|{pid[i]}|set|{set_key[rr[i]]}"
            elif memo:
                label = f"{col}|pid|{pid[i]}"
            else:
                label = f"{col}|occ|{ids[i]}"
            if sampler == "exact":
                from .exact_dlap import keyed_exact_noise
                z = keyed_exact_noise(ring.k, label, sens[col], eps_attr)
            elif sampler == "inversion":
                u1, u2 = ring.uniform(label)
                z = two_sided_geometric(u1, u2, alpha)
            else:
                raise ValueError(f"unknown sampler {sampler!r}")
            y = int(v) + z
            vals[i] = max(0, y) if clamp else y
        ent[col] = vals
    for col in ("out_rows", "out_bytes"):
        if col in ent.columns:
            ent[col] = SUPP
    tag = memo if isinstance(memo, str) else ("partition" if memo else "fresh")
    h.meta.setdefault("ops", []).append(f"DPN[{unit},eps={eps:g},{tag}]")
    return h


def flip_probability(x: int, threshold: int, eps_attr: float, sens: float) -> float:
    """Exact probability that [x + Z >= threshold] differs from [x >= threshold]
    under two-sided geometric noise with alpha = exp(-eps_attr/sens)."""
    alpha = math.exp(-eps_attr / sens)
    if x >= threshold:
        k = x - threshold + 1          # need Z <= -k
    else:
        k = threshold - x              # need Z >= k
    return alpha ** k / (1 + alpha)


# ------------------------------------------------------------------ pipelines

def sanitise(g: Graph, spec: Spec, key: bytes, ops=("VG", "TC", "BS", "DPN"), *,
             delta: float = 60.0, date_granularity: float | None = 7 * 86400,
             eps: float = 1.0, unit: str = "bucket", memo: bool | str = True,
             scopes: dict | None = None, aggregate_inputs: bool = True,
             salted: bytes | None = None, keep_rank: bool = True,
             bucket_width=(1000, 500_000), max_row_bytes: int = 4096,
             sampler: str = "exact") -> Graph:
    """Composition S = BS o TC o VG o DPN (paper Theorem 5).

    Apply operators in the order DPN, VG, TC, BS. The order matters: DPN reads
    raw partition identifiers (before VG pseudonymises them), and BS needs the
    ranks that TC publishes."""
    ring = KeyRing(key)
    h = g
    if "DPN" in ops:
        h = dpn(h, spec, ring, eps, unit=unit, memo=memo, bucket_width=bucket_width,
                max_row_bytes=max_row_bytes, sampler=sampler)
    if "VG" in ops:
        h = vg(h, spec, ring, scopes=scopes, aggregate_inputs=aggregate_inputs, salted=salted)
    if "TC" in ops:
        h = tc(h, delta, date_granularity, keep_rank=keep_rank)
        if not keep_rank:
            h = naive_order(h, ring)
    if "BS" in ops:
        h = bs(h, ring)
    return h
