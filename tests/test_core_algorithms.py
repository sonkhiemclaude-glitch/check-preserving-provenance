"""Core algorithm tests (ported from the code that produced the reference results).

Covers: Declare semantics (vs an LTLf evaluator and pm4py), Theorems 1-3 and 5
(exact preservation), Theorem 2 (timed bound), Theorem 3 (block shuffle),
parallel = sequential, ablation counterexamples, collision detection, Theorem 4
memoisation scopes, Proposition 4 (flip formula), PROV-JSON / OTel round
trips, Assumptions 2-3, Definition 1 (spurious instances), three-valued
semantics, salted adversary, block-shuffle privacy, lifted specification."""
import math

import numpy as np
import pandas as pd
import pytest

from provpub.baselines import structural as B
from provpub.checks.checks import _traces, align, compare, declare_holds, run_checks
from provpub.data.schemas import concat_graphs, split_by_runs
from provpub.data.spec import Constraint
from provpub.data.synthetic import GENERATORS
from provpub.methods import operators as O

KEY = bytes(range(32))
PM4PY_NAME = {"init": "init", "existence": "existence", "absence": "absence", "response": "response",
              "precedence": "precedence", "chain_response": "chainresponse",
              "responded_existence": "responded_existence", "not_coexistence": "noncoexistence"}
# pm4py's 'nonsuccession' is implemented as the negation of 'succession' (see
# pm4py/algo/conformance/declare/variants/classic.py), not as the Declare
# NotSuccession formula G(a -> !F b); it is therefore checked only against LTLf.


def ltlf(f, s, i=0):
    """Minimal LTLf evaluator on a finite trace s at position i."""
    op = f[0]
    if op == "atom":
        return i < len(s) and s[i] == f[1]
    if op == "not":
        return not ltlf(f[1], s, i)
    if op == "and":
        return ltlf(f[1], s, i) and ltlf(f[2], s, i)
    if op == "or":
        return ltlf(f[1], s, i) or ltlf(f[2], s, i)
    if op == "imp":
        return (not ltlf(f[1], s, i)) or ltlf(f[2], s, i)
    if op == "X":   # strong next
        return i + 1 < len(s) and ltlf(f[1], s, i + 1)
    if op == "F":
        return any(ltlf(f[1], s, j) for j in range(i, len(s)))
    if op == "G":
        return all(ltlf(f[1], s, j) for j in range(i, len(s)))
    if op == "U":
        for j in range(i, len(s)):
            if ltlf(f[2], s, j):
                return all(ltlf(f[1], s, k) for k in range(i, j))
        return False
    raise ValueError(op)


def formula(c):
    A, B = ("atom", c.a), ("atom", c.b)
    return {
        "init": A, "existence": ("F", A), "absence": ("not", ("F", A)),
        "absence2": ("not", ("F", ("and", A, ("X", ("F", A))))),
        "responded_existence": ("imp", ("F", A), ("F", B)),
        "response": ("G", ("imp", A, ("F", B))),
        "precedence": ("or", ("U", ("not", B), A), ("G", ("not", B))),
        "chain_response": ("G", ("imp", A, ("X", B))),
        "not_coexistence": ("not", ("and", ("F", A), ("F", B))),
        "not_succession": ("G", ("imp", A, ("not", ("F", B)))),
    }[c.template]


def test_declare_matches_ltlf():
    rng = np.random.default_rng(2)
    alph = list("abcd")
    from provpub.data.spec import DECLARE_TEMPLATES
    for _ in range(600):
        s = [str(x) for x in rng.choice(alph, size=int(rng.integers(0, 10)))]
        for t in DECLARE_TEMPLATES:
            for a in alph:
                for b in alph:
                    if a == b:
                        continue
                    c = Constraint(t, a, b)
                    assert declare_holds(c, s) == ltlf(formula(c), s), (c, s)


def pm4py_agreement(n_traces=400, seed=0):
    """Agreement of declare_holds with pm4py.conformance_declare on random traces."""
    pm4py = pytest.importorskip("pm4py")
    rng = np.random.default_rng(seed)
    alph = list("abcd")
    traces = [list(rng.choice(alph, size=int(rng.integers(1, 13)))) for _ in range(n_traces)]
    rows = []
    for ci, tr in enumerate(traces):
        for k, act in enumerate(tr):
            rows.append({"case:concept:name": f"{ci:05d}", "concept:name": str(act),
                         "time:timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=k)})
    log = pd.DataFrame(rows)
    res = {}
    for tmpl, pname in PM4PY_NAME.items():
        for a in alph:
            for b in (alph if tmpl not in ("init", "existence", "absence") else [None]):
                if b == a:
                    continue
                c = Constraint(tmpl, a, b)
                key = a if b is None else (a, b)
                model = {pname: {key: {"support": 1.0, "confidence": 1.0}}}
                out = pm4py.conformance_declare(log, model)
                agree = sum(int(bool(o["is_fit"]) == declare_holds(c, [str(x) for x in tr]))
                            for o, tr in zip(out, traces))
                res[c.cid] = (agree, len(traces))
    return res


def test_declare_matches_pm4py():
    res = pm4py_agreement(150, seed=1)
    bad = {k: v for k, v in res.items() if v[0] != v[1]}
    assert not bad, bad


def _same(g, spec, h):
    raw = run_checks(g, spec)
    san = align(h, run_checks(h, spec))
    return compare(raw, san)


@pytest.mark.parametrize("w", ["W1", "W2", "W3"])
@pytest.mark.parametrize("ops", [("VG",), ("TC",), ("VG", "TC"), ("VG", "TC", "BS")])
def test_exact_preservation(w, ops):
    gen, specf = GENERATORS[w]
    spec = specf()
    for seed in range(2):
        g = gen(200, seed=seed, p_dev=0.3)
        h = O.sanitise(g, spec, KEY, ops, delta=60.0)
        m = _same(g, spec, h)
        assert m["same"].all(), m[~m["same"]].head()


@pytest.mark.parametrize("w", ["W2", "W3"])
def test_dpn_only_affects_thresholds(w):
    gen, specf = GENERATORS[w]
    spec = specf()
    g = gen(200, seed=3, p_dev=0.3)
    h = O.sanitise(g, spec, KEY, ("VG", "TC", "BS", "DPN"), unit="record", eps=1.0)
    m = _same(g, spec, h)
    assert m[m["family"] != "thr"]["same"].all()


def test_timed_bound():
    gen, specf = GENERATORS["W3"]
    spec = specf()
    g = gen(300, seed=4, p_dev=0.3)
    for delta in (60.0, 600.0, 1800.0):
        h = O.sanitise(g, spec, KEY, ("TC",), delta=delta)
        m = _same(g, spec, h)
        t = m[(m["family"] == "time") & (m["cid"].str.startswith("time(span"))]
        act = g.act
        span = (act.groupby("run")["end"].max() - act.groupby("run")["start"].min())
        lim = 6 * 3600.0
        far = set(span[(span - lim).abs() >= delta].index)
        assert t[t["key"].isin(far)]["same"].all()


def test_block_shuffle_keeps_traces():
    gen, _specf = GENERATORS["W3"]
    g = gen(100, seed=5)
    ring = O.KeyRing(KEY)
    h1 = O.tc(g, 60.0)
    h2 = O.bs(h1, ring)
    t1, _ = _traces(h1)
    t2, _ = _traces(h2)
    assert t1 == t2
    # the slots actually moved for some block
    assert (h1.act["rank"].to_numpy() != h2.act["rank"].to_numpy()).any()


def test_parallel_equals_sequential():
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(120, seed=6)
    full = O.sanitise(g, spec, KEY)
    parts = [O.sanitise(p, spec, KEY) for p in split_by_runs(g, 3)]
    merged = concat_graphs(parts)
    for tab in ("runs", "act", "ent"):
        a = getattr(full, tab)
        b = getattr(merged, tab)
        key = "run" if tab == "runs" else "id"
        a = a.sort_values(key).reset_index(drop=True).astype(str)
        b = b.sort_values(key).reset_index(drop=True)[a.columns].astype(str)
        pd.testing.assert_frame_equal(a, b)


def test_run_scoped_config_loses_replica_checks():
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(200, seed=7)
    scopes = O.default_scopes(spec)
    scopes[("runs", "config")] = "run"
    h = O.sanitise(g, spec, KEY, ("VG",), scopes=scopes)
    m = _same(g, spec, h)
    assert not m[m["family"].str.startswith("rep_")]["same"].any()


def test_norank_coarsening_changes_conformance():
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(300, seed=8, p_dev=0.3)
    h = O.sanitise(g, spec, KEY, ("VG", "TC"), delta=600.0, keep_rank=False)
    m = _same(g, spec, h)
    assert not m[m["family"] == "conf"]["same"].all()


def test_collision_detected():
    ring = O.KeyRing(KEY)
    vals = [str(i) for i in range(5000)]
    with pytest.raises(O.CollisionError):
        O.pseudonymise(vals, ["release"] * len(vals), "x", ring, nbytes=1)


def test_memoised_noise_consistent():
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(200, seed=9)
    ring = O.KeyRing(KEY)
    h = O.dpn(g, spec, ring, 1.0, unit="record")
    src = g.ent["type"] == "data_partition"
    df = pd.DataFrame({"d": g.ent.loc[src, "digest"], "v": h.ent.loc[src, "n_records"]})
    assert (df.groupby("d")["v"].nunique() == 1).all()


def test_flip_probability_formula():
    ring = O.KeyRing(KEY)
    eps_attr, sens, thr = 0.5, 1.0, 500
    alpha = math.exp(-eps_attr / sens)
    for x in (497, 500, 503):
        n = 20000
        flips = 0
        for i in range(n):
            u1, u2 = ring.uniform(f"t|{x}|{i}")
            z = O.two_sided_geometric(u1, u2, alpha)
            flips += int((x + z >= thr) != (x >= thr))
        p = O.flip_probability(x, thr, eps_attr, sens)
        assert abs(flips / n - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-3


@pytest.mark.parametrize("ops", [None, ("VG", "TC", "BS", "DPN")])
def test_prov_json_roundtrip(tmp_path, ops):
    from provpub.data.prov_io import from_prov_json, to_prov_json
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(60, seed=10, p_dev=0.3)
    h = g if ops is None else O.sanitise(g, spec, KEY, ops)
    p = tmp_path / "doc.json"
    to_prov_json(h, str(p))
    back = from_prov_json(str(p))
    a = run_checks(h, spec)
    b = run_checks(back, spec)
    m = compare(a, b)
    assert m["same"].all()


@pytest.mark.parametrize("ops", [None, ("VG", "TC", "BS", "DPN")])
def test_prov_json_parses_with_reference_library(tmp_path, ops):
    """Optional: the document parses with the W3C 'prov' reference library (skipped if absent)."""
    prov_model = pytest.importorskip("prov.model")
    from provpub.data.prov_io import to_prov_json
    gen, specf = GENERATORS["W2"]
    g = gen(30, seed=10)
    h = g if ops is None else O.sanitise(g, specf(), KEY, ops)
    p = tmp_path / "doc.json"
    to_prov_json(h, str(p))
    doc = prov_model.ProvDocument.deserialize(str(p), format="json")
    assert len(list(doc.get_records())) > 0


def test_otel_roundtrip():
    from provpub.data.prov_io import from_otel_spans, to_otel_spans
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(60, seed=11, p_dev=0.3)
    back = from_otel_spans(to_otel_spans(g))
    m = compare(run_checks(g, spec), run_checks(back, spec))
    assert m["same"].all()


def test_input_set_scoped_noise():
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(300, seed=12)
    ring = O.KeyRing(KEY)
    h = O.dpn(g, spec, ring, 1.0, unit="bucket", memo="group")
    src = (g.ent["type"] == "data_partition").to_numpy()
    sets = g.ent[src].groupby("run")["digest"].apply(lambda s: ";".join(sorted(s)))
    df = pd.DataFrame({"d": g.ent.loc[src, "digest"].to_numpy(), "s": g.ent.loc[src, "run"].map(sets).to_numpy(),
                       "v": h.ent.loc[src, "n_records"].to_numpy()})
    # same partition and same input set: identical noise
    assert (df.groupby(["d", "s"])["v"].nunique() == 1).all()
    # same partition across different input sets: noise differs for most partitions
    multi = df.groupby("d")["s"].nunique()
    vals = df.groupby("d")["v"].nunique()
    assert (vals[multi > 1] > 1).mean() > 0.9
    m = _same(g, spec, O.sanitise(g, spec, KEY, memo="group"))
    assert m[m["family"] != "thr"]["same"].all()


def test_release_rows_carry_no_collection_order():
    """Edge rows and node rows are emitted in keyed order: the order of 'used'
    edges must not reveal the pre-shuffle position of load activities."""
    from scipy.stats import spearmanr
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(200, seed=13)
    h = O.sanitise(g, spec, KEY, ("VG", "TC", "BS"))
    # map released activity ids back to raw collector seq (publisher side)
    import hmac
    nid = {int.from_bytes(hmac.digest(KEY, f"nid|{i}".encode(), "sha256")[:7], "big") >> 4: q
           for i, q in zip(g.act["id"], g.act["seq"])}
    loads = set(h.act.loc[h.act["type"] == "load_partition", "id"])
    used = h.edge[(h.edge["kind"] == "used") & h.edge["src"].isin(loads)]
    pos = np.arange(len(used))
    seqs = np.array([nid[int(a)] for a in used["src"]])
    rho, _ = spearmanr(pos, seqs)
    assert abs(rho) < 0.1
    # edge table is sorted by keyed labels
    assert h.edge.equals(h.edge.sort_values(["kind", "src", "dst"], kind="mergesort").reset_index(drop=True))


def test_ties_in_collector_order_rejected():
    gen, specf = GENERATORS["W1"]
    g = gen(20, seed=14)
    g.act["seq"] = 0
    with pytest.raises(ValueError):
        run_checks(g, specf())


def test_otel_import_assigns_unique_sequence():
    from provpub.data.prov_io import from_otel_spans, to_otel_spans
    gen, specf = GENERATORS["W3"]
    g = gen(20, seed=15)
    spans = to_otel_spans(g)
    for sp in spans:
        sp.pop("seq", None)
    back = from_otel_spans(spans)
    assert not back.act.duplicated(["run", "seq"]).any()


def test_specification_must_be_well_typed():
    from dataclasses import replace

    from provpub.data.spec import VocabRule
    spec = GENERATORS["W2"][1]()
    with pytest.raises(ValueError):
        replace(spec, vocab_rules=[VocabRule("load_partition", "site", frozenset({"site-A"}))])
    with pytest.raises(ValueError):
        replace(spec, vocab_rules=[VocabRule("agent_reason", "model", frozenset({"not-in-vocab"}))])
    with pytest.raises(ValueError):
        Constraint("response", "a", "a")


def test_compare_counts_spurious_instances():
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(100, seed=16)
    raw = run_checks(g, spec)
    extra = pd.DataFrame([("rep_outputs", "agree", (10**9, 10**9 + 1), 0)], columns=raw.columns)
    m = compare(raw, pd.concat([raw, extra], ignore_index=True))
    assert (~m["same"]).sum() == 1


def test_noise_keyed_by_partition_identifier_not_content():
    """Two partitions with identical content but different identifiers get
    independent noise (content changes cannot merge memo keys)."""
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(50, seed=17)
    src = g.ent["type"] == "data_partition"
    g.ent.loc[src, "digest"] = "same-content"
    ring = O.KeyRing(KEY)
    h = O.dpn(g, spec, ring, 1.0, unit="bucket")
    df = pd.DataFrame({"p": g.ent.loc[src, "part_id"], "v": h.ent.loc[src, "n_records"]})
    assert df.groupby("p")["v"].nunique().max() == 1
    assert df.drop_duplicates("p")["v"].nunique() > 1


def test_suppressed_inputs_make_replica_checks_undetermined():
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(200, seed=18)
    h = B.modhide(g, {"load_partition", "validate_schema"})
    m = _same(g, spec, h)
    rep_in = m[m["family"] == "rep_inputs"]
    assert len(rep_in) > 0
    assert (rep_in["outcome_san"] == -1).all()


def test_salted_attacker_matches_operator():
    """The published-salt variant is reversible by an attacker who knows the
    construction: user names and prompts are recovered by enumeration."""
    from provpub.attacks import attacks as A
    from provpub.data.synthetic import W1_DATASETS, W1_QUESTIONS, w1_generate, w1_prompt, w1_spec
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(60, seed=3)
    salt = b"public-salt"
    h = O.sanitise(g, spec, KEY, ops=("VG",), salted=salt)
    users = set(g.runs["user"])
    recovered = {u for u in users if A._salted_ps(salt, "user", u) in set(h.runs["user"])}
    assert recovered == users
    g1 = w1_generate(40, seed=1)
    h1 = O.sanitise(g1, w1_spec(), KEY, ops=("VG",), salted=salt)
    cands = [w1_prompt(qi, ds) + sfx for ds in W1_DATASETS for qi in range(len(W1_QUESTIONS))
             for sfx in [f" [step {k}]" for k in range(40)] + [" [final]"]]
    rate, _ = A.dictionary_attack(h1, cands, salt)
    assert rate == 1.0
    rate_keyed, _ = A.dictionary_attack(O.sanitise(g1, w1_spec(), KEY, ops=("VG",)), cands, None)
    assert rate_keyed == 0.0


def test_block_shuffle_removes_order_side_channel():
    from provpub.attacks import attacks as A
    gen, specf = GENERATORS["W2"]
    spec = specf()
    g = gen(400, seed=5)
    a_tc, chance = A.position_attack(O.sanitise(g, spec, KEY, ops=("VG", "TC")), g, spec, KEY)
    a_bs, _ = A.position_attack(O.sanitise(g, spec, KEY, ops=("VG", "TC", "BS")), g, spec, KEY)
    assert a_tc > chance + 0.15
    assert abs(a_bs - chance) < 0.05


def test_lifted_spec_drops_collapsed_types():
    from provpub.data.spec import lift_spec
    gen, specf = GENERATORS["W1"]
    spec = specf()
    grp = {"agent_step", "load_csv"}
    ls = lift_spec(spec, grp)
    assert all(c.a not in grp and (c.b is None or c.b not in grp) for c in ls.constraints)
    assert all(r.act_type not in grp for r in ls.vocab_rules)
    assert len(ls.constraints) < len(spec.constraints)
