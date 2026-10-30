"""Schemas, synthetic workloads W1-W3 and their declared specifications."""
import pytest

from provpub.data.schemas import EDGE_KINDS
from provpub.data.spec import Constraint, Spec, VocabRule
from provpub.data.spec_io import spec_from_dict, spec_to_dict
from provpub.data.synthetic import GENERATORS, make_sites, size_class_of, w2_generate
from provpub.utils.validation import consistent_provenance, validate_graph


@pytest.mark.parametrize("wl", ["W1", "W2", "W3"])
def test_generators_produce_valid_graphs(wl):
    gen, specf = GENERATORS[wl]
    g = gen(30, seed=1)
    assert validate_graph(g) == []
    assert set(g.edge["kind"]) <= set(EDGE_KINDS)
    assert len(g.runs) == 30
    assert g.n_records == len(g.act) + len(g.ent) + len(g.edge)
    spec = specf()
    assert set(g.act["type"]) - set(spec.alphabet) <= {"shell_exec", "export_raw", "delete_outputs"} | set(spec.alphabet)
    assert consistent_provenance(g, spec.source_type)


@pytest.mark.parametrize("wl", ["W1", "W2", "W3"])
def test_generators_are_deterministic(wl):
    gen, _ = GENERATORS[wl]
    a, b = gen(25, seed=7), gen(25, seed=7)
    assert a.act.equals(b.act) and a.ent.equals(b.ent) and a.edge.equals(b.edge)
    c = gen(25, seed=8)
    assert not a.act.equals(c.act)


def test_deviations_recorded():
    g = w2_generate(200, seed=3)
    devs = g.meta["deviations"]
    assert devs and all(int(r) in set(g.runs["run"]) for r, _ in devs)


def test_size_classes_and_sensitivity_variant():
    import numpy as np
    sites = make_sites(12, np.random.default_rng(0))
    assert [s.size_class for s in sites] == [size_class_of(s.base_records) for s in sites]
    cont = make_sites(12, np.random.default_rng(0), size_mode="continuous")
    assert all(450 <= s.base_records <= 150000 for s in cont)
    g = w2_generate(30, seed=0, prompt_sites=False, load_order="random")
    prompts = g.ent[g.ent["type"] == "prompt"]["text"].astype(str)
    assert not prompts.str.contains("site-").any()


@pytest.mark.parametrize("wl", ["W1", "W2", "W3"])
def test_spec_yaml_roundtrip(wl):
    spec = GENERATORS[wl][1]()
    back = spec_from_dict(spec_to_dict(spec))
    assert spec_to_dict(back) == spec_to_dict(spec)
    g = GENERATORS[wl][0](20, seed=0)
    from provpub.checks.checks import run_checks
    assert run_checks(g, spec)["outcome"].tolist() == run_checks(g, back)["outcome"].tolist()


def test_ill_typed_spec_rejected():
    with pytest.raises(ValueError):
        Spec(name="x", alphabet=frozenset({"a"}), constraints=[], vocab={"tool": frozenset({"t"})},
             vocab_rules=[VocabRule("a", "user", frozenset({"u"}))], generalisers={}, nd_types=frozenset())
    with pytest.raises(ValueError):
        Constraint("response", "a", "a")
    with pytest.raises(ValueError):
        Constraint("eventually", "a")


def test_validate_graph_reports_problems():
    g = w2_generate(10, seed=0)
    h = g.copy()
    h.edge.loc[0, "dst"] = -12345
    assert any("unknown nodes" in p for p in validate_graph(h))
