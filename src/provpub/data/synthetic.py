"""Workload generators emitting PROV-AGENT-model provenance.

All three workloads use scripted stochastic planners in place of an LLM:
the decisions an agent would take (which tools, which parameters, which
sites) are sampled from documented distributions, and every LLM step is
recorded as a model invocation with a prompt entity, a response entity, a
model identifier, a temperature and a seed. No LLM inference is performed.

W1  exploratory analysis agent (ReAct-style tool loop on public datasets)
W2  multi-institution analytics (per-site partitions, harmonise, train, evaluate)
W3  HPC simulation campaign (agent-chosen parameter sets, many parallel solver runs)

Deviations are injected per run with probability p_dev and recorded in
Graph.meta['deviations'] as (run, kind).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import Graph
from .spec import Constraint as C
from .spec import Spec, ThresholdRule, TimedRule, VocabRule, path_generaliser

APPROVED_MODELS = ("approved-llm-a:1.2", "approved-llm-b:0.9")
ROGUE_MODEL = "unvetted-llm-x:0.1"


def sha(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


ACT_COLS = ["id", "run", "type", "start", "end", "seq", "agent", "tool", "model",
            "temperature", "seed", "site", "p_path", "p_value"]
ENT_COLS = ["id", "run", "type", "text", "digest", "part_id", "site", "n_records", "n_bytes",
            "out_rows", "out_bytes"]


class Builder:
    def __init__(self):
        self.act, self.ent, self.edge, self.runs = [], [], [], []
        self.nid = 0
        self.dev = []
        self.truth = {}   # private ground truth for attacks: run -> dict

    def new_id(self):
        self.nid += 1
        return self.nid

    def a(self, run, typ, start, end, seq, **kw):
        i = self.new_id()
        row = dict.fromkeys(ACT_COLS)
        row.update(id=i, run=run, type=typ, start=float(start), end=float(end), seq=seq, agent="data-agent")
        row.update(kw)
        self.act.append(row)
        return i

    def e(self, run, typ, **kw):
        i = self.new_id()
        row = dict.fromkeys(ENT_COLS)
        row.update(id=i, run=run, type=typ)
        row.update(kw)
        self.ent.append(row)
        return i

    def used(self, run, act, ent):
        self.edge.append(("used", act, ent, run))

    def gen(self, run, ent, act):
        self.edge.append(("wasGeneratedBy", ent, act, run))

    def informed(self, run, later, earlier):
        self.edge.append(("wasInformedBy", later, earlier, run))

    def llm(self, run, typ, t, seq, prompt, model, seed, rng, latency=(2.0, 12.0), uses=()):
        """An LLM step: Prompt entity -> model invocation -> Response entity."""
        dur = float(rng.uniform(*latency))
        p = self.e(run, "prompt", text=prompt, digest=sha("prompt", prompt))
        a = self.a(run, typ, t, t + dur, seq, model=model, temperature="0", seed=seed, tool=None)
        resp_text = f"[{typ} response to: {prompt[:40]}] seed={seed}"
        r = self.e(run, "response", text=resp_text, digest=sha("resp", prompt, model, seed))
        self.used(run, a, p)
        for u in uses:
            self.used(run, a, u)
        self.gen(run, r, a)
        return a, r, t + dur

    def graph(self, meta=None) -> Graph:
        act = pd.DataFrame(self.act, columns=ACT_COLS)
        ent = pd.DataFrame(self.ent, columns=ENT_COLS)
        for c in ("n_records", "n_bytes", "out_rows", "out_bytes"):
            ent[c] = ent[c].astype(object)
        edge = pd.DataFrame(self.edge, columns=["kind", "src", "dst", "run"])
        runs = pd.DataFrame(self.runs)
        m = {"deviations": self.dev, "truth": self.truth}
        if meta:
            m.update(meta)
        return Graph(runs, act, ent, edge, m)


# ====================================================================== W2

@dataclass
class Site:
    name: str
    size_class: int          # 0 small, 1 medium, 2 large
    base_records: int
    row_width: int
    throughput: float        # records per second when loading
    latency: float           # fixed load latency in seconds


SIZE_BOUNDS = (2000, 20000)  # class 0 < 2000 <= class 1 < 20000 <= class 2


def make_sites(n_sites: int, rng: np.random.Generator, size_mode: str = "classes") -> list[Site]:
    """size_mode='classes': sizes drawn within decade-separated class ranges
    (default workload); 'continuous': every site drawn log-uniformly from the
    whole range, so neighbouring sites can have similar sizes (sensitivity run)."""
    sites = []
    for i in range(n_sites):
        c = i % 3
        lo, hi = [(450, 1900), (2500, 18000), (25000, 150000)][c] if size_mode == "classes" else (450, 150000)
        rec = int(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        c = c if size_mode == "classes" else size_class_of(rec)
        sites.append(Site(
            name=f"site-{chr(65 + i)}", size_class=c, base_records=rec,
            row_width=int(rng.uniform(80, 600)),
            throughput=float(math.exp(rng.uniform(math.log(2000), math.log(20000)))),
            latency=float(rng.uniform(0.5, 5.0)),
        ))
    return sites


def size_class_of(n: float) -> int:
    return 0 if n < SIZE_BOUNDS[0] else (1 if n < SIZE_BOUNDS[1] else 2)


@dataclass
class Analyst:
    name: str
    hour_mu: float           # typical start hour
    hour_sd: float
    approval_median: float   # seconds of human approval wait before training


def make_analysts(n: int, rng) -> list[Analyst]:
    out = []
    for i in range(n):
        out.append(Analyst(f"analyst-{i:02d}", hour_mu=float(rng.uniform(7, 20)), hour_sd=float(rng.uniform(0.5, 1.5)),
                           approval_median=float(math.exp(rng.uniform(math.log(20), math.log(600))))))
    return out


W2_DEV_KINDS = ("order", "model", "seed", "nonrepro", "export", "path", "dangling", "skipval")


def w2_spec() -> Spec:
    alphabet = frozenset({"agent_plan", "load_partition", "validate_schema", "harmonise", "agent_reason",
                          "train_model", "evaluate", "agent_report", "export_raw"})
    cons = [
        C("init", "agent_plan"), C("existence", "agent_report"), C("absence2", "agent_report"),
        C("precedence", "load_partition", "validate_schema"), C("response", "load_partition", "validate_schema"),
        C("precedence", "validate_schema", "harmonise"), C("chain_response", "agent_reason", "train_model"),
        C("precedence", "train_model", "evaluate"), C("response", "train_model", "evaluate"),
        C("precedence", "evaluate", "agent_report"), C("absence", "export_raw"),
        C("not_succession", "agent_report", "train_model"),
    ]
    tools = frozenset({"sql_load", "schema_check", "harmonise", "train", "evaluate"})
    vocab = {"tool": tools, "model": frozenset(APPROVED_MODELS), "temperature": frozenset({"0"})}
    rules = [VocabRule(t, "tool", tools) for t in ("load_partition", "validate_schema", "harmonise", "train_model", "evaluate")]
    rules += [VocabRule(t, "model", frozenset(APPROVED_MODELS)) for t in ("agent_plan", "agent_reason", "agent_report")]
    rules += [VocabRule(t, "temperature", frozenset({"0"})) for t in ("agent_plan", "agent_reason", "agent_report")]
    rules += [VocabRule("load_partition", "p_path", frozenset({"allowed"}))]
    return Spec(
        name="W2-multisite-analytics", alphabet=alphabet, constraints=cons, vocab=vocab, vocab_rules=rules,
        generalisers={"p_path": path_generaliser(("/data/consortium/",))},
        nd_types=frozenset({"agent_plan", "agent_reason", "agent_report"}),
        thresholds=[ThresholdRule("data_partition", "n_records", 500)],
        timed=[TimedRule("span", 3600.0), TimedRule("max_dur", 120.0, "load_partition")],
    )


def w2_generate(n_runs: int, seed: int = 0, n_sites: int = 12, n_analysts: int = 6, n_configs: int | None = None,
                p_dev: float = 0.1, replica_p: float = 0.35, sites=None, analysts=None,
                horizon_days: int = 180, prompt_sites: bool = True, load_order: str = "config") -> Graph:
    """prompt_sites=False: the planning prompt states only the number of sites;
    load_order='random': partitions are loaded in a random order per run
    (both used for the sensitivity analysis)."""
    rng = np.random.default_rng(seed)
    sites = sites or make_sites(n_sites, rng)
    analysts = analysts or make_analysts(n_analysts, rng)
    if n_configs is None:
        n_configs = max(4, n_runs // 4)
    targets = ["readmission", "length_of_stay", "mortality", "cost_band"]
    models_ml = ["logreg", "gbm", "rf"]
    configs = []
    for c in range(n_configs):
        k = int(rng.integers(3, min(9, len(sites)) + 1))
        sub = sorted(rng.choice(len(sites), size=k, replace=False).tolist())
        configs.append(dict(cid=f"study-{c:05d}", sites=sub, ml=str(rng.choice(models_ml)),
                            target=str(rng.choice(targets)), owner=int(rng.integers(len(analysts))),
                            hp=f"max_depth={int(rng.integers(3, 9))};lr={rng.choice([0.03, 0.1, 0.3])}"))
    b = Builder()
    run = 0
    t_day = 86400.0
    while run < n_runs:
        cfg = configs[int(rng.integers(n_configs))]
        an = analysts[cfg["owner"]] if rng.random() < 0.8 else analysts[int(rng.integers(len(analysts)))]
        day = int(rng.integers(horizon_days))
        n_rep = 1 + (int(rng.integers(1, 4)) if rng.random() < replica_p else 0)
        n_rep = min(n_rep, n_runs - run)
        group_dev = None
        if rng.random() < p_dev:
            group_dev = str(rng.choice(W2_DEV_KINDS))
            if group_dev == "nonrepro" and n_rep < 2:
                group_dev = None if n_rep < 2 else group_dev
        hour = float(np.clip(rng.normal(an.hour_mu, an.hour_sd), 0, 23.9))
        t0 = day * t_day + hour * 3600.0
        batch = run
        for rep in range(n_rep):
            dev = group_dev if (group_dev and (group_dev != "nonrepro" or rep == n_rep - 1)) else None
            if group_dev and group_dev != "nonrepro" and rep > 0:
                dev = None  # one deviating run per group for non-replica deviations
            _w2_run(b, run, cfg, sites, an, t0 + rep * float(rng.uniform(600, 5400)), day, dev, rng, batch,
                    prompt_sites, load_order)
            run += 1
    return b.graph({"workload": "W2", "sites": sites, "analysts": analysts, "configs": configs})


def _w2_run(b: Builder, run, cfg, sites, an, t, day, dev, rng, batch, prompt_sites=True, load_order="config"):
    seq = 0
    def nxt():
        nonlocal seq
        seq += 1
        return seq
    b.runs.append({"run": run, "config": f"{cfg['cid']}#b{batch}", "user": an.name})
    version = day // 30
    site_list = ", ".join(sites[i].name for i in cfg["sites"])
    model = APPROVED_MODELS[run % 2]
    seed = f"{int(rng.integers(1, 2**31))}"
    if not prompt_sites:
        site_list = f"{len(cfg['sites'])} consortium sites"
    prompt = (f"Study {cfg['cid']}: using partitions from {site_list}, build a {cfg['ml']} model "
              f"for {cfg['target']}; validate schemas before pooling.")
    plan, plan_resp, t = b.llm(run, "agent_plan", t, nxt(), prompt, model, seed, rng)
    if dev == "seed":
        b.act[-1]["seed"] = None
    truth = {"sites": [sites[i].name for i in cfg["sites"]], "user": an.name, "start": None, "dev": dev}
    tables = []
    t += float(rng.uniform(0.5, 3.0))
    load_ids = []
    t_load0 = t
    ends = []
    order = list(cfg["sites"]) if load_order == "config" else [int(x) for x in rng.permutation(cfg["sites"])]
    for j, si in enumerate(order):           # default: collector order follows configuration order (by name)
        s = sites[si]
        rec = int(s.base_records * (1 + 0.04 * (version % 5)))
        nbytes = int(rec * s.row_width * float(rng.uniform(0.99, 1.01)))
        dig = sha("partition", s.name, version)
        nbytes = int(rec * s.row_width * (1 + ((int(dig[:4], 16) % 200) - 100) / 10000.0))
        path = f"/data/consortium/{s.name}/cohort_v{version}.parquet"
        if dev == "path" and j == 0:
            path = f"/home/{an.name}/scratch/{s.name}_extract.csv"
        src = b.e(run, "data_partition", digest=dig, part_id=f"{s.name}/cohort/v{version}", site=s.name,
                  n_records=rec, n_bytes=nbytes)
        st = t_load0 + float(rng.uniform(0.0, 0.4)) * j
        dur = s.latency + rec / s.throughput * float(rng.uniform(0.9, 1.1))
        la = b.a(run, "load_partition", st, st + dur, nxt(), tool="sql_load", site=s.name, p_path=path)
        b.used(run, la, src)
        b.informed(run, la, plan)
        tab = b.e(run, "table", digest=sha("table", dig), site=s.name, out_rows=rec, out_bytes=nbytes)
        b.gen(run, tab, la)
        tables.append((s, tab, rec, dig))
        load_ids.append(la)
        ends.append(st + dur)
    t = max(ends) + float(rng.uniform(0.2, 1.0))
    validated = []
    if dev != "skipval":
        vends = []
        for j, (s, tab, rec, dig) in enumerate(tables):
            st = t + float(rng.uniform(0.0, 0.2)) * j
            dur = 0.2 + rec / 50000.0 * float(rng.uniform(0.9, 1.1))
            va = b.a(run, "validate_schema", st, st + dur, nxt(), tool="schema_check", site=s.name)
            b.used(run, va, tab)
            vt = b.e(run, "table", digest=sha("valid", dig), site=s.name, out_rows=rec, out_bytes=None)
            b.gen(run, vt, va)
            validated.append(vt)
            vends.append(st + dur)
        t = max(vends) + float(rng.uniform(0.2, 1.0))
    else:
        validated = [tab for (_, tab, _, _) in tables]
    total = sum(r for (_, _, r, _) in tables)
    ha = b.a(run, "harmonise", t, t + 1.0 + total / 100000.0, nxt(), tool="harmonise")
    for v in validated:
        b.used(run, ha, v)
    pooled = b.e(run, "table", digest=sha("pooled", *[d for (_, _, _, d) in tables]), out_rows=total)
    b.gen(run, pooled, ha)
    t = t + 1.0 + total / 100000.0
    prompt2 = (f"Pooled {total} rows from {len(tables)} sites for {cfg['target']}. "
               f"Propose hyperparameters for {cfg['ml']}.")
    rmodel = ROGUE_MODEL if dev == "model" else model
    reason, reason_resp, t = b.llm(run, "agent_reason", t, nxt(), prompt2, rmodel, seed, rng, uses=(pooled,))
    t += float(np.exp(rng.normal(math.log(an.approval_median), 0.5)))   # human approval wait
    ext = None
    if dev == "dangling":
        ext = b.e(run, "table", digest=sha("external", run), out_rows=int(rng.integers(100, 1000)))
    def train(t):
        dur = 5.0 + total / 20000.0 * float(rng.uniform(0.9, 1.2))
        tr = b.a(run, "train_model", t, t + dur, nxt(), tool="train", p_value=cfg["hp"])
        b.used(run, tr, pooled)
        if ext is not None:
            b.used(run, tr, ext)
        b.informed(run, tr, reason)
        mdl = b.e(run, "model_artifact", digest=sha("model", cfg["cid"], *[d for (_, _, _, d) in tables]))
        b.gen(run, mdl, tr)
        return tr, mdl, t + dur
    def evaluate(t, mdl):
        dur = 2.0 + total / 80000.0
        ev = b.a(run, "evaluate", t, t + dur, nxt(), tool="evaluate")
        if mdl is not None:
            b.used(run, ev, mdl)
        b.used(run, ev, pooled)
        met = b.e(run, "metrics", digest=sha("metrics", cfg["cid"], *[d for (_, _, _, d) in tables]))
        b.gen(run, met, ev)
        return ev, met, t + dur
    if dev == "order":
        ev, met, t = evaluate(t, None)
        tr, mdl, t = train(t)
    else:
        tr, mdl, t = train(t)
        ev, met, t = evaluate(t, mdl)
    if dev == "export":
        ex = b.a(run, "export_raw", t, t + 3.0, nxt(), tool="sql_export", p_path=f"/tmp/{cfg['cid']}.csv")
        b.used(run, ex, pooled)
        t += 3.0
    prompt3 = f"Summarise evaluation of {cfg['ml']} for {cfg['target']} in study {cfg['cid']}."
    rep, rep_resp, t = b.llm(run, "agent_report", t, nxt(), prompt3, model, seed, rng, uses=(met,))
    out_digest = sha("result", cfg["cid"], *[d for (_, _, _, d) in tables])
    if dev == "nonrepro":
        out_digest = sha("result-drift", run, rng.random())
    res = b.e(run, "result", digest=out_digest, text=f"Report for {cfg['cid']}")
    b.gen(run, res, rep)
    truth["start"] = b.act[-1]["start"]
    b.truth[run] = truth
    if dev:
        b.dev.append((run, dev))


# ====================================================================== W1

W1_DATASETS = ["iris", "wine", "breast_cancer", "diabetes", "digits", "penguins", "titanic", "adult", "mpg", "tips"]
W1_QUESTIONS = [f"q{i:02d}" for i in range(40)]
W1_QTEXT = ["Which features best predict the target?", "Is there a difference between groups?",
            "Summarise the distribution of each column.", "Fit a baseline model and report accuracy.",
            "Find outliers in the numeric columns.", "Plot the correlation matrix.",
            "Which subgroup has the highest mean outcome?", "Compare two models by cross-validation."]
W1_TOOLS = ("load_csv", "describe", "filter_rows", "group_by", "plot", "fit_model")
W1_DEV_KINDS = ("path", "seed", "shell", "order", "nonrepro")


def w1_prompt(question_idx: int, dataset: str) -> str:
    q = W1_QTEXT[question_idx % len(W1_QTEXT)]
    return f"You are a data analysis agent. Dataset: {dataset}. Task {question_idx:02d}: {q}"


def w1_spec() -> Spec:
    alphabet = frozenset({"agent_step", *W1_TOOLS, "final_answer", "shell_exec"})
    cons = [C("init", "agent_step"), C("existence", "load_csv"), C("precedence", "load_csv", "describe"),
            C("precedence", "load_csv", "fit_model"), C("existence", "final_answer"), C("absence2", "final_answer"),
            C("absence", "shell_exec"), C("not_succession", "final_answer", "agent_step"),
            C("response", "agent_step", "final_answer")]
    tools = frozenset(set(W1_TOOLS) | {"answer"})
    vocab = {"tool": tools, "model": frozenset(APPROVED_MODELS), "temperature": frozenset({"0"})}
    rules = [VocabRule(t, "tool", tools) for t in W1_TOOLS] + [VocabRule("agent_step", "model", frozenset(APPROVED_MODELS)),
             VocabRule("agent_step", "temperature", frozenset({"0"})), VocabRule("load_csv", "p_path", frozenset({"allowed"}))]
    return Spec(name="W1-exploratory-agent", alphabet=alphabet, constraints=cons, vocab=vocab, vocab_rules=rules,
                generalisers={"p_path": path_generaliser(("/data/public/",))}, nd_types=frozenset({"agent_step"}),
                source_type="data_partition", thresholds=[], timed=[TimedRule("span", 1800.0)])


def w1_generate(n_runs: int, seed: int = 0, n_analysts: int = 6, p_dev: float = 0.1, replica_p: float = 0.3) -> Graph:
    rng = np.random.default_rng(seed)
    analysts = make_analysts(n_analysts, rng)
    b = Builder()
    run = 0
    while run < n_runs:
        ds = str(rng.choice(W1_DATASETS))
        qi = int(rng.integers(len(W1_QUESTIONS)))
        an = analysts[int(rng.integers(n_analysts))]
        n_rep = 1 + (int(rng.integers(1, 3)) if rng.random() < replica_p else 0)
        n_rep = min(n_rep, n_runs - run)
        dev = str(rng.choice(W1_DEV_KINDS)) if rng.random() < p_dev else None
        if dev == "nonrepro" and n_rep < 2:
            dev = None
        day = int(rng.integers(180))
        t0 = day * 86400.0 + float(np.clip(rng.normal(an.hour_mu, an.hour_sd), 0, 23.9)) * 3600
        steps = [str(x) for x in rng.choice(W1_TOOLS[1:], size=int(rng.integers(4, 20)))]
        sd = f"{int(rng.integers(1, 2**31))}"
        for rep in range(n_rep):
            d = dev if (dev and (dev != "nonrepro" or rep == n_rep - 1) and (dev == "nonrepro" or rep == 0)) else None
            _w1_run(b, run, ds, qi, an, t0 + rep * 900.0, steps, sd, d, rng)
            run += 1
    return b.graph({"workload": "W1", "analysts": analysts})


def _w1_run(b, run, ds, qi, an, t, steps, sd, dev, rng):
    seq = 0
    def nxt():
        nonlocal seq
        seq += 1
        return seq
    b.runs.append({"run": run, "config": f"session-{ds}-{qi:02d}-{sd}", "user": an.name})
    src = b.e(run, "data_partition", digest=sha("public", ds), part_id=f"public/{ds}", n_records=None, n_bytes=None)
    model = APPROVED_MODELS[0]
    prompt = w1_prompt(qi, ds)
    last = src
    tools = ["load_csv"] + steps
    if dev == "order":
        tools = [t for t in tools if t != "load_csv"]
        tools.insert(min(2, len(tools)), "load_csv")
        if "fit_model" not in tools[:tools.index("load_csv")]:
            tools.insert(0, "fit_model")
    if dev == "shell":
        tools.insert(len(tools) // 2, "shell_exec")
    for k, tool in enumerate(tools):
        a, r, t = b.llm(run, "agent_step", t, nxt(), prompt + f" [step {k}]", model, sd, rng, latency=(1.0, 6.0))
        if dev == "seed" and k == 1:
            b.act[-1]["seed"] = None
        dur = float(rng.uniform(0.1, 3.0))
        path = f"/data/public/{ds}.csv" if tool == "load_csv" else None
        if dev == "path" and tool == "load_csv":
            path = f"/home/{an.name}/private/{ds}_raw.csv"
        ta = b.a(run, tool, t, t + dur, nxt(), tool=("bash" if tool == "shell_exec" else tool), p_path=path,
                 p_value=(f"cols={int(rng.integers(1, 8))}" if tool in ("filter_rows", "group_by") else None))
        b.used(run, ta, last if k > 0 else src)
        b.informed(run, ta, a)
        out = b.e(run, "table" if tool != "plot" else "figure", digest=sha("w1", ds, qi, k, tool), out_rows=None)
        b.gen(run, out, ta)
        last = out
        t += dur + float(np.exp(rng.normal(math.log(an.approval_median / 10), 0.5)))
    a, r, t = b.llm(run, "agent_step", t, nxt(), prompt + " [final]", model, sd, rng, latency=(1.0, 6.0))
    fa = b.a(run, "final_answer", t, t + 0.5, nxt(), tool="answer")
    b.informed(run, fa, a)
    b.used(run, fa, last)
    dig = sha("w1-answer", ds, qi, sd) if dev != "nonrepro" else sha("drift", run, rng.random())
    res = b.e(run, "result", digest=dig, text=f"answer {ds} {qi}")
    b.gen(run, res, fa)
    b.truth[run] = {"user": an.name, "dataset": ds, "q": qi, "dev": dev}
    if dev:
        b.dev.append((run, dev))


# ====================================================================== W3

W3_DEV_KINDS = ("order", "model", "seed", "nonrepro", "path", "timeout")


def w3_spec() -> Spec:
    alphabet = frozenset({"agent_plan", "run_simulation", "aggregate", "agent_reason", "plot", "agent_report", "delete_outputs"})
    cons = [C("init", "agent_plan"), C("existence", "run_simulation"), C("precedence", "run_simulation", "aggregate"),
            C("response", "run_simulation", "aggregate"), C("precedence", "aggregate", "agent_report"),
            C("existence", "agent_report"), C("absence2", "agent_report"), C("absence", "delete_outputs"),
            C("chain_response", "agent_reason", "plot")]
    tools = frozenset({"sim_solver", "aggregate", "plotter"})
    vocab = {"tool": tools, "model": frozenset(APPROVED_MODELS), "temperature": frozenset({"0"})}
    rules = [VocabRule(t, "tool", tools) for t in ("run_simulation", "aggregate", "plot")]
    rules += [VocabRule(t, "model", frozenset(APPROVED_MODELS)) for t in ("agent_plan", "agent_reason", "agent_report")]
    rules += [VocabRule("run_simulation", "p_path", frozenset({"allowed"}))]
    return Spec(name="W3-hpc-campaign", alphabet=alphabet, constraints=cons, vocab=vocab, vocab_rules=rules,
                generalisers={"p_path": path_generaliser(("/scratch/project/",))},
                nd_types=frozenset({"agent_plan", "agent_reason", "agent_report"}), source_type="data_partition",
                thresholds=[ThresholdRule("data_partition", "n_records", 50_000)],
                timed=[TimedRule("span", 6 * 3600.0), TimedRule("max_dur", 1200.0, "run_simulation")])


def w3_generate(n_runs: int, seed: int = 0, n_inst: int = 5, p_dev: float = 0.1, replica_p: float = 0.3,
                sims=(8, 64)) -> Graph:
    rng = np.random.default_rng(seed)
    analysts = make_analysts(4, rng)
    meshes = [dict(owner=f"inst-{i}", cells=int(math.exp(rng.uniform(math.log(2e4), math.log(2e6)))),
                   width=int(rng.uniform(40, 200))) for i in range(n_inst)]
    b = Builder()
    run = 0
    while run < n_runs:
        mi = int(rng.integers(n_inst))
        an = analysts[int(rng.integers(len(analysts)))]
        m = int(rng.integers(sims[0], sims[1] + 1))
        params = [f"re={float(rng.choice([100, 200, 400, 800]))};dt={float(rng.choice([1e-3, 5e-4]))};k={int(rng.integers(1, 9))}"
                  for _ in range(m)]
        n_rep = 1 + (1 if rng.random() < replica_p else 0)
        n_rep = min(n_rep, n_runs - run)
        dev = str(rng.choice(W3_DEV_KINDS)) if rng.random() < p_dev else None
        if dev == "nonrepro" and n_rep < 2:
            dev = None
        day = int(rng.integers(180))
        t0 = day * 86400.0 + float(np.clip(rng.normal(an.hour_mu, an.hour_sd), 0, 23.9)) * 3600
        sd = f"{int(rng.integers(1, 2**31))}"
        cid = f"campaign-{run:05d}"
        for rep in range(n_rep):
            d = dev if (dev and (dev != "nonrepro" or rep == n_rep - 1) and (dev == "nonrepro" or rep == 0)) else None
            _w3_run(b, run, cid, meshes[mi], params, an, t0 + rep * 7200.0, sd, d, rng)
            run += 1
    return b.graph({"workload": "W3", "meshes": meshes})


def _w3_run(b, run, cid, mesh, params, an, t, sd, dev, rng):
    seq = 0
    def nxt():
        nonlocal seq
        seq += 1
        return seq
    b.runs.append({"run": run, "config": cid, "user": an.name})
    model = APPROVED_MODELS[1]
    src = b.e(run, "data_partition", digest=sha("mesh", mesh["owner"], mesh["cells"]), part_id=f"{mesh['owner']}/mesh",
              site=mesh["owner"],
              n_records=mesh["cells"], n_bytes=mesh["cells"] * mesh["width"])
    plan, _, t = b.llm(run, "agent_plan", t, nxt(), f"Plan a parameter study on mesh from {mesh['owner']} with {len(params)} cases.",
                       model if dev != "model" else ROGUE_MODEL, sd if dev != "seed" else None, rng)
    t += 30.0
    outs, ends = [], []
    for j, p in enumerate(params):
        st = t + float(rng.uniform(0, 2.0))      # batch launch: near-simultaneous starts
        cost = mesh["cells"] / 5000.0 * (1 + 0.2 * (j % 3)) * float(rng.uniform(0.8, 1.2))
        if dev == "timeout" and j == 0:
            cost = 1500.0
        path = f"/scratch/project/{cid}/case{j}" if not (dev == "path" and j == 0) else f"/home/{an.name}/case{j}"
        sa = b.a(run, "run_simulation", st, st + cost, nxt(), tool="sim_solver", p_value=p, p_path=path)
        b.used(run, sa, src)
        b.informed(run, sa, plan)
        o = b.e(run, "sim_output", digest=sha("sim", mesh["owner"], p, sd), out_bytes=int(mesh["cells"] * 8))
        b.gen(run, o, sa)
        outs.append(o)
        ends.append(st + cost)
    t = max(ends) + 5.0
    def aggregate(t):
        ag = b.a(run, "aggregate", t, t + 20.0, nxt(), tool="aggregate")
        for o in outs:
            b.used(run, ag, o)
        s = b.e(run, "summary", digest=sha("agg", *params, sd))
        b.gen(run, s, ag)
        return ag, s, t + 20.0
    if dev == "order":
        # report drafted before aggregation completes
        _, r0, t = b.llm(run, "agent_report", t, nxt(), f"Draft report for {cid}", model, sd, rng)
    ag, summ, t = aggregate(t)
    rs, _, t = b.llm(run, "agent_reason", t, nxt(), f"Interpret aggregated results of {cid}", model, sd, rng, uses=(summ,))
    pl = b.a(run, "plot", t, t + 4.0, nxt(), tool="plotter")
    b.used(run, pl, summ)
    fig = b.e(run, "figure", digest=sha("fig", cid, sd))
    b.gen(run, fig, pl)
    t += 4.0
    rep, _, t = b.llm(run, "agent_report", t, nxt(), f"Write the report for {cid}", model, sd, rng, uses=(fig,))
    dig = sha("w3-result", cid, sd) if dev != "nonrepro" else sha("drift", run, rng.random())
    res = b.e(run, "result", digest=dig)
    b.gen(run, res, rep)
    b.truth[run] = {"user": an.name, "dev": dev}
    if dev:
        b.dev.append((run, dev))


GENERATORS: dict[str, tuple[Callable[..., Graph], Callable[[], Spec]]] = {
    "W1": (w1_generate, w1_spec), "W2": (w2_generate, w2_spec), "W3": (w3_generate, w3_spec)}
