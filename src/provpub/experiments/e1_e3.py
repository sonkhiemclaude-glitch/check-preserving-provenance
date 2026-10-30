"""E1 utility preservation and E3 non-conformance visibility.

For every workload (W1-W3), seed and configuration: sanitise a release of
>= 1e5 PROV records, run all verifier checks on raw and released graphs, and
record per family how many outcomes are unchanged. Also records:
  - timed-check flips and whether any flip violates the coarsening bound
    (|d - D| >= delta must never flip),
  - observed vs. exactly expected threshold-check flips under DPN,
  - detection of each injected deviation in the release.
Output (in out_dir): e1_utility.csv, e1_bounds.csv, e3_deviations.csv, e3_false_alarms.csv
Reference protocol: seeds 0-4, runs per workload RUNS_1E5 (configs/e1_e3.yaml).
"""
from __future__ import annotations

import logging
import math
import os
import time

import pandas as pd

from ..checks.checks import align, compare, run_checks
from ..data.schemas import FAIL, PASS, UNDET
from ..data.spec import lift_spec
from ..data.synthetic import GENERATORS
from .configurations import ABSTRACT_GROUP, CONFIGS, RUNS_1E5, build

log = logging.getLogger(__name__)

SEEDS = [0, 1, 2, 3, 4]
DEV_FAMILY = {
    "W1": {"path": "vocab", "seed": "nd", "shell": "conf", "order": "conf", "nonrepro": "rep_outputs"},
    "W2": {"order": "conf", "model": "vocab", "seed": "nd", "nonrepro": "rep_outputs", "export": "conf",
           "path": "vocab", "dangling": "flow", "skipval": "conf"},
    "W3": {"order": "conf", "model": "vocab", "seed": "nd", "nonrepro": "rep_outputs", "path": "vocab",
           "timeout": "time"},
}


def expected_thr_flips(g, spec, kw, raw_checks):
    """Exact expected number of per-run threshold-check flips under DPN."""
    if not spec.thresholds or "DPN" not in kw.get("ops", ()):
        return None
    eps_attr = kw.get("eps", 1.0) / 2.0
    sens = 1.0 if kw.get("unit") == "record" else 1000.0
    alpha = math.exp(-eps_attr / sens)
    thr = spec.thresholds[0]
    src = g.ent[g.ent["type"] == thr.ent_type]

    def p_pass(x):
        if x >= thr.minimum:
            return 1 - alpha ** (x - thr.minimum + 1) / (1 + alpha)
        return alpha ** (thr.minimum - x) / (1 + alpha)
    q = src.assign(q=src["n_records"].astype(float).map(p_pass)).groupby("run")["q"].prod()
    rc = raw_checks[raw_checks["family"] == "thr"].set_index("key")["outcome"]
    exp = 0.0
    for run, out in rc.items():
        pr = q.get(run, 1.0)
        exp += (1 - pr) if out == 1 else pr
    return exp


def partition_thr_flips(g, h, spec, kw):
    """Partition-level validation of the flip formula: each distinct source
    partition is one independent trial of [x + Z >= theta] vs [x >= theta]."""
    if not spec.thresholds or "DPN" not in kw.get("ops", ()):
        return None, None, None
    eps_attr = kw.get("eps", 1.0) / 2.0
    sens = 1.0 if kw.get("unit") == "record" else 1000.0
    alpha = math.exp(-eps_attr / sens)
    th = spec.thresholds[0].minimum
    src = g.ent["type"] == spec.source_type
    # sanitised values in the same row order: DPN keeps rows; VG renumbers ids,
    # so recover values through the publisher-side node-id map
    import hmac
    key = kw["_key"]
    nid = {int(i): int.from_bytes(hmac.digest(key, f"nid|{i}".encode(), "sha256")[:7], "big") >> 4
           for i in g.ent.loc[src, "id"]}
    pub = dict(zip(h.ent["id"].astype("int64"), h.ent["n_records"]))
    pid = g.ent.loc[src, "part_id"].astype(str)
    runs = g.ent.loc[src, "run"]
    iset = pid.groupby(runs).apply(lambda s: ";".join(sorted(s)))
    # one independent draw per memo key: the partition identifier (partition
    # scope) or the identifier and the run's input set (input-set scope)
    d = pid if kw.get("memo", True) is True else pid + "|" + runs.map(iset)
    df = pd.DataFrame({"d": d, "x": g.ent.loc[src, "n_records"].astype(float),
                       "y": [float(pub[nid[int(i)]]) for i in g.ent.loc[src, "id"]]})
    if not kw.get("memo", True):
        u = df
    else:
        u = df.drop_duplicates("d")
    obs = int(((u["y"] >= th) != (u["x"] >= th)).sum())
    exp = float(sum(alpha ** ((x - th + 1) if x >= th else (th - x)) / (1 + alpha) for x in u["x"]))
    return obs, exp, len(u)


def timed_bound(g, m, delta):
    """Count timed flips, and flips that violate the coarsening bound."""
    if delta is None:
        return None
    act = g.act
    span = act.groupby("run")["end"].max() - act.groupby("run")["start"].min()
    out = []
    for _, r in m[m["family"] == "time"].iterrows():
        if r["same"]:
            continue
        if r["cid"].startswith("time(span"):
            lim = float(r["cid"].split("<=")[1].rstrip(")"))
            d = span.get(r["key"])
            out.append(abs(d - lim) >= delta)
        else:
            typ = r["cid"].split(":")[1].split("<=")[0]
            lim = float(r["cid"].split("<=")[1].rstrip(")"))
            sub = act[(act["run"] == r["key"]) & (act["type"] == typ)]
            d = (sub["end"] - sub["start"]).max()
            out.append(abs(d - lim) >= delta)
    return len(out), int(sum(out))


def _runs_of(k):
    if isinstance(k, tuple):
        return [int(x) for x in k]
    return [] if k == -1 else [int(k)]


def main(seeds=SEEDS, configs=None, out_dir: str = "outputs/e1_e3", runs: dict | None = None,
         workloads=None):
    """Run E1/E3. ``runs`` maps workload -> number of runs (default RUNS_1E5)."""
    runs = dict(RUNS_1E5, **(runs or {}))
    os.makedirs(out_dir, exist_ok=True)
    rows, brows, drows, frows = [], [], [], []
    t_all = time.time()
    for wl, (gen, specf) in GENERATORS.items():
        if workloads and wl not in workloads:
            continue
        spec = specf()
        for seed in seeds:
            g = gen(runs[wl], seed=seed)
            raw = run_checks(g, spec)
            key = (f"release-key-{wl}-{seed}".encode() * 2)[:32]
            for name, cfg in CONFIGS.items():
                if configs and name not in configs:
                    continue
                t0 = time.time()
                h = build(g, spec, wl, cfg, key)
                t_s = time.time() - t0
                if name.endswith("-lifted"):
                    # the verifier of a user view receives the view: lift the
                    # specification and evaluate raw and release against it
                    spec_c = lift_spec(spec, ABSTRACT_GROUP[wl])
                    raw_c = run_checks(g, spec_c)
                else:
                    spec_c, raw_c = spec, raw
                san = align(h, run_checks(h, spec_c))
                m = compare(raw_c, san)
                o_r, o_s = m["outcome_raw"], m["outcome_san"]
                for fam, sub in m.groupby("family"):
                    r_, s_ = sub["outcome_raw"], sub["outcome_san"]
                    rows.append(dict(workload=wl, seed=seed, config=name, family=fam, n=len(sub),
                                     same=int(sub["same"].sum()), undet=int((s_ == UNDET).sum()),
                                     raw_fail=int((r_ == FAIL).sum()),
                                     pass_undet=int(((r_ == PASS) & (s_ == UNDET)).sum()),
                                     fail_undet=int(((r_ == FAIL) & (s_ == UNDET)).sum()),
                                     pass_fail=int(((r_ == PASS) & (s_ == FAIL)).sum()),
                                     fail_pass=int(((r_ == FAIL) & (s_ == PASS)).sum()),
                                     records=g.n_records, runs=len(g.runs), t_sanitise=t_s,
                                     n_full_spec=int((raw["family"] == fam).sum())))
                # false alarms: runs without an injected deviation that receive a
                # FAIL in the release where the raw outcome was PASS
                dev_runs = {int(r) for r, _ in g.meta["deviations"]}
                fa: set = set()
                for k in m.loc[(o_r == PASS) & (o_s == FAIL), "key"]:
                    fa.update(x for x in _runs_of(k) if x not in dev_runs)
                n_clean = len(g.runs) - len(dev_runs & set(int(x) for x in g.runs["run"]))
                frows.append(dict(workload=wl, seed=seed, config=name, clean_runs=n_clean,
                                  false_alarm_runs=len(fa)))
                kw = cfg if isinstance(cfg, dict) else {}
                tb = timed_bound(g, m, kw.get("delta") if "TC" in kw.get("ops", ()) else None)
                ef = expected_thr_flips(g, spec, kw, raw)
                obs_thr = int((~m[m["family"] == "thr"]["same"]).sum()) if spec.thresholds else None
                pobs, pexp, pn = partition_thr_flips(g, h, spec, dict(kw, _key=key))
                brows.append(dict(workload=wl, seed=seed, config=name,
                                  timed_flips=None if tb is None else tb[0],
                                  timed_bound_violations=None if tb is None else tb[1],
                                  thr_flips_obs=obs_thr, thr_flips_exp=ef,
                                  part_flips_obs=pobs, part_flips_exp=pexp, part_trials=pn))
                # E3: deviation visibility
                fails = m[m["outcome_san"] == FAIL]
                fail_keys = {}
                for fam, sub in fails.groupby("family"):
                    ks: set = set()
                    for k in sub["key"]:
                        if isinstance(k, tuple):
                            ks.update(int(x) for x in k)
                        else:
                            ks.add(int(k))
                    fail_keys[fam] = ks
                for run, kind in g.meta["deviations"]:
                    fam = DEV_FAMILY[wl][kind]
                    drows.append(dict(workload=wl, seed=seed, config=name, kind=kind, family=fam,
                                      detected=int(run) in fail_keys.get(fam, set())))
            log.info("%s seed %s done (%.0fs)", wl, seed, time.time() - t_all)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "e1_utility.csv"), index=False)
    pd.DataFrame(brows).to_csv(os.path.join(out_dir, "e1_bounds.csv"), index=False)
    pd.DataFrame(drows).to_csv(os.path.join(out_dir, "e3_deviations.csv"), index=False)
    pd.DataFrame(frows).to_csv(os.path.join(out_dir, "e3_false_alarms.csv"), index=False)
    return {"seeds": list(seeds), "runs": runs, "wall_seconds": time.time() - t_all,
            "configs": {k: str(v) for k, v in CONFIGS.items() if not configs or k in configs}}
