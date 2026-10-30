"""E2: inference by an honest-but-curious verifier.

Attacks A1 (site membership), A2 (size class), A3 (analyst), A5 (load order) on
W2 releases; A4 (prompt dictionary) on W1; sweeps over the privacy budget and
the coarsening width; group-budget accounting m_p; and the
sensitivity run without the generator choices that ease the attacks.

Reference protocol: seeds 0-9, 1,500 W2 runs per release, eps grid
(0.01, 0.03, 0.1, 0.3, 1, 3, 10), Delta grid (1, 10, 60, 300, 900, 3600) s,
sensitivity seeds 0-4 (configs/e2.yaml). Keys are the public
experiment keys of the reference runs.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from ..attacks import attacks as A
from ..checks.checks import align, compare, run_checks
from ..data.synthetic import (
    W1_DATASETS,
    W1_QUESTIONS,
    make_analysts,
    make_sites,
    w1_generate,
    w1_prompt,
    w1_spec,
    w2_generate,
    w2_spec,
)
from .configurations import E2_CONFIGS, SALT, build

log = logging.getLogger(__name__)

EPS_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DELTA_GRID = (1.0, 10.0, 60.0, 300.0, 900.0, 3600.0)
SENS_CONFIGS = ["RAW", "ABSTRACT", "VG", "VG+TC", "VG+TC+BS", "FULL-bucket", "FULL-bucket-group", "FULL-nosize"]
ATTACKER = ("known_frac=0.2 labelled runs; aux record count/row width/load duration within +/-10%; "
            "shadow release with own population and key")


def populations(seed: int, size_mode: str = "classes"):
    """Target and shadow site/analyst populations (shadow is the adversary's own)."""
    rng = np.random.default_rng(1000 + seed)
    sites, ans = make_sites(12, rng, size_mode), make_analysts(6, rng)
    rng2 = np.random.default_rng(5000 + seed)
    return sites, ans, make_sites(12, rng2, size_mode), make_analysts(6, rng2)


def evaluate(g, gs, spec, sites, ans, cfg, name, seed, raw_checks) -> dict:
    """All W2 attacks plus check preservation for one configuration and seed."""
    key = (f"target-key-{seed}".encode() * 3)[:32]
    skey = (f"shadow-key-{seed}".encode() * 3)[:32]
    h = build(g, spec, "W2", cfg, key)
    hs = build(gs, spec, "W2", cfg, skey)
    A.attach_truth(g, h, key, spec)
    A.attach_truth(gs, hs, skey, spec)
    salt = SALT if name == "VG-salted" else None
    mia, _per, mia_fresh = A.membership_attack(h, spec, g.meta["truth"], sites, list(range(12)), seed=seed, salt=salt)
    if name.startswith("ABSTRACT"):
        aia, base = float("nan"), float("nan")
    else:
        aia, base = A.size_class_attack(h, hs, spec)
    ana, abase = A.analyst_attack(h, g.meta["truth"], ans, seed=seed, salt=salt)
    pos, pos_base = A.position_attack(h, g, spec, key)
    m = compare(raw_checks, align(h, run_checks(h, spec)))
    return dict(config=name, seed=seed, mia_auc=mia, mia_auc_unlinked=mia_fresh, aia_acc=aia, aia_base=base,
                analyst_acc=ana, analyst_base=abase, pos_acc=pos, pos_base=pos_base,
                utility_all=float(m["same"].mean()),
                utility_thr=float(m[m["family"] == "thr"]["same"].mean()),
                utility_time=float(m[m["family"] == "time"]["same"].mean()),
                utility_nonnumeric=float(m[~m["family"].isin(["thr", "time"])]["same"].mean()))


def group_budget(g, spec, seed) -> dict:
    """m_p = number of distinct input sets containing partition p, and the
    per-site sum over versions."""
    src = g.ent[g.ent["type"] == spec.source_type]
    sets = src.groupby("run")["part_id"].apply(lambda s: ";".join(sorted(s)))
    m_p = src.assign(s=src["run"].map(sets)).groupby("part_id")["s"].nunique()
    site = m_p.index.to_series().str.split("/").str[0]
    per_site = m_p.groupby(site.values).agg(["sum", "size"])
    return dict(seed=seed, partitions=len(m_p), median=float(m_p.median()), mean=float(m_p.mean()),
                p90=float(m_p.quantile(0.9)), max=int(m_p.max()), versions_max=int(per_site["size"].max()),
                site_sum_median=float(per_site["sum"].median()), site_sum_max=int(per_site["sum"].max()))


def main(seeds=tuple(range(10)), out_dir: str = "outputs/e2", n_runs: int = 1500, eps_grid=EPS_GRID,
         delta_grid=DELTA_GRID, configs=None, w1_runs: int = 720, dictionary: bool = True) -> dict:
    """Run E2. Results are written after every seed."""
    os.makedirs(out_dir, exist_ok=True)
    spec = w2_spec()
    rows, eps_rows, d_rows, mp_rows = [], [], [], []
    t0 = time.time()
    for seed in seeds:
        sites, ans, sites2, ans2 = populations(seed)
        g = w2_generate(n_runs, seed=seed, sites=sites, analysts=ans)
        gs = w2_generate(n_runs, seed=seed + 50, sites=sites2, analysts=ans2)
        raw = run_checks(g, spec)
        mp_rows.append(group_budget(g, spec, seed))
        for name, cfg in E2_CONFIGS.items():
            if configs and name not in configs:
                continue
            rows.append(evaluate(g, gs, spec, sites, ans, cfg, name, seed, raw))
        for eps in eps_grid:
            for unit, memo in (("bucket", True), ("record", True), ("bucket", "group")):
                cfg = dict(ops=("VG", "TC", "BS", "DPN"), delta=60.0, unit=unit, eps=eps, memo=memo)
                r = evaluate(g, gs, spec, sites, ans, cfg, f"FULL-{unit}", seed, raw)
                r["eps"] = eps
                r["unit"] = unit if memo is True else f"{unit}-group"
                eps_rows.append(r)
        for delta in delta_grid:
            r = evaluate(g, gs, spec, sites, ans, dict(ops=("VG", "TC", "BS"), delta=delta), "VG+TC+BS", seed, raw)
            r["delta"] = delta
            d_rows.append(r)
        log.info("E2 seed %s done (%.0fs)", seed, time.time() - t0)
        pd.DataFrame(mp_rows).to_csv(os.path.join(out_dir, "e2_group_budget.csv"), index=False)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "e2_privacy.csv"), index=False)
        pd.DataFrame(eps_rows).to_csv(os.path.join(out_dir, "e2_eps_sweep.csv"), index=False)
        pd.DataFrame(d_rows).to_csv(os.path.join(out_dir, "e2_delta_sweep.csv"), index=False)
    if dictionary:
        dictionary_attack(seeds, out_dir, w1_runs)
    return {"seeds": list(seeds), "n_runs": n_runs, "wall_seconds": time.time() - t0, "attacker": ATTACKER,
            "configs": {k: str(v) for k, v in E2_CONFIGS.items() if not configs or k in configs}}


def dictionary_candidates() -> list[str]:
    """The 16,400 prompts the W1 planner can produce (A4)."""
    return [w1_prompt(qi, ds) + suffix for ds in W1_DATASETS for qi in range(len(W1_QUESTIONS))
            for suffix in [f" [step {k}]" for k in range(40)] + [" [final]"]]


def dictionary_attack(seeds, out_dir: str, w1_runs: int = 720) -> pd.DataFrame:
    """A4 on W1 prompts: raw, published salt and keyed pseudonyms."""
    spec1 = w1_spec()
    cands = dictionary_candidates()
    drows = []
    for seed in seeds:
        g1 = w1_generate(w1_runs, seed=seed)
        key = (f"w1-key-{seed}".encode() * 3)[:32]
        for name, cfg, salt in (("RAW", "raw", None), ("VG-salted", dict(ops=("VG",), salted=SALT), SALT),
                                ("VG", dict(ops=("VG",)), None)):
            h = build(g1, spec1, "W1", cfg, key)
            rate, n = A.dictionary_attack(h, cands, salt)
            drows.append(dict(config=name, seed=seed, recovered=rate, distinct_prompts=n, candidates=len(cands)))
    df = pd.DataFrame(drows)
    df.to_csv(os.path.join(out_dir, "e2_dictionary.csv"), index=False)
    return df


def sensitivity(seeds=tuple(range(5)), out_dir: str = "outputs/e2", n_runs: int = 1500, configs=SENS_CONFIGS) -> dict:
    """W2 variant without the generator choices that ease the attacks:
    continuous site sizes, prompts without site names, random load order."""
    os.makedirs(out_dir, exist_ok=True)
    spec = w2_spec()
    rows = []
    t0 = time.time()
    for seed in seeds:
        sites, ans, sites2, ans2 = populations(seed, "continuous")
        kw: dict = dict(prompt_sites=False, load_order="random")
        g = w2_generate(n_runs, seed=seed, sites=sites, analysts=ans, **kw)
        gs = w2_generate(n_runs, seed=seed + 50, sites=sites2, analysts=ans2, **kw)
        raw = run_checks(g, spec)
        for name in configs:
            r = evaluate(g, gs, spec, sites, ans, E2_CONFIGS[name], name, seed, raw)
            r["variant"] = "continuous-sizes, no-site-prompts, random-order"
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "e2_sensitivity.csv"), index=False)
        log.info("sensitivity seed %s done (%.0fs)", seed, time.time() - t0)
    return {"sensitivity_seeds": list(seeds), "n_runs": n_runs, "wall_seconds": time.time() - t0}
