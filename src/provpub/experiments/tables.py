"""Result tables, generated from result CSVs (no manual numbers).

Writes LaTeX bodies (tab_*.tex) and CSV
versions (tab_*.csv) to out_dir. Percentages are shares of check instances.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

E1_ORDER = ["RAW", "VG", "TC-60", "TC-600", "TC-3600", "VG+TC+BS", "FULL-record", "FULL-bucket", "FULL-bucket-group",
            "VG-salted", "VG-runscope", "VG+TC-norank-600", "FULL-bucket-fresh", "SUPPRESS", "MODHIDE", "ABSTRACT", "ABSTRACT-lifted"]
E1_LABEL = {"RAW": "Raw (reference)", "VG": "\\VG", "TC-60": "\\TC{} ($\\Delta$=60\\,s)", "TC-600": "\\TC{} ($\\Delta$=600\\,s)",
            "TC-3600": "\\TC{} ($\\Delta$=3600\\,s)", "VG+TC+BS": "\\VG+\\TC+\\BS",
            "FULL-record": "Full, record level", "FULL-bucket": "Full, bucket level", "FULL-bucket-group": "Full, bucket, input-set scope",
            "VG-salted": "Ablation: published salt", "VG-runscope": "Ablation: all run scope",
            "VG+TC-norank-600": "Ablation: no rank ($\\Delta$=600\\,s)", "FULL-bucket-fresh": "Ablation: fresh noise",
            "SUPPRESS": "Suppress all", "MODHIDE": "Module hiding", "ABSTRACT": "Structural abstraction",
            "ABSTRACT-lifted": "Structural abstraction, lifted spec."}
FAMS = [("conf", "Conf."), ("vocab", "Vocab."), ("flow", "Flow"), ("nd", "ND"), ("rep_inputs", "Rep.\\ in"),
        ("rep_counts", "Rep.\\ cnt"), ("rep_outputs", "Rep.\\ out"), ("thr", "Thr."), ("time", "Timed")]


def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x >= 1.0:
        return "100"
    if x >= 0.9995:
        return "$>$99.9"
    if x == 0:
        return "0"
    if x < 0.0005:
        return "$<$0.1"
    return f"{100 * x:.1f}"


def e1_table(RES, OUT):
    u = pd.read_csv(os.path.join(RES, "e1_utility.csv"))
    d = pd.read_csv(os.path.join(RES, "e3_deviations.csv"))
    g = u.groupby(["config", "family"])[["n", "same"]].sum()
    det = d.groupby("config")["detected"].agg(["sum", "size"])
    lines = []
    lines.append("\\begin{tabular}{@{}l" + "r" * len(FAMS) + "@{}}")
    lines.append("\\toprule")
    lines.append("Configuration & " + " & ".join(f for _, f in FAMS) + " \\\\")
    lines.append("\\midrule")
    present = set(u["config"])
    for c in E1_ORDER:
        if c not in present:
            continue
        cells = []
        for fam, _ in FAMS:
            if (c, fam) in g.index:
                n, s = g.loc[(c, fam)]
                cells.append(pct(s / n))
            else:
                cells.append("--")
        if c == "VG-salted":
            lines.append("\\midrule")
        if c == "SUPPRESS":
            lines.append("\\midrule")
        lines.append(E1_LABEL[c] + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(OUT, "tab_e1_utility.tex"), "w").write("\n".join(lines) + "\n")
    ref = "RAW" if "RAW" in present else sorted(present)[0]
    n_checks = int(u[u["config"] == ref]["n"].sum())
    n_dev = int(det.loc[ref, "size"]) if ref in det.index else 0
    changes_table(u, d, RES, OUT)
    return n_checks, n_dev


CH_ORDER = ["TC-600", "FULL-bucket", "FULL-bucket-group", "VG-runscope", "VG+TC-norank-600", "FULL-bucket-fresh",
            "SUPPRESS", "MODHIDE", "ABSTRACT", "ABSTRACT-lifted"]


def changes_table(u, d, RES, OUT):
    """Split of changed outcomes into 'cannot verify' (to UNDET) and wrong
    verdicts (PASS->FAIL, FAIL->PASS); deviation detection and false alarms."""
    fa = pd.read_csv(os.path.join(RES, "e3_false_alarms.csv"))
    lines = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Configuration & Changed & $\\to\\undet$ & PASS$\\to$FAIL & FAIL$\\to$PASS & Dev.\\ detected & False alarms \\\\",
             "\\midrule"]
    for c in ["VG+TC+BS", "FULL-record"] + CH_ORDER:
        s = u[u["config"] == c]
        if s.empty:
            continue
        n = s["n"].sum()
        ch = (s["n"] - s["same"]).sum()
        dv = d[d["config"] == c]["detected"].mean()
        f = fa[fa["config"] == c]
        far = f["false_alarm_runs"].sum() / f["clean_runs"].sum()
        lines.append(f"{E1_LABEL[c]} & {pct(ch / n)} & {pct(s['pass_undet'].sum() / n + s['fail_undet'].sum() / n)} & "
                     f"{pct(s['pass_fail'].sum() / n)} & {pct(s['fail_pass'].sum() / n)} & {pct(dv)} & {pct(far)} \\\\")
        if c == "FULL-record" or c == "FULL-bucket-fresh":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "tab_e1_changes.tex"), "w").write("\n".join(lines) + "\n")


def e1_bounds(RES):
    b = pd.read_csv(os.path.join(RES, "e1_bounds.csv"))
    s = b.groupby(["workload", "config"])[["timed_flips", "timed_bound_violations", "thr_flips_obs", "thr_flips_exp",
                                           "part_flips_obs", "part_flips_exp", "part_trials"]].sum(min_count=1)
    return s


def setup_table(RES, OUT, runs: dict | None = None, seeds=range(5)):
    """Workload table, regenerated from the generators."""
    from ..data.synthetic import GENERATORS
    from .configurations import RUNS_1E5
    runs = dict(RUNS_1E5, **(runs or {}))
    rows = []
    for wl, (gen, specf) in GENERATORS.items():
        spec = specf()
        recs, acts, ents, edges, devs, groups = [], [], [], [], [], []
        for seed in seeds:
            g = gen(runs[wl], seed=seed)
            recs.append(g.n_records); acts.append(len(g.act)); ents.append(len(g.ent)); edges.append(len(g.edge))
            devs.append(len(g.meta["deviations"]))
            vc = g.runs["config"].value_counts()
            groups.append(int((vc > 1).sum()))
        rows.append(dict(wl=wl, runs=runs[wl], records=np.mean(recs), acts=np.mean(acts), ents=np.mean(ents),
                         edges=np.mean(edges), devs=np.mean(devs), groups=np.mean(groups),
                         types=len(spec.alphabet), cons=len(spec.constraints), vocab=len(spec.vocab_rules),
                         thr=len(spec.thresholds), timed=len(spec.timed)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "workload_stats.csv"), index=False)
    names = {"W1": "W1 exploratory agent", "W2": "W2 multi-site analytics", "W3": "W3 HPC campaign"}
    lines = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
             "Workload & Runs & PROV records & Activities & Replica groups & Deviations & Declare & Vocab./Thr./Timed \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(f"{names[r['wl']]} & {r['runs']:,} & {r['records']:,.0f} & {r['acts']:,.0f} & {r['groups']:.0f} & "
                     f"{r['devs']:.0f} & {r['cons']} & {r['vocab']}/{r['thr']}/{r['timed']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "tab_workloads.tex"), "w").write("\n".join(lines) + "\n")
    return df


E2_ORDER = ["RAW", "SUPPRESS", "MODHIDE", "ABSTRACT-clear", "ABSTRACT", "VG-salted", "VG-release", "VG", "VG+TC", "VG+TC+BS",
            "FULL-record", "FULL-bucket", "FULL-bucket-group", "FULL-nosize", "NAIVE-DP"]
E2_LABEL = {"RAW": "Raw", "SUPPRESS": "Suppress all", "MODHIDE": "Module hiding", "ABSTRACT": "Structural abstraction",
            "ABSTRACT-clear": "Structural abstraction, values in clear",
            "VG-salted": "Published salt (all release scope)", "VG-release": "\\VG, all release scope, no aggregate",
            "VG": "\\VG{} (scope rule)", "VG+TC": "\\VG+\\TC", "VG+TC+BS": "\\VG+\\TC+\\BS",
            "FULL-record": "Full, record level ($\\varepsilon$=1)", "FULL-bucket": "Full, bucket level ($\\varepsilon$=1)", "FULL-bucket-group": "Full, bucket, input-set scope ($\\varepsilon$=1 per draw)",
            "FULL-nosize": "\\VG+\\TC+\\BS, sizes suppressed", "NAIVE-DP": "Naive DP (fresh noise, release scope)"}


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    m = x.mean()
    # 95% t-interval over seeds
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, h


def fmt(m, h, digits=2):
    if np.isnan(m):
        return "--"
    return f"{m:.{digits}f}$\\pm${h:.{digits}f}"


def e2_table(RES, OUT):
    p = pd.read_csv(os.path.join(RES, "e2_privacy.csv"))
    dpath = os.path.join(RES, "e2_dictionary.csv")
    dic = pd.read_csv(dpath) if os.path.exists(dpath) else None
    lines = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Configuration & A1 AUC & A2 acc. & A3 acc. & A5 acc. & Exact checks & Thr. kept \\\\", "\\midrule"]
    for c in E2_ORDER:
        s = p[p["config"] == c]
        if len(s) == 0:
            continue
        a1 = fmt(*ci(s["mia_auc"]))
        a2 = fmt(*ci(s["aia_acc"]))
        a3 = fmt(*ci(s["analyst_acc"]))
        a5 = fmt(*ci(s["pos_acc"]))
        ex = pct(s["utility_nonnumeric"].mean())
        th = pct(s["utility_thr"].mean())
        lines.append(f"{E2_LABEL[c]} & {a1} & {a2} & {a3} & {a5} & {ex} & {th} \\\\")
    base2 = fmt(*ci(p[p.config == "RAW"]["aia_base"]))
    base3 = fmt(*ci(p[p.config == "RAW"]["analyst_base"]))
    base5 = fmt(*ci(p[p.config == "RAW"]["pos_base"]))
    lines += ["\\midrule", f"Chance / majority baseline & 0.50 & {base2} & {base3} & {base5} & & \\\\", "\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "tab_e2_privacy.tex"), "w").write("\n".join(lines) + "\n")
    return p, dic


def _tex_to_csv(path_tex: str) -> None:
    """Plain CSV copy of a generated LaTeX table body (for readers without LaTeX)."""
    rows = []
    for line in open(path_tex, encoding="utf-8"):
        line = line.strip()
        if not line.endswith("\\\\") or line.startswith(("\\toprule", "\\midrule", "\\bottomrule", "\\begin", "\\end")):
            continue
        cells = [c.strip().replace("$\\pm$", "+/-").replace("\\,", " ").replace("{}", "").replace("\\to", "->")
                 .replace("\\undet", "UNDET").replace("\\", "").replace("$", "") for c in line[:-2].split("&")]
        rows.append(cells)
    if rows:
        pd.DataFrame(rows[1:], columns=rows[0]).to_csv(path_tex[:-4] + ".csv", index=False)


def make_all(res_dir: str, out_dir: str, runs: dict | None = None, seeds_table3=range(5)) -> dict:
    """Generate every table whose input CSVs exist in res_dir."""
    os.makedirs(out_dir, exist_ok=True)
    made: dict = {}
    if os.path.exists(os.path.join(res_dir, "e1_utility.csv")):
        n_checks, n_dev = e1_table(res_dir, out_dir)
        made["e1"] = {"check_instances_per_config": n_checks, "deviations": n_dev}
        e1_bounds(res_dir).to_csv(os.path.join(out_dir, "e1_bounds_summary.csv"))
        made["table3"] = len(setup_table(res_dir, out_dir, runs, seeds_table3))
    if os.path.exists(os.path.join(res_dir, "e2_privacy.csv")):
        e2_table(res_dir, out_dir)
        made["e2"] = True
    for f in os.listdir(out_dir):
        if f.endswith(".tex"):
            _tex_to_csv(os.path.join(out_dir, f))
    return made
