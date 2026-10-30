#!/usr/bin/env python3
"""Table 7 (E5, real LLM agent) LaTeX body and CSV from the E5 study results.

Utility is pooled over both workflows; A1/A2/A3/A5 are reported on the
regression (W2-style) workflow and A4 on the exploration (W1-style) workflow,
as in E2. Intervals are the 95% cluster-bootstrap intervals of the analysis.

Usage: python scripts/make_e5_tables.py --input results/e5_study --output results/e5_study/tables
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

ROWS = [
    ("RAW", "Raw"),
    ("SUPPRESS", "Suppress all"),
    ("MODHIDE", "Module hiding"),
    ("ABSTRACT-lifted", "Structural abstraction, lifted spec."),
    ("VG-salted", "Published salt"),
    ("VG", r"\VG{} (scope rule)"),
    ("VG-runscope", r"\VG, run scope"),
    ("VG+TC-norank-600", r"\VG+\TC{} without ranks ($\Delta$=600\,s)"),
    ("VG+TC+BS", r"\VG+\TC+\BS"),
    ("FULL-record", r"Full, record level ($\varepsilon$=1)"),
    ("FULL-bucket", r"Full, bucket level ($\varepsilon$=1)"),
    ("FULL-bucket-group", r"Full, bucket, input-set scope"),
    ("FULL-bucket-fresh", r"Full, bucket, fresh noise"),
    ("FULL-nosize", r"\VG+\TC+\BS, sizes suppressed"),
]
# Attacks on an abstracted release do not depend on the specification used to check it.
ATTACK_SOURCE = {"ABSTRACT-lifted": "ABSTRACT"}


def interval(x: dict | None) -> tuple[str, float | None, float | None, float | None]:
    if not x or x.get("estimate") is None:
        return "--", None, None, None
    est, lo, hi = x["estimate"], x["ci_low"], x["ci_high"]
    return f"{est:.2f}\\,{{\\scriptsize[{lo:.2f}, {hi:.2f}]}}", est, lo, hi


def result_file(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"none of the expected E5 result files exist in {root}: {', '.join(names)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    input_dir = Path(a.input)
    utility = pd.read_csv(result_file(input_dir, "utility_by_family.csv", "e5_utility_by_family.csv"))
    attacks_path = result_file(input_dir, "attacks.json", "e5_attacks.json")
    with open(attacks_path, encoding="utf-8") as f:
        attacks = {(r["workflow"], r["config"]): r for r in json.load(f)}
    exact = utility[~utility.family.isin(["thr", "thr_counts", "time"])].groupby("config")[["instances", "preserved"]].sum()
    thr = utility[utility.family == "thr"].groupby("config")[["instances", "preserved"]].sum()
    lines, rows = [], []
    for config, label in ROWS:
        reg = attacks[("regression", ATTACK_SOURCE.get(config, config))]
        exp = attacks[("explore", ATTACK_SOURCE.get(config, config))]
        e = 100 * exact.loc[config, "preserved"] / exact.loc[config, "instances"]
        t = 100 * thr.loc[config, "preserved"] / thr.loc[config, "instances"]
        cells = {k: interval(reg[k]) for k in ("A1", "A2", "A3_profiles", "A5")}
        a4 = exp["A4"]["initial_prompt_recovery"]
        lines.append(" & ".join([label, *(cells[k][0] for k in ("A1", "A2", "A3_profiles", "A5")),
                                 f"{a4:.2f}", f"{e:.1f}", f"{t:.0f}"]) + r" \\")
        rows.append({"config": config, "label": label, "exact_checks_pct": e, "thr_kept_pct": t, "a4_initial_recovery": a4,
                     **{f"{k}_{s}": v for k in cells for s, v in zip(("est", "lo", "hi"), cells[k][1:])}})
    base = attacks[("regression", "RAW")]
    body = "\n".join([
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Configuration & A1 AUC & A2 acc. & A3 acc. & A5 acc. & A4 rec. & Exact checks & Thr. kept \\",
        r"\midrule",
        *lines,
        r"\midrule",
        (f"Chance / majority baseline & 0.50 & {base['A2']['baseline']:.2f} & {base['A3_profiles']['baseline']:.2f}"
         f" & {base['A5']['chance']:.2f} & & & \\\\"),
        r"\bottomrule",
        r"\end{tabular}",
    ])
    os.makedirs(a.output, exist_ok=True)
    with open(os.path.join(a.output, "tab_e5.tex"), "w", encoding="utf-8") as f:
        f.write(body + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(a.output, "tab_e5.csv"), index=False)
    print(json.dumps({"rows": len(rows), "output": a.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
