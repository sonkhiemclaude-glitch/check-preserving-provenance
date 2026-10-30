#!/usr/bin/env python3
"""Result tables (LaTeX bodies and CSV) from result CSVs.

Usage: python scripts/make_tables.py --input outputs/<run>/results --output outputs/<run>/tables
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from provpub.experiments.runner import label_smoke_tables  # noqa: E402
from provpub.experiments.tables import make_all  # noqa: E402
from provpub.utils.io import read_run_kind  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    run_dir = os.path.dirname(os.path.normpath(a.input))
    runs, seeds = None, [0]
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if os.path.exists(cfg_path):          # the workload table uses the sizes of the run
        import yaml
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        runs = cfg.get("e1", {}).get("runs")
        seeds = cfg.get("tables", {}).get("table3_seeds", seeds)
    made = make_all(a.input, a.output, runs=runs, seeds_table3=seeds)
    if read_run_kind(run_dir) in (None, "smoke"):
        label_smoke_tables(a.output)
    print(json.dumps(made, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
