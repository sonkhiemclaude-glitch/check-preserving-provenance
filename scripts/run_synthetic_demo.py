#!/usr/bin/env python3
"""Synthetic smoke test of every stage (E1-E4, tables, figures).

Output is labelled "synthetic smoke-test output, not a reference result".
Usage: python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from provpub.experiments.runner import load_config, run  # noqa: E402
from provpub.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/synthetic_demo.yaml")
    ap.add_argument("--out", help="override run.out_dir")
    a = ap.parse_args()
    over = {"run": {"kind": "smoke"}}
    if a.out:
        over["run"]["out_dir"] = a.out
    cfg = load_config(a.config, over)
    setup_logging(cfg["run"]["log_level"])
    s = run(cfg)
    print(f"synthetic smoke-test output, not a reference result: {cfg['run']['out_dir']} ({s['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
