#!/usr/bin/env python3
"""Run experiment stages from a config (reference protocol or real data).

Usage:
  python scripts/run_experiment.py --config configs/e1_e3.yaml
  python scripts/run_experiment.py --config configs/e2.yaml --stage e2
  python scripts/run_experiment.py --config configs/experiment_template.yaml --stage real

Stages: e1 (E1/E3), e2, sensitivity, e4, tables, plots, real. Nothing is downloaded.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from provpub.experiments.runner import STAGES, load_config, run  # noqa: E402
from provpub.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", action="append", choices=STAGES, help="repeatable; default: stages in the config")
    ap.add_argument("--out", help="override run.out_dir")
    a = ap.parse_args()
    cfg = load_config(a.config, {"run": {"out_dir": a.out}} if a.out else None)
    setup_logging(cfg["run"]["log_level"])
    s = run(cfg, a.stage)
    print(f"{cfg['run']['kind']} run finished in {s['wall_seconds']:.0f}s: {cfg['run']['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
