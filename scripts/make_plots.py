#!/usr/bin/env python3
"""Figures 2-4 from result CSVs.

Usage: python scripts/make_plots.py --input outputs/<run>/results --output outputs/<run>/figures
The run kind is read from outputs/<run>/run_metadata.json; smoke runs (or runs
without metadata) are labelled "synthetic smoke-test output, not a reference result".
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from provpub.experiments.plots import make_all  # noqa: E402
from provpub.utils.io import read_run_kind  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    kind = read_run_kind(os.path.dirname(os.path.normpath(a.input)))
    for p in make_all(a.input, a.output, run_kind=kind):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
