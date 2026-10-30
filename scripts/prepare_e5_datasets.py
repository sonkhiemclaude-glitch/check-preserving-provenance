#!/usr/bin/env python3
"""Explicitly provision the five public regression benchmarks for the E5 extension."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provpub.e5.benchmarks import provision  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--download-public", action="store_true", help="explicitly download four small UCI archives")
    ap.add_argument("--sites", type=int, default=12)
    args = ap.parse_args()
    print(provision(args.out, download_public=args.download_public, sites=args.sites))


if __name__ == "__main__":
    main()
