#!/usr/bin/env python3
"""Compare measured scientific CSVs against reference CSVs (for example results/).

Timing is machine dependent and excluded. Reference rows are restricted to the
seed/config/workload combinations in the measured file. E4 is validated by
the benchmark's release hashes, not by matching timings from another machine.
Exits nonzero on a mismatch; emits a machine-readable report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FILES = ("e1_utility.csv", "e1_bounds.csv", "e3_deviations.csv", "e3_false_alarms.csv",
         "e2_privacy.csv", "e2_dictionary.csv", "e2_group_budget.csv", "e2_eps_sweep.csv",
         "e2_delta_sweep.csv", "e2_sensitivity.csv")


def compare(measured: Path, reference: Path) -> dict:
    report: dict = {"files": {}, "matched": True, "tolerance": {"rtol": 1e-12, "atol": 1e-12}}
    for name in FILES:
        path = measured / name
        if not path.is_file():
            continue
        try:
            actual = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            report["files"][name] = {"status": "skipped", "reason": "empty optional sweep"}
            continue
        if actual.empty:
            report["files"][name] = {"status": "skipped", "reason": "empty optional sweep"}
            continue
        try:
            expected = pd.read_csv(reference / name)
            keys = [k for k in ("workload", "seed", "config", "eps", "unit", "delta", "variant") if k in actual.columns]
            combinations = actual[keys].drop_duplicates()
            expected = expected.merge(combinations, on=keys, how="inner", validate="many_to_one")
            columns = [c for c in actual.columns if c != "t_sanitise"]
            if set(columns) != set(expected.columns) - {"t_sanitise"}:
                raise AssertionError("scientific column schema differs")
            # Sorting the full rows also handles repeated deviation kinds, whose
            # prototype CSV has no run identifier. Preserve duplicate counts.
            a = actual[columns].sort_values(columns, kind="stable").reset_index(drop=True)
            b = expected[columns].sort_values(columns, kind="stable").reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
            report["files"][name] = {"status": "matched", "rows": len(a)}
        except (AssertionError, OSError, ValueError, KeyError) as exc:
            report["matched"] = False
            report["files"][name] = {"status": "mismatch", "detail": str(exc)[:2000]}
    if not any(f["status"] in ("matched", "mismatch") for f in report["files"].values()):
        report.update(matched=False, error="no nonempty scientific result files to compare")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured", type=Path, required=True, help="results directory of the new run")
    ap.add_argument("--reference", type=Path, required=True, help="reference CSV directory, e.g. results/")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = compare(args.measured, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
