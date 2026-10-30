"""Reference comparisons must detect changed values and missing measurements."""
import runpy
from pathlib import Path

import pandas as pd

compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "compare_reference.py"))["compare"]


def test_reference_subset_excludes_only_machine_timings(tmp_path):
    ref, actual = tmp_path / "reference", tmp_path / "actual"
    ref.mkdir()
    actual.mkdir()
    rows = pd.DataFrame([{"config": "VG", "seed": 0, "family": "conf", "same": 10, "t_sanitise": 9.0},
                         {"config": "VG", "seed": 0, "family": "flow", "same": 10, "t_sanitise": 9.0},
                         {"config": "VG", "seed": 1, "family": "conf", "same": 11, "t_sanitise": 9.0}])
    rows.to_csv(ref / "e1_utility.csv", index=False)
    measured = rows[rows.seed == 0].copy()
    measured["t_sanitise"] = 0.1
    measured.to_csv(actual / "e1_utility.csv", index=False)
    assert compare(actual, ref)["matched"]
    measured.iloc[:1].to_csv(actual / "e1_utility.csv", index=False)
    assert not compare(actual, ref)["matched"]
    measured.loc[0, "same"] = 9
    measured.to_csv(actual / "e1_utility.csv", index=False)
    assert not compare(actual, ref)["matched"]


def test_empty_directory_does_not_count_as_reproduced(tmp_path):
    assert not compare(tmp_path, tmp_path)["matched"]
