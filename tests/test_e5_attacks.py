"""Scientific guards for E5 attacks: cluster uncertainty and public-only dictionaries."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from provpub.e5.workflow import Dataset, MockClient, make_spec, run_agent
from provpub.experiments.configurations import SALT, build

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
import e5_attacks as attacks  # noqa: E402


def test_cluster_intervals_do_not_treat_duplicates_as_independent():
    rows = []
    for ds in ("a", "b"):
        for i in range(8):
            rows.append(
                {
                    "dataset": ds,
                    "task": f"{ds}-{i}",
                    "target": "t",
                    "split": 0,
                    "y": i % 2,
                    "score": i / 8,
                    "correct": float(i % 3 == 0),
                }
            )
    frame = pd.DataFrame(rows)
    duplicated = pd.concat([frame] * 50, ignore_index=True)
    for fn in (attacks.auc_interval, attacks.accuracy_interval):
        first, second = fn(frame), fn(duplicated)
        assert first["clusters"] == second["clusters"] == 16
        for k in ("estimate", "ci_low", "ci_high"):
            assert np.isclose(first[k], second[k])


def test_cluster_bootstrap_names_are_scoped_to_dataset():
    # The same task IDs can occur in separate dataset cohorts. Opposite outcomes
    # should not accidentally receive the same resampled cluster weights.
    frame = pd.DataFrame(
        [
            {"dataset": "a", "task": "0", "correct": 0.0},
            {"dataset": "a", "task": "1", "correct": 1.0},
            {"dataset": "b", "task": "0", "correct": 1.0},
            {"dataset": "b", "task": "1", "correct": 0.0},
        ]
    )
    result = attacks.accuracy_interval(frame)
    assert result["clusters"] == 4
    assert result["estimate"] == 0.5
    assert result["ci_low"] < result["ci_high"]


def test_public_dictionary_and_load_order_score_actual_capture(tmp_path):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(360, 5)), columns=["a", "b", "c", "d", "target"])
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    ds = Dataset("fixture", path, "target", "offline fixture", 12, 42, "weighted")
    captures = tmp_path / "captures"
    captures.mkdir()
    graph, detail = run_agent(
        ds,
        ds.site_ids[:10],
        MockClient(),
        "MOCK",
        0.0,
        0,
        "g",
        captures,
        workflow="explore",
        batch_tools=True,
        profile="profile-0",
    )
    from provpub.data.prov_io import from_prov_json, to_prov_json

    to_prov_json(graph, str(tmp_path / "wire.prov.json"))
    graph = from_prov_json(str(tmp_path / "wire.prov.json"))
    spec = make_spec("MOCK", 0.0, "explore", True)
    key = b"0123456789abcdef0123456789abcdef"
    candidates = attacks.dictionary_candidates([ds])
    assert len(candidates) == 66 * 2 * 6
    # Later histories include measured tool output and are outside the public initial dictionary.
    from provpub.e5.workflow import canonical

    assert canonical(detail["captures"][0]["messages"]) in candidates
    assert canonical(detail["captures"][-1]["messages"]) not in candidates
    labels = pd.DataFrame(
        [{"gid": 0, "dataset": "fixture", "task": "g", "user": "profile-0", "sites": ";".join(ds.site_ids[:10])}]
    )
    for _name, cfg, salt, expected in [
        ("RAW", "raw", None, 1.0),
        ("salt", {"ops": ["VG"], "salted": SALT}, SALT, 1.0),
        ("keyed", {"ops": ["VG"]}, None, 0.0),
    ]:
        release = build(graph, spec, "W1", cfg, key)
        score = attacks.dictionary(release, graph, key, candidates, salt)
        assert score["initial_prompt_recovery"] == expected
    raw_score, _ = attacks.load_order(graph, graph, spec, key, labels)
    shuffled = build(graph, spec, "W1", {"ops": ["VG", "TC", "BS"], "delta": 60.0}, key)
    shuffled_score, _ = attacks.load_order(shuffled, graph, spec, key, labels)
    assert raw_score["estimate"] == 1.0 and shuffled_score["estimate"] < 1.0
    assert shuffled_score["chance"] == 0.1


def test_suppressed_release_without_timestamps_is_scored_not_crashed(tmp_path):
    # Live E5 analysis crashed on SUPPRESS, whose released activities carry no start/end columns.
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(360, 5)), columns=["a", "b", "c", "d", "target"])
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    ds = Dataset("fixture", path, "target", "offline fixture", 12, 42, "weighted")
    graph, _ = run_agent(ds, ds.site_ids[:10], MockClient(), "MOCK", 0.0, 0, "g", tmp_path,
                         workflow="explore", batch_tools=True, profile="profile-0")
    spec = make_spec("MOCK", 0.0, "explore", True)
    key = b"0123456789abcdef0123456789abcdef"
    release = build(graph, spec, "W1", "raw", key)
    release.act = release.act.drop(columns=[c for c in ("start", "end", "rank") if c in release.act])
    labels = pd.DataFrame(
        [{"gid": 0, "dataset": "fixture", "task": "g", "user": "profile-0", "sites": ";".join(ds.site_ids[:10])}]
    )
    score, predictions = attacks.load_order(release, graph, spec, key, labels)
    assert score["estimate"] is None and predictions.empty
