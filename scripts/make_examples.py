#!/usr/bin/env python3
"""Regenerate examples/ (small SYNTHETIC files for documentation and tests).

Creates examples/synthetic_data/ (one W2 workload in the three supported
formats plus its specification) and examples/real_data_layout/ (a directory in
the layout expected for real traces, filled with synthetic data so that the
'real' stage can be tried end to end). None of these files is real data.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from provpub.data.adapters import spans_to_otlp_json  # noqa: E402
from provpub.data.prov_io import to_otel_spans, to_prov_json  # noqa: E402
from provpub.data.spec_io import save_spec_yaml  # noqa: E402
from provpub.data.synthetic import make_sites, w1_spec, w2_generate, w2_spec, w3_spec  # noqa: E402

EX = os.path.join(HERE, "..", "examples")


def main() -> int:
    sd = os.path.join(EX, "synthetic_data")
    os.makedirs(sd, exist_ok=True)
    g = w2_generate(15, seed=0)
    to_prov_json(g, os.path.join(sd, "w2_15runs.prov.json"))
    spans = to_otel_spans(g)
    with open(os.path.join(sd, "w2_15runs.provpub-spans.json"), "w") as f:
        json.dump(spans, f)
    with open(os.path.join(sd, "w2_15runs.otlp.json"), "w") as f:
        json.dump(spans_to_otlp_json(spans), f)
    for name, spec in (("w1_spec.yaml", w1_spec()), ("w2_spec.yaml", w2_spec()), ("w3_spec.yaml", w3_spec())):
        save_spec_yaml(spec, os.path.join(sd, name))
    root = os.path.join(EX, "real_data_layout")
    os.makedirs(os.path.join(root, "traces"), exist_ok=True)
    os.makedirs(os.path.join(root, "truth"), exist_ok=True)
    with open(os.path.join(root, "manifest.yaml"), "w") as f:
        yaml.safe_dump({"format": "prov-json", "source": "SYNTHETIC example (W2 generator), not real data",
                        "collector": "provpub synthetic generator", "collected_at": None,
                        "notes": "replace traces/, spec.yaml and truth/ with real files"}, f, sort_keys=False)
    save_spec_yaml(w2_spec(), os.path.join(root, "spec.yaml"))
    rows = []
    sites = make_sites(12, np.random.default_rng(1001))
    for i, seed in enumerate((1, 2)):
        gi = w2_generate(25, seed=seed, sites=sites)
        name = f"release_part{i}.json"
        to_prov_json(gi, os.path.join(root, "traces", name))
        for r in gi.runs["run"]:
            t = gi.meta["truth"][int(r)]
            rows.append({"run_file": name, "run": int(r), "user": t["user"], "sites": ";".join(t["sites"])})
    pd.DataFrame(rows).to_csv(os.path.join(root, "truth", "runs.csv"), index=False)
    pd.DataFrame([{"site": s.name, "approx_records": s.base_records, "row_width": s.row_width,
                   "load_seconds": round(s.latency + s.base_records / s.throughput, 3)} for s in sites]).to_csv(
        os.path.join(root, "truth", "sites_aux.csv"), index=False)
    print("examples regenerated (synthetic)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
