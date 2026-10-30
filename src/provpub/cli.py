"""Command-line interface: python -m provpub.cli --help

Publisher side : generate, sanitize (optionally with a verdict-split report)
Verifier side  : check
Experiments    : run (config-driven), demo (synthetic smoke test)
Utilities      : spec-export
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .utils.logging import setup_logging

log = logging.getLogger("provpub")


def _load(path: str, fmt: str):
    from .data.adapters import load_graph
    return load_graph(path, fmt)


def _write(g, path: str, fmt: str) -> None:
    from .data.adapters import spans_to_otlp_json
    from .data.prov_io import to_otel_spans, to_prov_json
    if fmt == "prov-json":
        to_prov_json(g, path)
        return
    spans = to_otel_spans(g)
    doc = spans if fmt == "provpub-spans" else spans_to_otlp_json(spans)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def cmd_generate(a) -> int:
    from .data.spec_io import save_spec_yaml
    from .data.synthetic import GENERATORS
    gen, specf = GENERATORS[a.workload]
    g = gen(a.runs, seed=a.seed)
    _write(g, a.out, a.format)
    if a.spec_out:
        save_spec_yaml(specf(), a.spec_out)
    log.info("wrote %s (%d runs, %d PROV records); synthetic workload %s", a.out, len(g.runs), g.n_records, a.workload)
    return 0


def cmd_sanitize(a) -> int:
    from .checks.checks import align, compare, run_checks
    from .data.prov_io import to_prov_json
    from .data.spec_io import load_spec_yaml
    from .experiments.real_data import read_release_key
    from .methods.operators import sanitise
    from .metrics.metrics import verdict_transitions
    from .utils.random import key_fingerprint
    spec = load_spec_yaml(a.spec)
    g = _load(a.input, a.format)
    key = read_release_key(a.key_file, a.key_env)
    memo_map: dict[str, bool | str] = {"partition": True, "input-set": "group", "fresh": False}
    memo = memo_map[a.memo]
    h = sanitise(g, spec, key, ops=tuple(a.ops.split(",")), delta=a.delta, eps=a.eps, unit=a.unit, memo=memo,
                 sampler=a.sampler)
    to_prov_json(h, a.out)
    log.info("release written to %s (key fingerprint %s)", a.out, key_fingerprint(key))
    if a.report:
        m = compare(run_checks(g, spec), align(h, run_checks(h, spec)))
        rep = verdict_transitions(m)
        rep["by_family_unchanged"] = m.groupby("family")["same"].mean().round(6).to_dict()
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        log.info("verdict report written to %s", a.report)
    return 0


def cmd_check(a) -> int:
    from .checks.checks import run_checks
    from .data.spec_io import load_spec_yaml
    spec = load_spec_yaml(a.spec)
    g = _load(a.input, a.format)
    out = run_checks(g, spec)
    out["key"] = out["key"].astype(str)
    out.to_csv(a.out, index=False)
    counts = out["outcome"].map({1: "PASS", 0: "FAIL", -1: "UNDET"}).value_counts().to_dict()
    log.info("%d check instances: %s -> %s", len(out), counts, a.out)
    return 0


def cmd_run(a) -> int:
    from .experiments.runner import load_config, run
    over = {"run": {"out_dir": a.out}} if a.out else None
    cfg = load_config(a.config, over)
    setup_logging(cfg["run"]["log_level"])
    summary = run(cfg, a.stages.split(",") if a.stages else None)
    log.info("done in %.1fs; outputs in %s", summary["wall_seconds"], cfg["run"]["out_dir"])
    return 0


def cmd_demo(a) -> int:
    from .experiments.runner import load_config, run
    cfg = load_config(a.config, {"run": {"kind": "smoke", "out_dir": a.out}})
    stages = a.stages.split(",") if a.stages else ["e1", "e2", "sensitivity", "e4", "tables", "plots"]
    run(cfg, stages)
    log.info("synthetic smoke-test output, not a reference result: %s", a.out)
    return 0


def cmd_spec_export(a) -> int:
    from .data.spec_io import save_spec_yaml
    from .data.synthetic import GENERATORS
    save_spec_yaml(GENERATORS[a.workload][1](), a.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provpub", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    fmts = ("prov-json", "provpub-spans", "otlp-json")

    s = sub.add_parser("generate", help="write a synthetic workload (W1, W2, W3) to a trace file")
    s.add_argument("--workload", choices=("W1", "W2", "W3"), required=True)
    s.add_argument("--runs", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--format", choices=fmts, default="prov-json")
    s.add_argument("--out", required=True)
    s.add_argument("--spec-out", help="also write the workload's declared specification (YAML)")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("sanitize", help="publisher: sanitise a trace file into a PROV-JSON release")
    s.add_argument("--input", required=True)
    s.add_argument("--format", choices=fmts, default="prov-json")
    s.add_argument("--spec", required=True, help="declared specification (YAML)")
    s.add_argument("--out", required=True)
    s.add_argument("--key-file", help="file holding the secret release key")
    s.add_argument("--key-env", help="environment variable holding the secret release key")
    s.add_argument("--ops", default="VG,TC,BS,DPN")
    s.add_argument("--delta", type=float, default=60.0, help="coarsening width in seconds")
    s.add_argument("--eps", type=float, default=1.0, help="privacy budget per noise draw")
    s.add_argument("--unit", choices=("record", "bucket", "suppress"), default="bucket")
    s.add_argument("--memo", choices=("partition", "input-set", "fresh"), default="partition")
    s.add_argument("--sampler", choices=("exact", "inversion"), default="exact")
    s.add_argument("--report", help="write the verdict split raw vs release (JSON)")
    s.set_defaults(func=cmd_sanitize)

    s = sub.add_parser("check", help="verifier: run all declared checks on a trace or release")
    s.add_argument("--input", required=True)
    s.add_argument("--format", choices=fmts, default="prov-json")
    s.add_argument("--spec", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("run", help="run experiment stages from a YAML config")
    s.add_argument("--config", required=True)
    s.add_argument("--stages", help="comma-separated subset of: e1,e2,sensitivity,e4,tables,plots,real")
    s.add_argument("--out", help="override run.out_dir")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("demo", help="synthetic smoke test of every stage (not a reference result)")
    s.add_argument("--config", help="smoke config (default: built-in small sizes)")
    s.add_argument("--out", default=os.path.join("outputs", "synthetic_demo"))
    s.add_argument("--stages")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("spec-export", help="write the declared specification of W1, W2 or W3 as YAML")
    s.add_argument("--workload", choices=("W1", "W2", "W3"), required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_spec_export)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    setup_logging(a.log_level)
    try:
        return a.func(a)
    except (ValueError, KeyError, FileNotFoundError, NotImplementedError) as e:
        log.error("%s: %s", type(e).__name__, e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
