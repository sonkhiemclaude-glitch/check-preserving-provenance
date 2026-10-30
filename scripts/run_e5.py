#!/usr/bin/env python3
"""E5 Bedrock pilot: config check, tool-call probe, collection, replay and evaluation.

Examples (run from code/):
  python scripts/run_e5.py --prepare-benchmark ../data/e5
  python scripts/run_e5.py --check-config
  python scripts/run_e5.py --probe
  python scripts/run_e5.py --mock --datasets ../data/e5/manifest.yaml --out ../runs/e5_mock
  python scripts/run_e5.py --datasets ../data/e5/manifest.yaml --out ../runs/e5_live
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provpub.e5.provider import BedrockClient, ProviderConfig, ProviderError, read_env  # noqa: E402
from provpub.e5.runner import load_config, run  # noqa: E402
from provpub.e5.workflow import prepare_benchmark  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/e5/pilot.yaml")
    ap.add_argument("--datasets", type=Path, help="CSV dataset manifest; no automatic download")
    ap.add_argument("--out", type=Path, help="new private output directory")
    ap.add_argument("--resume", action="store_true")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--check-config", action="store_true", help="validate connection config without network requests")
    modes.add_argument("--probe", action="store_true", help="one small live tool-calling capability probe")
    modes.add_argument("--discover", action="store_true", help="read the available Mantle model catalog; no generation")
    modes.add_argument("--mock", action="store_true", help="offline fixture, always labelled smoke")
    modes.add_argument("--prepare-benchmark", type=Path, help="export sklearn's bundled diabetes CSV; no API call")
    args = ap.parse_args(argv)
    # Private capture files are local publisher artifacts; publishing is a separate operation.
    os.umask(0o077)
    try:
        if args.prepare_benchmark:
            print(prepare_benchmark(args.prepare_benchmark))
            return 0
        cfg = load_config(args.config)
        provider = None
        if not args.mock:
            provider = ProviderConfig.from_env(read_env(args.env))
            provider.validate()
        if args.check_config:
            print(json.dumps({"status": "configuration_valid", "provider": provider.public(),
                              "credential_present": True}, indent=2))
        elif args.probe:
            print(json.dumps(BedrockClient(provider).probe(), indent=2))
        elif args.discover:
            print(json.dumps(BedrockClient(provider).discover_models(), indent=2))
        else:
            if args.datasets is None or args.out is None:
                raise ValueError("collection requires --datasets and --out")
            result = run(cfg, args.datasets, args.out.resolve(), provider, mock=args.mock, resume=args.resume)
            print(json.dumps(result, indent=2))
        return 0
    except ProviderError as exc:
        print(f"E5 provider error: {exc}; check the configured model/route/access. No model fallback was used.", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        # These messages are generated locally. Provider exception bodies are never rendered.
        print(f"E5 configuration or trace error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
