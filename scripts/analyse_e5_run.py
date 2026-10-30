#!/usr/bin/env python3
"""Offline E5 readiness report from captured traces; never calls an LLM.

Separates primary/replay metrics, records the collected volume, and
reports threshold changes without hiding them behind a nonnumeric average.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from provpub.e5.workflow import atomic_json, digest  # noqa: E402
from provpub.utils.io import source_manifest  # noqa: E402


def analyse(root: Path, out: Path) -> dict:
    state = json.loads((root / "collection_state.json").read_text())
    if state['status'] != 'completed':
        raise ValueError('collection is not complete')
    meta = json.loads((root / 'run_metadata.json').read_text())
    if meta['source_sha256'] != state['source_sha256']:
        raise ValueError('capture and run metadata source snapshots differ')
    rows = []
    for record in state['runs'].values():
        trace = root / 'raw/traces' / f"run_{record['run']:05d}.prov.json"
        capture = root / 'raw/captures' / f"run_{record['run']:05d}.json"
        if digest(trace.read_bytes()) != record['trace_sha256'] or digest(capture.read_bytes()) != record['capture_sha256']:
            raise ValueError('completed capture or trace changed')
        captures = json.loads(capture.read_text())
        rows.append({'run': record['run'], 'dataset': record['dataset'], 'mode': record['mode'],
                     'steps': record['steps'], 'sites': record['sites'], **record['metrics'],
                     'input_tokens': sum(r['input_tokens'] for r in captures),
                     'output_tokens': sum(r['output_tokens'] for r in captures)})
    frame = pd.DataFrame(rows)
    outcomes = pd.read_csv(root / 'results/outcomes.csv')
    if frame[frame['mode'] == 'live'].empty:
        raise ValueError('readiness evidence requires real LLM runs')
    journal = [json.loads(s) for s in (root / 'requests.jsonl').read_text().splitlines()]
    success = [r for r in journal if r['status'] == 'ok']
    prompt, completion = (sum(r[k] for r in success) for k in ('input_tokens', 'output_tokens'))
    report = {'status': 'completed', 'run_directory': str(root.resolve()), 'summary': state['summary'],
              'dataset_mode_counts': frame.groupby(['dataset', 'mode']).size().rename('runs').reset_index().to_dict('records'),
              'verdicts_by_family': outcomes.groupby('family').agg(instances=('same', 'size'),
                                                               preserved=('same', 'sum')).reset_index().to_dict('records'),
              'replica_protocol': 'recorded-decision replay and tool re-execution; no fresh inference',
              'usage': {'input_tokens': prompt, 'output_tokens': completion,
                        'attempts': sum(r['status'] == 'started' for r in journal),
                        'rejected_responses': sum(r['status'] == 'response_rejected' for r in journal)},
              'source_sha256': state['source_sha256'],
              'analysis_source_matches_capture': source_manifest() == state['source_sha256'],
              'not_evaluated': ['W1 standalone exploration', 'A1-A5 privacy attacks', 'independent model-inference replicas']}
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'metrics_by_run.csv', index=False)
    atomic_json(out / 'readiness.json', report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(analyse(args.run, args.out), indent=2))


if __name__ == '__main__':
    main()
