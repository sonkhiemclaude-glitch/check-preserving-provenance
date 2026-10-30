"""Offline evaluation is reversible and keeps scripted faults distinct from capture."""
import importlib.util
from pathlib import Path

import pandas as pd
from test_e5 import dataset_manifest

from provpub.e5.runner import load_config, run
from provpub.e5.workflow import digest


def test_offline_faults_and_ablations_do_not_modify_capture(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'analysis/e5_checks.py'
    module = importlib.util.spec_from_file_location('e5_checks', script)
    checks = importlib.util.module_from_spec(module)
    module.loader.exec_module(checks)
    out = tmp_path / 'mock'
    run({**load_config(None), 'live_runs': 1}, dataset_manifest(tmp_path), out, mock=True)
    before = {str(p): digest(p.read_bytes()) for p in (out / 'raw').rglob('*') if p.is_file()}
    key = next((tmp_path / '.provpub-release-keys').glob('*.key'))
    receipt = checks.evaluate(out / 'raw', tmp_path / 'analysis', str(key))
    assert receipt['model_requests'] == 0 and receipt['fault_evaluations'] == 18
    assert receipt['all_injected_faults_detected']
    assert before == {str(p): digest(p.read_bytes()) for p in (out / 'raw').rglob('*') if p.is_file()}
    utility = pd.read_csv(tmp_path / 'analysis/operator_utility.csv')
    exact = utility[(utility['config'] == 'FULL-record') & ~utility['family'].isin(['thr', 'time'])]
    assert (exact['instances'] == exact['preserved']).all()
    ablation = utility[(utility['config'] == 'VG-runscope') & utility['family'].str.startswith('rep_')]
    assert (ablation['instances'] > ablation['preserved']).any()
