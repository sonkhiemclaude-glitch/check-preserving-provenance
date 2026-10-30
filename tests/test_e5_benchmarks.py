"""Public benchmark preparation preserves targets and records verifiable source transformations."""
import io
import json
import zipfile

import pandas as pd
import pytest

from provpub.e5 import benchmarks
from provpub.e5.workflow import load_datasets


def test_auto_mpg_retains_missing_feature_without_losing_quoted_names():
    raw = b'18.0 8 307.0 130.0 3504. 12.0 70 1 "chevrolet chevelle malibu"\n15.0 8 350.0 ? 3693. 11.5 70 1 "buick skylark 320"\n'
    frame, note = benchmarks.parse_public('auto_mpg', raw)
    assert len(frame) == 2 and list(frame['mpg']) == [18., 15.]
    assert pd.isna(frame.loc[1, 'horsepower']) and 'car_name' not in frame
    assert 'no rows removed' in note


def test_abalone_onehot_does_not_encode_target_or_invent_categories():
    raw = b'M,0.4,0.3,0.1,0.5,0.2,0.1,0.15,15\nF,0.5,0.4,0.2,0.6,0.3,0.2,0.2,7\n'
    frame, _ = benchmarks.parse_public('abalone', raw)
    assert list(frame['rings']) == [15, 7]
    assert list(frame['sex_M']) == [1, 0] and list(frame['sex_F']) == [0, 1]
    assert list(frame['sex_I']) == [0, 0] and 'sex' not in frame
    with pytest.raises(ValueError, match='category'):
        benchmarks.parse_public('abalone', raw.replace(b'M,', b'X,'))


def test_public_provision_is_explicit_atomic_and_hash_checked(tmp_path, monkeypatch):
    root = tmp_path / 'suite'
    with pytest.raises(ValueError, match='explicit'):
        benchmarks.provision(root)
    assert not root.exists()
    sources = {'auto_mpg': b'18 8 307 130 3504 12 70 1 "example car"\n',
               'abalone': b'M,0.4,0.3,0.1,0.5,0.2,0.1,0.15,15\n',
               'airfoil': b'100 1 0.3 30 0.01 120\n', 'yacht': b'1 2 3 4 5 6 7\n'}
    catalog = {name: {**v, 'rows': 168} for name, v in benchmarks.CATALOG.items()}
    monkeypatch.setattr(benchmarks, 'CATALOG', catalog)
    calls = []
    def fetch(url, timeout):
        calls.append(url)
        name, item = next((n, i) for n, i in catalog.items() if f"/{i['uci']}/" in url)
        b = io.BytesIO()
        with zipfile.ZipFile(b, 'w') as z:
            z.writestr(str(item['member']), sources[name] * 168)
        return io.BytesIO(b.getvalue())
    monkeypatch.setattr(benchmarks, 'urlopen', fetch)
    manifest = benchmarks.provision(root, download_public=True)
    assert len(calls) == 4 and len(load_datasets(manifest)) == 5
    receipt = json.loads((root / 'provision.json').read_text())
    assert len(receipt['datasets']) == 5 and len(receipt['files_sha256']) == 10
    assert benchmarks.provision(root) == manifest and len(calls) == 4
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        benchmarks.provision(root)
    monkeypatch.setattr(benchmarks, 'urlopen', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('offline')))
    with pytest.raises(OSError):
        benchmarks.provision(tmp_path / 'failed', download_public=True)
    assert not (tmp_path / 'failed').exists()
    assert not list(tmp_path.glob('.e5-provision-*'))
