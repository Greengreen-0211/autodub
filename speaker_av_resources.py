"""Verify the installed pinned source/assets; never download or install."""
from pathlib import Path
import os
from speaker_av_support import AV_COMMIT, read_json, sha


def verify_resources(options):
    source, models = Path(options['source']).resolve(), Path(options['models']).resolve()
    code = read_json(source / 'AUTODUB_SOURCE_MANIFEST.json')
    assets = read_json(models / 'VERIFIED_MANIFEST.json')
    if code.get('commit') != AV_COMMIT or assets.get('source', {}).get('commit') != AV_COMMIT:
        raise RuntimeError('3D-Speaker source/model manifest version differs from the tested commit')
    if len(code.get('files', [])) != 84 or len(assets.get('assets', [])) != 14:
        raise RuntimeError('3D-Speaker source/model inventory is incomplete')
    for row in code['files']:
        path = source / row['path']
        if not path.resolve().is_relative_to(source):
            raise RuntimeError('Upstream source path escapes the installed repository')
        actual = __import__('hashlib').sha256(os.readlink(path).encode()).hexdigest() if row.get('mode') == '120000' else sha(path)
        if actual != row['sha256']:
            raise RuntimeError('Upstream source checksum differs: ' + str(path))
    for row in assets['assets']:
        path = Path(row['path'])
        if not path.is_relative_to(models) or not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError('Existing model checksum/path differs: ' + str(path))
    return assets['assets']
