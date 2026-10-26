"""Create a portable source + trained inference artifact archive; exclude data and environments."""
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'outputs' / 'SamacharSetu-project.zip'
OUT.parent.mkdir(parents=True, exist_ok=True)
excluded_dirs = {'.venv', 'node_modules', '.next', '.git', '__pycache__', '.pytest_cache', 'test-results', 'playwright-report'}
included = []
for parent, dirs, files in os.walk(ROOT):
    relative = Path(parent).relative_to(ROOT)
    dirs[:] = [d for d in dirs if d not in excluded_dirs and not (relative == Path('data') and d in ('raw', 'prepared'))]
    for name in files:
        path = Path(parent) / name
        if name.endswith(('.log', '.pyc', '.tsbuildinfo', '.model')) or (name.startswith('.env') and name != '.env.example'):
            continue
        # NLTK ships both extracted resources and their source zip; include one copy.
        if name.endswith('.zip') and path.with_suffix('').is_dir():
            continue
        included.append(path)

with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(included):
        archive.write(path, 'SamacharSetu/' + path.relative_to(ROOT).as_posix())
    manifest = {str(p.relative_to(ROOT).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest() for p in included}
    archive.writestr('SamacharSetu/FILE_CHECKSUMS.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(OUT) as archive:
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f'Archive CRC check failed: {bad}')
print(json.dumps({'path': str(OUT), 'files': len(included)+1, 'bytes': OUT.stat().st_size, 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2))
