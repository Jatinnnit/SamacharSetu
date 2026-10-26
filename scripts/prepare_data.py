"""Acquire pinned source splits, audit schemas, remove duplicates before fitting."""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {'en': 'English', 'hi': 'Hindi', 'mr': 'Marathi'}


def fingerprint(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(c for c in text if unicodedata.category(c)[0] in 'LMN')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = {x.lower(): x for x in (reader.fieldnames or [])}
        label = fields.get('labels', fields.get('label'))
        if 'text' not in fields or not label:
            raise ValueError(f'Unsupported schema in {path.name}: {reader.fieldnames}')
        return [{'text': unicodedata.normalize('NFC', row[fields['text']]).strip(),
                 'label': row[label].strip().casefold()} for row in reader]


def clean_splits(splits):
    labels = defaultdict(set)
    for rows in splits.values():
        for row in rows:
            labels[fingerprint(row['text'])].add(row['label'])
    conflicts = {key for key, values in labels.items() if len(values) > 1}
    seen, cleaned, audit = set(), {}, {}
    # Preserve official split membership, giving training priority on duplicates.
    for split in ('train', 'validation', 'test'):
        kept, rejected = [], Counter()
        for row in splits[split]:
            key = fingerprint(row['text'])
            if len(key) < 8 or not row['label']:
                rejected['empty_or_short'] += 1
            elif key in conflicts:
                rejected['conflicting_labels'] += 1
            elif key in seen:
                rejected['duplicate'] += 1
            else:
                seen.add(key)
                kept.append(row)
        cleaned[split] = kept
        audit[split] = {'raw': len(splits[split]), 'retained': len(kept),
                        'removed': dict(rejected), 'labels': dict(Counter(r['label'] for r in kept))}
    return cleaned, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    source = json.loads((ROOT / 'data/source-manifest.json').read_text())
    raw, prepared = ROOT / 'data/raw', ROOT / 'data/prepared'
    raw.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    if args.download:
        for item in source['files']:
            target = raw / item['file']
            if not target.exists():
                print('Downloading', target.name, flush=True)
                with urllib.request.urlopen(item['url'], timeout=120) as response:
                    target.write_bytes(response.read())
    report = {'source': source['repository'], 'revision': source['revision'], 'languages': {}}
    for lang, name in LANGUAGES.items():
        splits = {}
        for split, pattern in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]:
            item = next(x for x in source['files'] if x['file'].lower().startswith(name.lower()) and pattern in x['file'].lower())
            path = raw / item['file']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item['sha256']:
                raise ValueError(f'Checksum mismatch: {path.name}')
            splits[split] = read_csv(path)
        cleaned, audit = clean_splits(splits)
        supported = {label for label, count in Counter(r['label'] for r in cleaned['train']).items() if count >= 100}
        for split, rows in cleaned.items():
            excluded = Counter(r['label'] for r in rows if r['label'] not in supported)
            cleaned[split] = [r for r in rows if r['label'] in supported]
            audit[split]['excluded_low_training_support'] = dict(excluded)
            audit[split]['retained'] = len(cleaned[split])
            audit[split]['labels'] = dict(Counter(r['label'] for r in cleaned[split]))
        for split, rows in cleaned.items():
            with (prepared / f'{lang}-{split}.jsonl').open('w', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        report['languages'][lang] = audit
    (ROOT / 'models/data-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
