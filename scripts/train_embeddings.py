"""Train genuine CBOW and Skip-gram models on training splits only."""
import json
import os
import subprocess
import sys
from pathlib import Path
from gensim.models import Word2Vec
import regex
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def tokens(text):
    return regex.findall(r'[\p{L}\p{M}]+', text.casefold())


def main():
    for lang in ('en', 'hi', 'mr'):
        rows = [json.loads(x)['text'] for x in (ROOT / f'data/prepared/{lang}-train.jsonl').read_text(encoding='utf-8').splitlines()]
        sentences = [tokens(x) for x in rows]
        for mode, sg in [('cbow', 0), ('skipgram', 1)]:
            print(f'Training {lang} {mode}', flush=True)
            model = Word2Vec(sentences, vector_size=48, window=4, min_count=5, workers=1, sg=sg, seed=42, epochs=10, max_final_vocab=15000)
            model.save(str(ROOT / f'models/word2vec-{lang}-{mode}.model'))
            np.savez_compressed(ROOT / f'models/word2vec-{lang}-{mode}.npz', words=np.asarray(model.wv.index_to_key), vectors=model.wv.vectors)
            meta = {'algorithm': mode, 'language': lang, 'dimensions': 48, 'vocabulary': len(model.wv), 'training_headlines': len(rows), 'epochs': 10, 'seed': 42, 'python_hash_seed': 42, 'workers': 1, 'dataset': 'L3Cube-IndicNews SHC training split', 'evaluation': 'Teaching demonstration; nearest-neighbour quality not independently benchmarked.'}
            (ROOT / f'models/word2vec-{lang}-{mode}.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != '42':
        subprocess.run([sys.executable, __file__], env={**os.environ, 'PYTHONHASHSEED': '42'}, check=True)
    else:
        main()
