"""Export already-trained Gensim word vectors to safe, compact inference arrays."""
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec

root = Path(__file__).resolve().parents[1]
for path in (root / 'models').glob('word2vec-*.model'):
    model = Word2Vec.load(str(path))
    np.savez_compressed(path.with_suffix('.npz'), words=np.asarray(model.wv.index_to_key), vectors=model.wv.vectors)
    print(path.stem, len(model.wv))
