"""Train language-specific TF-IDF + LinearSVC; select C on validation only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def load(lang, split):
    rows = [json.loads(line) for line in (ROOT / f'data/prepared/{lang}-{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    return [r['text'] for r in rows], [r['label'] for r in rows]


def main():
    audit = json.loads((ROOT / 'models/data-audit.json').read_text())
    for lang in ('en', 'hi', 'mr'):
        started = time.time()
        train_x, train_y = load(lang, 'train')
        val_x, val_y = load(lang, 'validation')
        test_x, test_y = load(lang, 'test')
        features = FeatureUnion([
            ('words', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=70000, sublinear_tf=True, dtype=np.float32)),
            ('characters', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=70000, sublinear_tf=True, dtype=np.float32)),
        ])
        print(f'{lang}: fitting vocabulary on {len(train_x):,} training headlines', flush=True)
        x_train = features.fit_transform(train_x)
        x_val = features.transform(val_x)
        trials, best = [], None
        for c in (0.5, 2.0):
            classifier = LinearSVC(C=c, random_state=42, dual=True, max_iter=3000)
            with threadpool_limits(limits=2):
                classifier.fit(x_train, train_y)
            pred = classifier.predict(x_val)
            score = float(f1_score(val_y, pred, labels=classifier.classes_, average='macro', zero_division=0))
            trials.append({'C': c, 'validation_macro_f1': score})
            if best is None or score > best[0]:
                best = score, classifier, c
        classifier = best[1]
        # Do not refit on validation. Test remains untouched during selection.
        pred = classifier.predict(features.transform(test_x))
        labels = list(classifier.classes_)
        model_path = ROOT / f'models/classifier-{lang}.joblib'
        joblib.dump({'features': features, 'classifier': classifier}, model_path, compress=3)
        report = {
            'language': lang, 'status': 'trained', 'dataset': 'L3Cube-IndicNews SHC',
            'license': 'CC-BY-NC-SA-4.0; research use', 'source_revision': audit['revision'],
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'algorithm': 'Word (1,2) and character (3,5) TF-IDF + LinearSVC',
            'seed': 42, 'selected_C': best[2], 'selection_trials': trials,
            'train_count': len(train_x), 'validation_count': len(val_x), 'test_count': len(test_x),
            'accuracy': float(accuracy_score(test_y, pred)), 'macro_f1': float(f1_score(test_y, pred, labels=labels, average='macro', zero_division=0)),
            'majority_baseline_accuracy': max(Counter(test_y).values()) / len(test_y),
            'labels': labels, 'confusion_matrix': confusion_matrix(test_y, pred, labels=labels).tolist(),
            'per_class': classification_report(test_y, pred, labels=labels, output_dict=True, zero_division=0),
            'split_audit': audit['languages'][lang],
            'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
            'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'duration_seconds': round(time.time() - started, 2),
            'limitations': ['Historical headline benchmark; not live-news accuracy.', 'Exact normalized duplicates removed; related events may remain across splits.', 'Native labels retained; no universal topic classifier.', 'SVM margins are uncalibrated scores, not probabilities.'],
        }
        (ROOT / f'models/metrics-{lang}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'{lang}: accuracy={report["accuracy"]:.4f}, macro-F1={report["macro_f1"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
