import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reports = [json.loads(p.read_text()) for p in sorted((ROOT / 'models').glob('metrics-*.json'))]
lines = ['# Model evaluation', '', 'Generated from the actual saved metrics files. No values are manually invented.', '', '| Language | Training | Validation | Test | Accuracy | Macro-F1 | Majority baseline | C |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in reports:
    lines.append(f'| {r["language"]} | {r["train_count"]:,} | {r["validation_count"]:,} | {r["test_count"]:,} | {r["accuracy"]:.4%} | {r["macro_f1"]:.4f} | {r["majority_baseline_accuracy"]:.4%} | {r["selected_C"]} |')
lines += ['', f'Total training examples: **{sum(r["train_count"] for r in reports):,}**. Validation and test examples are not counted as training.', '', '## Method', '', 'Language-specific TF-IDF word 1–2 grams and character 3–5 grams (up to 70,000 features each), sublinear term frequency, LinearSVC. C ∈ {0.5, 2.0} selected by validation macro-F1. Seed 42, training only; no validation refit. Test labels are not used for hyperparameter selection.', '', 'Class support threshold is 100 clean training records. Source split membership is retained. A full-pool duplicate/conflicting-label integrity audit happens before any feature fitting. This creates cleaned-subset evaluation, not a directly comparable reproduction of the publisher’s original benchmark.', '', '## Limitations', '', '- These metrics measure historical SHC classification, not factuality, trustworthiness, bias or live-news accuracy.', '- Exact normalized duplicates are removed, but near-duplicate events and publisher/label cues can remain. No time-based or outlet-based split was available or claimed.', '- Native category coverage differs by language. Unavailable categories are not silently mapped.', '- SVM margins are uncalibrated. The margin-gap uncertainty threshold (0.35) is a display heuristic, not calibrated on the test set.', '- Word2Vec models are genuine training artifacts, but nearest-neighbour quality is not independently benchmarked.', '- The small interactive n-gram perplexity and LSA modules are teaching tools; their results are not classifier metrics.', '', '## Per-language reports']
for r in reports:
    lines += ['', f'### {r["language"]}', '', f'Trained: {r["trained_at"]}. Source revision: `{r["source_revision"]}`.', '', f'Artifact SHA-256: `{r["model_sha256"]}`.', '', f'Software: {r["versions"]}.', '', 'Validation selection:', '']
    for trial in r['selection_trials']:
        lines.append(f'- C={trial["C"]}: macro-F1={trial["validation_macro_f1"]:.6f}')
    lines += ['', '| Label | Precision | Recall | F1 | Test support |','|---|---:|---:|---:|---:|']
    for label in r['labels']:
        c=r['per_class'][label]
        lines.append(f'| {label} | {c["precision"]:.4f} | {c["recall"]:.4f} | {c["f1-score"]:.4f} | {int(c["support"])} |')
    lines += ['', 'Confusion matrix: rows = actual, columns = predicted, in this order: '+', '.join(r['labels'])+'.', '', '```text']
    lines.extend(' '.join(map(str,row)) for row in r['confusion_matrix'])
    lines += ['```', '', 'Raw split and removal details are in the corresponding metrics JSON and `models/data-audit.json`.']
(ROOT/'docs/model-evaluation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote docs/model-evaluation.md')
