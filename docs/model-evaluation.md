# Model evaluation

Generated from the actual saved metrics files. No values are manually invented.

| Language | Training | Validation | Test | Accuracy | Macro-F1 | Majority baseline | C |
|---|---:|---:|---:|---:|---:|---:|---:|
| en | 23,031 | 2,897 | 2,858 | 89.8530% | 0.8943 | 10.6368% | 0.5 |
| hi | 26,834 | 3,184 | 3,142 | 87.3966% | 0.8708 | 11.1394% | 0.5 |
| mr | 17,647 | 2,109 | 2,060 | 90.4854% | 0.9002 | 11.7961% | 0.5 |

Total training examples: **67,512**. Validation and test examples are not counted as training.

## Method

Language-specific TF-IDF word 1–2 grams and character 3–5 grams (up to 70,000 features each), sublinear term frequency, LinearSVC. C ∈ {0.5, 2.0} selected by validation macro-F1. Seed 42, training only; no validation refit. Test labels are not used for hyperparameter selection.

Class support threshold is 100 clean training records. Source split membership is retained. A full-pool duplicate/conflicting-label integrity audit happens before any feature fitting. This creates cleaned-subset evaluation, not a directly comparable reproduction of the publisher’s original benchmark.

## Limitations

- These metrics measure historical SHC classification, not factuality, trustworthiness, bias or live-news accuracy.
- Exact normalized duplicates are removed, but near-duplicate events and publisher/label cues can remain. No time-based or outlet-based split was available or claimed.
- Native category coverage differs by language. Unavailable categories are not silently mapped.
- SVM margins are uncalibrated. The margin-gap uncertainty threshold (0.35) is a display heuristic, not calibrated on the test set.
- Word2Vec models are genuine training artifacts, but nearest-neighbour quality is not independently benchmarked.
- The small interactive n-gram perplexity and LSA modules are teaching tools; their results are not classifier metrics.

## Per-language reports

### en

Trained: 2026-10-25T12:12:09.794250+00:00. Source revision: `0c6014be94385b357e31ce6250da7094e8a7c0fb`.

Artifact SHA-256: `1946e9e7f446d1992ec7ca8ff1b8140fdf6786432332ec2647cc588d2c25bc96`.

Software: {'python': '3.11.9', 'sklearn': '1.9.0', 'numpy': '2.4.6'}.

Validation selection:

- C=0.5: macro-F1=0.890435
- C=2.0: macro-F1=0.886169

| Label | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| auto | 0.9632 | 0.9225 | 0.9424 | 142 |
| business | 0.9185 | 0.9185 | 0.9185 | 270 |
| education | 0.9412 | 0.9632 | 0.9521 | 299 |
| elections | 0.9508 | 0.9431 | 0.9469 | 246 |
| entertainment | 0.9028 | 0.9353 | 0.9187 | 278 |
| health | 0.9008 | 0.8906 | 0.8956 | 265 |
| india | 0.7897 | 0.8075 | 0.7985 | 265 |
| lifestyle | 0.8805 | 0.8089 | 0.8432 | 246 |
| science | 0.8302 | 0.8148 | 0.8224 | 108 |
| sports | 0.9578 | 0.9704 | 0.9641 | 304 |
| technology | 0.8920 | 0.8771 | 0.8845 | 179 |
| world | 0.8302 | 0.8594 | 0.8445 | 256 |

Confusion matrix: rows = actual, columns = predicted, in this order: auto, business, education, elections, entertainment, health, india, lifestyle, science, sports, technology, world.

```text
131 3 1 0 1 1 0 2 1 1 0 1
0 248 2 0 0 2 6 3 0 0 2 7
0 0 288 0 1 2 2 1 1 0 1 3
1 0 0 232 1 0 12 0 0 0 0 0
0 0 1 0 260 0 4 2 0 2 3 6
0 0 5 0 0 236 4 9 4 0 2 5
2 6 2 9 2 5 214 3 1 2 2 17
0 1 1 1 18 9 4 199 5 6 2 0
0 1 0 0 1 5 2 4 88 0 3 4
0 0 1 0 2 1 3 0 1 295 0 1
2 7 3 1 1 0 4 1 2 0 157 1
0 4 2 1 1 1 16 2 3 2 4 220
```

Raw split and removal details are in the corresponding metrics JSON and `models/data-audit.json`.

### hi

Trained: 2026-10-25T12:12:29.511181+00:00. Source revision: `0c6014be94385b357e31ce6250da7094e8a7c0fb`.

Artifact SHA-256: `8f23df4127e7d09585ecadbf6041217ed0a3704a960645e695ff01ee9b304f2c`.

Software: {'python': '3.11.9', 'sklearn': '1.9.0', 'numpy': '2.4.6'}.

Validation selection:

- C=0.5: macro-F1=0.866632
- C=2.0: macro-F1=0.861020

| Label | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| automobile | 0.8519 | 0.8000 | 0.8251 | 115 |
| business | 0.8503 | 0.8632 | 0.8567 | 329 |
| crime-news-hindi | 0.7988 | 0.8792 | 0.8371 | 298 |
| education | 0.9850 | 0.9647 | 0.9747 | 340 |
| entertainment | 0.7785 | 0.8923 | 0.8315 | 130 |
| health-news-hindi | 0.9612 | 0.9738 | 0.9675 | 229 |
| international | 0.8276 | 0.7564 | 0.7904 | 349 |
| khel | 0.9471 | 0.9612 | 0.9541 | 335 |
| national | 0.7508 | 0.7418 | 0.7463 | 337 |
| politics | 0.8445 | 0.8394 | 0.8419 | 330 |
| technology-news | 0.9704 | 0.9371 | 0.9535 | 350 |

Confusion matrix: rows = actual, columns = predicted, in this order: automobile, business, crime-news-hindi, education, entertainment, health-news-hindi, international, khel, national, politics, technology-news.

```text
92 20 0 0 1 1 0 0 0 0 1
15 284 3 1 2 3 7 0 6 2 6
0 2 262 1 4 0 7 1 17 3 1
1 1 0 328 0 0 2 0 5 3 0
0 0 2 0 116 0 4 1 1 6 0
0 1 0 0 1 223 1 0 1 2 0
0 4 23 1 12 3 264 13 17 11 1
0 0 1 0 3 0 5 322 3 1 0
0 5 32 1 5 0 20 1 250 22 1
0 4 5 1 5 0 7 0 31 277 0
0 13 0 0 0 2 2 2 2 1 328
```

Raw split and removal details are in the corresponding metrics JSON and `models/data-audit.json`.

### mr

Trained: 2026-10-25T12:12:41.650692+00:00. Source revision: `0c6014be94385b357e31ce6250da7094e8a7c0fb`.

Artifact SHA-256: `e5b292de77fae933e6cd016f636d413761352cd9153a9da4438166b10c2ef1a8`.

Software: {'python': '3.11.9', 'sklearn': '1.9.0', 'numpy': '2.4.6'}.

Validation selection:

- C=0.5: macro-F1=0.901792
- C=2.0: macro-F1=0.901419

| Label | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| auto | 0.9303 | 0.9397 | 0.9350 | 199 |
| bhakti | 0.9368 | 0.9760 | 0.9560 | 167 |
| crime | 0.8059 | 0.9139 | 0.8565 | 209 |
| education | 0.9733 | 0.8488 | 0.9068 | 86 |
| fashion | 0.9091 | 0.8370 | 0.8716 | 227 |
| health | 0.9196 | 0.9537 | 0.9364 | 216 |
| international | 0.8246 | 0.7945 | 0.8093 | 219 |
| politics | 0.9247 | 0.9444 | 0.9345 | 234 |
| sports | 0.9721 | 0.9721 | 0.9721 | 179 |
| tech | 0.9336 | 0.9259 | 0.9298 | 243 |
| travel | 0.8571 | 0.7407 | 0.7947 | 81 |

Confusion matrix: rows = actual, columns = predicted, in this order: auto, bhakti, crime, education, fashion, health, international, politics, sports, tech, travel.

```text
187 0 4 0 0 2 1 1 0 2 2
0 163 0 0 1 2 0 1 0 0 0
2 0 191 1 1 2 5 5 0 1 1
1 2 3 73 1 0 5 1 0 0 0
3 3 7 0 190 2 11 2 1 4 4
1 2 1 0 0 206 1 2 0 2 1
3 2 25 0 2 5 174 2 3 2 1
0 2 5 0 0 0 5 221 0 1 0
0 0 0 0 2 0 3 0 174 0 0
3 0 1 0 6 3 3 0 1 225 1
1 0 0 1 6 2 3 4 0 4 60
```

Raw split and removal details are in the corresponding metrics JSON and `models/data-audit.json`.
