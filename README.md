# SamacharSetu — समाचारसेतु

**Understand the news. In your language.**

A local Next.js + Python FastAPI project that combines a real news reader with a working NLP Learning Lab. This is a separate project, not JanSahay. No account, paid API, database or API key is required.

## Start here in VS Code (Windows / Python 3.11)

1. Extract the ZIP. Open the **SamacharSetu** folder in VS Code, not the parent folder.
2. Install Node.js 20.9+ (Node 24 was tested) and Python 3.11 if necessary.
3. Open a terminal in the project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd ci
```

No activation is necessary. If VS Code offers to select `.venv` as the Python interpreter, select **Yes**. Internet is needed for this initial dependency installation.

4. Start the backend in terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

5. Open a second terminal in the same folder and start the frontend:

```powershell
npm.cmd run dev
```

6. Open **http://127.0.0.1:3000**. For offline examples, choose **Offline examples** or open **http://127.0.0.1:3000/?demo=1**. Keep both terminals running. Press Ctrl+C in each terminal to stop.

The ZIP includes three trained classifiers, six trained word-vector arrays, their measured evaluations, and English NLTK resources. You do **not** need to train the models to open the app. Raw and prepared training datasets are intentionally excluded.

For macOS/Linux, replace the Windows Python path with `.venv/bin/python`, create the environment with `python3.11 -m venv .venv`, and use `npm` in place of `npm.cmd`.

## What works

- Home and Explore: live GDELT news metadata, keyword search, news language, publisher-country filter, date range, topic query and newest/relevance ordering.
- Reader: original links, correctly labelled GDELT first-seen timestamps, available-text summaries, supported entity extraction, related wording, save/copy/share, and Explore NLP.
- Compare: 2–6 stories, source details, shared terms/entities, distinct terms, and a same-language TF-IDF cosine matrix. This is not a fact-checker or bias detector.
- Saved: persistent browser bookmarks, reading preferences, and preference reset.
- Lab: web scraper with HTML element extraction, language detection, Unicode tokens and counts, English normalization/POS/NER, trained category classification, six Word2Vec models, embedding-based text classification, TF-IDF term weighting & cosine matrix, n-grams with sequence probability and perplexity, LSA topic exploration, topic-based opinion mining with sentiment/emotions, and an extractive Question Answering (QA) pipeline.
- Evaluation: actual training/validation/test counts, held-out accuracy, macro-F1, class reports, confusion matrices and cleaning audit.
- Assignment map: all 9 official practical syllabus assignments (Assignments 1–9) visibly mapped to pages, functions, languages, tests, and standalone runnable scripts in `scripts/`.

## Actual training

The selected source is **L3Cube-IndicNews Short Headlines Classification (SHC)**, from a pinned revision of the publisher's repository. 112,177 raw records were audited, with **67,512 cleaned training headlines**, 8,190 validation headlines and 8,060 held-out test headlines retained. No synthetic expansion, repeated fixtures or HuffPost data were counted as training data.

| Language | Train | Validation | Test | Test accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| English | 23,031 | 2,897 | 2,858 | 89.85% | 0.8943 |
| Hindi | 26,834 | 3,184 | 3,142 | 87.40% | 0.8708 |
| Marathi | 17,647 | 2,109 | 2,060 | 90.49% | 0.9002 |

These are historical dataset benchmarks, not measured accuracy on live news. Duplicate/conflicting source rows were removed before fitting. A rare Marathi label, `manoranjan`, had only 24 clean training examples and no validation/test examples; it is excluded under the rule requiring at least 100 training examples per class. See [model evaluation](docs/model-evaluation.md) and the machine-readable `models/metrics-*.json` files.

The models use word and character TF-IDF features and a linear SVM. `C` is selected using validation macro-F1. SVM scores are **not confidence probabilities**. Word2Vec CBOW and Skip-gram are trained separately on the training splits only, with 48 dimensions, 10 epochs, seed 42 and one worker.

## Language/task coverage

| Task | Actual coverage |
|---|---|
| Main navigation and primary reader controls | English, Hindi, Marathi; detailed lab explanations remain English |
| News discovery / source language filters | English and eleven listed Indian languages, subject to GDELT availability |
| Tokenization / simple sentence splitting | Unicode script-aware rules; not linguistically perfect segmentation |
| Language detection | Pretrained langid.py; short and mixed text marked uncertain |
| News classifiers | English (12 labels), Hindi (11), Marathi (11) |
| Word2Vec | English, Hindi, Marathi; separate CBOW and Skip-gram models |
| Stemming, lemmatization, POS and NER | English only; unsupported languages display an explanation |
| Related-story similarity | Same-language lexical similarity only; cross-language cells unavailable |
| Translation and simpler paraphrases | Not configured; original text retained |

## Retrain or reproduce

Review [dataset sources and licenses](docs/dataset-sources.md) first. The data publisher licenses resources under **CC BY-NC-SA 4.0** and specifies research use. This is a research/college project, not a commercially licensed news service. Dataset/model terms are separate from software library licenses.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --download
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\python.exe scripts\train_embeddings.py
.\.venv\Scripts\python.exe scripts\report_evaluation.py
```

The downloader uses pinned source URLs and verifies SHA-256 hashes. It downloads only nine manageable CSVs (about 22 MB), not a multi-gigabyte corpus. The training process preserves the publisher's split membership and removes normalized duplicates across splits. It does not train at application startup. Restart the backend after replacing model artifacts so its bounded model caches refresh.

To reinstall missing English linguistic resources:

```powershell
.\.venv\Scripts\python.exe scripts\setup_nltk.py
```

## Build and test

```powershell
npm.cmd run typecheck
npm.cmd run build
npm.cmd test
.\.venv\Scripts\python.exe -m pytest -q
```

For a production-style local frontend, use `npm.cmd start` after the build instead of `npm.cmd run dev`. Keep the backend running.

Browser integration tests require both servers on ports 3000 and 8000:

```powershell
npx.cmd playwright install chromium
npm.cmd run test:e2e
```

One dataset leakage test is skipped in the downloaded ZIP until `prepare_data.py --download` has recreated the excluded training data. All other tests use bundled models, fixtures or mocked provider responses. See [verification](docs/verification.md) for the recorded results.

## Project structure

- `app/` — Next.js route and editorial responsive stylesheet.
- `components/App.tsx` — reader, search, comparison, bookmarks and preferences.
- `components/Lab.tsx` — working learning modules and model evaluation.
- `lib/` — typed API helpers and browser persistence functions.
- `backend/main.py` — validated FastAPI endpoints.
- `backend/news.py` — GDELT adapter, bounded cache, duplicate removal and offline fixtures.
- `backend/nlp.py` — local NLP inference and academic algorithms.
- `scripts/` — acquisition, preparation, training, resource setup and report generation.
- `models/` — trained inference artifacts, provenance, metrics and NLTK resources.
- `data/source-manifest.json` — exact revision, download URLs and checksums.
- `data/demo/` — authored fictional teaching fixtures, never shown as live news.
- `docs/` — corrected specification, architecture, source audit, assignment mapping and evaluation.
- `tests/` — backend tests, frontend state tests and browser flows.

## Known limits and troubleshooting

- GDELT may be rate-limited or unavailable. The app shows an error and offers an explicit offline mode. Requests have a timeout, five-second spacing and a five-minute in-memory cache. At most 50 provider results are requested.
- GDELT's `seendate` is first observation, **not publication time**. News is metadata-only; no articles are scraped or redistributed.
- Most live records contain only a headline. Summaries are available only when enough genuine source text is present (or for clearly fictional learning fixtures). No made-up facts, translations or publication dates are added.
- No model has been independently evaluated on recent articles, cross-language event matching, or all Indian languages. Topics in the reader are provider keyword queries, not the classifier's native labels.
- A direct reader URL needs the story in this browser's session or saved collection. Share/copy uses the publisher URL for live stories. Offline fixture URLs can reload independently.
- Pasted text is processed in memory on the local backend; it is not saved or sent to GDELT. User-initiated news search terms are sent to GDELT. Bookmarks/preferences remain in browser storage.
- If PowerShell blocks `npm.ps1`, use `npm.cmd` as shown. No execution-policy change is required.
- If port 3000 or 8000 is occupied, stop the older project process or change ports consistently. `NLP_API_URL` in `.env.local` can point Next.js to another local backend; never expose secrets with `NEXT_PUBLIC_`.
- Do not load untrusted `.joblib` files. Only the bundled, locally trained artifacts are intended for this app.
- Browser storage can be cleared by private mode or browser settings. Storage failures display an explanation. No authentication, cloud sync, background task or automatic article harvesting is included.
