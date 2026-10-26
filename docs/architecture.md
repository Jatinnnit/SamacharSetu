# Architecture

SamacharSetu is a local two-process app: Next.js serves the interface on 127.0.0.1:3000; FastAPI serves NLP/news endpoints on 127.0.0.1:8000. Next.js rewrites `/api/*` to the Python service, so browser requests remain same-origin. Both servers bind to loopback by default.

## Responsibilities

| Layer | Files | Responsibility |
|---|---|---|
| Presentation | `app/`, `components/` | Responsive reader, forms, state, lab visualizations and model audit |
| Browser persistence | `lib/storage.mjs` | Saved stories/preferences in localStorage; recent articles, comparison and filters in sessionStorage |
| HTTP boundary | `backend/main.py` | Pydantic limits, enum checks, date validation and API contracts |
| Provider | `backend/news.py` | Fixed GDELT endpoint, metadata normalization, URL validation, duplicate filtering, bounded TTL cache |
| Inference | `backend/nlp.py` | Language-aware models, n-grams, LSA, summaries and comparison |
| Acquisition/training | `scripts/` | Pinned downloads, checksums, split hygiene, model fitting and evaluation |
| Artifacts | `models/` | Three trusted joblib classifiers, six safe NumPy word-vector files, metrics, resource files |

## API

| Method / path | Input | Output |
|---|---|---|
| GET `/api/health` | none | service identity/status |
| GET `/api/news` | q, language, country, topic, sort, start, end, demo | mode, articles, notice, cache status |
| GET `/api/models` | none | per-language measured metrics and capability coverage |
| POST `/api/analyze` | text, language or auto | detection, tokens/counts, linguistic results, category and extractive summary |
| POST `/api/embeddings` | language, mode, word, other | learned vector, neighbours, cosine and metadata |
| POST `/api/compare` | 2–6 limited article metadata objects | same-language similarity matrix, shared/distinct terms and entities |
| POST `/api/ngram` | training, evaluation, n, alpha | probabilities, vocabulary, OOV count, perplexity/infinite flag |
| POST `/api/lsa` | 3–30 bounded documents | component terms, variance and document coordinates |
| POST `/api/translate` | none | unavailable status; original text retained |

## Data and privacy boundaries

Only user-initiated news search requests leave the machine via GDELT. Article URLs are links opened by the reader, never backend scrape targets. Pasted NLP text is processed in memory and is neither logged as a request body nor retained. No authentication, accounts, remote analytics, cookies for profiling or database are used. Resource/model downloads occur in explicit setup/training scripts only.

News cache: maximum 64 query keys, default five-minute TTL, five-second spacing between provider requests, 15-second network timeout, 50 requested records. Cache is process-local and lost on restart. Failed responses are not presented as successful cached news. Classifier/embedding caches are bounded by the supported language/model count. Inference results containing private pasted text are not cached.

Live article models have safe HTTP(S) source URLs and hashed URL IDs. `timestamp_kind=first_seen` explicitly labels GDELT's observation time. Offline examples have no publisher URL or invented publication date. Comparisons return null for unsupported cross-language pairs.

## Implementation tradeoffs

The default architecture in the prompt was retained (Next.js/TypeScript + FastAPI). Native controls and a custom CSS system replace Tailwind/shadcn to reduce dependencies. The typography uses system fonts so no font download is needed. Lucide supplies a consistent icon family. No database, Docker or paid service is introduced.

TF-IDF + SVM is an explainable CPU baseline, not a transformer. English language tools use pretrained NLTK resources; Indic tools do not borrow English models. Same-language lexical matching provides a small, inspectable baseline in place of large multilingual sentence-transformer downloads. Bookmarks work on one browser, without sync. A full article reader is only possible when actual text is available; the application does not pretend headline-only metadata is a full article.

## Safety and deployment scope

The input schemas cap text and document sizes. No arbitrary local paths, pickle uploads or outbound URL-fetch endpoints are accepted. All untrusted article/pasted text renders through React escaping. External links are scheme-checked and open with `noopener noreferrer`. The joblib files are executable serialization; only trusted bundled models should be loaded.

This is a local college/research MVP, not a hardened multi-user service. Public hosting would require authentication/rate limiting appropriate to the deployment, data-license review and an operational security review. It has not been published externally.
