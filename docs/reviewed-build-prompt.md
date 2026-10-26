# Reviewed build specification — SamacharSetu

This is the corrected implementation specification derived from the supplied 488-line master prompt. The product, name and assignment requirements are preserved. This project is independent of JanSahay.

## Corrections made before implementation

| Gap or ambiguity in the original prompt | Resolved requirement |
|---|---|
| “Good amount of data” had no acceptance criterion | Target tens of thousands of unique, relevant training headlines per primary language where available; report raw, removed, training, validation and test counts separately. Never inflate volume by repeating fixtures. |
| Listed datasets do not necessarily cover the same tasks or languages | Audit schemas, labels, source ownership and licenses. Prefer the task-matched L3Cube SHC dataset after reviewing AI4Bharat alternatives. |
| Topic menus could be mistaken for universal classifier labels | Keep provider keyword topics and model-native labels separate. Display each language's supported labels. |
| Random splitting alone would not prevent duplicated-headline leakage | Preserve source splits, normalize fingerprints, remove cross-split duplicates and conflicting labels before fitting. Disclose remaining event/near-duplicate leakage risk. |
| A rare class can produce unsupported evaluation claims | Require at least 100 cleaned training examples per class. Report excluded labels. Require nonzero held-out support for advertised evaluated classes. |
| “Confidence” can incorrectly describe SVM margins | Display a raw margin and a clearly labelled uncertainty heuristic; do not convert the score into a fake percentage. |
| GDELT often returns no article body, summary or verified publication date | Retain the original headline and link; label `seendate` as first seen. Summarize only sufficiently long available text, otherwise explain the absence. |
| Multilingual interface and multilingual NLP were conflated | Support primary navigation in English/Hindi/Marathi, with a task-level capability matrix. Detailed lab teaching copy remains English in this MVP. |
| Similarity can suggest more than lexical evidence supports | Use a documented same-language TF-IDF baseline. Cross-language matching is unavailable unless a genuine multilingual model is added. Never infer bias or truth. |
| Demo data could be confused with training or live news | Keep three separate locations and modes. Authored fixture cards are visibly fictional and undated. |
| Translation and simpler explanations could imply an unavailable generative model | Preserve source wording. Translation and paraphrasing show an honest unavailable state. |
| Reproducibility needed more detail | Pin data revision and checksums, record software versions/seed/configuration, select hyperparameters on validation only and save test predictions as aggregate metrics. |
| Desktop appearance alone was insufficient | Check the actual rendered browser at desktop and mobile widths, test overflow and critical actions. |
| Next.js/Tailwind/shadcn were suggested as defaults, not a reason for unnecessary infrastructure | Use Next.js App Router/TypeScript, Lucide icons and accessible native components with a custom responsive stylesheet; FastAPI for Python NLP. No Tailwind or component framework is needed for this small app. |

## Final build instructions

Build a local, account-free project called SamacharSetu with the tagline “Understand the news. In your language.” Use an editorial paper-and-ink visual system, deep-teal accents, readable text, visible focus states, meaningful empty/error states and responsive layouts.

Implement these routes: `/`, `/explore`, `/story/{id}`, `/compare`, `/saved`, `/lab`. The story reader opens actual selected metadata, retains publisher URLs and known timestamps, and offers bookmarking, original-link copying/sharing, related wording and a transfer of the exact available text into the Lab. Store bookmarks/preferences locally and remember search filters on navigation.

Implement server-side GDELT search with validated language/country/topic/sort/date inputs, bounded results, timeouts, request spacing, a bounded TTL cache and duplicate removal. Use source-country as publisher country, not event location. Date range requests must fall within the supported recent-news window. Query failures must not silently substitute fictional content. Do not scrape article bodies or fetch arbitrary user-supplied URLs.

Train independent English, Hindi and Marathi news classifiers on genuine SHC data. Fit TF-IDF only on training text, choose linear-SVM C on validation macro-F1, evaluate the selected model once on the cleaned test set and save its artifact plus metadata. Report baseline accuracy, accuracy, macro-F1, class support, precision/recall/F1 and confusion matrix. Freeze the source revision and verify checksums. Retain training data only locally, outside the downloadable source archive and version control.

Train genuine CBOW and Skip-gram word embeddings separately on those training splits. Use a fixed seed, fixed Python hash seed and one worker. Export learned vectors and their vocabulary to portable arrays. Show nearest neighbours, complete vector values and two-word cosine similarity, with explicit out-of-vocabulary handling. Do not claim sentence-level or cross-language semantic equivalence from these word vectors.

The Lab must provide:

1. Language detection with a real pretrained model and uncertain short/mixed-text handling.
2. Unicode mark-aware word/punctuation tokens, original character spans, deterministic rule-based sentence segments and reproducible counts.
3. English Snowball stems and POS-aware WordNet lemmas, with unsupported-language notices.
4. English contextual POS tags and readable descriptions.
5. English model-predicted entity spans, highlighted exactly in the input text.
6. Trained word vectors, CBOW/Skip-gram comparison and cosine similarity.
7. Trained news classification with native labels and feature-contribution inspection.
8. Interactive unigram/bigram/trigram counts, no smoothing or additive smoothing, token probabilities and held-out-text perplexity including the infinite unsmoothed case.
9. TF-IDF/SVD-based LSA on a user-supplied document collection, exposing component terms and document vectors.
10. A measured evaluation view and visible assignment map linked to actual backend functions and tests.

All pasted text stays in the local inference process and is not persisted. Translation is a capability check until a provider/model is genuinely configured. Related-story matching is a lexical discovery aid, not proof of shared events, factuality or bias. Supported language/task claims must be generated from the actual available artifacts and resources where possible.

## Acceptance checklist

- [x] Next.js frontend and FastAPI backend run locally.
- [x] Live provider tested with actual metadata; offline fixtures visibly labelled.
- [x] Search/language/topic filters, reader, comparison and bookmarks work.
- [x] Real training completed and artifacts included.
- [x] Dataset audit, source license, split hygiene and limitations documented.
- [x] All twelve required assignment mappings point to working functions and tests.
- [x] TypeScript and production build checked.
- [x] Backend/frontend/browser tests and visual desktop/mobile checks performed; see verification.md for recorded results.
- [x] Setup, retraining and missing-resource recovery commands included.

This specification does not claim that translation, Indic POS/NER, neural language models, LDA or multilingual sentence embeddings were implemented. They were optional or capability-dependent extensions, not fabricated MVP features.
