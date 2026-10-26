# Dataset and source audit

Audited 25 October 2026. Training data, live metadata and offline fixtures are separate.

## Selected: L3Cube-IndicNews SHC

- Publisher: L3Cube Pune. [Official repository](https://github.com/l3cube-pune/indic-nlp), [SHC dataset description](https://github.com/l3cube-pune/indic-nlp/tree/main/L3Cube-IndicNews), [research paper](https://arxiv.org/abs/2401.02254).
- Authors: Aishwarya Mirashi, Srushti Sonavane, Purva Lingayat, Tejas Padhiyar and Raviraj Joshi. *L3Cube-IndicNews: News-based Short Text and Long Document Classification Datasets in Indic Languages* (2024 arXiv version).
- Exact repository revision: `0c6014be94385b357e31ce6250da7094e8a7c0fb`.
- Terms: publisher's repository states **Creative Commons Attribution-NonCommercial-ShareAlike 4.0**, research purposes only. [License text](https://creativecommons.org/licenses/by-nc-sa/4.0/). This research project preserves attribution and the noncommercial/share-alike treatment of dataset-derived artifacts. This does not grant rights to republish publisher articles commercially.
- Source acquisition: nine official SHC CSVs, using URLs and checksums in `data/source-manifest.json`. No scraping, login or manual acceptance needed for these public files. Raw CSVs and cleaned text records are excluded from Git and the project ZIP.
- Why selected: the inputs are **headlines**, matching GDELT's typical available text. It has substantially more Hindi/Marathi headline training data and richer class coverage than the small alternative subsets reviewed below.

### Actual schemas and splits

| Language | CSV columns | Raw train | Raw validation | Raw test |
|---|---|---:|---:|---:|
| English | `text`, `labels` | 36,877 | 4,610 | 4,610 |
| Hindi | `text`, `labels` | 30,851 | 3,835 | 3,836 |
| Marathi | `Text`, `Label` | 22,025 | 2,761 | 2,772 |

Total: **112,177 raw records**. The loader resolves supported column names case-insensitively and rejects an unexpected schema. It normalizes Unicode to NFC, trims text and lowercases labels. It does not merge different semantic labels across languages.

Native supported labels after cleaning:

- English: auto, business, education, elections, entertainment, health, india, lifestyle, science, sports, technology, world.
- Hindi: automobile, business, crime-news-hindi, education, entertainment, health-news-hindi, international, khel, national, politics, technology-news.
- Marathi: auto, bhakti, crime, education, fashion, health, international, politics, sports, tech, travel.

`manoranjan` in Marathi is excluded: 24 cleaned training records, no evaluation support. This is not mapped to a different category. The app's “Culture”/“Economy” topic buttons are provider-search terms, not aliases for these training labels. No classifier is advertised for Assamese, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil or Telugu in this build.

### Cleaning and evaluation hygiene

1. Preserve the supplied train/validation/test membership.
2. Compute NFKC/casefolded fingerprints retaining Unicode letters, combining marks and numbers. Ignore punctuation/spacing for duplicate detection.
3. Remove all entries whose fingerprint has conflicting labels, using the full source pool only for the integrity audit—not feature fitting or model selection.
4. Remove short/empty records and repeated fingerprints, with priority train → validation → test.
5. Exclude any class with fewer than 100 cleaned training records. Record the exact exclusion per split.
6. Fit word/character TF-IDF and SVM on training only. Select C on validation. Evaluate the selected model on test once.

The full-data integrity audit means these figures are **cleaned-subset benchmarks**, not directly comparable to the paper's official unfiltered results. Duplicate removal alone does not remove all near-duplicate headlines, related events, publisher cues or label hints embedded in headlines. There is no time-based or outlet-based generalization evaluation. Class balance changes after cleaning; macro-F1 and class support are exposed.

## Other requested sources reviewed

| Source | What was verified | Decision |
|---|---|---|
| [AI4Bharat IndicNLP corpus](https://github.com/AI4Bharat/indicnlp_corpus) | News-category datasets span nine languages; Marathi has entertainment/lifestyle/sports with roughly 1.5K articles per class; Hindi is not in this article-classification table. Corpus license stated CC BY-NC-SA 4.0. Public external datasets retain their own original licenses. | Valid alternative, but article-level input and narrower Marathi labels fit this headline task less well. Not downloaded or blended into training. |
| [AI4Bharat IndicGLUE](https://huggingface.co/datasets/ai4bharat/indic_glue) | Dataset metadata inspected at revision `6dcbebd49d6b12965d433bffcd07e9a786211703`. `bbca.hi` has `text`/`label`, 3,467 train and 866 test records; it is article classification. The collection card says license `other`, and individual tasks/subsets vary. | A benchmark collection, not one uniform news dataset. No blanket license or universal label coverage assumed. |
| [MIND](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news) | Research-only Microsoft license, English news recommendation, `news.tsv` contains titles/abstracts/categories/entities; complete article bodies are not provided for download because of licensing. | Not needed; L3Cube's English headlines keep the task consistent and avoid recommendation logs. |
| [IndicCorp](https://indicnlp.ai4bharat.org/corpora/) | Large sentence-level corpus; license varies by release (the original IndicCorp page states CC BY-NC 4.0). No news labels implied. | Not needed for this lightweight model. No multi-gigabyte corpus downloaded. |

No HuffPost dataset was used. No other corpus was silently merged into the source labels.

## Live provider: GDELT DOC 2.0

[Official DOC API documentation](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) defines ArtList mode and language/source-country/query/time/sort filters. The implementation requests at most 50 records, caches for five minutes and spaces outbound calls. The API is unauthenticated but can be rate-limited or unavailable.

Only metadata is presented: original title, source domain, source-reported language, safe publisher URL and first-seen timestamp. Missing description/body stays empty. Images are deliberately omitted in this MVP because their retrieval/privacy/attribution quality is not independently validated. Source links retain the reader's route to original reporting. This source is not model-training data.

## Offline fixtures

`data/demo/stories.json` contains nine original, explicitly fictional learning scenarios in English, Hindi and Marathi. They have no publication dates, real-publisher attribution or invented source URLs. They are used to test reader flows and NLP inputs only, and are not counted in training or evaluation.

## Pretrained linguistic resources

Language identification uses langid.py (BSD-style project license). English tools use NLTK (Apache 2.0 library) with its English averaged-perceptron POS model, maximum-entropy entity chunker, words corpus and WordNet resource. Resource-specific notices accompany their distributions; WordNet has its own Princeton license. Gensim (LGPL library) trains word vectors; NumPy performs inference from the exported arrays. These library licenses do not override the dataset's noncommercial/share-alike conditions.
