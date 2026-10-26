# Third-party notices

## Data-derived artifacts

The news classifiers and word-vector arrays were trained on **L3Cube-IndicNews SHC** (English, Hindi, Marathi), sourced from L3Cube Pune's repository at revision `0c6014be94385b357e31ce6250da7094e8a7c0fb`.

Authors: Aishwarya Mirashi, Srushti Sonavane, Purva Lingayat, Tejas Padhiyar, Raviraj Joshi. Paper: *L3Cube-IndicNews: News-based Short Text and Long Document Classification Datasets in Indic Languages*, arXiv:2401.02254 (2024).

Source: https://github.com/l3cube-pune/indic-nlp

Publisher terms: Creative Commons Attribution-NonCommercial-ShareAlike 4.0, research use. License: https://creativecommons.org/licenses/by-nc-sa/4.0/ . The dataset-derived artifacts in this project are supplied under the same noncommercial/share-alike terms; raw publisher texts are not redistributed. Changes: normalized/deduplicated rows, conflicting/low-support label removal, training new TF-IDF/SVM and Word2Vec models. L3Cube does not endorse this project.

## Software and language resources

- Next.js / React / TypeScript: MIT licenses.
- Lucide: ISC license.
- FastAPI: MIT. Uvicorn: BSD. HTTPX: BSD.
- NumPy, SciPy and scikit-learn: BSD licenses. Joblib: BSD.
- Gensim: LGPL-2.1 library; used for word-vector training.
- NLTK: Apache-2.0 library. POS/chunker/word-list resources retain their distributed resource notices in `models/nltk_data/`.
- WordNet: Princeton University WordNet license, included in the resource archive; retain its copyright and license notices.
- langid.py: BSD-style project license.

Dependency packages retain their upstream notices when installed. No software license changes the applicable dataset/resource terms.

GDELT metadata links to third-party reporting. Rights in the original articles remain with the publishers. No article bodies or photos are scraped or redistributed as live content. Authored offline learning scenarios are explicitly fictional and separate from those sources.
