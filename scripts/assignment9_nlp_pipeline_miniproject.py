#!/usr/bin/env python3
"""
NLP Lab - Assignment 9:
Develop a mini project by implementing suitable NLP tasks in NLP pipeline for different topics
like question answering system, sentiment analysis, speech recognition, machine translation, etc. (3 turns)
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from backend import nlp
from backend.scraper import scrape_web_data

async def run_pipeline():
    print("=" * 85)
    print("NLP Lab | Assignment 9: End-to-End NLP Pipeline Mini Project (SamacharSetu)")
    print("Tasks: Web Scraping -> Preprocessing -> Linguistics (POS/NER) -> Embeddings ->")
    print("       Classification -> Sentiment Analysis -> Extractive Question Answering System")
    print("=" * 85)

    # -------------------------------------------------------------
    # STAGE 1: Data Gathering (Web Scraping / Corpus Ingestion)
    # -------------------------------------------------------------
    print("\n[STAGE 1] Data Gathering & Web Scraping:")
    target_url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    scraped = await scrape_web_data(target_url)
    print(f"  * Source URL        : {scraped.get('url')}")
    print(f"  * Document Title    : {scraped.get('title')}")
    print(f"  * Paragraphs Gathered: {len(scraped.get('paragraphs', []))}")
    print(f"  * Total Words       : {scraped['stats']['word_count']}")

    sample_article = (
        "Google and Microsoft researchers announced breakthrough artificial intelligence software algorithms in California. "
        "The newly trained language models understand multilingual sentences with exceptional accuracy and joy. "
        "Dr. Rajesh Sharma led the natural language processing research group during this major achievement. "
        "However, some critics expressed serious concern and worry regarding computational power demands and energy costs."
    )

    print(f"\nProcessing Corpus Passage:\n\"{sample_article}\"")

    # -------------------------------------------------------------
    # STAGE 2: Language Detection & Text Preprocessing
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[STAGE 2] Language Detection, Sentence Segmentation & Tokenization:")
    det = nlp.detect(sample_article)
    tokens = nlp.tokenize(sample_article)
    sents = nlp.sentences(sample_article)
    words = [t for t in tokens if t['is_word']]
    print(f"  * Language Detected : {nlp.LANGUAGES.get(det['language'], det['language'])} (confidence: {det['confidence']})")
    print(f"  * Sentence Count    : {len(sents)}")
    print(f"  * Token Count       : {len(tokens)} (Words: {len(words)})")

    # -------------------------------------------------------------
    # STAGE 3: Linguistic Analysis (Stemming, WordNet Lemmatization, POS & NER)
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[STAGE 3] Linguistic Normalization, POS Tagging & Named Entity Recognition:")
    ling = nlp.linguistics(sample_article, 'en')
    if ling['status'] == 'available':
        print("  * POS & Morphological Tokens (Sample):")
        for r in ling['rows'][:6]:
            if r['is_word']:
                print(f"    - {r['text']:<16} Tag: {r['tag']:<6} Role: {r['description']:<18} Lemma: {r['lemma']}")

        print("\n  * Extracted Named Entities (MaxEnt Chunker):")
        for ent in ling['entities']:
            print(f"    - [{ent['type']}] \"{ent['text']}\" (Span: {ent['start']}..{ent['end']})")

    # -------------------------------------------------------------
    # STAGE 4: Category Classification & Word Embeddings
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[STAGE 4] News Classification & Word2Vec Embeddings:")
    emb_res = nlp.classify_with_embeddings(sample_article, 'en', 'skipgram')
    if emb_res['status'] == 'available':
        print(f"  * Predicted Category       : {emb_res['predicted_category'].upper()} (Similarity: {emb_res['similarity_score']:.4f})")
        print(f"  * Document Vector Sample   : {emb_res['vector_sample'][:6]}")
        print("  * Category Centroid Scores : " + ", ".join(f"{s['category']}: {s['similarity']:.3f}" for s in emb_res['scores'][:4]))

    # -------------------------------------------------------------
    # STAGE 5: Sentiment Analysis & Emotion Spectrum
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[STAGE 5] Sentiment Analysis & Emotion Identification:")
    sent = nlp.sentiment_emotion(sample_article, 'en')
    if sent['status'] == 'available':
        print(f"  * Polarity Label    : {sent['label'].upper()} (Compound: {sent['compound']:+.4f})")
        print(f"  * Dominant Emotion  : {sent['dominant_emotion']}")
        print(f"  * Polarity Breakdown: Pos: {sent['polarity']['pos']} | Neu: {sent['polarity']['neu']} | Neg: {sent['polarity']['neg']}")
        print(f"  * Emotion Evidence  : {sent.get('evidence')}")

    # -------------------------------------------------------------
    # STAGE 6: Extractive Question Answering System
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[STAGE 6] Question Answering (QA) System over Corpus:")
    test_questions = [
        "Who led the natural language processing research group?",
        "Where did researchers announce the algorithms?",
        "What concerns did critics express?"
    ]

    for q in test_questions:
        qa_res = nlp.extractive_qa(sample_article, q)
        print(f"\n  Question : \"{q}\"")
        print(f"  Answer   : \"{qa_res['best_answer']}\"")
        print(f"  Confidence: {qa_res['confidence']:.4f} | Evaluated {qa_res['total_sentences_evaluated']} candidate sentences")

    # -------------------------------------------------------------
    # STAGE 7: Machine Translation Notice
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[STAGE 7] Machine Translation Specification Check:")
    print("  * Machine Translation status: Local offline privacy boundary preserved.")
    print("    No external cloud translation API is invoked to keep user data private and local.")

    print("\n" + "=" * 85)
    print("Assignment 9 Mini Project NLP Pipeline executed successfully!")
    print("=" * 85)

if __name__ == '__main__':
    asyncio.run(run_pipeline())
