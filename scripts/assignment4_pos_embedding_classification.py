#!/usr/bin/env python3
"""
PCCOE Pune - T.Y. BTech CSE (AI & ML)
Assignment 4:
Write a program to perform POS Tagging and word embeddings. Perform text classification based on it.
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from backend import nlp

def main():
    print("=" * 75)
    print("PCCOE Pune - T.Y. BTech CSE (AI & ML) | NLP Lab Assignment 4")
    print("Topic: POS Tagging, Word Embeddings & Embedding-Based Text Classification")
    print("=" * 75)

    sample_sentences = [
        "Google and Microsoft researchers launch artificial intelligence algorithms for mobile devices",
        "The national cricket team won the championship match with a thrilling final goal",
        "Students at the university study for engineering exams in the campus library",
        "The hospital doctors treated patients during the viral infection outbreak"
    ]

    for idx, text in enumerate(sample_sentences, 1):
        print(f"\n--- Sample {idx}: \"{text}\" ---")

        # Step 1: Part-of-Speech (POS) Tagging
        print("\n[Step 1] Part-of-Speech (POS) Tagging:")
        ling = nlp.linguistics(text, 'en')
        if ling['status'] == 'available':
            for r in ling['rows'][:8]:
                print(f"  Token: {r['text']:<15} Tag: {r['tag']:<6} Role: {r['description']}")
            if len(ling['rows']) > 8:
                print(f"  ... and {len(ling['rows']) - 8} more tokens")

        # Step 2: Word Embeddings (Word2Vec)
        print("\n[Step 2] Word Embeddings (Word2Vec 48-dim vectors):")
        res = nlp.classify_with_embeddings(text, lang='en', mode='skipgram')
        print(f"  Words found in vocabulary: {res['words_used']}")
        print(f"  Averaged Document Vector (first 6 dims): {res['vector_sample'][:6]}")

        # Step 3: Text Classification based on Word Embeddings
        print("\n[Step 3] Text Classification based on Embedding Centroid Similarity:")
        print(f"  -> Predicted Category: {res['predicted_category'].upper()} (Cosine Sim: {res['similarity_score']:.4f})")
        print("  Top category scores:")
        for score_item in res['scores'][:4]:
            print(f"     * {score_item['category']:<15} : {score_item['similarity']:.4f}")
        print(f"  Method: {res['method']}")

    print("\n" + "=" * 75)
    print("Assignment 4 demonstration completed successfully.")
    print("=" * 75)

if __name__ == '__main__':
    main()
