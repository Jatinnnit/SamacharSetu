#!/usr/bin/env python3
"""
NLP Lab - Assignment 6:
Write a program to implement TD-IDF method. (1 turn)
"""
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from backend import nlp

def manual_tfidf_demo(corpus):
    """Demonstrates step-by-step mathematical calculation of TF, IDF, and TF-IDF."""
    print("\n--- Step-by-Step Mathematical Calculation of TF-IDF ---")

    tokenized_docs = []
    for doc in corpus:
        words = [t['text'].casefold() for t in nlp.tokenize(doc) if t['is_word']]
        tokenized_docs.append(words)

    N = len(corpus)

    # 1. Document Frequency
    df = Counter()
    for words in tokenized_docs:
        for w in set(words):
            df[w] += 1

    # Select top terms
    terms = sorted([t for t, count in df.items() if count >= 1])

    # 2. IDF calculation: idf(t) = ln((1 + N) / (1 + df(t))) + 1
    idf = {}
    print("\n[Step 1] Inverse Document Frequency (IDF) Scores:")
    print(f"{'Term':<18} {'Doc Frequency (DF)':<22} {'IDF Score':<15}")
    print("-" * 55)
    for t in terms[:12]:
        idf[t] = math.log((1 + N) / (1 + df[t])) + 1.0
        print(f"{t:<18} {df[t]:<22} {idf[t]:.4f}")

    # 3. Term Frequency & TF-IDF table
    print("\n[Step 2] Term Frequency (TF) & TF-IDF Vectors across Documents:")
    for i, (doc, words) in enumerate(zip(corpus, tokenized_docs), 1):
        total_w = len(words)
        counts = Counter(words)
        print(f"\nDocument {i}: \"{doc}\" (Total Words: {total_w})")
        print(f"  {'Term':<16} {'Raw Count':<12} {'TF (count/len)':<16} {'IDF':<12} {'TF-IDF Weight':<15}")
        print("  " + "-" * 70)
        for t in terms[:8]:
            c = counts[t]
            tf_val = c / total_w
            tfidf_val = tf_val * idf[t]
            print(f"  {t:<16} {c:<12} {tf_val:<16.4f} {idf[t]:<12.4f} {tfidf_val:<15.4f}")


def main():
    print("=" * 80)
    print("NLP Lab | Assignment 6: Term Frequency - Inverse Document Frequency (TF-IDF)")
    print("=" * 80)

    corpus = [
        "Natural language processing uses statistical algorithms for computational linguistics.",
        "Machine learning algorithms improve natural language processing models.",
        "Cricket players participate in regional championship tournaments."
    ]

    print("Input Corpus Documents:")
    for i, d in enumerate(corpus, 1):
        print(f"  Doc {i}: \"{d}\"")

    # Step-by-step math demonstration
    manual_tfidf_demo(corpus)

    # API / module demonstration via backend.nlp.compute_tfidf
    print("\n" + "=" * 80)
    print("[Step 3] Integrated Backend Vectorizer & Document Cosine Similarity:")
    res = nlp.compute_tfidf(corpus)

    if res['status'] == 'available':
        print(f"\nVocabulary size analyzed: {len(res['vocabulary'])} terms")
        print("\nPairwise Cosine Similarity Matrix (Normalized TF-IDF Space):")
        print("       " + " ".join(f"Doc {j+1:<6}" for j in range(len(corpus))))
        for i, row in enumerate(res['similarity_matrix']):
            row_str = " ".join(f"{val:.4f}  " for val in row)
            print(f"Doc {i+1} : {row_str}")

        print("\nKey Extracted Terms per Document:")
        for doc_info in res['documents']:
            terms_str = ", ".join(f"{item['term']} ({item['tfidf']:.3f})" for item in doc_info['top_terms'])
            print(f"  * Doc {doc_info['doc_id']}: {terms_str}")

    print("\n" + "=" * 80)
    print("Assignment 6 demonstration completed successfully.")
    print("=" * 80)

if __name__ == '__main__':
    main()
