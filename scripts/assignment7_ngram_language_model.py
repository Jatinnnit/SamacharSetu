#!/usr/bin/env python3
"""
NLP Lab - Assignment 7:
Write a program to implement N-gram models and determine the probability of a sequence of words using N-gram Language Model. (2 turns)
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

def main():
    print("=" * 80)
    print("NLP Lab | Assignment 7: N-gram Language Models & Word Sequence Probability")
    print("=" * 80)

    training_corpus = """the library opens early in the morning.
the library welcomes students and readers.
students read books and study in the library.
readers enjoy quiet study in the community library.
the university supports student education and reading."""

    test_sequences = [
        "students read books in the library.",
        "the library welcomes readers.",
        "students enjoy space exploration."  # Contains unseen vocabulary
    ]

    print("Training Corpus:")
    for line in training_corpus.strip().split('\n'):
        print(f"  > {line}")

    # Evaluate N-gram models across orders (Unigram, Bigram, Trigram)
    orders = [(1, "Unigram (N=1)"), (2, "Bigram (N=2)"), (3, "Trigram (N=3)")]

    for seq in test_sequences:
        print("\n" + "=" * 80)
        print(f"EVALUATING SEQUENCE: \"{seq}\"")
        print("=" * 80)

        for n, order_name in orders:
            print(f"\n--- Model: {order_name} with Laplace Add-1 Smoothing (alpha=1.0) ---")

            res = nlp.ngram(training_corpus, seq, n=n, alpha=1.0)

            print(f"Vocabulary Size (|V|)    : {res['vocabulary_size']}")
            print(f"Unknown (OOV) Words Count: {res['oov_count']}")

            print("\nWord-by-Word Conditional Probabilities P(w_i | context):")
            print(f"  {'Context':<24} {'Word (w_i)':<16} {'Count(h,w)':<12} {'Count(h)':<10} {'P(w_i|h)':<12}")
            print("  " + "-" * 75)

            prob_factors = []
            for ev in res['events']:
                ctx_display = ev['context'] if ev['context'] else "<s>"
                print(f"  {ctx_display:<24} {ev['word']:<16} {ev['count']:<12} {ev['context_count']:<10} {ev['probability']:<12.6f}")
                prob_factors.append(ev['probability'])

            print("\nSequence Probability Calculation:")
            print("  Formula: P(W) = ∏ P(w_i | h_i)")
            print(f"  * Total Sequence Probability P(W) : {res['sequence_probability']:.8e}")
            if res['log_probability'] is not None:
                print(f"  * Log Probability log P(W)        : {res['log_probability']:.4f}")
            print(f"  * Model Perplexity                : {res['perplexity']:.4f}" if res['perplexity'] else "  * Model Perplexity                : ∞ (Zero probability event)")

    # Demonstration without smoothing (MLE) vs with smoothing
    print("\n" + "=" * 80)
    print("[Comparison] MLE (No Smoothing, alpha=0) vs. Laplace (alpha=1.0) on Bigram Model:")
    target_seq = "the library welcomes readers."
    mle_res = nlp.ngram(training_corpus, target_seq, n=2, alpha=0.0)
    laplace_res = nlp.ngram(training_corpus, target_seq, n=2, alpha=1.0)

    mle_p = f"{mle_res['perplexity']:.4f}" if mle_res['perplexity'] is not None else "∞"
    lap_p = f"{laplace_res['perplexity']:.4f}" if laplace_res['perplexity'] is not None else "∞"

    print(f"Sequence: \"{target_seq}\"")
    print(f"  * MLE (alpha=0.0)    -> P(W) = {mle_res['sequence_probability']:.8e} | Perplexity = {mle_p}")
    print(f"  * Laplace (alpha=1.0) -> P(W) = {laplace_res['sequence_probability']:.8e} | Perplexity = {lap_p}")

    print("\n" + "=" * 80)
    print("Assignment 7 demonstration completed successfully.")
    print("=" * 80)

if __name__ == '__main__':
    main()
