#!/usr/bin/env python3
"""
NLP Lab - Assignment 3:
Write a program to perform stemming words. Also, write a program for lemmatizing words using WordNet. (1 turn)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

import nltk
local_nltk = str(ROOT / 'models/nltk_data')
if local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk)

from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from backend import nlp

def main():
    print("=" * 80)
    print("NLP Lab | Assignment 3: Word Stemming and WordNet Lemmatization")
    print("=" * 80)

    # Initialize stemmers and lemmatizer
    porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    # Part 1: Word-level comparative table
    print("\n[PART 1] Word-level Comparison: Stemming vs. WordNet Lemmatization")
    print("-" * 80)
    print(f"{'Original Word':<16} {'POS Tag':<10} {'Porter Stem':<15} {'Snowball Stem':<15} {'WordNet Lemma':<15}")
    print("-" * 80)

    word_test_cases = [
        ('studies', 'v', 'Verb'),
        ('studying', 'v', 'Verb'),
        ('students', 'n', 'Noun'),
        ('universities', 'n', 'Noun'),
        ('corpora', 'n', 'Noun'),
        ('better', 'a', 'Adjective'),
        ('running', 'v', 'Verb'),
        ('leaves', 'n', 'Noun'),
        ('easily', 'r', 'Adverb'),
        ('matrices', 'n', 'Noun'),
        ('connected', 'v', 'Verb'),
        ('connectivity', 'n', 'Noun')
    ]

    for word, wn_pos, label in word_test_cases:
        p_stem = porter.stem(word)
        s_stem = snowball.stem(word)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        print(f"{word:<16} {label:<10} {p_stem:<15} {s_stem:<15} {lemma:<15}")

    # Part 2: Full Sentence Contextual Normalization
    print("\n" + "-" * 80)
    print("[PART 2] Contextual Sentence Lemmatization (POS-Aware via backend.nlp)")
    print("-" * 80)

    sentence = "The university researchers were studying complex corpora and finding better linguistic models."
    print(f"Input Sentence: \"{sentence}\"\n")

    result = nlp.linguistics(sentence, 'en')
    if result['status'] == 'available':
        print(f"{'Token':<14} {'POS Tag':<8} {'Grammar Role':<16} {'Snowball Stem':<16} {'WordNet Lemma':<16}")
        print("-" * 75)
        for row in result['rows']:
            if row['is_word']:
                print(f"{row['text']:<14} {row['tag']:<8} {row['description']:<16} {row['stem']:<16} {row['lemma']:<16}")
        print("\nNotice:")
        print("  - Stemming uses algorithmic suffix stripping and may produce non-dictionary roots (e.g. 'studi').")
        print("  - Lemmatization uses WordNet dictionary lookup with morphological analysis to return valid base forms.")
    else:
        print(f"Notice: {result.get('notice')}")

    print("\n" + "=" * 80)
    print("Assignment 3 demonstration completed successfully.")
    print("=" * 80)

if __name__ == '__main__':
    main()
