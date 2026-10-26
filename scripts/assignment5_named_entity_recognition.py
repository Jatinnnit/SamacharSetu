#!/usr/bin/env python3
"""
NLP Lab - Assignment 5:
Write a program to implement Named Entity Recognition. (1 turn)
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

import nltk
local_nltk = str(ROOT / 'models/nltk_data')
if local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk)

from backend import nlp

def main():
    print("=" * 80)
    print("NLP Lab | Assignment 5: Named Entity Recognition (NER)")
    print("=" * 80)

    test_articles = [
        "Sundar Pichai and Satya Nadella met at the Google headquarters in California to discuss artificial intelligence.",
        "The Indian Space Research Organisation in Bengaluru announced that Chandrayaan reached the Moon successfully.",
        "Prime Minister Narendra Modi visited Paris and Berlin for bilateral summits with European leaders."
    ]

    entity_type_counter = Counter()

    for idx, text in enumerate(test_articles, 1):
        print(f"\n--- Article {idx} ---")
        print(f"Text: \"{text}\"")

        ling = nlp.linguistics(text, 'en')

        if ling['status'] == 'available':
            entities = ling['entities']
            print(f"\nExtracted Named Entities ({len(entities)} found):")
            print(f"  {'Entity Text':<30} {'Entity Type':<16} {'Character Span':<15}")
            print("  " + "-" * 65)

            for ent in entities:
                entity_type_counter[ent['type']] += 1
                span_str = f"{ent['start']:02d}..{ent['end']:02d}"
                print(f"  {ent['text']:<30} {ent['type']:<16} {span_str:<15}")

            # Visual highlighted text
            highlighted = text
            # Offset tracking
            for ent in sorted(entities, key=lambda e: e['start'], reverse=True):
                s, e, t = ent['start'], ent['end'], ent['type']
                highlighted = highlighted[:s] + f"[{ent['text']}]({t})" + highlighted[e:]
            print(f"\nVisual Inline Tagging:\n  {highlighted}")
        else:
            print(f"Notice: {ling.get('notice')}")

    print("\n" + "-" * 80)
    print("Aggregate Named Entity Frequencies across Corpus:")
    for etype, count in entity_type_counter.most_common():
        print(f"  * {etype:<16} : {count} entities identified")

    print("\n" + "=" * 80)
    print("Assignment 5 demonstration completed successfully.")
    print("=" * 80)

if __name__ == '__main__':
    main()
