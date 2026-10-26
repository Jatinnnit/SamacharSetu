#!/usr/bin/env python3
"""
NLP Lab - Assignment 2:
Implement a program to perform language detection, Word Count, Sentence Count, and tokenization. (1 turn)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from backend import nlp

def main():
    print("=" * 75)
    print("NLP Lab | Assignment 2: Language Detection, Tokenization & Text Counts")
    print("=" * 75)

    test_samples = [
        ("English", "Natural language processing enables computers to understand text. Students analyze syntax and semantics daily!"),
        ("Hindi", "प्राकृतिक भाषा प्रसंस्करण कंप्यूटर को मानव भाषा समझने में सक्षम बनाता है। छात्र प्रतिदिन पुस्तकालय में अध्ययन करते हैं।"),
        ("Marathi", "संगणकीय भाषाशास्त्र आणि नैसर्गिक भाषा प्रक्रिया ही आधुनिक विज्ञानाची महत्त्वाची शाखा आहे. विद्यार्थी वाचनालयात पुस्तके वाचतात.")
    ]

    for expected_lang, text in test_samples:
        print(f"\n--- Analysis for Sample ({expected_lang}) ---")
        print(f"Input Text: \"{text}\"")

        # 1. Language Detection
        detection = nlp.detect(text)
        detected_name = nlp.LANGUAGES.get(detection['language'], detection['language'])
        print("\n[Step 1] Language Detection:")
        print(f"  * Detected Language : {detected_name} (code: '{detection['language']}')")
        print(f"  * Confidence Score  : {detection.get('confidence')}")
        print(f"  * Uncertain Flag    : {detection.get('uncertain')}")
        print(f"  * Method Used       : {detection.get('method')}")

        # 2. Tokenization
        tokens = nlp.tokenize(text)
        words_only = [t for t in tokens if t['is_word']]
        print("\n[Step 2] Tokenization (Unicode Script-Aware):")
        print(f"  * Total Tokens (including punctuation) : {len(tokens)}")
        print(f"  * Word Tokens Only                     : {len(words_only)}")
        print("  * First 10 Tokens with Spans:")
        for t in tokens[:10]:
            kind = "WORD" if t['is_word'] else "PUNCT"
            print(f"    - {t['text']:<18} [{kind}] (char {t['start']:02d}..{t['end']:02d})")

        # 3. Sentence Splitting & Count
        sents = nlp.sentences(text)
        print("\n[Step 3] Sentence Segmentation:")
        print(f"  * Sentence Count : {len(sents)}")
        for idx, s in enumerate(sents, 1):
            print(f"    Sentence {idx}: \"{s}\"")

        # 4. Comprehensive Statistics Summary
        print("\n[Step 4] Aggregate Text Statistics:")
        print(f"  * Word Count      : {len(words_only)}")
        print(f"  * Token Count     : {len(tokens)}")
        print(f"  * Sentence Count  : {len(sents)}")
        print(f"  * Character Count : {len(text)}")
        print(f"  * Avg Word Length : {sum(len(w['text']) for w in words_only) / max(len(words_only), 1):.2f} chars")

    print("\n" + "=" * 75)
    print("Assignment 2 demonstration completed successfully.")
    print("=" * 75)

if __name__ == '__main__':
    main()
