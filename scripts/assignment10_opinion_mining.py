#!/usr/bin/env python3
"""
PCCOE Pune - T.Y. BTech CSE (AI & ML)
Assignment 10:
Write a program for identifying sentiment (positive or negative) and emotions expressed in text
and develop a system for opinion mining with suitable topic modelling technique.
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import nlp

def main():
    print("=" * 80)
    print("PCCOE Pune - T.Y. BTech CSE (AI & ML) | NLP Lab Assignment 10")
    print("Topic: Sentiment Analysis, Emotion Detection & Topic-Based Opinion Mining")
    print("=" * 80)

    # -------------------------------------------------------------
    # Part 1: Sentiment & Emotion Analysis on Individual Texts
    # -------------------------------------------------------------
    print("\n[PART 1] Single Text Sentiment & Emotion Identification:")
    test_cases = [
        "The newly inaugurated solar power initiative is a brilliant triumph, bringing joy and hope to all citizens!",
        "Devastating flash floods and terrible destruction triggered deep grief, sorrow and panic across communities.",
        "Commuters are in absolute rage and outrage over constant train cancellations, potholes and corrupt municipal delays."
    ]

    for i, sample in enumerate(test_cases, 1):
        print(f"\nExample {i}: \"{sample}\"")
        result = nlp.sentiment_emotion(sample)
        print(f"  * Polarity Label    : {result['label'].upper()} (Compound Score: {result['compound']:+.4f})")
        print(f"  * Scores Breakdown  : Pos: {result['polarity']['pos']:.2f} | Neu: {result['polarity']['neu']:.2f} | Neg: {result['polarity']['neg']:.2f}")
        print(f"  * Dominant Emotion  : {result['dominant_emotion']}")
        print(f"  * Emotion Spectrum  : {', '.join(f'{k}: {v:.1%}' for k, v in result['emotions'].items() if v > 0)}")
        if result['evidence']:
            print(f"  * Evidential Words  : {result['evidence']}")

    # -------------------------------------------------------------
    # Part 2: Opinion Mining System with Topic Modelling (LSI/SVD)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PART 2] Opinion Mining System with Topic Modelling (LSI / TruncatedSVD):")
    print("Analyzing a multi-topic collection of public opinions & citizen feedback...")

    opinion_dataset = [
        # Topic A: Public Library & Reading Culture (Positive)
        "The municipal library renovation is magnificent with spacious quiet study rooms and joyful readers.",
        "Staff at the community reading hall are extraordinarily helpful, friendly and supportive.",
        "I love spending afternoons in the reading section, it offers wonderful peace and inspiration.",
        
        # Topic B: Roads, Traffic & Infrastructure (Negative)
        "Dangerous open potholes and terrible road quality cause awful daily traffic jams and furious frustration.",
        "Commuters face immense anger due to reckless speeding trucks and broken traffic signals creating hazards.",
        "Severe road waterlogging during monsoon causes panic, delays and endless rage among drivers.",

        # Topic C: Sports & Youth Recreation (Positive/Mixed)
        "The youth sports arena opened with an exciting cricket tournament and proud celebration.",
        "Young athletes are thrilled and joyous about the newly surfaced badminton and tennis courts.",
        "Local sports facilities still need better maintenance and regular water supply."
    ]

    mined = nlp.opinion_mining(opinion_dataset)

    print("\n--- Opinion Mining Executive Summary ---")
    summary = mined['summary']
    print(f"Total Feedback Statements : {summary['total_opinions']}")
    print(f"Positive Sentiment Ratio  : {summary['positive_count']} ({summary['overall_positive_pct']}%)")
    print(f"Negative Sentiment Ratio  : {summary['negative_count']} ({summary['overall_negative_pct']}%)")
    print(f"Neutral / Mixed Ratio     : {summary['neutral_count']}")
    print(f"Identified Discussion Themes (LSA Topics): {summary['topics_identified']}")

    print("\n--- Topic-by-Topic Mined Opinions ---")
    for topic in mined['topics']:
        print(f"\n[Topic {topic['topic_id']}] Key Themes: {', '.join(topic['keywords'][:5])}")
        print(f"  * Opinions Counted     : {topic['count']}")
        print(f"  * Stance Classification : {topic['sentiment_stance']} (Avg Polarity: {topic['avg_polarity']:+.4f})")
        print(f"  * Polarity Breakdown   : {topic['positive_pct']}% Positive | {topic['negative_pct']}% Negative | {topic['neutral_pct']}% Neutral")
        print(f"  * Dominant Emotion     : {topic['dominant_emotion']}")
        print("  * Sample Quotes:")
        for op in topic['opinions'][:2]:
            print(f"      - [{op['label'].upper()}] \"{op['text'][:80]}...\"")

    print("\n" + "=" * 80)
    print("Assignment 10 demonstration completed successfully.")
    print("=" * 80)

if __name__ == '__main__':
    main()
