#!/usr/bin/env python3
"""
NLP Lab - Assignment 8:
Write a program to implement Latent Semantic Indexing for some application or task
like document categorization, opinion mining, sentiment analysis etc. (3 turns)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

import numpy as np
from backend import nlp

def main():
    print("=" * 80)
    print("NLP Lab | Assignment 8: Latent Semantic Indexing (LSI / LSA)")
    print("Application Tasks: Document Categorization & Topic-Based Opinion Mining")
    print("=" * 80)

    # -------------------------------------------------------------
    # Application Task 1: Document Categorization & Semantic Space
    # -------------------------------------------------------------
    print("\n[TASK 1] LSI for Unsupervised Document Categorization & Semantic Indexing:")
    documents = [
        # Category A: Space Science
        "NASA and ISRO launched astronomical satellites to investigate lunar soil and deep space.",
        "Space telescopes capture distant galaxies and planetary atmospheres in orbit.",
        "Astronomers discover exoplanet orbits using space observatory data.",

        # Category B: Sports & Athletics
        "The national cricket team won the international championship series with a brilliant final victory.",
        "Athletes train on the stadium track for the world athletics tournament match.",
        "Football players celebrate the winning goal in the tournament final."
    ]

    print(f"Total Input Documents: {len(documents)} across 2 thematic categories (Space & Sports)\n")

    lsa_result = nlp.lsa(documents)

    if lsa_result['status'] == 'available':
        print("Discovered Latent Semantic Topics (SVD Components):")
        for topic in lsa_result['topics']:
            top_words = ", ".join(f"{t['word']} ({t['weight']:+.3f})" for t in topic['terms'][:6])
            print(f"  * Component {topic['component']}: {top_words}")

        print(f"\nExplained Variance Ratio: {[round(float(v), 3) for v in lsa_result['explained_variance']]}")

        print("\nDocument Coordinates in 2D Latent Semantic Space (U * Sigma):")
        coords = lsa_result['coordinates']
        for i, (doc, coord) in enumerate(zip(documents, coords), 1):
            category_guess = "Space Science" if coord[0] > coord[1] else "Sports & Athletics"
            print(f"  Doc {i} [{category_guess:<18}]: C1={coord[0]:+7.4f}, C2={coord[1]:+7.4f} -> \"{doc[:55]}...\"")

    # -------------------------------------------------------------
    # Application Task 2: Topic-Based Opinion Mining & Sentiment Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[TASK 2] LSI Applied to Public Opinion Mining & Sentiment Analysis:")

    opinions = [
        "The newly renovated municipal library provides magnificent quiet study rooms and joyful support.",
        "Reading hall staff are exceptionally supportive and kind to young students.",
        "Horrible traffic jams and dangerous open potholes cause endless delays and deep fury.",
        "Broken traffic signals and reckless drivers cause severe panic, anger and road hazards.",
        "The youth cricket academy opened with proud celebrations and fantastic tournament matches."
    ]

    mined = nlp.opinion_mining(opinions)
    if mined['status'] == 'available':
        print(f"Analyzed {mined['summary']['total_opinions']} citizen opinion statements:")
        print(f"  * Positive Opinions: {mined['summary']['positive_count']} ({mined['summary']['overall_positive_pct']}%)")
        print(f"  * Negative Opinions: {mined['summary']['negative_count']} ({mined['summary']['overall_negative_pct']}%)")

        print("\nLSI Discovered Discussion Themes with Stance Classification:")
        for topic in mined['topics']:
            print(f"\n  [Topic {topic['topic_id']}] Key Concepts: {', '.join(topic['keywords'][:5])}")
            print(f"    - Sentiment Stance : {topic['sentiment_stance']} (Polarity: {topic['avg_polarity']:+.4f})")
            print(f"    - Dominant Emotion : {topic['dominant_emotion']}")
            print(f"    - Opinions Counted : {topic['count']}")

    print("\n" + "=" * 80)
    print("Assignment 8 demonstration completed successfully.")
    print("=" * 80)

if __name__ == '__main__':
    main()
