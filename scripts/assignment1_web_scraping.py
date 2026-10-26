#!/usr/bin/env python3
"""
NLP Lab - Assignment 1:
Write a program for scraping the Web to Gather Data. (1 turn)
"""
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.scraper import scrape_web_data

async def main():
    print("=" * 75)
    print("NLP Lab | Assignment 1: Web Scraping to Gather Data")
    print("=" * 75)

    test_urls = [
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://example.com/news/sample"
    ]

    for idx, url in enumerate(test_urls, 1):
        print(f"\n[{idx}] Scraping Web Data from: {url}")
        print("-" * 65)

        data = await scrape_web_data(url)

        if data['status'] == 'available':
            print(f"  * Document Title   : {data.get('title')}")
            print(f"  * Domain           : {data.get('domain')}")
            print(f"  * Meta Description : {data.get('meta_description')}")
            print(f"  * Paragraph Count  : {data['stats']['paragraph_count']}")
            print(f"  * Total Words      : {data['stats']['word_count']:,}")
            print(f"  * Total Characters : {data['stats']['character_count']:,}")

            print("\n  [Extracted Headings / Topics]:")
            for h in data.get('headings', [])[:5]:
                print(f"    - {h}")

            print("\n  [Sample Paragraphs Gathered]:")
            for p in data.get('paragraphs', [])[:2]:
                print(f"    > \"{p}\"")

            print("\n  [Extracted Hyperlinks]:")
            for l in data.get('links', [])[:3]:
                print(f"    * {l['text']} -> {l['url']}")

            print(f"\n  Notice: {data.get('notice')}")
        else:
            print(f"  Error scraping {url}: {data.get('notice')}")

    # Output file demonstration
    output_path = ROOT / "scraped_data_demo.json"
    demo_data = await scrape_web_data(test_urls[0])
    output_path.write_text(json.dumps(demo_data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n-> Saved gathered data to: {output_path.name}")

    print("\n" + "=" * 75)
    print("Assignment 1 demonstration completed successfully.")
    print("=" * 75)

if __name__ == '__main__':
    asyncio.run(main())
