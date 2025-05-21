"""
@file: build_dataset.py
@author: Hai Phong Nguyen

This script scrapes statements from Demagog.cz and provides scraped evidence documents for each statement.
"""

import asyncio
from datetime import datetime
import json
import os
import shutil
import sys
import aiohttp
from itertools import islice

from tqdm.asyncio import tqdm

from dataset_manager import Dataset
import sqlite3
from .article_scraper import ArticleScraper

SCRATCHDIR=os.environ.get("SCRATCHDIR")

def get_missing_urls():
    dataset_path = "datasets/dataset.sqlite"
    con = sqlite3.connect(dataset_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("""
        SELECT * FROM failed_scrapes
    """)

    rows = cur.fetchall()

    row_dicts = [dict(row) for row in rows]

    return row_dicts

def get_missing_urls_from_json():
    """
    Reads the missing URLs from a JSON file.
    """
    with open("missing_urls.json", "r") as f:
        missing_urls = json.load(f)
    return missing_urls

async def main():
    print("Starting the script...")
    scraper = ArticleScraper()
    dataset = Dataset("datasets/demagog_deduplicated.sqlite")
    articles = dataset.get_articles()
    missing_articles = get_missing_urls_from_json()
    scraped_urls = [a.url for a in articles]
    
    # Filter out identical URLs
    missing_urls = []

    for article in missing_articles:
        if article.get("url", None) not in scraped_urls:
            missing_urls.append(article)

    with open("missing_urls.json", "w") as f:
        json.dump(missing_urls, f, indent=4, ensure_ascii=False)

    missing_urls = [i["url"] for i in missing_urls if i["url"] and not i["url"].endswith(".pdf")]
    missing_urls = [i for i in missing_urls if not "195.46.72.16" in i]
    missing_urls = list(set(missing_urls))

    # Define batch size
    batch_size = 100

    # Split missing_urls into batches of 2000
    def chunks(iterable, size):
        it = iter(iterable)
        for first in it:
            yield [first] + list(islice(it, size - 1))

    # Process each batch sequentially
    iteration_number = 3
    print("Processing batches of missing URLs...")
    for i, batch in enumerate(chunks(missing_urls, batch_size)):
        # Scrape and extract articles for the batch
        if i < iteration_number:
            continue

        articles, unprocessed = await scraper.scrape_extractus_async(batch)

        # Add timestamp to the articles
        for a in articles:
            a["accessed"] = datetime.now().isoformat()

        # Save the articles in a batch-specific file
        with open(f"articles/{iteration_number}_articles.json", "w") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        # Save unprocessed articles if any
        with open(f"articles/{iteration_number}_unprocessed.json", "w") as f:
            json.dump(unprocessed, f, indent=4, ensure_ascii=False)

        print(f"Successfully scraped {len(articles)} articles in batch {iteration_number}.")
        # Increment iteration number for the next batch
        iteration_number += 1

    print("Batch processing completed.")

if __name__ == "__main__":
    asyncio.run(main())
