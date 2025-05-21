import json
import asyncio
import os
from .config import CONFIG
from .scraper import DemagogScraper

if __name__ == "__main__":
    scraper = DemagogScraper(CONFIG['FromYear'], CONFIG['ToYear'], CONFIG['FirstNPages'])
    output_path = os.path.join(CONFIG['OutputDir'], "statements.json")

    os.makedirs(CONFIG['OutputDir'], exist_ok=True)
    if os.path.isfile(output_path):
        print(f"File {output_path} already exists, overwrite? [y/n]")
        if input().lower() != "y":
            print("Exiting...")
            exit(0)

    statements = asyncio.run(scraper.run())
    print(f"Scraped {len(statements)} statements")

    with open(output_path, "w") as f:
        json.dump(statements, f, indent=4, ensure_ascii=False)
