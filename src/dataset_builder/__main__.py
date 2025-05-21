"""
@file: build_dataset.py
@author: Hai Phong Nguyen

This script scrapes statements from Demagog.cz and provides scraped evidence documents for each statement.
"""

import asyncio
from .build_dataset import build_dataset

if __name__ == "__main__":
    asyncio.run(build_dataset())
    print("Dataset built successfully.")
