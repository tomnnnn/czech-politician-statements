"""
This script creates a hybrid search index using the HybridSearch class.

@author: Hai Phong Nguyen
"""

import argparse
import asyncio
import os

from .hybrid import HybridSearch


async def main():
    argparser = argparse.ArgumentParser(description="Create hybrid search index")
    argparser.add_argument("-d", "--dataset-path", type=str, help="Path to the dataset")
    argparser.add_argument("-o", "--output-path", type=str, help="Path to the index")
    args = argparser.parse_args()

    retriever = HybridSearch(args.dataset_path, args.output_path)
    await retriever.create_indices()
    # await retriever.load_indices()
    #
    # query = "STAN obvineni politici v kauze Dozimetr"
    # results = await retriever.search(query, "merged")
    # results = [i.text for i in results]
    #
    # while True:
    #     # Accept user input from stdin
    #     query = input("Enter your query (or type 'exit' to quit): ").strip()
    #
    #     if query.lower() == 'exit':
    #         break
    #
    #     # Perform the search
    #     results = await retriever.search(query, "merged")
    #     results = [i.text for i in results]
    #
    #     # Print the results
    #     if results:
    #         for r in results:
    #             print(r, end="\n\n")
    #     else:
    #         print("No results found.\n")
    #
    #     for r in results:
    #         print(r, end="\n\n")
    #


asyncio.run(main())
