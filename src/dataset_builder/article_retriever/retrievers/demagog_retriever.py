import asyncio
import datetime
import sys
import pprint

import aiohttp

from ..article_retriever import ArticleRetriever


class DemagogRetriever(ArticleRetriever):
    """
    Assumes evidence links are already scraped and stored in the statements database.
    """

    def __prepare_query(self, claim: str) -> str:
        return claim

    async def retrieve(self, statement: dict, top_k: int = 10) -> dict:
        return {
            "id": statement["id"],
            "query": None,
            "date": None,
            "results": statement['evidence_links'],
        }
