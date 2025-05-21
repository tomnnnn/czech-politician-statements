import asyncio
import datetime
import sys
import aiohttp
from ..article_retriever import ArticleRetriever


class CriteriaRetriever(ArticleRetriever):
    def __prepare_query(self, claim: str) -> str:
        return claim

    async def retrieve(self, statement: dict, top_k: int = 10) -> dict:
        query = self.__prepare_query(statement["statement"])
        endpoint = "https://lab.idiap.ch/criteria/search_tom"
        response_data = {
            "id": statement["id"],
            "query": query,
            "date": datetime.datetime.now().isoformat(),
            "results": [],
        }

        headers = {
            "Content-Type": "application/json",
        }
        payload = dict(claim=query, num_results=top_k)

        timeout = aiohttp.ClientTimeout(total=10)

        async with self.fetch_sem, aiohttp.ClientSession() as session:
            try:
                async with session.post(endpoint, headers=headers, timeout=timeout, json=payload) as response:
                    response.raise_for_status()
                    results = await response.json()

                    await asyncio.sleep(self.fetch_delay)
                    response_data["results"] = [
                        {
                            "title": result["title"],
                            "snippet": result["fulltext"],
                            "url": result["url"],
                            "date": result["date"],
                            "score": result["score"],
                        }
                        for result in results
                    ]
                    return response_data

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"Failed to fetch Criteria search results: {e}", file=sys.stderr)
                await asyncio.sleep(self.fetch_delay)
                return {}
