import asyncio
import datetime
import sys
import aiohttp
from ..article_retriever import ArticleRetriever


class GoogleRetriever(ArticleRetriever):
    """
    Retrieves evidence from Google search engine using Serper API

    Args:
        api_key (str): API key for Serper API
        fetch_concurrency (int): Maximum number of concurrent requests
        fetch_delay (int): Delay between requests
    """
    def __prepare_query(self, claim: str) -> str:
        return "-site:demagog.cz -site:x.com -site:facebook.com -site:instagram.com -site:reddit.com -filetype:pdf -filetype:xls -filetype:doc -filetype:docx -filetype:csv -filetype:xml " + claim

    async def retrieve(self, statement: dict, top_k: int = 10) -> dict:
        """
        Uses Serper API to search for articles
        """
        query = self.__prepare_query(statement["statement"])
        endpoint = "https://google.serper.dev/search"

        headers = {}
        params = {
            "q": query,
            "location": "Czechia",
            "gl": "cz",
            "hl": "cs",
            "autocorrect": "false",
            "apiKey": self.api_key,
            "num": top_k,
        }

        timeout = aiohttp.ClientTimeout(total=10)

        async with self.fetch_sem, aiohttp.ClientSession() as session:
            try:
                async with session.get(endpoint, headers=headers, params=params, timeout=timeout) as response:
                    response.raise_for_status()
                    results = await response.json()

                    await asyncio.sleep(self.fetch_delay)

                    return {
                        "id": statement["id"],
                        "query": query,
                        "date": datetime.datetime.now().isoformat(),
                        "results": results.get("organic", []),
                    }
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"Failed to fetch Google search results: {e}", file=sys.stderr)
                await asyncio.sleep(self.fetch_delay)
                return {}
