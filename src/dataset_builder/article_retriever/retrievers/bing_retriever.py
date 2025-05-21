import asyncio
import sys
import aiohttp
import datetime
from ..article_retriever import ArticleRetriever


class BingRetriever(ArticleRetriever):
    def __prepare_query(self, claim: str) -> str:
         return "-site:demagog.cz -site:x.com -site:facebook.com -site:instagram.com -site:reddit.com -filetype:pdf -filetype:xls -filetype:doc -filetype:docx -filetype:csv -filetype:xml " + claim


    async def retrieve(self, statement: dict, top_k: int = 10) -> dict:
        api_endpoint = "https://api.bing.microsoft.com/v7.0/search"
        query = self.__prepare_query(statement['statement'])
        params = {"q": query, "count": top_k}
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        timeout = aiohttp.ClientTimeout(total=10)
        response_data = {"id": statement['id'], "query": query, "date": datetime.datetime.now().isoformat(), "results": []}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_endpoint, headers=headers, params=params, timeout=timeout) as response:
                    response.raise_for_status()
                    results = (await response.json()).get("webPages", {}).get("value", [])

                    response_data["results"] = [
                        {"title": item["name"], "snippet": item["snippet"], "url": item["url"]}
                        for item in results
                    ]

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Error fetching Bing search results: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)

        return response_data
