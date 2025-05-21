from tqdm.asyncio import tqdm
import asyncio

class ArticleRetriever:
    def __init__(self, api_key: str, fetch_concurrency: int = 5, fetch_delay: int = 1):
        self.fetch_sem = asyncio.Semaphore(fetch_concurrency)
        self.fetch_delay = fetch_delay
        self.api_key = api_key

    def set_fetch_delay(self, fetch_delay: int):
        self.fetch_delay = fetch_delay

    def set_fetch_concurrency(self, fetch_concurrency: int):
        self.fetch_sem = asyncio.Semaphore(fetch_concurrency)

    def __prepare_query(self, claim: str) -> str:
        return claim

    async def retrieve(self, statement: dict, top_k: int = 10) -> dict:
        raise NotImplementedError

    async def batch_retrieve(self, statements: list[dict], top_k: int = 10) -> list:
        tasks = [self.retrieve(statement, top_k) for statement in statements]
        return await tqdm.gather(*tasks, desc="Searching for evidence", unit="claim")
