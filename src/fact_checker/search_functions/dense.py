"""
DenseSearch class for creating, managing and searching FAISS indices using the BGE-M3 model.

@author: Hai Phong Nguyen
"""

from typing import Optional, TypedDict
import asyncio
import numpy as np
import torch
from dataset_manager.models import Segment
import os
import faiss
from fact_checker.search_functions.base import SearchFunction
from FlagEmbedding import BGEM3FlagModel
import time

class IndexEntry(TypedDict):
    index: faiss.Index
    corpus: list[Segment]


class DenseSearch(SearchFunction):
    indices: dict[str|int, IndexEntry]
    model: BGEM3FlagModel

    def __init__(self, batch_size=32, **kwargs):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self.batch_size = batch_size
        self.indices = {}
        self.sem = asyncio.Semaphore(20)
        self.sem_encode = asyncio.Semaphore(1)

    async def _encode_documents_async(self, documents: list[str]) -> np.ndarray:
        async with self.sem_encode:
            embeddings = await asyncio.to_thread(self.model.encode, documents)
            return embeddings["dense_vecs"] # type: ignore

    async def _encode_query_async(self, query: str) -> np.ndarray:
        async with self.sem_encode:
            embeddings = await asyncio.to_thread(self.model.encode_queries, [query])
            return embeddings["dense_vecs"] # type: ignore

    def unload_index(self, key: str|int):
        self.indices.pop(key, None)
        torch.cuda.empty_cache()

    def key_exists(self, key: str|int = "_default") -> bool:
        """
        Check if the key exists in the indices dictionary.

        Args:
            key (str): Key for the index.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.indices

    async def create_index(self, texts: list[str], save_path: str|None = None) -> faiss.Index:
        """
        Creates a FAISS index from the given texts and saves it to the specified path if provided.
        """
        async with self.sem:
            embeddings = await self._encode_documents_async(texts)
            dim = embeddings.shape[1]

            index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
            index.add(embeddings)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                faiss.write_index(index, save_path)

            return index

    async def add_index(self, segments: list[Segment], save_path: Optional[str] = None, load_if_exists: bool = True, save: bool = True, key: str|int = "_default"):
        """
        Creates or loads an index and adds it to internal indices dictionary.

        Args:
            segments (list[Segment]): List of segments to index.
            key (str): Key for the index.
            save_path (Optional[str]): Path to save the index.
            load (Union[Literal["auto"], bool]): Whether to load the index if it exists.
        """
        if not segments:
            raise ValueError("No segments provided for indexing.")

        if load_if_exists and save_path and os.path.exists(save_path):
            index = faiss.read_index(save_path)
        else:
            async with self.sem:
                start = time.time()
                print("creating index")
                texts = [i.text for i in segments]
                embeddings = await self._encode_documents_async(texts)
                dim = embeddings.shape[1]

                index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
                index.add(embeddings)

                if save and save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    faiss.write_index(index, save_path)

                print("Creating index took:", time.time() - start, "seconds")

                torch.cuda.empty_cache()

        self.indices[key] = {
            "index": index,
            "corpus": segments
        }


    def _search_index(self, query_embeddings: np.ndarray, k: int, key: str|int = "_default") -> list[list[Segment]]:
        index = self.indices[key]["index"]
        corpus = self.indices[key]["corpus"]

        _, ids = index.search(query_embeddings, k=k)
        results = [
            [corpus[int(i)] for i in ids[j] if i != -1] for j in range(len(ids))
        ]
        return results

    def search(self, query: str, k: int = 10, key: str|int = "_default") -> list[Segment]:
        query_embeddings = self.model.encode([query], convert_to_numpy=True, batch_size=self.batch_size)["dense_vecs"]
        results = self._search_index(query_embeddings, k, key)

        torch.cuda.empty_cache()
        return results[0]

    async def search_async(self, query: str, k: int = 10, key: str|int = "_default") -> list[Segment]:
        query_embeddings = await self._encode_query_async(query)
        results = self._search_index(query_embeddings, k, key)

        torch.cuda.empty_cache()
        return results[0]

    async def search_external_index(self, query: str, index: faiss.Index, corpus: list[Segment], k: int = 10) -> list[Segment]:
        query_embeddings = await self._encode_documents_async([query])
        _, ids = index.search(query_embeddings, k=k)
        results = [
            [corpus[int(i)] for i in ids[j] if i != -1] for j in range(len(ids))
        ]
        return results[0]
