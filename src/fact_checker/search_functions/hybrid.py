"""
HybridSearch class for searching segments using a combination of dense and sparse retrieval methods.
Utilizes BGE-M3 for dense retrieval and BM25 for sparse retrieval.

@author: Hai Phong Nguyen
"""

from .dense import DenseSearch
from tqdm.asyncio import tqdm_asyncio
import tqdm
from .sparse import SparseSearch
import os
from collections import defaultdict
from dataset_manager import Dataset
from dataset_manager.models import Segment
from FlagEmbedding import FlagReranker

class HybridSearch():
    def __init__(self, dataset_path: str, storage_dir: str, merged: bool = True):
        self.storage_dir = storage_dir

        self.dense_retriever = DenseSearch(2500)
        self.sparse_retriever = SparseSearch()
        self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

        dataset = Dataset(dataset_path)
        segments = dataset.get_segments_with_statement_ids()
        self.segment_map = defaultdict(list)

        if merged:
            # Load all segments into a single index
            self.segment_map["merged"] = dataset.get_segments()
            self.statement_ids = ["merged"] # single index for all segments
        else:
            # Load segments with their statement IDs
            for statement_id, segment in segments:
                self.segment_map[statement_id].append(segment)

            self.statement_ids = self.segment_map.keys()


    async def create_indices(self):
        for key, segments in tqdm.tqdm(self.segment_map.items(), desc="Creating dense indices"):
            await self.dense_retriever.create_index(
                [segment.text for segment in segments], 
                os.path.join(self.storage_dir, "dense", f"{key}.faiss")
            )

        sparse_create_tasks = [
            self.sparse_retriever.create_index(
                [segment.text for segment in segments],
                os.path.join(self.storage_dir, "sparse", str(statement_id))
            )
            for statement_id, segments in self.segment_map.items()
        ]

        await tqdm_asyncio.gather(*sparse_create_tasks, desc="Creating sparse indices")


    async def load_indices(self):
        """
        Loads the indices for the given statement IDs.

        Args:
            statement_ids (list): List of statement IDs to load.
        """
        for statement_id in tqdm.tqdm(self.statement_ids, desc="Loading indices"):
            await self.dense_retriever.add_index(
                self.segment_map[statement_id],
                save_path=os.path.join(self.storage_dir, "dense", f"{statement_id}.faiss"),
                load_if_exists=True,
                save=False,
                key=statement_id
            )

            await self.sparse_retriever.add_index(
                self.segment_map[statement_id],
                os.path.join(self.storage_dir, "sparse", str(statement_id)),
                load_if_exists=True,
                save=False,
                key=statement_id
            )

    def _rerank(self, query: str, segments: list[Segment], k: int):
        sentence_pairs = [(query, result.text) for result in segments]
        scores = self.reranker.compute_score(sentence_pairs)
        sorted_results = [res for res, _ in sorted(zip(segments, scores), key=lambda x: x[1], reverse=True)]
        return sorted_results[:k]

    async def search_dense(
        self,
        query: str,
        statement_id: int|None = None,
        k: int = 3,
        n: int|None = None,
        rerank=True,
    ) -> list[Segment]:
        """
        Search a statement's article index using BGE-M3 dense embeddings.

        Args:
            query (str): Search query to be used in search
            statement_id (int): Statement whose articles to search
            k (int): How many top segments to retrieve
            n (int|None): Number of retrieved segments from index (can be higher than k in case we use reranking)

        Returns:
            List of segments.

        """
        if not n or n < k:
            n = k

        index_key = statement_id or "merged"

        results = await self.dense_retriever.search_async(
            query,
            k=n,
            key=index_key
        )

        if rerank:
            return self._rerank(query, results, k)
        else:
            return results[:k]

    async def search_on_the_fly(
        self,
        query: str,
        corpus: list[Segment],
        k: int = 3,
        n: int|None = None,
    ):
        """
        Search a statement's article index using BGE-M3 dense embeddings and BM25 vectors.

        Args:
            query (str): Search query to be used in search
            dense_index (faiss.Index): Dense index to search
            sparse_index (bm25s.BM25): Sparse index to search
            k (int): How many top segments to return
            n (int|None): Number of retrieved segments from each index

        Returns:
            List of segments.
        """
        if not n or 2*n < k:
            n = k 

        if not corpus:
            return []

        dense_index = await self.dense_retriever.create_index(
            [segment.text for segment in corpus]
        )
        sparse_index = await self.sparse_retriever.create_index(
            [segment.text for segment in corpus]
        )

        dense_results = await self.dense_retriever.search_external_index(
            query,
            index=dense_index,
            corpus=corpus,
            k=k,
        )

        sparse_results = await self.sparse_retriever.search_external_index(
            query,
            index=sparse_index,
            corpus=corpus,
            k=k,
        )

        combined_results = dense_results + sparse_results
        combined_results = list({seg.text:seg for seg in combined_results}.values())
        return self._rerank(query, combined_results, k)

    async def search(
        self,
        query: str,
        statement_id: int|None = None,
        k:int = 3,
        n:int|None = None,
    ):
        """
        Search a statement's article index using BGE-M3 dense embeddings and BM25 vectors.

        Args:
            query (str): Search query to be used in search
            statement_id (int): Statement whose articles to search
            k (int): How many top segments to return
            n (int|None): Number of retrieved segments from each index

        Returns:
            List of segments.
        """
        if not n or 2*n < k:
            n = k 

        index_key = statement_id or "merged"

        dense_results = await self.dense_retriever.search_async(
            query,
            k=k,
            key=index_key
        )

        sparse_results = await self.sparse_retriever.search_async(
            query,
            k=k,
            key=index_key
        )

        combined_results = dense_results + sparse_results
        combined_results = list({seg.text:seg for seg in combined_results}.values())
        return self._rerank(query, combined_results, k)

